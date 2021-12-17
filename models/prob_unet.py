from models.unet import UNet
import torch
import torch.nn as nn
from torch import distributions
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Conv1x1Decoder(nn.Module):

    def __init__(self,
                 latent_dim,
                 init_features,
                 num_classes=1,
                 num_1x1_convs=3):
        super().__init__()

        self._num_1x1_convs = num_1x1_convs
        self._latent_dim = latent_dim
        self._num_classes = num_classes
        self._features = init_features

        self.net = self._build()


    def forward(self, z, unet_features):
        """ Add the noise to the output of the UNet model, then pass it through several 1x1 convolutions
            z: [Batch size, latent_dim]
            unet_feature: [Batch size, input_channels, H, W]
        """

        *_, h, w = unet_features.shape
        out = torch.cat([unet_features, z[..., None, None].tile(dims=(1, 1, h, w))], dim=1)
        logits = self.net(out)

        return logits

    def _build(self):
        layers = []
        in_channels = self._latent_dim + self._features
        for i in range(self._num_1x1_convs - 1):
            layers += [nn.Conv2d(in_channels, in_channels, (1, 1)),
                       nn.LeakyReLU(0.1)]

        layers += [nn.Conv2d(in_channels, self._num_classes, (1, 1)),
                   nn.Sigmoid()]


        return nn.Sequential(*layers)


class AxisAlignedConvGaussian(nn.Module):
    """
    Takes in RGB image and a segmentation ground truth.
    Outputs the mean and log std of of the input.
    """

    def __init__(self,
                 latent_dim,
                 in_channels=3,
                 init_features=32,
                 ):

        super().__init__()
        self._latent_dim = latent_dim
        features = init_features
        self.encoder1 = AxisAlignedConvGaussian._block(in_channels, features)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder2 = AxisAlignedConvGaussian._block(features, 2 * features)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = AxisAlignedConvGaussian._block(features * 2, features * 4)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder4 = AxisAlignedConvGaussian._block(features * 4, features * 8)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder5 = AxisAlignedConvGaussian._block(features * 8, features * 8)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = AxisAlignedConvGaussian._block(features * 8, features * 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self._mu_log_sigma = nn.Conv2d(8 * features, 2 * self._latent_dim, (1, 1))

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        bottleneck = self.bottleneck(self.pool5(enc5))

        mu_log_sigma = self._mu_log_sigma(self.avg_pool(bottleneck))
        mu = mu_log_sigma[:, :self._latent_dim, 0, 0]
        log_sigma = mu_log_sigma[:, self._latent_dim:, 0, 0]
# distributions.Independent(distributions.Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return mu, log_sigma


    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=True),
            nn.GroupNorm(4, features),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=True,
            ),
            nn.GroupNorm(4, features),
            nn.LeakyReLU(0.1)
        )


class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        bce = self.bce_loss(input, target)
        pt = torch.exp(-bce)
        if self.alpha is None:
            # use inverse weighting
            zeros = (1 - target).sum()
            total = target.numel()
            alpha = zeros / total
        else:
            # use fixed weighting
            alpha = self.alpha

        focal_loss = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt) ** self.gamma * bce

        return focal_loss.mean()

def iou_loss(pred, target):
    y_pred = torch.sigmoid(pred.view(-1))
    y_target = target.view(-1)
    intersection = (y_pred * y_target).sum()
    iou = (intersection + 1) / (y_pred.sum() + y_target.sum() - intersection + 1)

    return -iou

class ProbUNet(nn.Module):

    """ Probabilistic UNet"""

    def __init__(self,
                 latent_dim,
                 in_channels,
                 num_classes,
                 num_1x1_convs=3,
                 init_features=32):
        super().__init__()
        self._latent_dim = latent_dim
        self._unet = UNet(in_channels, init_features)
        self._f_comb = Conv1x1Decoder(latent_dim, init_features, num_classes, num_1x1_convs)
        # self._prior = AxisAlignedConvGaussian(latent_dim, in_channels, init_features)  # RGB image
        self._posterior = AxisAlignedConvGaussian(latent_dim, in_channels + 1, init_features)  # Image + ground truth

        self.apply(self.init_weight)


    def forward(self, *args):

        if self.training:
            img, mask = args
            unet_features = self._unet(img)
            self.mu, self.log_sigma = self._posterior(torch.cat([img, mask], dim=1))
            self.q = distributions.Normal(self.mu, torch.exp(self.log_sigma) + 1e-3)
            z_q = self.q.sample()
            logits = self._f_comb(z_q, unet_features)

            return logits
        else:
            img = args[0]
            batch_size = img.shape[0]
            mean = torch.zeros(batch_size, self._latent_dim, device=img.device)
            cov = torch.eye(self._latent_dim, device=img.device)
            prior = distributions.MultivariateNormal(mean, cov)
            z_p = prior.sample()
            unet_features = self._unet(img)
            logits = self._f_comb(z_p, unet_features)
            return logits

    @staticmethod
    def init_weight(m):

        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.kaiming_normal_(m.weight)

            if hasattr(m, 'bias'):
                torch.nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.GroupNorm):
            torch.nn.init.normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def kl(self):
        kld = torch.mean(-0.5 * torch.sum(1 + 2 * self.log_sigma - self.mu ** 2 - self.log_sigma.exp()**2, dim=1))
        return kld


    @staticmethod
    def reconstruction_loss(pred, target):
        return nn.BCELoss()(pred, target)
        # return iou_loss(pred, target)
        # return nn.MSELoss()(pred, target)
        # return FocalLoss()(pred, target)
        # return iou_loss(pred, target)


    def elbo(self, pred, target, beta=0.01):

        kl = self.kl()
        recon_loss = self.reconstruction_loss(pred, target)

        print("Kl  ", kl.item(), "Recon loss  ", recon_loss.item())

        return beta * + kl + recon_loss

