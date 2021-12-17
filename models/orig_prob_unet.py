import torch
import torch.nn as nn
from torch import distributions

def down_block(input_channels,
               output_channels,
               kernel_size,
               padding,
               stride=(1, 1),
               num_convs=2,
               down_sample_input=True):
    """
        Several convolutions followed by one down sampling
    :param input_channels:
    :param output_channels:
    :param kernel_size:
    :param padding:
    :param stride:
    :param num_convs:
    :param down_sample_input:
    :return:
    """

    layers = []
    for i in range(num_convs):
        if i > 0:
            input_channels = output_channels
        layers += [nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                   nn.GroupNorm(4, output_channels),
                   nn.ReLU()
                   ]
    if down_sample_input:
        layers += [nn.AvgPool2d(2)]

    return nn.Sequential(*layers)


def up_block(
        input_channels,
        output_channels,
        kernel_size,
        padding,
        stride,
        num_convs=2):
    """
     One upsampling followed by several convolutions
    :param input_channels:
    :param output_channels:
    :param kernel_size:
    :param padding:
    :param stride:
    :param num_convs:
    :return:
    """
    layers = [nn.ConvTranspose2d(input_channels, output_channels, (2, 2), (2, 2)),
              nn.ReLU()
              ]
    for i in range(num_convs):
        if i == 0:
            layers += [nn.Conv2d(2 * output_channels, output_channels, kernel_size, stride, padding),
                       nn.GroupNorm(4, output_channels),
                       nn.ReLU()
                       ]
        else:
            layers += [nn.Conv2d(output_channels, output_channels, kernel_size, stride, padding),
                       nn.GroupNorm(4, output_channels),
                       nn.ReLU()
                       ]


    return nn.Sequential(*layers)



class VGGEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 num_channels,
                 num_convs_per_block=3):
        super().__init__()
        self._input_dim = input_dim
        self._num_channels = num_channels
        self._num_convs = num_convs_per_block

        self.net = self._build()

    def forward(self, x):

        """
        returns a list of features by passing input through the encoder
        :param x:
        :return:
        """
        features = []
        for i in range(len(self._num_channels)):
            x = self.net[i](x)
            features.append(x)

        return features

    def _build(self):

        num_channels = [self._input_dim] + self._num_channels
        layers = []
        for i, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):

            if i == 0:
                down_sample = False
            else:
                down_sample = True

            layers += [down_block(input_channels=in_channels,
                                  output_channels=out_channels,
                                  kernel_size=(3, 3),
                                  padding=(1, 1),
                                  stride=(1, 1),
                                  num_convs=self._num_convs,
                                  down_sample_input=down_sample)
                       ]

        return nn.Sequential(*layers)

class VGGDecoder(nn.Module):

    def __init__(self,
                 num_channels,
                 num_convs_per_block=3):
        super().__init__()
        self._num_channels = num_channels
        self._num_convs = num_convs_per_block

        self.net = self._build()

    def forward(self, input_features):
        """
        gets features from the encoder and outputs a prediction
        :param input_features:
        :return:
        """
        n = len(input_features) - 1
        lower_res_features = input_features[-1]
        for i in range(n):
            same_res_features = input_features[n - i - 1]
            lower_res_features = self.net[i][0](lower_res_features)
            lower_res_features = torch.cat([lower_res_features, same_res_features], dim=1)
            lower_res_features = self.net[i][1:](lower_res_features)

        return lower_res_features

    def _build(self):
        layers = []
        for (input_channels, output_channels) in zip(self._num_channels[:-1], self._num_channels[1:]):

            layers += [
                        up_block(input_channels=input_channels,
                                 output_channels=output_channels,
                                 kernel_size=(3, 3),
                                 padding=(1, 1),
                                 stride=(1, 1),
                                 num_convs=self._num_convs)
                    ]


        return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self,
                 input_dim,
                 num_channels,
                 num_convs_per_block=3):
        super().__init__()

        self._encoder = VGGEncoder(input_dim, num_channels, num_convs_per_block)
        self._decoder = VGGDecoder(list(reversed(num_channels)), num_convs_per_block)

    def forward(self, x):
        encoder_features = self._encoder(x)
        logits = self._decoder(encoder_features)

        return logits



class Conv1x1Decoder(nn.Module):

    def __init__(self,
                 num_classes,
                 num_1x1_convs,
                 latent_dim,
                 num_channels):
        super().__init__()

        self._num_1x1_convs = num_1x1_convs
        self._latent_dim = latent_dim
        self._num_classes = num_classes
        self._num_channels = num_channels

        self.net = self._build()


    def forward(self, unet_features):
        """ Add the noise to the output of the UNet model, then pass it through several 1x1 convolutions
            z: [Batch size, latent_dim]
            unet_feature: [Batch size, input_channels, H, W]
        """

        # *_, h, w = unet_features.shape
        # out = torch.cat([z[..., None, None].tile(dims=(1, 1, h, w)), unet_features], dim=1)

        return self.net(unet_features)

    def _build(self):
        layers = []
        in_channels = self._latent_dim + self._num_channels
        for i in range(self._num_1x1_convs):
            if i > 0:
                in_channels = self._num_classes
            layers += [nn.Conv2d(in_channels, self._num_classes, (1, 1))]
            if i < self._num_1x1_convs - 1:
                layers += [nn.ReLU()]



        return nn.Sequential(*layers)

class UNetSegmentation(nn.Module):
    def __init__(self,
                 input_dim,
                 num_channels,
                 num_convs_per_block=3):
        super().__init__()

        self._encoder = VGGEncoder(input_dim, num_channels, num_convs_per_block)
        self._decoder = VGGDecoder(list(reversed(num_channels)), num_convs_per_block)
        self._fcomb = Conv1x1Decoder(1, 3, 0, 32)

    def forward(self, x):
        encoder_features = self._encoder(x)
        logits = self._decoder(encoder_features)
        logits = self._fcomb(logits)

        return logits


class AxisAlignedConvGaussian(nn.Module):

    def __init__(self,
                 input_dim,
                 latent_dim,
                 num_channels,
                 num_convs_per_block):
        super().__init__()

        self._latent_dim = latent_dim
        self._num_channels = num_channels
        self._num_convs_per_blocks = num_convs_per_block

        self._encoder = VGGEncoder(input_dim, num_channels, num_convs_per_block)

        self._avg_pool = nn.AdaptiveAvgPool2d(1)

        self._mu_log_sigma = nn.Conv2d(num_channels[-1], 2 * self._latent_dim, (1, 1))

    def forward(self, x):
        encoding = self._encoder(x)
        encoding = self._avg_pool(encoding[-1])
        mu_log_sigma = self._mu_log_sigma(encoding)
        mu = mu_log_sigma[:, :self._latent_dim, 0, 0]
        log_sigma = mu_log_sigma[:, self._latent_dim:, 0, 0]
        return distributions.Independent(distributions.Normal(loc=mu, scale=torch.exp(log_sigma)), 1)



class ProbUNet(nn.Module):

    """ Probabilistic UNet"""

    def __init__(self,
                 input_dim,
                 latent_dim,
                 num_channels,
                 num_convs_per_block,
                 num_classes,
                 num_1x1_convs=3):
        super().__init__()


        self._unet = UNet(input_dim, num_channels, num_convs_per_block)
        self._f_comb = Conv1x1Decoder(num_classes, num_1x1_convs, latent_dim, num_channels[0])
        self._prior = AxisAlignedConvGaussian(input_dim, latent_dim, num_channels, num_convs_per_block)  # RGB image
        self._posterior = AxisAlignedConvGaussian(input_dim + 1, latent_dim, num_channels, num_convs_per_block)  # Image + ground truth


    def forward(self, *args, train=True):
        if train:
            img, mask = args
        else:
            img = args
        p = self._prior(img)
        unet_features = self._unet(img)
        z_p = p.sample()
        pred = self._f_comb(z_p, unet_features)

        if train:
            img, mask = args
            q = self._posterior(torch.cat([img, mask], dim=1))

            return p, q, pred
        return p, pred


    @staticmethod
    def kl(p, q):
        return torch.distributions.kl_divergence(q, p).sum()
        # z_q = q.rsample()
        # log_q = q.log_prob(z_q)
        # log_p = p.log_prob(z_q)
        # kld = log_q - log_p
        # return kld.mean()

    @staticmethod
    def reconstruction_loss(pred, target):
        return nn.BCEWithLogitsLoss(reduction='none')(pred, target).sum()

    def elbo(self, pred, target, p, q):
        kl = self.kl(p, q)
        recon_loss = self.reconstruction_loss(pred, target)

        return 10 * kl + recon_loss
