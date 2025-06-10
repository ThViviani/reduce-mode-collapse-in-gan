import torch
import torch.nn as nn

from models.utils import ModelFreezeMixin, MLP


class DiscriminatorCNNBlock(nn.Module):
    """Defines a discriminator CNN block"""

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size=4, 
            stride=2, 
            use_dropout=False, 
            norm_layer=nn.BatchNorm2d,
            negative_slope=0.2,
            padding=1,
        ):
        """Construct a convolutional block.
        Parameters:
            in_channels (int)  -- the number of channels in the input
            out_channels (int) -- the number of channels in the ouput when applyed this block
            norm_layer         -- normalization layer
            stride (int)       -- the stride of conv layer
            negative_slope (float) -- controls the angle of the negative slope (which is used for negative input values). 
        """

        super(DiscriminatorCNNBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

        if use_dropout:
            self.conv.append(nn.Dropout(0.3))

    def forward(self, x):
        return self.conv(x)
    
class CriticMNIST(ModelFreezeMixin, nn.Module):
    def __init__(self, ndf=28, nc=1):
        """Construct a base MNIST Critic.
        Parameters:
            ndf (int) -- Size of feature maps in discriminator
            nc (int) -- Number of input channels.
        """
        super().__init__()

        self.feature_extractor = nn.Sequential( # 1 x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 28 x 14 x 14
            nn.LeakyReLU(0.2, inplace=True),
            DiscriminatorCNNBlock(ndf, ndf * 2), # 56 x 7 x 7
            DiscriminatorCNNBlock(ndf * 2, ndf * 4), # 112 x 3 x 3
        )

        self.logits = nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False) # 1 x 1 x 1

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        logits_ = self.logits(features)
        return (logits_, features) if return_features else logits_
    
class Critic2D(MLP):
    def forward(self, x, return_features=False):
        logits = super().forward(x)
        return (logits, logits) if return_features else logits
    
class CriticStackedMNIST(ModelFreezeMixin, nn.Module):
  def __init__(self, x_factor=2, nc=3):
    super().__init__()

    self.feature_extractor = nn.Sequential( # 3 x 32 x 32
        DiscriminatorCNNBlock(nc, 8 // x_factor, kernel_size=3, negative_slope=0.3), # 4 x 16 x 16
        DiscriminatorCNNBlock(8 // x_factor, 16 // x_factor, kernel_size=3, negative_slope=0.3), # 8 x 8 x 8
        DiscriminatorCNNBlock(16 // x_factor, 32 // x_factor, kernel_size=3, negative_slope=0.3), # 16 x 4 x 4
    )

    self.logits = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 1) if x_factor == 2 else nn.Linear(8 * 4 * 4, 1),
    )

  def forward(self, x, return_features=False):
    features = self.feature_extractor(x)
    logits_ = self.logits(features)
    return (logits_, features) if return_features else logits_
  
class CriticCIFAR10(ModelFreezeMixin, nn.Module):
  def __init__(self, nc=3, ndf=64):
    super().__init__()

    self.feature_extractor = nn.Sequential( # 3 x 32 x 32
        self._conv_lrelu(nc, ndf, 3, 1, 1),
        self._conv_lrelu(ndf, ndf, 4, 2, 1),
        self._conv_lrelu(ndf, ndf * 2, 3, 1, 1),
        self._conv_lrelu(ndf * 2, ndf * 2, 4, 2, 1),
        self._conv_lrelu(ndf * 2, ndf * 4, 3, 1, 1),
        self._conv_lrelu(ndf * 4, ndf * 4, 4, 2, 1),
        nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1) # 512 x 4 x 4
    )

    self.logits = nn.Sequential(
        nn.Flatten(),
        nn.Linear(ndf * 8 * 4 * 4, 1),
    )

  def forward(self, x, return_features=False):
    features = self.feature_extractor(x)
    logits_ = self.logits(features)
    return (logits_, features) if return_features else logits_

  def _conv_lrelu(self, in_channels, out_channels, kernel_size, stride, padding, negative_slope=0.1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
            ),
            nn.LeakyReLU(negative_slope, inplace=True),
        )