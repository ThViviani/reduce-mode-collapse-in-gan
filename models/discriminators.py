import torch
import torch.nn as nn


class DiscriminatorCNNBlock(nn.Module):
    """Defines a discriminator CNN block"""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, use_dropout=False, norm_layer=nn.BatchNorm2d):
        """Construct a convolutional block.
        Parameters:
            in_channels (int)  -- the number of channels in the input
            out_channels (int) -- the number of channels in the ouput when applyed this block
            norm_layer         -- normalization layer
            stride (int)       -- the stride of conv layer
        """

        super(DiscriminatorCNNBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if use_dropout:
            self.conv.append(nn.Dropout(0.3))

    def forward(self, x):
        return self.conv(x)
    
class CriticMNIST(nn.Module):
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

        self.logits = nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False), # 1 x 1 x 1

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        logits_ = self.logits(features)
        return (logits_, features) if return_features else logits_

    # TODO: move this logic to Mixin
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True