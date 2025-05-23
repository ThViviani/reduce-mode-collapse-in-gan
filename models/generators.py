import torch
import torch.nn as nn


class GeneratorCNNBlock(nn.Module):
    """Defines a generator CNN block"""

    def __init__(self, in_channels, out_channels, activation="relu", use_dropout=False, norm_layer=nn.BatchNorm2d, **kwargs):
        """Construct a generator CNN block
        Parameters:
            in_channels (int)            -- the num
        |ber of channels in the input
            out_channels (int)           -- the number of channels in the ouput when applyed this block
            activation (str)             -- the name of activation function: relu | another names
            use_dropout (bool)           -- if the flag is True then will add nn.Dropout after conv layer
            norm_layer (torch.nn)        -- normalization layer
        """

        super(GeneratorCNNBlock, self).__init__()

        kernel_size    = kwargs.get("kernel_size", 4)
        stride         = kwargs.get("stride", 2)
        padding        = kwargs.get("padding", 1)
        output_padding = kwargs.get("output_padding", 0)
        padding_mode   = kwargs.get("padding_mode", "zeros")

        use_bias = True if norm_layer == nn.InstanceNorm2d else False

        activation_module = None
        if activation == "relu":
            activation_module = nn.ReLU(inplace=True)
        elif activation == "identity":
            activation_module = nn.Identity()
        else:
            activation_module = nn.LeakyReLU(0.2)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=use_bias,
                output_padding=output_padding
            ),
            norm_layer(out_channels),
            activation_module,
        )

        if use_dropout:
            self.conv.append(nn.Dropout(0.5))

    def forward(self, x):
        return self.conv(x)

class GeneratorMNIST(nn.Module):
    def __init__(self, ngf=64, nc=1, latent_space_dim=100):
        super().__init__()

        self.generator = nn.Sequential(
            GeneratorCNNBlock(latent_space_dim, ngf * 4, kernel_size=3, stride=2, padding=0), # (ngf * 4) x 3 x 3
            GeneratorCNNBlock(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=0), # (ngf * 2) x 8 x 8
            GeneratorCNNBlock(ngf * 2, ngf, kernel_size=3, stride=2, padding=0), # (ngf * ) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 3, 2, 2, 1), # nc x 28 x 88
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.generator(z)