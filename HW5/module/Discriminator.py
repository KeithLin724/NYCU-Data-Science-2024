import torch
import torch.nn as nn
from ops import spectral_norm as SpectralNorm


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""

    def __init__(
        self,
        image_size: int = 128,
        conv_dim: int = 64,
        repeat_num: int = 3,
        norm: str = "SN",
    ):
        super().__init__()

        if norm == "SN":
            layer_type = lambda in_channels, out_channels, kernel_size, stride, padding, bias=True: SpectralNorm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        else:
            layer_type = lambda in_channels, out_channels, kernel_size, stride, padding, bias=True: nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )

        layers = [
            layer_type(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        ]

        curr_dim = conv_dim
        for _ in range(1, repeat_num):
            layers.append(
                layer_type(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        # k_size = int(image_size / np.power(2, repeat_num))

        layers.append(
            layer_type(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1)
        )

        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)

        self.conv1 = layer_type(
            curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False
        )

        # conv1 remain the last square size, 256*256-->30*30
        # self.conv2 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=False))
        # conv2 output a single number

    def forward(self, x):
        h = self.main(x)
        # out_real = self.conv1(h)
        out_makeup = self.conv1(h)
        # return out_real.squeeze(), out_makeup.squeeze()
        return out_makeup.squeeze()
