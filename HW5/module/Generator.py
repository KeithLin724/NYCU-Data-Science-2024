import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
        )

    def forward(self, x):
        return x + self.main(x)


class SamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        up: bool = False,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False,
    ):
        super().__init__()

        first_conv = nn.Conv2d if not up else nn.ConvTranspose2d

        self.sampling_seq = nn.Sequential(
            first_conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        return

    def forward(self, x):
        return self.sampling_seq(x)


class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self, conv_dim: int = 64, repeat_num: int = 6):
        super().__init__()

        layers = [
            SamplingBlock(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        ]

        # Down-Sampling
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(SamplingBlock(curr_dim, curr_dim * 2))
            curr_dim = curr_dim * 2

        # Bottleneck
        for _ in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for _ in range(2):
            layers.append(SamplingBlock(curr_dim, curr_dim // 2, up=True))
            curr_dim = curr_dim // 2

        layers.append(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        )
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out


class Generator_makeup(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    # input 2 images and output 2 images as well
    def __init__(self, conv_dim=64, repeat_num=6, input_nc=6):
        super().__init__()

        layers = [
            SamplingBlock(
                input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False
            )
        ]

        # Down-Sampling
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(SamplingBlock(curr_dim, curr_dim * 2))
            curr_dim = curr_dim * 2

        # Bottleneck
        for _ in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for _ in range(2):
            layers.append(SamplingBlock(curr_dim, curr_dim // 2, up=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        self.branch_1 = nn.Sequential(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, y):
        input_x = torch.cat((x, y), dim=1)
        out = self.main(input_x)
        out_A = self.branch_1(out)
        out_B = self.branch_2(out)
        return out_A, out_B


class Generator_branch(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    # input 2 images and output 2 images as well
    def __init__(self, conv_dim=64, repeat_num=6, input_nc=3):
        super(Generator_branch, self).__init__()

        # Branch input
        self.Branch_0 = nn.Sequential(
            SamplingBlock(
                input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False
            ),
            SamplingBlock(conv_dim, conv_dim * 2),
        )

        # Branch input
        self.Branch_1 = nn.Sequential(
            SamplingBlock(
                input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False
            ),
            SamplingBlock(conv_dim, conv_dim * 2),
        )

        # Down-Sampling, branch merge
        layers = []
        curr_dim = conv_dim * 2
        layers.append(SamplingBlock(curr_dim * 2, curr_dim * 2))
        curr_dim = curr_dim * 2

        # Bottleneck
        for _ in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for _ in range(2):
            layers.append(SamplingBlock(curr_dim, curr_dim // 2, up=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        self.branch_1 = nn.Sequential(
            SamplingBlock(
                curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            SamplingBlock(
                curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        )

        self.branch_2 = nn.Sequential(
            SamplingBlock(
                curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            SamplingBlock(
                curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, y):
        input_x = self.Branch_0(x)
        input_y = self.Branch_1(y)
        input_fuse = torch.cat((input_x, input_y), dim=1)
        out = self.main(input_fuse)
        out_A = self.branch_1(out)
        out_B = self.branch_2(out)
        return out_A, out_B
