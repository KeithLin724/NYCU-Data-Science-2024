import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, dim_in: int, dim_out: int, net_mode: str | None = None):
        super().__init__()

        if net_mode == "p" or (net_mode is None):
            use_affine = True
        elif net_mode == "t":
            use_affine = False

        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
        )

    def forward(self, x):
        return x + self.main(x)


class GetMatrix(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.get_gamma = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.get_beta = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return x, gamma, beta


class NONLocalBlock2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0
        )

    def forward(self, source, weight):
        """(b, c, h, w)
        src_diff: (3, 136, 32, 32)
        """
        batch_size = source.size(0)

        g_source = source.view(batch_size, 1, -1)  # (N, C, H*W)
        g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)

        y = torch.bmm(weight.to_dense(), g_source)
        y = y.permute(0, 2, 1).contiguous()  # (N, C, H*W)
        y = y.view(batch_size, 1, *source.size()[2:])
        return y


class BaseComponents:

    @staticmethod
    def build_ResidualBlock(
        dim_in: int, dim_out: int, net_mode: str | None = None
    ) -> ResidualBlock:
        return ResidualBlock(dim_in, dim_out, net_mode)

    @staticmethod
    def build_GetMatrix(dim_in: int, dim_out: int) -> GetMatrix:
        return GetMatrix(dim_in, dim_out)

    @staticmethod
    def build_NONLocalBlock2D() -> NONLocalBlock2D:
        return NONLocalBlock2D()
