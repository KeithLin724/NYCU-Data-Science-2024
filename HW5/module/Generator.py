import torch
import torch.nn as nn
import torch.nn.functional as F

from base import Track

import lightning as L
from BaseComponent import NONLocalBlock2D, GetMatrix, ResidualBlock


class Generator(L.LightningModule, Track):
    def __init__(self):
        super().__init__()
        self.build_MDNet()
        self.build_MANet()
        Track.__init__(self)

    def build_MDNet(self):
        "PNet(MDNet) for obtaining makeup matrices"
        layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        self.pnet_in = layers

        curr_dim = 64
        curr_dim = self.build_p_down_sampling(curr_dim=curr_dim)
        curr_dim = self.build_p_bottleneck(curr_dim=curr_dim)
        return curr_dim

    def build_p_down_sampling(self, curr_dim: int):
        """build Down-Sampling"""
        for i in range(2):
            next_dim = curr_dim * 2
            layers = nn.Sequential(
                nn.Conv2d(
                    curr_dim,
                    next_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(next_dim, affine=True),
                nn.ReLU(inplace=True),
            )

            setattr(self, f"pnet_down_{i+1}", layers)
            curr_dim = next_dim
        return curr_dim

    def build_p_bottleneck(self, curr_dim: int) -> None:
        """Bottleneck. All bottlenecks share the same attention module"""
        self.atten_bottleneck_g = NONLocalBlock2D()
        self.atten_bottleneck_b = NONLocalBlock2D()
        self.simple_spade = GetMatrix(curr_dim, 1)  # get the makeup matrix
        for i in range(3):
            setattr(
                self,
                f"pnet_bottleneck_{i+1}",
                ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode="p"),
            )
        return

    def build_MANet(self):
        self.tnet_in_conv = nn.Conv2d(
            3, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.tnet_in_spade = nn.InstanceNorm2d(64, affine=False)
        self.tnet_in_relu = nn.ReLU(inplace=True)

        curr_dim = 64
        curr_dim = self.build_t_down_sampling(curr_dim=curr_dim)
        curr_dim = self.build_t_bottleneck(curr_dim=curr_dim)
        self.build_t_up_sampling(curr_dim=curr_dim)
        return

    def build_t_down_sampling(self, curr_dim: int) -> int:
        for i in range(2):
            setattr(
                self,
                f"tnet_down_conv_{i+1}",
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            setattr(
                self,
                f"tnet_down_spade_{i+1}",
                nn.InstanceNorm2d(curr_dim * 2, affine=False),
            )
            setattr(
                self,
                f"tnet_down_relu_{i+1}",
                nn.ReLU(inplace=True),
            )
            curr_dim *= 2
        return curr_dim

    def build_t_bottleneck(self, curr_dim: int):
        for i in range(6):
            setattr(
                self,
                f"tnet_bottleneck_{i+1}",
                ResidualBlock(
                    dim_in=curr_dim,
                    dim_out=curr_dim,
                    net_mode="t",
                ),
            )
        return curr_dim

    def build_t_up_sampling(self, curr_dim: int) -> None:
        for i in range(2):
            setattr(
                self,
                f"tnet_up_conv_{i+1}",
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            setattr(
                self,
                f"tnet_up_spade_{i+1}",
                nn.InstanceNorm2d(curr_dim // 2, affine=False),
            )
            setattr(self, f"tnet_up_relu_{i+1}", nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.tnet_out = nn.Sequential(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        )
        return

    @staticmethod
    def atten_feature(mask_s, weight, gamma_s, beta_s, atten_module_g, atten_module_b):
        """
        feature size: (1, c, h, w)
        mask_c(s): (3, 1, h, w)
        diff_c: (1, 138, 256, 256)
        return: (1, c, h, w)
        """
        channel_num = gamma_s.shape[1]

        mask_s_re = F.interpolate(mask_s, size=gamma_s.shape[2:]).repeat(
            1, channel_num, 1, 1
        )
        gamma_s_re = gamma_s.repeat(3, 1, 1, 1)
        gamma_s = gamma_s_re * mask_s_re  # (3, c, h, w)
        beta_s_re = beta_s.repeat(3, 1, 1, 1)
        beta_s = beta_s_re * mask_s_re

        gamma = atten_module_g(gamma_s, weight)  # (3, c, h, w)
        beta = atten_module_b(beta_s, weight)

        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(
            0
        )  # (c, h, w) combine the three parts
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta

    def get_weight(self, mask_c, mask_s, fea_c, fea_s, diff_c, diff_s):
        """s --> source; c --> target
        feature size: (1, 256, 64, 64)
        diff: (3, 136, 32, 32)
        """
        HW = 64 * 64
        batch_size = 3
        assert fea_s is not None  # fea_s when i==3
        # get 3 part fea using mask
        channel_num = fea_s.shape[1]

        mask_c_re = F.interpolate(mask_c, size=64).repeat(
            1, channel_num, 1, 1
        )  # (3, c, h, w)
        fea_c = fea_c.repeat(3, 1, 1, 1)  # (3, c, h, w)
        fea_c = fea_c * mask_c_re  # (3, c, h, w) 3 stands for 3 parts

        mask_s_re = F.interpolate(mask_s, size=64).repeat(1, channel_num, 1, 1)
        fea_s = fea_s.repeat(3, 1, 1, 1)
        fea_s = fea_s * mask_s_re

        theta_input = torch.cat((fea_c * 0.01, diff_c), dim=1)
        phi_input = torch.cat((fea_s * 0.01, diff_s), dim=1)

        theta_target = theta_input.view(batch_size, -1, HW)  # (N, C+136, H*W)
        theta_target = theta_target.permute(0, 2, 1)  # (N, H*W, C+136)

        phi_source = phi_input.view(batch_size, -1, HW)  # (N, C+136, H*W)
        self.track("before mask")

        weight = torch.bmm(theta_target, phi_source)  # (3, HW, HW)
        self.track("among bmm")
        with torch.no_grad():
            v = weight.detach().nonzero().long().permute(1, 0)
            # This clone is required to correctly release cuda memory.
            weight_ind = v.clone()
            del v
            torch.cuda.empty_cache()

        weight *= 200  # hyper parameters for visual feature
        weight = F.softmax(weight, dim=-1)
        weight = weight[weight_ind[0], weight_ind[1], weight_ind[2]]
        ret = torch.sparse.FloatTensor(weight_ind, weight, torch.Size([3, HW, HW]))
        self.track("after bmm")
        return ret

    def forward(
        self, c, s, mask_c, mask_s, diff_c, diff_s, gamma=None, beta=None, ret=False
    ):
        c, s, mask_c, mask_s, diff_c, diff_s = [
            x.squeeze(0) if x.ndim == 5 else x
            for x in [c, s, mask_c, mask_s, diff_c, diff_s]
        ]
        """attention version
        c: content, stands for source image. shape: (b, c, h, w)
        s: style, stands for reference image. shape: (b, c, h, w)
        mask_list_c: lip, skin, eye. (b, 1, h, w)
        """

        self.track("start")
        # forward c in tnet(MANet)
        c_tnet = self.tnet_in_conv(c)
        s = self.pnet_in(s)
        c_tnet = self.tnet_in_spade(c_tnet)
        c_tnet = self.tnet_in_relu(c_tnet)

        # down-sampling
        for i in range(2):
            if gamma is None:
                cur_pnet_down = getattr(self, f"pnet_down_{i+1}")
                s = cur_pnet_down(s)

            cur_tnet_down_conv = getattr(self, f"tnet_down_conv_{i+1}")
            cur_tnet_down_spade = getattr(self, f"tnet_down_spade_{i+1}")
            cur_tnet_down_relu = getattr(self, f"tnet_down_relu_{i+1}")
            c_tnet = cur_tnet_down_conv(c_tnet)
            c_tnet = cur_tnet_down_spade(c_tnet)
            c_tnet = cur_tnet_down_relu(c_tnet)
        self.track("downsampling")

        # bottleneck
        for i in range(6):
            if gamma is None and i <= 2:
                cur_pnet_bottleneck = getattr(self, f"pnet_bottleneck_{i+1}")
            cur_tnet_bottleneck = getattr(self, f"tnet_bottleneck_{i+1}")

            # get s_pnet from p and transform
            if i == 3:
                if gamma is None:  # not in test_mix
                    s, gamma, beta = self.simple_spade(s)
                    weight = self.get_weight(mask_c, mask_s, c_tnet, s, diff_c, diff_s)
                    gamma, beta = self.atten_feature(
                        mask_s,
                        weight,
                        gamma,
                        beta,
                        self.atten_bottleneck_g,
                        self.atten_bottleneck_b,
                    )
                    if ret:
                        return [gamma, beta]
                # else:                       # in test mode
                # gamma, beta = param_A[0]*w + param_B[0]*(1-w), param_A[1]*w + param_B[1]*(1-w)

                c_tnet = (
                    c_tnet * (1 + gamma) + beta
                )  # apply makeup transfer using makeup matrices

            if gamma is None and i <= 2:
                s = cur_pnet_bottleneck(s)
            c_tnet = cur_tnet_bottleneck(c_tnet)
        self.track("bottleneck")

        # up-sampling
        for i in range(2):
            cur_tnet_up_conv = getattr(self, f"tnet_up_conv_{i+1}")
            cur_tnet_up_spade = getattr(self, f"tnet_up_spade_{i+1}")
            cur_tnet_up_relu = getattr(self, f"tnet_up_relu_{i+1}")

            c_tnet = cur_tnet_up_conv(c_tnet)
            c_tnet = cur_tnet_up_spade(c_tnet)
            c_tnet = cur_tnet_up_relu(c_tnet)

        self.track("upsampling")

        c_tnet = self.tnet_out(c_tnet)
        return c_tnet


# a = Generator()

# print(a)
