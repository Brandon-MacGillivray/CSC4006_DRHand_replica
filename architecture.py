import torch
import torch.nn as nn
import torch.nn.functional as F
from block import Conv2D, DSConv2D, MaxPool, DeConv2D

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.convM = Conv2D(3, 24, 3, 1, 1)
        self.convR1 = Conv2D(24, 48, 1, 1, 0)
        self.convR2 = Conv2D(48, 96, 1, 1, 0)
        self.convR3 = Conv2D(96, 192, 1, 1, 0)
        self.convR4 = Conv2D(96, 288, 1, 1, 0)

        self.mp = MaxPool(2,2)

        self.dsconvL1 = DSConv2D(24, 48, 5, 2, 2)
        self.dsconvL2 = DSConv2D(48, 96, 5, 2, 2)
        self.dsconvL3 = DSConv2D(96, 192, 5, 2, 2)
        self.dsconvL4 = DSConv2D(192, 192, 5, 2, 2)
        self.dsconvL5 = DSConv2D(192, 192, 5, 2, 2)
        self.dsconvL6 = DSConv2D(96, 96, 5, 2, 2)
        self.dsconvL7 = DSConv2D(96, 288, 5, 2, 2)
        self.dsconvL8 = DSConv2D(288, 288, 5, 2, 2)
        self.dsconvL9 = DSConv2D(288, 288, 5, 2, 2)
        self.dsconvL10 = DSConv2D(288, 288, 5, 2, 2)

        self.dsconvM1_1 = DSConv2D(24, 24, 5, 1, 2)
        self.dsconvM1_2 = DSConv2D(24, 24, 5, 1, 2)
        self.dsconvM2_1 = DSConv2D(48, 48, 5, 1, 2)
        self.dsconvM2_2 = DSConv2D(48, 48, 5, 1, 2)
        self.dsconvM3_1 = DSConv2D(96, 96, 5, 1, 2)
        self.dsconvM3_2 = DSConv2D(96, 96, 5, 1, 2)
        self.dsconvM4_1 = DSConv2D(192, 192, 5, 1, 2)
        self.dsconvM4_2 = DSConv2D(192, 192, 5, 1, 2)
        self.dsconvM5_1 = DSConv2D(384, 192, 5, 1, 2)
        self.dsconvM5_2 = DSConv2D(192, 192, 5, 1, 2)
        self.dsconvM6 = DSConv2D(288, 192, 5, 1, 2)
        self.dsconvM7 = DSConv2D(192, 96, 5, 1, 2)
        self.dsconvM8_1 = DSConv2D(144, 96, 5, 1, 2)
        self.dsconvM8_2 = DSConv2D(96, 96, 5, 1, 2)
        self.dsconvM8_3 = DSConv2D(96, 96, 5, 1, 2)
        self.dsconvM8_4 = DSConv2D(96, 96, 5, 1, 2)
        self.dsconvM9_1 = DSConv2D(288, 288, 5, 1, 2)
        self.dsconvM9_2 = DSConv2D(288, 288, 5, 1, 2)

        self.deconvM1 = DeConv2D(192, 192, 3, 2, 1, 1)
        self.deconvM2 = DeConv2D(192, 192, 3, 2, 1, 1)
        self.deconvM3 = DeConv2D(96, 96, 3, 2, 1, 1)

    def forward(self, x):

        x = self.convM(x)

        #

        x = self.dsconvM1_1(x)
        x = self.dsconvM1_2(x)

        xl = self.dsconvL1(x)
        xr = self.mp(x)
        xr = self.convR1(xr)

        x = xl + xr
        x_skip_1 = x

        #

        x = self.dsconvM2_1(x)
        x = self.dsconvM2_2(x)

        xl = self.dsconvL2(x)
        xr = self.mp(x)
        xr = self.convR2(xr)

        x = xl + xr
        x_skip_2 = x

        #

        x = self.dsconvM3_1(x)
        x = self.dsconvM3_2(x)

        xl = self.dsconvL3(x)
        xr = self.mp(x)
        xr = self.convR3(xr)

        x = xl + xr
        x_skip_3 = x

        #

        x = self.dsconvM4_1(x)
        x = self.dsconvM4_2(x)

        xl = self.dsconvL4(x)
        xr = self.mp(x)

        x = xl + xr

        x = self.deconvM1(x)

        x = torch.cat([x, x_skip_3], dim=1)

        #

        x = self.dsconvM5_1(x)
        x = self.dsconvM5_2(x)

        xl = self.dsconvL5(x)
        xr = self.mp(x)

        x = xl + xr

        x = self.deconvM2(x)

        x = F.interpolate(x, size=x_skip_2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x_skip_2], dim=1)

        #

        x = self.dsconvM6(x)
        x = self.dsconvM7(x)

        xl = self.dsconvL6(x)
        xr = self.mp(x)

        x = xl + xr

        x = self.deconvM3(x)

        x = F.interpolate(x, size=x_skip_1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x_skip_1], dim=1)

        #

        x = self.dsconvM8_1(x)
        x = self.dsconvM8_2(x)
        x = self.dsconvM8_3(x)
        x = self.dsconvM8_4(x)

        xl = self.dsconvL7(x)
        xr = self.mp(x)
        xr = self.convR4(xr)

        x = xl + xr

        x = self.dsconvM9_1(x)
        x = self.dsconvM9_2(x)

        xl = self.dsconvL8(x)
        xr = self.mp(x)

        x = xl + xr

        x = self.dsconvL9(x)
        x = self.dsconvL10(x)

        return x
    
class Heatmap_reg(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconvM1 = DeConv2D(288, 192, 3, 2, 1, 1)
        self.deconvM2 = DeConv2D(192, 96, 3, 2, 1, 1)
        self.deconvM3 = DeConv2D(96, 48, 3, 2, 1, 1)

        self.convM = Conv2D(48, 21, 1, 1, 0)

    @staticmethod
    def heatmap_to_coords_argmax(hm):
        # hm: (N, J, H, W)
        n, j, h, w = hm.shape
        flat = hm.reshape(n, j, -1)
        idx = flat.argmax(dim=-1)
        y = (idx // w).float()
        x = (idx % w).float()
        return torch.stack([x, y], dim=-1)

    def forward(self, x, return_coords=False):
        x = self.deconvM1(x)
        x = self.deconvM2(x)
        x = self.deconvM3(x)
        x = self.convM(x)
        if return_coords:
            return x, self.heatmap_to_coords_argmax(x)
        return x
