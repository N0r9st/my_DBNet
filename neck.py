import torch
import torch.nn.functional as F
from torch import nn

from backbone import *


class FPN(nn.Module):
    def __init__(self, in_channels, inner_channels=256, **kwargs):
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.reduce_conv_l1 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_l2 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_l3 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_l4 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers
        self.smooth_l1 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_l2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_l3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )
        self.out_channels = self.conv_out

    def forward(self, x):
        #c2, c3, c4, c5 = x
        # Top-down
        rl4 = self.reduce_conv_l4(x['layer4'])
        rl3 = self._upsample_add(rl4, self.reduce_conv_l3(x['layer3']))
        rl3 = self.smooth_l3(rl3)
        rl2 = self._upsample_add(rl3, self.reduce_conv_l2(x['layer2']))
        rl2 = self.smooth_l2(rl2)
        rl1 = self._upsample_add(rl2, self.reduce_conv_l1(x['layer1']))
        rl1 = self.smooth_l1(rl1)

        x = self._upsample_cat(rl1, rl2, rl3, rl4)
        x = self.conv(x)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, rl1, rl2, rl3, rl4):
        h, w = rl1.size()[2:]
        rl2 = F.interpolate(rl2, size=(h, w))
        rl3 = F.interpolate(rl3, size=(h, w))
        rl4 = F.interpolate(rl4, size=(h, w))
        return torch.cat([rl1, rl2, rl3, rl4], dim=1)