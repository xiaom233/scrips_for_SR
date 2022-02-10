'''
This repository is used to implement all blocks and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''
from functools import partial

import torch
import torch.nn as nn
import math
from distutils.version import LooseVersion

# Depthwise Separable Convolution
class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthWiseConv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


# Blueprint Separable Convolutions
# BN will be replaced by LN
class BSConvU(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None):
        super().__init__()

        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.add_module("pw", torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        ))

        # batchnorm
        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))

        # depthwise
        self.add_module("dw", torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        ))


class BSConvS(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", p=0.25, min_mid_channels=4, with_bn=False, bn_kwargs=None):
        super().__init__()

        # check arguments
        assert 0.0 <= p <= 1.0
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise 1
        self.add_module("pw1", torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # batchnorm
        if with_bn:
            self.add_module("bn1", torch.nn.BatchNorm2d(num_features=mid_channels, **bn_kwargs))

        # pointwise 2
        self.add_module("pw2", torch.nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # batchnorm
        if with_bn:
            self.add_module("bn2", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))

        # depthwise
        self.add_module("dw", torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        ))

    def _reg_loss(self):
        W = self[0].weight[:, :, 0, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p="fro")


class BSConvS_ModelRegLossMixin():
    def reg_loss(self, alpha=0.1):
        loss = 0.0
        for sub_module in self.modules():
            if hasattr(sub_module, "_reg_loss"):
                loss += sub_module._reg_loss()
        return alpha * loss


# replicate the input RGB channels at the head of the net
class ChannelReplicate(nn.Module):
    def __init__(self, factor=3):
        super(ChannelReplicate, self).__init__()
        self.factor = factor

    def forward(self, input):
        template = input
        for i in range(0, self.factor-1):
            input = torch.cat((template, input), 1)
        return input


# Performer
# for simpler you can just import and use like below
# from performer_pytorch import FastAttention
# https://github.com/lucidrains/performer-pytorch for more details




# Linear Transformer
