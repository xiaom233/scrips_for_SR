'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Blocks as Blocks
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Linear(num_feat, f)
        self.conv_f = nn.Linear(f, f)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = nn.Conv2d(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Linear(f, num_feat)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        x = input.permute(0, 2, 3, 1)
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_.permute(0, 3, 1, 2))
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3.permute(0, 2, 3, 1) + cf))
        m = self.sigmoid(c4.permute(0, 3, 1, 2))

        return input * m


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, conv=nn.Conv2d, p=0.25):
        super(RFDB, self).__init__()
        kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Linear(in_channels, self.dc)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        self.c2_d = nn.Linear(self.remaining_channels, self.dc)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.c3_d = nn.Linear(self.remaining_channels, self.dc)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.c5 = nn.Linear(self.dc * 4, in_channels)
        self.att = MultiSpectralAttentionLayer(in_channels, 56, 56, reduction=16,
                                               freq_sel_method='top16')
        # self.esa = ESA(in_channels, conv)

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input.permute(0, 2, 3, 1)))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1.permute(0, 2, 3, 1)))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2.permute(0, 2, 3, 1)))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4.permute(0, 2, 3, 1)], dim=3)
        out = self.c5(out).permute(0, 3, 1, 2)
        out_fused = self.att(out)

        return out_fused


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class RFDNFCA(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=50, num_block=4, num_out_ch=3, upscale=4,
                 conv='DepthWiseConv', upsampler='pixelshuffledirect', p=0.25):
        super(RFDNFCA, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'BSConvS':
            kwargs = {'p': p}
        print(conv)
        if conv == 'DepthWiseConv':
            self.conv = Blocks.DepthWiseConv
        elif conv == 'BSConvU':
            self.conv = Blocks.BSConvU
        elif conv == 'BSConvS':
            self.conv = Blocks.BSConvS
        else:
            self.conv = nn.Conv2d
        self.fea_conv = self.conv(num_in_ch, num_feat, kernel_size=3, **kwargs)

        # RFDB_block_f = functools.partial(RFDB, in_channels=num_feat, conv=self.conv, p=p)
        # RFDB_trunk = make_layer(RFDB_block_f, num_block)
        self.B1 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
        self.B2 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
        self.B3 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
        self.B4 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
        self.B5 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
        self.B6 = RFDB(in_channels=num_feat, conv=self.conv, p=p)
        # self.B7 = RFDB(in_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Linear(num_feat * num_block, num_feat)
        self.GELU = nn.GELU()

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)


        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        # out_B7 = self.B7(out_B6)
        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1)
        out_B = self.c1(trunk.permute(0, 2, 3, 1))
        out_B = self.GELU(out_B.permute(0, 3, 1, 2))
        # print(out_B.shape)
        out_lr = self.c2(out_B) + out_fea


        # output = self.c3(out_lr)
        output = self.upsampler(out_lr)

        return output

