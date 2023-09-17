#encoding:utf-8
import torch
#encoding:utf-8
from Fusion_Net_test import *
import math

import numpy as np
class EnergyFusion(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(EnergyFusion, self).__init__()
        laplace_filter1 = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]])
        laplace_filter2 = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)


        self.convx.weight.data.copy_(torch.from_numpy(laplace_filter1))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(laplace_filter2))

        self.conv_v_weight = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,padding='same',bias=False),
            nn.BatchNorm2d(32),
            nn.Sigmoid()
        )
        self.conv_v_weight2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.Sigmoid()
        )
        self.SOB = Sobelxy(channels)
        self.conv1 = ConvBnLeakeyReLu2d(2,2)
        self.conv2 = OneConvLeakeyReLu2d(2,1)
        self.conv3 = ConvBnLeakeyReLu2d(2,1)
        self.BN = nn.BatchNorm2d(channels)
        # self.conv_i_weight = OneConvBnRelu2d(2, 1, 1)
    def forward(self, x, y):
        x1 = self.convx(x)
        y1 = self.convx(y)
        x_w = self.conv_v_weight(x1)
        y_w = self.conv_v_weight2(y1)
        x2 = torch.mul(x, x_w)+x
        y2 = torch.mul(y, y_w)+y
        return torch.cat((x2,y2),1)




class Spa(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spa, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)
class Fuse_model(nn.Module):
    def __init__(self, in_channel, mid_channel1, mid_channel2, out_channel):
        super(Fuse_model, self).__init__()
        self.conv1 = ConvBnReLu2d(in_channel, mid_channel1)
        self.conv2 = ConvBnReLu2d(mid_channel1, mid_channel2)
        self.conv3 = ConvBnReLu2d(mid_channel2, out_channel)#ConvBnReLu2d(mid_channel, out_channel)
        self.conv4 = ConvBnTanh2d(out_channel, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x4


class Dusion_model(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Dusion_model, self).__init__()
        # self.conv1 = ConvBnReLu2d(in_channel, in_channel, groups=in_channel)
        self.conv2 = ConvBnLeakeyReLu2d(in_channel, out_channel)
        # self.conv3 = OneConvBnRelu2d(in_channel, out_channel)#, groups=out_channel

    def forward(self, x):
        x2 = self.conv2(x)
        return x2
class dec_model(nn.Module):
    def __init__(self, in_channel,mid_channel,out_channel,):
        super(dec_model, self).__init__()
        self.conv1 = ConvBnLeakeyReLu2d(mid_channel, out_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, padding='same', bias=False,groups=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.conv3 = OneConvLeakeyReLu2d(out_channel,out_channel)
        self.conv4 = OneConvLeakeyReLu2d(in_channel,mid_channel)
        self.conv5 = OneConvBnRelu2d(in_channel, out_channel)
        self.sob = Sobelxy(in_channel)

    def forward(self, x):
        x1 = self.conv4(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x5 = self.conv3(x3)
        return x5


class dec_mode1(nn.Module):
    def __init__(self, in_channel,mid_channel,out_channel,):
        super(dec_model, self).__init__()
        self.conv1 = ConvBnLeakeyReLu2d(mid_channel, out_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, padding='same', bias=False,groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.conv3 = OneConvLeakeyReLu2d(out_channel,out_channel)
        self.conv4 = OneConvLeakeyReLu2d(in_channel,mid_channel)
        self.conv5 = OneConvBnRelu2d(in_channel, out_channel)
        self.sob = Sobelxy(in_channel)

    def forward(self, x):
        x1 = self.conv4(x)
        lp = self.sob(x)
        lpx = self.conv5(lp)

        x2 = self.conv1(x1)+lpx
        x3 = self.conv2(x2)
        x4 = x2+x3
        x5 = self.conv3(x4)
        return x5

class Dense_model(nn.Module):
    def __init__(self, channel,out_channel):
        super(Dense_model, self).__init__()
        self.conv1 = OneConvLeakeyReLu2d(channel, out_channel)
        self.conv2 = ConvBnLeakeyReLu2d(out_channel,out_channel)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2


class SpatialAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (1, 3, 7), 'kernel size must be 3 or 7'
        if kernel_size == 1:
            padding = 0
        else:
            padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, channel, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, x, y, fg=0):
        if fg == 0:
            x = torch.cat((x, y), 1)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            # min_out, _ = torch.min(x, dim=1, keepdim=True)
            x = torch.cat((avg_out, max_out), dim=1)
            x = self.conv2(x)
        if fg == 1:
            x = torch.cat([x, y], 1)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            # min_out, _ = torch.min(x, dim=1, keepdim=True)
            x = torch.cat((max_out, avg_out), 1)
            x = self.conv1(x)
        return self.relu(x)#改成sigmoid


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class Fusion_model(nn.Module):
    def __init__(self, channel):
        super(Fusion_model, self).__init__()
        self.sp = Spa(7)

    def forward(self, image_ir, image_vis, ir_features, vis_features):
        ad = ir_features+vis_features
        feature_all_weight = torch.split(ad, 1, dim=1)
        flag = 0
        for fe in feature_all_weight:
            feature_f_ok = torch.mul(self.sp(fe),fe)

            if flag == 0:
                n_f_all = feature_f_ok
                flag = 1
                continue
            if flag:
                n_f_all = torch.cat((feature_f_ok, n_f_all), dim=1)

        return n_f_all


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class FusionNet1(nn.Module):
    def __init__(self):
        super(FusionNet1, self).__init__()
        inf_ch = [1, 16, 32, 48, 64]
        self.vis_encoder1 = OneConvLeakeyReLu2d(1, 16)
        self.dec1 = dec_model(1,16,32)

    def forward(self, image_vis):
        x_vis_origin = image_vis[:, :1]
        x = self.dec1(x_vis_origin)

        return x


class FusionNet2(nn.Module):
    def __init__(self):
        super(FusionNet2, self).__init__()
        self.inf_encoder1 = OneConvLeakeyReLu2d(1, 16)
        self.dec = dec_model(1,16, 32)

    def forward(self, image_ir):

        x = self.dec(image_ir)
        return x


class FusionNet3(nn.Module):
    def __init__(self, output):
        super(FusionNet3, self).__init__()
        fusion_ch = [64, 32, 16, 1]
        self.decoder1_1 = Dusion_model(96, 64)
        self.decoder2_1 = Dusion_model(64, 48)
        self.decoder3_1 = Dusion_model(48, 32)
        self.decoder4_1 = ConvBnLeakeyReLu2d(32, 16)
        self.out = ConvBnTanh2d(16, 1)


        self.v_en = FusionNet1()
        self.i_en = FusionNet2()

        self.Fusion_mode1 = EnergyFusion(64)
        self.Fusion_mode2 = EnergyFusion(32)
        self.fus = Fusion_model(64)
        self.dec_1 = ConvBnLeakeyReLu2d(64,48)
        self.dec_2 = ConvBnLeakeyReLu2d(48, 32)
        self.dec_3 = ConvBnLeakeyReLu2d(32, 16)
        self.sc = Sobelxy(1)
        self.conv = OneConvBn(1,16)
        self.atten = CBAMLayer(64)



    def forward(self, image_vis, image_ir):
        ven_3 = self.v_en(image_vis[:, :1])
        ien_3 = self.i_en(image_ir)



        iv_1 = self.Fusion_mode2(ven_3,ien_3)

        x1 = self.dec_1(iv_1)
        x2= self.dec_2(x1)
        x3 = self.dec_3(x2)

        output = self.out(x3)
        return output

