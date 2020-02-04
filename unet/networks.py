import torch
import torch.nn as nn
from torchsummary import summary

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


def single_conv(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))




class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, t):
        g1 = self.W_g(g)
        print(g1.shape)
        x1 = self.W_x(t)
        print(x1.shape)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        print("psi shape :{}".format(psi.shape))
        print(t.shape)
        out = t * psi
        # print(print(out.shape)
        return out

class UNet(nn.Module):

    def __init__(self, n_class,attention = False):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.dconv_up1_atten =  nn.Conv2d(64, 64,1)

        self.conv_last_triple = nn.Conv2d(64, 3, 1)
        self.conv_last_single = nn.Conv2d(64, 1, 1)

        self.atten = Attention_block(64,128,192)
        self.attention = attention


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        if self.attention:
            x = self.atten(x,conv1)
            print(x.shape)
            x = self.dconv_up1_atten(x)
        else:
            x = torch.cat([x, conv1], dim=1)

            x = self.dconv_up1(x)

        out_single = self.conv_last_single(x)
        out_triple = self.conv_last_triple(x)
#        print("Out", out.shape)
        return out_single,out_triple

#
# model = UNet(1,attention = True)
# summary(model,(1,512,512))



class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = single_conv(1, 64)
        self.dconv_down2 = single_conv(64, 128)
        self.dconv_down3 = single_conv(128, 256)
        self.dconv_down4 = single_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = single_conv(256 + 512, 256)
        self.dconv_up2 = single_conv(128 + 256, 128)
        self.dconv_up1 = single_conv(128 + 64, 64)
        self.dconv_up1_atten =  nn.Conv2d(64, 64,1)

        self.conv_last_triple = nn.Conv2d(64, 3, 1)
        self.conv_last_single = nn.Conv2d(64, 1, 1)




    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        # if self.attention:
        #     x = self.atten(x,conv1)
        #     print(x.shape)
        #     x = self.dconv_up1_atten(x)
        # else:
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out_single = self.conv_last_single(x)
        # out_triple = self.conv_last_triple(x)
#        print("Out", out.shape)
        return out_single
