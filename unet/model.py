import torch
import torch.nn as nn
from torchsummary import summary


def single_conv(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))

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
        self.conv_last_single = nn.Conv2d(64, n_class, 1)




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


# m = UNet(3)
# print(summary(m,(1,512,512)))
