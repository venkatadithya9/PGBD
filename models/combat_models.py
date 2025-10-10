import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import Module


class UnetGenerator(Module):
    def __init__(self, in_channels=3, nf=64, use_bias=True, out_channel=None):
        super(UnetGenerator, self).__init__()
        if out_channel is None:
            out_channel = in_channels
        self.act = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=(2, 2), mode="bilinear")
        self.conv0_0 = nn.Conv2d(
            in_channels, nf, kernel_size=3, stride=2, padding=1, bias=use_bias
        )
        # self.bn0_0 = nn.InstanceNorm2d(nf)
        self.conv0_1 = nn.Conv2d(
            nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.bn0_1 = nn.InstanceNorm2d(nf)
        self.conv1_0 = nn.Conv2d(
            nf, nf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias
        )
        self.bn1_0 = nn.InstanceNorm2d(nf * 2)
        self.conv1_1 = nn.Conv2d(
            nf * 2, nf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.bn1_1 = nn.InstanceNorm2d(nf * 2)
        self.conv2_0 = nn.Conv2d(
            nf * 2, nf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias
        )
        self.bn2_0 = nn.InstanceNorm2d(nf * 4)
        self.conv2_1 = nn.Conv2d(
            nf * 4, nf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.bn2_1 = nn.InstanceNorm2d(nf * 4)
        self.conv3_0 = nn.Conv2d(
            nf * 4, nf * 8, kernel_size=3, stride=2, padding=1, bias=use_bias
        )
        self.bn3_0 = nn.InstanceNorm2d(nf * 8)
        self.conv3_1 = nn.Conv2d(
            nf * 8, nf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.bn3_1 = nn.InstanceNorm2d(nf * 8)

        # self.upconv3_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn3_2 = nn.InstanceNorm2d(nf*8)
        self.upconv3_1 = nn.Conv2d(
            nf * 8, nf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.upbn3_1 = nn.InstanceNorm2d(nf * 8)
        self.upconv3_0 = nn.Conv2d(
            nf * 8, nf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.upbn3_0 = nn.InstanceNorm2d(nf * 4)
        # self.upconv2_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn2_2 = nn.InstanceNorm2d(nf*4)
        self.upconv2_1 = nn.Conv2d(
            nf * 4, nf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.upbn2_1 = nn.InstanceNorm2d(nf * 4)
        self.upconv2_0 = nn.Conv2d(
            nf * 4, nf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.upbn2_0 = nn.InstanceNorm2d(nf * 2)
        # self.upconv1_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn1_2 = nn.InstanceNorm2d(nf*2)
        self.upconv1_1 = nn.Conv2d(
            nf * 2, nf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.upbn1_1 = nn.InstanceNorm2d(nf * 2)
        self.upconv1_0 = nn.Conv2d(
            nf * 2, nf, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.upbn1_0 = nn.InstanceNorm2d(nf)
        # self.upconv0_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias)
        # self.upbn0_2 = nn.InstanceNorm2d(nf)
        self.upconv0_1 = nn.Conv2d(
            nf, nf, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.upbn0_1 = nn.InstanceNorm2d(nf)
        self.upconv0_0 = nn.Conv2d(
            nf, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias
        )
        self.do = nn.Dropout(p=0.3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        f0 = self.conv0_0(x)
        f0 = self.bn0_1(self.conv0_1(self.act(f0)))
        f1 = self.bn1_0(self.conv1_0(self.act(f0)))
        f1 = self.bn1_1(self.conv1_1(self.act(f1)))
        f2 = self.bn2_0(self.conv2_0(self.act(f1)))
        f2 = self.bn2_1(self.conv2_1(self.act(f2)))
        f3 = self.bn3_0(self.conv3_0(self.act(f2)))
        f3 = self.bn3_1(self.conv3_1(self.act(f3)))
        # f3 = self.do(f3)

        # u3 = self.upbn3_2(self.upconv3_2(self.act(self.up(f3))))
        u3 = self.upbn3_1(self.upconv3_1(self.act(self.up(f3))))
        u3 = self.upbn3_0(self.upconv3_0(self.act(u3))) + f2
        # u2 = self.upbn2_2(self.upconv2_2(self.act(self.up(u3))))
        u2 = self.upbn2_1(self.upconv2_1(self.act(self.up(u3))))
        u2 = self.upbn2_0(self.upconv2_0(self.act(u2))) + f1
        # u1 = self.upbn1_2(self.upconv1_2(self.act(self.up(u2))))
        u1 = self.upbn1_1(self.upconv1_1(self.act(self.up(u2))))
        u1 = self.upbn1_0(self.upconv1_0(self.act(u1))) + f0
        # u0 = self.upbn0_2(self.upconv0_2(self.act(self.up(u1))))
        u0 = self.upbn0_1(self.upconv0_1(self.act(self.up(u1))))
        u0 = self.tanh(self.upconv0_0(self.act(u0)))
        return u0
