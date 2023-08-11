import torch.nn.init as init
import torch.nn as nn
import torch
import torch.nn.functional as F
from dilation_supervised.networks.mine.attention import batch_attention, ASPP_attention, SpatialAttention, dual_direction, SELayer, adam, asdam


class MemoryEfficientMish(nn.Module):
    # Mish activation memory-efficient
    class F(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)

class down_pooling(nn.Module):
    def __init__(self, ch):
        super(down_pooling, self).__init__()
        self.down = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.down(x)
        return x

class conv_block0(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(conv_block0, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        return x3

class conv_block1(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(conv_block1, self).__init__()
        self.conv1 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_in),
            nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2

class conv_block2(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(conv_block2, self).__init__()
        self.conv1 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_in),
            nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )
        self.conv4 = nn.Sequential(
            MemoryEfficientMish(),
            # nn.ReLU(inplace=True),
            nn.BatchNorm2d(channel_out),
            nn.Conv2d(channel_out, channel_out, 3, 1, 1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        x4 = self.conv4(x3 + x2 + x1)
        return x4


class conv_standard(nn.Module):
    def __init__(self, in_ch, middle_ch, out_ch):
        super(conv_standard, self).__init__()
        self.doudouble_conv = nn.Sequential(
            conv_block0(in_ch, middle_ch),
            conv_block2(middle_ch, middle_ch),
            conv_block2(middle_ch, out_ch),
        )

    def forward(self, x):
        x = self.doudouble_conv(x)

        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = conv_block0(1, 32)
        self.conv2 = conv_block1(32, 64)
        self.conv3 = conv_block1(64, 128)
        self.conv4 = conv_block1(128, 128*2)
        self.conv5 = conv_block1(128*2, 128*4)

        self.pool32 = down_pooling(32)
        self.pool64 = down_pooling(64)
        self.pool128 = down_pooling(128)
        self.pool256 = down_pooling(256)

        self.upconv1 = self.upconv(64, 32)
        self.upconv2 = self.upconv(128, 64)
        self.upconv3 = self.upconv(128*2, 128)
        self.upconv4 = self.upconv(128*4, 128*2)

        self.conv6 = conv_block1(128 * 4, 128 * 2)
        self.conv7 = conv_block1(256, 128)
        self.conv8 = conv_block1(128, 64)
        self.conv9 = conv_block1(64, 32)

        self.outc1 = outconv(32, 1)
        self.standard1 = conv_standard(1, 32, 32)
        self.da = dual_direction(channel=32)
        self.outc2 = outconv(32, 1)
        self.standard2 = conv_standard(2, 32, 32)
        self.outc3 = outconv(32, 1)
        self.sigmoid = nn.Sigmoid()

        # self.batch_attention = batch_attention(512)
        # self.adam = adam(32, 512)
        self.asdam = asdam(32, 512)


    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in,channel_out,kernel_size=2,stride=2)

    def forward(self, x):

        # dilation_supervised learning stage
        x1 = self.conv1(x)
        x2 = self.pool32(x1)
        x2 = self.conv2(x2)
        x3 = self.pool64(x2)
        x3 = self.conv3(x3)
        x4 = self.pool128(x3)
        x4 = self.conv4(x4)
        x5 = self.pool256(x4)
        x5 = self.conv5(x5)

        u4 = self.upconv4(x5)
        u4 = self.conv6(torch.cat([u4, x4], dim=1))
        u3 = self.upconv3(u4)
        u3 = self.conv7(torch.cat([u3, x3], dim=1))
        u2 = self.upconv2(u3)
        u2 = self.conv8(torch.cat([u2, x2], dim=1))
        u1 = self.upconv1(u2)
        u1 = self.conv9(torch.cat([u1, x1], dim=1))
        j = self.outc1(u1)
        j1 = F.sigmoid(j)

        #standard_supervised learning stage
        s1 = self.standard1(x)
        s1 = self.asdam(s1)
        # s1 = self.batch_attention(s1)
        # s1 = self.adam(s1)
        s2 = self.outc2(s1)
        s0 = self.sigmoid(s2)

        s2 = self.standard2(torch.cat([j, s2], dim=1))
        s2 = self.da(s1, s2)
        j2 = self.sigmoid(self.outc3(s2))

        return s0, j1, j2







