import torch
import torch.nn as nn
import torch.nn.functional as F
import hiddenlayer as hl
from torchvision.transforms import Resize

class batch_attention(nn.Module):
    def __init__(self, size):
        super(batch_attention, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0))
        self.conv1_2 = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1))
        self.conv2_1 = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0))
        self.conv2_2 = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1))
        self.softmax = nn.Softmax(dim=3)
        self.avgpooling1 = nn.MaxPool2d(kernel_size=(1, size))
        self.avgpooling2 = nn.MaxPool2d(kernel_size=(size, 1))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, h, w = x.size()
        x1, x2 = x.split(1, dim=0)  # batchsize=2
        x11 = torch.mean(x1, dim=1, keepdim=True)
        x22 = torch.mean(x2, dim=1, keepdim=True)
        x111 = self.avgpooling1(x11)
        x112 = self.avgpooling2(x11)
        x221 = self.avgpooling1(x22)
        x222 = self.avgpooling2(x22)
        x111 = self.conv1_1(x111)
        x112 = self.conv1_2(x112)
        x221 = self.conv2_1(x221)
        x222 = self.conv2_2(x222)
        # cat
        x11_ = torch.cat([x111, ((x112.reshape(1, w)).t()).reshape(1, 1, w, 1)], dim=2)
        x22_ = torch.cat([((x221.reshape(h, 1)).t()).reshape(1, 1, 1, h), x222], dim=3)
        # -&*
        x_c = self.softmax(abs(x11_ - x22_))
        x_c_1 = ((x_c.reshape(h + w, h + w)).t()).reshape(1, 1, h + w, h + w)
        x11_bian = ((torch.matmul(x22_, x_c_1).reshape(1, h + w)).t()).reshape(1, 1, h + w, 1) + x11_
        x_c_2 = x_c
        x22_bian = torch.matmul((((x11_.reshape(h + w, 1)).t()).reshape(1, 1, 1, h + w)), x_c_2) + x22_
        # split
        x111, x112 = x11_bian.split([h, w], dim=2)
        x221, x222 = x22_bian.split([h, w], dim=3)
        # t
        x112 = x112.reshape(w, 1).t().reshape(1, 1, 1, w)
        x221 = x221.reshape(1, h).t().reshape(1, 1, h, 1)
        # weight
        x1 = (x11 * self.sigmoid(x111) * self.sigmoid(x112)) * x1
        x2 = (x22 * self.sigmoid(x221) * self.sigmoid(x222)) * x2
        x = torch.cat([x1, x2], dim=0)
        return x

class ASPP_attention(nn.Module):
    def __init__(self, in_channel):
        super(ASPP_attention, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(5, 1, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
        image_features = torch.mean(image_features, dim=1, keepdim=True)

        atrous_block1 = torch.mean(self.atrous_block1(x), dim=1, keepdim=True)
        atrous_block6 = torch.mean(self.atrous_block6(x), dim=1, keepdim=True)
        atrous_block12 = torch.mean(self.atrous_block12(x), dim=1, keepdim=True)
        atrous_block18 = torch.mean(self.atrous_block18(x), dim=1, keepdim=True)

        cat = torch.cat([image_features, atrous_block1, atrous_block6,
                         atrous_block12, atrous_block18], dim=1)
        attention = self.sigmoid(self.conv_1x1_output(cat))
        return attention

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.sigmoid = nn.Sigmoid()
        self.atrous_block1 = nn.Conv2d(2, 1, 1, 1)
        self.atrous_block6 = nn.Conv2d(2, 1, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(2, 1, 3, 1, padding=12, dilation=12)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x2 = self.atrous_block1(x)
        x3 = self.atrous_block6(x)
        x1 = self.atrous_block12(x)
        x = x1 + x2 + x3
        return self.sigmoid(x)

#Dual_direction Attention Mechanism
class dual_direction(nn.Module):
    def __init__(self, channel):
        super(dual_direction, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.sconv13 = nn.Conv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1))
        self.sconv31 = nn.Conv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, y, x):
        b, c, H, W = x.size()

        x1 = self.sconv13(x)
        x2 = self.sconv31(x)

        y1 = self.sconv13(y)
        y2 = self.sconv31(y)

        map_y13 = torch.sigmoid(self.avg_pool(y1).view(b, c, 1, 1))
        map_y31 = torch.sigmoid(self.avg_pool(y2).view(b, c, 1, 1))

        k = x1 * map_y31 + x2 * map_y13 + x

        return k

#Adaptive Dimensional Attention Mechanism
class adam(nn.Module):
    def __init__(self, out_ch, size):
        super(asdam, self).__init__()

        self.dilated1 = nn.Conv2d(in_channels=out_ch, out_channels=int(out_ch / 2), kernel_size=1, stride=1)
        self.dilated2 = nn.Conv2d(in_channels=out_ch, out_channels=int(out_ch / 2), kernel_size=1, stride=1)
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(int(out_ch / 2), int(out_ch / 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(out_ch / 2)),
            nn.ReLU(inplace=True)
        )
        self.conv3_3_dilated = nn.Sequential(
            nn.Conv2d(int(out_ch / 2), int(out_ch / 2), kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(int(out_ch / 2)),
            nn.ReLU(inplace=True)
        )
        self.conv1_1_1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.se11 = SELayer(out_ch)
        self.se12 = SELayer(size)
        self.se13 = SELayer(size)

    def forward(self, x):
        x_1 = self.conv3_3(self.dilated1(x))
        x_2 = self.conv3_3_dilated(self.dilated2(x))
        xx = self.conv1_1_1(torch.cat([x_1, x_2], dim=1))
        x1_1 = self.se11(xx)
        x1_2 = xx.permute(0, 2, 1, 3)
        x1_2 = self.se12(x1_2).permute(0, 2, 1, 3)
        x1_3 = xx.permute(0, 3, 2, 1)
        x1_3 = self.se13(x1_3).permute(0, 3, 2, 1)
        x = x1_1 + x1_2 + x1_3 + xx + x

        return x

#Adaptive Scale Dimensional Attention Mechanism
class asdam(nn.Module):
    def __init__(self, out_ch, size):
        super(asdam, self).__init__()

        self.dilated1 = nn.Conv2d(in_channels=out_ch, out_channels=int(out_ch / 2), kernel_size=1, stride=1)
        self.dilated2 = nn.Conv2d(in_channels=out_ch, out_channels=int(out_ch / 2), kernel_size=1, stride=1)
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(int(out_ch / 2), int(out_ch / 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(out_ch / 2)),
            nn.ReLU(inplace=True)
        )
        self.conv3_3_dilated = nn.Sequential(
            nn.Conv2d(int(out_ch / 2), int(out_ch / 2), kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(int(out_ch / 2)),
            nn.ReLU(inplace=True)
        )
        self.conv1_1_1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.wc1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 1), stride=1, padding=0)
        self.wc3 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.wc5 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 5), stride=1, padding=(0, 2))

        self.hc1 = nn.Conv2d(in_channels=size, out_channels=size, kernel_size=(1, 1), stride=1, padding=0)
        self.hc3 = nn.Conv2d(in_channels=size, out_channels=size, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.hc5 = nn.Conv2d(in_channels=size, out_channels=size, kernel_size=(1, 5), stride=1, padding=(0, 2))

        self.cc1 = nn.Conv2d(in_channels=size, out_channels=size, kernel_size=(1, 1), stride=1, padding=0)
        self.cc3 = nn.Conv2d(in_channels=size, out_channels=size, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.cc5 = nn.Conv2d(in_channels=size, out_channels=size, kernel_size=(1, 5), stride=1, padding=(0, 2))

        self.se11 = SELayer(out_ch)
        self.se12 = SELayer(out_ch)
        self.se13 = SELayer(out_ch)
        self.se21 = SELayer(size)
        self.se22 = SELayer(size)
        self.se23 = SELayer(size)
        self.se31 = SELayer(size)
        self.se32 = SELayer(size)
        self.se33 = SELayer(size)

    def forward(self, x):
        x_1 = self.conv3_3(self.dilated1(x))
        x_2 = self.conv3_3_dilated(self.dilated2(x))
        xx = self.conv1_1_1(torch.cat([x_1, x_2], dim=1))

        wc1 = self.wc1(xx)
        wc3 = self.wc3(xx)
        wc5 = self.wc5(xx)
        x1_1 = self.se11(wc1) * wc1 + self.se12(wc3) * wc3 + self.se13(wc5) * wc5 + xx

        x1_2 = xx.permute(0, 2, 1, 3)
        hc1 = self.hc1(x1_2)
        hc3 = self.hc3(x1_2)
        hc5 = self.hc5(x1_2)
        x1_2 = self.se21(hc1) * hc1 + self.se22(hc3) * hc3 + self.se23(hc5) * hc5 + x1_2
        x1_2 = x1_2.permute(0, 2, 1, 3)

        x1_3 = xx.permute(0, 3, 2, 1)
        cc1 = self.cc1(x1_3)
        cc3 = self.cc3(x1_3)
        cc5 = self.cc5(x1_3)
        x1_3 = self.se31(cc1) * cc1 + self.se32(cc3) * cc3 + self.se33(cc5) * cc5 + x1_3
        x1_3 = x1_3.permute(0, 3, 2, 1)

        x = x1_1 + x1_2 + x1_3 + xx + x

        return x
