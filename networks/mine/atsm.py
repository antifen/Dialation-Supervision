import torch
import torch.nn as nn

class LL(nn.Module):
    def __init__(self, input_size=262144, common_size=1):
        super(LL, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, 512),
            nn.Tanh(),
            nn.Linear(512, 32),
            nn.Tanh(),
            nn.Linear(32, common_size)

        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        self.linear.apply(init_weights)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
        # out = self.tanh(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(1).unsqueeze(2)
        # print("outd", out)

        return out

class LL1(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(LL1, self).__init__()
        self.conv1 = nn.Sequential(#25*25
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Conv2d(1, 1, kernel_size=9, stride=1, padding=16,dilation=4),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )
        self.conv11 = nn.Sequential(  # 25*25
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Conv2d(1, 1, kernel_size=9, stride=1, padding=16, dilation=4),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )
        self.avg_pool1 = nn.AvgPool2d(4)

        self.conv2 = nn.Sequential(  # 19*19
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=9, dilation=3),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )
        self.conv22 = nn.Sequential(  # 19*19
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=9, dilation=3),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )
        self.avg_pool2 = nn.AvgPool2d(4)

        self.conv3 = nn.Sequential(  # 13*13
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=6, dilation=3),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )
        self.conv33 = nn.Sequential(  # 13*13
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=6, dilation=3),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )
        self.avg_pool3 = nn.AvgPool2d(4)

        self.conv4 = nn.Sequential(#5*5
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = (self.conv1(x))*(x.clone())
        x1 = self.avg_pool1((self.conv11(x1))*(x1.clone()))
        x2 = (self.conv2(x1)) * (x1.clone())
        x2 = self.avg_pool2((self.conv22(x2)) * (x2.clone()))
        x3 = (self.conv3(x2)) * (x2.clone())
        x3 = self.avg_pool3((self.conv33(x2)) * (x3.clone()))
        x4 = self.conv4(x3)
        x = self.avg_pool(x4)
        x = self.sigmoid(x)

        return x


class LL2(nn.Module):
    def __init__(self, in_ch=1):
        super(LL2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )
        self.avg_pool1 = nn.AvgPool2d(4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        )
        self.avg_pool2 = nn.AvgPool2d(4)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )
        self.avg_pool3 = nn.AvgPool2d(4)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )
        self.avg_pool4 = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.avg_pool1(x1)
        x2 = self.conv2(x1)
        x2 = self.avg_pool2(x2)
        x3 = self.conv3(x2)
        x3 = self.avg_pool3(x3)
        x4 = self.conv4(x3)
        x = self.avg_pool4(x4)
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.sigmoid(x)

        return x

class LL3(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(LL3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.avg_pool1 = nn.AvgPool2d(4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.avg_pool2 = nn.AvgPool2d(4)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.avg_pool3 = nn.AvgPool2d(4)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )
        self.avg_pool4 = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.avg_pool1(x1)
        x2 = self.conv2(x1)
        x2 = self.avg_pool2(x2)
        x3 = self.conv3(x2)
        x3 = self.avg_pool3(x3)
        x4 = self.conv4(x3)
        x = self.avg_pool4(x4)
        x = self.sigmoid(x)

        return x

class LL4(nn.Module):
    def __init__(self, in_ch=1):
        super(LL4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = torch.mean(x3, dim=1, keepdim=True)
        x = self.sigmoid(x)

        return x