import torch.nn as nn


class Type1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Type2(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.type1 = Type1(channels, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.type1(x)
        out = self.bn(self.conv(out))

        return out + x


class Type3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.type1 = Type1(in_channels, out_channels)

        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(3, 2, padding=1)

        self.shortcut_conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=2, bias=False
        )
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.type1(x)
        out = self.pool(self.bn(self.conv(out)))

        shortcut = self.shortcut_bn(self.shortcut_conv(x))

        return out + shortcut


class Type4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.type1 = Type1(in_channels, out_channels)

        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.type1(x)
        return self.pool(out).flatten(1)


class AdvGuard(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        self.l1 = Type1(in_channels, 64)
        self.l2 = Type1(64, 16)

        self.l3 = Type2(16)
        self.l4 = Type2(16)
        self.l5 = Type2(16)
        self.l6 = Type2(16)
        self.l7 = Type2(16)

        self.l8 = Type3(16, 16)
        self.l9 = Type3(16, 64)
        self.l10 = Type3(64, 128)
        self.l11 = Type3(128, 256)

        self.l12 = Type4(256, 512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)

        return self.fc(x)
