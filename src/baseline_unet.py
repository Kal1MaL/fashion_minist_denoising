import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    """
    专门为 28x28 图像设计的传统 U-Net 绿叶基线。
    完美兼容你现有的前向传播接口 (y_noisy, cond_dict)。
    """

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128]):
        super().__init__()
        self.in_channels = in_channels
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part (28 -> 14 -> 7)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck (7x7)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up part (7 -> 14 -> 28)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, y_noisy, cond_dict):
        if self.in_channels == 3:
            x = torch.cat([y_noisy, cond_dict['y_blur'], cond_dict['y_flip']], dim=1)
        elif self.in_channels == 1:
            x = y_noisy
        else:
            raise ValueError("in_channels must be 1 or 3")

        skip_connections = []

        # 下采样路径
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈层
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # 上采样路径
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            # 拼接跳跃连接
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)


class ResidualBlock(nn.Module):
    """
    标准的残差块：Conv -> BN -> ReLU -> Conv -> BN -> + Skip -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出通道数不一致，需要用 1x1 卷积调整通道数以匹配相加
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 残差连接
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    """
    基于残差块的 U-Net。
    完美兼容 28x28 图像以及 3 通道输入先验 (y_noisy, y_blur, y_flip)。
    """

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128]):
        super().__init__()
        self.in_channels = in_channels
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part (28 -> 14 -> 7)
        curr_channels = in_channels
        for feature in features:
            self.downs.append(ResidualBlock(curr_channels, feature))
            curr_channels = feature

        # Bottleneck (7x7)
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)

        # Up part (7 -> 14 -> 28)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # 注意：拼接后的通道数是 feature * 2
            self.ups.append(ResidualBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, y_noisy, cond_dict):
        if self.in_channels == 3:
            x = torch.cat([y_noisy, cond_dict['y_blur'], cond_dict['y_flip']], dim=1)
        elif self.in_channels == 1:
            x = y_noisy
        else:
            raise ValueError("in_channels must be 1 or 3")

        skip_connections = []

        # 下采样路径
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈层
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # 上采样路径
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            # 拼接跳跃连接
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)