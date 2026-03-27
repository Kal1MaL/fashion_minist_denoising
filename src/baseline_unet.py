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
        # Input/Output: [B, out_channels, H, W]
        return self.conv(x)

class SimpleUNet(nn.Module):
    """
    Simple U-Net baseline for 28x28 images.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128]):
        super().__init__()
        self.in_channels = in_channels
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down: 28 -> 14 -> 7
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck: 7x7
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up: 7 -> 14 -> 28
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, y_noisy, cond_dict):
        # Input y_noisy: [B, C, 28, 28]
        if self.in_channels == 3:
            x = torch.cat([y_noisy, cond_dict['y_blur'], cond_dict['y_flip']], dim=1)
        elif self.in_channels == 1:
            x = y_noisy
        else:
            raise ValueError("in_channels must be 1 or 3")

        skip_connections = []

        # Downsample
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsample
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        # Output: [B, out_channels, 28, 28]
        return self.final_conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

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
        out += identity
        out = self.relu(out)
        return out

class ResUNet(nn.Module):
    """
    Residual U-Net baseline for 28x28 images.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128]):
        super().__init__()
        self.in_channels = in_channels
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        curr_channels = in_channels
        for feature in features:
            self.downs.append(ResidualBlock(curr_channels, feature))
            curr_channels = feature

        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ResidualBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, y_noisy, cond_dict):
        # Input y_noisy: [B, C, 28, 28]
        if self.in_channels == 3:
            x = torch.cat([y_noisy, cond_dict['y_blur'], cond_dict['y_flip']], dim=1)
        elif self.in_channels == 1:
            x = y_noisy
        else:
            raise ValueError("in_channels must be 1 or 3")

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        # Output: [B, out_channels, 28, 28]
        return self.final_conv(x)