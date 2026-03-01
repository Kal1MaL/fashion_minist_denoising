import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================
# 方案 1: 极简 3层 CNN (Baseline：防止过拟合的保底方案)
# ====================================================
class SimpleCNNRouter(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 输入: y_noisy(1) + pred_vit(1) + pred_flow(1) = 3 通道
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        # self._initialize_weights()

    # def _initialize_weights(self):
    #     # 强制偏置初始化为 2.0，让初始 sigmoid(2.0) ≈ 0.88，绝对偏向 ViT (0.0032)
    #     nn.init.zeros_(self.net[-1].weight)
    #     nn.init.constant_(self.net[-1].bias, 2.0)

    def forward(self, y_noisy, pred_vit, pred_flow):
        x = torch.cat([y_noisy, pred_vit, pred_flow], dim=1)  # [B, 3, 28, 28]
        return torch.sigmoid(self.net(x))


# ====================================================
# 方案 2: 轻量级 ResNet Router (扩大感受野，看清宏观结构)
# ====================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, channels)  # 使用 GroupNorm 应对小 Batch Size
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, channels)

    def forward(self, x):
        res = x
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        x = self.gn2(self.conv2(x))
        return F.relu(x + res, inplace=True)


class ResCNNRouter(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, num_blocks=3):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_blocks)])
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

        # 保底初始化
        # nn.init.zeros_(self.final_conv.weight)
        # nn.init.constant_(self.final_conv.bias, 2.0)

    def forward(self, y_noisy, pred_vit, pred_flow):
        x = torch.cat([y_noisy, pred_vit, pred_flow], dim=1)
        x = F.relu(self.init_conv(x), inplace=True)
        x = self.blocks(x)
        return torch.sigmoid(self.final_conv(x))


# ====================================================
# 方案 3: 极轻量 DenseNet Router (极致特征复用，上限最高)
# ====================================================
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(4, growth_rate)
        # 少量 Dropout 防止过度依赖特定像素噪声
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        out = F.relu(self.gn(self.conv(x)), inplace=True)
        out = self.drop(out)
        # 将原始输入和新特征拼接，保留到底层图像的视觉差异记忆
        return torch.cat([x, out], dim=1)


class LightweightDenseRouter(nn.Module):
    def __init__(self, in_channels=3, growth_rate=8, num_layers=4):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)

        self.layers = nn.ModuleList()
        current_channels = 16
        for _ in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate

        self.final_conv = nn.Conv2d(current_channels, 1, kernel_size=3, padding=1)

        # 保底初始化
        # nn.init.zeros_(self.final_conv.weight)
        # nn.init.constant_(self.final_conv.bias, 2.0)

    def forward(self, y_noisy, pred_vit, pred_flow):
        x = torch.cat([y_noisy, pred_vit, pred_flow], dim=1)
        x = F.relu(self.init_conv(x), inplace=True)
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.final_conv(x))