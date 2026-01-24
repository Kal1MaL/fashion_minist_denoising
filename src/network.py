import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super(ResidualConvBlock, self).__init__()
        self.is_res = is_res
        self.same_channels = in_channels == out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

        self.handle_channel = None
        if self.is_res and not self.same_channels:
            self.handle_channel = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x_out = self.conv2(self.conv1(x))
        if not self.is_res:
            return x_out
        else:
            if self.handle_channel:
                x_out = x_out + self.handle_channel(x)
            else:
                x_out = x_out + x
            return x_out / 1.414


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, 1)
        self.proj = nn.Conv2d(channels, channels, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, C, -1).permute(0, 2, 1)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1).permute(0, 2, 1)

        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = torch.softmax(w, dim=-1)

        h = torch.bmm(w, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h)


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels, True),
                  ResidualConvBlock(out_channels, out_channels),
                  nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels, True),
            ResidualConvBlock(out_channels, out_channels)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, x_skip):
        if x.shape != x_skip.shape:
            diffY = x_skip.size(2) - x.size(2)
            diffX = x_skip.size(3) - x.size(3)
            x = nn.functional.pad(x, [0, diffX, 0, diffY])
        return self.model(torch.cat([x, x_skip], dim=1))


class EmbedFC(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.model(x)[:, :, None, None]


class SimpleUNet(nn.Module):
    def __init__(self, in_channels, base_channels=32, n_downs=2):
        super(SimpleUNet, self).__init__()

        n_feat = base_channels
        self.in_channels = in_channels
        self.n_downs = n_downs

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down_blocks = nn.ModuleList()
        for i in range(n_downs):
            self.down_blocks.append(UNetDown(2 ** i * n_feat, 2 ** (i + 1) * n_feat))

        # Bottleneck processing
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 ** n_downs * n_feat, 2 ** n_downs * n_feat, 7, 1, 0),
            nn.GroupNorm(8, 2 ** n_downs * n_feat),
            nn.GELU()
        )

        # Attention at Bottleneck
        self.mid_attn = AttentionBlock(2 ** n_downs * n_feat)

        # --- [NEW] Auxiliary Classification Head ---
        # 强制 Bottleneck 学习语义特征
        # Feature dim = base_channels * 2^n_downs (e.g., 32 * 4 = 128)
        feature_dim = (2 ** n_downs) * base_channels
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Pooling -> (B, C, 1, 1)
            nn.Flatten(),  # Flatten -> (B, C)
            nn.Linear(feature_dim, 128),  # Projection
            nn.GELU(),
            nn.Linear(128, 10)  # Output: 10 Classes
        )
        # -------------------------------------------

        self.up_blocks = nn.ModuleList()
        for i in range(n_downs, 0, -1):
            self.up_blocks.append(UNetUp(2 ** (i + 1) * n_feat, 2 ** (i - 1) * n_feat))

        self.final_conv = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.GELU(),
            nn.Conv2d(n_feat, 1, 1, 1)
        )

        # Time Embeddings
        self.timeembs = nn.ModuleList([EmbedFC(1, 2 ** i * n_feat) for i in range(n_downs, 0, -1)])

        # Sigma Embeddings (for Noise Level)
        self.sigmaembs = nn.ModuleList([EmbedFC(1, 2 ** i * n_feat) for i in range(n_downs, 0, -1)])

    def forward(self, x, t, condition, sigma):
        # Concatenate noisy image and condition (original noisy input)
        x = torch.cat([x, condition], dim=1)
        x = self.init_conv(x)

        downs = []
        for i, down_block in enumerate(self.down_blocks):
            if i == 0:
                downs.append(down_block(x))
            else:
                downs.append(down_block(downs[-1]))

        # Bottleneck
        up = self.up0(self.to_vec(downs[-1]))

        # Apply Attention
        up = self.mid_attn(up)

        # --- [NEW] Classification Branch ---
        # 利用 Bottleneck 特征进行分类预测
        class_logits = self.class_head(up)
        # -----------------------------------

        # Decoder Loop with Time & Sigma Conditioning
        for up_block, down, timeemb, sigmaemb in zip(self.up_blocks, downs[::-1], self.timeembs, self.sigmaembs):
            t_e = timeemb(t)  # Time embedding
            s_e = sigmaemb(sigma)  # Sigma embedding

            # Combine features + time + sigma
            up = up_block(up + t_e + s_e, down)

        # Return both Predicted Noise AND Class Logits
        return self.final_conv(torch.cat([up, x], axis=1)), class_logits