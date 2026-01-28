import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 基础组件 (保持你原有的，或者确保已定义)
# ==========================================
# 假设 EmbedFC, ResidualConvBlock, UNetDown, UNetUp 你原来就有
# 这里为了完整性我把它们列出来，如果你有引用可以直接用你的

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.layers(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        self.layers = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.layers(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        self.conv = ResidualConvBlock(in_channels // 2 + out_channels, out_channels)  # concat后通道数

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理可能得padding问题，这里假设尺寸是2的倍数，不做额外pad处理
        # x2 是 skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ==========================================
# 2. 新增的高级模块 (New Modules) 🚀
# ==========================================

# --- A. 多头注意力 (替代老的 AttentionBlock) ---
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, 1)
        self.proj = nn.Conv2d(channels, channels, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        h = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(h)


# --- B. CBAM (空间+通道注意力，用于修复边缘) ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(scale))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 确保 reduction 不会导致维度为 0
        mid_channels = max(channels // reduction, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# --- C. SSG (语义-空间门控) ---
class SelectiveSSG(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super().__init__()

        # 1. 语义与噪声的映射 (保持不变)
        self.cls_proj = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.SiLU(),
        )
        self.sigma_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
        )

        # 2. 先验特征生成器 (Prior Generator)
        # 将 [B, 128] 的向量扩展并卷积成与图像特征同维度的 "建议图"
        self.prior_gen = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.SiLU(),
            nn.Conv2d(128, in_channels, 3, padding=1),  # 输出通道对齐特征图
            nn.SiLU()
        )

        # 3. 选择性门控网络 (The Selector)
        # 输入: 原特征图(in) + 先验特征图(in)
        # 输出: 0~1 的门控值 (1 channel)
        # 作用: 决定在每个像素点上，听多少先验的建议
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 1),  # 融合两者信息
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()  # 输出 0-1 概率
        )

        # 可选：先验特征的缩放系数，防止初始化时波动太大
        self.prior_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, class_logits, sigma):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # --- A. 准备先验信息 ---
        # 修复 sigma 维度: [B, 1, 1, 1] -> [B, 1]
        sigma = sigma.view(B, -1)

        cls_probs = torch.softmax(class_logits, dim=-1)
        cls_feat = self.cls_proj(cls_probs)  # [B, 64]
        sigma_feat = self.sigma_proj(sigma)  # [B, 64]

        # 拼接并广播: [B, 128, H, W]
        prior_emb = torch.cat([cls_feat, sigma_feat], dim=1)
        prior_emb_map = prior_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # 生成 "先验建议特征" (Proposed Prior Features)
        # 这是一个包含了类别形状信息的特征图
        prior_features = self.prior_gen(prior_emb_map)  # [B, C, H, W]

        # --- B. 计算选择门控 (Gate) ---
        # 核心：模型看着 "原图特征 x" 和 "先验建议 prior_features"
        # 思考：这俩匹配吗？这个位置我需要先验吗？
        combined = torch.cat([x, prior_features], dim=1)
        gate = self.gate_net(combined)  # [B, 1, H, W]

        # --- C. 柔性注入 ---
        # 残差连接：原图 x + (门控 * 先验)
        # 如果 gate=0 (比如 Dress 的变异区域)，则完全保留 x，不受先验干扰
        # 如果 gate=1 (比如 Sneaker 的暗部)，则强力注入先验

        return x + self.prior_scale * gate * prior_features


# ==========================================
# 3. 完整的 UNet 主类
# ==========================================

class SimpleUNet(nn.Module):
    def __init__(self, in_channels, base_channels=32, n_downs=2, use_class_head=True, use_sigma_emb=True):
        super(SimpleUNet, self).__init__()

        self.use_class_head = use_class_head
        self.use_sigma_emb = use_sigma_emb

        n_feat = base_channels
        self.in_channels = in_channels
        self.n_downs = n_downs

        # 1. 初始卷积
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # 2. 下采样路径
        self.down_blocks = nn.ModuleList()
        for i in range(n_downs):
            self.down_blocks.append(UNetDown(2 ** i * n_feat, 2 ** (i + 1) * n_feat))

        # 3. 瓶颈层 (Bottleneck)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 ** n_downs * n_feat, 2 ** n_downs * n_feat, 7, 1, 0),
            nn.GroupNorm(8, 2 ** n_downs * n_feat),
            nn.GELU()
        )

        # MHA
        self.mid_attn = MultiHeadAttentionBlock(2 ** n_downs * n_feat, num_heads=4)

        # Class Head
        if self.use_class_head:
            feature_dim = (2 ** n_downs) * base_channels
            self.class_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, 128),
                nn.GELU(),
                nn.Linear(128, 10)
            )
        else:
            self.class_head = None

        # 4. 上采样路径 (这里修复了通道数计算错误) 🔧
        self.up_blocks = nn.ModuleList()
        self.cbam_blocks = nn.ModuleList()

        for i in range(n_downs, 0, -1):
            # [修复点 1] 输入通道应该是 2^i，而不是 2^(i+1)
            # 例如 i=2 (Bottleneck出来), 输入是 4*32=128, 对应 2^n_downs
            input_channels = 2 ** i * n_feat
            output_channels = 2 ** (i - 1) * n_feat

            self.up_blocks.append(UNetUp(input_channels, output_channels))
            self.cbam_blocks.append(CBAMBlock(output_channels))

        # SSG
        if self.use_class_head and self.use_sigma_emb:
            self.ssg = SelectiveSSG(n_feat, num_classes=10)
        else:
            self.ssg = None

        self.final_conv = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.GELU(),
            nn.Conv2d(n_feat, 1, 1, 1)
        )

        self.timeembs = nn.ModuleList([EmbedFC(1, 2 ** i * n_feat) for i in range(n_downs, 0, -1)])

        if self.use_sigma_emb:
            self.sigmaembs = nn.ModuleList([EmbedFC(1, 2 ** i * n_feat) for i in range(n_downs, 0, -1)])
        else:
            self.sigmaembs = None

    def forward(self, x, t, condition, sigma, force_class_logits=None):
        # Input fusion
        x_in = torch.cat([x, condition], dim=1)
        x_feat = self.init_conv(x_in)

        # --- Downsampling ---
        downs = []
        for i, down_block in enumerate(self.down_blocks):
            if i == 0:
                downs.append(down_block(x_feat))
            else:
                downs.append(down_block(downs[-1]))

        # --- Bottleneck ---
        # downs[-1] 进入了 bottleneck，所以它不能作为第一个 skip connection
        up = self.up0(self.to_vec(downs[-1]))
        up = self.mid_attn(up)

        # --- Class Head ---
        class_logits = None
        if self.use_class_head:
            class_logits = self.class_head(up)

        if force_class_logits is not None:
            class_logits = force_class_logits

        # --- Upsampling ---
        if self.sigmaembs is None:
            sigma_iterator = [None] * len(self.up_blocks)
        else:
            sigma_iterator = self.sigmaembs

        # [修复点 2] Skip Connection 列表构建 🔧
        # downs 的最后一个元素 downs[-1] 已经被 bottleneck 消化了。
        # 上采样的第一层应该和 downs[-2] 拼接 (14x14)
        # 上采样的第二层应该和 x_feat 拼接 (28x28)
        # 所以 skip_connections 应该是 [downs[0], x_feat]
        skip_connections = downs[:-1][::-1] + [x_feat]

        for up_block, cbam, down, timeemb, sigmaemb in zip(self.up_blocks, self.cbam_blocks, skip_connections,
                                                           self.timeembs, sigma_iterator):
            # 扩展维度以支持广播 [B, C] -> [B, C, 1, 1]
            t_e = timeemb(t)[..., None, None]

            if sigmaemb is not None:
                s_e = sigmaemb(sigma)[..., None, None]
            else:
                s_e = 0

            # 现在 up (128ch) 和 t_e (128ch) 可以相加了
            up = up_block(up + t_e + s_e, down)

            # CBAM
            up = cbam(up)

        # --- SSG Injection ---
        if self.ssg is not None and class_logits is not None:
            up = self.ssg(up, class_logits, sigma)

        # --- Final Output ---
        return self.final_conv(torch.cat([up, x_feat], axis=1)), class_logits