import torch
import torch.nn as nn
import math

# 如果你的环境没有 src.rope，确保它在路径中
from src.rope import RopePositionEmbedding


# ==========================================
# DropPath (Stochastic Depth) 基础组件
# ==========================================
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape 处理：确保能在 batch 维度上随机 mask，而序列长度和特征维度保持一致
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# ==========================================
# 1. 通用的 RoPE 旋转应用函数 (解耦逻辑)
# ==========================================
def apply_rope_dino(q, k, sin, cos):
    """
    接收 吐出的 sin, cos [HW, D_head]，并应用到 q, k 上。
    q, k shape: [B, num_heads, HW, head_dim]
    """
    # 调整 sin/cos 形状以支持广播: [1, 1, HW, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)

    # 旋转一半的通用函数
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # 应用旋转矩阵
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


# ==========================================
# 2. 改造后的 DiT Block (已接通 DropPath)
# ==========================================
class DiTBlockWithRoPE(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_spatial_tokens, drop_path=0.0):  # 👈 新增 drop_path 参数
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_spatial_tokens = num_spatial_tokens

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # 👈 实例化 DropPath (如果概率为0则相当于恒等映射)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, sin, cos):
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # --- 完美避开全局 Token ---
        q_spatial, q_global = q[:, :, :self.num_spatial_tokens, :], q[:, :, self.num_spatial_tokens:, :]
        k_spatial, k_global = k[:, :, :self.num_spatial_tokens, :], k[:, :, self.num_spatial_tokens:, :]

        # --- 注入 DINOv3 的魔法 ---
        q_spatial, k_spatial = apply_rope_dino(q_spatial, k_spatial, sin, cos)

        q = torch.cat((q_spatial, q_global), dim=2)
        k = torch.cat((k_spatial, k_global), dim=2)

        # Attention 与前馈网络
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # 👈 给残差连接包裹上 DropPath 装甲
        x = x + self.drop_path(self.proj(out))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ==========================================
# 3. 极度模块化的主干网络 (带线性递增 DropPath)
# ==========================================
class DiTPixelMeanFlow(nn.Module):
    def __init__(self, in_channels=3, img_size=28, patch_size=2, hidden_dim=256,
                 depth=6, num_heads=8, num_classes=10, num_time_tokens=4, num_cls_tokens=4,
                 drop_path_rate=0.1):  # 👈 新增 drop_path_rate 控制全局随机深度概率
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_spatial_tokens = self.grid_size ** 2
        self.hidden_dim = hidden_dim

        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # --- 实例化 DINOv3 的 RoPE 模块 ---
        self.rope_embed = RopePositionEmbedding(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            jitter_coords=1.05,
            normalize_coords="separate"
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * num_time_tokens)
        )
        self.cls_emb = nn.Embedding(num_classes, hidden_dim * num_cls_tokens)

        # 🌟 核心魔法：生成线性递增的 DropPath 概率 (从 0 慢慢涨到 drop_path_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            DiTBlockWithRoPE(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_spatial_tokens=self.num_spatial_tokens,
                drop_path=dpr[i]  # 👈 为每一层传入专属的丢弃概率
            ) for i in range(depth)
        ])

        self.norm_final = nn.LayerNorm(hidden_dim)
        # 预测的通道数是 1 (Fashion-MNIST 原图通道)
        self.head = nn.Linear(hidden_dim, 1 * patch_size * patch_size)

    def forward(self, z_t, t, cond_dict):
        B = z_t.shape[0]

        img_input = torch.cat([z_t, cond_dict['y_blur'], cond_dict['y_flip']], dim=1)
        x = self.patch_embed(img_input).flatten(2).transpose(1, 2)

        t_tokens = self.time_mlp(t.view(-1, 1)).view(B, -1, self.hidden_dim)
        c_tokens = self.cls_emb(cond_dict['label']).view(B, -1, self.hidden_dim)
        x = torch.cat([x, t_tokens, c_tokens], dim=1)

        # --- 动态生成当前网格的 sin, cos ---
        sin, cos = self.rope_embed(H=self.grid_size, W=self.grid_size)

        for block in self.blocks:
            x = block(x, sin, cos)

        x_spatial = self.norm_final(x[:, :self.num_spatial_tokens, :])
        x_pred = self.head(x_spatial)  # shape: [B, H_grid*W_grid, 1*patch_size*patch_size]

        # ==========================================
        # 🛡️ 修复：绝对安全、标准化的 Unpatchify (图像折叠) 逻辑
        # ==========================================
        # 1. 恢复出网格空间和 Patch 内部空间
        # shape 变成: [B, grid_H, grid_W, channels, patch_H, patch_W]
        x_pred = x_pred.reshape(B, self.grid_size, self.grid_size, 1, self.patch_size, self.patch_size)

        # 2. 精准转置，把相邻的维度拼到一起
        # shape 变成: [B, channels, grid_H, patch_H, grid_W, patch_W]
        x_pred = x_pred.permute(0, 3, 1, 4, 2, 5)

        # 3. 展平成最终图像尺寸
        # shape 变成: [B, 1, img_size, img_size]
        x_pred = x_pred.reshape(B, 1, self.img_size, self.img_size)

        return x_pred