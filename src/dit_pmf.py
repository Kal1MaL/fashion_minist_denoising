import torch
import torch.nn as nn
import math


from src.rope import RopePositionEmbedding


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
# 2. 改造后的 DiT Block
# ==========================================
class DiTBlockWithRoPE(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_spatial_tokens):
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

    # 现在的 forward 极其干净，直接接收外部算好的 sin 和 cos
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

        # Attention 与前馈网络保持不变
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


# ==========================================
# 3. 极度模块化的主干网络
# ==========================================
class DiTPixelMeanFlow(nn.Module):
    def __init__(self, in_channels=3, img_size=28, patch_size=2, hidden_dim=256,
                 depth=6, num_heads=8, num_classes=10, num_time_tokens=4, num_cls_tokens=4):
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
            # 开启极微小的抖动来防止过拟合
            jitter_coords=1.05,
            normalize_coords="separate"
        )

        # 其他组件 (Time MLP, Class Embedding 等) 与之前一致
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, hidden_dim * num_time_tokens))
        self.cls_emb = nn.Embedding(num_classes, hidden_dim * num_cls_tokens)

        self.blocks = nn.ModuleList(
            [DiTBlockWithRoPE(hidden_dim, num_heads, self.num_spatial_tokens) for _ in range(depth)])
        self.norm_final = nn.LayerNorm(hidden_dim)
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
        x_pred = self.head(x_spatial)

        x_pred = x_pred.transpose(1, 2).reshape(B, 1, self.grid_size, self.grid_size, self.patch_size, self.patch_size)
        x_pred = x_pred.permute(0, 1, 2, 4, 3, 5).reshape(B, 1, self.img_size, self.img_size)
        return x_pred