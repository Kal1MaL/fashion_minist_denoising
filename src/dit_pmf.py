import torch
import torch.nn as nn
import math

from src.rope import RopePositionEmbedding

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

def apply_rope_dino(q, k, sin, cos):
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out

class DiTBlockWithRoPE(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_spatial_tokens, drop_path=0.0):
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

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, sin, cos):
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_spatial, q_global = q[:, :, :self.num_spatial_tokens, :], q[:, :, self.num_spatial_tokens:, :]
        k_spatial, k_global = k[:, :, :self.num_spatial_tokens, :], k[:, :, self.num_spatial_tokens:, :]

        q_spatial, k_spatial = apply_rope_dino(q_spatial, k_spatial, sin, cos)

        q = torch.cat((q_spatial, q_global), dim=2)
        k = torch.cat((k_spatial, k_global), dim=2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)

        x = x + self.drop_path(self.proj(out))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DiTPixelMeanFlow(nn.Module):
    """
    现在它是一个纯粹的强力端到端自编码器，移除了时间步 t 的输入，专注图像回归。
    """
    def __init__(self, in_channels=3, img_size=28, patch_size=2, hidden_dim=256,
                 depth=6, num_heads=8, num_classes=10, num_cls_tokens=4, drop_path_rate=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_spatial_tokens = self.grid_size ** 2
        self.hidden_dim = hidden_dim

        # 🌟 优化：重叠卷积 (kernel=3, padding=1)，消除28x28小图分块带来的网格伪影
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=patch_size, padding=1)

        self.rope_embed = RopePositionEmbedding(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            jitter_coords=1.05,
            normalize_coords="separate"
        )

        self.cls_emb = nn.Embedding(num_classes, hidden_dim * num_cls_tokens)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            DiTBlockWithRoPE(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_spatial_tokens=self.num_spatial_tokens,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])

        self.norm_final = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1 * patch_size * patch_size)

    def forward(self, y_noisy, cond_dict):
        # 移除了 t，直接输入含噪图像 y_noisy
        B = y_noisy.shape[0]

        img_input = torch.cat([y_noisy, cond_dict['y_blur'], cond_dict['y_flip']], dim=1)
        x = self.patch_embed(img_input).flatten(2).transpose(1, 2)

        c_tokens = self.cls_emb(cond_dict['label']).view(B, -1, self.hidden_dim)
        x = torch.cat([x, c_tokens], dim=1)  # 只拼接 cls tokens

        sin, cos = self.rope_embed(H=self.grid_size, W=self.grid_size)

        for block in self.blocks:
            x = block(x, sin, cos)

        x_spatial = self.norm_final(x[:, :self.num_spatial_tokens, :])
        x_pred = self.head(x_spatial)

        x_pred = x_pred.reshape(B, self.grid_size, self.grid_size, 1, self.patch_size, self.patch_size)
        x_pred = x_pred.permute(0, 3, 1, 4, 2, 5)
        x_pred = x_pred.reshape(B, 1, self.img_size, self.img_size)

        return x_pred