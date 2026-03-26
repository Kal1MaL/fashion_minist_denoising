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
    def __init__(self, hidden_dim, num_heads, num_spatial_tokens, drop_path=0.0,
                 has_global_tokens=True, decouple_proj=False):  # 🌟 新增 decouple_proj 参数
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_spatial_tokens = num_spatial_tokens
        self.has_global_tokens = has_global_tokens
        self.decouple_proj = decouple_proj  # 🌟 记录是否需要独立投影

        # 始终存在的共享/空间投影层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)

        # 🌟 只有当既有全局 Token，又被要求解耦时，才初始化专属层
        if self.has_global_tokens and self.decouple_proj:
            self.norm1_global = nn.LayerNorm(hidden_dim)
            self.qkv_global = nn.Linear(hidden_dim, hidden_dim * 3)

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

        if self.has_global_tokens:
            if self.decouple_proj:
                # ==========================================
                # 🌟 模式 A：前 1/3 层的解耦投影
                # ==========================================
                x_spatial = x[:, :self.num_spatial_tokens, :]
                x_global = x[:, self.num_spatial_tokens:, :]
                N_global = x_global.shape[1]

                h_spatial = self.norm1(x_spatial)
                h_global = self.norm1_global(x_global)

                qkv_spatial = self.qkv(h_spatial).reshape(B, self.num_spatial_tokens, 3, self.num_heads,
                                                          self.head_dim).permute(2, 0, 3, 1, 4)
                qkv_global = self.qkv_global(h_global).reshape(B, N_global, 3, self.num_heads, self.head_dim).permute(2,
                                                                                                                      0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      4)

                q_s, k_s, v_s = qkv_spatial[0], qkv_spatial[1], qkv_spatial[2]
                q_g, k_g, v_g = qkv_global[0], qkv_global[1], qkv_global[2]
            else:
                # ==========================================
                # 🌟 模式 B：后 2/3 层的共享投影
                # ==========================================
                h = self.norm1(x)
                qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # 投影后进行物理切分，为了不让 RoPE 污染全局 Token
                q_s, q_g = q[:, :, :self.num_spatial_tokens, :], q[:, :, self.num_spatial_tokens:, :]
                k_s, k_g = k[:, :, :self.num_spatial_tokens, :], k[:, :, self.num_spatial_tokens:, :]
                v_s, v_g = v[:, :, :self.num_spatial_tokens, :], v[:, :, self.num_spatial_tokens:, :]

            # 🌟 共同逻辑：RoPE 仅作用于空间特征
            q_s, k_s = apply_rope_dino(q_s, k_s, sin, cos)

            # 重新拼接
            q = torch.cat([q_s, q_g], dim=2)
            k = torch.cat([k_s, k_g], dim=2)
            v = torch.cat([v_s, v_g], dim=2)
        else:
            # 纯空间模式 (用于全消融实验)
            h = self.norm1(x)
            qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = apply_rope_dino(q, k, sin, cos)

        # 统一的 Attention 计算
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)

        x = x + self.drop_path(self.proj(out))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DiTPixelMeanFlow(nn.Module):
    def __init__(self, in_channels=3, img_size=28, patch_size=2, hidden_dim=256,
                 depth=6, num_heads=8, num_classes=10, num_cls_tokens=4, drop_path_rate=0.1, num_register_tokens=0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_spatial_tokens = self.grid_size ** 2
        self.hidden_dim = hidden_dim
        self.num_cls_tokens = num_cls_tokens
        self.num_register_tokens = num_register_tokens

        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=patch_size, padding=1)

        self.rope_embed = RopePositionEmbedding(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            jitter_coords=1.05,
            normalize_coords="separate"
        )

        if self.num_cls_tokens > 0:
            self.cls_emb = nn.Embedding(num_classes, hidden_dim * num_cls_tokens)

        if self.num_register_tokens > 0:
            # 定义为可学习的 Parameter，形状为 (1, num_registers, hidden_dim)
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, hidden_dim))
            # 用截断正态分布初始化
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        #  判断是否包含任何全局 Token (条件 Token 或 寄存器 Token)
        has_global = (num_cls_tokens > 0) or (num_register_tokens > 0)

        decouple_layers_limit = depth // 3

        self.blocks = nn.ModuleList([
            DiTBlockWithRoPE(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_spatial_tokens=self.num_spatial_tokens,
                drop_path=dpr[i],
                has_global_tokens=has_global,
                decouple_proj=(i < decouple_layers_limit)
            ) for i in range(depth)
        ])

        self.norm_final = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1 * patch_size * patch_size)

    def forward(self, y_noisy, cond_dict):
        B = y_noisy.shape[0]

        if hasattr(self, 'in_channels') and self.in_channels == 1:
            img_input = y_noisy
        else:
            img_input = torch.cat([y_noisy, cond_dict['y_blur'], cond_dict['y_flip']], dim=1)

        x = self.patch_embed(img_input).flatten(2).transpose(1, 2)

        # 拼接所有全局 Tokens 到序列末尾
        global_tokens = []

        if self.num_cls_tokens > 0:
            c_tokens = self.cls_emb(cond_dict['label']).view(B, -1, self.hidden_dim)
            global_tokens.append(c_tokens)

        if self.num_register_tokens > 0:
            # 将 1 个 batch 的寄存器扩展到 B 个 batch
            r_tokens = self.register_tokens.expand(B, -1, -1)
            global_tokens.append(r_tokens)

        # 如果有全局 Tokens，就和空间 Tokens 拼起来
        if len(global_tokens) > 0:
            x = torch.cat([x] + global_tokens, dim=1)

        sin, cos = self.rope_embed(H=self.grid_size, W=self.grid_size)

        for block in self.blocks:
            x = block(x, sin, cos)

        # Transformer 出口：完美剥离所有全局 Tokens！
        # 无论后面挂了多少 cls token 还是 register token，
        # 我们永远只取前面 self.num_spatial_tokens 个空间特征来还原图像！
        x_spatial = self.norm_final(x[:, :self.num_spatial_tokens, :])
        x_pred = self.head(x_spatial)

        x_pred = x_pred.reshape(B, self.grid_size, self.grid_size, 1, self.patch_size, self.patch_size)
        x_pred = x_pred.permute(0, 3, 1, 4, 2, 5)
        x_pred = x_pred.reshape(B, 1, self.img_size, self.img_size)

        return x_pred