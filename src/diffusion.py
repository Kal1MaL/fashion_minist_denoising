import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from .network import SimpleUNet


class ConditionalDDPM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 初始化模型 (Residual-UNet + Attention + Sigma + ClassHead)
        # 确保 network.py 里的 SimpleUNet forward 返回 (noise, logits)
        self.model = SimpleUNet(in_channels=2, base_channels=cfg.model.channels)
        self.timesteps = cfg.model.timesteps

        # --- Cosine Schedule 设置 ---
        def cosine_beta_schedule(timesteps, s=0.008):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)

        beta = cosine_beta_schedule(self.timesteps)

        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        # 注册缓冲区 (不作为参数更新，但随模型保存)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1. - alpha_bar))
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x_noisy, t, condition, sigma):
        return self.model(x_noisy, t, condition, sigma)

    def _common_step(self, batch, batch_idx, stage='train'):
        """
        统一的训练/验证步逻辑，包含 Curriculum Learning (课程学习)
        stage: 'train' 或 'val'
        """
        condition, clean_img, labels = batch
        batch_size = clean_img.shape[0]

        # 1. 计算 Sigma (用于 Condition)
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        # ====================================================
        # 🔥 课程学习：动态难度调度器 (Dynamic Difficulty Slider)
        # ====================================================

        warmup_epochs = 5  # 前5轮：预热（只看原图，练分类）
        rampup_length = 10  # 接下来的10轮：爬坡（难度逐渐增加）
        full_start_epoch = warmup_epochs + rampup_length  # 第15轮开始：完全体

        max_t_limit = self.timesteps  # 默认最大难度

        if stage == 'val':
            # 【验证集策略】
            # 验证集必须始终诚实，测试全范围难度，这样才能通过 val_mse 看出真实水平
            max_t_limit = self.timesteps

        else:
            # 【训练集策略】
            if self.current_epoch < warmup_epochs:
                # 阶段一：预热 (Warm-up)
                # 强制 t=0，让模型先在无噪图上学会分类，打通 Encoder
                max_t_limit = 0

            elif self.current_epoch < full_start_epoch:
                # 阶段二：爬坡 (Ramp-up)
                # t 的上限随着 epoch 线性增加
                progress = (self.current_epoch - warmup_epochs) / rampup_length
                max_t_limit = int(progress * self.timesteps)
                max_t_limit = max(10, max_t_limit)  # 至少保留一点点难度

            else:
                # 阶段三：完全体 (Full)
                # 火力全开，随机采样 [0, 1000]
                max_t_limit = self.timesteps

        # ====================================================
        # 🎲 采样时间步 t
        # ====================================================

        if max_t_limit == 0:
            # 作弊模式：纯净图
            t = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        else:
            # 正常/爬坡模式：在允许的范围内随机采样
            t = torch.randint(0, max_t_limit, (batch_size,), device=self.device)

        # 2. 加噪过程
        noise = torch.randn_like(clean_img)
        x_t = (
                self.sqrt_alpha_bar[t, None, None, None] * clean_img +
                self.sqrt_one_minus_alpha_bar[t, None, None, None] * noise
        )

        # 归一化时间步
        t_float = t.float() / self.timesteps

        # 3. 模型前向传播
        predicted_noise, class_logits = self.model(x_t, t_float, condition, current_sigma)

        # 4. 计算 Loss
        # Loss A: 去噪 (MSE)
        noise_loss = F.mse_loss(predicted_noise, noise)

        # Loss B: 分类 (CrossEntropy)
        # 因为我们已经限制了 max_t，所以在当前难度下，所有样本都应该尝试分类
        loss_mask = torch.ones_like(t, dtype=torch.float)
        raw_class_loss = F.cross_entropy(class_logits, labels, reduction='none')

        # 避免除以 0 的安全措施
        valid_samples = loss_mask.sum()
        if valid_samples > 0:
            class_loss = (raw_class_loss * loss_mask).sum() / valid_samples
        else:
            class_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # 5. 总 Loss 动态权重分配
        if self.current_epoch < 5:
            # [阶段1] 预热：只看分类
            total_loss = 5.0 * class_loss
        elif self.current_epoch < 15:
            # [阶段2] 协同：分类辅助去噪
            total_loss = noise_loss + 1.0 * class_loss
        else:
            # [阶段3] 冲刺：分类滚粗，全力去噪！
            # 把权重降到 0.01 甚至 0，让梯度完全由 MSE 主导
            total_loss = noise_loss + 0.01 * class_loss

        return total_loss, noise_loss, class_loss

    def training_step(self, batch, batch_idx):
        # 必须传入 stage='train' 以启用 Curriculum Learning
        total_loss, noise_loss, class_loss = self._common_step(batch, batch_idx, stage='train')

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_mse", noise_loss, prog_bar=True)
        self.log("train_cls", class_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # 必须传入 stage='val' 以禁用作弊，确保 val_mse 真实
        total_loss, noise_loss, class_loss = self._common_step(batch, batch_idx, stage='val')

        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        # 🔥 ModelCheckpoint 监控这个指标
        self.log("val_mse", noise_loss, prog_bar=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        # 1. 定义优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)

        # 2. 定义 One Cycle LR 调度器
        # max_lr: 也就是我们在 config 里设定的 1e-3
        # total_steps: 自动计算总步数 (steps_per_epoch * epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.training.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,  # 前 30% 时间用来热身 (Warm-up)
            div_factor=25,  # 初始 LR = max_lr / 25
            final_div_factor=1e4  # 最终 LR 极其微小，利于收敛
        )

        # 3. 返回 Lightning 要求的格式
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"  # 必须是 step 级更新，不是 epoch 级
            }
        }

    @torch.no_grad()
    def sample(self, condition):
        """
        生成/推理函数
        """
        b, c, h, w = condition.shape
        img = torch.randn((b, c, h, w), device=self.device)

        # 推理时也需要计算 Sigma
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            t_float = t.float() / self.timesteps

            # 推理时只需要 predicted_noise，忽略 class_logits
            predicted_noise, _ = self.model(img, t_float, condition, current_sigma)

            alpha = self.alpha[i]
            alpha_bar = self.alpha_bar[i]
            beta = self.beta[i]

            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)

            img = (1 / torch.sqrt(alpha)) * (
                    img - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise
            ) + torch.sqrt(beta) * noise

            # Clipping 防止数值爆炸
            img = torch.clamp(img, -1.0, 1.0)

        img = torch.clamp(img, 0.0, 1.0)
        return img