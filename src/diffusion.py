import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from .network import SimpleUNet


class ConditionalDDPM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 1. 判断模式
        # 现有模式: "baseline" | "sigma" | "ours"
        self.mode = cfg.model.get("mode", "ours")

        # 配置开关
        if self.mode == "ours":
            use_class_head = True
            use_sigma_emb = True
            self.use_curriculum = True
        elif self.mode == "sigma":
            # 【新增模式】: 只有 Sigma，没有分类头，没有课程学习
            use_class_head = False
            use_sigma_emb = True
            self.use_curriculum = False
        else:  # baseline
            use_class_head = False
            use_sigma_emb = False
            self.use_curriculum = False

        # 2. 传参给 Network
        self.model = SimpleUNet(
            in_channels=2,
            base_channels=cfg.model.channels,
            use_class_head=use_class_head,
            use_sigma_emb=use_sigma_emb
        )
        self.timesteps = cfg.model.timesteps

        # ... (Cosine Schedule 部分保持不变) ...
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

        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1. - alpha_bar))
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x_noisy, t, condition, sigma):
        return self.model(x_noisy, t, condition, sigma)

    def _common_step(self, batch, batch_idx, stage='train'):
        condition, clean_img, labels = batch
        batch_size = clean_img.shape[0]
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        # ====================================================
        # 🎲 t 采样策略
        # ====================================================
        if not self.use_curriculum:
            # 【Baseline / Sigma】: 没有任何花里胡哨，直接全范围随机
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        else:
            # 【Ours】: 课程学习 (Curriculum Learning)
            warmup_epochs = 5
            rampup_length = 10
            full_start_epoch = warmup_epochs + rampup_length

            max_t_limit = self.timesteps
            if stage == 'train':
                if self.current_epoch < warmup_epochs:
                    max_t_limit = 0
                elif self.current_epoch < full_start_epoch:
                    progress = (self.current_epoch - warmup_epochs) / rampup_length
                    max_t_limit = int(progress * self.timesteps)
                    max_t_limit = max(10, max_t_limit)

            if max_t_limit == 0:
                t = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
            else:
                t = torch.randint(0, max_t_limit, (batch_size,), device=self.device)

        # ====================================================
        # 加噪与前向
        # ====================================================
        noise = torch.randn_like(clean_img)
        x_t = (
                self.sqrt_alpha_bar[t, None, None, None] * clean_img +
                self.sqrt_one_minus_alpha_bar[t, None, None, None] * noise
        )
        t_float = t.float() / self.timesteps

        predicted_noise, class_logits = self.model(x_t, t_float, condition, current_sigma)

        # ====================================================
        # Loss 计算
        # ====================================================
        noise_loss = F.mse_loss(predicted_noise, noise)

        if not self.use_curriculum:
            # 【Baseline / Sigma】: 只看去噪，不管分类
            # 即使 model 输出了 logits (实际上如果是 sigma 模式也不会输出)，我们也不算它的 loss
            total_loss = noise_loss
            class_loss = torch.tensor(0.0, device=self.device)

        else:
            # 【Ours】: 双头 Loss + 动态权重
            loss_mask = torch.ones_like(t, dtype=torch.float)
            if class_logits is not None:
                raw_class_loss = F.cross_entropy(class_logits, labels, reduction='none')
                if loss_mask.sum() > 0:
                    class_loss = (raw_class_loss * loss_mask).sum() / loss_mask.sum()
                else:
                    class_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                class_loss = torch.tensor(0.0, device=self.device)

            # 动态权重逻辑
            warmup_epochs = 5
            if self.current_epoch < warmup_epochs:
                if stage == 'train':
                    total_loss = 5.0 * class_loss
                else:
                    total_loss = noise_loss + class_loss
            elif self.current_epoch < 20:
                total_loss = noise_loss + 1.0 * class_loss
            else:
                total_loss = noise_loss + 0.05 * class_loss

        return total_loss, noise_loss, class_loss

    def training_step(self, batch, batch_idx):
        total_loss, noise_loss, class_loss = self._common_step(batch, batch_idx, stage='train')
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_mse", noise_loss, prog_bar=True)
        if self.mode == "ours":
            self.log("train_cls", class_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # 1. 获取常规 Loss (Noise MSE)
        total_loss, noise_loss, class_loss = self._common_step(batch, batch_idx, stage='val')

        # 2. 计算 Image MSE
        condition, clean_img, labels = batch
        batch_size = clean_img.shape[0]

        # 随机采样一个 t 用于验证
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(clean_img)

        # 构造 x_t
        x_t = (
                self.sqrt_alpha_bar[t, None, None, None] * clean_img +
                self.sqrt_one_minus_alpha_bar[t, None, None, None] * noise
        )
        t_float = t.float() / self.timesteps
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        # 预测噪声
        predicted_noise, _ = self.model(x_t, t_float, condition, current_sigma)

        # === 利用 DDPM 公式从 x_t 和 predicted_noise 逆推 x_0 (Pred Image) ===
        # x_0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t, None, None, None]

        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        pred_x0 = torch.clamp(pred_x0, 0, 1)  # 像素截断

        # 计算图像 MSE
        real_image_mse = F.mse_loss(pred_x0, clean_img)

        # 记录日志
        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_mse_epsilon", noise_loss, prog_bar=True, sync_dist=True)
        self.log("val_mse_image", real_image_mse, prog_bar=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.training.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }

    @torch.no_grad()
    def sample(self, condition):
        b, c, h, w = condition.shape
        img = torch.randn((b, c, h, w), device=self.device)
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            t_float = t.float() / self.timesteps

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
            img = torch.clamp(img, -1.0, 1.0)

        img = torch.clamp(img, 0.0, 1.0)
        return img