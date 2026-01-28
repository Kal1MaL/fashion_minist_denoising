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

        # ==========================================
        # 1. 模式配置
        # ==========================================
        # 默认为 'ours' (全功能模式)
        self.mode = cfg.model.get("mode", "ours")

        # 根据模式决定开关
        if self.mode == "ours":
            use_class_head = True
            use_sigma_emb = True
            self.use_curriculum = True
        elif self.mode == "sigma":
            use_class_head = False
            use_sigma_emb = True
            self.use_curriculum = False
        else:  # baseline
            use_class_head = False
            use_sigma_emb = False
            self.use_curriculum = False

        # ==========================================
        # 2. 模型初始化
        # ==========================================
        self.model = SimpleUNet(
            in_channels=2,
            base_channels=cfg.model.channels,
            use_class_head=use_class_head,
            use_sigma_emb=use_sigma_emb
        )
        self.timesteps = cfg.model.timesteps

        # ==========================================
        # 3. 注册 Sobel 算子 (用于 Edge Loss)
        # ==========================================
        # 定义为 buffer，随模型移动设备，但不参与梯度更新
        self.register_buffer('sobel_kernel_x',
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_kernel_y',
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

        # ==========================================
        # 4. Diffusion 参数 (Cosine Schedule)
        # ==========================================
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

    # === 辅助函数：提取图像边缘 ===
    def get_edges(self, img):
        # img: (B, 1, H, W)
        edge_x = F.conv2d(img, self.sobel_kernel_x, padding=1)
        edge_y = F.conv2d(img, self.sobel_kernel_y, padding=1)
        # 加上 1e-6 防止 sqrt(0) 梯度 NaN
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

    def _common_step(self, batch, batch_idx, stage='train'):
        condition, clean_img, labels = batch
        batch_size = clean_img.shape[0]
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        # ====================================================
        # 1. t 采样策略 (Curriculum Learning)
        # ====================================================
        if not self.use_curriculum:
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        else:
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
        # 2. 加噪与模型预测
        # ====================================================
        noise = torch.randn_like(clean_img)
        x_t = (
                self.sqrt_alpha_bar[t, None, None, None] * clean_img +
                self.sqrt_one_minus_alpha_bar[t, None, None, None] * noise
        )
        t_float = t.float() / self.timesteps

        predicted_noise, class_logits = self.model(x_t, t_float, condition, current_sigma)

        # ====================================================
        # 🧠 动态权重调度 (Loss Scheduling) - 适配 CBAM + SSG 版
        # ====================================================

        # 1. Edge Loss 权重 (lambda_edge)
        # 新架构自带边缘修复能力，不需要太大的 Edge Loss
        if self.current_epoch < 40:
            # 早期：几乎不加 Edge Loss，让模型先学会看懂图
            lambda_edge = 0.01
            current_beta = 0.1
        elif self.current_epoch < 60:
            # 中期：温和爬坡，辅助 CBAM 收敛
            # 0.01 -> 0.1 (封顶 0.1 就够了，之前 0.2 对新架构来说太大了)
            progress = (self.current_epoch - 20) / (60 - 20)
            lambda_edge = 0.01 + progress * (0.1 - 0.01)
            current_beta = 0.1 - progress * (0.1 - 0.05)
        else:
            # 稳定期
            lambda_edge = 0.005
            current_beta = 0.02

        # ====================================================
        # 3. Loss 计算 (核心升级部分) 🚀
        # ====================================================

        # --- A. 噪声 Loss (Smooth L1) ---
        # 这能像显微镜一样放大暗部细节的梯度
        loss_noise = F.smooth_l1_loss(predicted_noise, noise, beta=current_beta)

        # --- B. 边缘 Loss (Edge Loss) ---
        # 计算预测的原图 x0
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t, None, None, None]

        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        pred_x0 = torch.clamp(pred_x0, 0, 1)  # 截断回图像范围

        # 提取边缘并计算 L1 Loss
        pred_edge = self.get_edges(pred_x0)
        true_edge = self.get_edges(clean_img)
        loss_edge = F.l1_loss(pred_edge, true_edge)

        # --- C. 分类 Loss (辅助头) ---
        if self.mode == "ours" and class_logits is not None:
            # 只在 t 较大的时候计算分类 loss (可选优化，这里保持简单全程计算)
            # 或者用之前的 loss_mask 逻辑
            loss_mask = torch.ones_like(t, dtype=torch.float)
            raw_class_loss = F.cross_entropy(class_logits, labels, reduction='none')
            if loss_mask.sum() > 0:
                class_loss = (raw_class_loss * loss_mask).sum() / loss_mask.sum()
            else:
                class_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            class_loss = torch.tensor(0.0, device=self.device)

        # ====================================================
        # 4. 动态权重 (Loss Scheduling)
        # ====================================================

        recon_loss = loss_noise + lambda_edge * loss_edge

        if self.mode == "ours":
            # SSG 模块非常依赖分类精度，所以我们要保分类头！

            if self.current_epoch < 10:
                # Phase 1: 语义建立期 (Semantic Priming)
                # 必须先把分类头训准了，SSG 才能瞎指挥
                # 此时分类 Loss 权重极大
                w_cls = 2.0
                if stage == 'train':
                    # 甚至可以让 total_loss 主要由 class_loss 主导
                    total_loss = recon_loss + w_cls * class_loss
                else:
                    total_loss = recon_loss + class_loss

            elif self.current_epoch < 40:
                # Phase 2: 联合优化期
                # 分类和去噪并重
                w_cls = 0.3
                total_loss = recon_loss + w_cls * class_loss

            else:
                # Phase 3: 像素精修期
                # 分类头已经准了，降低权重，防止它干扰去噪微调
                w_cls = 0.1
                total_loss = recon_loss + w_cls * class_loss
        else:
            total_loss = recon_loss

        return total_loss, loss_noise, loss_edge, class_loss

    def training_step(self, batch, batch_idx):
        total_loss, loss_noise, loss_edge, class_loss = self._common_step(batch, batch_idx, stage='train')
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_smoothL1_mse", loss_noise, prog_bar=False)
        self.log("train_edge", loss_edge, prog_bar=False)
        if self.mode == "ours":
            self.log("train_cls", class_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, loss_noise, loss_edge, class_loss = self._common_step(batch, batch_idx, stage='val')

        condition, clean_img, _ = batch
        batch_size = clean_img.shape[0]

        # 随机采样 t (为了验证泛化性)
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

        # 反推 x0
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t, None, None, None]
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        pred_x0 = torch.clamp(pred_x0, 0, 1)

        # 计算真正的图像 MSE
        real_image_mse = F.mse_loss(pred_x0, clean_img)
        # ==========================================

        self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val_smooth_l1", loss_noise, prog_bar=False, sync_dist=True)
        self.log("val_edge", loss_edge, prog_bar=False, sync_dist=True)
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
    def sample(self, condition, gt_label_override=None):
        """
        gt_label_override: 如果不为 None，则强制使用这个标签生成 one-hot，
                           而不使用分类头的预测结果。这就是 'Oracle' 模式。
        """
        b, c, h, w = condition.shape
        img = torch.randn((b, c, h, w), device=self.device)
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        # 准备 Oracle Logits (只有在需要的时候才计算)
        oracle_logits = None
        if gt_label_override is not None:
            # 创建一个极端的 logits，让目标类概率接近 1.0
            # [B, 10]
            oracle_logits = torch.full((b, 10), -100.0, device=self.device)
            rows = torch.arange(b, device=self.device)
            oracle_logits[rows, gt_label_override] = 100.0  # 强制激活 GT 类别

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            t_float = t.float() / self.timesteps


            predicted_noise, _ = self.model(
                img, t_float, condition, current_sigma,
                force_class_logits=oracle_logits
            )

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