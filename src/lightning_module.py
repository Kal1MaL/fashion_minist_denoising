import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
import random

# 导入你的纯回归版 DiT 骨干网络
from src.dit_pmf import DiTPixelMeanFlow


class LitPixelMeanFlow(pl.LightningModule):
    def __init__(self, in_channels, img_size, patch_size, hidden_dim,
                 depth, num_heads, num_classes, num_cls_tokens,
                 learning_rate, drop_path_rate):
        super().__init__()
        self.save_hyperparameters()

        # 实例化骨干网络 (移除时间 t，专注图像端到端回归)
        self.net = DiTPixelMeanFlow(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=num_classes,
            num_cls_tokens=num_cls_tokens,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, y_noisy, cond_dict):
        """测试/推理阶段直接调用：一步吐出极其干净的原图"""
        return self.net(y_noisy, cond_dict)

    def training_step(self, batch, batch_idx):
        x_clean = batch['x_clean']
        y_noisy = batch['y']

        y_blur = batch['y_blur'].clone()
        y_flip = batch['y_flip'].clone()

        # ==========================================
        # 🌟 1. 基础几何数据增强 (严格同步所有先验图)
        # ==========================================
        if random.random() > 0.5:
            x_clean = TF.hflip(x_clean)
            y_noisy = TF.hflip(y_noisy)
            y_blur = TF.hflip(y_blur)
            y_flip = TF.hflip(y_flip)

        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        if shift_x != 0 or shift_y != 0:
            x_clean = TF.affine(x_clean, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
            y_noisy = TF.affine(y_noisy, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
            y_blur = TF.affine(y_blur, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
            y_flip = TF.affine(y_flip, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)

        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            x_clean = torch.rot90(x_clean, k, dims=[-2, -1])
            y_noisy = torch.rot90(y_noisy, k, dims=[-2, -1])
            y_blur = torch.rot90(y_blur, k, dims=[-2, -1])
            y_flip = torch.rot90(y_flip, k, dims=[-2, -1])

        # ==========================================
        # 🌟 2. 核心魔法：噪声解耦与重采样增强 (Noise Resampling)
        # ==========================================
        # 提取当前 batch 绝对真实的物理噪声分布
        real_noise = y_noisy - x_clean

        aug_prob = random.random()
        if self.current_epoch >= int(self.trainer.max_epochs * 0.8):
            aug_prob = 1.0

        if aug_prob < 0.3:
            # 策略 A: 【跨图组合 - Noise Swapping】
            # 将噪声在 Batch 维度向下滚动一格，把 B 图的真实噪声嫁接给 A 图
            noise_shifted = torch.roll(real_noise, shifts=1, dims=0)
            y_noisy = x_clean + noise_shifted

            # 重新实时生成物理先验
            y_blur = TF.gaussian_blur(y_noisy, kernel_size=[5, 5], sigma=[1.5, 1.5])
            y_flip = TF.hflip(y_noisy)

        elif aug_prob < 0.6:
            # 策略 B: 【采样生成 - Synthetic Noise Injection】
            # 基于当前 batch 真实噪声的方差，凭空采样全新的高斯噪声
            # 计算每张图的真实噪声标准差 sigma (为了支持广播，shape 转为 [B, 1, 1, 1])
            sigma_real = real_noise.reshape(real_noise.shape[0], -1).std(dim=1).reshape(-1, 1, 1, 1)

            # 生成全新的、但统计分布绝对严谨的高斯白噪声
            synthetic_noise = torch.randn_like(x_clean) * sigma_real
            y_noisy = x_clean + synthetic_noise

            # 重新实时生成物理先验
            y_blur = TF.gaussian_blur(y_noisy, kernel_size=[5, 5], sigma=[1.5, 1.5])
            y_flip = TF.hflip(y_noisy)

        # 注意：如果题目原始的加噪逻辑中没有做截断，这里也不需要 clamp，保持正宗的 Gaussian 即可。

        # ==========================================
        # 🌟 3. 打包条件字典，准备前向传播
        # ==========================================
        cond_dict = {
            'y_blur': y_blur,
            'y_flip': y_flip,
            'sigma': batch['sigma'],
            'label': batch['label']
        }

        # ====================================================
        # 纯粹的端到端 MSE 回归
        # ====================================================
        x_pred = self.net(y_noisy, cond_dict)
        loss_mse = F.mse_loss(x_pred, x_clean)

        self.log('train_mse_loss', loss_mse, prog_bar=True)
        return loss_mse

    def validation_step(self, batch, batch_idx):
        """验证集上严格遵循原始噪声，不做任何增强"""
        x_clean = batch['x_clean']
        y_noisy = batch['y']

        cond_dict = {
            'y_blur': batch['y_blur'],
            'y_flip': batch['y_flip'],
            'sigma': batch['sigma'],
            'label': batch['label']
        }

        # 极速单步推导
        x_pred = self(y_noisy, cond_dict)

        # 计算验证集 MSE
        val_mse = F.mse_loss(x_pred, x_clean)
        self.log('val_mse', val_mse, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # 定义优化器 (AdamW)
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )

        total_steps = self.trainer.estimated_stepping_batches

        # 实例化 OneCycleLR
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=total_steps,
            pct_start=0.2,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }