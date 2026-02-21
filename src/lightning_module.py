import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd.functional import jvp
import torchvision.transforms.functional as TF
import random

# 导入你之前写好的 DiTPixelMeanFlow 主干网络
# 假设保存在 src.models.dit_pmf 中
from src.dit_pmf import DiTPixelMeanFlow


class LitPixelMeanFlow(pl.LightningModule):
    def __init__(self, in_channels, img_size, patch_size, hidden_dim,
                 depth, num_heads, num_classes, num_time_tokens, num_cls_tokens,
                 learning_rate, lambda_pmf, lambda_mse,drop_path_rate):
        super().__init__()
        # 自动保存所有传入的超参数到 self.hparams
        self.save_hyperparameters()

        # 实例化骨干网络 (带有 DINOv3 RoPE 和 多模态 Tokens)
        self.net = DiTPixelMeanFlow(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=num_classes,
            num_time_tokens=num_time_tokens,
            num_cls_tokens=num_cls_tokens,
            drop_path_rate= drop_path_rate,
        )

    def forward(self, z_t, t, cond_dict):
        """测试/推理阶段直接调用：一步吐出极其干净的原图"""
        return self.net(z_t, t, cond_dict)

    def training_step(self, batch, batch_idx):
        x_clean = batch['x_clean']
        y_noisy = batch['y']

        # 🌟 1. 提前把空间图像先验提出来，准备同步“受刑”
        y_blur = batch['y_blur'].clone()
        y_flip_flag = batch['y_flip'].clone()

        # ==========================================
        # 🌟 2. 联合水平翻转 (同步翻转 y_blur 和 flag)
        # ==========================================
        if random.random() > 0.5:
            x_clean = TF.hflip(x_clean)
            y_noisy = TF.hflip(y_noisy)
            y_blur = TF.hflip(y_blur)  # 👈 必须同步翻转先验图像！
            y_flip_flag = 1.0 - y_flip_flag

        # ==========================================
        # 🌟 3. 联合随机平移 (同步平移 y_blur)
        # ==========================================
        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        x_clean = TF.affine(x_clean, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
        y_noisy = TF.affine(y_noisy, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
        y_blur = TF.affine(y_blur, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)  # 👈 必须同步平移！

        # ==========================================
        # 🌟 4. 无损正交旋转 (同步旋转 y_blur)
        # ==========================================
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            x_clean = torch.rot90(x_clean, k, dims=[-2, -1])
            y_noisy = torch.rot90(y_noisy, k, dims=[-2, -1])
            y_blur = torch.rot90(y_blur, k, dims=[-2, -1])  # 👈 必须同步旋转！

        # ==========================================
        # 🌟 5. 动力学 MixUp (同步混合 y_blur)
        # ==========================================
        if random.random() > 0.3:
            x_clean_roll = torch.roll(x_clean, shifts=1, dims=0)
            y_noisy_roll = torch.roll(y_noisy, shifts=1, dims=0)
            y_blur_roll = torch.roll(y_blur, shifts=1, dims=0)  # 👈 先验图也要 Roll

            lam = torch.distributions.Beta(0.5, 0.5).sample().item()

            x_clean = lam * x_clean + (1.0 - lam) * x_clean_roll
            y_noisy = lam * y_noisy + (1.0 - lam) * y_noisy_roll
            y_blur = lam * y_blur + (1.0 - lam) * y_blur_roll  # 👈 按照相同的比例混合先验图！

            # 注意：对于全局离散条件（label），我们保持原样不融合。
            # 这在学术上叫做 "Label-Preserving Mixup"，强迫模型在混合的图像中，
            # 依然以主图像 (lam 较大的那张) 的标签作为主要的去噪引导，是一种极强的正则化。

        # ==========================================
        # 🌟 6. 重新打包极度安全的 cond_dict
        # ==========================================
        cond_dict = {
            'y_blur': y_blur,
            'y_flip': y_flip_flag,
            'sigma': batch['sigma'],
            'label': batch['label']
        }

        B = x_clean.shape[0]

        # ====================================================
        # pMF 算法核心：Data-to-Data 流匹配 + JVP 重参数化
        # ====================================================

        # 1. 采样时间步 t ~ U(0, 1) (为避免除零，给一个小小的 epsilon)
        t = torch.rand((B,), device=self.device) * 0.999 + 0.001
        t_view = t.view(B, 1, 1, 1)

        # 2. 构造 Data-to-Data 直线插值路径
        # t=0 是干净图像(终点)，t=1 是含噪图像(起点)
        z_t = (1 - t_view) * x_clean + t_view * y_noisy

        # 3. 真实理想的瞬时速度 v (Tangent)
        v_target = y_noisy - x_clean

        # 4. 定义 u_fn 闭包：为了计算 JVP (Jacobian-Vector Product)
        # pMF 论文指出: x(z_t, r, t) = z_t - t * u(z_t, r, t)
        # 所以反推速度: u = (z_t - x_pred) / t
        def u_fn(z_in, t_in):
            x_pred = self.net(z_in, t_in, cond_dict)
            t_safe = torch.clamp(t_in.view(-1, 1, 1, 1), min=1e-2)
            return (z_in - x_pred) / t_safe

        # 5. 计算 JVP (Jacobian-Vector Product)
        # 我们对 u_fn 在 (z_t, t) 处求导。
        # z_t 的变化率(切向量)是 v_target；t 的变化率是 1 (torch.ones_like(t))
        u_theta, dudt = jvp(
            func=u_fn,
            inputs=(z_t, t),
            v=(v_target, torch.ones_like(t)),
            create_graph=True  # 允许梯度反向传播穿过 JVP！
        )

        # 6. 计算复合速度 V (因为 r=0，公式简化为 V = u + t * dudt)
        # stop_gradient 等效于 dudt.detach()
        V_theta = u_theta + t_view * dudt.detach()

        # 7. 再次前向传播获取 x_pred，用于计算 MSE Loss
        # (因为 u_fn 是在 JVP 内部调用的，我们需要显式的 x_pred 来做混合 Loss)
        x_pred_explicit = self.net(z_t, t, cond_dict)

        # ====================================================
        # 混合 Loss 宣判时刻：动力学优雅 + 竞赛功利 的完美缝合
        # ====================================================

        # pMF 动力学损失：拟合真实速度 v
        loss_pmf = F.mse_loss(V_theta, v_target)

        # MSE 暴击损失：强行把 x_prediction 拉向真实图像 x_clean (竞赛 30% 分数核心)
        loss_mse = F.mse_loss(x_pred_explicit, x_clean)

        # 总损失
        loss_total = self.hparams.lambda_pmf * loss_pmf + self.hparams.lambda_mse * loss_mse

        # 日志记录 (能在 TensorBoard 或 Wandb 中实时查看)
        self.log('train_loss', loss_total, prog_bar=True)
        self.log('loss_pmf', loss_pmf)
        self.log('loss_mse', loss_mse, prog_bar=True)

        return loss_total

    def validation_step(self, batch, batch_idx):
        """在验证集上模拟 1-NFE (一步生成) 的真实去噪表现"""
        x_clean = batch['x_clean']
        y_noisy = batch['y']

        cond_dict = {
            'y_blur': batch['y_blur'],
            'y_flip': batch['y_flip'],
            'sigma': batch['sigma'],
            'label': batch['label']
        }

        # 测试时：直接把含噪图 y 当作 t=1 时的 z_1 喂进去，一步推导原图！
        t_val = torch.ones(y_noisy.shape[0], device=self.device)
        x_pred = self(y_noisy, t_val, cond_dict)

        # 计算验证集 MSE
        val_mse = F.mse_loss(x_pred, x_clean)
        self.log('val_mse', val_mse, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # 1. 定义优化器 (AdamW)
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )

        # 2. 获取极其精确的总训练步数 (Lightning 的神仙属性)
        # 相比于手动算 len(dataloader) * max_epochs，它能自动兼容多卡、梯度累加等复杂情况
        total_steps = self.trainer.estimated_stepping_batches

        # 3. 实例化 OneCycleLR
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,  # 在这里，你的 config 里的 lr 就是最高学习率
            total_steps=total_steps,
            pct_start=0.2,  # 用前 10% 的步数做线性 Warmup 预热
            anneal_strategy='cos',  # 后续采用余弦退火
            div_factor=25.0,  # 初始学习率 = max_lr / 25 (极其安全的极低起点，绝不 NaN)
            final_div_factor=1e4  # 最终学习率 = 初始学习率 / 10000 (榨干最后一滴性能)
        )

        # 4. 组装返回字典 (Lightning 的标准格式)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }