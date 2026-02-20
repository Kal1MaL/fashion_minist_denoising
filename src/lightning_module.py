import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd.functional import jvp

# 导入你之前写好的 DiTPixelMeanFlow 主干网络
# 假设保存在 src.models.dit_pmf 中
from src.dit_pmf import DiTPixelMeanFlow


class LitPixelMeanFlow(pl.LightningModule):
    def __init__(self, in_channels, img_size, patch_size, hidden_dim,
                 depth, num_heads, num_classes, num_time_tokens, num_cls_tokens,
                 learning_rate, lambda_pmf, lambda_mse):
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
            num_cls_tokens=num_cls_tokens
        )

    def forward(self, z_t, t, cond_dict):
        """测试/推理阶段直接调用：一步吐出极其干净的原图"""
        return self.net(z_t, t, cond_dict)

    def training_step(self, batch, batch_idx):
        x_clean = batch['x_clean']  # 干净原图 [B, 1, 28, 28]
        y_noisy = batch['y']  # 含噪原图 [B, 1, 28, 28]

        # 将各种先验打包
        cond_dict = {
            'y_blur': batch['y_blur'],
            'y_flip': batch['y_flip'],
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
            t_in_view = t_in.view(-1, 1, 1, 1)
            return (z_in - x_pred) / t_in_view

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
        # 使用 AdamW，搭配较好的 Weight Decay 抑制 Transformer 过拟合
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )
        # 如果需要更激进的收敛，可以在这里配置 CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}