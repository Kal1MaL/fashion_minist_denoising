import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .network import SimpleUNet
from .diffusion import ConditionalDDPM


# ==========================================
# 1. 定制版感知损失: 拔下老师的“眼睛”当裁判 (保持不变)
# ==========================================
class FashionPerceptualLoss(nn.Module):
    def __init__(self, teacher_unet):
        super().__init__()
        self.init_conv = teacher_unet.init_conv
        self.down_blocks = teacher_unet.down_blocks
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred_x0, target_x0, condition):
        def extract_features(x):
            x_in = torch.cat([x, condition], dim=1)
            feat = self.init_conv(x_in)
            feats = [feat]
            for down in self.down_blocks:
                feat = down(feat)
                feats.append(feat)
            return feats

        p_feats = extract_features(pred_x0)
        t_feats = extract_features(target_x0)

        loss_perc = 0.0
        for pf, tf in zip(p_feats, t_feats):
            loss_perc += F.mse_loss(pf, tf)

        return loss_perc


# ==========================================
# 2. 回归本质的 pMF 蒸馏模型
# ==========================================
class XPredictDistiller(pl.LightningModule):
    def __init__(self, cfg, teacher_ckpt_path):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 1. 加载老司机 Teacher
        print(f"Loading Teacher Model from {teacher_ckpt_path}...")
        self.teacher = ConditionalDDPM.load_from_checkpoint(teacher_ckpt_path, weights_only=False)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 2. 初始化学生网络 Student (任务是直接预测 x0)
        self.student = SimpleUNet(
            in_channels=2,
            base_channels=cfg.model.channels,
            use_class_head=self.teacher.model.use_class_head,
            use_sigma_emb=self.teacher.model.use_sigma_emb
        )

        self.perc_loss = FashionPerceptualLoss(self.teacher.model)

        self.timesteps = self.teacher.timesteps
        self.sqrt_alpha_bar = self.teacher.sqrt_alpha_bar
        self.sqrt_one_minus_alpha_bar = self.teacher.sqrt_one_minus_alpha_bar

    def training_step(self, batch, batch_idx):
        condition, clean_img, labels = batch
        batch_size = clean_img.shape[0]
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        # --------------------------------------------------
        # 🌟 核心魔法：课程学习 (Curriculum Learning)
        # --------------------------------------------------
        # 根据当前 Epoch 占总 Epoch 的比例，动态计算当前允许的最大噪声步数 max_t
        # 刚开始只加很小的噪声，最后逐渐放开到最大噪声 (self.timesteps - 1)
        progress = self.current_epoch / max(1, self.trainer.max_epochs)

        # 比如：起始阶段让 t 最高只能到 20%，随训练拉满到 100%
        current_max_t = int(self.timesteps * (0.2 + 0.8 * progress))
        current_max_t = min(current_max_t, self.timesteps - 1)

        # 在当前允许的难度范围内随机抽题
        t = torch.randint(0, current_max_t + 1, (batch_size,), device=self.device)
        t_float = t.float() / self.timesteps

        noise = torch.randn_like(clean_img)
        x_t = (
                self.sqrt_alpha_bar[t, None, None, None] * clean_img +
                self.sqrt_one_minus_alpha_bar[t, None, None, None] * noise
        )

        # --------------------------------------------------
        # 老师获取目标 (Teacher Target)
        # --------------------------------------------------
        with torch.no_grad():
            teacher_noise, _ = self.teacher.model(x_t, t_float, condition, current_sigma)
            sqrt_alpha_bar_t = self.sqrt_alpha_bar[t, None, None, None]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t, None, None, None]
            teacher_x0 = (x_t - sqrt_one_minus_alpha_bar_t * teacher_noise) / sqrt_alpha_bar_t
            teacher_x0 = torch.clamp(teacher_x0, 0.0, 1.0)

            # --------------------------------------------------
        # 学生进行 X-Prediction
        # --------------------------------------------------
        student_x0_pred, class_logits = self.student(x_t, t_float, condition, current_sigma)
        student_x0_pred_clamped = torch.clamp(student_x0_pred, 0.0, 1.0)

        # --------------------------------------------------
        # 返璞归真的 Loss 计算
        # --------------------------------------------------
        # 1. 基础 MSE：直接对齐老师的 x0 预测结果，继承老师的生成流形
        loss_mse = F.mse_loss(student_x0_pred_clamped, teacher_x0)

        # 2. 你的定制感知损失：提供高频细节补充，目标也可以直接用 clean_img
        loss_perceptual = self.perc_loss(student_x0_pred_clamped, clean_img, condition)

        loss_cls = F.cross_entropy(class_logits, labels) if class_logits is not None else 0.0

        # 总 Loss，MSE 占主导，感知损失辅助 (权重可调)
        total_loss = loss_mse + 5.0 * loss_perceptual + 0.1 * loss_cls

        self.log("train_total_loss", total_loss, prog_bar=True)
        self.log("train_mse", loss_mse, prog_bar=False)
        # 监控一下当前的训练难度
        self.log("curriculum_max_t", float(current_max_t), prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        condition, clean_img, labels = batch
        b = clean_img.shape[0]
        pure_noise = torch.randn_like(clean_img)

        # 验证时永远是最难的题目：1-step 从纯噪声出图！
        t = torch.full((b,), self.timesteps - 1, device=self.device, dtype=torch.long)
        t_float = t.float() / self.timesteps
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        student_x0_pred, _ = self.student(pure_noise, t_float, condition, current_sigma)
        student_x0_pred = torch.clamp(student_x0_pred, 0.0, 1.0)

        real_image_mse = F.mse_loss(student_x0_pred, clean_img)
        self.log("val_1step_mse", real_image_mse, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    @torch.no_grad()
    def sample(self, condition):
        b, c, h, w = condition.shape
        pure_noise = torch.randn((b, c, h, w), device=self.device)
        t = torch.full((b,), self.timesteps - 1, device=self.device, dtype=torch.long)
        t_float = t.float() / self.timesteps
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        student_x0_pred, _ = self.student(pure_noise, t_float, condition, current_sigma)
        return torch.clamp(student_x0_pred, 0.0, 1.0)