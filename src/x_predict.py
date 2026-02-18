import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .network import SimpleUNet
from .diffusion import ConditionalDDPM
from .unet import RegressionUNet


# ==========================================
# 1. 定制版感知损失: 拔下老师的“眼睛”当裁判
# ==========================================
class FashionPerceptualLoss(nn.Module):
    def __init__(self, teacher_unet):
        super().__init__()
        # 借用老师模型已经训好的下采样模块提取特征
        self.init_conv = teacher_unet.init_conv
        self.down_blocks = teacher_unet.down_blocks

        # 冻结这些层，只做评估不更新
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred_x0, target_x0, condition):
        # 辅助函数：将图像输入网络，截获各层的中间特征
        def extract_features(x):
            x_in = torch.cat([x, condition], dim=1)
            feat = self.init_conv(x_in)
            feats = [feat]  # 第 0 层特征
            for down in self.down_blocks:
                feat = down(feat)
                feats.append(feat)  # 后续深层特征
            return feats

        # 提取特征
        p_feats = extract_features(pred_x0)
        t_feats = extract_features(target_x0)

        # 计算每一层特征的 L2/MSE 损失并累加
        loss_perc = 0.0
        for pf, tf in zip(p_feats, t_feats):
            loss_perc += F.mse_loss(pf, tf)

        return loss_perc


# ==========================================
# 2. X-Prediction 蒸馏模型 (LightningModule)
# ==========================================
class XPredictDistiller(pl.LightningModule):
    def __init__(self, cfg, teacher_ckpt_path):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 1. 加载并冻结你的满级老司机 (Teacher)
        print(f"Loading Teacher Model from {teacher_ckpt_path}...")
        self.teacher = ConditionalDDPM.load_from_checkpoint(teacher_ckpt_path)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 2. 初始化学生网络 (Student)
        # 架构完全一样，但是它的直接输出将被解释为 x_0 (清晰图像)
        self.student = SimpleUNet(
            in_channels=2,
            base_channels=cfg.model.channels,
            use_class_head=self.teacher.model.use_class_head,
            use_sigma_emb=self.teacher.model.use_sigma_emb
        )

        # 3. 注册你的定制感知损失
        self.perc_loss = FashionPerceptualLoss(self.teacher.model)

        # 沿用 Teacher 的调度器参数，用于加噪计算
        self.timesteps = self.teacher.timesteps
        self.sqrt_alpha_bar = self.teacher.sqrt_alpha_bar
        self.sqrt_one_minus_alpha_bar = self.teacher.sqrt_one_minus_alpha_bar

    def training_step(self, batch, batch_idx):
        condition, clean_img, labels = batch
        batch_size = clean_img.shape[0]
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        # --------------------------------------------------
        # A. 构造偏置时间采样 (训练“一步到终点”绝技)
        # --------------------------------------------------
        if torch.rand(1).item() < 0.5:
            # 50% 概率：喂它最难的纯噪声 (t=999)，强制练 1-step
            t = torch.full((batch_size,), self.timesteps - 1, device=self.device, dtype=torch.long)
        else:
            # 50% 概率：正常的全局流形探索，防止模型走偏
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        t_float = t.float() / self.timesteps
        noise = torch.randn_like(clean_img)

        # 生成带噪状态 x_t
        x_t = (
                self.sqrt_alpha_bar[t, None, None, None] * clean_img +
                self.sqrt_one_minus_alpha_bar[t, None, None, None] * noise
        )

        # --------------------------------------------------
        # B. 老师生成完美目标 (Teacher Target)
        # --------------------------------------------------
        with torch.no_grad():
            teacher_noise, _ = self.teacher.model(x_t, t_float, condition, current_sigma)
            # 神奇公式：直接利用扩散模型的数学反解出老师眼里的 x_0
            sqrt_alpha_bar_t = self.sqrt_alpha_bar[t, None, None, None]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t, None, None, None]
            teacher_x0 = (x_t - sqrt_one_minus_alpha_bar_t * teacher_noise) / sqrt_alpha_bar_t
            teacher_x0 = torch.clamp(teacher_x0, 0.0, 1.0)  # 截断回像素空间

        # --------------------------------------------------
        # C. 学生进行 X-Prediction
        # --------------------------------------------------
        student_x0_pred, class_logits = self.student(x_t, t_float, condition, current_sigma)
        student_x0_pred_clamped = torch.clamp(student_x0_pred, 0.0, 1.0)

        # --------------------------------------------------
        # 🔥 核心魔法：靶向解耦 (Target Decoupling)
        # --------------------------------------------------
        # 1. 像素级 Loss 靶子 -> 老师 (Teacher)
        # 作用：老师给的路径是确定的。这能死死稳住衣服的大体轮廓和背景，避免一团模糊。
        loss_recon = F.smooth_l1_loss(student_x0_pred_clamped, teacher_x0, beta=0.05)

        # 2. 感知级 Loss 靶子 -> 绝对真实的 Ground Truth (clean_img)！
        # 作用：我们不要去学老师那微弱的"管子残影"了！
        # 让网络的卷积特征直接去对齐最完美的真实高清图像，强制磨利边缘！
        loss_perceptual = self.perc_loss(student_x0_pred_clamped, clean_img, condition)

        # 3. 辅助头 Loss
        if class_logits is not None:
            loss_cls = F.cross_entropy(class_logits, labels)
        else:
            loss_cls = torch.tensor(0.0, device=self.device)

        # ==================================================
        # D. 终极混合
        # ==================================================
        # 权重配比：以 Recon 为主干，10% 的真实高清特征牵引
        total_loss = loss_recon + 0.1 * loss_perceptual + 0.1 * loss_cls

        # 日志
        self.log("train_total_loss", total_loss, prog_bar=True)
        self.log("train_perc", loss_perceptual, prog_bar=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        condition, clean_img, labels = batch
        # 验证时，我们只看它“1-step从纯噪声还原”的功力！
        b = clean_img.shape[0]
        pure_noise = torch.randn_like(clean_img)
        t = torch.full((b,), self.timesteps - 1, device=self.device, dtype=torch.long)
        t_float = t.float() / self.timesteps
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        student_x0_pred, _ = self.student(pure_noise, t_float, condition, current_sigma)
        student_x0_pred = torch.clamp(student_x0_pred, 0.0, 1.0)

        # 直接和真实干净图像比对 MSE
        real_image_mse = F.mse_loss(student_x0_pred, clean_img)
        self.log("val_1step_mse", real_image_mse, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # 降低蒸馏学习率，增加一点 weight_decay 防止过拟合
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=3e-4, weight_decay=1e-4)

        # 引入余弦退火，让学习率在后期变得极其温柔
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.epochs,  # 假设跑 50 或 100 个 epoch
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }

    @torch.no_grad()
    def sample(self, condition):
        """1-Step 极速推理！不再有 For 循环！"""
        b, c, h, w = condition.shape
        pure_noise = torch.randn((b, c, h, w), device=self.device)
        t = torch.full((b,), self.timesteps - 1, device=self.device, dtype=torch.long)
        t_float = t.float() / self.timesteps
        current_sigma = condition.std(dim=(1, 2, 3), keepdim=True)

        # 一步直接输出结果
        student_x0_pred, _ = self.student(pure_noise, t_float, condition, current_sigma)
        return torch.clamp(student_x0_pred, 0.0, 1.0)