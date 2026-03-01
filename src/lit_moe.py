import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# 导入你的两个异构专家
from src.lightning_module import LitPixelMeanFlow
from src.x_predict import XPredictDistiller

# 导入我们刚刚写好的三种 Router
from src.moe_routers import SimpleCNNRouter, ResCNNRouter, LightweightDenseRouter


class LitMoEFusion(pl.LightningModule):
    def __init__(self, vit_ckpt, flow_ckpt, teacher_ckpt, router_type="resnet", lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        print(f"Loading Frozen Experts... (Router Type: {router_type})")

        # ==========================================
        # 1. 加载并彻底冻结 ViT 专家
        # ==========================================
        self.expert_vit = LitPixelMeanFlow.load_from_checkpoint(vit_ckpt,weights_only=False)
        self.expert_vit.eval()
        for param in self.expert_vit.parameters():
            param.requires_grad = False

        # ==========================================
        # 2. 加载并彻底冻结 Flow 专家 (附带覆写 teacher 路径)
        # ==========================================
        self.expert_flow = XPredictDistiller.load_from_checkpoint(
            flow_ckpt,
            teacher_ckpt_path=teacher_ckpt,
            weights_only=False
        )
        self.expert_flow.eval()
        for param in self.expert_flow.parameters():
            param.requires_grad = False

        # ==========================================
        # 3. 实例化我们唯一需要训练的 Router (门控网络)
        # ==========================================
        if router_type == "cnn":
            self.router = SimpleCNNRouter(in_channels=3)
        elif router_type == "resnet":
            self.router = ResCNNRouter(in_channels=3, base_channels=16, num_blocks=3)
        elif router_type == "dense":
            self.router = LightweightDenseRouter(in_channels=3, growth_rate=8, num_layers=4)
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

    def forward(self, y_noisy, cond_dict):
        # 1. 两位专家各自给出预测 (无梯度)
        with torch.no_grad():
            pred_vit = self.expert_vit(y_noisy, cond_dict)
            pred_flow = self.expert_flow.sample(condition=y_noisy)

        # 2. 裁判 (Router) 出场，生成像素级权重图 alpha
        # alpha 越接近 1，越信任 ViT；越接近 0，越信任 Flow
        alpha = self.router(y_noisy, pred_vit, pred_flow)

        # 3. 融合最终答案
        pred_final = alpha * pred_vit + (1.0 - alpha) * pred_flow
        return pred_final, alpha, pred_vit, pred_flow

    def training_step(self, batch, batch_idx):
        x_clean = batch['x_clean']
        y_noisy = batch['y']
        cond_dict = {
            'y_blur': batch['y_blur'],
            'y_flip': batch['y_flip'],
            'sigma': batch['sigma'],
            'label': batch['label']
        }

        # 前向传播
        pred_final, alpha, _, _ = self(y_noisy, cond_dict)

        # 计算 Loss
        loss = F.mse_loss(pred_final, x_clean)

        self.log("train_moe_mse", loss, prog_bar=True)
        # 记录一下 alpha 的均值，如果一直是 0.88 说明没学到，如果开始波动说明开始精细选特征了
        self.log("train_alpha_mean", alpha.mean(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_clean = batch.get('x_clean', None)
        if x_clean is None:
            return

        y_noisy = batch['y']
        cond_dict = {
            'y_blur': batch['y_blur'],
            'y_flip': batch['y_flip'],
            'sigma': batch['sigma'],
            'label': batch['label']
        }

        pred_final, alpha, pred_vit, pred_flow = self(y_noisy, cond_dict)

        mse_final = F.mse_loss(pred_final, x_clean)
        # 顺便监控一下专家的表现，作为 baseline 对比
        mse_vit = F.mse_loss(pred_vit, x_clean)
        mse_flow = F.mse_loss(pred_flow, x_clean)

        self.log("val_moe_mse", mse_final, prog_bar=True, sync_dist=True)
        self.log("val_vit_baseline", mse_vit, prog_bar=False, sync_dist=True)
        self.log("val_flow_baseline", mse_flow, prog_bar=False, sync_dist=True)
        self.log("val_alpha_mean", alpha.mean(), prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        # 只有 router 的参数参与优化
        optimizer = torch.optim.AdamW(self.router.parameters(), lr=self.lr, weight_decay=1e-4)

        # 引入 OneCycleLR
        # 注意：这里需要 trainer.estimated_stepping_batches 来自动计算总步数
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,  # 峰值学习率
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,  # 前 10% 的步数用于缓慢预热升温 (Warmup)
            div_factor=25,  # 初始学习率 = max_lr / 25 (极其温柔的开局，防止瞬间崩溃)
            final_div_factor=10000  # 训练结束时学习率降到极低，方便收敛
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"  # 必须是 step 级别更新
            }
        }