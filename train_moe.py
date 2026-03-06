import os
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar

# 导入你在 src 目录下写好的 MoE 闪电模块
from src.lit_moe import LitMoEFusion


class HighPrecisionProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # 获取原始的监控指标字典
        items = super().get_metrics(trainer, model)

        # 1. 无情删掉烦人的 v_num
        items.pop("v_num", None)

        # 2. 精度扩展魔法：把我们关心的指标强制转为 6 位小数
        for key in ["train_moe_mse", "val_moe_mse", "train_alpha_mean"]:
            if key in items and isinstance(items[key], (float, int)):
                items[key] = f"{items[key]:.6f}"

        return items

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. 设定随机种子，保证实验可复现
    pl.seed_everything(42)
    print("🚀 启动 MoE (混合专家) 训练流程...")

    # 2. 实例化数据模块 (完美复用你之前的配置)
    print("📦 正在加载 Fashion MNIST 数据模块...")
    datamodule = hydra.utils.instantiate(cfg.data)

    # ==========================================
    # 3. 设定 Checkpoint 路径与 Router 类型
    # ==========================================
    # 这里你可以随时修改为 "cnn", "resnet", "dense" 来做消融实验
    router_type = "resnet"

    vit_ckpt = "checkpoints/best-pmf-epoch=159-val_mse=0.00325.ckpt"
    flow_ckpt = "checkpoints_distilled/x-pred-best-epoch=130-val_1step_mse=0.0037.ckpt"
    teacher_ckpt = "checkpoints/best-model-epoch=96-val_mse_image=0.0042.ckpt"

    print(f"🧠 正在实例化 MoE 模型 (Router: {router_type})...")
    model = LitMoEFusion(
        vit_ckpt=vit_ckpt,
        flow_ckpt=flow_ckpt,
        teacher_ckpt=teacher_ckpt,
        router_type=router_type,
        lr=1e-4
    )

    # ==========================================
    # 4. 配置训练回调 (Callbacks)
    # ==========================================
    # a. 保存最佳模型 (紧盯 val_moe_mse)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/moe_routers/",
        filename=f"moe-{router_type}-best-{{epoch:02d}}-{{val_moe_mse:.6f}}",
        monitor="val_moe_mse",
        mode="min",
        save_top_k=2,
        save_last=False
    )

    # b. 早停机制 (由于网络极其轻量，极易过拟合，连续5个epoch不降就停)
    early_stop_callback = EarlyStopping(
        monitor="val_moe_mse",
        min_delta=0.000001,
        patience=5,
        verbose=True,
        mode="min"
    )

    # ==========================================
    # 5. 配置 Logger 与 Trainer
    # ==========================================
    logger = TensorBoardLogger("logs", name=f"moe_fusion_{router_type}")

    trainer = pl.Trainer(
        max_epochs=20,  # 因为只需训几层 CNN，收敛极快，20 个 Epoch 绰绰有余
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, HighPrecisionProgressBar(), LearningRateMonitor(logging_interval="step")],
        precision=32,  # 建议保持 32 位精度，因为我们在扣万分之几的 MSE
        log_every_n_steps=10
    )

    # ==========================================
    # 6. 开始训练！(移花接木大法)
    # ==========================================
    print("🔥 万事俱备，开始采用 OOF (Out-of-Fold) 策略训练 Router!")

    # 提取 datamodule 里的各种 loader
    datamodule.setup(stage='fit')
    datamodule.setup(stage='test')

    # 【核心破局点】：
    # 1. 用专家们没见过的 Validation Set 来当作 Router 的训练集！
    router_train_loader = datamodule.val_dataloader()

    # 2. 用 Test Set 来当作 Router 的验证集！
    router_val_loader = datamodule.test_dataloader()

    # 将这两个全新的 loader 喂给 trainer
    trainer.fit(
        model,
        train_dataloaders=router_train_loader,
        val_dataloaders=router_val_loader
    )

    print("✅ 训练结束！请查看 checkpoints/moe_routers 目录获取最佳权重。")

if __name__ == "__main__":
    main()