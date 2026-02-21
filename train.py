import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import StochasticWeightAveraging

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 设定全局随机种子保证竞赛可复现性
    pl.seed_everything(42)

    # 1. 实例化你之前写的 DataModule
    datamodule = hydra.utils.instantiate(cfg.data)

    # 2. 实例化这个史诗级的 LightningModule
    model = hydra.utils.instantiate(cfg.model)

    # 3. 设置回调函数：自动保存在验证集上 MSE 最低的权重！
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-pmf-{epoch:02d}-{val_mse:.5f}",
        monitor="val_mse",
        mode="min",
        save_top_k=3
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    swa_callback = StochasticWeightAveraging(swa_lrs=1e-5, swa_epoch_start=0.8)

    # 4. 实例化 Trainer 并启动
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    print("🚀 启动混合先验 pMF-DiT 训练战车...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()