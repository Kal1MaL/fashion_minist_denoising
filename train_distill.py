import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.dataset import FashionMNISTDenoisingDataset
from src.x_predict import XPredictDistiller


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(42)

    # 1. 准备数据 (与原来一致)
    full_dataset = FashionMNISTDenoisingDataset(cfg.data.train_path, cfg.data.clean_path, mode='train')
    total_size = len(full_dataset)
    val_size = int(total_size * cfg.data.val_split)
    train_size = total_size - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    # 2. 载入老师权重，启动蒸馏器！
    TEACHER_CKPT = "checkpoints/best-model-epoch=96-val_mse_image=0.0042.ckpt"  # 填入你的神级权重路径

    model = XPredictDistiller(cfg, teacher_ckpt_path=TEACHER_CKPT)

    # 3. 设置保存回调 (监控 1-step 生成真实度)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_1step_mse",
        dirpath="checkpoints_distilled",
        filename="x-pred-best-{epoch:02d}-{val_1step_mse:.4f}",
        save_top_k=1,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    print("\n🚀 开始 X-Prediction 极速蒸馏训练！")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()