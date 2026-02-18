import os
import glob
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from src.dataset import FashionMNISTDenoisingDataset
from src.diffusion import ConditionalDDPM
from src.unet import RegressionUNet


def generate_benchmark_csv(model, cfg, output_filename="submission.csv"):
    """
    推理函数：包含 TTA (Test-Time Augmentation) 和 Ensemble
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Starting Final Inference with TTA on {device}...")

    test_dataset = FashionMNISTDenoisingDataset(
        noisy_csv=cfg.data.test_path,
        clean_csv=None,
        mode='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)

    model.to(device)
    model.eval()

    all_denoised_images = []


    # N=1 (TTA后是2张平均)
    # N=3 (TTA后是6张平均)
    N_SAMPLES = 1

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Inference (TTA + Ensemble x{N_SAMPLES})"):
            noisy_imgs = batch.to(device)

            # --- 累加器初始化 ---
            ensemble_accumulator = torch.zeros_like(noisy_imgs)

            # === Part A: 原图采样 ===
            for _ in range(N_SAMPLES):
                sample = model.sample(condition=noisy_imgs)
                ensemble_accumulator += sample

            # === Part B: TTA (水平翻转) 采样 ===
            # 1. 翻转
            noisy_flip = torch.flip(noisy_imgs, dims=[3])

            for _ in range(N_SAMPLES):
                # 2. 采样翻转图
                sample_flip = model.sample(condition=noisy_flip)
                # 3. 翻转回来
                sample_flip_back = torch.flip(sample_flip, dims=[3])
                ensemble_accumulator += sample_flip_back

            # === Part C: 取平均 ===
            # 总次数 = N_SAMPLES * 2
            averaged_img = ensemble_accumulator / (N_SAMPLES * 2)

            # 关键：数值截断 (防止 Cosine Schedule 的数值爆炸)
            averaged_img = torch.clamp(averaged_img, 0.0, 1.0)

            # 展平
            flat_imgs = averaged_img.cpu().numpy().reshape(averaged_img.shape[0], -1)
            all_denoised_images.append(flat_imgs)

    final_data = np.concatenate(all_denoised_images, axis=0)
    print(f" Saving final TTA results to {output_filename}...")
    df = pd.DataFrame(final_data)
    df.to_csv(output_filename, index=False, header=False)
    print(" Submission file generated successfully!")


class DetailedMetricsCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # 获取所有 logged 的指标 (包括 prog_bar=False 的)
        metrics = trainer.callback_metrics

        # 准备打印字符串
        print_msg = f"\nEpoch {trainer.current_epoch}:"

        # 分类打印 Train 和 Val 指标
        train_metrics = []
        val_metrics = []

        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            if "val" in k:
                val_metrics.append(f"{k}: {val:.5f}")
            elif "train" in k:
                train_metrics.append(f"{k}: {val:.5f}")

        # 格式化多行输出
        if train_metrics:
            print_msg += f"\n  [Train] " + " | ".join(train_metrics)
        if val_metrics:
            print_msg += f"\n  [Val]   " + " | ".join(val_metrics)

        # 打印 (tqdm 会自动处理 print，不会打断进度条)
        print(print_msg + "\n")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(42)

    # --- 数据准备 ---
    full_dataset = FashionMNISTDenoisingDataset(cfg.data.train_path, cfg.data.clean_path, mode='train')

    total_size = len(full_dataset)
    val_size = int(total_size * cfg.data.val_split)
    train_size = total_size - val_size

    print(f"Train Size: {train_size}, Validation Size: {val_size}")

    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4,
                            persistent_workers=True)

    # --- 模型初始化 ---
    # model = ConditionalDDPM(cfg)
    model = RegressionUNet(lr=cfg.training.lr)

    metrics_callback = DetailedMetricsCallback()


    early_stop_callback = EarlyStopping(
        monitor="val_mse_image",
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mse_image",
        dirpath="checkpoints",
        filename="best-model-{epoch:02d}-{val_mse_image:.4f}",
        save_top_k=1,
        mode="min"
    )

    # --- 训练器设置 ---
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        callbacks=[early_stop_callback, checkpoint_callback, metrics_callback],
        enable_checkpointing=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=1
    )

    resume_path = None

    # 获取原始目录下的 checkpoints 文件夹
    ckpt_dir = os.path.join(get_original_cwd(), "checkpoints")

    # 搜索所有 .ckpt 文件
    if os.path.exists(ckpt_dir):
        list_of_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))

        if list_of_files:
            # 找到最后修改时间最新的那个文件
            latest_ckpt = max(list_of_files, key=os.path.getctime)
            print(f"\n🚀 检测到 Checkpoint，准备从断点恢复: {latest_ckpt}")
            print("   (如果不希望续训，请手动删除 checkpoints 文件夹或修改代码)\n")
            # resume_path = latest_ckpt
            resume_path = None

    # 开始训练 (传入 ckpt_path 即可实现续训)
    # 如果 resume_path 是 None，它就会从头开始
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)

    # 训练结束
    print("\n" + "=" * 40)
    print("Training Finished (or Early Stopped)!")

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading Best Model from: {best_model_path}")
        model = ConditionalDDPM.load_from_checkpoint(best_model_path)
    print("\nCalculating Best Validation MSE (Local Benchmark)...")
    val_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(val_device)
    model.eval()

    val_result = trainer.validate(model, val_loader, verbose=True)

    final_mse = val_result[0]['val_loss']
    print(f"\n" + "=" * 40)
    print(f" Final Best Validation MSE: {final_mse:.6f}")
    print("=" * 40 + "\n")

    generate_benchmark_csv(model, cfg, output_filename="final_submission.csv")


if __name__ == "__main__":
    main()