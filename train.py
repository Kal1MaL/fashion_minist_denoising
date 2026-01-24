import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from src.dataset import FashionMNISTDenoisingDataset
from src.diffusion import ConditionalDDPM


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
    model = ConditionalDDPM(cfg)


    early_stop_callback = EarlyStopping(
        monitor="val_mse_image",
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mse_image",
        dirpath="checkpoints",
        filename="best-model-{epoch:02d}-{val_mse:.4f}",
        save_top_k=1,
        mode="min"
    )

    # --- 训练器设置 ---
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=1
    )

    # 开始训练
    trainer.fit(model, train_loader, val_loader)

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