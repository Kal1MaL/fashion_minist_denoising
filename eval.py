import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path, get_original_cwd

# 引入你的 Dataset 和 Model 类
from src.dataset import FashionMNISTDenoisingDataset
from src.diffusion import ConditionalDDPM


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 1. 自动定位 Checkpoint
    # ==========================================
    # Hydra 会修改工作目录，必须用 get_original_cwd() 找回根目录
    orig_cwd = get_original_cwd()
    ckpt_dir = os.path.join(orig_cwd, "checkpoints")

    # 搜索 .ckpt 文件
    list_of_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    if not list_of_files:
        print(f" 错误: 在 {ckpt_dir} 下没找到任何 .ckpt 模型文件！")
        return

    # 找最新的那个
    # latest_ckpt = max(list_of_files, key=os.path.getctime)
    latest_ckpt = "checkpoints/best-model-epoch=43-val_mse=0.0228.ckpt"
    print(f" 正在加载模型: {latest_ckpt}")

    # ==========================================
    # 2. 加载模型
    # ==========================================
    # load_from_checkpoint 会自动处理网络结构初始化
    model = ConditionalDDPM.load_from_checkpoint(latest_ckpt, cfg=cfg)
    model.to(device)
    model.eval()

    # ==========================================
    # 3. 加载数据
    # ==========================================

    noisy_path = to_absolute_path(cfg.data.train_path)
    clean_path = to_absolute_path(cfg.data.clean_path)

    print(f" 读取数据: {noisy_path}")

    # 使用训练集模式，因为只有训练集才有 Clean 图片做对比
    dataset = FashionMNISTDenoisingDataset(
        noisy_csv=noisy_path,
        clean_csv=clean_path,
        mode='train'
    )

    # ==========================================
    # 4. 可视化推理
    # ==========================================
    num_samples = 5
    # 随机选几张图
    indices = random.sample(range(len(dataset)), num_samples)

    print(f" 正在生成 {num_samples} 组对比图...")

    plt.figure(figsize=(12, 4 * num_samples))

    for i, idx in enumerate(indices):
        # Dataset 返回 (noisy, clean, label)
        noisy, clean, label = dataset[idx]

        # 增加 Batch 维度 (1, 1, 28, 28) 并送入 GPU
        noisy_input = noisy.unsqueeze(0).to(device)

        # --- 模型推理 ---
        with torch.no_grad():
            # 调用 sample 函数
            denoised = model.sample(condition=noisy_input)

        # 转回 CPU numpy 用于画图
        noisy_np = noisy.squeeze().numpy()
        clean_np = clean.squeeze().numpy()
        denoised_np = denoised.cpu().squeeze().numpy()

        # --- 绘图 (左中右三联) ---

        # 1. 输入 (Noisy)
        ax1 = plt.subplot(num_samples, 3, i * 3 + 1)
        ax1.imshow(noisy_np, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f"Input (Noisy)\nLabel: {label.item()}")
        ax1.axis('off')

        # 2. 模型输出 (Denoised)
        ax2 = plt.subplot(num_samples, 3, i * 3 + 2)
        ax2.imshow(denoised_np, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f"Output (Ours)")
        ax2.axis('off')

        # 3. 原图 (Ground Truth)
        ax3 = plt.subplot(num_samples, 3, i * 3 + 3)
        ax3.imshow(clean_np, cmap='gray', vmin=0, vmax=1)
        ax3.set_title(f"Ground Truth")
        ax3.axis('off')

    plt.tight_layout()

    # 保存结果到项目根目录
    save_path = os.path.join(orig_cwd, "result_visualization.png")
    plt.savefig(save_path)
    print(f" 可视化结果已保存至: {save_path}")

    # 如果你在本地运行，可以取消注释下面这行直接显示
    # plt.show()


if __name__ == "__main__":
    main()