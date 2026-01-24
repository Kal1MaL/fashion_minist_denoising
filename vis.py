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
    orig_cwd = get_original_cwd()
    ckpt_dir = os.path.join(orig_cwd, "checkpoints")

    # 优先寻找手动指定的模型，没有则自动找最新的
    # 你可以把这里改成你想测的具体文件
    manual_ckpt = "checkpoints/best-model-epoch=43-val_mse=0.0228.ckpt"
    ckpt_path = os.path.join(orig_cwd, manual_ckpt)

    if not os.path.exists(ckpt_path):
        # 自动搜寻
        list_of_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
        if not list_of_files:
            print(f"❌ 错误: 在 {ckpt_dir} 下没找到任何 .ckpt 模型文件！")
            return
        ckpt_path = max(list_of_files, key=os.path.getctime)

    print(f"🚀 正在加载模型: {ckpt_path}")

    # ==========================================
    # 2. 加载模型
    # ==========================================
    model = ConditionalDDPM.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.to(device)
    model.eval()

    # ==========================================
    # 3. 加载测试数据 (Test Set)
    # ==========================================
    # 逻辑：如果你想用官方测试集 (Clean)，请在这里手动指定 clean_csv
    # 如果是用原来的 noisy test set，保持 cfg.data.test_path 即可

    # 这里我们读取 config 里的配置，但加一个兼容逻辑
    test_path = to_absolute_path(cfg.data.test_path)
    print(f" 读取测试数据: {test_path}")

    # ⚠ 注意：如果你修改了 dataset.py 支持 synthesize_noise，
    # 可以在这里把 synthesize_noise=True 加上
    # 这里默认还是用你的旧逻辑，只传 noisy_csv 或 clean_csv
    dataset = FashionMNISTDenoisingDataset(
        noisy_csv=None,  # 如果是官方测试集，这可能是 clean csv 的路径
        clean_csv=test_path,  # 如果你有对应的 clean csv，填这里
        mode='test',
        synthesize_noise=True
    )

    # ==========================================
    # 4. 可视化推理
    # ==========================================
    num_samples = 10
    # 防止数据量不够
    sample_indices = range(len(dataset))
    if len(dataset) < num_samples:
        indices = sample_indices
    else:
        indices = random.sample(sample_indices, num_samples)

    print(f" 正在生成 {len(indices)} 组对比图...")

    # 预先检查 Dataset 返回的是 Tuple 还是 Tensor，决定画几列
    sample_data = dataset[0]
    has_gt = isinstance(sample_data, (tuple, list)) and len(sample_data) == 3

    # 如果有 GT 画 3 列，没有画 2 列
    cols = 3 if has_gt else 2
    plt.figure(figsize=(4 * cols, 4 * len(indices)))

    for i, idx in enumerate(indices):
        data = dataset[idx]

        # --- 智能解包逻辑 ---
        if has_gt:
            # 返回了 (noisy, clean, label)
            noisy, clean, label = data
            label_text = f"{label.item()}"
        else:
            # 只返回了 noisy
            noisy = data
            clean = None
            label_text = "?"

        # 增加 Batch 维度 (1, 1, 28, 28) 并送入 GPU
        noisy_input = noisy.unsqueeze(0).to(device)

        # --- 模型推理 ---
        with torch.no_grad():
            denoised = model.sample(condition=noisy_input)

        # 转回 CPU numpy
        noisy_np = noisy.squeeze().numpy()
        denoised_np = denoised.cpu().squeeze().numpy()
        clean_np = clean.squeeze().numpy() if clean is not None else None

        # --- 绘图 ---

        # 1. Input (Noisy)
        ax1 = plt.subplot(len(indices), cols, i * cols + 1)
        ax1.imshow(noisy_np, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f"Input (Noisy)\nLabel: {label_text}")
        ax1.axis('off')

        # 2. Output (Restored)
        ax2 = plt.subplot(len(indices), cols, i * cols + 2)
        ax2.imshow(denoised_np, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f"Output (Ours)")
        ax2.axis('off')

        # 3. Ground Truth (如果有)
        if has_gt and clean_np is not None:
            ax3 = plt.subplot(len(indices), cols, i * cols + 3)
            ax3.imshow(clean_np, cmap='gray', vmin=0, vmax=1)
            ax3.set_title(f"Ground Truth")
            ax3.axis('off')

    plt.tight_layout()

    save_path = os.path.join(orig_cwd, "test_visualization.png")
    plt.savefig(save_path)
    print(f" 可视化结果已保存至: {save_path}")

    # plt.show() # 本地运行时可解开


if __name__ == "__main__":
    main()