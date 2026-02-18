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
from src.unet import RegressionUNet


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # ==========================================
    # 1. 自动定位 Checkpoint
    # ==========================================
    orig_cwd = get_original_cwd()
    ckpt_dir = os.path.join(orig_cwd, "checkpoints")

    # 🔥 建议修改：指定你刚刚训练出的最佳模型（30~40 epoch 或最新修复后的）
    manual_ckpt = "b"  # <--- 修改这里文件名

    # 路径拼接
    ckpt_path = os.path.join(orig_cwd, manual_ckpt)

    # 如果指定的文件不存在，自动找最新的
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 指定模型 {manual_ckpt} 不存在，尝试自动搜索...")
        list_of_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
        if not list_of_files:
            print(f"❌ 错误: 在 {ckpt_dir} 下没找到任何 .ckpt 模型文件！")
            return
        # 按修改时间排序找最新的
        ckpt_path = max(list_of_files, key=os.path.getctime)

    print(f"📂 加载模型: {ckpt_path}")

    # ==========================================
    # 2. 加载模型
    # ==========================================
    # strict=False 有助于忽略掉一些不匹配的 key (防止之前改结构留下的缓存问题)
    # 但由于我们结构改动大，最好还是 strict=True 确保加载正确
    model = RegressionUNet.load_from_checkpoint(ckpt_path, cfg=cfg, strict=False)
    model.to(device)
    model.eval()

    # ==========================================
    # 3. 加载测试数据
    # ==========================================
    test_path = to_absolute_path(cfg.data.test_path)
    dataset = FashionMNISTDenoisingDataset(
        noisy_csv="data/fashion-mnist_noisy_test.csv",
        clean_csv="data/fashion-mnist_clean_test.csv",
        mode='test',
        synthesize_noise=True
    )

    # ==========================================
    # 4. 可视化推理 (含 TTA) 🚀
    # ==========================================
    num_samples = 10  # 生成 8 组图

    # 随机采样
    sample_indices = range(len(dataset))
    if len(dataset) < num_samples:
        indices = sample_indices
    else:
        indices = random.sample(sample_indices, num_samples)

    print(f"🎨 正在生成 {len(indices)} 组对比图 (启用 TTA 增强)...")

    # 检查是否有 GT
    sample_data = dataset[0]
    has_gt = isinstance(sample_data, (tuple, list)) and len(sample_data) == 3

    # 列数：Input | Output(TTA) | GT | Error Map
    cols = 4 if has_gt else 2

    # 设置大一点的画布，interpolation='nearest' 防止显示模糊
    fig = plt.figure(figsize=(4 * cols, 4 * len(indices)))

    for i, idx in enumerate(indices):
        data = dataset[idx]

        # --- 解包 ---
        if has_gt:
            noisy, clean, label = data
            label_text = f"Class: {label.item()}"
        else:
            noisy = data
            clean = None
            label_text = "?"

        # 送入 GPU [1, 1, 28, 28]
        noisy_input = noisy.unsqueeze(0).to(device)

        # ==========================================
        # 🔥 核心：TTA 推理 (Test-Time Augmentation)
        # ==========================================
        with torch.no_grad():
            denoised = model.sample(condition=noisy_input)

            # pred_normal = model.sample(condition=noisy_input)
            #
            #
            # noisy_flipped = torch.flip(noisy_input, dims=[3])
            # pred_flipped = model.sample(condition=noisy_flipped)
            # pred_flipped_back = torch.flip(pred_flipped, dims=[3])
            #
            #
            # denoised = (pred_normal + pred_flipped_back) / 2.0

        # 转回 Numpy
        noisy_np = noisy.squeeze().numpy()
        denoised_np = denoised.cpu().squeeze().numpy()
        clean_np = clean.squeeze().numpy() if clean is not None else None

        # --- 绘图 ---
        row_idx = i * cols

        # 1. Input (Noisy)
        ax1 = plt.subplot(len(indices), cols, row_idx + 1)
        ax1.imshow(noisy_np, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        if i == 0: ax1.set_title("Noisy Input")
        ax1.set_ylabel(label_text, rotation=90, size='large')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # 2. Output (TTA)
        ax2 = plt.subplot(len(indices), cols, row_idx + 2)
        ax2.imshow(denoised_np, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        if i == 0: ax2.set_title("Ours (TTA)")
        ax2.axis('off')

        if has_gt and clean_np is not None:
            # 3. Ground Truth
            ax3 = plt.subplot(len(indices), cols, row_idx + 3)
            ax3.imshow(clean_np, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            if i == 0: ax3.set_title("Ground Truth")
            ax3.axis('off')

            # 4. Error Map (残差热力图)
            # 越黑越好，亮的地方表示误差大
            # 这能让你一眼看出鞋底有没有修好
            error_map = np.abs(clean_np - denoised_np)
            ax4 = plt.subplot(len(indices), cols, row_idx + 4)
            # 使用 'inferno' 或 'hot' 色阶，让误差醒目
            im = ax4.imshow(error_map, cmap='inferno', vmin=0, vmax=0.3, interpolation='nearest')
            if i == 0: ax4.set_title("|Diff| (Darker is Better)")
            ax4.axis('off')

    plt.tight_layout()

    save_path = os.path.join(orig_cwd, "test_visualization_tta.png")
    plt.savefig(save_path, dpi=150)  # 高 DPI 保证看清像素
    print(f"✅ 可视化完成！结果已保存至: {save_path}")
    print("   请打开图片检查第四列的 Error Map，如果鞋底位置是黑色的，说明修复成功！")


if __name__ == "__main__":
    main()