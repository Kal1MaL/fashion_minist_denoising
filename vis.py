import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def visualize_comparison(noisy_path, result_path, num_samples=5, save_path="denoising_result.png"):
    print("正在加载数据，请稍候...")

    # 1. 读取噪声输入 (测试集)
    # 根据任务书，第一列通常是 Label，后面 784 列是像素
    # 这里的 header=0 表示第一行是表头
    df_noisy = pd.read_csv(noisy_path)
    # 取出像素数据：跳过第一列(label)，取后面所有列
    noisy_data = df_noisy.iloc[:, 1:].values

    # 2. 读取你的去噪结果
    # 假设你的结果文件没有表头 (header=None)
    df_result = pd.read_csv(result_path, header=None)
    result_data = df_result.values

    # 3. 随机选择几个样本的索引
    total_images = len(noisy_data)
    indices = random.sample(range(total_images), num_samples)

    # 4. 创建画布
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 6))

    for i, idx in enumerate(indices):
        # --- 处理噪声图 ---
        # Reshape: (784,) -> (28, 28)
        img_noisy = noisy_data[idx].reshape(28, 28)

        axes[0, i].imshow(img_noisy, cmap='gray')
        axes[0, i].set_title(f"Input (Noisy)\nID: {idx}")
        axes[0, i].axis('off')

        # --- 处理去噪图 ---
        # Reshape: (784,) -> (28, 28)
        # 注意：如果你的结果包含 header 或 index，这里可能要调整
        img_result = result_data[idx].reshape(28, 28)

        axes[1, i].imshow(img_result, cmap='gray')
        axes[1, i].set_title(f"Output (Denoised)")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n 可视化完成！图片已保存为: {save_path}")
    print("请在文件浏览器中打开该图片查看效果。")


if __name__ == "__main__":
    # 配置你的文件路径
    NOISY_TEST_CSV = "data/fashion-mnist_noisy_test.csv"  # 官方给的测试集
    MY_SUBMISSION_CSV = "final_submission.csv"  # 你刚才跑出来的结果

    visualize_comparison(NOISY_TEST_CSV, MY_SUBMISSION_CSV, num_samples=8)