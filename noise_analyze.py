import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def check_sigma_consistency():
    print("正在逐张图片核查噪声强度...")
    # 读取前 5000 张图
    df_clean = pd.read_csv("data/fashion-mnist_normalised_train.csv", nrows=5000)
    df_noisy = pd.read_csv("data/fashion-mnist_noisy_train.csv", nrows=5000)

    clean = df_clean.iloc[:, 1:].values
    noisy = df_noisy.iloc[:, 1:].values

    # 计算每张图的独立噪声图
    noise_maps = noisy - clean

    # 【关键】计算每一行的标准差 (axis=1)
    # 这代表了每一张图自己的“噪声烈度”
    sigmas_per_image = np.std(noise_maps, axis=1)

    mean_sigma = np.mean(sigmas_per_image)
    std_sigma = np.std(sigmas_per_image)  # 这是“标准差的标准差”，用来衡量波动

    print("\n" + "=" * 40)
    print("🕵️‍♂️ 单图噪声波动分析")
    print("=" * 40)
    print(f"平均 Sigma          : {mean_sigma:.5f}")
    print(f"Sigma 的波动幅度    : {std_sigma:.5f}")
    print(f"Sigma 最小值        : {np.min(sigmas_per_image):.5f}")
    print(f"Sigma 最大值        : {np.max(sigmas_per_image):.5f}")

    # 判定逻辑
    variation_ratio = std_sigma / mean_sigma
    print(f"波动率 (CV)         : {variation_ratio:.2%}")

    if variation_ratio < 0.05:
        print("结论：👉 噪声强度基本恒定。无需 Sigma-Conditioning。")
    else:
        print("结论：👉 噪声强度波动较大！Sigma-Conditioning 会有奇效！")

    # 画图
    plt.figure(figsize=(10, 5))
    plt.hist(sigmas_per_image, bins=100, color='purple', alpha=0.7)
    plt.axvline(mean_sigma, color='k', linestyle='dashed', linewidth=1)
    plt.title(f"Distribution of Per-Image Noise Levels (Mean={mean_sigma:.3f})")
    plt.xlabel("Sigma of individual image")
    plt.ylabel("Count")
    plt.savefig("sigma_distribution.png")
    print("📊 分布图已保存为 sigma_distribution.png")


if __name__ == "__main__":
    check_sigma_consistency()