import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 优化后的数据加载 ---
clean_path = 'data/fashion-mnist_clean_train.csv'
noisy_path = 'data/fashion-mnist_noisy_train.csv'

print("正在加载数据集 (自动解析表头)...")
# header=0 表示把第一行作为列名，不再计入数据体
df_clean = pd.read_csv(clean_path, header=0)
df_noisy = pd.read_csv(noisy_path, header=0)

# 第 0 列是标签 (类别 0~9)
labels = df_clean.iloc[:, 0].values.astype(np.int64)

# 第 1 列到最后是 784 维像素特征
X_clean = df_clean.iloc[:, 1:].values.astype(np.float32)
X_noisy = df_noisy.iloc[:, 1:].values.astype(np.float32)

print(f"数据加载完毕！")
print(f"-> 干净特征矩阵维度: {X_clean.shape} (完美符合 60000 张图)")
print(f"-> 标签维度: {labels.shape}")

# ==========================================
# 核心任务 A：重新进行精准的噪声分析
# ==========================================
print("\n--- A. 噪声先验分析 ---")
noise = X_noisy - X_clean

global_mean = np.mean(noise)
global_std = np.std(noise)
print(f"-> 全局高斯噪声均值 (Mean): {global_mean:.6f}")
print(f"-> 全局高斯噪声标准差 (Std): {global_std:.6f}")

# 重新计算单张图片的噪声标准差
image_stds = np.std(noise, axis=1)
print(f"-> 单图噪声标准差 最小值: {np.min(image_stds):.6f}")
print(f"-> 单图噪声标准差 最大值: {np.max(image_stds):.6f}")

# ==========================================
# 核心任务 B：精准提取空间稀疏性先验
# ==========================================
print("\n--- B. 空间稀疏性先验提取 ---")
mean_clean_image = np.mean(X_clean, axis=0)
var_clean_image = np.var(X_clean, axis=0)

background_pixels = np.sum((mean_clean_image < 0.05) & (var_clean_image < 0.01))
print(f"-> 784 个像素中，有 {background_pixels} 个像素属于‘纯黑死角’")

np.save('data/spatial_prior_mean.npy', mean_clean_image)
print("-> 已将全局平均图像保存为 'data/spatial_prior_mean.npy'")

# ==========================================
# 可视化 (已修复 LaTeX Bug)
# ==========================================
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(mean_clean_image.reshape(28, 28), cmap='gray')
plt.title("Spatial Prior: Mean Image")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(var_clean_image.reshape(28, 28), cmap='hot')
plt.title("Spatial Variance Map")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.hist(noise.flatten()[:100000], bins=100, color='blue', alpha=0.7, density=True)
# 注意这里加了 'r'，变成了原生字符串，防止 \a 被转义
plt.title(rf"Noise Dist ($\sigma \approx {global_std:.2f}$)")
plt.xlabel("Noise Value")
plt.ylabel("Density")

plt.subplot(1, 4, 4)
sample_idx = 0
combined = np.concatenate((X_clean[sample_idx].reshape(28,28), X_noisy[sample_idx].reshape(28,28)), axis=1)
plt.imshow(combined, cmap='gray')
plt.title(f"Clean vs Noisy (Label: {labels[sample_idx]})")
plt.axis('off')

plt.tight_layout()
plt.savefig('prior_analysis.png', dpi=300)
print("\n-> 分析图表已保存为 'prior_analysis.png'！")