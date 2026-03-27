import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

clean_path = 'data/fashion-mnist_clean_train.csv'
noisy_path = 'data/fashion-mnist_noisy_train.csv'

print("Loading datasets...")
df_clean = pd.read_csv(clean_path, header=0)
df_noisy = pd.read_csv(noisy_path, header=0)

labels = df_clean.iloc[:, 0].values.astype(np.int64)

# Features: 784 dims
X_clean = df_clean.iloc[:, 1:].values.astype(np.float32)
X_noisy = df_noisy.iloc[:, 1:].values.astype(np.float32)

print("Data loaded successfully!")
print(f"Clean features shape: {X_clean.shape}")
print(f"Labels shape: {labels.shape}")

# A. Noise prior analysis
print("\n--- A. Noise Analysis ---")
noise = X_noisy - X_clean

global_mean = np.mean(noise)
global_std = np.std(noise)
print(f"Global Noise Mean: {global_mean:.6f}")
print(f"Global Noise Std: {global_std:.6f}")

image_stds = np.std(noise, axis=1)
print(f"Min image noise std: {np.min(image_stds):.6f}")
print(f"Max image noise std: {np.max(image_stds):.6f}")

# B. Extract spatial sparsity prior
print("\n--- B. Spatial Sparsity Prior ---")
mean_clean_image = np.mean(X_clean, axis=0)
var_clean_image = np.var(X_clean, axis=0)

background_pixels = np.sum((mean_clean_image < 0.05) & (var_clean_image < 0.01))
print(f"Background pixels count: {background_pixels} / 784")

np.save('data/spatial_prior_mean.npy', mean_clean_image)
print("Saved mean image to 'data/spatial_prior_mean.npy'")

# Plotting
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
print("Saved analysis plots to 'prior_analysis.png'")