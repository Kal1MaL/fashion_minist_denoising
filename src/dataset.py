import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FashionMNISTDenoisingDataset(Dataset):
    def __init__(self, noisy_csv, clean_csv=None, mode='train'):
        self.mode = mode

        # 1. 加载 Noisy Data
        # 假设 csv 格式: [label, pixel1, pixel2, ...]
        df_noisy = pd.read_csv(noisy_csv)

        # 提取标签 (第 0 列)
        self.labels = df_noisy.iloc[:, 0].values

        # 提取像素 (第 1 列之后)
        self.noisy_data = df_noisy.iloc[:, 1:].values.astype(np.float32)

        # 2. 加载 Clean Data (如果是训练/验证模式)
        self.clean_data = None
        if clean_csv:
            df_clean = pd.read_csv(clean_csv)
            self.clean_data = df_clean.iloc[:, 1:].values.astype(np.float32)

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        # 1. 获取 Noisy Image
        # Reshape (784,) -> (1, 28, 28)

        # 【🔥 关键修正】
        # 既然 CSV 里已经是小数了，绝对不要再除以 255.0！
        # 直接转 Tensor 即可
        noisy_img = self.noisy_data[idx].reshape(1, 28, 28)
        noisy_img = torch.tensor(noisy_img, dtype=torch.float32)

        # 2. 获取 Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # 3. 获取 Clean Image (如果是 Train/Val)
        if self.clean_data is not None:
            clean_img = self.clean_data[idx].reshape(1, 28, 28)
            # 【🔥 关键修正】这里也不要除以 255.0
            clean_img = torch.tensor(clean_img, dtype=torch.float32)

            # 返回三个值：(noisy, clean, label)
            return noisy_img, clean_img, label

        else:
            # Test 模式
            return noisy_img