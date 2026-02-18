import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FashionMNISTDenoisingDataset(Dataset):
    # 新增 synthesize_noise 参数
    def __init__(self, noisy_csv=None, clean_csv=None, mode='train', synthesize_noise=False):
        self.mode = mode
        self.synthesize_noise = synthesize_noise

        # 逻辑分支 A: 传统模式 (读取现成的 noisy csv)
        if not synthesize_noise:
            if noisy_csv is None:
                raise ValueError("In standard mode, noisy_csv is required!")
            df_noisy = pd.read_csv(noisy_csv)
            self.labels = df_noisy.iloc[:, 0].values
            self.noisy_data = df_noisy.iloc[:, 1:].values.astype(np.float32)

            self.clean_data = None
            if clean_csv:
                df_clean = pd.read_csv(clean_csv)
                self.clean_data = df_clean.iloc[:, 1:].values.astype(np.float32)

        # 逻辑分支 B: 合成模式 (只读 clean csv，自动加噪)
        else:
            if clean_csv is None:
                raise ValueError("In synthesize mode, clean_csv is required!")
            df_clean = pd.read_csv(clean_csv)
            # 假设官方 csv 第一列是 label
            self.labels = df_clean.iloc[:, 0].values
            # 剩下的全是 clean pixel
            self.clean_data = df_clean.iloc[:, 1:].values.astype(np.float32)
            # 不需要 self.noisy_data，因为我们会在 getitem 里实时生成

    def __len__(self):
        # 长度取决于 clean_data (合成模式) 或 noisy_data (传统模式)
        if self.synthesize_noise:
            return len(self.clean_data)
        return len(self.noisy_data)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # --- 模式 A: 读取硬盘上的固定噪声图 ---
        if not self.synthesize_noise:
            noisy_img = self.noisy_data[idx].reshape(1, 28, 28)
            noisy_img = torch.tensor(noisy_img, dtype=torch.float32)

            if self.clean_data is not None:
                clean_img = self.clean_data[idx].reshape(1, 28, 28)
                clean_img = torch.tensor(clean_img, dtype=torch.float32)
                return noisy_img, clean_img, label
            else:
                return noisy_img

        # --- 模式 B: 在线生成噪声 (Dynamic Noise) ---
        else:
            # 1. 拿 Clean 图
            clean_img = self.clean_data[idx].reshape(1, 28, 28)
            clean_img = torch.tensor(clean_img, dtype=torch.float32)

            # 3. 🔥 注入高斯噪声 (模拟真实场景)
            # 这里的 0.154 是你之前算出的数据集平均 sigma，或者你可以设别的强度
            noise = torch.randn_like(clean_img) * 0.154
            noisy_img = clean_img + noise

            # 4. 截断到有效范围 (0, 1) 或者保持原样，取决于你的预处理逻辑
            # noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

            return noisy_img, clean_img, label