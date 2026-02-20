import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF


class FashionMNISTPriorDataset(Dataset):
    """
    底层 Dataset，纯靠内存中的 Tensor/Numpy 数组驱动，速度极快。
    负责在 __getitem__ 中实时组装所有多模态先验。
    """

    def __init__(self, y_noisy, labels, x_clean=None, spatial_prior=None):
        self.y_noisy = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1, 28, 28)
        self.labels = torch.tensor(labels, dtype=torch.long)

        self.has_clean = x_clean is not None
        if self.has_clean:
            self.x_clean = torch.tensor(x_clean, dtype=torch.float32).view(-1, 1, 28, 28)

        self.spatial_prior = spatial_prior if spatial_prior is not None else torch.zeros((1, 28, 28))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y_tensor = self.y_noisy[idx]  # [1, 28, 28]
        label = self.labels[idx]

        # 1. 频域先验: 低频模糊图 (y_blur)
        # 用 PyTorch 原生操作，比 PIL 快得多。kernel_size=5, sigma=1.5
        y_blur_tensor = TF.gaussian_blur(y_tensor, kernel_size=[5, 5], sigma=[1.5, 1.5])

        # 2. 动态噪声强度先验 (sigma)
        sigma_est = torch.std(y_tensor - y_blur_tensor)

        # 3. 对称性先验: 镜像翻转图 (y_flip)
        y_flip_tensor = TF.hflip(y_tensor)

        # 组装返回字典
        data_dict = {
            'y': y_tensor,
            'y_blur': y_blur_tensor,
            'y_flip': y_flip_tensor,
            'spatial_prior': self.spatial_prior,  # [1, 28, 28] 全局背景
            'label': label,
            'sigma': sigma_est
        }

        if self.has_clean:
            data_dict['x_clean'] = self.x_clean[idx]

        return data_dict


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data",
            batch_size: int = 128,
            num_workers: int = 4,
            val_split: float = 0.1,
            seed: int = 42
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir

    def setup(self, stage=None):
        # 加载空间稀疏性先验 (我们在上一轮脚本中生成的)
        prior_path = os.path.join(self.data_dir, 'spatial_prior_mean.npy')
        if os.path.exists(prior_path):
            spatial_prior = torch.tensor(np.load(prior_path), dtype=torch.float32).view(1, 28, 28)
        else:
            spatial_prior = torch.zeros((1, 28, 28))
            print("Warning: spatial_prior_mean.npy not found. Using zero prior.")

        # --- Fit 阶段: 准备训练和验证集 ---
        if stage == 'fit' or stage is None:
            # 读取含有 60,000 个样本的训练集 [cite: 12, 13]
            df_noisy = pd.read_csv(os.path.join(self.data_dir, 'fashion-mnist_noisy_train.csv'), header=0)
            df_clean = pd.read_csv(os.path.join(self.data_dir, 'fashion-mnist_clean_train.csv'), header=0)

            labels = df_noisy.iloc[:, 0].values
            y_noisy = df_noisy.iloc[:, 1:].values
            x_clean = df_clean.iloc[:, 1:].values

            # 实例化全量 Dataset
            full_dataset = FashionMNISTPriorDataset(y_noisy, labels, x_clean, spatial_prior)

            # 划分 Train 和 Val
            total_size = len(full_dataset)
            val_size = int(total_size * self.hparams.val_split)
            train_size = total_size - val_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.hparams.seed)
            )

        # --- Test/Predict 阶段: 准备测试集 ---
        if stage == 'test' or stage == 'predict':
            # 读取含有 10,000 个样本的测试集 [cite: 14]
            df_test_noisy = pd.read_csv(os.path.join(self.data_dir, 'fashion-mnist_noisy_test.csv'), header=0)
            labels_test = df_test_noisy.iloc[:, 0].values
            y_test_noisy = df_test_noisy.iloc[:, 1:].values

            # 尝试读取 clean_test 用于本地评估 MSE (如果是比赛提交环节可能没有这个文件)
            clean_test_path = os.path.join(self.data_dir, 'fashion-mnist_clean_test.csv')
            x_test_clean = None
            if os.path.exists(clean_test_path):
                df_test_clean = pd.read_csv(clean_test_path, header=0)
                x_test_clean = df_test_clean.iloc[:, 1:].values

            self.test_dataset = FashionMNISTPriorDataset(y_test_noisy, labels_test, x_test_clean, spatial_prior)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,  # 加速数据向 GPU 传输
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )