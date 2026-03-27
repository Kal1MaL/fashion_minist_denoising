import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
import random

from src.dit_pmf import DiTPixelMeanFlow
from src.baseline_unet import SimpleUNet, ResUNet

class LitPixelMeanFlow(pl.LightningModule):
    def __init__(self, in_channels, img_size, patch_size, hidden_dim,
                 depth, num_heads, num_classes, num_cls_tokens,
                 learning_rate, drop_path_rate, backbone_type="vit",use_dynamic_aug=True,num_register_tokens=0):
        super().__init__()
        self.save_hyperparameters()

        if backbone_type == "vit":
            self.net = DiTPixelMeanFlow(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                depth=depth,
                num_heads=num_heads,
                num_classes=num_classes,
                num_cls_tokens=num_cls_tokens,
                drop_path_rate=drop_path_rate,
                num_register_tokens=num_register_tokens
            )
        elif backbone_type == "unet":
            self.net = SimpleUNet(in_channels=in_channels, out_channels=1)
        elif backbone_type == "resunet":
            self.net = ResUNet(in_channels=in_channels, out_channels=1)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

    def forward(self, y_noisy, cond_dict):
        """Forward pass for inference."""
        return self.net(y_noisy, cond_dict)

    def training_step(self, batch, batch_idx):
        x_clean = batch['x_clean']
        y_noisy = batch['y']

        y_blur = batch['y_blur'].clone()
        y_flip = batch['y_flip'].clone()

        # 1. Geometric augmentations
        if random.random() > 0.5:
            x_clean = TF.hflip(x_clean)
            y_noisy = TF.hflip(y_noisy)
            y_blur = TF.hflip(y_blur)
            y_flip = TF.hflip(y_flip)

        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        if shift_x != 0 or shift_y != 0:
            x_clean = TF.affine(x_clean, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
            y_noisy = TF.affine(y_noisy, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
            y_blur = TF.affine(y_blur, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
            y_flip = TF.affine(y_flip, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)

        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            x_clean = torch.rot90(x_clean, k, dims=[-2, -1])
            y_noisy = torch.rot90(y_noisy, k, dims=[-2, -1])
            y_blur = torch.rot90(y_blur, k, dims=[-2, -1])
            y_flip = torch.rot90(y_flip, k, dims=[-2, -1])

        # 2. Noise resampling augmentations
        if self.hparams.use_dynamic_aug:
            real_noise = y_noisy - x_clean
            aug_prob = random.random()

            if self.current_epoch >= int(self.trainer.max_epochs * 0.8):
                aug_prob = 1.0

            if aug_prob < 0.3:
                # Strategy A: Noise Swapping
                noise_shifted = torch.roll(real_noise, shifts=1, dims=0)
                y_noisy = x_clean + noise_shifted
                y_blur = TF.gaussian_blur(y_noisy, kernel_size=[5, 5], sigma=[1.5, 1.5])
                y_flip = TF.hflip(y_noisy)

            elif aug_prob < 0.6:
                # Strategy B: Synthetic Noise Injection
                sigma_real = real_noise.reshape(real_noise.shape[0], -1).std(dim=1).reshape(-1, 1, 1, 1)
                synthetic_noise = torch.randn_like(x_clean) * sigma_real
                y_noisy = x_clean + synthetic_noise
                y_blur = TF.gaussian_blur(y_noisy, kernel_size=[5, 5], sigma=[1.5, 1.5])
                y_flip = TF.hflip(y_noisy)

        # 3. Prepare condition dict
        cond_dict = {
            'y_blur': y_blur,
            'y_flip': y_flip,
            'sigma': batch['sigma'],
            'label': batch['label']
        }

        # End-to-end MSE regression
        x_pred = self.net(y_noisy, cond_dict)
        loss_mse = F.mse_loss(x_pred, x_clean)

        self.log('train_mse_loss', loss_mse, prog_bar=True)
        return loss_mse

    def validation_step(self, batch, batch_idx):
        x_clean = batch['x_clean']
        y_noisy = batch['y']

        cond_dict = {
            'y_blur': batch['y_blur'],
            'y_flip': batch['y_flip'],
            'sigma': batch['sigma'],
            'label': batch['label']
        }

        x_pred = self(y_noisy, cond_dict)
        val_mse = F.mse_loss(x_pred, x_clean)
        self.log('val_mse', val_mse, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )

        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=total_steps,
            pct_start=0.2,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }