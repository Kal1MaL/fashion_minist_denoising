import os
import torch
import hydra
from omegaconf import DictConfig
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl

from src.lightning_module import LitPixelMeanFlow

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting adversarial condition injection test...")

    # 1. Load checkpoint
    ckpt_path = cfg.get("ckpt_path", r"checkpoints/best-pmf-epoch=159-val_mse=0.00325.ckpt")
    model = LitPixelMeanFlow.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    if model.hparams.get('num_cls_tokens', 4) == 0:
        raise ValueError("Error: Model does not support conditional injection (num_cls_tokens is 0).")

    # 2. Get test data batch
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    batch = next(iter(test_loader))
    num_samples = 8

    x_clean = batch['x_clean'][:num_samples].to(device)
    y_noisy = batch['y'][:num_samples].to(device)
    true_labels = batch['label'][:num_samples].to(device)

    cond_dict_base = {
        'y_blur': batch['y_blur'][:num_samples].to(device),
        'y_flip': batch['y_flip'][:num_samples].to(device),
        'sigma': batch['sigma'][:num_samples].to(device)
    }

    # 3. Normal inference
    cond_dict_normal = cond_dict_base.copy()
    cond_dict_normal['label'] = true_labels

    with torch.no_grad():
        x_pred_normal = model(y_noisy, cond_dict_normal)

    # 4. Adversarial inference (injecting target class 7)
    adv_labels = torch.full_like(true_labels, fill_value=7).to(device)

    cond_dict_adv = cond_dict_base.copy()
    cond_dict_adv['label'] = adv_labels

    with torch.no_grad():
        x_pred_adv = model(y_noisy, cond_dict_adv)

    # 5. Save grid visualization
    os.makedirs("results", exist_ok=True)

    print(f"True Labels: {true_labels.cpu().tolist()}")
    print(f"Adv Labels: {adv_labels.cpu().tolist()}")

    # Output grid layout: Noisy | Ground Truth | Normal Denoise | Adversarial Denoise
    grid_input = torch.cat([y_noisy, x_clean, x_pred_normal, x_pred_adv], dim=0)

    vis_grid = make_grid(grid_input, nrow=num_samples, padding=2, normalize=True, scale_each=True)
    save_path = "results/adversarial_injection_demo.png"
    save_image(vis_grid, save_path)
    print(f"Saved adversarial results to {save_path}")

if __name__ == "__main__":
    main()