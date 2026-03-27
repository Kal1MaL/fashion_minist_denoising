import os
import time
import hydra
import torch
import torch.nn.functional as F
import pandas as pd
from omegaconf import DictConfig
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl

from src.lightning_module import LitPixelMeanFlow
from src.fashion_datamodule import FashionMNISTDataModule

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting multi-model evaluation on device: {device}")

    # 1. Model configurations
    checkpoints = {
        "ResUNet w/o prior,data_aug": "checkpoints/best-pure_resunet-epoch=152-val_mse=0.00363.ckpt",
        "ResUNet w/ prior,data_aug": "checkpoints/best-resunet-epoch=153-val_mse=0.00357.ckpt",
        "ViT (Ours)": "checkpoints/best-vitdecouple-epoch=158-val_mse=0.00325.ckpt"
    }

    models = {}
    metrics = {name: {"total_mse": 0.0, "total_infer_time": 0.0, "total_images": 0} for name in checkpoints.keys()}

    # 2. Load models
    for model_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found for {model_name}: {ckpt_path}")

        print(f"Loading {model_name}...")
        model = LitPixelMeanFlow.load_from_checkpoint(ckpt_path)
        model.to(device)
        model.eval()
        models[model_name] = model

    # 3. Data setup
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    vis_data = {}
    num_vis_samples = 10

    # 4. Evaluation Loop
    print("Running inference over test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x_clean = batch.get('x_clean', None)
            y_noisy = batch['y'].to(device)
            cond_dict = {
                'y_blur': batch['y_blur'].to(device),
                'y_flip': batch['y_flip'].to(device),
                'sigma': batch['sigma'].to(device),
                'label': batch['label'].to(device)
            }
            B = y_noisy.shape[0]

            if batch_idx == 3 and x_clean is not None:
                vis_data['Input'] = y_noisy[:num_vis_samples].cpu()
                vis_data['GT'] = x_clean[:num_vis_samples].cpu()

            for model_name, model in models.items():
                if device.type == 'cuda': torch.cuda.synchronize()
                start_time = time.time()

                x_pred = model(y_noisy, cond_dict)

                if device.type == 'cuda': torch.cuda.synchronize()
                end_time = time.time()

                metrics[model_name]["total_infer_time"] += (end_time - start_time)
                metrics[model_name]["total_images"] += B

                if x_clean is not None:
                    mse = F.mse_loss(x_pred, x_clean.to(device), reduction='sum')
                    metrics[model_name]["total_mse"] += mse.item()

                if batch_idx == 3 and x_clean is not None:
                    vis_data[model_name] = x_pred[:num_vis_samples].cpu()

    # 5. Output metrics
    print("\n" + "=" * 50)
    print("Multi-Model Evaluation Report")
    print("=" * 50)

    csv_data = {"Model Name": [], "Params (M)": [], "Test MSE": [], "Latency (ms/img)": [], "Throughput (FPS)": []}

    for model_name, model in models.items():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        avg_mse = metrics[model_name]["total_mse"] / (metrics[model_name]["total_images"] * 28 * 28)
        avg_latency_ms = (metrics[model_name]["total_infer_time"] / metrics[model_name]["total_images"]) * 1000
        throughput_fps = metrics[model_name]["total_images"] / metrics[model_name]["total_infer_time"]

        print(f"[{model_name}]")
        print(f"  MSE:        {avg_mse:.5f}")
        print(f"  Params:     {trainable_params / 1e6:.2f} M")
        print(f"  Latency:    {avg_latency_ms:.2f} ms")
        print("-" * 50)

        csv_data["Model Name"].append(model_name)
        csv_data["Params (M)"].append(round(trainable_params / 1e6, 2))
        csv_data["Test MSE"].append(round(avg_mse, 5))
        csv_data["Latency (ms/img)"].append(round(avg_latency_ms, 2))
        csv_data["Throughput (FPS)"].append(round(throughput_fps, 2))

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(csv_data).to_csv("results/multi_model_comparison.csv", index=False)
    print("Saved evaluation CSV to results/multi_model_comparison.csv")

    # 6. Save visualization grid
    if 'Input' in vis_data:
        print("Generating comparison visualization...")

        diff_map = torch.abs(vis_data['ResUNet w/o prior,data_aug'] - vis_data['ViT (Ours)'])
        diff_map = torch.clamp(diff_map * 3.0, 0, 1)

        grid_input = torch.cat([
            vis_data['Input'],
            vis_data['ResUNet w/o prior,data_aug'],
            vis_data['ResUNet w/ prior,data_aug'],
            vis_data['ViT (Ours)'],
            diff_map,
            vis_data['GT']
        ], dim=0)

        vis_grid = make_grid(grid_input, nrow=num_vis_samples, padding=2, normalize=False)
        save_image(vis_grid, "results/ultimate_comparison.png")
        print("Visualization saved to results/ultimate_comparison.png")

if __name__ == "__main__":
    main()