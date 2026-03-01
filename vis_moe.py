import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from src.lit_moe import LitMoEFusion
import pytorch_lightning as pl


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def visualize_moe(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载你的冠军模型
    moe_ckpt = "checkpoints/moe_routers/moe-resnet-best-epoch=11-val_moe_mse=0.003214.ckpt"
    model = LitMoEFusion.load_from_checkpoint(moe_ckpt).to(device).eval()

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    # 拿一个 Batch
    batch = next(iter(test_loader))
    x_clean = batch['x_clean'].to(device)
    y_noisy = batch['y'].to(device)
    cond_dict = {k: v.to(device) for k, v in batch.items() if k in ['y_blur', 'y_flip', 'sigma', 'label']}

    with torch.no_grad():
        pred_final, alpha, pred_vit, pred_flow = model(y_noisy, cond_dict)

    # 挑选第一张图进行可视化
    idx = 10
    img_clean = x_clean[idx, 0].cpu().numpy()
    img_noisy = y_noisy[idx, 0].cpu().numpy()
    img_vit = pred_vit[idx, 0].cpu().numpy()
    img_flow = pred_flow[idx, 0].cpu().numpy()
    img_final = pred_final[idx, 0].cpu().numpy()
    alpha_map = alpha[idx, 0].cpu().numpy()  # 这就是那张神奇的权重图！

    # 画图
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    axes[0].imshow(img_noisy, cmap='gray');
    axes[0].set_title('Noisy Input')
    axes[1].imshow(img_vit, cmap='gray');
    axes[1].set_title('ViT (Conservative)')
    axes[2].imshow(img_flow, cmap='gray');
    axes[2].set_title('Flow (Generative)')

    # 划重点：画出 Alpha 热力图 (越接近红/黄，代表越偏向 ViT；越接近蓝/深，代表越偏向 Flow)
    im3 = axes[3].imshow(alpha_map, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title('Router Alpha Map')
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    axes[4].imshow(img_final, cmap='gray');
    axes[4].set_title('MoE Final')
    axes[5].imshow(img_clean, cmap='gray');
    axes[5].set_title('Ground Truth')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("moe_visualization.png", dpi=300)
    print("✅ 可视化已保存为 moe_visualization.png")


if __name__ == "__main__":
    visualize_moe()