import os
import torch
import hydra
from omegaconf import DictConfig
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl

# 导入你的 LightningModule
from src.lightning_module import LitPixelMeanFlow


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动对抗性条件注入实验...")

    # ==========================================
    # 1. 加载你训练得最好的带有 Prior 和 Aug 的 ViT 模型
    # ==========================================
    ckpt_path = cfg.get("ckpt_path", r"checkpoints/best-pmf-epoch=159-val_mse=0.00325.ckpt")
    model = LitPixelMeanFlow.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    # 检查模型是否使用了条件 Token
    if model.hparams.get('num_cls_tokens', 4) == 0:
        raise ValueError("❌ 当前加载的模型 num_cls_tokens 为 0，无法进行条件注入实验！请加载带条件标签的模型。")

    # ==========================================
    # 2. 获取一批测试数据
    # ==========================================
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    # 我们只取第一个 Batch 的前 8 张图片来做精细可视化
    batch = next(iter(test_loader))
    num_samples = 8

    x_clean = batch['x_clean'][:num_samples].to(device)
    y_noisy = batch['y'][:num_samples].to(device)
    true_labels = batch['label'][:num_samples].to(device)

    # 准备共享的物理先验
    cond_dict_base = {
        'y_blur': batch['y_blur'][:num_samples].to(device),
        'y_flip': batch['y_flip'][:num_samples].to(device),
        'sigma': batch['sigma'][:num_samples].to(device)
    }

    # ==========================================
    # 3. 正常推理 (正确的标签)
    # ==========================================
    cond_dict_normal = cond_dict_base.copy()
    cond_dict_normal['label'] = true_labels

    with torch.no_grad():
        x_pred_normal = model(y_noisy, cond_dict_normal)

    # ==========================================
    # 4. 对抗性推理 (注入错误的恶意标签)
    # ==========================================
    # 策略：我们将所有的标签强行改为 7 (Sneaker 运动鞋)
    # 看看当输入是一件 T恤 或 裤子 时，网络会不会强行画出鞋底或鞋带
    adv_labels = torch.full_like(true_labels, fill_value=7).to(device)

    cond_dict_adv = cond_dict_base.copy()
    cond_dict_adv['label'] = adv_labels

    with torch.no_grad():
        x_pred_adv = model(y_noisy, cond_dict_adv)

    # ==========================================
    # 5. 拼接并保存极致的对比图
    # ==========================================
    os.makedirs("results", exist_ok=True)

    print("真实标签:", true_labels.cpu().tolist())
    print("恶意注入标签:", adv_labels.cpu().tolist())

    # 按照以下顺序从上到下拼接：
    # 第一行: 含噪输入 (Noisy Input)
    # 第二行: 干净原图 (Ground Truth)
    # 第三行: 正常去噪结果 (Normal Denoised)
    # 第四行: 对抗注入结果 (Adversarially Denoised - "被迫变成鞋子")
    grid_input = torch.cat([y_noisy, x_clean, x_pred_normal, x_pred_adv], dim=0)

    # nrow=num_samples 保证每一行有 8 张图
    vis_grid = make_grid(grid_input, nrow=num_samples, padding=2, normalize=True, scale_each=True)
    save_path = "results/adversarial_injection_demo.png"
    save_image(vis_grid, save_path)

    print(f"✅ 实验完成！快去查看生成的绝赞对比图: {save_path}")


if __name__ == "__main__":
    main()