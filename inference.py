import os
import time
import hydra
import torch
import torch.nn.functional as F
import pandas as pd
from omegaconf import DictConfig
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl

# 导入你的 LightningModule 和 DataModule
from src.lightning_module import LitPixelMeanFlow
from src.fashion_datamodule import FashionMNISTDataModule


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动超级多模型横评脚本，使用设备: {device}")

    # ==========================================
    # 1. 配置你的三个模型 Checkpoint 路径
    # ==========================================
    # ⚠️ 请在这里替换为你实际的 checkpoint 路径！
    checkpoints = {
        "ResUNet w/o prior,data_aug": "checkpoints/best-pure_resunet-epoch=152-val_mse=0.00363.ckpt",  # 替换为真实的 UNet 权重
        "ResUNet w/ prior,data_aug": "checkpoints/best-resunet-epoch=153-val_mse=0.00357.ckpt",  # 替换为真实的 ResUNet 权重
        "ViT (Ours)": "checkpoints/best-vitdecouple-epoch=158-val_mse=0.00325.ckpt"  # 你的终极模型
    }

    models = {}
    metrics = {name: {"total_mse": 0.0, "total_infer_time": 0.0, "total_images": 0} for name in checkpoints.keys()}

    # ==========================================
    # 2. 依次加载所有模型到显存
    # ==========================================
    for model_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"❌ 找不到 {model_name} 的权重文件: {ckpt_path}")

        print(f"📦 正在加载 {model_name}: {ckpt_path}")
        model = LitPixelMeanFlow.load_from_checkpoint(ckpt_path)
        model.to(device)
        model.eval()
        models[model_name] = model

    # ==========================================
    # 3. 实例化数据
    # ==========================================
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    # 用于画图的样本缓存
    vis_data = {}
    num_vis_samples = 10  # 选 10 张图画并排对比图

    # ==========================================
    # 4. 核心评测循环：跑遍测试集
    # ==========================================
    print("⏳ 开始在测试集上进行多模型同步推导...")
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

            # 遍历三个模型进行推理
            for model_name, model in models.items():
                if device.type == 'cuda': torch.cuda.synchronize()
                start_time = time.time()

                x_pred = model(y_noisy, cond_dict)

                if device.type == 'cuda': torch.cuda.synchronize()
                end_time = time.time()

                # 记录指标
                metrics[model_name]["total_infer_time"] += (end_time - start_time)
                metrics[model_name]["total_images"] += B

                if x_clean is not None:
                    mse = F.mse_loss(x_pred, x_clean.to(device), reduction='sum')
                    metrics[model_name]["total_mse"] += mse.item()

                # 提取第 10 个 Batch 用于可视化
                if batch_idx == 3 and x_clean is not None:
                    vis_data[model_name] = x_pred[:num_vis_samples].cpu()

    # ==========================================
    # 5. 打印性能指标 & 保存 CSV
    # ==========================================
    print("\n" + "=" * 50)
    print("🏆 超级多模型横评结果报告")
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
    print("💾 性能表格已保存到 results/multi_model_comparison.csv")

    # ==========================================
    # 6. 🎨 生成极其震撼的可视化六宫格网格图
    # ==========================================
    if 'Input' in vis_data:
        print("🎨 正在生成绝杀对比图 (results/ultimate_comparison.png)...")

        # 🌟 绝杀图层：ResUNet 和 ViT 的差异放大图
        # 我们用绝对值差异，并乘以 3 放大差异，方便肉眼观察到底差在哪里
        diff_map = torch.abs(vis_data['ResUNet w/o prior,data_aug'] - vis_data['ViT (Ours)'])
        diff_map = torch.clamp(diff_map * 3.0, 0, 1)  # 放大 3 倍并截断到 0-1

        # 按照你构思的极品顺序拼接：
        grid_input = torch.cat([
            vis_data['Input'],  # 第一行: 含噪图
            vis_data['ResUNet w/o prior,data_aug'],  # 第二行: UNet 预测
            vis_data['ResUNet w/ prior,data_aug'],  # 第三行: ResUNet 预测
            vis_data['ViT (Ours)'],  # 第四行: ViT 预测
            diff_map,  # 第五行: ResUNet 和 ViT 的差异 (放大3倍)
            vis_data['GT']  # 第六行: 干净原图
        ], dim=0)

        # 生成并保存
        vis_grid = make_grid(grid_input, nrow=num_vis_samples, padding=2, normalize=False)
        save_image(vis_grid, "results/ultimate_comparison.png")
        print("✅ 终极对比图保存成功！可以直接贴到报告里了！")


if __name__ == "__main__":
    main()