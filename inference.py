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
    print(f"🚀 启动评估脚本，使用设备: {device}")

    # ==========================================
    # 1. 检查 Checkpoint 路径
    # ==========================================

    ckpt_path = r"checkpoints/best-pmf-epoch=156-val_mse=0.00475.ckpt"
    print(f" 正在加载模型权重: {ckpt_path}")

    # ==========================================
    # 2. 实例化数据和模型
    # ==========================================
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    model = LitPixelMeanFlow.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()  # 开启评估模式

    # ==========================================
    # 3. 统计参数量 (Parameters)
    # ==========================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型总参数量: {total_params / 1e6:.2f} M")

    # ==========================================
    # 4. 核心评测循环：计算 MSE 与 推理延迟
    # ==========================================
    total_mse = 0.0
    total_images = 0
    total_infer_time = 0.0

    # 用于画图的样本缓存
    vis_y_noisy, vis_x_pred, vis_x_clean = None, None, None
    num_vis_samples = 10  # 我们选 10 张图画并排对比图

    print("⏳ 开始在测试集上进行 1-NFE 极速推导...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 将数据推到 GPU
            x_clean = batch.get('x_clean', None)
            y_noisy = batch['y'].to(device)
            cond_dict = {
                'y_blur': batch['y_blur'].to(device),
                'y_flip': batch['y_flip'].to(device),
                'sigma': batch['sigma'].to(device),
                'label': batch['label'].to(device)
            }

            B = y_noisy.shape[0]
            t_val = torch.ones(B, device=device)  # 1-NFE 一步到位

            # --- 测速开始 (为了准确测量 GPU 时间，使用 cuda.synchronize) ---
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()

            # 极速前向传播
            x_pred = model(y_noisy, t_val, cond_dict)

            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.time()
            # --- 测速结束 ---

            total_infer_time += (end_time - start_time)
            total_images += B

            # 累计 MSE
            if x_clean is not None:
                x_clean = x_clean.to(device)
                mse = F.mse_loss(x_pred, x_clean, reduction='sum')
                total_mse += mse.item()

            # 保存第一批的 10 张图用于后续可视化
            if batch_idx == 3 and x_clean is not None:
                vis_y_noisy = y_noisy[:num_vis_samples].cpu()
                vis_x_pred = x_pred[:num_vis_samples].cpu()
                vis_x_clean = x_clean[:num_vis_samples].cpu()

    # 计算最终指标
    avg_mse = total_mse / (total_images * 28 * 28) if x_clean is not None else -1
    avg_latency_ms = (total_infer_time / total_images) * 1000  # 毫秒/张
    throughput_fps = total_images / total_infer_time  # 帧/秒 (FPS)

    print("\n" + "=" * 40)
    print("🏆 评测结果报告")
    print("=" * 40)
    print(f"Test MSE:           {avg_mse:.5f}")
    print(f"Parameters:         {trainable_params / 1e6:.2f} M")
    print(f"Latency per image:  {avg_latency_ms:.2f} ms")
    print(f"Throughput (FPS):   {throughput_fps:.2f} img/s")
    print("=" * 40)

    # ==========================================
    # 5. 生成极其震撼的可视化网格图
    # ==========================================
    os.makedirs("results", exist_ok=True)
    if vis_y_noisy is not None:
        print("🎨 正在生成残差对比图 (results/denoising_comparison.png)...")
        # 把三组图拼成一个长条：
        # 第一行: 含噪图 (Input)
        # 第二行: 模型预测图 (pMF-DiT Output)
        # 第三行: 干净真实图 (Ground Truth)
        # 将它们全部拉到 0-1 范围用于显示（如果是标准化的记得反归一化，这里假设已经是 0-1）
        grid_input = torch.cat([vis_y_noisy, vis_x_pred, vis_x_clean], dim=0)

        # 使用 make_grid，nrow 设置为我们要展示的样本数
        vis_grid = make_grid(grid_input, nrow=num_vis_samples, padding=2, normalize=True, scale_each=True)
        save_image(vis_grid, "results/denoising_comparison.png")
        print("✅ 对比图保存成功！")

    # ==========================================
    # 6. 保存性能指标到 CSV (为报告表格准备)
    # ==========================================
    csv_path = "results/model_performance_metrics.csv"
    print(f"💾 正在将性能指标保存到 {csv_path}...")

    metrics_data = {
        "Model Name": ["pMF-DiT (1-NFE)"],
        "Params (M)": [round(trainable_params / 1e6, 2)],
        "Test MSE": [round(avg_mse, 5)],
        "Latency (ms/img)": [round(avg_latency_ms, 2)],
        "Throughput (FPS)": [round(throughput_fps, 2)]
    }

    df_metrics = pd.DataFrame(metrics_data)

    # 如果文件已存在，我们可以追加写入，方便你对比其他模型 (比如 ResUNet)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_metrics = pd.concat([df_existing, df_metrics], ignore_index=True)

    df_metrics.to_csv(csv_path, index=False)
    print("✅ CSV 性能表格保存成功！报告里的表格有素材了！")


if __name__ == "__main__":
    main()