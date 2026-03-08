import os
import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import pytorch_lightning as pl

# 💡 核心修改：同时导入两个不同时代的模型类
from src.lightning_module import LitPixelMeanFlow
from src.x_predict import XPredictDistiller  # 假设你的 x_predict.py 放在 src 目录下


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def evaluate_delta_mse(cfg: DictConfig):
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 ΔMSE 评估，使用设备: {device}")

    # 1. 实例化数据
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    vit_path = "checkpoints/best-pmf-epoch=159-val_mse=0.00325.ckpt"
    flow_path = "checkpoints_distilled/x-pred-best-epoch=83-val_1step_mse=0.0037.ckpt"

    print("⏳ 正在加载异构模型: ViT (LitPixelMeanFlow) 与 Flow (XPredictDistiller)...")

    # 按照各自的架构正确加载权重
    model_vit = LitPixelMeanFlow.load_from_checkpoint(vit_path,weights_only=False).to(device).eval()

    teacher_path = "checkpoints/best-model-epoch=96-val_mse_image=0.0042.ckpt"
    model_flow = XPredictDistiller.load_from_checkpoint(
        flow_path,
        teacher_ckpt_path=teacher_path,  # 强行覆盖掉原来 checkpoint 里存的旧路径！
        weights_only=False  # 加上 strict=False 防止一些无关紧要的 buffer key 不匹配
    ).to(device).eval()

    total_mse_vit = 0.0
    total_mse_flow = 0.0
    total_delta_mse = 0.0
    total_images = 0

    print("🏃 开始双轨推理计算...")
    with torch.no_grad():
        for batch in test_loader:
            x_clean = batch.get('x_clean', None)
            if x_clean is None:
                continue

            x_clean = x_clean.to(device)
            # 在新的 DataLoader 里，含噪图像的 key 叫 'y'
            y_noisy = batch['y'].to(device)

            # 新版 ViT 需要的复杂条件字典
            cond_dict = {
                'y_blur': batch['y_blur'].to(device),
                'y_flip': batch['y_flip'].to(device),
                'sigma': batch['sigma'].to(device),
                'label': batch['label'].to(device)
            }

            # ==========================================
            # 💡 核心：调用各自不同的推理接口
            # ==========================================
            # 1. 新 ViT 极速前向
            pred_vit = model_vit(y_noisy, cond_dict)

            # 2. 旧 Flow 1-Step 采样 (内部会自动生成纯噪声并预测)
            # 在旧架构中，condition 通常就是含噪/模糊图像本身
            pred_flow = model_flow.sample(condition=y_noisy)

            # 计算各项 MSE
            total_mse_vit += F.mse_loss(pred_vit, x_clean, reduction='sum').item()
            total_mse_flow += F.mse_loss(pred_flow, x_clean, reduction='sum').item()

            # 🔥 计算两者之间的像素级分歧
            total_delta_mse += F.mse_loss(pred_vit, pred_flow, reduction='sum').item()

            total_images += y_noisy.shape[0]

    num_pixels = total_images * 28 * 28
    print("\n" + "=" * 40)
    print("🎯 MSE 测算结果 (Pixel-wise)")
    print("=" * 40)
    print(f"ViT 自身 MSE:   {total_mse_vit / num_pixels:.6f}")
    print(f"Flow 自身 MSE:  {total_mse_flow / num_pixels:.6f}")
    print(f"ΔMSE (分歧度):  {total_delta_mse / num_pixels:.6f}")
    print("=" * 40)


if __name__ == "__main__":
    evaluate_delta_mse()