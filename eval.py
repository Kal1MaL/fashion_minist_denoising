import torch
import torch.nn.functional as F
import pandas as pd
import time
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm

from src.dataset import FashionMNISTDenoisingDataset
from src.diffusion import ConditionalDDPM
from src.x_predict import XPredictDistiller
from src.unet import RegressionUNet


def count_parameters(model):
    """精确统计真正用于推理的网络参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_model(model_name, model_fn, dataloader, device, desc):
    """
    通用评估器: 测试 MSE, Throughput, VRAM
    """
    total_mse = 0.0
    total_images = 0

    # --- 预热 GPU ---
    print(f"\n[{model_name}] 正在进行 GPU 预热 (Warm-up)...")
    warmup_batch = next(iter(dataloader))
    cond_warmup, clean_warmup, _ = warmup_batch
    cond_warmup = cond_warmup.to(device)
    sigma_warmup = cond_warmup.std(dim=(1, 2, 3), keepdim=True)

    # 预热2次
    for _ in range(2):
        model_fn(cond_warmup, sigma_warmup)
    torch.cuda.synchronize()

    # --- 开始测速与算分 ---
    print(f"[{model_name}] 开始全量评估...")
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for batch in tqdm(dataloader, desc=f"Evaluating {desc}"):
        conditions, clean_imgs, _ = batch
        conditions = conditions.to(device)
        clean_imgs = clean_imgs.to(device)
        current_sigma = conditions.std(dim=(1, 2, 3), keepdim=True)

        b = conditions.shape[0]
        total_images += b

        # 推理
        pred_x0 = model_fn(conditions, current_sigma)

        # 算 MSE
        mse = F.mse_loss(pred_x0, clean_imgs, reduction='mean').item()
        total_mse += mse * b

    end_event.record()
    torch.cuda.synchronize()

    # --- 指标计算 ---
    avg_mse = total_mse / total_images

    elapsed_time_s = start_event.elapsed_time(end_event) / 1000.0
    throughput = total_images / elapsed_time_s
    time_per_image_ms = (elapsed_time_s * 1000.0) / total_images
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "Model": model_name,
        "Test Images": total_images,
        "MSE (↓)": f"{avg_mse:.5f}",
        "Throughput (Imgs/s) (↑)": f"{throughput:.2f}",
        "Time/Img (ms) (↓)": f"{time_per_image_ms:.2f}",
        "Peak VRAM (MB) (↓)": f"{peak_vram_mb:.2f}"
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 Test Set 全面评估平台 on {device}...")

    # ==========================================
    # 🎛️ 评测任务控制台 (在这里增删你需要对比的 Baseline)
    # ==========================================
    EVAL_CONFIG = {
        "run_student": True,  # 蒸馏后的 1-Step 模型
        "run_hybrid": False,  # 混合模型 (需要同时有 Teacher 和 Student 权重)
        "run_teacher": False,  # 原版 DDPM (1000 步)
        "run_unet": True  # 🚀 纯回归 Res-UNet
    }

    # ⚠️ 请确保这里的路径是你本地实际存在的 ckpt 路径！
    CKPT_PATHS = {
        "teacher": "checkpoints/best-model-epoch=96-val_mse_image=0.0042.ckpt",
        "student": "checkpoints_distilled/x-pred-best-epoch=08-val_1step_mse=0.0048.ckpt",
        "unet": "checkpoints/best-model-epoch=13-val_mse_image=0.0038.ckpt"
    }

    # ==========================================
    # 1. 加载真正的 Test Pair 数据集
    # ==========================================
    pl.seed_everything(42)

    TEST_CLEAN_CSV = "data/fashion-mnist_clean_test.csv"
    TEST_NOISY_CSV = "data/fashion-mnist_noisy_test.csv"

    print(f"加载测试集数据: {TEST_CLEAN_CSV} 和 {TEST_NOISY_CSV}")
    test_dataset = FashionMNISTDenoisingDataset(
        noisy_csv=TEST_NOISY_CSV,
        clean_csv=TEST_CLEAN_CSV,
        mode='test',
        synthesize_noise=False
    )

    full_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"✅ Test Set 总量: {len(test_dataset)} 张。")

    # ==========================================
    # 2. 根据开关动态加载权重与模型
    # ==========================================
    teacher_net = None
    student_net = None
    unet_net = None
    timesteps = 1000
    refine_steps = 100

    if EVAL_CONFIG["run_teacher"] or EVAL_CONFIG["run_hybrid"]:
        print("\n加载 Teacher DDPM...")
        teacher = ConditionalDDPM.load_from_checkpoint(CKPT_PATHS["teacher"]).to(device)
        teacher.eval()
        teacher_net = teacher.model
        timesteps = teacher.timesteps
        refine_steps = int(timesteps * 0.1)

    if EVAL_CONFIG["run_student"] or EVAL_CONFIG["run_hybrid"]:
        print("\n加载 Student X-Predictor...")
        student_distiller = XPredictDistiller.load_from_checkpoint(CKPT_PATHS["student"]).to(device)
        student_distiller.eval()
        student_net = student_distiller.student

    if EVAL_CONFIG["run_unet"]:
        print("\n加载 Regression UNet...")
        unet_model = RegressionUNet.load_from_checkpoint(CKPT_PATHS["unet"]).to(device)
        unet_model.eval()
        unet_net = unet_model

    # ==========================================
    # 3. 注册激活的评估任务清单
    # ==========================================
    eval_tasks = []

    if EVAL_CONFIG["run_student"]:
        def student_fn(cond, sigma):
            b, c, h, w = cond.shape
            pure_noise = torch.randn((b, c, h, w), device=device)
            t_max = torch.full((b,), timesteps - 1, device=device, dtype=torch.long)
            pred, _ = student_net(pure_noise, t_max.float() / timesteps, cond, sigma)
            return torch.clamp(pred, 0.0, 1.0)

        eval_tasks.append({
            "name": "Student (1-Step)", "fn": student_fn, "desc": "Student",
            "params": count_parameters(student_net), "nfe": 1
        })

    if EVAL_CONFIG["run_hybrid"]:
        def hybrid_fn(cond, sigma):
            b, c, h, w = cond.shape
            pure_noise = torch.randn((b, c, h, w), device=device)
            t_max = torch.full((b,), timesteps - 1, device=device, dtype=torch.long)
            student_pred, _ = student_net(pure_noise, t_max.float() / timesteps, cond, sigma)
            student_pred = torch.clamp(student_pred, 0.0, 1.0)

            t_refine = torch.full((b,), refine_steps, device=device, dtype=torch.long)
            noise = torch.randn_like(student_pred)
            hybrid_img = teacher.sqrt_alpha_bar[t_refine, None, None, None] * student_pred + \
                         teacher.sqrt_one_minus_alpha_bar[t_refine, None, None, None] * noise

            for i in reversed(range(refine_steps)):
                t_curr = torch.full((b,), i, device=device, dtype=torch.long)
                pred_noise, _ = teacher_net(hybrid_img, t_curr.float() / timesteps, cond, sigma)
                z = torch.randn_like(hybrid_img) if i > 0 else torch.zeros_like(hybrid_img)
                hybrid_img = (1 / torch.sqrt(teacher.alpha[i])) * (
                        hybrid_img - ((1 - teacher.alpha[i]) / (torch.sqrt(1 - teacher.alpha_bar[i]))) * pred_noise
                ) + torch.sqrt(teacher.beta[i]) * z
                hybrid_img = torch.clamp(hybrid_img, -1.0, 1.0)
            return torch.clamp(hybrid_img, 0.0, 1.0)

        eval_tasks.append({
            "name": f"Hybrid (1+{refine_steps}-Step)", "fn": hybrid_fn, "desc": "Hybrid",
            "params": count_parameters(student_net) + count_parameters(teacher_net), "nfe": 1 + refine_steps
        })

    if EVAL_CONFIG["run_teacher"]:
        def teacher_fn(cond, sigma):
            b, c, h, w = cond.shape
            img = torch.randn((b, c, h, w), device=device)
            for i in reversed(range(timesteps)):
                t_curr = torch.full((b,), i, device=device, dtype=torch.long)
                pred_noise, _ = teacher_net(img, t_curr.float() / timesteps, cond, sigma)
                z = torch.randn_like(img) if i > 0 else torch.zeros_like(img)
                img = (1 / torch.sqrt(teacher.alpha[i])) * (
                        img - ((1 - teacher.alpha[i]) / (torch.sqrt(1 - teacher.alpha_bar[i]))) * pred_noise
                ) + torch.sqrt(teacher.beta[i]) * z
                img = torch.clamp(img, -1.0, 1.0)
            return torch.clamp(img, 0.0, 1.0)

        eval_tasks.append({
            "name": "Teacher (1000-Step)", "fn": teacher_fn, "desc": "Teacher",
            "params": count_parameters(teacher_net), "nfe": timesteps
        })

    if EVAL_CONFIG["run_unet"]:
        def unet_fn(cond, sigma):
            # Res-UNet 直接输入条件(带噪图)，不需要 sigma
            return unet_net(cond)

        eval_tasks.append({
            "name": "Res-UNet (Regression)", "fn": unet_fn, "desc": "UNet",
            "params": count_parameters(unet_net), "nfe": 1
        })

    if not eval_tasks:
        print("❌ 没有启用任何模型，请检查 EVAL_CONFIG！")
        return

    # ==========================================
    # 4. 执行动态全面效能评测
    # ==========================================
    results = []
    for task in eval_tasks:
        res = evaluate_model(task["name"], task["fn"], full_test_loader, device, task["desc"])
        res["Params (M)"] = f"{task['params'] / 1e6:.2f}"
        res["NFE"] = task["nfe"]
        results.append(res)

    # ==========================================
    # 5. 保存评测报告
    # ==========================================
    df = pd.DataFrame(results)
    cols = ["Model", "Params (M)", "NFE", "Test Images", "MSE (↓)", "Throughput (Imgs/s) (↑)", "Time/Img (ms) (↓)",
            "Peak VRAM (MB) (↓)"]
    df = df[cols]

    print("\n" + "=" * 90)
    print("🏆 Test Set 终极全面评估报告 🏆")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    csv_path = "comprehensive_test_evaluation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 完整效能报告已保存至: {csv_path}")

    # ==========================================
    # 6. 生成动态可视化图像
    # ==========================================
    print("\n📸 正在生成视觉去噪对比图...")
    vis_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)
    cond_vis, clean_vis, _ = next(iter(vis_loader))
    cond_vis = cond_vis.to(device)
    clean_vis = clean_vis.to(device)
    sigma_vis = cond_vis.std(dim=(1, 2, 3), keepdim=True)

    # 动态组装绘图数据
    plot_data = [cond_vis.cpu().numpy()]
    titles = ["1. Noisy Input\n(Condition)"]

    with torch.no_grad():
        for idx, task in enumerate(eval_tasks):
            pred_np = task["fn"](cond_vis, sigma_vis).cpu().numpy()
            plot_data.append(pred_np)
            titles.append(f"{idx + 2}. {task['desc']}\n({task['name']})")

    plot_data.append(clean_vis.cpu().numpy())
    titles.append(f"{len(eval_tasks) + 2}. Ground Truth\n(Target)")

    # 动态计算行数，自动适配图片高度
    num_rows = len(plot_data)
    fig, axes = plt.subplots(num_rows, 8, figsize=(16, 2.2 * num_rows))
    plt.suptitle("Comprehensive Denoising Evaluation on Actual Test Set", fontsize=18)

    for row_idx, (data, title) in enumerate(zip(plot_data, titles)):
        for col_idx in range(8):
            axes[row_idx, col_idx].imshow(data[col_idx, 0], cmap='gray')
            axes[row_idx, col_idx].axis('off')
            if col_idx == 0:
                axes[row_idx, col_idx].set_title(title, fontsize=12, loc='left', pad=10)

    plt.tight_layout()
    img_path = "comprehensive_test_results.png"
    plt.savefig(img_path, dpi=300)
    print(f"✅ 可视化结果已保存至: {img_path}")
    print("\n🎉 全部评测圆满完成！")


if __name__ == "__main__":
    main()