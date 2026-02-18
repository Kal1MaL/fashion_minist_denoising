import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.dataset import FashionMNISTDenoisingDataset
from src.x_predict import XPredictDistiller


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备数据（我们取 train 或 val 里的一小批数据，这样有 Ground Truth 可以对比）
    dataset = FashionMNISTDenoisingDataset(cfg.data.train_path, cfg.data.clean_path, mode='train')
    # 随机抽 8 张图来看看
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    conditions, clean_imgs, labels = batch
    conditions = conditions.to(device)
    clean_imgs = clean_imgs.to(device)

    # 2. 载入我们刚刚出炉的 1-Step 神器
    # ⚠️ 请确保这里的名字和你的 ckpt 文件名完全一致
    CKPT_PATH = "checkpoints_distilled/x-pred-best-epoch=08-val_1step_mse=0.0049.ckpt"

    print(f"Loading 1-step model from {CKPT_PATH}...")
    model = XPredictDistiller.load_from_checkpoint(CKPT_PATH)
    model.to(device)
    model.eval()

    # 3. 极速推理 (见证奇迹的时刻，绝对是一瞬间的事)
    print("Running 1-step inference...")
    with torch.no_grad():
        # 这里调用的就是我们在 XPredictDistiller 里写的直接预测代码
        pred_imgs = model.sample(condition=conditions)

    # 4. 画图大比拼
    conditions_np = conditions.cpu().numpy()
    pred_imgs_np = pred_imgs.cpu().numpy()
    clean_imgs_np = clean_imgs.cpu().numpy()

    fig, axes = plt.subplots(3, 16, figsize=(16, 6))
    plt.suptitle("1-Step X-Prediction Distillation Results", fontsize=16)

    for i in range(16):
        # 第一行：输入的带噪图像 (Condition)
        axes[0, i].imshow(conditions_np[i, 0], cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Noisy Input\n(Condition)", fontsize=12)

        # 第二行：学生模型 1-Step 一步预测的结果
        axes[1, i].imshow(pred_imgs_np[i, 0], cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("1-Step Pred\n(Student)", fontsize=12)

        # 第三行：真实的干净图像 (Ground Truth)
        axes[2, i].imshow(clean_imgs_np[i, 0], cmap='gray')
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title("Ground Truth\n(Target)", fontsize=12)

    plt.tight_layout()
    plt.savefig("1_step_results.png", dpi=300)
    print("✅ Visualization saved to '1_step_results.png'!")

    # 如果你在有界面的系统上，可以直接弹窗显示
    # plt.show()


if __name__ == "__main__":
    main()