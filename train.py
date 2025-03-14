import torch
from torch.utils.data import DataLoader
import json
from datasets.data_preprocessing import ShapeNet15kPointClouds

DATASET_PATH = "datasets/shapenet_data_5000_splitted"
DATA_POINTS_SIZE = 5000
RESULTS_JSON_PATH = "datasets/results.json"
CATEGORY = ['chair', 'airplane']
BATCH_SIZE = 8
WINDOW_SIZE = 4
WORKERS = 4
DEPTH = 12
window_block_indexes = (0,3,6,9)
voxel_size = 32
lr=1e-4
beta_start = 1e-5
beta_end = 0.008
time_num = 1000
iteration_num = 5000
checkpoints_dir = "checkpoints"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
# from models.dit3d import DiT3D_models
from models.dit3d_window_attn import DiT3D_models_WindAttn

def get_dit3d_model():
    if WINDOW_SIZE > 0:
        return DiT3D_models_WindAttn["DiT-S/4"](
        pretrained=False,
        input_size=voxel_size,
        window_size=WINDOW_SIZE,
        window_block_indexes=window_block_indexes, 
        )
    # else :
    #     return DiT3D_models["DiT_S_4"](
    #     pretrained=False,
    #     input_size=voxel_size,
    #     num_classes=num_classes
    # )


# -------------------------------------
# 2. Diffusion 过程定义
# -------------------------------------
class GaussianDiffusion:
    def __init__(self, betas, device, loss_type="mse"):
        self.loss_type = loss_type
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.num_timesteps = len(betas)

        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)


    def q_sample(self, x_start, t, noise=None):
        """前向扩散（Forward Process）"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t]).reshape(-1, 1, 1)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - self.alphas_cumprod[t]).reshape(-1, 1, 1)
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha * noise

    def p_losses(self, model, x_start, t, noise=None, y=None):
        """计算损失"""
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        print("x_noisy", x_noisy.shape)
        pred_noise = model(x_noisy, t, y)

        if self.loss_type == "mse":
            return nn.functional.mse_loss(pred_noise, noise)


def train():
    # 设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(RESULTS_JSON_PATH, "r") as f:
            text_annotations = json.load(f)  # 加载 JSON 文件

    train_dataset = ShapeNet15kPointClouds(categories=CATEGORY, split="train", text_annotations=text_annotations)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    # 选择模型
    model = get_dit3d_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 生成 beta schedule（用于扩散过程）
    betas = np.linspace(beta_start, beta_end, time_num)
    diffusion = GaussianDiffusion(betas, device=device)

    # 训练循环
    for epoch in range(iteration_num):
        for i, data in enumerate(train_loader):
            print(data["train_points"].shape)
            x = data["train_points"].transpose(1, 2).to(device)  # 形状: [B, 3, 2048]
            print(x.shape)
            y = data["text"]#TODO

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)  # 随机时间步
            loss = diffusion.p_losses(model, x, t, y=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch}/{iteration_num}] [Batch {i}/{len(train_loader)}] Loss: {loss.item():.6f}")

        # 每 100 轮保存一次模型
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"{checkpoints_dir}/dit3D_epoch{epoch}.pth")


# -------------------------------------
# 5. 运行训练
# -------------------------------------
if __name__ == "__main__":
    train()
