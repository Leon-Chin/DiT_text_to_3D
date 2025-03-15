import torch
import numpy as np
import argparse
import os

from models.dit3d import DiT3D_models  # 假设你的 DiT 模型在这里定义
from modules.trilinear_devoxelize import trilinear_devoxelize

DATASET_PATH = "datasets/shapenet_data_5000_splitted"
DATA_POINTS_SIZE = 3500
RESULTS_JSON_PATH = "datasets/results.json"
CATEGORY = ['chair', 'airplane']
BATCH_SIZE = 8
WINDOW_SIZE = 4
WORKERS = 4
DEPTH = 24
window_block_indexes = (0,3,6,9)
voxel_size = 32
lr=1e-5
beta_start = 1e-5
beta_end = 0.008
time_num = 1000
iteration_num = 10000
checkpoints_dir = "checkpoints"

# 定义扩散过程类，与训练时一致
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

@torch.no_grad()
def ddim_sample(model, diffusion, shape, num_steps, device, y=None, cfg_scale=0.0):
    """
    利用 DDIM 采样生成 3D 点云。
    
    参数:
      - model: 训练好的 3D DiT 模型
      - diffusion: GaussianDiffusion 实例
      - shape: 输出形状，(B, 3, N)，如 (1, 3, 2048)
      - num_steps: 采样步数（通常小于 diffusion.num_timesteps）
      - y: 条件信息（文本描述或其它）
      - cfg_scale: 若 >0，则使用 classifier-free guidance

    返回:
      - (B, 3, N) 的采样结果
    """
    print("ddim",shape)
    B, C, N = shape
    # 1) 初始化 x_t 为随机噪声
    x = torch.randn(shape, device=device)  # (B, 3, N)

    # 2) 定义采样时间表 (简单用 linspace 等距，也可用自定义 schedule)
    timesteps = torch.linspace(diffusion.num_timesteps - 1, 0, num_steps, device=device, dtype=torch.long)

    # 3) 逐步反演
    for step_idx, t in enumerate(timesteps):
        t_long = t.long()  # 当前整型时间步
        t_batch = torch.full((B,), t_long, device=device, dtype=torch.long)

        # 3.1 通过模型预测 eps (或 x0 等)
        # if cfg_scale > 0 and hasattr(model, "forward_with_cfg"):
        #     eps = model.forward_with_cfg(x, t_batch, y, cfg_scale)  # (B, 3, N)
        # else:
        eps = model(x, t_batch, y)  # (B, 3, N)

        # 3.2 计算 alpha_bar (对应 t)
        alpha_bar = diffusion.alphas_cumprod[t_long]
        if t_long > 0:
            alpha_bar_prev = diffusion.alphas_cumprod[t_long - 1]
        else:
            alpha_bar_prev = diffusion.alphas_cumprod[0]

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_bar_prev)

        # 3.3 DDIM 公式：预测 x0
        #     x0 = (x - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
        x0 = (x - sqrt_one_minus_alpha * eps) / sqrt_alpha_bar

        # 3.4 如果还没到最后一步，则往前一步
        if step_idx < num_steps - 1:
            # 这里演示一个最简约的“无梯度 DDIM”，
            # x_{t-1} = sqrt(alpha_bar_{t-1}) * x0 + sqrt(1 - alpha_bar_{t-1}) * eps
            # 你可以在此处插入自定义的 DDIM/DDPM 等更多细节。
            x = sqrt_alpha_bar_prev * x0 + sqrt_one_minus_alpha_prev * eps
        else:
            # 最后一步就直接用 x0
            x = x0

    return x

def load_partial_checkpoint(model, checkpoint_path, ignore_prefixes=["y_embedder."]):
    """
    加载 checkpoint 中与当前模型匹配的参数，忽略那些键以 ignore_prefixes 开头的部分
    :param model: 当前模型
    :param checkpoint_path: checkpoint 文件路径
    :param ignore_prefixes: 一个列表，包含要忽略加载的层的前缀，例如 "y_embedder." 表示不加载该层的参数
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")["model_state"]
    model_state = model.state_dict()
    # print("model_state", model_state.keys())
    new_state = {}
    for key, value in checkpoint.items():
        key = key.replace("model.module.", "")
        # print("key", key)
        # 如果键不以任一 ignore_prefix 开头，并且在当前模型中存在，则加载该参数
        if key in model_state and not any(key.startswith(prefix) for prefix in ignore_prefixes):
            new_state[key] = value
        else:
            print(f"Skip loading key: {key}")

    # 更新当前模型的 state_dict
    model_state.update(new_state)
    model.load_state_dict(model_state)
    print("Partial checkpoint loaded. Loaded {} parameters.".format(len(new_state)))
    return model
    
checkpoint_path = "/Users/qinleiheng/Documents/秦磊恒/IP Paris/Master 1/Computer Vision/Project/Project Code/DiT-text-to-3D/checkpoints/checkpoint.pth"

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 这里选择 DiT-S/4 模型（请根据实际情况选择合适的模型）
    model = DiT3D_models["DiT-S/4"](pretrained=False, input_size=args.voxel_size).to(device)
    model = load_partial_checkpoint(model, checkpoint_path, ignore_prefixes=["y_embedder."])
    
    model.eval()
    print("Model loaded.")

    # 生成 beta schedule，与训练时一致
    betas = np.linspace(args.beta_start, args.beta_end, args.time_num)
    diffusion = GaussianDiffusion(betas, device=device)

    # 定义采样输出形状
    # 假设输出为 devoxelized 3D 对象，形状为 (B, 3, voxel_size, voxel_size, voxel_size)
    shape = (len(args.conditions), 3, DATA_POINTS_SIZE)
    
    # 使用 DDIM 采样生成 3D 对象
    samples = ddim_sample(model, diffusion, shape, args.num_steps, device, y=args.conditions, cfg_scale=args.cfg_scale)
    print("predicted", samples.shape)
    # 如果需要，可以对生成结果进行后处理，比如 devoxelize（这里假设 trilinear_devoxelize 已内嵌于模型 forward）
    # 此处直接保存为 numpy 数组
    output_path = args.output
    np.save(output_path, samples.cpu().numpy())
    print(f"Generated samples saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Object Generation using DiT and DDIM Sampling")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="generated_samples.npy", help="File to save generated samples")
    parser.add_argument("--voxel_size", type=int, default=32, help="Voxel resolution (assumed cubic)")
    parser.add_argument("--beta_start", type=float, default=1e-5, help="Beta start value")
    parser.add_argument("--beta_end", type=float, default=0.008, help="Beta end value")
    parser.add_argument("--time_num", type=int, default=1000, help="Total diffusion timesteps used during training")
    parser.add_argument("--num_steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale (0 for none)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--conditions", type=str, nargs="+", default=[], help="Conditional text prompt (if any)")
    args = parser.parse_args()
    main(args)
