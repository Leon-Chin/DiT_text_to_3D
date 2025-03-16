import torch
from torch.utils.data import DataLoader
import json

from datasets.data_preprocessing import ShapeNet15kPointClouds
import visualize

DATASET_PATH = "datasets/shapenet_data_5000_splitted"
RESULTS_JSON_PATH = "datasets/results.json"
GENERATED_PATH = "output/generated_samples.npy"
DATA_POINTS_SIZE = 5000
DATA_POINTS_IN_MODEL_SIZE = 3500
CATEGORY = ['chair']
BATCH_SIZE = 8
WINDOW_SIZE = 4
WORKERS = 4
DEPTH = 24
window_block_indexes = (0,3,6,9)
voxel_size = 32
lr=2e-4
beta_start = 1e-5
beta_end = 0.008
time_num = 1000
iteration_num = 0
checkpoints_dir = "checkpoints"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from models.dit3d import DiT3D_models
from models.dit3d_window_attn import DiT3D_models_WindAttn

class Model(nn.Module):
    def __init__(self, diffusion, base_model):
        super(Model, self).__init__()
        self.diffusion = diffusion
        self.model = base_model

    def _denoise(self, data, t, y):
        return self.model(data, t, y)

    def get_loss_iter(self, data, noise=None, y=None):
        B = data.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=data.device)
        return self.diffusion.p_losses(self._denoise, data, t, noise, y)

    def gen_samples(self, shape, device, y, noise_fn=torch.randn, clip_denoised=True):
        return self.diffusion.p_sample_loop(self._denoise, shape, device, y, noise_fn, clip_denoised)

    def gen_sample_traj(self, shape, device, y, freq, noise_fn=torch.randn, clip_denoised=True):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape, device, y, freq, noise_fn, clip_denoised)
    
    def gen_samples_ddim(self, shape, device, y, eta=0.0, clip_denoised=True):
        return self.diffusion.ddim_sample_loop(self._denoise, shape, device, y, eta, clip_denoised)

def get_dit3d_model():
    if WINDOW_SIZE > 0:
        return DiT3D_models_WindAttn["DiT-S/4"](
        pretrained=True,
        input_size=voxel_size,
        window_size=WINDOW_SIZE,
        window_block_indexes=window_block_indexes, 
        )
    # return DiT3D_models["DiT-S/4"](
    #     pretrained=False,
    #     input_size=voxel_size,
    # )

def freeze_model_except(model, training_layers=[]):
    for name, param in model.named_parameters():
        # 默认冻结所有参数
        param.requires_grad = False
        # 如果名称中包含任一指定关键词，则解冻
        for layer_keyword in training_layers:
            print(layer_keyword, name, layer_keyword in name)
            if layer_keyword in name:
                param.requires_grad = True
                break

def print_frozen_params(model):
    total_params = 0
    frozen_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        if param.requires_grad:
            trainable_params += n
        else:
            frozen_params += n
    print(f"总参数量: {total_params}")
    print(f"可训练参数量: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")
    print(f"冻结参数量: {frozen_params} ({100 * frozen_params / total_params:.2f}%)")


class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        betas = betas.astype(np.float64)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        bs, = t.shape
        out = torch.gather(a, 0, t)
        return torch.reshape(out, [bs] + [1] * (len(x_shape) - 1))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    def p_mean_variance(self, denoise_fn, data, t, y, clip_denoised: bool, return_pred_xstart: bool):
        model_output = denoise_fn(data, t, y)
        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            model_variance, model_log_variance = {
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)
            if clip_denoised:
                x_recon = torch.clamp(x_recon, -0.5, 0.5)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def p_sample(self, denoise_fn, data, t, noise_fn, y, clip_denoised=False, return_pred_xstart=False):
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data, t, y, clip_denoised, return_pred_xstart=True)
        noise = noise_fn(data.shape, dtype=data.dtype, device=data.device)
        nonzero_mask = (1 - (t == 0).float()).view(data.shape[0], *([1] * (len(data.shape) - 1)))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        if return_pred_xstart:
            return sample, pred_xstart
        else:
            return sample

    def p_sample_loop(self, denoise_fn, shape, device, y, noise_fn=torch.randn, clip_denoised=True):
        img_t = noise_fn(shape, dtype=torch.float, device=device)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.int64, device=device)
            img_t = self.p_sample(denoise_fn, img_t, t_tensor, noise_fn, y, clip_denoised=clip_denoised)
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, y, freq, noise_fn=torch.randn, clip_denoised=True):
        total_steps = self.num_timesteps
        img_t = noise_fn(shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(total_steps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.int64, device=device)
            img_t = self.p_sample(denoise_fn, img_t, t_tensor, noise_fn, y, clip_denoised=clip_denoised)
            if t % freq == 0 or t == total_steps - 1:
                imgs.append(img_t)
        return imgs

    def p_losses(self, denoise_fn, data_start, t, noise=None, y=None):
        B = data_start.shape[0]
        if noise is None:
            noise = torch.randn_like(data_start)
        data_t = self.q_sample(data_start, t, noise)
        if self.loss_type == 'mse':
            eps_recon = denoise_fn(data_t, t, y)
            losses = ((noise - eps_recon) ** 2).mean(dim=list(range(1, len(data_start.shape))))
        else:
            raise NotImplementedError(self.loss_type)
        return losses

    def ddim_sample(self, denoise_fn, x_t, t, y, eta=0.2, clip_denoised=True):
        """
        DDIM 单步采样更新
        eta：控制随机性，eta=0时为确定性采样
        """
        # 预测噪声
        eps = denoise_fn(x_t, t, y)
        # 提取当前时间步的 alpha_bar
        alpha_bar = self._extract(self.alphas_cumprod.to(x_t.device), t, x_t.shape)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        # 根据公式预测 x0
        x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar
        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -0.5, 0.5)
        # 提取上一步的 alpha_bar，即 alpha_bar_{t-1}
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev.to(x_t.device), t, x_t.shape)
        # 计算 DDIM 更新中的 sigma
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
        # 当 eta = 0 时噪声项为 0，实现确定性采样
        noise = torch.randn_like(x_t) if eta > 0 else 0.
        # DDIM 更新公式
        x_t_prev = torch.sqrt(alpha_bar_prev) * x0_pred + \
                   torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps + \
                   sigma * noise
        return x_t_prev

    def ddim_sample_loop(self, denoise_fn, shape, device, y, eta=0.0, clip_denoised=True):
        """
        DDIM 多步采样
        """
        x = torch.randn(shape, dtype=torch.float, device=device)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.int64, device=device)
            x = self.ddim_sample(denoise_fn, x, t_tensor, y, eta, clip_denoised)
        return x

def train_model(model, dataloader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            # 假设 data 字典中包含 'train_points' 与 'cate_idx'
            x = data['train_points'].transpose(1, 2).to(device)
            y = data['cate_idx'].to(device)
            loss = model.get_loss_iter(x, None, y).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch}/ Iteration {i}/{len(dataloader)}]: Loss: {loss.item():.4f}")
        
        # 每 50 轮保存一次模型
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"{checkpoints_dir}/dit3D_epoch{epoch}.pth")

def main(sampling_method):
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with open(RESULTS_JSON_PATH, "r") as f:
        text_annotations = json.load(f)  # 加载 JSON 文件

    train_dataset = ShapeNet15kPointClouds(categories=CATEGORY, split="train", text_annotations=text_annotations)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True,)

    # 生成 beta schedule（用于扩散过程）
    betas = np.linspace(beta_start, beta_end, time_num)
    diffusion = GaussianDiffusion(betas, loss_type='mse', model_mean_type='eps', model_var_type='fixedsmall')
    
    # 选择模型
    model_type = get_dit3d_model()
    freeze_model_except(model_type, ["y_embedder"])
    print_frozen_params(model=model_type)
    model = Model(diffusion, model_type).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_model(model, dataloader, optimizer, iteration_num, device)

    # 生成样本示例
    model.eval()
    with torch.no_grad():
        sample_shape = (BATCH_SIZE, 3, DATA_POINTS_IN_MODEL_SIZE)  # 假设点云3通道
        # 此处 y 可以视情况设定，例如全 0 表示某一类别
        y_gen = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        if sampling_method == "ddpm":
            samples = model.gen_samples(sample_shape, device, y_gen, noise_fn=torch.randn, clip_denoised=False)
            print("Generated samples shape:", samples.shape)
        elif sampling_method == "ddim":
            samples = model.gen_samples_ddim(sample_shape, device, y_gen, eta=0.5, clip_denoised=False)
            print("DDIM generated samples shape:", samples.shape)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
        np.save(GENERATED_PATH, samples.cpu().numpy())

        # # 生成采样轨迹（可用于观察生成过程）
        # traj = model.gen_sample_traj(sample_shape, device, y_gen, freq=40, noise_fn=torch.randn, clip_denoised=False)
        # print("Generated sample trajectory length:", len(traj))

if __name__ == "__main__":
    main(sampling_method = "ddim")
