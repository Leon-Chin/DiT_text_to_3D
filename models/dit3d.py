# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from modules.myvoxelization import Voxelization
from modules.trilinear_devoxelize import trilinear_devoxelize
from models.text_encoder import ClipTextEncoder




def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttention(nn.Module):
    """
    Flash Attention for input shape (B, N, dim).
    
    其中:
      - B: batch size
      - N: sequence length (可以是 flatten 后的 patch/voxel 数量)
      - dim: embedding dimension
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (B, N, dim)
        返回: (B, N, dim)
        """
        B, N, C = x.shape
        # 1. 线性映射得到 q, k, v
        qkv = self.qkv(x)              # (B, N, 3*dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # 2. 重新排列: (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别是 (B, num_heads, N, head_dim)

        # 3. 使用 PyTorch 2.0 内置 flash attention
        # scaled_dot_product_attention 会自动除以 sqrt(head_dim)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        # out: (B, num_heads, N, head_dim)

        # 4. 还原回 (B, N, dim)
        out = out.transpose(1, 2).reshape(B, N, C)

        # 5. 再做一次线性映射
        out = self.proj(out)
        return out


class PatchEmbed_Voxel(nn.Module):
    """ Voxel to Patch Embedding
    """
    def __init__(self, voxel_size=32, patch_size=4, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        voxel_size = (voxel_size, voxel_size, voxel_size)
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (voxel_size[0] // patch_size[0]) * (voxel_size[1] // patch_size[1]) * (voxel_size[2] // patch_size[2])
        self.patch_xyz = (voxel_size[0] // patch_size[0], voxel_size[1] // patch_size[1], voxel_size[2] // patch_size[2])
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        # print("patchEmbed_Voxel", x.shape)
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2)
        # print('patchEmbed_Voxel output shape', x.shape)
        return x
    

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = ClipTextEncoder(dropout_prob=dropout_prob if self.training else 0)
        self.proj = nn.Linear(self.encoder.output_dim, hidden_size)
        
    def forward(self, labels, force_drop_ids=None):
        with torch.no_grad():  # 确保 CLIP 编码器不更新
            text_features = self.encoder.encode_text(labels, force_drop_ids=force_drop_ids).float()  # (batch, 512)
        
        embeddings = self.proj(text_features)  # 线性映射到 hidden_size 维度
        return embeddings 


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = FlashAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        approx_gelu = lambda: nn.GELU() # for torch 1.7.1
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.input_size = input_size
        self.voxelization = Voxelization(resolution=input_size, normalize=True, eps=1e-7)

        self.x_embedder = PatchEmbed_Voxel(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(hidden_size, class_dropout_prob)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.input_size//self.patch_size))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.proj.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify_voxels(self, x0):
        """
        input: (N, T, patch_size * patch_size * patch_size * C)    (N, 64, 8*8*8*3)
        voxels: (N, C, X, Y, Z)          (N, 3, 32, 32, 32)
        """
        c = self.out_channels
        p = self.patch_size
        x = y = z = self.input_size // self.patch_size
        assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        points = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return points

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, P) tensor of spatial inputs (point clouds or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        # Voxelization
        features = x.permute(0, 2, 1).contiguous()  # (B, num_points, 3)
        coords = x.permute(0, 2, 1).contiguous()    # (B, num_points, 3)
        print(x.shape)
        x, voxel_coords = self.voxelization(features, coords)

        x = self.x_embedder(x) 
        x = x + self.pos_embed 

        t = self.t_embedder(t)  
        y = self.y_embedder(y)    
        c = t + y                                

        for block in self.blocks:
            x = block(x, c)                      
        x = self.final_layer(x, c)                
        x = self.unpatchify_voxels(x)                   

        # Devoxelization
        x = trilinear_devoxelize(x, voxel_coords, self.input_size, self.training)
        print("output", x.shape)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale=1.0):
        """
        Forward pass of DiT with Classifier-Free Guidance (CFG).
        
        Args:
            x (Tensor): 形状为 (B, 3, N) 的点云（或其他输入），B是批大小，N是点数。
            t (Tensor): 形状为 (B,) 的扩散时间步。
            y (List[str] 或其他): 文本标签或类别标签批。
            cfg_scale (float): CFG 强度系数。1.0 表示无指导，大于1会加大条件的影响。
            
        Returns:
            Tensor: shape (B, 3, N) 的输出，结合了无条件和有条件的结果。
        """
        # -------------------------------------------------------------------------
        # 1) 先进行体素化，把连续点云（x）转为体素 (x_voxel, voxel_coords)
        # -------------------------------------------------------------------------
        features = x.permute(0, 2, 1).contiguous()  # (B, N, 3)
        coords   = x.permute(0, 2, 1).contiguous()  # (B, N, 3)
        x_voxel, voxel_coords = self.voxelization(features, coords)
        
        # -------------------------------------------------------------------------
        # 2) 无条件 (unconditional) 分支
        #    这里通过 force_drop_ids 来让文本编码器全部“熔断”，从而得到无条件的 y embedding
        # -------------------------------------------------------------------------
        x_uncond = self.x_embedder(x_voxel)          # (B, num_patches, hidden_size)
        x_uncond = x_uncond + self.pos_embed         # 加上固定的 sine-cos 位置编码

        t_emb = self.t_embedder(t)  # (B, hidden_size)
        # 根据你的 LabelEmbedder 实现，force_drop_ids 通常可以传入整批的索引，强制让文本全部 dropout
        y_uncond_emb = self.y_embedder(y, force_drop_ids=torch.arange(len(y), device=x.device))
        c_uncond = t_emb + y_uncond_emb              # (B, hidden_size)

        for block in self.blocks:
            x_uncond = block(x_uncond, c_uncond)
        out_uncond = self.final_layer(x_uncond, c_uncond)  # (B, num_patches, patch_size^3 * out_channels)
        out_uncond = self.unpatchify_voxels(out_uncond)    # (B, out_channels, X, Y, Z)
        out_uncond = trilinear_devoxelize(
            out_uncond, voxel_coords, self.input_size, self.training
        )  # (B, out_channels, N)，在你的代码里 out_channels 通常是3 或 6

        # -------------------------------------------------------------------------
        # 3) 有条件 (conditional) 分支
        #    使用真实的文本标签 y 来获取正常的文本 embedding
        # -------------------------------------------------------------------------
        x_cond = self.x_embedder(x_voxel)
        x_cond = x_cond + self.pos_embed

        y_cond_emb = self.y_embedder(y)  
        c_cond = t_emb + y_cond_emb

        for block in self.blocks:
            x_cond = block(x_cond, c_cond)
        out_cond = self.final_layer(x_cond, c_cond)
        out_cond = self.unpatchify_voxels(out_cond)
        out_cond = trilinear_devoxelize(
            out_cond, voxel_coords, self.input_size, self.training
        )  # (B, out_channels, N)

        # -------------------------------------------------------------------------
        # 4) 用 CFG 的公式组合无条件和有条件结果
        #    out = out_uncond + cfg_scale * (out_cond - out_uncond)
        # -------------------------------------------------------------------------
        out = out_uncond + cfg_scale * (out_cond - out_uncond)
        return out



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    print('grid_size:', grid_size)

    grid_x = np.arange(grid_size, dtype=np.float32)
    grid_y = np.arange(grid_size, dtype=np.float32)
    grid_z = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (X*Y*Z, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def DiT_S_4(pretrained=True, **kwargs):

    model = DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)
    if pretrained:
        checkpoint = torch.load('checkpoints/dit3D_epoch499.pth', map_location='cpu')
        checkpoint_blocks = {k: checkpoint[k] for k in checkpoint if k.startswith('blocks')}
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model

DiT3D_models = {
    'DiT-S/4':  DiT_S_4
}