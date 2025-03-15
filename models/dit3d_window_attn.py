import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
# from modules.myvoxelization import Voxelization
# from modules.trilinear_devoxelize import trilinear_devoxelize
from models.text_encoder import ClipTextEncoder
from visualize import visualize_point_cloud

from modules.voxelization import Voxelization
import modules.functional as F


checkpoint_path = "/Users/qinleiheng/Documents/秦磊恒/IP Paris/Master 1/Computer Vision/Project/Project Code/DiT-text-to-3D/checkpoints/checkpoint.pth"

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, X, Y, Z, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, window_size, C].
        (Xp, Yp, Zp): padded height and width before partition
    """
    print("window x.shape", x.shape)
    B, X, Y, Z, C = x.shape

    pad_x = (window_size - X % window_size) % window_size
    pad_y = (window_size - Y % window_size) % window_size
    pad_z = (window_size - Z % window_size) % window_size
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        x = F.pad(x, (0, 0, 0, pad_x, 0, pad_y))

    Xp, Yp, Zp = X + pad_x, Y + pad_y, Z + pad_z

    x = x.view(B, Xp // window_size, window_size, Yp // window_size, window_size, Zp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Xp, Yp, Zp)


def window_unpartition(windows, window_size, pad_xyz, xyz):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, window_size, C].
        window_size (int): window size.
        pad_xyz (Tuple): padded height and width (Xp, Yp, Zp).
        xyz (Tuple): original height and width (X, Y, Z) before padding.

    Returns:
        x: unpartitioned sequences with [B, X, Y, Z, C].
    """
    Xp, Yp, Zp = pad_xyz
    X, Y, Z = xyz
    B = windows.shape[0] // (Xp * Yp * Zp // window_size // window_size // window_size)
    x = windows.view(B, Xp // window_size, Yp // window_size, Zp // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Xp, Yp, Zp, -1)

    if Xp > X or Yp > Y or Zp > Z:
        x = x[:, :X, :Y, :Z, :].contiguous()
    return x
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, X, Y, Z, _ = x.shape
        qkv = self.qkv(x).reshape(B, X*Y*Z, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, X * Y * Z, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, X, Y, Z, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, X, Y, Z, -1)
        x = self.proj(x)
        return x

class FlashAttention(nn.Module):
    """基于 Flash Attention 的多头注意力模块。
    
    输入 x 的形状为 (B, X, Y, Z, C)，我们会先用一个线性层生成 q, k, v，
    然后调用 torch.nn.functional.scaled_dot_product_attention 来计算注意力。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # 注意这里输出维度为 3*dim，后面再拆分成 q, k, v
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        # x: (B, X, Y, Z, C)
        B, X, Y, Z, C = x.shape
        seq_len = X * Y * Z
        # 生成 q, k, v 并转换形状为 (B, num_heads, seq_len, head_dim)
        qkv = self.qkv(x).reshape(B, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 使用 PyTorch 内置的 flash attention 计算注意力，内部会自动除以 sqrt(head_dim)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        # out: (B, num_heads, seq_len, head_dim)
        # 将注意力输出还原成原始的空间维度
        out = out.transpose(1, 2).reshape(B, X, Y, Z, C)
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
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2)
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


# class LabelEmbedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """
#     def __init__(self, hidden_size, dropout_prob):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.encoder = ClipTextEncoder(dropout_prob=dropout_prob if self.training else 0)
#         self.proj = nn.Linear(self.encoder.output_dim, hidden_size)
        
#     def forward(self, labels, force_drop_ids=None):
#         with torch.no_grad():  # 确保 CLIP 编码器不更新
#             text_features = self.encoder.encode_text(labels, force_drop_ids=force_drop_ids)  # (batch, 512)
        
#         embeddings = self.proj(text_features)  # 线性映射到 hidden_size 维度
#         return embeddings 


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, 
                 num_heads, 
                 mlp_ratio=4.0, 
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 window_size=0,
                 use_residual_block=False,
                 input_size=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = FlashAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
        )

        self.input_size = input_size
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU() # for torch 1.7.1
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.window_size = window_size

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
                
        shortcut = x
        x = self.norm1(x)
        B = x.shape[0]
        x = x.reshape(B, self.input_size[0], self.input_size[1], self.input_size[2], -1)
        
        # Window partition
        # print("daozhele", self.window_size)
        if self.window_size > 0:
            X, Y, Z = x.shape[1], x.shape[2], x.shape[3]
            x, pad_xyz = window_partition(x, self.window_size)

        x = x.reshape(B, self.input_size[0] * self.input_size[1] * self.input_size[2], -1)
        x = modulate(x, shift_msa, scale_msa)
        x = x.reshape(B, self.input_size[0], self.input_size[1], self.input_size[2], -1)

        x = self.attn(x)
        
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_xyz, (X, Y, Z))
        x = x.reshape(B, self.input_size[0] * self.input_size[1] * self.input_size[2], -1)

        x = shortcut + gate_msa.unsqueeze(1) * x
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
        learn_sigma=False,
        window_size=0,
        num_classes=1,
        window_block_indexes=(),
        use_rel_pos=False,
        rel_pos_zero_init=True
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
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # print("depth", depth)
        # print("hiddensize", hidden_size)
        # print("window_block_indexes", window_block_indexes)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio,
                     use_rel_pos=use_rel_pos,
                     rel_pos_zero_init=rel_pos_zero_init,
                     window_size=window_size if i in window_block_indexes else 0,
                     input_size=(input_size // patch_size, input_size // patch_size, input_size // patch_size)
                     ) for i in range(depth)
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
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

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
        print("forward,",x.shape)
        features, coords = x, x

        x, voxel_coords = self.voxelization(features, coords)

        x = self.x_embedder(x) 
        x = x + self.pos_embed

        t = self.t_embedder(t)                  
        # y = self.y_embedder(y)    
        y = self.y_embedder(y, self.training)    
        c = t + y                           

        for block in self.blocks:
            x = block(x, c)                      
        x = self.final_layer(x, c)
        x = self.unpatchify_voxels(x)           

        # Devoxelization
        x = F.trilinear_devoxelize(x, voxel_coords, self.input_size, self.training)
        # print(x[0].transpose(0,1).shape)
        # visualize_point_cloud(x[0].transpose(0,1))
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


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
    # print('grid_size:', grid_size)

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
        load_partial_checkpoint(model, checkpoint_path, ignore_prefixes=["y_embedder."])
    return model

# def load_partial_checkpoint(model, checkpoint_path, ignore_prefixes=["y_embedder."]):
def load_partial_checkpoint(model, checkpoint_path, ignore_prefixes=[]):
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
        new_state[key] = value
        # if key in model_state and not any(key.startswith(prefix) for prefix in ignore_prefixes):
        #     new_state[key] = value
        # else:
        #     print(f"Skip loading key: {key}")

    # 更新当前模型的 state_dict
    model_state.update(new_state)
    model.load_state_dict(model_state)
    print("Partial checkpoint loaded. Loaded {} parameters.".format(len(new_state)))
    # 加上这行:
    return model

DiT3D_models_WindAttn = {
    'DiT-S/4':  DiT_S_4
}