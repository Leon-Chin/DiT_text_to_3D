import torch
import torch.nn.functional as F

def trilinear_devoxelize(voxel_features, voxel_coords, resolution, training=True):
    """
    用 PyTorch 实现三线性插值替代 `F.trilinear_devoxelize`
    
    :param voxel_features: 体素特征, shape=(B, C, X, Y, Z)
    :param voxel_coords: 归一化的点云坐标, shape=(B, N, 3), 取值范围 [0, 1]
    :param resolution: 体素网格的分辨率 (X, Y, Z)
    :param training: 是否为训练模式，默认 True
    
    :return: 点云特征, shape=(B, C, N)
    """

    B, C, X, Y, Z = voxel_features.shape  # 体素特征的形状
    _, N, _ = voxel_coords.shape          # 点云坐标的形状

    # 归一化到 [-1, 1]，适配 `grid_sample` 输入要求
    norm_coords = 2.0 * voxel_coords - 1.0  # [B, N, 3] -> [-1, 1]

    # 调整形状适配 `grid_sample` 需要的 (B, D, H, W, C) 格式
    voxel_features = voxel_features.unsqueeze(2)  # (B, C, 1, X, Y, Z)
    
    # 使用 `grid_sample` 进行三线性插值
    sampled_features = F.grid_sample(
        voxel_features,           # (B, C, 1, X, Y, Z)
        norm_coords.unsqueeze(1), # (B, 1, N, 3) 作为 grid
        mode='bilinear',          # 采用双线性插值（三维情况下仍然有效）
        align_corners=True
    )  # 结果形状: (B, C, 1, N)

    return sampled_features.squeeze(2)  # 返回 (B, C, N)
