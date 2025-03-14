import torch
import torch.nn.functional as F

def trilinear_devoxelize(voxel_features, voxel_coords, resolution, training=True):
    """
    将体素网格中的特征根据连续坐标通过三线性插值还原出来。
    
    参数：
        voxel_features: torch.Tensor, 形状 [B, C, R, R, R]，体素化后的特征。
        voxel_coords: torch.Tensor, 形状 [B, 3, N]，每个点云中 N 个点的连续坐标，取值范围 [0, 1]。
        resolution: int, 网格分辨率 R。
        training: bool, 训练模式标志（本实现中不作特殊处理）。
    
    返回：
        输出: torch.Tensor, 形状 [B, C, N]，每个点通过三线性插值得到的特征。
    """
    # 将 voxel_coords 从 [B, 3, N] 转换为 [B, N, 3]
    coords = voxel_coords  # [B, N, 3]
    # 将坐标从 [0, 1] 映射到 [-1, 1]（grid_sample 的要求）
    grid = 2.0 * coords - 1.0  # [B, N, 3]
    
    # grid_sample 要求网格的形状为 [B, D_out, H_out, W_out, 3]，这里我们希望对每个点进行采样，
    # 所以将 grid 扩展为 [B, N, 1, 1, 3]
    grid = grid.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1, 3]
    
    # 使用 grid_sample 进行三线性插值，注意：mode='bilinear' 在 3D 中也实现了三线性插值
    sampled = F.grid_sample(voxel_features, grid, mode='bilinear', align_corners=True)
    # grid_sample 返回 [B, C, N, 1, 1]，去除多余的维度得到 [B, C, N]
    sampled = sampled.squeeze(-1).squeeze(-1)
    
    return sampled