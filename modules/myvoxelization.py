import torch
import torch.nn as nn

class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.long)

        return self.avg_voxelize(features, vox_coords, self.r), norm_coords

    def avg_voxelize(self, features, vox_coords, resolution):
        """
        使用 PyTorch 实现 avg_voxelize：
        1. 将点云特征 `features` 投影到 `vox_coords` 体素网格。
        2. 计算每个体素中的平均特征。
        """
        batch_size, num_points, feature_dim = features.shape

        # 计算每个体素的索引
        vox_indices = vox_coords[..., 0] * resolution**2 + vox_coords[..., 1] * resolution + vox_coords[..., 2]

        # 创建一个空的体素特征存储空间
        voxel_features = torch.zeros((batch_size, resolution**3, feature_dim), device=features.device)
        voxel_counts = torch.zeros((batch_size, resolution**3, 1), device=features.device)

        # 将点云特征投影到体素网格中
        voxel_features.scatter_add_(1, vox_indices.unsqueeze(-1).expand(-1, -1, feature_dim), features)
        voxel_counts.scatter_add_(1, vox_indices.unsqueeze(-1), torch.ones_like(voxel_counts))

        # 避免除以 0
        voxel_counts = torch.clamp(voxel_counts, min=1.0)
        voxel_features /= voxel_counts

        # 重新调整成 (batch_size, resolution, resolution, resolution, feature_dim)
        voxel_features = voxel_features.view(batch_size, resolution, resolution, resolution, feature_dim)
        
        voxel_features = voxel_features.permute(0, 4, 1, 2, 3)
        
        return voxel_features

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
