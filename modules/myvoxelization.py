import torch
import torch.nn as nn

class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=1e-6):
        """
        :param resolution: 体素网格的分辨率，例如 32 表示生成一个 32x32x32 的体素网格。
        :param normalize: 是否对点坐标归一化到 [0, 1] 区间。
        :param eps: 防止除 0 的小常数。
        """
        super().__init__()
        self.resolution = resolution
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        """
        :param features: 点云特征，形状为 [B, N, C]，其中 B 为批次大小，N 为点数，C 为特征维度。
        :param coords: 点云坐标，形状为 [B, N, 3]，假设坐标原始范围大致在 [-1, 1]。
        :return:
            avg_voxel_features: 平均体素特征，形状为 [B, C, R, R, R]
            normalized_coords: 归一化后的点坐标，范围在 [0, 1]（用于后续反体素化）。
        """
        # 1. 中心化坐标：减去每个点云的均值
        mean_coords = coords.mean(dim=1, keepdim=True)  # [B, 1, 3]
        centered_coords = coords - mean_coords  # [B, N, 3]

        # 2. 如果需要，则归一化（这里将点云扩展到大约 [0, 1]）
        if self.normalize:
            # 计算每个点云所有点的最大范数，乘以 2 后再归一化，再加上 0.5使得坐标大致落在 [0, 1]
            max_norm = centered_coords.norm(dim=2, keepdim=True).max(dim=1, keepdim=True).values  # [B, 1, 1]
            normalized_coords = centered_coords / (max_norm * 2 + self.eps) + 0.5
        else:
            normalized_coords = (centered_coords + 1) / 2

        # 3. 将归一化坐标映射到体素网格上
        scaled_coords = normalized_coords * (self.resolution - 1)
        # 四舍五入得到每个点所属的体素索引（整数坐标）
        voxel_indices = torch.round(scaled_coords).long()  # [B, N, 3]

        B, N, C = features.shape
        # 将三维体素坐标转化为一个唯一的一维索引： index = x * (R^2) + y * R + z
        indices = voxel_indices[..., 0] * (self.resolution ** 2) + voxel_indices[..., 1] * self.resolution + voxel_indices[..., 2]  # [B, N]

        # 4. 将点云特征投影到体素网格中
        # 创建用于存储累加的体素特征和计数的张量
        voxel_features = torch.zeros(B, self.resolution**3, C, device=features.device)  # [B, R^3, C]
        voxel_counts = torch.zeros(B, self.resolution**3, 1, device=features.device)      # [B, R^3, 1]

        # 使用 scatter_add 将每个点的特征累加到对应体素位置上
        voxel_features = voxel_features.scatter_add(1, indices.unsqueeze(-1).expand(-1, -1, C), features)
        # 统计每个体素内有多少个点
        voxel_counts = voxel_counts.scatter_add(1, indices.unsqueeze(-1), torch.ones(B, N, 1, device=features.device))

        # 防止除 0
        voxel_counts = torch.clamp(voxel_counts, min=1.0)
        avg_voxel_features = voxel_features / voxel_counts  # [B, R^3, C]

        # 5. 将平均特征重新 reshape 成 3D 体素网格形式： [B, C, R, R, R]
        avg_voxel_features = avg_voxel_features.transpose(1, 2).view(B, C, self.resolution, self.resolution, self.resolution)

        return avg_voxel_features, normalized_coords

    def extra_repr(self):
        return f'resolution={self.resolution}, normalize={self.normalize}, eps={self.eps}'

# 示例使用：
if __name__ == "__main__":
    # 假设有 2 个点云，每个点云有 10000 个点，每个点 3 个坐标，3 维特征
    B, N, C = 2, 10000, 3
    features = torch.randn(B, N, C)
    # 假设原始坐标在 [-1, 1] 范围内
    coords = torch.rand(B, N, 3) * 2 - 1

    voxelizer = Voxelization(resolution=32, normalize=True)
    voxel_feats, norm_coords = voxelizer(features, coords)
    print("Voxelized Features shape:", voxel_feats.shape)  # 预期：[B, C, 32, 32, 32]
    print("Normalized Coordinates shape:", norm_coords.shape)  # 预期：[B, N, 3]
