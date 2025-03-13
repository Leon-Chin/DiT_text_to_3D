import numpy as np
import open3d as o3d

# 读取 .npy 文件
file_path = "/Users/qinleiheng/Documents/秦磊恒/IP Paris/Master 1/Computer Vision/Project/Project Code/DiT-text-to-3D/data_1024/02691156/1a54a2319e87bd4071d03b466c72ce41.npy"  # 替换为你的 .npy 文件路径
point_cloud = np.load(file_path)  # 加载点云数据 (N, 3)

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# 可视化
o3d.visualization.draw_geometries([pcd])
