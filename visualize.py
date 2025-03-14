import numpy as np
import open3d as o3d

# 读取 .npy 文件
file_path = "/Users/qinleiheng/Documents/秦磊恒/IP Paris/Master 1/Computer Vision/Project/Project Code/DiT-text-to-3D/shapenet_versions/shapenet_data_2048/02691156/1a04e3eab45ca15dd86060f189eb133.npy"
point_cloud = np.load(file_path)  # 加载点云数据 (N, 3)

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# 可视化
o3d.visualization.draw_geometries([pcd])
