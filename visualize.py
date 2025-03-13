import numpy as np
import open3d as o3d

# 读取 .npy 文件
file_path = "/Users/qinleiheng/Documents/秦磊恒/IP Paris/Master 1/Computer Vision/Project/Project Code/DiT-text-to-3D/data_5000/02691156/1b626fd06226b600adcbeb54f3d014e9.npy"
point_cloud = np.load(file_path)  # 加载点云数据 (N, 3)

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# 可视化
o3d.visualization.draw_geometries([pcd])
