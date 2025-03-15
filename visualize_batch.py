import numpy as np
import open3d as o3d

# 读取 .npy 文件
file_path = "output/generated_samples.npy"
point_clouds = np.load(file_path)  # 加载点云数据 (N, 3)
print(point_clouds.shape)
for pc in point_clouds:
    pc = pc.transpose(1, 0)
    print(pc.shape)
    pcd = o3d.geometry.PointCloud()
    # 创建 Open3D 点云对象
    pcd.points = o3d.utility.Vector3dVector(pc)
    # 可视化
    o3d.visualization.draw_geometries([pcd])
