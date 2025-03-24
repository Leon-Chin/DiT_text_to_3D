import numpy as np
import open3d as o3d

# file_path = "output/generated_samples_epoch0.npy"
file_path = "output/generated_samples_temp.npy"
point_clouds = np.load(file_path)  # (N, 3)
print(point_clouds.shape)
index = 0
for pc in point_clouds:
    if index == 0:
        pc = pc.transpose(1, 0)
        print(pc.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([pcd])
