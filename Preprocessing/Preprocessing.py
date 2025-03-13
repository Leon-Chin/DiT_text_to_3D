import os
import open3d as o3d
import numpy as np

# 原始 ShapeNet 数据路径
DATASET_DIR = "/Users/qinleiheng/Documents/秦磊恒/IP Paris/Master 1/Computer Vision/Project/Project Code/DiT-text-to-3D/shapenet"
# 目标存储路径
N_POINTS = 5000
OUTPUT_DIR = f"data_{N_POINTS}"
# 设定每个点云的点数

synset_to_label = {
    '02691156': 'airplane', 
    '03001627': 'chair',
}


def process_obj_to_pcd(obj_path, output_path):
    """
    读取 .obj 文件，采样点云，归一化并存储 .npy 文件
    """
    # 读取 3D 网格模型
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if not mesh.has_vertices():
        print(f"❌ 跳过 {obj_path}，没有顶点数据")
        return

    # 进行点云采样
    pcd = mesh.sample_points_uniformly(number_of_points=N_POINTS)
    point_cloud = np.asarray(pcd.points)

    # 归一化处理：将点云居中，并缩放到 [-1, 1]
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    max_distance = np.max(np.linalg.norm(point_cloud, axis=1))
    point_cloud /= max_distance

    # 保存点云为 .npy
    np.save(output_path, point_cloud)
    print(f"✅ 已处理 {output_path}")

def preprocess_shapenet(dataset_dir, output_dir):
    """
    遍历 ShapeNet 目录，找到所有 .obj 文件，并转换为点云数据
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for synset_id in os.listdir(dataset_dir):#类别
        synset_path = os.path.join(dataset_dir, synset_id)
        if not os.path.isdir(synset_path):
            continue
        
        output_synset_dir = os.path.join(output_dir, synset_id)
        if not os.path.exists(output_synset_dir):
            os.makedirs(output_synset_dir)

        # 遍历所有模型
        for model_id in os.listdir(synset_path):
            model_path = os.path.join(synset_path, model_id, "models", "model_normalized.obj")
            if not os.path.exists(model_path):
                continue

            output_pcd_path = os.path.join(output_synset_dir, f"{model_id}.npy")
            process_obj_to_pcd(model_path, output_pcd_path)

# 运行数据预处理
preprocess_shapenet(DATASET_DIR, OUTPUT_DIR)
