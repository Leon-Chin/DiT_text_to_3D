import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import json
import open3d as o3d
import numpy as np
import torch.nn.functional as F
from pathlib import Path

synset_to_label = {
    '02691156': 'airplane', 
    '03001627': 'chair',
}

cate_to_synsetid = {v: k for k, v in synset_to_label.items()}

DATASET_PATH = "datasets/shapenet_data"
RESULTS_JSON_PATH = "datasets/results.json"

with open(RESULTS_JSON_PATH, "r") as f:
    text_annotations = json.load(f)  # 加载 JSON 文件

# for sysnet_id, models in text_annotations.items():
#     num_models = len(models)
#     print(f"Category {sysnet_id} has {num_models} models.")

class PointCloudMasks:
    """ 生成 3D 点云的 Mask，模拟不同视角的遮挡 """
    def __init__(self, radius=10, elev_range=(30, 60), azim_range=(0, 360), randomize=True):
        self.radius = radius
        self.elev_range = elev_range
        self.azim_range = azim_range
        self.randomize = randomize  # 是否随机选择视角

    def __call__(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 选取随机观察角度
        elev = np.random.uniform(*self.elev_range) if self.randomize else np.mean(self.elev_range)
        azim = np.random.uniform(*self.azim_range) if self.randomize else np.mean(self.azim_range)

        # 计算摄像机位置
        camera = [self.radius * np.sin(np.radians(90 - elev)) * np.cos(np.radians(azim)),
                  self.radius * np.cos(np.radians(90 - elev)),
                  self.radius * np.sin(np.radians(90 - elev)) * np.sin(np.radians(azim))]

        _, pt_map = pcd.hidden_point_removal(camera, self.radius)

        # 创建 Mask，遮挡不可见点
        mask = np.zeros(points.shape[0], dtype=np.float32)
        mask[pt_map] = 1
        return mask


class ShapeNet15kPointClouds(Dataset):
    def __init__(self, root_dir, categories=['chair'], train_data_rate=0.7,
                 split='train', normalize=True, random_subsample=False, augment=False, use_mask=False):
        self.root_dir = root_dir
        self.split = split
        self.random_subsample = random_subsample
        self.augment = augment  # 是否使用数据增强
        self.use_mask = use_mask  # 是否使用遮挡



        # 获取 synset ID
        self.synset_ids = [cate_to_synsetid[c] for c in categories]
        self.mask_transform = PointCloudMasks() if use_mask else None

        categorys_size = {}
        for i, category_id in enumerate(self.synset_ids):
            category_path = os.path.join(self.root_dir, category_id)
            category_number = len([f for f in os.listdir(category_path) if f.endswith('.npy')])
            categorys_size[category_id] = category_number

        self.all_points = []
        self.cate_idx_lst = []
        self.all_texts = []
        self.all_cate_mids = []

        for cate_idx, synset_id in enumerate(self.synset_ids):
            cate_path = os.path.join(root_dir, synset_id)
            if not os.path.isdir(cate_path):
                print(f"Skipping {synset_id}, directory not found: {cate_path}")
                continue

            for file_name in os.listdir(cate_path):
                if not file_name.endswith(".npy"):
                    continue

                model_id = file_name[:-4]
                file_path = os.path.join(cate_path, file_name)

                # 读取点云数据
                point_cloud = np.load(file_path)

                # 归一化
                if normalize:
                    mean = point_cloud.mean(axis=0)
                    std = point_cloud.std(axis=0)
                    point_cloud = (point_cloud - mean) / std

                # 数据增强
                if self.augment:
                    point_cloud = self.apply_augmentations(point_cloud)

                # 训练集 & 测试集
                training_sample_size = int(categorys_size[synset_id] * train_data_rate)
                test_sample_size = categorys_size[synset_id] - training_sample_size

                train_points = point_cloud[:training_sample_size]
                test_points = point_cloud[training_sample_size:training_sample_size + test_sample_size]

                print(train_points.shape, test_points.shape)

                # 获取文本描述
                text_description = text_annotations.get(synset_id, {}).get(model_id, "No description available.")

                # 存入列表
                self.all_points.append((train_points, test_points))
                self.cate_idx_lst.append(cate_idx)
                self.all_texts.append(text_description)
                self.all_cate_mids.append((synset_id, model_id))

        # 随机打乱
        self.shuffle_idx = list(range(len(self.all_points)))
        random.shuffle(self.shuffle_idx)
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_texts = [self.all_texts[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        print(f"Loaded {len(self.all_points)} samples from {categories}")

    def __len__(self):
        return len(self.all_points)

    def apply_augmentations(self, points):
        """ 应用数据增强 """
        # 随机旋转
        angle = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        points = np.dot(points, rot_matrix)

        # 随机缩放
        scale = np.random.uniform(0.8, 1.2)
        points *= scale

        # 添加高斯噪声
        noise = np.random.normal(0, 0.02, size=points.shape)
        points += noise

        return points

    def __getitem__(self, idx):
        train_points, test_points = self.all_points[idx]
        cate_idx = self.cate_idx_lst[idx]
        text = self.all_texts[idx]
        sid, mid = self.all_cate_mids[idx]

        # 生成 Mask
        train_mask = self.mask_transform(train_points) if self.use_mask else np.ones(train_points.shape[0])
        test_mask = self.mask_transform(test_points) if self.use_mask else np.ones(test_points.shape[0])

        return {
            'train_points': torch.from_numpy(train_points).float(),
            'test_points': torch.from_numpy(test_points).float(),
            'train_mask': torch.from_numpy(train_mask).float(),
            'test_mask': torch.from_numpy(test_mask).float(),
            'cate_idx': cate_idx,
            'text': text,
            'sid': sid,
            'mid': mid
        }


# 运行数据处理并保存
if __name__ == "__main__":
    dataset = ShapeNet15kPointClouds(DATASET_PATH, categories=['chair', 'airplane'])

    torch.save(dataset, "processed_shapenet.pt")
    print("✅ 数据处理完成，已保存到 processed_shapenet.pt")