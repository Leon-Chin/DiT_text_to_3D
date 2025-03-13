import os
import shutil
import random

# 你的数据集路径
root_dir = "shapenet_data_5000"
split_ratio = {'train': 0.7, 'val': 0.15, 'test': 0.15}  # 70% 训练，15% 验证，15% 测试

for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)
    if not os.path.isdir(category_path):
        continue

    npy_files = [f for f in os.listdir(category_path) if f.endswith('.npy')]
    random.shuffle(npy_files)

    train_split = int(len(npy_files) * split_ratio['train'])
    val_split = int(len(npy_files) * (split_ratio['train'] + split_ratio['val']))

    for split, files in zip(['train', 'val', 'test'],
                             [npy_files[:train_split], npy_files[train_split:val_split], npy_files[val_split:]]):
        split_path = os.path.join(category_path, split)
        os.makedirs(split_path, exist_ok=True)
        for file in files:
            shutil.move(os.path.join(category_path, file), os.path.join(split_path, file))

print("✅ 数据集划分完成！")

import os

# 你的数据集路径
root_dir = "shapenet_data_5000"

# 统计结果存储
split_counts = {}

for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)
    if not os.path.isdir(category_path):
        continue

    split_counts[category] = {}

    for split in ["train", "val", "test"]:
        split_path = os.path.join(category_path, split)
        if os.path.exists(split_path):
            num_files = len([f for f in os.listdir(split_path) if f.endswith(".npy")])
            split_counts[category][split] = num_files
        else:
            split_counts[category][split] = 0

# 打印统计结果
print("\n📊 数据集划分统计结果：")
for category, splits in split_counts.items():
    print(f"\n类别: {category}")
    for split, count in splits.items():
        print(f"  {split}: {count} 个 .npy 文件")
