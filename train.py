import torch
from torch.utils.data import DataLoader
import json

dataset = torch.load("processed_shapenet.pt")

print(f"数据集中样本总数: {len(dataset)}")  # 打印数据集的大小

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# 迭代 DataLoader
for batch in dataloader:
    print(f"Batch idx: {batch['idx']}")
    print(f"训练点云形状: {batch['train_points'].shape}")  # (batch_size, 3500, 3)
    print(f"测试点云形状: {batch['test_points'].shape}")   # (batch_size, 1500, 3)
    print(f"类别索引: {batch['cate_idx']}")               # 类别编号
    print(f"文本描述: {batch['text']}")                   # 该模型的文本描述
    break  # 只打印第一个 batch
