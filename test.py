window_block_indexes=(0,3,6,9)

def test():
    for i in range(4):
        if i in window_block_indexes:
            print('window')
        else:
            print('not window')

test()

# if __name__ == "__main__":
#     with open(RESULTS_JSON_PATH, "r") as f:
#             text_annotations = json.load(f)  # 加载 JSON 文件

#     for sysnet_id, models in text_annotations.items():
#         num_models = len(models)
#         print(f"Category {sysnet_id} has {num_models} models.")
#     # dataset = ShapeNet15kPointClouds(categories=['chair', 'airplane'], text_annotations=text_annotations)
#     dataset = ShapeNet15kPointClouds(categories=CATEGORY, split="val", text_annotations=text_annotations)

#     dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
#     # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

#     # 迭代 DataLoader
#     for batch in dataloader:
#         print(f"Batch idx: {batch['idx']}")
#         print(f"训练点云形状: {batch['train_points'].shape}")  # (batch_size, 3500, 3)
#         print(f"测试点云形状: {batch['test_points'].shape}")   # (batch_size, 1500, 3)
#         print(f"类别索引: {batch['cate_idx']}")               # 类别编号
#         print(f"mid: {batch['mid']}")                         # 模型编号
#         print(f"文本描述: {batch['text']}")                   # 该模型的文本描述
#         break  # 只打印第一个 batch