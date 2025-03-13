import torch
from torch.utils.data import DataLoader
import json

from datasets.data_preprocessing import ShapeNet15kPointClouds

DATASET_PATH = "datasets/shapenet_data_5000_splitted"
DATA_POINTS_SIZE = 5000
RESULTS_JSON_PATH = "datasets/results.json"
if __name__ == "__main__":
    with open(RESULTS_JSON_PATH, "r") as f:
            text_annotations = json.load(f)  # 加载 JSON 文件

    for sysnet_id, models in text_annotations.items():
        num_models = len(models)
        print(f"Category {sysnet_id} has {num_models} models.")
    dataset = ShapeNet15kPointClouds(categories=['chair', 'airplane'], text_annotations=text_annotations)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 迭代 DataLoader
    for batch in dataloader:
        print(f"Batch idx: {batch['idx']}")
        print(f"训练点云形状: {batch['train_points'].shape}")  # (batch_size, 3500, 3)
        print(f"测试点云形状: {batch['test_points'].shape}")   # (batch_size, 1500, 3)
        print(f"类别索引: {batch['cate_idx']}")               # 类别编号
        print(f"mid: {batch['mid']}")                         # 模型编号
        print(f"文本描述: {batch['text']}")                   # 该模型的文本描述
        break  # 只打印第一个 batch

['A large, four-engine commercial airliner with a wide-body fuselage, two vertical stabilizers, and four rear-mounted engines. The wings are swept back, and the aircraft features a large, curved tail fin. The fuselage is elongated, with multiple rows of windows, and the aircraft is designed for long-haul flights, carrying hundreds of passengers.', 
 'A modern cantilever chair with a continuous metal frame, a floating padded seat, a slightly curved backrest, and integrated armrests. Its structure relies on a cantilevered support system for ergonomic flexibility, making it suitable for contemporary interior settings.', 
 "A red velvet chair with a curved backrest, a padded seat, and a metal frame. The chair features a unique design with a combination of curves and angles, providing both comfort and visual interest. The red color adds a bold and luxurious touch to the chair's overall appearance.", 
 'A large commercial airplane with two engines, a delta wing, and a streamlined fuselage. The aircraft features a smooth exterior and a delta wing shape, with twin outward-angled vertical stabilizers and rear-mounted exhaust nozzles. The wings are sharply angled with integrated control surfaces, and the aircraft is designed for high-speed maneuverability and aerodynamic efficiency.', 
 'A wooden chair with a curved backrest, a padded seat, and a metal frame. The chair features a continuous armrest and a slightly curved structure, providing comfort and support for the user.',
   'A green and black airplane with a delta wing design, two engines, and a streamlined fuselage. The aircraft features a smooth exterior and a tail with two vertical stabilizers. The wings are sharply angled with integrated control surfaces, and the aircraft is designed for high-speed maneuverability and aerodynamic efficiency.', 
 'A modern cantilever chair with a continuous metal frame, a floating padded seat, a slightly curved backrest, and integrated armrests. Its structure relies on a cantilevered support system for ergonomic flexibility, making it suitable for contemporary interior settings.',
   'A black and white image of a chair.']