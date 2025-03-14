import torch
import clip

class ClipTextEncoder:
    """
    Encodes text prompts into feature vectors using OpenAI CLIP.
    Supports Classifier-Free Guidance (CFG) by randomly dropping text prompts.
    """
    def __init__(self, model_name="ViT-B/32", dropout_prob=0.3):
        """
        :param model_name: CLIP 模型名称，可选: "ViT-B/32", "ViT-B/16", "ViT-L/14" 等
        :param dropout_prob: 以概率 `dropout_prob` 将文本替换为空字符串，默认 0.3
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dropout_prob = dropout_prob  # 控制文本丢弃概率
        self.output_dim = 512  # CLIP 默认输出维度
        
        # 加载 CLIP 模型
        self.model, self.preprocess = clip.load(model_name, device=device)
        
    @torch.no_grad()
    def encode_text(self, text_list, force_drop_ids=None):
        """
        将文本列表编码为 (batch, hidden_size) 的张量，并支持 CFG 文本随机丢弃
        :param text_list: List[str], 包含多条文本描述
        :param force_drop_ids: (可选) 一个 bool Tensor, 指定哪些文本需要丢弃
        :return: torch.FloatTensor(shape: (batch, hidden_dim))
        """
        batch_size = len(text_list)

        if force_drop_ids is not None:
            drop_mask = force_drop_ids.bool().to(self.device)
        else:
            drop_mask = torch.rand(batch_size, device=self.device) < self.dropout_prob

        modified_texts = ["" if drop_mask[i] else text_list[i] for i in range(batch_size)]

        tokens = clip.tokenize(modified_texts, truncate=True).to(self.device)
        text_features = self.model.encode_text(tokens)  # (B, 512)

        zero_vector = torch.zeros_like(text_features)
        text_features = torch.where(drop_mask[:, None], zero_vector, text_features)

        return text_features  # (batch, 512)

# texts = ["A chair with a curved backrest, a floating padded seat, and integrated armrests. Its structure relies on a cantilevered support system for ergonomic flexibility, making it suitable for contemporary interior settings.",
# "A modern cantilever chair with a continuous metal frame, a floating padded seat, a slightly curved backrest, and integrated armrests. Its structure relies on a cantilevered support system for ergonomic flexibility, making it suitable for contemporary interior settings.",
# "A wooden chair with a green seat, a backrest that curves slightly, and a metal frame. The chair has a simple and functional design, making it suitable for various settings.",
# "A wooden chair with a curved backrest, armrests, and legs. The chair features a simple yet elegant design, with a focus on comfort and functionality.",
# "A metal chair with a curved backrest, a metal armrest, and a metal frame. The chair has a unique design and is likely to be used in an office or home setting.",
# "A wooden chair with a metal frame, a padded seat, and a curved backrest. The chair features a simple and functional design, with a focus on comfort and durability.",
# "A modern cantilever chair with a continuous metal frame, a floating padded seat, a slightly curved backrest, and integrated armrests. Its structure relies on a cantilevered support system for ergonomic flexibility, making it suitable for contemporary interior settings.",
# "A chair with a wooden frame, a padded seat, and a backrest that curves slightly upwards. The chair has a simple and elegant design, making it suitable for various settings.",
# "A modern cantilever chair with a continuous metal frame, a floating padded seat, a slightly curved backrest, and integrated armrests. Its structure relies on a cantilevered support system for ergonomic flexibility, making it suitable for contemporary interior settings.",
# "A wooden chair with a brown seat and backrest, featuring a curved design and a simple, functional structure. The chair is supported by four legs, providing stability and comfort.",
# ]
# encoder = ClipTextEncoder(dropout_prob=0.3)
# features = encoder.encode_text(text_list=texts)
# print("嵌入向量形状:", features.shape) 
# for i in range(len(texts)):
#     print(f"文本 {i+1} 的嵌入向量:\n{features[i]}\n")
