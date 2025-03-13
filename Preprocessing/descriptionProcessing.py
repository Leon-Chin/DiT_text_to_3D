import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

synset_to_label = {
    '02691156': 'airplane', 
    '03001627': 'chair',
}

def main():
    # 1. 设置设备：如果支持 mps 则使用 mps，否则使用 CPU
    device = "mps"
    print(f"Device: {device}")

    # 2. 加载 LLaVA 7B 模型及处理器（trust_remote_code=True）
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        trust_remote_code=True
    )
    print("LLaVA 7B 模型加载成功！")

    # 准备一个字典存储结果 {filename: caption}
    results = {}

    # 3. 批量遍历 data/ 文件夹下所有 .npy 文件
    valid_filenames = {"0.png", "2.png", "6.png", "8.png"}
    data_folder = "/Users/qinleiheng/Documents/秦磊恒/IP Paris/Master 1/Computer Vision/Project/Project Code/DiT-text-to-3D/render_images"  # 你的数据文件夹
    for model_type in os.listdir(data_folder):
        print(model_type)
        type_path = os.path.join(data_folder, model_type)
        if not os.path.isdir(type_path):
            continue  # 跳过非目录（比如 .DS_Store）
        for obj in os.listdir(os.path.join(data_folder, model_type)):
            obj_path = os.path.join(data_folder, model_type, obj)
            if obj_path == "/Users/qinleiheng/Documents/秦磊恒/IP Paris/Master 1/Computer Vision/Project/Project Code/DiT-text-to-3D/render_images/02691156/1a6ad7a24bb89733f412783097373bdc":
                print(os.listdir(os.path.join(data_folder, model_type, obj)))
                images = []
                for image_path in os.listdir(os.path.join(data_folder, model_type, obj)):
                    print("image_path")
                    if image_path.endswith(".png") and image_path in valid_filenames:
                        image = Image.open(os.path.join(data_folder, model_type, obj, image_path))

                        images.append(image)
                # limit images length
                images = images[:3]
                # 构造文本提示
                image_placeholder = "<image>\n"
                if synset_to_label[model_type] == "plane":
                    item_examples = "fighter jet, bomber, drone" 
                    item_features = "wing shape, tail design, engine placement"
                else: 
                    item_examples = "office chair, lounge chair, dining chair" 
                    item_features = "legs, armrests, backrest shape"

                prompt = (
                    f"USER: {image_placeholder * 3}\n"
                    "Analyze this image and answer concisely:"
                    f"1. What type of {synset_to_label[model_type]} is this? (e.g., {item_examples})"
                    f"2. What are its key silhouette features? (e.g., {item_features})"
                    "3. Provide a one-sentence description of its 3D shape for text-to-3D generation."
                    "Format your response like this:"
                    "A modern twin-engine fighter jet with delta wings, twin vertical stabilizers, and front air intakes."
                    "A modern office chair with a mesh backrest, five-wheel base, and adjustable armrests."
                    "ASSISTANT:"
                    )
                # 利用处理器构造输入
                inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)

                # 生成文本描述
                output_ids = model.generate(**inputs, max_new_tokens=100)
                output_text = processor.decode(output_ids[0], skip_special_tokens=True)

                # 打印或存储结果
                print(f"{obj} -> {output_text}")
                results[obj] = output_text
            # torch.cuda.empty_cache()
            # torch.cuda.ipc_collect()

    # 4. 将结果写入 JSON 文件
    output_json = "results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n所有结果已保存到 {output_json} 中。")

if __name__ == "__main__":
    main()


