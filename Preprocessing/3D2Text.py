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
    device = "cuda"
    print(f"Device: {device}")

    # 2. 加载 LLaVA 7B 模型及处理器
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
    torch.cuda.empty_cache()

    # 3. 读取已有的 JSON 文件（如果存在）
    output_json = "/workspace/results.json"
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                results = json.load(f)
        except ():
            print("⚠️ 无法解析 JSON 文件，重新初始化 results")
            # print()
    else:
        results = {}

    # 4. 遍历数据文件夹
    data_folder = "/workspace/assets/render_images"
    valid_filenames = {"0.png", "2.png", "6.png", "8.png"}

    for model_type in os.listdir(data_folder):
        type_path = os.path.join(data_folder, model_type)
        if not os.path.isdir(type_path):
            continue  # 跳过非目录

        # 确保 model_type 在 results 中有初始化
        if model_type not in results:
            results[model_type] = {}

        for obj in os.listdir(type_path):
            obj_path = os.path.join(type_path, obj)
            if not os.path.isdir(obj_path):
                continue  # 跳过非目录
            # **跳过已处理的对象**
            if obj in results[model_type]:
                print(f"Skipping {obj}, already processed.")
                continue  

            images = []
            for image_path in sorted(os.listdir(obj_path)):  # 确保顺序一致
                if image_path in valid_filenames and image_path.endswith(".png"):
                    image = Image.open(os.path.join(obj_path, image_path))
                    images.append(image)

            # 限制最大 3 张图片
            images = [img.resize((128, 128)) for img in images]  # 降低分辨率

            # 5. 构造 LLaVA 生成文本的 prompt
            image_placeholder = "<image>\n"
            if synset_to_label.get(model_type) == "airplane":
                item_examples = "fighter jet, bomber, drone"
                item_features = "wing shape, tail design, engine placement"
            else: 
                item_examples = "office chair, lounge chair, dining chair"
                item_features = "legs, armrests, backrest shape"

            prompt = (
                f"USER: {image_placeholder * len(images)}\n"
                "Analyze this image and answer concisely:"
                f"1. What type of {synset_to_label[model_type]} is this? (e.g., {item_examples})"
                f"2. What are its key silhouette features? (e.g., {item_features})"
                "3. Provide a detailed description of its shape, structure, and components without mentioning color or texture."
                "Format your response like this:"
                "A twin-engine, delta-wing fighter jet with a streamlined fuselage, twin outward-angled vertical stabilizers, and rear-mounted exhaust nozzles. The wings are sharply angled with integrated control surfaces, and the aircraft features a smooth, stealth-optimized exterior. Designed for high-speed maneuverability and aerodynamic efficiency."
                "A modern cantilever chair with a continuous metal frame, a floating padded seat, a slightly curved backrest, and integrated armrests. Its structure relies on a cantilevered support system for ergonomic flexibility, making it suitable for contemporary interior settings."
                "A three-seater sofa with a boxy structure, wide padded armrests, three large back cushions, and thick seat cushions. Supported by four short sturdy legs, its symmetrical and structured design provides balanced comfort and support, ideal for a modern or classic living room."
                "ASSISTANT:"
            )

            # 6. 处理输入
            inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)

            # 生成文本描述
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=100)
                output_text = processor.decode(output_ids[0], skip_special_tokens=True)

            # 只保留模型输出的文本
            if "ASSISTANT:" in output_text:
                output_text = output_text.split("ASSISTANT:")[-1].strip()

            # ✅ 存储结果
            print(f"{obj} -> {output_text}")
            results[model_type][obj] = output_text

            # ✅ **每处理完一个对象，就立即写入 JSON 文件**
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"Saved {obj} to {output_json}")

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    print(f"\n所有结果已保存到 {output_json} 中。")

if __name__ == "__main__":
    main()
