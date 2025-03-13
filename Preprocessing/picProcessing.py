#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil

synset_to_label = {
    '02691156': 'airplane', 
    '03001627': 'chair',
}

handled_datasets = [k for k in synset_to_label.keys()]

def main():
    """
    假设:
      - 原始图像所在目录为 RAW_DIR = "rendering"
      - 每个视角的文件名为 <view_index>.png
      - 需要整理为 render_images/<synset_id>/<model_id>/<view_index>.png
      - RAW_PATHS 中存放了若干个形如 "rendering/<synsetId>_<modelId>/<viewIndex>" 的相对路径
    """

    # 原始图像所在的根目录
    RAW_DIR = "rendering"

    # 整理后输出的根目录
    OUT_DIR = "render_images"

    # 依次处理列表中的每一条路径
    for path_str in os.listdir(RAW_DIR):
        # 1) 去掉开头的 "rendering/" (如果存在)
        parts = path_str.split("_")
        synset_id, model_id = parts[0], parts[1]
        if synset_id in handled_datasets:
            for view_index in os.listdir(os.path.join(RAW_DIR, path_str)):
                # 2) 获取视角索引
                view_index = view_index.split(".")[0]
                src_path = os.path.join(RAW_DIR, f"{synset_id}_{model_id}", view_index + ".png")
                dst_path = os.path.join(OUT_DIR, synset_id, model_id, view_index + ".png")

                # 6) 创建目标目录 & 复制
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                if not os.path.exists(src_path):
                    print(f"[警告] 源文件不存在: {src_path}")
                    continue

                shutil.copy2(src_path, dst_path)
                print(f"[复制完成] {src_path} -> {dst_path}")

if __name__ == "__main__":
    main()
