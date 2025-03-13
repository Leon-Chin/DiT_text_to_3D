import os
import shutil

DIR_RENDER = "render_images"
DIR_DATA = "data_5000"

def remove_dir(path):
    """安全地删除一个文件夹（包含其中所有内容）。"""
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"[删除文件夹] {path}")

def remove_file(path):
    """安全地删除一个文件。"""
    if os.path.isfile(path):
        os.remove(path)
        print(f"[删除文件] {path}")

def main():
    for type in os.listdir(DIR_RENDER):
        if os.path.isdir(os.path.join(DIR_RENDER, type)):
            rendered_models = set(os.listdir(os.path.join(DIR_RENDER, type)))
            models = set(filename.split(".")[0] for filename in os.listdir(os.path.join(DIR_DATA, type)))
            without_image = models - rendered_models

            print(without_image)
            # 只保留其中是文件夹的
            # rendered_models = {t for t in rendered_models if os.path.exists(os.path.join(DIR_RENDER, t+".npy"))}
            for model in without_image:
                remove_file(os.path.join(DIR_DATA, type, model+".npy"))



if __name__ == "__main__":
    main()