import os
import numpy as np
import open3d as o3d
from PIL import Image

def render_pointcloud_to_image(npy_path, output_path):
    """
    读取 .npy 点云文件，使用 Open3D 渲染为 2D 图像，并保存为 output_path 指定的文件。
    """
    # 读取点云数据
    point_cloud = np.load(npy_path)

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 创建不可见窗口并渲染
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # 获取屏幕截图 (float array in [0,1])
    img_buffer = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    # 转换为 PIL Image
    img_np = (np.array(img_buffer) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    pil_img.save(output_path)
    print(f"保存图片至 {output_path}")

if __name__ == "__main__":
    # 示例：对 data 文件夹下的所有 .npy 文件进行渲染
    data_folder = "data"
    output_folder = "rendered_images"
    os.makedirs(output_folder, exist_ok=True)
    for type_id in os.listdir(data_folder):
        type_path = os.path.join(data_folder, type_id)
        for npy_file in os.listdir(type_path):
            npy_path = os.path.join(type_path, npy_file)
            # 构造输出文件名，例如将 .npy 替换成 .png
            output_path = os.path.join(output_folder, npy_file.replace(".npy", ".png"))
            render_pointcloud_to_image(npy_path, output_path)
