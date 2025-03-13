import os
import numpy as np
import open3d as o3d
from PIL import Image

def get_camera_extrinsic(eye, center, up):
    """
    根据 eye、center、up 计算相机外参矩阵。
    外参矩阵将世界坐标转换到相机坐标，公式为：
      z = (eye - center) / ||eye - center||
      x = cross(up, z) / ||cross(up, z)||
      y = cross(z, x)
      R = [x; y; z]  (作为 3x3 矩阵，行排列)
      t = -R * eye
      extrinsic = [ R   t ]
                   [0 0 0 1]
    """
    eye = np.array(eye)
    center = np.array(center)
    up = np.array(up)
    
    z = eye - center
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ eye
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic

def render_3D_to_image(obj_path, output_path, view):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if mesh.is_empty():
        raise ValueError("模型加载失败，请检查文件路径或模型文件是否正确。")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    # 修改模型颜色以获得更好的对比效果
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.2, 0.2, 0.8])
    
    # 创建 Visualizer（不可见窗口）
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=640, height=480)
    
    # 设置背景颜色为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    
    # 获取模型中心和尺寸
    center = mesh.get_center()
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    d = np.linalg.norm(extent) * 1.5  # 摄像机距离
    
    # 根据视角计算摄像机位置（eye）和 up 向量
    if view == "top":
        eye = [center[0], center[1], center[2] + d]
        up = [0, 1, 0]
    elif view == "bottom":
        eye = [center[0], center[1], center[2] - d]
        up = [0, -1, 0]
    elif view == "front":
        eye = [center[0], center[1] - d, center[2]]
        up = [0, 0, 1]
    elif view == "back":
        eye = [center[0], center[1] + d, center[2]]
        up = [0, 0, 1]
    elif view == "left":
        eye = [center[0] - d, center[1], center[2]]
        up = [0, 0, 1]
    elif view == "right":
        eye = [center[0] + d, center[1], center[2]]
        up = [0, 0, 1]
    else:
        raise ValueError("Unknown view type: {}".format(view))
    
    # 修改 Visualizer 中的相机参数
    view_ctl = vis.get_view_control()
    cam_param = view_ctl.convert_to_pinhole_camera_parameters()
    cam_param.extrinsic = get_camera_extrinsic(eye, center, up)
    view_ctl.convert_from_pinhole_camera_parameters(cam_param)
    
    # 刷新窗口
    vis.poll_events()
    vis.update_renderer()
    
    # 截图（返回 float 数组，取值 [0,1]）
    img_buffer = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    
    # 转换为 PIL Image 并保存
    img_np = (np.array(img_buffer) * 255).astype(np.uint8)
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    pil_img = Image.fromarray(img_np)
    pil_img.save(output_path)
    print(f"保存图片至 {output_path}")

def render_all_views_for_obj(obj_path, output_dir):
    """
    对单个 .obj 模型生成六个固定视角的图片，保存至 output_dir 下，
    文件名为 0.png ~ 5.png，对应：
      0: 俯视图 (top)
      1: 仰视图 (bottom)
      2: 正视图 (front)
      3: 左视图 (left)
      4: 右视图 (right)
      5: 后视图 (back)
    """
    views = ["top", "bottom", "front", "left", "right", "back"]
    os.makedirs(output_dir, exist_ok=True)
    for i, view in enumerate(views):
        output_path = os.path.join(output_dir, f"{i}.png")
        render_3D_to_image(obj_path, output_path, view)

if __name__ == "__main__":
    # 示例：对 data 文件夹下的所有 .npy 文件进行渲染
    data_folder = "shapenet"
    output_folder = "rendered_images"
    os.makedirs(output_folder, exist_ok=True)
    for synset_id in os.listdir(data_folder):#类别
        synset_path = os.path.join(data_folder, synset_id) #模型plane
        if not os.path.isdir(synset_path):
            continue
        
        output_synset_dir = os.path.join(output_folder, synset_id)# plane chair
        if not os.path.exists(output_synset_dir):
            os.makedirs(output_synset_dir)

        # 遍历所有模型
        for model_id in os.listdir(synset_path):
            model_path = os.path.join(synset_path, model_id, "models", "model_normalized.obj")
            if not os.path.exists(model_path):
                continue
            output_dir = os.path.join(output_synset_dir, model_id)
            render_all_views_for_obj(model_path, output_dir)



