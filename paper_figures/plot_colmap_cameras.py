import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scene import colmap_loader

# 路径配置
colmap_dir = 'LF/statue/sparse/0'
cameras_bin = os.path.join(colmap_dir, 'cameras.bin')
images_bin = os.path.join(colmap_dir, 'images.bin')

# 读取相机参数和位姿
cameras = colmap_loader.read_intrinsics_binary(cameras_bin)
images = colmap_loader.read_extrinsics_binary(images_bin)


# 只绘制train_split.txt中的相机
split_path = 'LF/statue/train_split.txt'
with open(split_path) as f:
    split_stems = set(line.strip() for line in f if line.strip())


# 分别收集所有相机和train_split相机
all_cam_centers = []
all_img_names = []
split_cam_centers = []
split_img_names = []
other_cam_centers = []
other_img_names = []
print("[DEBUG] 前5个相机的qvec/tvec/center:")
cnt = 0
for img in images.values():
    stem = os.path.splitext(os.path.basename(img.name))[0]
    R = colmap_loader.qvec2rotmat(img.qvec)
    t = img.tvec
    center = R.T @ (-t)
    all_cam_centers.append(center)
    all_img_names.append(img.name)
    if stem in split_stems:
        split_cam_centers.append(center)
        split_img_names.append(img.name)
        if cnt < 10:
            print(f"  name: {img.name}")
            print(f"    qvec: {img.qvec}")
            print(f"    tvec: {img.tvec}")
            print(f"    center: {center}")
        cnt += 1
    else:
        other_cam_centers.append(center)
        other_img_names.append(img.name)
all_cam_centers = np.stack(all_cam_centers, axis=0)
split_cam_centers = np.stack(split_cam_centers, axis=0) if split_cam_centers else np.zeros((0,3))
other_cam_centers = np.stack(other_cam_centers, axis=0) if other_cam_centers else np.zeros((0,3))

# 绘制三维相机分布

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_camera(ax, center, R, size=0.2, color='b', alpha=0.7):
    # 三棱锥相机形状，底面朝z轴负方向
    # 定义局部坐标下的锥体
    p0 = np.zeros(3)
    p1 = np.array([size, size, -size*1.5])
    p2 = np.array([-size, size, -size*1.5])
    p3 = np.array([0, -size, -size*1.5])
    # 旋转到世界坐标
    pts = np.stack([p0, p1, p2, p3], axis=0)
    pts = (R @ pts.T).T + center
    p0w, p1w, p2w, p3w = pts
    verts = [[p0w, p1w, p2w], [p0w, p2w, p3w], [p0w, p3w, p1w], [p1w, p2w, p3w]]
    poly = Poly3DCollection(verts, facecolors=color, edgecolors='k', linewidths=0.5, alpha=alpha)
    ax.add_collection3d(poly)


# 加载并绘制点云（物体）
points3d_bin = os.path.join(colmap_dir, 'points3D.bin')
points3d = None
if os.path.exists(points3d_bin):
    try:
        points3d = colmap_loader.read_points3d_binary(points3d_bin)
        print(f"[INFO] 加载点云: {points3d_bin}, 点数: {len(points3d)}")
    except Exception as e:
        print(f"[WARN] 读取点云失败: {e}")
else:
    print(f"[WARN] 未找到点云文件: {points3d_bin}")




# 只绘制train_split中的相机锥体，前4个高亮
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')


# 只遍历train_split中的相机，前4个高亮，并收集所有锥体顶点
split_img_list = [img for img in images.values() if os.path.splitext(os.path.basename(img.name))[0] in split_stems]
all_vertices = []
for idx, img in enumerate(split_img_list):
    R = colmap_loader.qvec2rotmat(img.qvec)
    t = img.tvec
    center = R.T @ (-t)
    size = 0.08
    # 计算锥体顶点
    p0 = np.zeros(3)
    p1 = np.array([size, size, -size*1.5])
    p2 = np.array([-size, size, -size*1.5])
    p3 = np.array([0, -size, -size*1.5])
    pts = (R @ np.stack([p0, p1, p2, p3], axis=0).T).T + center
    all_vertices.append(pts)
    if idx < 5:
        plot_camera(ax, center, R, size=size, color='magenta', alpha=1.0)
    else:
        plot_camera(ax, center, R, size=size, color='gray', alpha=0.18)

all_vertices = np.concatenate(all_vertices, axis=0)
min_xyz = all_vertices.min(axis=0)
max_xyz = all_vertices.max(axis=0)
center_xyz = (min_xyz + max_xyz) / 2
max_range = (max_xyz - min_xyz).max() / 2
cube_min = center_xyz - max_range
cube_max = center_xyz + max_range
ax.set_xlim([cube_min[0], cube_max[0]])
ax.set_ylim([cube_min[1], cube_max[1]])
ax.set_zlim([cube_min[2], cube_max[2]])

# 设置视角
ax.view_init(elev=10, azim=200)

# 去除坐标轴和刻度
ax.set_axis_off()
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()
output_dir = 'LF/statue/colmap_camera_plot'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'colmap_camera_plot_3d_cameras_only.png')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'三维相机锥体分布图已保存到: {output_path}')


# 计算所有点的包围盒，并设置为正方体（立方体坐标系）
all_xyz = [all_cam_centers]
if points3d is not None and len(points3d) > 0:
    pts = np.array([p.xyz for p in points3d.values()])
    all_xyz.append(pts)
all_xyz = np.concatenate(all_xyz, axis=0)
min_xyz = all_xyz.min(axis=0)
max_xyz = all_xyz.max(axis=0)
center_xyz = (min_xyz + max_xyz) / 2
max_range = (max_xyz - min_xyz).max() / 2
cube_min = center_xyz - max_range
cube_max = center_xyz + max_range
ax.set_xlim([cube_min[0], cube_max[0]])
ax.set_ylim([cube_min[1], cube_max[1]])
ax.set_zlim([cube_min[2], cube_max[2]])

# 美化坐标轴
ax.xaxis._axinfo['grid'].update({'linewidth': 0.5, 'color': '#cccccc'})
ax.yaxis._axinfo['grid'].update({'linewidth': 0.5, 'color': '#cccccc'})
ax.zaxis._axinfo['grid'].update({'linewidth': 0.5, 'color': '#cccccc'})
ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
# 已移除w_xaxis等属性设置，避免报错



# 去除顶部和右侧背景色
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()
output_dir = 'LF/statue/colmap_camera_plot'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'colmap_camera_plot_3d.png')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'三维相机点阵+物体分布图已保存到: {output_path}')