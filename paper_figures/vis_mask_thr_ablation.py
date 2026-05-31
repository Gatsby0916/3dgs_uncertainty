import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# 配置
images_dir = 'LF/statue/images'
mask_dir = 'LF/statue/mask'
out_dir = 'LF/statue/mask_thr_ablation'
train_split = 'LF/statue/train_split.txt'
THRS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
CMAP = 'viridis'

os.makedirs(out_dir, exist_ok=True)

# 读取train_split
with open(train_split) as f:
    stems = [line.strip() for line in f if line.strip()]


# 只可视化指定stem
stems = ['statue_alex01250.jpg']



for stem in tqdm(stems, desc='Ablation可视化'):
    img_path = os.path.join(images_dir, stem)
    mask_stem = os.path.splitext(stem)[0]
    mask_path = os.path.join(mask_dir, f'{mask_stem}_prob.npy')
    print(f"[DEBUG] img_path: {img_path}")
    print(f"[DEBUG] mask_path: {mask_path}")
    if not os.path.exists(img_path):
        print(f'[ERROR] 图像文件不存在: {img_path}')
    if not os.path.exists(mask_path):
        print(f'[ERROR] mask文件不存在: {mask_path}')
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f'跳过: {stem}')
        continue
    print(f"[INFO] 加载图像和mask: {stem}")
    img = np.array(Image.open(img_path).convert('RGB'))
    mask = np.load(mask_path).squeeze()
    print(f"[INFO] mask shape: {mask.shape}, max: {mask.max()}, min: {mask.min()}")
    if mask.max() > 1.0:
        mask = mask / mask.max()
        print(f"[INFO] mask已归一化到[0,1]")
    # 单独保存thr=0.1,0.3,0.5,0.7,0.9的heatmap（无文字、无坐标轴、无边框）
    for thr in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        mask_thr = mask.copy()
        mask_thr[mask_thr < thr] = 0.0
        fig = plt.figure(figsize=(4, 4))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        ax.imshow(mask_thr, cmap=CMAP)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        save_path = os.path.join(out_dir, f'{stem}_thr{thr:.2f}_heatmap.png')
        print(f"[INFO] 保存: {save_path}")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

print(f'已保存所有ablation可视化到: {out_dir}')
