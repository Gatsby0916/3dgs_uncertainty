import os
import numpy as np
from PIL import Image

# 输入输出路径
root = 'LF/basket/output/test/ours_18400'
mask_dir = 'LF/basket/mask'
folders = ['gt', 'renders']

for folder in folders:
    in_dir = os.path.join(root, folder)
    out_dir = os.path.join(root, folder + '_masked')
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(in_dir):
        if not fname.endswith('.png'):
            continue
        img_path = os.path.join(in_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace('.png', '_prob.npy'))
        if not os.path.exists(mask_path):
            print(f'[WARN] mask not found: {mask_path}')
            continue
        img = np.array(Image.open(img_path))
        mask = np.load(mask_path)
        # 二值化mask，阈值可调
        mask_bin = (mask > 0.5).astype(np.uint8)
        if img.ndim == 2:
            img = img * mask_bin
        else:
            img = img * mask_bin[..., None]
        # 去除多余维度，确保为(H,W)或(H,W,3)
        img = np.squeeze(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(out_dir, fname))
        print(f'Saved: {os.path.join(out_dir, fname)}')
print('全部处理完成！')
