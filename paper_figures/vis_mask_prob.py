import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

mask_dir = 'LF/statue/mask'
out_dir = 'LF/statue/mask_vis'
os.makedirs(out_dir, exist_ok=True)

mask_files = sorted(glob(os.path.join(mask_dir, '*.npy')))


for mask_path in tqdm(mask_files, desc='可视化mask'):
    mask = np.load(mask_path).squeeze()
    mask[mask < 0.5] = 0.0
    plt.imsave(
        os.path.join(out_dir, os.path.basename(mask_path).replace('.npy', '.png')),
        mask,
        cmap='plasma'
    )

print(f"已保存所有mask可视化到: {out_dir}")
