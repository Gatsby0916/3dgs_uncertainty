import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

# 配置路径
images_dir = 'LF/basket/images'
mask_dir = 'LF/basket/mask'
out_dir = 'LF/basket/gt_masked'
os.makedirs(out_dir, exist_ok=True)

# 匹配所有gt图像
image_files = sorted(glob(os.path.join(images_dir, '*.png')))

for img_path in tqdm(image_files, desc='过滤GT'):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    # 匹配mask文件（支持_Cropped_prob.npy等后缀）
    mask_path = os.path.join(mask_dir, f'{stem}_prob.npy')
    if not os.path.exists(mask_path):
        # 尝试大小写变体
        mask_candidates = glob(os.path.join(mask_dir, f'*{stem}*_prob.npy'))
        if mask_candidates:
            mask_path = mask_candidates[0]
        else:
            print(f'未找到mask: {mask_path}')
            continue
    # 加载gt和mask
    img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
    mask = np.load(mask_path).squeeze()
    if mask.max() > 1.0:
        mask = mask / mask.max()
    # 低于0.5的概率直接置0
    mask[mask < 0.5] = 0.0
    # mask扩展到3通道
    if mask.ndim == 2:
        mask3 = np.stack([mask]*3, axis=-1)
    else:
        mask3 = mask
    img_masked = img * mask3
    # 保存
    out_path = os.path.join(out_dir, f'{stem}_gt_masked.png')
    Image.fromarray((img_masked*255).astype(np.uint8)).save(out_path)

print(f'已保存所有过滤后的gt到: {out_dir}')
