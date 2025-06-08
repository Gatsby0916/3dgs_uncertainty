import os
import matplotlib.pyplot as plt
import numpy as np

# ———— 路径设置 ————  
base_path = os.path.join("data", "LF", "statue", "output", "train", "ours_7000")
gt_path = os.path.join(base_path, "gt", "00006.png")
render_path = os.path.join(base_path, "renders", "00006.png")
unc_path = os.path.join(base_path, "uncertainty", "00006.png")
new_unc_path = r"C:\Users\李海毅\Desktop\CVlearning\3DGS_uncertainty\gaussian-splatting\data\LF\statue\output\train\ours_7000\uncertainty\train_uncertainty_map_statue_alex01370.png"

# ———— 读取并归一化 ————
gt = plt.imread(gt_path).astype(np.float32) / 255.0
render = plt.imread(render_path).astype(np.float32) / 255.0
uncertainty = plt.imread(unc_path).astype(np.float32) / 255.0
new_uncertainty = plt.imread(new_unc_path).astype(np.float32) / 255.0

if uncertainty.ndim == 3:
    uncertainty = uncertainty.mean(axis=-1)
if new_uncertainty.ndim == 3:
    new_uncertainty = new_uncertainty.mean(axis=-1)

# ———— 计算误差图 ————
err = np.abs(gt - render).mean(axis=-1)

# ———— Gamma 增强 ————
def enhance_contrast(img, gamma=0.8):
    return np.power(img, gamma)

err_enh = enhance_contrast(err)
unc_enh = enhance_contrast(uncertainty)
new_unc_enh = enhance_contrast(new_uncertainty)

# ———— 放到同一个视窗里 ————
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Error Heatmap
im0 = axs[0].imshow(err_enh, cmap='cividis')
axs[0].set_title("Error Heatmap")
axs[0].axis('off')
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

# Ours Heatmap
im1 = axs[1].imshow(unc_enh, cmap='cividis')
axs[1].set_title("Ours Heatmap")
axs[1].axis('off')
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

# FisherRF Heatmap
im2 = axs[2].imshow(new_unc_enh, cmap='cividis')
axs[2].set_title("FisherRF Heatmap")
axs[2].axis('off')
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
