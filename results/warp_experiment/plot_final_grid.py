import numpy as np
import matplotlib.pyplot as plt

scenes = [
    ("Basket", "results/warp_experiment/basket_multi"),
    ("Statue", "results/warp_experiment/statue_multi"),
    ("Torch", "results/warp_experiment/torch_multi"),
    ("Africa", "results/warp_experiment/africa_multi")
]

fig, axes = plt.subplots(4, 4, figsize=(16, 9)) # 4 rows, 4 cols, tuned aspect ratio

# Titles for columns
col_titles = ["Warped Mask", "GT Mask", "Difference", "Overlay"]

for row_idx, (scene_name, base_name) in enumerate(scenes):
    pred_path = base_name + "_pred.npy"
    gt_path = base_name + "_gt.npy"
    
    try:
        pred_mask = np.load(pred_path)
        gt_mask = np.load(gt_path)
    except Exception as e:
        print(f"Error loading {base_name}: {e}")
        continue
        
    # Helpers
    pred_binary = (pred_mask > 0.5)
    gt_binary = (gt_mask > 0.5)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / union if union > 0 else 0

    # 1. Warped Mask
    ax = axes[row_idx, 0]
    ax.imshow(pred_mask, cmap='gray')
    ax.axis('off')
    # Row label on the left
    ax.text(-50, pred_mask.shape[0]//2, scene_name, fontsize=14, rotation=90, va='center', ha='right', weight='bold')

    # 2. GT Mask
    ax = axes[row_idx, 1]
    ax.imshow(gt_mask, cmap='gray')
    ax.axis('off')

    # 3. Difference
    ax = axes[row_idx, 2]
    diff = np.ones((*pred_mask.shape, 3)) * 0.1 # Dark grey bg
    # Colors
    # Missed (GT=1, Pred=0) -> Coral Red
    diff[np.logical_and(gt_binary, ~pred_binary)] = [0.93, 0.46, 0.44]
    # False Positive (GT=0, Pred=1) -> Periwinkle Blue
    diff[np.logical_and(~gt_binary, pred_binary)] = [0.46, 0.64, 0.93]
    # Correct -> Sage Green
    diff[np.logical_and(gt_binary, pred_binary)] = [0.56, 0.76, 0.48]
    
    ax.imshow(diff)
    ax.axis('off')

    # 4. Overlay
    ax = axes[row_idx, 3]
    ax.imshow(gt_mask, cmap='gray', alpha=0.5)
    ax.imshow(pred_mask, cmap='viridis', alpha=0.4)
    # Add IoU text inside the overlay or below?
    # User said "GT Mask... other don't need". 
    # But usually Overlay needs IoU. 
    # Let's put IoU in the corner of Overlay
    ax.text(5, 20, f"IoU={iou:.2f}", color="white", fontsize=10, bbox=dict(facecolor='black', alpha=0.5,linewidth=0))
    ax.axis('off')
    
    # Set Column Titles ONLY on top row
    if row_idx == 0:
        for col_idx in range(4):
            # Reduce pad for title
            axes[0, col_idx].set_title(col_titles[col_idx], fontsize=14, pad=5)

# Reduce margins and spacing to absolute minimum
plt.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.0, wspace=0.0, hspace=0.0)
plt.savefig("results/warp_experiment/combined_multi_grid_clean.png", dpi=150, bbox_inches='tight', pad_inches=0.0)
print("Saved results/warp_experiment/combined_multi_grid_clean.png")
