import numpy as np
import matplotlib.pyplot as plt
import os

scenes = ["basket", "statue", "torch", "africa"]

plt.figure(figsize=(10, 6))

for scene in scenes:
    file_path = f"results/warp_experiment/{scene}_random_curve.npy"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
        
    data = np.load(file_path)
    # data is list of (num_views, mean_iou)
    
    num_views = data[:, 0]
    mean_ious = data[:, 1]
    
    plt.plot(num_views, mean_ious, marker='o', label=f"{scene.capitalize()}")

plt.xlabel("Number of Selected Views")
plt.ylabel("Mean Test IoU of Aggregated Warped Mask")
plt.title("Effect of Increasing Views on Mask Quality (Random Selection)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/warp_experiment/mask_quality_curve.png")
print("Saved plot to results/warp_experiment/mask_quality_curve.png")
