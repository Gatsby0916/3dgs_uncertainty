import numpy as np
import matplotlib.pyplot as plt
import os

def plot_res():
    scenes = ['basket']
    modes = ['nbv', 'random']
    
    plt.figure(figsize=(10, 6))
    
    for scene in scenes:
        for mode in modes:
            path = f"results/warp_experiment/{scene}_{mode}_prop_curve.npy"
            if not os.path.exists(path):
                print(f"Missing {path}")
                continue
                
            data = np.load(path, allow_pickle=True)
            
            # Robust extraction
            clean_data = []
            for item in data:
                if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 4:
                    clean_data.append(item)
            
            if not clean_data:
                print(f"No valid data in {path}")
                continue
                
            clean_data = np.array(clean_data)
            steps = np.arange(len(clean_data)) + 1
            avg_iou_set = clean_data[:, 2]
            avg_iou_rem = clean_data[:, 3]
            
            plt.plot(steps, avg_iou_set, label=f"{scene} - {mode} (Added Set Quality)", marker='o', linestyle='--')
            plt.plot(steps, avg_iou_rem, label=f"{scene} - {mode} (Remaining Pool Quality)", marker='x', linestyle='-')

    plt.xlabel("Propagation Steps")
    plt.ylabel("Average IoU")
    plt.title("Mask Propagation: NBV (OUGS) vs Random\nDoes adding views improve remaining pool?")
    plt.legend()
    plt.grid(True)
    out_path = "results/warp_experiment/propagation_comparison.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_res()
