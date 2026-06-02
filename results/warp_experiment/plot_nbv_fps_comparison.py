import numpy as np
import matplotlib.pyplot as plt
import os

def plot_comparison():
    scene = 'basket'
    random_path = f"results/warp_experiment/{scene}_random_fps_curve.npy"
    nbv_path = f"results/warp_experiment/{scene}_nbv_fps_curve.npy"
    
    if not os.path.exists(random_path) or not os.path.exists(nbv_path):
        print("Data missing")
        return
        
    res_rand = np.load(random_path)
    res_nbv = np.load(nbv_path)
    
    x_rand = res_rand[:, 0]
    y_rand = res_rand[:, 1]
    
    x_nbv = res_nbv[:, 0]
    y_nbv = res_nbv[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_rand, y_rand, 'o--', label='Random Selection + FPS Init', linewidth=2, color='gray')
    plt.plot(x_nbv, y_nbv, 'o-', label='Uncertainty-Guided (NBV) + FPS Init', linewidth=2, color='red')
    
    plt.xlabel("Number of Views")
    plt.ylabel("Mask IoU (Test Set)")
    plt.title(f"NBV vs Random Selection: Mask Quality ({scene.capitalize()})\nInitial: Farthest Point Sampling(4) - Aggregation: Union")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = f"results/warp_experiment/{scene}_nbv_fps_comparison.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_comparison()
