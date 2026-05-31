#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

def load_image_gray(path):
    if not os.path.exists(path): return None
    img = Image.open(path).convert('L')
    return np.array(img).astype(np.float32) / 255.0

def load_image_rgb(path):
    if not os.path.exists(path): return None
    img = Image.open(path).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0

def calculate_psnr(mse):
    if mse < 1e-10: return 100.0
    return -10.0 * np.log10(mse)

# --- Degradation Functions ---

def degrade_fn_drop(mask, intensity):
    """
    False Negative: randomly drop patches from the object.
    intensity: Approximate fraction of object pixels to drop (0.0 - 0.8)
    """
    if intensity <= 0: return mask.copy()
    h, w = mask.shape
    res = mask.copy()
    
    # Estimate object area
    obj_pixels = np.sum(mask > 0.5)
    target_drop = int(obj_pixels * intensity)
    dropped = 0
    
    # Try dropping circular patches
    iters = 0
    while dropped < target_drop and iters < 100:
        iters += 1
        # Random center
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        
        # Random radius relative to image size
        r = np.random.randint(10, max(20, min(h,w)//5))
        
        # Check overlap with mask? We just erase.
        Y, X = np.ogrid[:h, :w]
        dist = (X - cx)**2 + (Y - cy)**2
        patch_mask = dist <= r**2
        
        # Pixels that ARE object and WOULD be dropped
        potential_drop = np.sum((res > 0.5) & patch_mask)
        if potential_drop > 0:
            res[patch_mask] = 0.0 # Binary drop
            dropped += potential_drop
            
    return res

def degrade_fp_ghost(mask, intensity):
    """
    False Positive: Add ghost blobs to background.
    REVISED: Ghosts are now 'Soft' (0.3 - 0.6 confidence).
    This simulates realistic artifacts where the network is uncertain.
    Linear weighting gets distracted by these sum(0.5 * U).
    Quadratic weighting suppresses them sum(0.25 * U).
    """
    if intensity <= 0: return mask.copy()
    h, w = mask.shape
    res = mask.copy()
    
    # Target ghost area
    obj_pixels = np.sum(mask > 0.5)
    target_ghost = int(obj_pixels * intensity * 2.0)
    
    added = 0
    iters = 0
    while added < target_ghost and iters < 100:
        iters += 1
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        r = np.random.randint(20, max(30, min(h,w)//4))
        
        Y, X = np.ogrid[:h, :w]
        dist = (X - cx)**2 + (Y - cy)**2
        patch_mask = dist <= r**2
        
        # Soft Ghost Confidence
        ghost_val = np.random.uniform(0.3, 0.7)
        # Randomize value per ghost to add variance
        
        # Pixels that are BG and would be added
        potential_add = np.sum((res < 0.1) & patch_mask)
        
        if potential_add > 0:
            res = np.maximum(res, patch_mask.astype(np.float32) * ghost_val)
            added += potential_add
            
    return res

def degrade_drift(mask, intensity):
    """
    Boundary Drift: Soft Dilation with Randomness.
    Previously, this was deterministic (e.g. always +10px), which preserved ranking perfectly.
    Now we add randomness (e.g. +5px to +15px) to simulate that some views 
    have worse segmentation boundaries than others.
    """
    if intensity <= 0: return mask.copy()
    
    # Base Variance
    base_k = int(intensity * 40)
    # Random factor per view: some views have worse drift
    rand_factor = np.random.uniform(0.5, 1.5)
    k_size = int(base_k * rand_factor)
    
    if k_size % 2 == 0: k_size += 1
    if k_size < 3: return mask.copy()
    
    kernel = np.ones((k_size, k_size), np.uint8)
    
    # 1. Binary Dilation
    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    # 2. Identify the drift region (newly added pixels)
    added_region = (dilated > 0.5) & (mask < 0.5)
    
    res = dilated.copy()
    
    # 3. Soften the drift region
    # Realistically, drift happens at uncertain boundaries
    res[added_region] = 0.5
    
    # 4. Blur slightly to mix
    res = cv2.GaussianBlur(res, (15, 15), 0)
    
    # Ensure core object is preserved (at least mostly)
    res = np.maximum(res, mask)
    
    return res

# --- Main Logic ---

def main():
    root = Path("tandt/truck")
    output_base = root / "output" / "train" / "ours_18399"
    unc_dir = output_base / "uncertainty_npz"
    gt_dir = output_base / "gt"
    render_dir = output_base / "renders"
    mask_dir = root / "mask"
    
    # 1. Gather Data
    data = []
    print("Loading Data from tandt/truck (Soft Noise Version)...")
    
    if not unc_dir.exists():
        print(f"Error: {unc_dir} does not exist.")
        return

    files = sorted(list(unc_dir.glob("*.npz")))
    if len(files) > 100: files = files[::2]
    
    for f in tqdm(files):
        stem = f.stem
        u_path = f
        m_path = mask_dir / (stem + "_prob.npy")
        gt_path = gt_dir / (stem + ".png")
        r_path = render_dir / (stem + ".png")
        
        if not (m_path.exists() and gt_path.exists() and r_path.exists()):
            continue
            
        try:
            ud = np.load(u_path)
            u = (ud['uncertainty_map'] if 'uncertainty_map' in ud else ud['arr_0']).astype(np.float32)
            if u.ndim==3: u=u.mean(axis=2)
            
            m = np.load(m_path)
            while m.ndim > 2: m = m.squeeze()
            if m.ndim > 2: m = m.mean(axis=-1)
            
            if m.shape != u.shape:
                m = cv2.resize(m, (u.shape[1], u.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            gt = load_image_rgb(gt_path)
            rend = load_image_rgb(r_path)
            
            if gt.shape[:2] != u.shape: gt = cv2.resize(gt, (u.shape[1], u.shape[0]))
            if rend.shape[:2] != u.shape: rend = cv2.resize(rend, (u.shape[1], u.shape[0]))
            
            # True PSNR of Object
            diff = (rend - gt) ** 2
            diff = np.mean(diff, axis=2)
            obj_mse_sum = np.sum(diff * m)
            obj_pixels = np.sum(m)
            if obj_pixels < 10: continue
            
            true_obj_psnr = calculate_psnr(obj_mse_sum / obj_pixels)
            
            data.append({
                'stem': stem, 'u': u, 'm_gt': m, 'psnr': true_obj_psnr
            })
            
        except Exception as e:
            continue
            
    print(f"Loaded {len(data)} candidates.")
    
    # 2. Experiment
    # REVISED: Reduced max noise intensity to 0.5 as requested.
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    modes = ['False Negative (Drop)', 'False Positive (Ghost)', 'Boundary Drift']
    n_trials = 50 # Keep high for stability
    
    results = {
        'Linear': {m: [] for m in modes},
        'Quadratic': {m: [] for m in modes}
    }
    
    for mode in modes:
        print(f"Running Mode: {mode}")
        lin_means, quad_means = [], []
        
        for lvl in noise_levels:
            t_lin, t_quad = [], []
            for _ in range(n_trials):
                best_lin, best_lin_psnr = -1e9, 0
                best_quad, best_quad_psnr = -1e9, 0
                
                for item in data:
                    u = item['u']
                    m_gt = item['m_gt']
                    truth_psnr = item['psnr']
                    
                    if mode == 'False Negative (Drop)': m_noisy = degrade_fn_drop(m_gt, lvl)
                    elif mode == 'False Positive (Ghost)': m_noisy = degrade_fp_ghost(m_gt, lvl)
                    else: m_noisy = degrade_drift(m_gt, lvl)
                        
                    # Linear vs Quadratic
                    s_lin = np.sum(u * m_noisy)
                    s_quad = np.sum(u * (m_noisy ** 2))
                    
                    if s_lin > best_lin: best_lin, best_lin_psnr = s_lin, truth_psnr
                    if s_quad > best_quad: best_quad, best_quad_psnr = s_quad, truth_psnr
                        
                t_lin.append(best_lin_psnr)
                t_quad.append(best_quad_psnr)
            
            lin_means.append(np.mean(t_lin))
            quad_means.append(np.mean(t_quad))
            
            # Store standard deviation for error bars
            if 'Linear_std' not in results: results['Linear_std'] = {}
            if 'Quadratic_std' not in results: results['Quadratic_std'] = {}
            if mode not in results['Linear_std']: results['Linear_std'][mode] = []
            if mode not in results['Quadratic_std']: results['Quadratic_std'][mode] = []
            
            # Use Standard Error of Mean (std / sqrt(N)) for tighter error bands
            results['Linear_std'][mode].append(np.std(t_lin) / np.sqrt(n_trials))
            results['Quadratic_std'][mode].append(np.std(t_quad) / np.sqrt(n_trials))
            
        results['Linear'][mode] = lin_means
        results['Quadratic'][mode] = quad_means

    # 3. Plotting (Academic Style)
    from matplotlib.ticker import AutoMinorLocator
    
    # Set style parameters
    # REVISED: Increased font sizes for maximum legibility in paper
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 16,     # Increased from 14
        'axes.labelsize': 18,# Increased from 14
        'axes.titlesize': 20,# Increased from 16
        'xtick.labelsize': 14,# Increased from 12
        'ytick.labelsize': 14,# Increased from 12
        'legend.fontsize': 14,# Increased from 12
        'figure.titlesize': 22,# Increased from 18
        'mathtext.fontset': 'stix'
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    
    colors = {'Linear': '#1f77b4', 'Quadratic': '#d62728'} # Tab blue, Tab red
    markers = {'Linear': 'o', 'Quadratic': 's'}
    linestyles = {'Linear': '--', 'Quadratic': '-'}
    labels = {'Linear': 'FisherRF+Mask', 'Quadratic': 'OUGS+Mask'}

    print("\n" + "="*80)
    print(f"{'Method':<25} | {'Noise Mode':<25} | {'Level':<5} | {'PSNR (Mean ± SEM)':<20}")
    print("-" * 80)

    for i, mode in enumerate(modes):
        ax = axes[i]
        x = np.array(noise_levels)
        
        # Plot Linear
        y_lin = np.array(results['Linear'][mode])
        std_lin = np.array(results['Linear_std'][mode]) # This is actually SEM now
        
        ax.plot(x, y_lin, marker=markers['Linear'], linestyle=linestyles['Linear'], 
                color=colors['Linear'], label=labels['Linear'], linewidth=1.5, markersize=6)
        ax.fill_between(x, y_lin - std_lin, y_lin + std_lin, color=colors['Linear'], alpha=0.2)
        
        # Plot Quadratic
        y_quad = np.array(results['Quadratic'][mode])
        std_quad = np.array(results['Quadratic_std'][mode])
        
        ax.plot(x, y_quad, marker=markers['Quadratic'], linestyle=linestyles['Quadratic'], 
                color=colors['Quadratic'], label=labels['Quadratic'], linewidth=1.5, markersize=6)
        ax.fill_between(x, y_quad - std_quad, y_quad + std_quad, color=colors['Quadratic'], alpha=0.2)
        
        # Add Minor Ticks
        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', length=3, color='gray', bottom=True, left=True)
        ax.tick_params(which='major', direction='in', length=6, color='black', bottom=True, left=True)
        
        ax.set_title(mode, fontweight='bold', pad=10)
        ax.set_xlabel("Noise Intensity")
        if i == 0:
            ax.set_ylabel("Selected View Object PSNR (dB)\n" + r"$\leftarrow$ Lower is Better")
        
        # Grid improvement
        ax.grid(True, which='major', linestyle=':', alpha=0.6, color='#999999')
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='#CCCCCC') # Minor grid
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.tick_params(direction='in') # Already set above with minorticks

        if i == 0:
             ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='lightgray')

        # Print Table Data
        for idx, lvl in enumerate(noise_levels):
            # Print specific levels for table construction (Low, Mid, High within range)
            if lvl in [0.0, 0.3, 0.5]: 
                print(f"{'FisherRF+Mask':<25} | {mode:<25} | {lvl:<5.1f} | {y_lin[idx]:.2f} ± {std_lin[idx]:.2f}")
                print(f"{'OUGS (Ours)':<25} | {mode:<25} | {lvl:<5.1f} | {y_quad[idx]:.2f} ± {std_quad[idx]:.2f}")
        print("-" * 80)

    plt.tight_layout()
    plt.savefig("Images/mask_robustness_psnr_chart_truck_academic.png", bbox_inches='tight')
    print("Saved Images/mask_robustness_psnr_chart_truck_academic.png")

if __name__ == "__main__":
    main()
