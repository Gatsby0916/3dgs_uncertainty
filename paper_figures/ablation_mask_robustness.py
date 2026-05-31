#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_mask_robustness.py

Implements the "Robustness to Segmentation Noise" experiment.
Simulates synthetic noise (random patch drops, false positives) on segmentation masks
and evaluates the stability of the Uncertainty-guided view selection score.

Comparison:
1. Linear Weighting: Sum(M * U)
2. Binary Weighting: Sum((M > 0.5) * U)
3. Quadratic Weighting (Ours): Sum(M^2 * U)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import random
import cv2

def load_mask(path):
    """Load mask as float32 in [0, 1]"""
    if str(path).endswith('.npy'):
        m = np.load(path)
        # Handle prob dicts if any, but usually just array
        if not isinstance(m, np.ndarray):
             # Try common keys
             if 'arr_0' in m: m = m['arr_0']
             elif 'prob' in m: m = m['prob']
             else: m = m[list(m.keys())[0]]
             
        if m.max() > 1.0: m = m / 255.0
        
        # Squeeze channel if needed
        while m.ndim > 2:
            m = m.squeeze()
            if m.ndim > 2 and m.shape[0] == 1: m = m[0]
            elif m.ndim > 2 and m.shape[-1] == 1: m = m[..., 0]
            elif m.ndim > 2: 
                # Take mean if channels
                m = m.mean(axis=0) if m.shape[0] < m.shape[-1] else m.mean(axis=-1)
            
    else:
        m = np.array(Image.open(path).convert('L')) / 255.0
    return m.astype(np.float32)

def load_uncertainty(path):
    """Load uncertainty map"""
    d = np.load(path)
    res = None
    # Check keys, usually 'uncertainty' or just the array
    if isinstance(d, np.ndarray):
        res = d.astype(np.float32)
    else:
        keys = list(d.keys())
        if 'uncertainty' in keys:
            res = d['uncertainty'].astype(np.float32)
        elif 'arr_0' in keys:
            res = d['arr_0'].astype(np.float32)
        elif len(keys) > 0:
            res = d[keys[0]].astype(np.float32)
    
    if res is not None:
        if res.ndim == 3:
            res = res.mean(axis=-1) # Flatten channels if any
        return res
        
    raise ValueError(f"Cannot load uncertainty from {path}")

def add_synthetic_noise(mask, intensity=0.1, patch_size=32):
    """
    Add synthetic noise to mask.
    intensity: fraction of image area to corrupt
    """
    h, w = mask.shape
    noisy_mask = mask.copy()
    
    # Number of patches to add
    total_pixels = h * w
    patch_area = patch_size * patch_size
    n_patches = int((total_pixels * intensity) / patch_area)
    
    for _ in range(n_patches):
        # Random location
        if h - patch_size <= 0 or w - patch_size <= 0: break
        y = np.random.randint(0, h - patch_size)
        x = np.random.randint(0, w - patch_size)
        
        # Force a change to verify
        # noise_val = np.random.uniform(0.3, 0.7) 
        
        # Flip logic: if mean of patch > 0.5 (object), set to 0.2 (drop)
        # if mean < 0.5 (bg), set to 0.8 (ghost)
        patch_mean = np.mean(mask[y:y+patch_size, x:x+patch_size])
        if patch_mean > 0.5:
             noise_val = 0.2
        else:
             noise_val = 0.8
             
        # Apply noise
        noisy_mask[y:y+patch_size, x:x+patch_size] = noise_val
        
    return np.clip(noisy_mask, 0.0, 1.0)

def compute_scores(mask, uncertainty):
    """Compute scores using different weightings"""
    # Ensure shapes match
    if mask.ndim == 3: mask = mask.squeeze()
    if uncertainty.ndim == 3: uncertainty = uncertainty.squeeze()
    
    if mask.size == 0 or uncertainty.size == 0:
        return 0, 0, 0

    if mask.shape != uncertainty.shape:
        # print(f"Resizing mask {mask.shape} to {uncertainty.shape}")
        try:
             # Check if width/height > 0
            h, w = uncertainty.shape[:2]
            if h <= 0 or w <= 0: return 0,0,0
            
            mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            # print(f"Resize error: {e}, Mask: {mask.shape}, Unc: {uncertainty.shape}")
            return 0, 0, 0
    else:
        mask_rs = mask
        
    # 1. Linear
    s_linear = np.sum(mask_rs * uncertainty)
    
    # 2. Binary
    s_binary = np.sum((mask_rs > 0.5).astype(np.float32) * uncertainty)
    
    # 3. Quadratic
    s_quad = np.sum((mask_rs ** 2) * uncertainty)
    
    return s_linear, s_binary, s_quad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/bonsai', help='Dataset root')
    parser.add_argument('--output_dir', type=str, default='ablation_results', help='Output directory')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    # Find uncertainty maps
    # Usually in output/train/ours_.../uncertainty_npz/
    output_dir = data_root / "output" / "train"
    if not output_dir.exists():
        print(f"No output dir found at {output_dir}")
        return

    ours_dirs = list(output_dir.glob("ours_*"))
    if not ours_dirs:
        print("No ours_* dir found")
        return
        
    # Pick best ours dir (one with most npz files)
    best_dir = None
    max_files = 0
    for od in ours_dirs:
        ud = od / "uncertainty_npz"
        if ud.exists():
            count = len(list(ud.glob("*.npz")))
            if count > max_files:
                max_files = count
                best_dir = od
                
    if best_dir is None:
        print("No populated uncertainty_npz dir found")
        return
        
    ours_dir = best_dir
    unc_dir = ours_dir / "uncertainty_npz"
    print(f"Using uncertainty from: {unc_dir}")
    
    mask_dir = data_root / "mask" # Assuming masks are here
    if not mask_dir.exists():
        print(f"No mask dir found at {mask_dir}")
        return

    # Gather files
    unc_files = sorted(list(unc_dir.glob("*.npz")))
    valid_pairs = []
    
    for u_path in unc_files:
        # Match mask filename
        # mask filename might be {stem}_prob.npy
        stem = u_path.stem
        
        # Try direct match
        m_path = mask_dir / (stem + ".png")
        if not m_path.exists():
             m_path = mask_dir / (stem + "_prob.npy")
        
        if not m_path.exists():
             # Try jpg
             m_path = mask_dir / (stem + ".jpg")

        if m_path.exists():
            valid_pairs.append((u_path, m_path))
            
    print(f"Found {len(valid_pairs)} pairs.")
    if len(valid_pairs) == 0:
        return
        
    # Use a subset of pairs to speed up
    subset = valid_pairs[:10]
    
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    
    results = {
        'noise': noise_levels,
        'linear_err': [],
        'binary_err': [],
        'quad_err': []
    }
    
    # For each noise level
    for nl in noise_levels:
        l_errs = []
        b_errs = []
        q_errs = []
        
        for u_p, m_p in tqdm(subset, desc=f"Noise {nl}"):
            u = load_uncertainty(u_p)
            m = load_mask(m_p)
            
            # Clean Scores (Reference)
            sl_c, sb_c, sq_c = compute_scores(m, u)
            if sl_c == 0: continue
            
            # Noisy Scores (Average over 5 trials)
            curr_l, curr_b, curr_q = [], [], []
            for _ in range(5):
                m_noisy = add_synthetic_noise(m, intensity=nl)
                # Debug noise
                # diff = np.sum(np.abs(m - m_noisy))
                # if diff == 0 and nl > 0: print("Warning: No noise added")
                
                sl, sb, sq = compute_scores(m_noisy, u)
                curr_l.append( abs(sl - sl_c) / (sl_c + 1e-6) )
                curr_b.append( abs(sb - sb_c) / (sb_c + 1e-6) )
                curr_q.append( abs(sq - sq_c) / (sq_c + 1e-6) )
            
            l_errs.append(np.mean(curr_l))
            b_errs.append(np.mean(curr_b))
            q_errs.append(np.mean(curr_q))
            
        results['linear_err'].append(np.mean(l_errs) if l_errs else 0)
        results['binary_err'].append(np.mean(b_errs) if b_errs else 0)
        results['quad_err'].append(np.mean(q_errs) if q_errs else 0)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(results['noise'], results['linear_err'], marker='o', label='Linear Weighting')
    plt.plot(results['noise'], results['binary_err'], marker='s', label='Binary Threshold')
    plt.plot(results['noise'], results['quad_err'], marker='^', label='Quadratic (Ours)')
    
    plt.xlabel('Noise Intensity (Patch Coverage)')
    plt.ylabel('Relative Score Error (vs Clean)')
    plt.title('Robustness of View Selection Score to Mask Noise')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'mask_robustness_chart.png'))
    print(f"Chart saved to {args.output_dir}/mask_robustness_chart.png")

    # Save CSV
    import csv
    with open(os.path.join(args.output_dir, 'robustness_data.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Noise', 'Linear_Err', 'Binary_Err', 'Quadratic_Err'])
        for i in range(len(noise_levels)):
            writer.writerow([
                noise_levels[i], 
                results['linear_err'][i], 
                results['binary_err'][i], 
                results['quad_err'][i]
            ])
    print("Data saved to CSV.")

if __name__ == "__main__":
    main()
