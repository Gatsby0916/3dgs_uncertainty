import os
import sys
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import random
import json

# Reuse WarpTool
from warp_masks import WarpTool

def calculate_iou_from_masks(pred, gt):
    if pred is None or gt is None: return 0.0
    pred_b = (pred > 0.5)
    gt_b = (gt > 0.5)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return inter / union if union > 0 else 0.0

# Global Cache for warped masks: (src_idx, tgt_idx) -> torch.Tensor
WARP_CACHE = {}

def get_cached_warp_mask(warper, src_idx, tgt_idx):
    key = (src_idx, tgt_idx)
    if key in WARP_CACHE:
        return WARP_CACHE[key]
    
    # Suppress output 
    sys.stdout = open(os.devnull, 'w')
    try:
        m = warper.warp_mask(src_idx, tgt_idx)
    except Exception:
        m = None
    finally:
        sys.stdout = sys.__stdout__
        
    if m is None:
        m = torch.zeros((warper.H_render, warper.W_render), device=warper.device, dtype=torch.float32)
    
    WARP_CACHE[key] = m
    return m

def calculate_uncertainty_score(warper, current_set_indices, candidate_idx):
    """
    Score = Sum of Variance of projected masks from current_set.
    """
    masks = []
    for src_idx in current_set_indices:
        m = get_cached_warp_mask(warper, src_idx, candidate_idx)
        masks.append(m)

    if not masks: return 0.0
    if len(masks) == 1: return 0.0 
        
    stack = torch.stack(masks)
    variance_map = torch.var(stack, dim=0, unbiased=False) 
    score = variance_map.sum().item()
    return score

def get_fps_indices(warper, pool_ids, n=4):
    """
    Select n views using Farthest Point Sampling based on camera positions.
    Starts with the first view in pool_ids (index 0).
    """
    # 1. Get positions
    positions = []
    valid_pool_ids = []
    
    for pid in pool_ids:
        if pid not in warper.image_id_to_idx: continue
        info = warper.get_cam_info(pid)
        # Check if cameras.json format load by WarpTool has 'position'
        # The tool output showed it does: "position": [-3.24...]
        pos = np.array(info['position'])
        positions.append(pos)
        valid_pool_ids.append(pid)
        
    if len(positions) < n:
        print(f"Error: Not enough cameras for FPS ({len(positions)} < {n})")
        return valid_pool_ids
        
    # FPS Algorithm
    # Start with index 0 (as per FisherRF/gen_split.py)
    selected_indices = [0] 
    selected_ids = [valid_pool_ids[0]]
    picked_centers = [positions[0]]
    
    print(f"FPS Seed: {selected_ids[0]}")
    
    for _ in range(n - 1):
        max_dist = -1
        best_r_idx = -1
        
        # Search among all (conceptually)
        for r_idx, center in enumerate(positions):
            if r_idx in selected_indices: continue
            
            # min dist to any selected
            dists = [np.linalg.norm(center - p) for p in picked_centers]
            min_d = min(dists)
            
            if min_d > max_dist:
                max_dist = min_d
                best_r_idx = r_idx
        
        if best_r_idx != -1:
            selected_indices.append(best_r_idx)
            selected_ids.append(valid_pool_ids[best_r_idx])
            picked_centers.append(positions[best_r_idx])
            
    return selected_ids

def run_simulation(source_path, model_output_path, colmap_path, train_pool_ids, target_set_ids, target_total_views=20, mode='nbv'):
    warper = WarpTool(source_path, model_output_path, colmap_path=colmap_path)
    
    # 1. FPS Initialization (4 views)
    print(f"--- Initialization (FPS, n=4) ---")
    current_set_ids = get_fps_indices(warper, train_pool_ids, n=4)
    remaining_pool_ids = [pid for pid in train_pool_ids if pid not in current_set_ids and pid in warper.image_id_to_idx]
    
    results = [] 
    
    # Evaluation helper
    def evaluate(view_set_ids, target_ids):
        ious = []
        for tgt_id in target_ids:
            if tgt_id not in warper.image_id_to_idx: continue
            tgt_idx = warper.image_id_to_idx[tgt_id]
            
            final_mask = None
            for src_id in view_set_ids:
                src_idx = warper.image_id_to_idx[src_id]
                m = get_cached_warp_mask(warper, src_idx, tgt_idx)
                if final_mask is None: final_mask = m.clone()
                else: final_mask = torch.max(final_mask, m)
            
            if final_mask is None:
                final_mask = torch.zeros((warper.H_render, warper.W_render), device=warper.device)
                
            tgt_info = warper.get_cam_info(tgt_id)
            gt_mask = warper.load_mask(tgt_info['name'])
            
            pred = final_mask.cpu().numpy()
            gt = gt_mask.cpu().numpy() if gt_mask is not None else None
            
            iou = calculate_iou_from_masks(pred, gt)
            ious.append(iou)
        return np.mean(ious)

    # Initial Eval
    print(f"Step 0: Initial Set {current_set_ids}")
    mean_iou = evaluate(current_set_ids, target_set_ids)
    print(f"  -> Mean IoU: {mean_iou:.4f}")
    results.append((len(current_set_ids), mean_iou))
    
    # Needs to reach 20 views. Currently 4. Steps needed = 16.
    steps_needed = target_total_views - len(current_set_ids)
    
    for step in range(steps_needed):
        if not remaining_pool_ids: break
        
        next_view_id = None
        
        if mode == 'random':
            next_view_id = random.choice(remaining_pool_ids)
            
        elif mode == 'nbv':
            # Score candidates
            best_score = -1.0
            best_id_candidate = None
            
            current_indices = [warper.image_id_to_idx[sid] for sid in current_set_ids]
            
            # Use tqdm if possible, otherwise silent
            # Optimization: check 20 random candidates? No, full scan is fine, dataset is small (~150).
            for cand_id in remaining_pool_ids:
                cand_idx = warper.image_id_to_idx[cand_id]
                score = calculate_uncertainty_score(warper, current_indices, cand_idx)
                if score > best_score:
                    best_score = score
                    best_id_candidate = cand_id
            
            next_view_id = best_id_candidate
            print(f"  [NBV] Best Score {best_score:.2f} -> View {next_view_id}")
            
        if next_view_id is not None:
            current_set_ids.append(next_view_id)
            remaining_pool_ids.remove(next_view_id)
            
            mean_iou = evaluate(current_set_ids, target_set_ids)
            print(f"Step {step+1}/{steps_needed}: Added {next_view_id} (Total {len(current_set_ids)}) -> IoU: {mean_iou:.4f}")
            results.append((len(current_set_ids), mean_iou))
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, choices=['basket', 'statue', 'torch', 'africa'])
    parser.add_argument("--mode", type=str, default="nbv", choices=['random', 'nbv'])
    args = parser.parse_args()
    
    base_dir = f"LF/ours/{args.scene}"
    output_dir = f"LF/ours/{args.scene}/output"
    
    with open(os.path.join(output_dir, "cameras.json"), 'r') as f:
        cams = json.load(f)
        
    ids = [c['id'] for c in cams]
    ids.sort() # Ensure consistent order
    
    test_ids = ids[::8]
    train_ids = [i for i in ids if i not in test_ids]
    
    print(f"Scene {args.scene}, Mode {args.mode}")
    print(f"Train Pool: {len(train_ids)}, Test Set: {len(test_ids)}")
    
    results = run_simulation(base_dir, output_dir, base_dir, train_ids, test_ids, target_total_views=20, mode=args.mode)
    
    out_file = f"results/warp_experiment/{args.scene}_{args.mode}_fps_curve.npy"
    if not os.path.exists("results/warp_experiment"):
        os.makedirs("results/warp_experiment")
    np.save(out_file, results)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
