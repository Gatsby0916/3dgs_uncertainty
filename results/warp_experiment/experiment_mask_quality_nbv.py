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
        # Store as zero tensor to indicate "valid but empty" or failure
        # To distinguish failure from empty, we arguably don't care for uncertainty
        # If we can't project, we are uncertain? Or we know nothing?
        # If we know nothing, variance is 0 (all zeros).
        m = torch.zeros((warper.H_render, warper.W_render), device=warper.device, dtype=torch.float32)
    
    WARP_CACHE[key] = m
    return m

def calculate_uncertainty_score(warper, current_set_indices, candidate_idx):
    """
    Calculates the 'Mask Uncertainty' score for a candidate view.
    Score = Sum of Variance of projected masks from current_set.
    """
    masks = []
    for src_idx in current_set_indices:
        m = get_cached_warp_mask(warper, src_idx, candidate_idx)
        masks.append(m)

    if not masks:
        return 0.0

    # Stack: (N, H, W)
    if len(masks) == 1:
        return 0.0 # Variance is 0 for single view
        
    stack = torch.stack(masks)
    
    # Compute Variance (N, H, W) -> (H, W)
    # var = mean(x^2) - mean(x)^2 for bernoulli? 
    # Or just torch.var.
    # Note: masks are 0.0 or 1.0 (float).
    variance_map = torch.var(stack, dim=0, unbiased=False) 
    
    # Sum of variance = Total Uncertainty in this view given current knowledge
    score = variance_map.sum().item()
    return score

def run_iterative_selection(source_path, model_output_path, colmap_path, train_pool_ids, target_set_ids, max_steps=10, mode='random'):
    warper = WarpTool(source_path, model_output_path, colmap_path=colmap_path)
    
    # Start with 4 random views (fixed seed for consistency across modes if desired)
    # To compare Random vs NBV effectively, they should start with SAME 4 views.
    random.seed(42)
    
    valid_pool = [pid for pid in train_pool_ids if pid in warper.image_id_to_idx]
    if len(valid_pool) < 4:
        print(f"Error: Pool size {len(valid_pool)} too small.")
        return []
        
    # Initial set
    current_set_ids = random.sample(valid_pool, 4)
    remaining_pool_ids = [pid for pid in valid_pool if pid not in current_set_ids]
    
    results = [] # List of (num_views, mean_iou)
    
    print(f"[{mode.upper()}] Step 0: Initial Set (4 views): {current_set_ids}")
    
    # Evaluation helper
    def evaluate(view_set_ids, target_ids):
        ious = []
        # Warping for evaluation (Union)
        # We can also use cache here!
        for tgt_id in target_ids:
            if tgt_id not in warper.image_id_to_idx: continue
            tgt_idx = warper.image_id_to_idx[tgt_id]
            
            final_mask = None
            
            for src_id in view_set_ids:
                src_idx = warper.image_id_to_idx[src_id]
                m = get_cached_warp_mask(warper, src_idx, tgt_idx)
                
                if final_mask is None:
                    final_mask = m.clone()
                else:
                    final_mask = torch.max(final_mask, m)
            
            if final_mask is None:
                final_mask = torch.zeros((warper.H_render, warper.W_render), device=warper.device)
                
            # Load GT
            # We need to access internal method to load GT using WarpTool logic
            # run_experiment does this. Let's replicate or assume correctness.
            # WarpTool.load_mask uses name.
            tgt_info = warper.get_cam_info(tgt_id)
            gt_mask = warper.load_mask(tgt_info['name'])
            
            pred = final_mask.cpu().numpy()
            gt = gt_mask.cpu().numpy() if gt_mask is not None else None
            
            iou = calculate_iou_from_masks(pred, gt)
            ious.append(iou)
        return np.mean(ious)

    # Initial Eval
    mean_iou = evaluate(current_set_ids, target_set_ids)
    print(f"  -> Mean IoU: {mean_iou:.4f}")
    results.append((len(current_set_ids), mean_iou))
    
    for step in range(max_steps):
        if not remaining_pool_ids: break
        
        next_view_id = None
        
        if mode == 'random':
            next_view_id = random.choice(remaining_pool_ids)
            
        elif mode == 'nbv':
            # Score all candidates
            best_score = -1.0
            best_id_candidate = None
            
            # Helper indices
            current_indices = [warper.image_id_to_idx[sid] for sid in current_set_ids]
            
            # Iterate pool
            # This is the slow part
            for cand_id in remaining_pool_ids:
                cand_idx = warper.image_id_to_idx[cand_id]
                score = calculate_uncertainty_score(warper, current_indices, cand_idx)
                
                if score > best_score:
                    best_score = score
                    best_id_candidate = cand_id
                    
            next_view_id = best_id_candidate
            print(f"  NBV Selection: Best Score {best_score:.2f} for View {next_view_id}")
            
        if next_view_id is not None:
            current_set_ids.append(next_view_id)
            remaining_pool_ids.remove(next_view_id)
            
            print(f"Step {step+1}: Added view {next_view_id}. Total {len(current_set_ids)} views.")
            mean_iou = evaluate(current_set_ids, target_set_ids)
            print(f"  -> Mean IoU: {mean_iou:.4f}")
            results.append((len(current_set_ids), mean_iou))
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, choices=['basket', 'statue', 'torch', 'africa'])
    parser.add_argument("--mode", type=str, default="nbv", choices=['random', 'nbv'])
    args = parser.parse_args()
    
    # Config
    base_dir = f"LF/ours/{args.scene}"
    output_dir = f"LF/ours/{args.scene}/output"
    
    # Load camera IDs
    with open(os.path.join(output_dir, "cameras.json"), 'r') as f:
        cams = json.load(f)
        
    ids = [c['id'] for c in cams]
    ids.sort()
    
    # LLFF split: every 8th is test
    test_ids = ids[::8]
    train_ids = [i for i in ids if i not in test_ids]
    
    print(f"Scene {args.scene}, Mode {args.mode}")
    print(f"Train Pool: {len(train_ids)}, Test Set: {len(test_ids)}")
    
    results = run_iterative_selection(base_dir, output_dir, base_dir, train_ids, test_ids, max_steps=10, mode=args.mode)
    
    # Save
    out_file = f"results/warp_experiment/{args.scene}_{args.mode}_curve.npy"
    if not os.path.exists("results/warp_experiment"):
        os.makedirs("results/warp_experiment")
    np.save(out_file, results)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
