import os
import sys
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import random

# Reuse WarpTool
from warp_masks import WarpTool

def calculate_iou_from_masks(pred, gt):
    if pred is None or gt is None: return 0.0
    pred_b = (pred > 0.5)
    gt_b = (gt > 0.5)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return inter / union if union > 0 else 0.0

def run_iterative_selection(source_path, model_output_path, colmap_path, train_pool_ids, target_set_ids, max_steps=10):
    warper = WarpTool(source_path, model_output_path, colmap_path=colmap_path)
    
    # Start with 4 random views
    random.seed(42)
    # Ensure they are in the pool
    valid_pool = [pid for pid in train_pool_ids if pid in warper.image_id_to_idx]
    if len(valid_pool) < 4:
        print(f"Error: Pool size {len(valid_pool)} too small.")
        return []
        
    current_set = random.sample(valid_pool, 4)
    # Remove from pool
    remaining_pool = [pid for pid in valid_pool if pid not in current_set]
    
    results = [] # List of (num_views, mean_iou)
    
    # 1. Evaluate Initial Set
    print(f"Step 0: Initial Set (4 views): {current_set}")
    
    # Evaluation function
    def evaluate(view_set, target_set):
        ious = []
        # Suppress prints from warper
        # Use simple progress bar maybe?
        for tgt_id in target_set:
            if tgt_id not in warper.image_id_to_idx: continue
            tgt_idx = warper.image_id_to_idx[tgt_id]
            # warp_masks uses indices relative to sorted list
            src_indices = [warper.image_id_to_idx[sid] for sid in view_set]
            
            # Run warping silently
            # Redirect stdout to devnull
            sys.stdout = open(os.devnull, 'w')
            try:
                pred, gt = warper.run_experiment(src_indices, tgt_idx)
            finally:
                sys.stdout = sys.__stdout__
            
            iou = calculate_iou_from_masks(pred, gt)
            ious.append(iou)
        return np.mean(ious)

    mean_iou = evaluate(current_set, target_set_ids)
    print(f"  -> Mean IoU: {mean_iou:.4f}")
    results.append((len(current_set), mean_iou))
    
    for step in range(max_steps):
        # Select best next view
        best_candidate = None
        best_iou = -1.0
        
        # Greedy Selection: Try adding each candidate and see which one improves IoU the most on Target Set
        # NOTE: In real NBV we use uncertainty, but here we simulate an "Oracle" or ask if "Adding views helps"
        # Since calculating Uncertainty requires training, let's assume we are testing the "upper bound" or 
        # checking if adding views *randomly* vs *greedily* helps. 
        # The user asks: "Can increasing selected views help".
        # Let's do RANDOM selection addition for this script as a baseline to just see if curve goes up.
        # Adding 'Oracle' NBV (Greedy on Test Set) is cheating but shows potential.
        # Adding 'Random' is the baseline. 
        
        # Let's implementation Random Selection first as it's faster.
        # Just pick one random from pool
        if not remaining_pool: break
        
        next_view = random.choice(remaining_pool)
        current_set.append(next_view)
        remaining_pool.remove(next_view)
        
        print(f"Step {step+1}: Added view {next_view}. Total {len(current_set)} views.")
        mean_iou = evaluate(current_set, target_set_ids)
        print(f"  -> Mean IoU: {mean_iou:.4f}")
        results.append((len(current_set), mean_iou))
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, choices=['basket', 'statue', 'torch', 'africa'])
    args = parser.parse_args()
    
    # Config
    base_dir = f"LF/ours/{args.scene}"
    output_dir = f"LF/ours/{args.scene}/output"
    
    # Load camera IDs from cameras.json to distinguish train/test
    import json
    with open(os.path.join(output_dir, "cameras.json"), 'r') as f:
        cams = json.load(f)
        
    # Simple split: First 20 as train stats, next 5 as target
    # Or use standard split if available. 
    # Usually 3DGS puts train/test in structure, but cameras.json has all.
    # Let's just pick a subset.
    ids = [c['id'] for c in cams]
    
    # Sort IDs
    ids.sort()
    
    # Use every 8th as test (standard standard LLFF)
    test_ids = ids[::8]
    train_ids = [i for i in ids if i not in test_ids]
    
    print(f"Scene {args.scene}: {len(train_ids)} Train, {len(test_ids)} Test")
    
    results = run_iterative_selection(base_dir, output_dir, base_dir, train_ids, test_ids, max_steps=10)
    
    # Save
    out_file = f"results/warp_experiment/{args.scene}_random_curve.npy"
    np.save(out_file, results)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
