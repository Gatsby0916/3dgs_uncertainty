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

def calculate_iou(pred, gt):
    if pred is None or gt is None: return 0.0
    pred_b = (pred > 0.5)
    gt_b = (gt > 0.5)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return inter / union if union > 0 else 0.0

def warp_mask_with_source(warper, src_idx, tgt_idx, src_mask_tensor):
    """
    Custom warp function that accepts a source mask tensor directly
    instead of loading from disk. 
    src_mask_tensor: (H, W) or (1, H, W) on device
    """
    # 1. Get Source Info
    src_id = warper.sorted_image_ids[src_idx]
    src_info = warper.get_cam_info(src_id)
    
    # Load geometric depth (always from disk)
    src_depth = warper.load_depth(src_info['name'])
    
    if src_depth is None:
        return None
        
    # Ensure dimensions match
    if src_mask_tensor.ndim == 3:
        src_mask_tensor = src_mask_tensor.squeeze(0)
        
    # 2. Filter Depth
    # src_depth is (H, W)
    object_depth = src_depth * (src_mask_tensor > 0.5).float()
    
    # 3. Unproject
    pts_world = warper.unproject(object_depth, src_info['K'], src_info['c2w'])
    
    if pts_world.shape[0] == 0:
        return torch.zeros((warper.H_render, warper.W_render), device=warper.device)
        
    # 4. Project to Target
    tgt_id = warper.sorted_image_ids[tgt_idx]
    tgt_info = warper.get_cam_info(tgt_id)
    
    u, v, z = warper.project(pts_world, tgt_info['K'], tgt_info['w2c'], warper.H_render, warper.W_render)
    
    # 5. Accumulate
    warped_mask = torch.zeros((warper.H_render, warper.W_render), device=warper.device)
    warped_mask[v, u] = 1.0
    
    return warped_mask

def aggregate_masks(masks_list):
    """
    Robust aggregation for propagation.
    Voting Strategy: If > 1 view sees it (sum > 1.0), keep it.
    Or Mean > threshold. 
    To be conservative against noise accumulation: sum >= 1 (Union) is bad.
    Let's try: Mean >= 0.2 (Soft Voting)
    """
    if not masks_list: return None
    stack = torch.stack(masks_list)
    # Mean (0..1)
    mean_mask = torch.mean(stack, dim=0)
    # Binary
    return (mean_mask > 0.2).float() # Slightly conservative

def load_uncertainty_map(warper, image_name):
    # Path: LF/ours/basket/object_uncertainty_png/basket_00001_object_uncertainty.png
    basename = os.path.basename(image_name)
    name_no_ext = os.path.splitext(basename)[0]
    
    uncert_dir = os.path.join(warper.source_path, "object_uncertainty_png")
    
    # Try multiple formats
    # 1. Exact match
    f1 = f"{name_no_ext}_object_uncertainty.png"
    # 2. With scene prefix (e.g. basket_00001...)
    f2 = f"basket_{name_no_ext}_object_uncertainty.png"
    # 3. Maybe just the number?
    
    for f in [f1, f2]:
        path = os.path.join(uncert_dir, f)
        # print(f"DEBUG: Checking {path}") # Uncomment to debug
        if os.path.exists(path):
            import cv2
            u = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if u is None: continue
            u = torch.tensor(u, device=warper.device, dtype=torch.float32) / 255.0
            return u
            
    # Fallback: Extract number
    import re
    match = re.search(r'(\d{4,5})', name_no_ext)
    if match:
        num = match.group(1)
        f3 = f"basket_{num}_object_uncertainty.png"
        path = os.path.join(uncert_dir, f3)
        if os.path.exists(path):
             import cv2
             u = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
             if u is not None:
                u = torch.tensor(u, device=warper.device, dtype=torch.float32) / 255.0
                return u
    
    # if np.random.rand() < 0.01: # Print rarely to avoid clutter
    #     print(f"DEBUG: Uncertainty map not found. Tried {f1}, {f2} in {uncert_dir}")
    return None

def calculate_uncertainty(warper, current_indices, cand_idx, propagated_masks):
    masks = []
    # Suppress output
    sys.stdout = open(os.devnull, 'w')
    cand_id = warper.sorted_image_ids[cand_idx]
    
    # 1. Generate Mask for Candidate using CURRENT POOL
    # Note: We need to warp FROM pool TO candidate
    # This matches the previous logic: warp_mask_with_source(src, tgt, src_mask)
    
    generated_mask_list = []
    try:
        for src_idx in current_indices:
            src_id = warper.sorted_image_ids[src_idx]
            src_mask = propagated_masks.get(src_id)
            if src_mask is None: continue
            
            m = warp_mask_with_source(warper, src_idx, cand_idx, src_mask)
            if m is not None: generated_mask_list.append(m)
    except Exception:
        pass
    finally:
        sys.stdout = sys.__stdout__
        
    if not generated_mask_list: return 0.0
    
    # Aggregate mask
    final_mask = aggregate_masks(generated_mask_list)
    
    # 2. Load Uncertainty Map (Pre-computed OUGS)
    info = warper.get_cam_info(cand_id)
    u_map = load_uncertainty_map(warper, info['name'])
    
    if u_map is None:
        # Fallback to Variance if map missing (or return 0)
        # return 0.0
        # Actually let's fallback to variance of the warped masks if OUGS is missing
        # But user insists on OUGS.
        return 0.0
        
    if u_map.shape != final_mask.shape:
        # Resize u_map to match final_mask
        import torch.nn.functional as F
        # Add batch/channel dims: (H,W) -> (1,1,H,W)
        u_in = u_map.unsqueeze(0).unsqueeze(0)
        u_out = F.interpolate(u_in, size=final_mask.shape, mode='bilinear', align_corners=False)
        u_map = u_out.squeeze(0).squeeze(0)
        
    # 3. Calculate Mean Uncertainty inside Mask
    # OUGS = Mean(Uncertainty * Mask) / Sum(Mask) ? 
    # Usually it's Sum or Mean. 
    # "Uncertainty score of the object" -> usually sum or mean.
    # If mask is empty, 0.
    
    mask_sum = final_mask.sum()
    if mask_sum < 1.0: return 0.0
    
    score = (u_map * final_mask).sum() / mask_sum
    return score.item()

def get_fps_indices(warper, pool_ids, n=4):
    positions = []
    valid_pool_ids = []
    
    for pid in pool_ids:
        if pid not in warper.image_id_to_idx: continue
        info = warper.get_cam_info(pid)
        if 'position' in info:
            pos = np.array(info['position'])
        else:
            # Recompute pos if missng
            c2w = info['c2w'].cpu().numpy()
            pos = c2w[:3, 3] # Approx
        positions.append(pos)
        valid_pool_ids.append(pid)
        
    if len(positions) < n: return valid_pool_ids
    
    # First item
    selected_indices = [0]
    selected_ids = [valid_pool_ids[0]]
    picked_centers = [positions[0]]
    
    for _ in range(n - 1):
        max_dist = -1
        best_r_idx = -1
        for r_idx, center in enumerate(positions):
            if r_idx in selected_indices: continue
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

def run_propagation_experiment(scene, mode='nbv', target_views=20):
    output_dir = f"LF/ours/{scene}/output"
    warper = WarpTool(f"LF/ours/{scene}", output_dir, colmap_path=f"LF/ours/{scene}")
    
    with open(os.path.join(output_dir, "cameras.json"), 'r') as f:
        cams = json.load(f)
    ids = [c['id'] for c in cams]
    ids.sort()
    
    test_ids = ids[::8]
    train_ids = [i for i in ids if i not in test_ids]
    
    # 1. Initialize with FPS using GT Masks
    init_ids = get_fps_indices(warper, train_ids, n=4)
    
    propagated_masks = {} # Memory Store for masks
    
    print(f"--- Initialization (4 Views GT) ---")
    for pid in init_ids:
        info = warper.get_cam_info(pid)
        gt_npy = warper.load_mask(info['name'])
        t = torch.tensor(gt_npy, device=warper.device, dtype=torch.float32)
        if t.ndim == 3: t = t.squeeze(0)
        propagated_masks[pid] = t
        
    current_set_ids = list(init_ids)
    remaining = [i for i in train_ids if i not in current_set_ids and i in warper.image_id_to_idx]
    
    # [(total_views, new_view_iou, avg_accum_iou)]
    results = [(len(current_set_ids), 1.0, 1.0)] 
    
    steps = target_views - len(current_set_ids)
    
    for step in range(steps):
        if not remaining: break
        
        # A. Choose Next View based on Uncertainty of *Propagated* Masks
        next_id = None
        
        if mode == 'random':
            next_id = random.choice(remaining)
        elif mode == 'nbv':
            best_score = -1.0
            best_cand = None
            
            curr_indices = [warper.image_id_to_idx[uid] for uid in current_set_ids]
            
            # Subsample candidates to speed up (every 5th?) if needed
            # For ~150 views, full scan takes ~20-30s. Acceptable.
            for cand_id in remaining:
                cand_idx = warper.image_id_to_idx[cand_id]
                score = calculate_uncertainty(warper, curr_indices, cand_idx, propagated_masks)
                if score > best_score:
                    best_score = score
                    best_cand = cand_id
            next_id = best_cand
            print(f"  [NBV] Highest Uncertainty Score: {best_score:.2f} -> View {next_id}")

        if next_id is None: break
        
        # B. "Replace SAM2": Generate Mask for this chosen view using Warping
        next_idx = warper.image_id_to_idx[next_id]
        projections = []
        
        # Warp all EXISTING masks to this new view to define it
        sys.stdout = open(os.devnull, 'w')
        try:
            for src_id in current_set_ids:
                src_idx = warper.image_id_to_idx[src_id]
                src_mask = propagated_masks[src_id]
                m = warp_mask_with_source(warper, src_idx, next_idx, src_mask)
                if m is not None: projections.append(m)
        finally:
            sys.stdout = sys.__stdout__
            
        generated_mask = aggregate_masks(projections)
        if generated_mask is None:
             generated_mask = torch.zeros((warper.H_render, warper.W_render), device=warper.device)
             
        # C. Evaluate This New Mask (Virtual vs GT)
        # This answers: "How good is the mask we just automatically labelled?"
        info_tgt = warper.get_cam_info(next_id)
        gt_mask = warper.load_mask(info_tgt['name'])
        
        # Ensure mask is cpu numpy
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
            
        iou_new = calculate_iou(generated_mask.cpu().numpy(), gt_mask)
        
        # D. Add to set (Propagate)
        propagated_masks[next_id] = generated_mask
        current_set_ids.append(next_id)
        remaining.remove(next_id)
        
        # Track average quality of generated masks in the set
        # AND Calculate Average Warp Quality for ALL Remaining Candidates using NEW Pool
        # This answers: "Does adding this view improve mask quality for the rest?"
        
        # 1. Intra-set quality
        gen_ious = []
        for pid in current_set_ids:
            if pid in init_ids: continue
            pred = propagated_masks[pid].cpu().numpy()
            info = warper.get_cam_info(pid)
            gt = warper.load_mask(info['name'])
            if isinstance(gt, torch.Tensor): gt = gt.cpu().numpy()
            gen_ious.append(calculate_iou(pred, gt))
        avg_iou_set = np.mean(gen_ious) if gen_ious else 1.0
        
        # 2. Remaining set quality (Sampled for speed, e.g. 20 random from remaining)
        rem_ious = []
        check_candidates = remaining if len(remaining) < 20 else np.random.choice(list(remaining), 20, replace=False)
        
        # Need to suppress print for this loop
        sys.stdout = open(os.devnull, 'w')
        try:
            for rid in check_candidates:
                # Generate mask for rid using CURRENT pool
                masks = []
                r_idx = warper.image_id_to_idx[rid]
                for src_id in current_set_ids:
                    src_idx = warper.image_id_to_idx[src_id]
                    src_mask = propagated_masks[src_id]
                    m = warp_mask_with_source(warper, src_idx, r_idx, src_mask)
                    if m is not None: masks.append(m)
                
                if masks:
                    final_mask = aggregate_masks(masks)
                    gt_r = warper.load_mask(warper.get_cam_info(rid)['name'])
                    if isinstance(gt_r, torch.Tensor): gt_r = gt_r.cpu().numpy()
                    rem_ious.append(calculate_iou(final_mask.cpu().numpy(), gt_r))
        except Exception:
            pass
        finally:
             sys.stdout = sys.__stdout__
             
        avg_iou_rem = np.mean(rem_ious) if rem_ious else 0.0
        
        print(f"Step {step+1}: Added {next_id}. Mask IoU: {iou_new:.4f}. Set Avg: {avg_iou_set:.4f}. Rem Avg: {avg_iou_rem:.4f}")
        results.append((len(current_set_ids), iou_new, avg_iou_set, avg_iou_rem))
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=['random', 'nbv'])
    args = parser.parse_args()
    
    print(f"Start: {args.scene} - {args.mode}")
    res = run_propagation_experiment(args.scene, mode=args.mode)
    
    out = f"results/warp_experiment/{args.scene}_{args.mode}_prop_curve.npy"
    if not os.path.exists("results/warp_experiment"): os.makedirs("results/warp_experiment")
    # Save as object array to avoid inhomogeneous shape error
    np.save(out, np.array(res, dtype=object))
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
