#!/usr/bin/env python3
import os
import sys
import argparse
import json
import numpy as np
import torch
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm

# Add the directory containing warp_masks to path
sys.path.append(os.path.join(os.path.dirname(__file__), "results/warp_experiment"))
try:
    from warp_masks import WarpTool
except ImportError:
    # Try local import if moved
    try:
        from results.warp_experiment.warp_masks import WarpTool
    except ImportError:
        print("Error: Could not import WarpTool from results/warp_experiment/warp_masks.py")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="NBV Selection using Warped Masks")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset root (e.g., data/basket)")
    parser.add_argument("--model-output-path", required=True, help="Path to 3DGS output root")
    parser.add_argument("--uncert-dir", required=True, help="Path to rendered uncertainty maps (npz/png)")
    parser.add_argument("--depth-dir", required=True, help="Path to rendered depth maps")
    parser.add_argument("--mask-dir", required=True, help="Path to MASK directory (dataset/mask)")
    parser.add_argument("--train-split", required=True, help="Path to train_split.txt")
    parser.add_argument("--out-score", default="nbv_scores.json", help="Output JSON for scores")
    parser.add_argument("--device", default="cuda", help="Device to use")
    return parser.parse_args()

def load_train_split(path):
    with open(path, 'r') as f:
        # Extract IDs from filenames (e.g. "images/00001.jpg" -> 1)
        # Assuming typical 3DGS format or simple filenames
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines

def get_image_id(name):
    # Try to extract integer id
    basename = os.path.basename(name)
    name_no_ext = os.path.splitext(basename)[0]
    # Handle "00001" or "basket_00001"
    import re
    match = re.search(r'(\d+)', name_no_ext)
    if match:
        return int(match.group(1))
    return name_no_ext # Fallback string ID

def load_uncertainty(path, shape=None):
    # Load 2D uncertainty map
    # Supports .npz (keys: 'sigma', 'uncertainty_map') or .png
    if path.endswith('.npz'):
        d = np.load(path)
        if 'uncertainty_map' in d: return d['uncertainty_map']
        if 'sigma' in d: return d['sigma']
        return list(d.values())[0]
    elif path.endswith('.png'):
        u = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if u is None: return None
        return u.astype(np.float32) / 255.0
    return None

def aggregate_masks(masks):
    if not masks: return None
    stack = torch.stack(masks)
    mean = torch.mean(stack, dim=0)
    # Robust voting
    return (mean > 0.2).float()

def main():
    args = parse_args()
    
    # 1. Setup Warper
    # WarpTool expects a source_path that contains 'sparse' folder
    print(f"[NBV-Warp] Initializing WarpTool on {args.dataset_path}")
    warper = WarpTool(args.dataset_path, args.model_output_path, device=args.device, depth_path=args.depth_dir)
    
    # 2. Identify Pool and Candidates
    current_train_lines = load_train_split(args.train_split)
    pool_ids = []
    
    # Map filenames to IDs used by Warper (colmap image_id)
    # This is tricky because train_split usually has names, Warper uses IDs.
    # We will iterate Warper's camera map to match names.
    name_to_id = {}
    for iid, info in warper.cameras_map.items():
        name_to_id[info.get('img_name', info.get('name'))] = iid
        # Also try basename
        name_to_id[os.path.basename(info.get('img_name', info.get('name')))] = iid
        # Also try stem
        name_to_id[os.path.splitext(os.path.basename(info.get('img_name', info.get('name'))))[0]] = iid
    
    # Build Pool IDs
    for line in current_train_lines:
        fname = os.path.basename(line)
        if fname in name_to_id:
            pool_ids.append(name_to_id[fname])
        else:
            print(f"Warning: Could not map train image {line} to colmap ID")
            
    pool_ids = list(set(pool_ids))
    all_ids = list(warper.image_id_to_idx.keys())
    candidate_ids = [i for i in all_ids if i not in pool_ids]
    
    print(f"[NBV-Warp] Pool size: {len(pool_ids)}, Candidates: {len(candidate_ids)}")
    
    if not candidate_ids:
        print("No candidates left!")
        return

    # 3. Score Candidates
    scores = {}
    best_score = -1.0
    best_id = None
    best_mask = None # Tensor
    
    # Cache pool masks to avoid reloading
    pool_masks = {}
    for pid in pool_ids:
        info = warper.get_cam_info(pid)
        # Load from mask dir (which might contain Warped masks from prev steps)
        # We use Warper's internal load_mask which looks in dataset/mask
        # BUT we need to be careful: Warper.load_mask might be hardcoded. 
        # Let's verify Warper code or write a custom loader.
        # Warper.load_mask looks for .npy in source_path/mask. Correct.
        m = warper.load_mask(info['name']) 
        if m is not None:
             pool_masks[pid] = m
        else:
            print(f"Warning: Missing mask for pool item {pid}")

    print(f"[NBV-Warp] Loaded {len(pool_masks)} pool masks")

    # Iterate candidates
    for cid in tqdm(candidate_ids, desc="Scoring Candidates"):
        c_idx = warper.image_id_to_idx[cid]
        c_info = warper.get_cam_info(cid)
        c_name = os.path.basename(c_info['name'])
        name_no_ext = os.path.splitext(c_name)[0]
        
        # A. Generate Mask via Warping
        # Use simple depth check or load loaded depth?
        # Warper needs depth. Since we rendered ALL views, we should check args.depth_dir
        # Warper.find_depth usually looks in specific structures.
        # We can override or rely on Warper logic logic if depth_dir matches.
        # The Warper.load_depth is not fully exposed for arbitrary paths.
        # We will assume Warper's "find_depth_dir" finds the one we just trained!
        # WARNING: Warper might find OLD depth from previous iterations if not careful.
        # But Unified Pipeline passes a specific output dir to Render, and Warper looks in output/train...
        # Ideally, we should forcefully tell Warper where the depth is.
        # Currently Warper scans. We'll trust it finds the latest 'ours_X' folder depth.
        
        generated_masks = []
        for pid in pool_ids:
            if pid not in pool_masks: continue
            p_idx = warper.image_id_to_idx[pid]
            
            # Warp!
            # Note: This uses the CANDIDATE's depth (to unproject? No, source depth).
            # Warp Logic: Source (Pool) -> Target (Candidate).
            # We need Source Depth (Pool Depth).
            # Is Pool Depth available? Yes, we rendered ALL views including training views.
            
            c_idx_idx = warper.image_id_to_idx[c_idx]
            m = warper.warp_mask_with_source(p_idx, c_idx_idx, pool_masks[pid])
            if m is not None:
                generated_masks.append(m)
        
        if not generated_masks:
            scores[str(cid)] = 0.0
            continue
            
        final_mask = aggregate_masks(generated_masks) # Tensor (H, W)
        
        # B. Load Uncertainty
        # Look in args.uncert_dir
        # Filename pattern: "basket_00001_uncertainty.npz" or similar?
        # Render.py usually outputs: <scene>_<iter>/uncertainty_npz/<name>.npz
        
        # Try finding the file
        u_path = None
        # Pattern 1: exact match
        p1 = os.path.join(args.uncert_dir, name_no_ext + ".npz")
        p2 = os.path.join(args.uncert_dir, name_no_ext + "_uncertainty.npz")
        p3 = os.path.join(args.uncert_dir, name_no_ext + ".png") # Render might output png
        
        if os.path.exists(p1): u_path = p1
        elif os.path.exists(p2): u_path = p2
        elif os.path.exists(p3): u_path = p3
        
        if not u_path:
            # Try searching dir
            # print(f"Missing uncertainty for {c_name}")
            scores[str(cid)] = 0.0
            continue
            
        u_map = load_uncertainty(u_path)
        if u_map is None:
            scores[str(cid)] = 0.0
            continue
            
        u_tensor = torch.tensor(u_map, device=args.device, dtype=torch.float32)
        
        # Resize if needed
        if u_tensor.shape != final_mask.shape:
             import torch.nn.functional as F
             u_tensor = u_tensor.unsqueeze(0).unsqueeze(0) #(1,1,H,W)
             u_tensor = F.interpolate(u_tensor, size=final_mask.shape, mode='bilinear', align_corners=False)
             u_tensor = u_tensor.squeeze(0).squeeze(0)
             
        # C. Calculate OUGS
        # Sum of uncertainty within mask / Total Mask Area (Mean)
        # Or Sum? User said "Uncertainty score of the object". Usually Mean is safer against large masks.
        # But if we want to find "Most uncertain object", sum might be biased by size.
        # Let's use Mean.
        
        mask_sum = final_mask.sum()
        if mask_sum < 10.0: # Too small, ignore
             score = 0.0
        else:
             score = (u_tensor * final_mask).sum() / mask_sum
             score = score.item()
             
        scores[str(cid)] = score
        
        if score > best_score:
            best_score = score
            best_id = cid
            best_mask = final_mask

    # 4. Save and Update
    print(f"[NBV-Warp] Best View: {best_id} (Score: {best_score:.4f})")
    
    if best_id:
        # A. Save the Warped Mask to Dataset Mask Dir
        # Convert to numpy 0-1
        mask_np = best_mask.cpu().numpy().astype(np.uint8)
        
        info = warper.get_cam_info(best_id)
        best_name = os.path.basename(info.get('img_name', info.get('name')))
        name_no_ext = os.path.splitext(best_name)[0]
        
        save_path = os.path.join(args.mask_dir, name_no_ext + ".npy")
        np.save(save_path, mask_np)
        print(f"[NBV-Warp] Saved Warped Mask to {save_path}")
        
        # B. Update Train Split
        # Need to append the RELATIVE path usually found in split file
        # We need to construct the line exactly as expected.
        # Usually: "images/00001.jpg"
        # We have the full path or name from Warper.
        # Warper info['name'] is usually "images/..." or just "..."
        # We will retrieve the original string from warper info if possible, or construct it.
        # Warper cameras.json usually has "img_name": "00001" or path.
        
        # We will read 3DGS dataset standard structure if needed, or just append the filename
        # But train_split usually matches format of gen_split output.
        # Let's verify what 'info['name']' contains. 
        # In 3DGS, it is usually the image name.
        
        # Let's trust info['name'] matches the relative path style.
        # FIX: train_split.txt expects STEMS (no extension)
        full_name = info.get('img_name', info.get('name'))
        new_line = os.path.splitext(os.path.basename(full_name))[0]
        
        with open(args.train_split, 'a') as f:
            f.write(f"\n{new_line}")
        print(f"[NBV-Warp] Updated {args.train_split}")
        
    # Save scores
    with open(args.out_score, 'w') as f:
        json.dump(scores, f, indent=4)

if __name__ == "__main__":
    main()
