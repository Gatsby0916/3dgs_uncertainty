import os
import sys
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Add current directory to path so we can import scene.colmap_loader
sys.path.append(os.getcwd())

try:
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
except ImportError:
    print("Error: Could not import scene.colmap_loader. Make sure you are in the project root.")
    sys.exit(1)

def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def focal2fov(focal, pixels):
    return 2*np.arctan(pixels/(2*focal))

class WarpTool:
    def __init__(self, source_path, model_output_path, colmap_path=None, device='cuda', depth_path=None):
        self.source_path = source_path
        self.model_output_path = model_output_path
        self.colmap_path = colmap_path if colmap_path else source_path
        self.device = device
        self.depth_path = depth_path
        
        self.cameras = {}
        self.images = {}
        self.image_names = []
        
        # Load Colmap Data
        self._load_colmap()
        
        # Determine Resolution Scaling
        # We assume the depth/mask maps are 320x180 based on previous checks
        # But we should verify strictly or take as arg. 
        # For now, we will read one depth map to determine target resolution.
        sample_depth = self._find_first_depth()
        if sample_depth is not None:
            self.H_render, self.W_render = sample_depth.shape[-2:]
        else:
            print("Warning: No depth maps found. Defaulting to 180x320")
            self.H_render, self.W_render = 180, 320

    def _load_colmap(self):
        # Load cameras.json from model path
        json_path = os.path.join(self.model_output_path, "cameras.json")
        if not os.path.exists(json_path):
            print(f"Error: cameras.json not found at {json_path}")
            return

        import json
        with open(json_path, 'r') as f:
            self.cameras_json = json.load(f)
        
        # Create dict for fast lookup
        self.cameras_map = {cam['id']: cam for cam in self.cameras_json}
        
        # Sort images by ID (using the 'id' field in json)
        self.sorted_image_ids = sorted([cam['id'] for cam in self.cameras_json])
        self.image_id_to_idx = {img_id: i for i, img_id in enumerate(self.sorted_image_ids)}
        
        # Image names
        self.image_names = [self.cameras_map[i]['img_name'] for i in self.sorted_image_ids]

    def _find_depth_dir(self):
        if self.depth_path:
            return self.depth_path
        # Candidates
        candidates = [
            os.path.join(self.model_output_path, "depth"),
            os.path.join(self.model_output_path, "train", "ours_30000", "depth"),
            os.path.join(self.model_output_path, "test", "ours_30000", "depth"),
            os.path.join(self.model_output_path, "train", "ours_7000", "depth"),
        ]
        print(f"DEBUG: Searching for depth dir in: {candidates}")
        for d in candidates:
            if os.path.exists(d):
                print(f"DEBUG: Found depth dir at {d}")
                return d
        print("DEBUG: No depth dir found")
        return None

    def _find_first_depth(self):
        depth_dir = self._find_depth_dir()
        if not depth_dir:
            return None
        files = sorted(os.listdir(depth_dir))
        for f in files:
            if f.endswith('.npy'):
                return np.load(os.path.join(depth_dir, f))
        return None
    
    def load_depth(self, image_name):
        depth_dir = self._find_depth_dir()
        if not depth_dir:
            print(f"Error: No depth directory found.")
            return None
            
        basename = os.path.basename(image_name) 
        name_no_ext = os.path.splitext(basename)[0]
        
        path1 = os.path.join(depth_dir, name_no_ext + ".npy")
        if os.path.exists(path1):
             d = np.load(path1)
             # Assumption: Depth is stored as inverse depth (disparity) based on range analysis
             d = torch.tensor(d, device=self.device, dtype=torch.float32)
             mask = (d > 1e-6)
             d[mask] = 1.0 / d[mask]
             d[~mask] = 1000.0 # Far plane
             return d
             
        print(f"Warning: Depth file for {image_name} not found at {path1}")
        return None

    def get_cam_info(self, image_id):
        cam_data = self.cameras_map[image_id]
        
        # Load C2W directly
        R_c2w = np.array(cam_data['rotation'])
        T_c2w = np.array(cam_data['position'])
        
        c2w = torch.eye(4, device=self.device)
        c2w[:3, :3] = torch.tensor(R_c2w, device=self.device)
        c2w[:3, 3] = torch.tensor(T_c2w, device=self.device)
        
        # W2C is inverse
        w2c = torch.inverse(c2w)
        
        # Intrinsics
        fx = cam_data['fx']
        fy = cam_data['fy']
        
        # CX, CY are usually width/2, height/2 in 3DGS exports unless specified?
        # cameras.json DOES NOT have cx, cy. 
        # Standard 3DGS usually assumes center principle point.
        cx = cam_data['width'] / 2.0
        cy = cam_data['height'] / 2.0
            
        # Scale Intrinsics to Render Resolution
        scale_x = self.W_render / cam_data['width']
        scale_y = self.H_render / cam_data['height']

        # print(f"DEBUG: Image {cam_data['img_name']} JSON Size: {cam_data['width']}x{cam_data['height']} Render Size: {self.W_render}x{self.H_render} Scale: {scale_x:.3f}")
        
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y
        
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        return {
            'K': K,
            'w2c': w2c,
            'c2w': c2w,
            'position': cam_data.get('position', c2w[:3, 3].cpu().numpy()),
            'height': self.H_render,
            'width': self.W_render,
            'name': cam_data['img_name']
        }



    def load_mask(self, image_name):
        # Similar logic for mask
        # Path: LF/ours/basket/mask/
        mask_dir = os.path.join(self.source_path, "mask")
        basename = os.path.basename(image_name)
        name_no_ext = os.path.splitext(basename)[0]
        
        # Try exact match first
        path1 = os.path.join(mask_dir, name_no_ext + ".npy")
        # Try with _prob suffix
        path2 = os.path.join(mask_dir, name_no_ext + "_prob.npy")
        
        if os.path.exists(path1):
            m = np.load(path1)
        elif os.path.exists(path2):
            m = np.load(path2)
        else:
            print(f"Warning: Mask for {image_name} not found. Checked {path1} and {path2}")
            return None
            
        # Mask shape (1,1,H,W) or (H,W)
        m = torch.tensor(m, device=self.device, dtype=torch.float32)
        if m.ndim == 4: m = m[0,0]
        if m.ndim == 3: m = m[0]
        return m

    def unproject(self, depth, K, c2w):
        """
        Unproject depth map to 3D world points.
        depth: (H, W)
        K: (3, 3) 
        c2w: (4, 4)
        """
        H, W = depth.shape
        y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        
        x = x.flatten()
        y = y.flatten()
        z = depth.flatten()
        
        # Filter invalid depth
        mask = z > 0.001
        x = x[mask]
        y = y[mask]
        z = z[mask]
        
        # Pixel to Camera coords
        # X_cam = (x - cx) * z / fx
        # Y_cam = (y - cy) * z / fy
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        X_cam = (x - cx) * z / fx
        Y_cam = (y - cy) * z / fy
        Z_cam = z
        
        # Camera coords (N, 4)
        ones = torch.ones_like(z)
        pts_cam = torch.stack([X_cam, Y_cam, Z_cam, ones], dim=1)
        
        # Camera to World
        # pts_world = (c2w @ pts_cam.T).T
        pts_world = torch.matmul(c2w, pts_cam.T).T
        
        return pts_world[:, :3] # (N, 3)

    def project(self, pts_world, K, w2c, H, W):
        """
        Project 3D world points to target image plane.
        """
        N = pts_world.shape[0]
        ones = torch.ones((N, 1), device=self.device)
        pts_world_homo = torch.cat([pts_world, ones], dim=1)
        
        # World to Camera
        pts_cam = torch.matmul(w2c, pts_world_homo.T).T #(N, 4)
        
        features = pts_cam[:, 2:3] # Depth Z
        
        # Normalize
        pts_cam_xyz = pts_cam[:, :3]
        
        # Camera to Pixel
        # u = fx * X/Z + cx
        # v = fy * Y/Z + cy
        X = pts_cam[:, 0]
        Y = pts_cam[:, 1]
        Z = pts_cam[:, 2]
        
        # Filter behind camera
        valid_z = Z > 0.001
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        u = (X * fx / Z) + cx
        v = (Y * fy / Z) + cy
        
        u = u.long()
        v = v.long()
        
        # Check bounds
        valid_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        
        valid = valid_z & valid_bounds
        
        return u[valid], v[valid], Z[valid]

    def warp_mask_with_source(self, src_idx, tgt_idx, src_mask):
        # 1. Get Source Data
        src_id = self.sorted_image_ids[src_idx]
        src_info = self.get_cam_info(src_id)
        # Handle key mismatch (3DGS saves 'name' or 'img_name')
        img_name = src_info.get('img_name', src_info.get('name'))
        src_depth = self.load_depth(img_name)
        
        if src_depth is None or src_mask is None:
            return None
            
        # 2. Get Target Camera
        tgt_id = self.sorted_image_ids[tgt_idx]
        tgt_info = self.get_cam_info(tgt_id)
        
        # 3. Filter Source Depth by Source Mask
        # Only warp pixels that are part of the object
        # shape (H, W)
        object_depth = src_depth * (src_mask > 0.5).float()
        
        # 4. Unproject Source
        pts_world = self.unproject(object_depth, src_info['K'], src_info['c2w'])
        
        if pts_world.shape[0] == 0:
            return torch.zeros((self.H_render, self.W_render), device=self.device)
            
        # 5. Project to Target
        u, v, z = self.project(pts_world, tgt_info['K'], tgt_info['w2c'], self.H_render, self.W_render)
        
        # 6. Accumulate (Splat)
        warped_mask = torch.zeros((self.H_render, self.W_render), device=self.device)
        warped_mask[v, u] = 1.0
        
        return warped_mask

    def warp_mask(self, src_idx, tgt_idx):
        # 1. Get Source Data
        src_id = self.sorted_image_ids[src_idx]
        src_info = self.get_cam_info(src_id)
        src_depth = self.load_depth(src_info['img_name'])
        src_mask = self.load_mask(src_info['img_name'])
        
        if src_depth is None or src_mask is None:
            return None
            
        print(f"DEBUG: Src Mask shape {src_mask.shape}, sum {src_mask.sum()}")
        print(f"DEBUG: Src Depth shape {src_depth.shape}, range {src_depth.min()}-{src_depth.max()}, mean {src_depth.mean()}")

        # 2. Get Target Camera
        tgt_id = self.sorted_image_ids[tgt_idx]
        tgt_info = self.get_cam_info(tgt_id)
        
        # 3. Filter Source Depth by Source Mask
        # Only warp pixels that are part of the object
        # shape (H, W)
        object_depth = src_depth * (src_mask > 0.5).float()
        
        # 4. Unproject Source
        pts_world = self.unproject(object_depth, src_info['K'], src_info['c2w'])
        
        if pts_world.shape[0] == 0:
            return torch.zeros((self.H_render, self.W_render), device=self.device)
            
        # 5. Project to Target
        u, v, z = self.project(pts_world, tgt_info['K'], tgt_info['w2c'], self.H_render, self.W_render)
        
        # DEBUG SIZES
        if u.shape[0] > 0:
            print(f"DEBUG: Projected {u.shape[0]} points. U range: {u.min()}-{u.max()}, V range: {v.min()}-{v.max()}")
        else:
            print(f"DEBUG: Projected 0 points! pts_world stats: min={pts_world.min(0)[0].tolist()}, max={pts_world.max(0)[0].tolist()}")

        # 6. Accumulate (Splat)
        # We simply mark these pixels as 1
        warped_mask = torch.zeros((self.H_render, self.W_render), device=self.device)
        warped_mask[v, u] = 1.0
        
        return warped_mask

    def run_experiment(self, source_indices, target_index):
        print(f"Running Warp Experiment: Sources={source_indices} -> Target={target_index}")
        
        final_mask = torch.zeros((self.H_render, self.W_render), device=self.device)
        
        for src_idx in source_indices:
            print(f"  Warping from view {src_idx}...")
            w_mask = self.warp_mask(src_idx, target_index)
            if w_mask is not None:
                final_mask = torch.max(final_mask, w_mask)
        
        # Load GT for evaluation
        tgt_id = self.sorted_image_ids[target_index]
        tgt_img_name = self.cameras_map[tgt_id]['img_name']
        gt_mask = self.load_mask(tgt_img_name)
        
        return final_mask.cpu().numpy(), gt_mask.cpu().numpy() if gt_mask is not None else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help="Path to Data source containing 'mask' folder")
    parser.add_argument("--colmap_path", type=str, default=None, help="Path to COLMAP source containing 'sparse/0' (default: same as source_path)")
    parser.add_argument("--model_output", type=str, required=True, help="Path to 3DGS output (e.g. LF/ours/basket/output/train/ours_30000)")
    parser.add_argument("--src_views", type=int, nargs='+', default=[0,1,2,3], help="Indices of source views")
    parser.add_argument("--tgt_view", type=int, default=10, help="Index of target view to test")
    parser.add_argument("--output_vis", type=str, default="warp_visualization.png")
    parser.add_argument("--save_numpy", action='store_true', help="Save raw numpy arrays")
    args = parser.parse_args()
    
    warper = WarpTool(args.source_path, args.model_output, colmap_path=args.colmap_path)
    
    pred_mask, gt_mask = warper.run_experiment(args.src_views, args.tgt_view)
    
    if args.save_numpy:
        base_name = os.path.splitext(args.output_vis)[0]
        np.save(base_name + "_pred.npy", pred_mask)
        if gt_mask is not None:
            np.save(base_name + "_gt.npy", gt_mask)
    
    # Calculate IoU
    if gt_mask is not None:
        pred_binary = (pred_mask > 0.5)
        gt_binary = (gt_mask > 0.5)
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        iou = intersection / union if union > 0 else 0
        print(f"IoU: {iou:.4f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("Warped Mask (Union)")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')
    
    if gt_mask is not None:
        plt.subplot(1, 4, 2)
        plt.title("GT Mask")
        plt.imshow(gt_mask, cmap='gray')
        plt.axis('off')
        
        # Difference
        plt.subplot(1, 4, 3)
        plt.title("Difference")
        diff = np.ones((*pred_mask.shape, 3)) * 0.1 # Dark background for contrast
        
        # Colors (R, G, B)
        # Missed (GT=1, Pred=0) -> Soft Red/Coral [0.93, 0.46, 0.44]
        diff[np.logical_and(gt_binary, ~pred_binary)] = [0.93, 0.46, 0.44]
        
        # False Positive (GT=0, Pred=1) -> Soft Blue/Periwinkle [0.46, 0.64, 0.93]
        diff[np.logical_and(~gt_binary, pred_binary)] = [0.46, 0.64, 0.93]
        
        # Correct -> Soft Green/Sage [0.56, 0.76, 0.48]
        diff[np.logical_and(gt_binary, pred_binary)] = [0.56, 0.76, 0.48]
        
        # Background (Both 0) -> Very Dark Gray [0.1, 0.1, 0.1]
        
        plt.imshow(diff)
        plt.axis('off')

        # Overlay
        plt.subplot(1, 4, 4)
        plt.title(f"Overlay (IoU={iou:.2f})")
        plt.imshow(gt_mask, cmap='gray', alpha=0.5)
        plt.imshow(pred_mask, cmap='viridis', alpha=0.4) # Change jet to viridis (better default)
        plt.axis('off')
        
    plt.tight_layout(pad=0.5)
    plt.savefig(args.output_vis, bbox_inches='tight', dpi=100)
    print(f"Visualization saved to {args.output_vis}")

if __name__ == "__main__":
    main()
