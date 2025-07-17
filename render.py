#!/usr/bin/env python3
# ==============================================================
# 3D Gaussian Splatting – Batch Render & Uncertainty Pipeline
#   • 文件保存名 = view.image_name 之 stem（原机名），避免重命名冲突
#   • 若 view 无 image_name 字段，则回退 idx:05d
#   • 其余参数、目录结构与官方 render.py 一致
# ==============================================================

import os, numpy as np, torch, torchvision
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render, estimate_uncertainty
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


# ---------- helper ----------
def stem_of(view, idx: int) -> str:
    """优先取 view.image_name / image_path stem；否则 idx 5 位数字"""
    name = getattr(view, "image_name", None)
    if name is None and hasattr(view, "image_path"):
        name = Path(view.image_path).name
    return (Path(name).stem if name else f"{idx:05d}").lower()


# ---------- main render set ----------
def render_set(model_path, name, iteration, views, gaussians, pipeline,
               background, train_test_exp, separate_sh,
               *, uncertainty_mode=False, patch_size=8,
               max_views=0, view_ids=None, mask_dir=None):



    # --------- 掩码准备（仅 uncertainty 模式才尝试） ----------
    use_mask = uncertainty_mode and mask_dir is not None
    if use_mask:
        mask_dir = os.path.normpath(mask_dir)
        if not os.path.isdir(mask_dir):
            print(f"[WARN] mask_dir={mask_dir} does not exist, disabling mask usage.")
            use_mask = False

    base_dir            = os.path.join(model_path, name, f"ours_{iteration}")
    render_dir          = os.path.join(base_dir, "renders")
    gt_dir              = os.path.join(base_dir, "gt")
    uncertainty_dir     = os.path.join(base_dir, "uncertainty")
    uncertainty_npz_dir = os.path.join(base_dir, "uncertainty_npz")
    depth_dir           = os.path.join(base_dir, "depth")
    for d in (render_dir, gt_dir, uncertainty_dir, uncertainty_npz_dir, depth_dir):
        os.makedirs(d, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"{name}-views")):
        if view_ids is not None and idx not in view_ids:
            continue
        if max_views and idx >= max_views:
            break

        stem = stem_of(view, idx)

        # Load Mask if Mask Directory is Provided
        mask_prob = None  # Initialize mask_prob as None before loading
        if use_mask:
            mask_path = os.path.join(mask_dir, f"{stem}_prob.npy")
            if os.path.exists(mask_path):
                mask_prob = np.load(mask_path)

        
        # ---------- render ----------
        if uncertainty_mode:
            with torch.enable_grad():
                out = estimate_uncertainty(
                    viewpoint_camera = view,
                    pc               = gaussians,
                    pipe             = pipeline,
                    bg_color         = background,
                    scaling_modifier = 1.0,
                    separate_sh      = separate_sh,
                    use_trained_exp  = train_test_exp,
                    patch_size       = patch_size,
                    return_raw       = True,
                    mask_prob        = mask_prob
                )
        else:
            with torch.no_grad():
                out = render(view, gaussians, pipeline, background,
                             use_trained_exp=train_test_exp, separate_sh=separate_sh)

        rendering   = out["render"]
        uncertainty = out.get("uncertainty")
        raw_uncert  = out.get("uncertainty_raw")
        depth_map   = out.get("depth")

        gt = view.original_image[0:3]
        if train_test_exp:
            W_half = rendering.shape[-1] // 2
            rendering, gt = rendering[...,W_half:], gt[...,W_half:]
            if uncertainty is not None: uncertainty = uncertainty[...,W_half:]
            if raw_uncert  is not None: raw_uncert  = raw_uncert[...,W_half:]

        # ---------- save ----------
        torchvision.utils.save_image(rendering.clamp(0,1),
                                     os.path.join(render_dir, f"{stem}.png"))
        torchvision.utils.save_image(gt.clamp(0,1),
                                     os.path.join(gt_dir, f"{stem}.png"))
        if uncertainty is not None:
            torchvision.utils.save_image(uncertainty.clamp(0,1),
                                         os.path.join(uncertainty_dir, f"{stem}.png"))
        if raw_uncert is not None:
            np.savez(os.path.join(uncertainty_npz_dir, f"{stem}.npz"),
                     uncertainty_map = raw_uncert.cpu().numpy(),
                     pixel_gaussian_counter = np.ones_like(raw_uncert.cpu(), dtype=np.int32))
        if depth_map is not None:
            np.save(os.path.join(depth_dir, f"{stem}.npy"),
                    depth_map.squeeze().cpu().numpy())

        # 清理显存
        del rendering, gt, uncertainty, raw_uncert, depth_map
        torch.cuda.empty_cache()


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, separate_sh: bool,
                *, uncertainty_mode: bool, patch_size: int,
                max_views: int = 0, view_ids=None, mask_dir=None):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene     = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg        = torch.tensor([1,1,1] if dataset.white_background else [0,0,0],
                                 dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline, bg,
                       dataset.train_test_exp, separate_sh,
                       uncertainty_mode=uncertainty_mode, patch_size=patch_size,
                       max_views=max_views, view_ids=view_ids, mask_dir=mask_dir)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline, bg,
                       dataset.train_test_exp, separate_sh,
                       uncertainty_mode=uncertainty_mode, patch_size=patch_size,
                       max_views=max_views, view_ids=view_ids, mask_dir=mask_dir)



# ------------------------- CLI ------------------------- #
if __name__ == "__main__":
    parser = ArgumentParser("3DGS 渲染脚本（含不确定性与深度输出）")
    model    = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    # 添加 --mask-dir 参数用于传递掩码目录
    parser.add_argument("--mask-dir", type=str, default=None, help="folder with *_prob.npy masks")

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test",  action="store_true")
    parser.add_argument("--quiet",      action="store_true")
    parser.add_argument("--uncertainty_mode", action="store_true")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--max_views",  type=int, default=0)
    parser.add_argument("--view_ids",   default="")
    args = get_combined_args(parser)
    


    view_ids = None
    if args.view_ids.strip():
        view_ids = sorted({int(v) for v in args.view_ids.split(",")})

    safe_state(args.quiet)
    mask_dir_pass = args.mask_dir if args.uncertainty_mode else None

    # 调用 render_sets 函数时，传递 mask-dir 参数
    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE,
                uncertainty_mode=args.uncertainty_mode, patch_size=args.patch_size,
                max_views=args.max_views, view_ids=view_ids, mask_dir=mask_dir_pass)
