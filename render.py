#!/usr/bin/env python3
# ==============================================================  
# 3D Gaussian Splatting – Batch Render & Uncertainty Pipeline  
#   • 支持 --max_views 限制渲染数量  
#   • 支持 --view_ids 精确指定要渲染的视角索引（0-based, 逗号分隔）  
#   • 默认保存深度图（.npy）、不确定性热图（.png）与原始不确定性数据（.npz）  
# ==============================================================  

import os
import numpy as np
import torch
import torchvision
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


def render_set(model_path,
               name,
               iteration,
               views,
               gaussians,
               pipeline,
               background,
               train_test_exp,
               separate_sh,
               *,
               uncertainty_mode=False,
               patch_size=8,
               top_k=15000,
               max_views: int = 0,
               view_ids: list[int] | None = None):
    """
    渲染一个视图集合（train 或 test），并保存：
      - RGB 渲染 (.png)
      - Ground-truth (.png)
      - 深度图 (.npy)
      - 不确定性热图 (.png)
      - 原始不确定性数据 (.npz)
    """
    base_dir            = os.path.join(model_path, name, f"ours_{iteration}")
    render_dir          = os.path.join(base_dir, "renders")
    gt_dir              = os.path.join(base_dir, "gt")
    uncertainty_dir     = os.path.join(base_dir, "uncertainty")
    uncertainty_npz_dir = os.path.join(base_dir, "uncertainty_npz")
    depth_dir           = os.path.join(base_dir, "depth")

    os.makedirs(render_dir,      exist_ok=True)
    os.makedirs(gt_dir,          exist_ok=True)
    os.makedirs(uncertainty_dir, exist_ok=True)
    os.makedirs(uncertainty_npz_dir, exist_ok=True)
    os.makedirs(depth_dir,       exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"{name}-views")):
        # 跳过不在 view_ids 列表内的索引
        if view_ids is not None and idx not in view_ids:
            continue
        # 达到上限即退出
        if max_views and idx >= max_views:
            break

        # 渲染（可选不确定性）
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
                    top_k            = top_k,
                    return_raw       = True   # 返回原始不确定性数据
                )
        else:
            with torch.no_grad():
                out = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=train_test_exp,
                    separate_sh=separate_sh
                )

        rendering   = out["render"]                          # [3, H, W]
        uncertainty = out.get("uncertainty", None)           # [3, H, W] 或 None
        raw_uncert  = out.get("uncertainty_raw", None)       # [H, W] 或 None
        depth_map   = out.get("depth", None)                 # [H, W] 或 None

        # 若 train_test_exp，截取后半图
        gt = view.original_image[0:3]
        if train_test_exp:
            W_half = rendering.shape[-1] // 2
            rendering   = rendering[..., W_half:]
            gt          = gt[...,        W_half:]
            if uncertainty is not None:
                uncertainty = uncertainty[..., W_half:]
            if raw_uncert is not None:
                raw_uncert = raw_uncert[..., W_half:]

        # 保存 RGB 渲染
        torchvision.utils.save_image(
            rendering.clamp(0,1),
            os.path.join(render_dir, f"{idx:05d}.png")
        )
        # 保存 Ground-truth
        torchvision.utils.save_image(
            gt.clamp(0,1),
            os.path.join(gt_dir, f"{idx:05d}.png")
        )
        # 保存不确定性热图
        if uncertainty is not None:
            torchvision.utils.save_image(
                uncertainty.clamp(0,1),
                os.path.join(uncertainty_dir, f"{idx:05d}.png")
            )
        # 保存原始不确定性数据 (.npz)
        if raw_uncert is not None:
            u_np = raw_uncert.cpu().numpy()
            counter = np.ones_like(u_np, dtype=np.int32)  # 占位计数器，可根据实际需求替换
            np.savez(
                os.path.join(uncertainty_npz_dir, f"{idx:05d}.npz"),
                uncertainty_map=u_np,
                pixel_gaussian_counter=counter
            )
        # 保存深度图（numpy .npy）
        if depth_map is not None:
            arr = depth_map.squeeze().cpu().numpy()
            np.save(
                os.path.join(depth_dir, f"{idx:05d}.npy"),
                arr
            )

        # 释放显存
        del rendering, gt
        if uncertainty is not None:
            del uncertainty
        if raw_uncert is not None:
            del raw_uncert
        if depth_map is not None:
            del depth_map
        torch.cuda.empty_cache()


def render_sets(dataset: ModelParams,
                iteration: int,
                pipeline: PipelineParams,
                skip_train: bool,
                skip_test: bool,
                separate_sh: bool,
                *,
                uncertainty_mode: bool,
                patch_size: int,
                top_k: int,
                max_views: int = 0,
                view_ids: list[int] | None = None):
    """
    渲染 train / test 两个集合
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene     = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color   = [1,1,1] if dataset.white_background else [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path, "train", scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians, pipeline, background, dataset.train_test_exp,
                separate_sh,
                uncertainty_mode=uncertainty_mode,
                patch_size=patch_size, top_k=top_k,
                max_views=max_views, view_ids=view_ids
            )
        if not skip_test:
            render_set(
                dataset.model_path, "test", scene.loaded_iter,
                scene.getTestCameras(),
                gaussians, pipeline, background, dataset.train_test_exp,
                separate_sh,
                uncertainty_mode=uncertainty_mode,
                patch_size=patch_size, top_k=top_k,
                max_views=max_views, view_ids=view_ids
            )


if __name__ == "__main__":
    parser = ArgumentParser("3DGS 渲染脚本（含不确定性与深度输出）")
    model    = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration",  type=int, default=-1,
                        help="加载的 checkpoint 迭代次数（-1=最新）")
    parser.add_argument("--skip_train", action="store_true",
                        help="跳过训练集渲染")
    parser.add_argument("--skip_test",  action="store_true",
                        help="跳过测试集渲染")
    parser.add_argument("--quiet",      action="store_true",
                        help="安静模式，少打印日志")

    # 不确定性渲染开关
    parser.add_argument("--uncertainty_mode", action="store_true",
                        help="渲染并保存不确定性热图及原始数据")
    parser.add_argument("--patch_size",       type=int, default=8,
                        help="不确定性补丁尺寸")
    parser.add_argument("--top_k",            type=int, default=15000,
                        help="保留 top-k 补丁计算梯度")

    # 渲染视角控制
    parser.add_argument("--max_views", type=int, default=0,
                        help="渲染的最多视图数（0=无限制）")
    parser.add_argument("--view_ids",  default="",
                        help="逗号分隔的视角索引列表（0-based）")

    args = get_combined_args(parser)
    print("Render model path:", args.model_path)

    # 解析 view_ids
    view_ids = None
    if args.view_ids.strip():
        view_ids = sorted({int(v) for v in args.view_ids.split(",")})

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
        uncertainty_mode=args.uncertainty_mode,
        patch_size=args.patch_size,
        top_k=args.top_k,
        max_views=args.max_views,
        view_ids=view_ids
    )
