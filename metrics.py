#!/usr/bin/env python3
# ==============================================================
# 3DGS Reconstruction Quality Evaluation
#   -- 计算 PSNR / SSIM / LPIPS（单场景或多场景批量）
#   -- 支持评估 train/ 与 test/ 子目录
#   -- GPU 加速（默认使用 cuda:0）
# ==============================================================
#
# Copyright (C) 2023-2025, Inria & Contributors
# All rights reserved – research & evaluation license, see LICENSE.md
# Contact: george.drettakis@inria.fr
#

from pathlib import Path
import os
import json
from argparse import ArgumentParser

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

from utils.loss_utils import ssim           # 原 3DGS SSIM (torch.cuda)
from lpipsPyTorch import lpips              # 需 pip install lpipsPyTorch
from utils.image_utils import psnr          # 原 3DGS PSNR (torch.cuda)

# ------------------------- I/O Helpers ------------------------- #

def read_images(renders_dir: Path, gt_dir: Path):
    """
    读取两个文件夹下同名 PNG/JPG 图片并转为 CUDA Tensor。
    返回：list[tensor], list[tensor], list[str]
    """
    exts = {".png", ".jpg", ".jpeg"}
    names = [f for f in os.listdir(renders_dir) if Path(f).suffix.lower() in exts]
    names.sort()

    if len(names) == 0:
        raise RuntimeError(f"No images with extension {exts} found in {renders_dir}")

    renders, gts = [], []
    for fname in names:
        rend = Image.open(renders_dir / fname)
        gt   = Image.open(gt_dir      / fname)

        # to_tensor → C×H×W, keep RGB only
        rend_t = tf.to_tensor(rend).unsqueeze(0)[:, :3, :, :].cuda(non_blocking=True)
        gt_t   = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda(non_blocking=True)

        renders.append(rend_t)
        gts.append(gt_t)
    return renders, gts, names

# ------------------------- Core Evaluation ------------------------- #

def evaluate(scene_dirs, split="test"):
    """
    scene_dirs: list[str] – 每个路径下应含 train/ 或 test/ 子目录
    split     : 'train' or 'test'
    """
    full_dict, per_view_dict = {}, {}

    for scene_dir in scene_dirs:
        print("\nScene:", scene_dir)
        scene_dir = Path(scene_dir)
        split_dir = scene_dir / split
        if not split_dir.exists():                   # 若指定 split 不存在，则自动 fallback
            alt = scene_dir / ("test" if split == "train" else "train")
            if alt.exists():
                print(f"[Info] '{split_dir.name}' not found, fallback to '{alt.name}'.")
                split_dir = alt
            else:
                print(f"[Warning] Neither train/ nor test/ exist in {scene_dir}; skip.")
                continue

        full_dict[str(scene_dir)] = {}
        per_view_dict[str(scene_dir)] = {}

        for method in os.listdir(split_dir):
            method_dir = split_dir / method
            gt_dir      = method_dir / "gt"
            renders_dir = method_dir / "renders"
            if not (gt_dir.exists() and renders_dir.exists()):
                print(f"  [Skip] {method}: missing gt/ or renders/")
                continue

            print("Method:", method)
            try:
                renders, gts, img_names = read_images(renders_dir, gt_dir)
            except RuntimeError as e:
                print(" ", e)
                continue

            ssims, psnrs, lpipss = [], [], []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation"):
                ssims.append(ssim(renders[idx], gts[idx]).item())
                psnrs.append(psnr(renders[idx], gts[idx]).item())
                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg').item())

            ssims_t  = torch.tensor(ssims)
            psnrs_t  = torch.tensor(psnrs)
            lpipss_t = torch.tensor(lpipss)

            print(f"  SSIM  : {ssims_t.mean():>10.6f}")
            print(f"  PSNR  : {psnrs_t.mean():>10.6f}")
            print(f"  LPIPS : {lpipss_t.mean():>10.6f}")

            # -------- 写入字典 -------- #
            full_dict[str(scene_dir)][method] = {
                "SSIM":  ssims_t.mean().item(),
                "PSNR":  psnrs_t.mean().item(),
                "LPIPS": lpipss_t.mean().item()
            }
            per_view_dict[str(scene_dir)][method] = {
                "SSIM":  dict(zip(img_names, ssims)),
                "PSNR":  dict(zip(img_names, psnrs)),
                "LPIPS": dict(zip(img_names, lpipss))
            }

        # -------- JSON 持久化 -------- #
        (scene_dir / "results.json").write_text(json.dumps(full_dict[str(scene_dir)], indent=2))
        (scene_dir / "per_view.json").write_text(json.dumps(per_view_dict[str(scene_dir)], indent=2))

# ------------------------- CLI ------------------------- #

if __name__ == "__main__":
    # GPU 默认使用 0 号卡
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        raise RuntimeError("CUDA is required for LPIPS/SSIM evaluation.")

    SPLIT_CHOICES = ["train", "test"]

    parser = ArgumentParser(description="Evaluate PSNR / SSIM / LPIPS for 3DGS renders")
    parser.add_argument('-m', '--model_paths', required=True, nargs="+",
                        help="Scene directories containing train/ or test/ folders")
    parser.add_argument('--split', default="test", choices=SPLIT_CHOICES,
                        help="Sub-folder to evaluate (default: test)")
    args = parser.parse_args()

    evaluate(args.model_paths, split=args.split)
