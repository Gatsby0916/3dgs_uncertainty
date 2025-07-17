#!/usr/bin/env python3
"""
run_fisher_pipeline.py  —  自动化 FisherRF-风格 3DGS NBV 流水线
========================================================
基于 fisher_config.yml 执行以下步骤：
 0. 若不存在 train_split.txt/candidate_split.txt，随机从前 max_pool 张图生成
 1. 循环 max_rounds 次：
    a) 训练至指定迭代步
    b) 渲染（mask-aware uncertainty）
    c) 打分选帧加入 train_split.txt
 2. 最终质量评估（PSNR/SSIM/LPIPS）

依赖: Python3, pyyaml, tqdm
"""
import os
import subprocess
import sys
import yaml
from pathlib import Path
from tqdm import trange

def sh(cmd):
    print(f">>> {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    cfg = yaml.safe_load(open("fisher_config.yml"))

    ds            = cfg["dataset"]
    img_dir       = Path(ds) / "images"
    mask_dir      = Path(cfg["mask_dir"])
    train_split   = Path("train_split.txt")
    cand_split    = Path("candidate_split.txt")
    max_pool      = cfg.get("max_pool", 100)
    init_pick     = cfg.get("init_pick", 4)
    max_rounds    = cfg.get("max_rounds", 16)
    max_views     = cfg.get("max_views", 50)  # 0 表示不限制
    iter_init     = cfg.get("train_iter_init", 3000)
    iter_delta    = cfg.get("train_iter_delta", 3000)
    patch_size    = cfg["uncert_mode"]["patch_size"]
    thr           = cfg["uncert_mode"]["thr"]
    mode          = cfg["uncert_mode"].get("mode", "max")


    # 0. 随机 split (仅从前 max_pool 张图中选)
    if not train_split.exists() or not cand_split.exists():
        sh(f"python make_random_split.py --img-dir {img_dir} "
           f"--num-train {init_pick} --max-pool {max_pool} "
           f"--train-out {train_split} --cand-out {cand_split}")

    prev_iter = 0
    for r in range(max_rounds):
        end_iter = iter_init + r * iter_delta

        # 1. 训练
        cmd_train = (
            f"python train.py -s {ds} -m {ds}/output "
            f"--iterations {end_iter} --checkpoint_iterations {end_iter} "
            f"--train_split {train_split} --densify_until_iter 2000 "
        )
        if iter_init > 1875:
            cmd_train += f" --start_checkpoint {ds}/output/chkpnt{prev_iter}.pth"
        sh(cmd_train)

        # 2. 渲染 uncertainty
        sh(
            f"python render.py -m {ds}/output --iteration {end_iter} "
            f"--uncertainty_mode --patch_size {patch_size} --mask-dir {mask_dir} --max_views {max_views} "
        )

        # 3. 打分 & 选帧
        sh(
            f"python object_nbv_score.py "
            f"--uncert-dir {ds}/output/train/ours_{end_iter}/uncertainty_npz "
            f"--mask-dir {mask_dir} --thr {thr} --mode {mode} "
            f"--out-json score_{end_iter}.json "
            f"--train-split {train_split} --cand-split {cand_split}"
        )

        prev_iter = end_iter

    # 4. 最终质量评估
    print("\n=== 开始最终质量评估 (PSNR/SSIM/LPIPS) ===")
    sh(f"python metrics.py -m {ds}/output --split train")

    print("=== NBV 流水线完成 ===")
