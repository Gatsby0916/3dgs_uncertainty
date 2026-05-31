#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
object_nbv_score.py  —  计算 σ̄_obj 分数并自动更新 train_split
---------------------------------------------------------
• σ_map : render.py 生成的 *.npz (字段 'sigma' or 'uncertainty_map')
• mask  : sam2_propagate 生成的 *_prob.npy
• --thr >0  ⇒ 二值掩码   =0 ⇒ 概率加权
• --mode sum|mean|max|pXX （sum: 累积分；mean: 面积归一化；max: 峰值；pXX: 百分位）
• 评分完后把最高分视图写入 --train-split
"""

import argparse
import json
import pathlib
import numpy as np
from pathlib import Path
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncert-dir", required=True,
                    help="含 *.npz (uncertainty_map) 的文件夹路径")
    ap.add_argument("--mask-dir",   required=True,
                    help="含 *_prob.npy 全概率掩码的目录")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="掩码阈值 (>0 二值化；=0 原概率加权)")
    modes = ["sum", "mean", "max"] + [f"p{p}" for p in (50,75,90,95)]
    ap.add_argument("--mode", choices=modes, default="mean",
                    help="打分模式：sum|mean|max|pXX (percent位)")
    ap.add_argument("--out-json",   required=True,
                    help="输出分数字典 JSON 文件路径")
    ap.add_argument("--train-split", required=True,
                    help="训练 split 文本路径")
    return ap.parse_args()


def compute_score(sigma: np.ndarray, prob: np.ndarray, thr: float, mode: str) -> float:
    # 构造权重 w
    if thr > 0:
        w = (prob >= thr).astype(np.float32)
    else:
        # For soft masks, variance scales with the square of the signal magnitude
        w = prob.astype(np.float32) ** 2

    # 加权不确定度
    prod = sigma * w

    # 只取前景（thr>0）或全部
    vals = prod[w > 0] if thr > 0 else prod.flatten()
    if vals.size == 0:
        return float('nan')

    if mode == "sum":
        return float(vals.sum())
    if mode == "mean":
        return float(vals.sum() / (w.sum() + 1e-12))
    if mode == "max":
        return float(vals.max())
    if mode.startswith("p"):
        q = float(mode[1:])
        return float(np.percentile(vals, q))
    raise ValueError(f"Unknown mode '{mode}'")


def main():
    args = parse_args()

    uncert_dir = pathlib.Path(args.uncert_dir)
    mask_dir   = pathlib.Path(args.mask_dir)
    npz_list   = sorted(uncert_dir.glob("*.npz"))
    assert npz_list, f"uncert-dir '{uncert_dir}' is empty"

    scores = {}
    skipped_missing = 0
    skipped_empty   = 0

    # 先建立 mask 目录下所有 stem 的映射（小写->真实名）
    mask_stem_map = {}
    for f in mask_dir.iterdir():
        if f.is_file() and f.name.endswith("_prob.npy"):
            mask_stem_map[f.stem.replace("_prob", "").lower()] = f.stem.replace("_prob", "")

    # uncertainty_npz 目录下所有 stem 的映射（小写->真实名）
    npz_stem_map = {f.stem.lower(): f.stem for f in npz_list}

    for npz_path in tqdm(npz_list, desc="scoring"):
        npz_stem = npz_path.stem
        npz_stem_lower = npz_stem.lower()
        # 找到 mask 匹配的真实 stem
        mask_stem = mask_stem_map.get(npz_stem_lower, None)
        if mask_stem is None:
            skipped_missing += 1
            continue
        mask_path = mask_dir / f"{mask_stem}_prob.npy"
        if not mask_path.exists():
            skipped_missing += 1
            continue

        data  = np.load(npz_path)
        sigma = data.get("sigma", data.get("uncertainty_map", None))
        if sigma is None:
            skipped_missing += 1
            continue

        prob  = np.load(mask_path).astype(np.float32)
        while prob.ndim > 3:
            prob = prob.squeeze(0)
        if prob.ndim == 3:
            prob = prob.max(0)

        if sigma.shape != prob.shape:
            skipped_missing += 1
            continue

        # 用 mask 目录下的真实 stem 作为 key
        score = compute_score(sigma, prob, args.thr, args.mode)
        if np.isnan(score):
            skipped_empty += 1
            continue

        scores[mask_stem] = score

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    out_path.write_text(json.dumps(sorted_scores, indent=2, ensure_ascii=False))
    print(f"✓  scores written to {out_path}")
    print(f"  总帧数 {len(npz_list)} | 有效 {len(scores)}"
          f" | 缺文件 {skipped_missing} | 掩码全零 {skipped_empty}")

    if not sorted_scores:
        print("⚠  no valid scores, stop updating split")
        return

    # 读取现有 train_split（保持原始名字）
    train_p = Path(args.train_split)
    train_set = set()
    if train_p.exists():
        train_set = {line.strip() for line in train_p.read_text().splitlines() if line.strip()}

    # 挑选第一个不在 train_set 中的视图（用 mask 真实 stem）
    best_view = None
    for s, _ in sorted_scores:
        if s not in train_set:
            best_view = s
            break

    if best_view is None:
        print("ℹ  所有候选都已在 train_split 中，跳过本轮更新")
        return

    # 加入 train_split（用 mask 真实 stem）
    train_set.add(best_view)
    train_p.write_text("\n".join(sorted(train_set)) + "\n")

    print(f"✅  added {best_view} to {train_p.name}; total train views {len(train_set)}")

if __name__ == "__main__":
    main()
