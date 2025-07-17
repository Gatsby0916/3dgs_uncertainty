#!/usr/bin/env python3
"""
object_nbv_score.py  —  计算 σ̄_obj 分数并自动更新 split
---------------------------------------------------------
• σ_map : render.py 生成的 *.npz (字段 'sigma' or 'uncertainty_map')
• mask  : sam2_propagate 生成的 *_prob.npy
• --thr >0  ⇒ 二值掩码   =0 ⇒ 概率加权
• --mode sum|mean|max|pXX （sum: 累积分；mean: 面积归一化；max: 峰值；pXX: 百分位）
• 评分完后把最高分视图写入 --train-split，并从 --cand-split 删除
"""

import argparse
import json
import pathlib
import numpy as np
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
    ap.add_argument("--cand-split",  required=True,
                    help="候选 split 文本路径")
    return ap.parse_args()

def compute_score(sigma: np.ndarray, prob: np.ndarray, thr: float, mode: str) -> float:
    # 构造权重 w
    if thr > 0:
        w = (prob >= thr).astype(np.float32)
    else:
        w = prob.astype(np.float32)

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

    for npz_path in tqdm(npz_list, desc="scoring"):
        # 统一使用大写 stem
        name = npz_path.stem.upper()

        # 找对应 mask
        mask_path = mask_dir / f"{name}_prob.npy"
        if not mask_path.exists():
            hits = list(mask_dir.glob(f"*{name}*_prob.npy"))
            if hits:
                mask_path = hits[0]
            else:
                skipped_missing += 1
                continue

        data  = np.load(npz_path)
        sigma = data.get("sigma", data.get("uncertainty_map", None))
        if sigma is None:
            skipped_missing += 1
            continue

        prob  = np.load(mask_path).astype(np.float32)
        # 处理多维度 mask：(B,n_obj,H,W) → (H,W)
        while prob.ndim > 3:
            prob = prob.squeeze(0)
        if prob.ndim == 3:
            prob = prob.max(0)

        if sigma.shape != prob.shape:
            skipped_missing += 1
            continue

        score = compute_score(sigma, prob, args.thr, args.mode)
        if np.isnan(score):
            skipped_empty += 1
            continue

        scores[name] = score

    # 保存 JSON
    out_path = pathlib.Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    out_path.write_text(json.dumps(sorted_scores, indent=2, ensure_ascii=False))
    print(f"✓  scores written to {out_path}")
    print(f"  总帧数 {len(npz_list)} | 有效 {len(scores)}"
          f" | 缺文件 {skipped_missing} | 掩码全零 {skipped_empty}")

    if not sorted_scores:
        print("⚠  no valid scores, stop updating split")
        return

    # 读取现有 train_split（大写）
    train_p   = pathlib.Path(args.train_split)
    cand_p    = pathlib.Path(args.cand_split)

    train_set = set()
    if train_p.exists():
        train_set = set(
            line.strip().upper()
            for line in train_p.read_text().splitlines()
            if line.strip()
        )

    cand_list = []
    if cand_p.exists():
        cand_list = [line.strip().upper() for line in cand_p.read_text().splitlines() if line.strip()]

    # 挑选第一个不在 train_set 中的视图
    best_view = None
    for name, _ in sorted_scores:
        if name not in train_set:
            best_view = name
            break

    if best_view is None:
        print("ℹ  所有候选都已在 train_split 中，跳过本轮更新")
        return

    # 从候选集中移除
    if best_view in cand_list:
        cand_list.remove(best_view)
        cand_p.write_text("\n".join(cand_list) + ("\n" if cand_list else ""))

    # 加入 train_split
    train_set.add(best_view)
    train_p.write_text("\n".join(sorted(train_set)) + "\n")

    print(f"✅  added {best_view} to {train_p.name}; remaining candidates {len(cand_list)}")

if __name__ == "__main__":
    main()
