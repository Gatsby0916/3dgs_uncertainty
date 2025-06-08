#!/usr/bin/env python3
# ==============================================================
# 3DGS Uncertainty Evaluation  (RGB + Depth, Depth 分支做 min–max 归一化)
#   • 遍历 uncertainty_npz/ 下所有 .npz 文件，与 RGB/Depth/GT 对应
#   • RGB-σ 分支保持不变（线性 σ → ause_and_curve）
#   • Depth-σ 分支：对深度误差与 σ 做 min–max 归一化，再调用 ause_and_curve
# ==============================================================

import argparse, math, csv, sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use("default")

# ---------- constants --------------------------------------------------------
NUM_VIEWS   = 10    # 最多评估前 N 个视图。若想评估全部可去掉 [:NUM_VIEWS] 切片
AUSE_STEPS  = 50    # 积分步数（50 → 2% bins）
REMOVE_FRAC = 0.5   # ΔMAE/ΔPSNR 去除像素比例 (50%)
EPSILON     = 1e-8  # 防止除零或归一化除以 0

# ---------- I/O helpers ------------------------------------------------------
def read_png(fp: Path):
    """
    读取 PNG 图，返回 float32 数组，归一化到 [0,1]。支持 H×W（灰度）或 H×W×3（彩色）。
    """
    arr = imageio.imread(fp).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr  # H, W, C 或 H, W

def read_depth_npy(fp: Path):
    """
    读取渲染深度的 .npy 文件，返回 float32 数组。
    """
    return np.load(fp).astype(np.float32)

def read_depth_gt(fp: Path):
    """
    读取深度真值，可以是 PNG（单位为毫米）或 NPY（单位为米）。PNG 会除以 1000 转为米。
    """
    if fp.suffix.lower() == '.png':
        return imageio.imread(fp).astype(np.float32) / 1000.0
    else:
        return np.load(fp).astype(np.float32)

# ---------- math utilities ---------------------------------------------------
def crop_common(a, b):
    """
    将 2D 数组 a, b 裁剪到相同的最小高/宽后返回 (a_crop, b_crop)。
    """
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]

def spearmanr(a, b):
    """
    计算 Spearman 秩相关系数。
    """
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0,1]

def roc_auc(scores, labels):
    """
    计算二分类 ROC-AUC:
      - scores: 连续分数数组，越大表示越可能为正类
      - labels: 0/1 二值标签
    若 TP 或 FP 全为 0，返回 0.0
    """
    order = np.argsort(-scores)
    tp = np.cumsum(labels[order] == 1)
    fp = np.cumsum(labels[order] == 0)
    if tp[-1] == 0 or fp[-1] == 0:
        return 0.0
    return np.trapz(tp / tp[-1], fp / fp[-1])

def ause_and_curve(err, unc, steps=AUSE_STEPS):
    """
    计算 AUSE（Area Under the Sparsification Error）及 ΔMAE 曲线。
      - err: shape (H, W) 或扁平化的误差向量
      - unc: shape (H, W) 或扁平化的不确定性向量
      - steps: 采样步数，例如 50
    返回:
      - ause: 一个标量值，np.trapz(delta, fracs)
      - delta: 大小为 (steps,) 的 ΔMAE 序列
    """
    e = err.flatten()
    s = unc.flatten()
    N = e.size
    idx_u = np.argsort(-s)   # 按不确定性从大到小排序的索引
    idx_o = np.argsort(-e)   # 按误差从大到小排序的索引
    csum_u = np.cumsum(e[idx_u])
    csum_o = np.cumsum(e[idx_o])
    fracs = np.linspace(0, 1, steps+1)[1:]        # (steps,) 从 1/steps ... 1
    keep  = ((1.0 - fracs) * N).astype(int)       # 保留的像素数
    mae_u = (csum_u[-1] - csum_u[keep-1]) / (N - keep)
    mae_o = (csum_o[-1] - csum_o[keep-1]) / (N - keep)
    delta = mae_u - mae_o  # (steps,) 数组
    return np.trapz(delta, fracs), delta

# ---------- main ------------------------------------------------------------
def main(root: Path, save_csv: bool):
    """
    评估脚本：
      - root: 指向 …/ours_<iter> 目录，例如 data/LF/basket/output/train/ours_3000
          该目录下应包含：
            renders/*.png         (RGB 渲染图)
            gt/*.png              (GT 图)
            depth/*.npy           (渲染深度)
            depth_images/*.png/.npy (深度真值，与 data/LF/basket/depth_gt_XX.npy 同级)
            uncertainty_npz/*.npz (原始不确定性文件，含 “uncertainty_map”, “pixel_gaussian_counter”)
      - save_csv: 为 True 时，会将每个视图指标写入 uq_eval/metrics.csv
    """

    # 1. 构建各子目录路径
    rgb_dir        = root / 'renders'
    gt_dir         = root / 'gt'
    depth_dir      = root / 'depth'
    uncert_npz_dir = root / 'uncertainty_npz'
    depth_gt_dir   = root / 'depth_images'

    # 2. 构建映射：key 为“去掉前导 0 后的文件名”（字符串），value 为 Path
    def key(fp: Path):
        return fp.stem.lstrip('0') or '0'

    rgb_map   = { key(f): f for f in sorted(rgb_dir.glob('*.png'))[:NUM_VIEWS] }
    gt_map    = { key(f): f for f in sorted(gt_dir.glob('*.png'))[:NUM_VIEWS] }
    depth_map = { key(f): f for f in sorted(depth_dir.glob('*.npy'))[:NUM_VIEWS] }
    npz_map   = { key(f): f for f in sorted(uncert_npz_dir.glob('*.npz'))[:NUM_VIEWS] }

    # 深度真值文件：位于 root/depth_images/depth_gt_XX.npy 或 .png
    depth_gt_map = {}
    for f in sorted(depth_gt_dir.iterdir()):
        if f.suffix.lower() in ('.png', '.npy') and f.stem.startswith('depth_gt_'):
            idx = f.stem.split('_')[-1].lstrip('0') or '0'
            depth_gt_map[idx] = f

    # 3. 找到可评估的视图集合
    common_rgb = sorted(set(rgb_map.keys()) & set(gt_map.keys()) & set(npz_map.keys()))
    common_dep = sorted(set(depth_map.keys()) & set(depth_gt_map.keys()) & set(npz_map.keys()))

    if len(common_rgb) == 0:
        raise RuntimeError("未找到同时具备 renders/gt/uncertainty_npz 的视图。")
    if len(common_dep) == 0:
        raise RuntimeError("未找到同时具备 depth/depth_gt/uncertainty_npz 的视图。")

    # 4. 初始化 CSV 输出（如需要）
    if save_csv:
        outdir = root / 'uq_eval'
        outdir.mkdir(exist_ok=True)
        csv_f = open(outdir / 'metrics.csv', 'w', newline='', encoding='utf-8-sig')
        writer = csv.writer(csv_f)
        writer.writerow([
            "view",
            "AUSE", "ΔMAE", "ΔPSNR", "ρ", "AUROC",
            "AUSE_D", "ρ_D", "AUROC_D"
        ])

    # 5. 累积容器
    accum = { k: [] for k in ("ause","dmae","dpsnr","rho","auc","ause_d","rho_d","auc_d") }
    curve_rgb = np.zeros(AUSE_STEPS, dtype=np.float64)

    # ================================
    #  6. RGB-σ 分支 (使用 .npz 中的线性 σ)
    # ================================
    for idx in common_rgb:
        # 6.1 读取 RGB 渲染与 GT
        rgb_pred = imageio.imread(rgb_map[idx]).astype(np.float32) / 255.0  # [H,W,3]
        rgb_gt   = imageio.imread(gt_map[idx]).astype(np.float32) / 255.0    # [H,W,3]

        # 6.2 从 .npz 加载原始不确定性
        data       = np.load(npz_map[idx])
        uncert_map = data["uncertainty_map"].astype(np.float32)    # [H0,W0]
        counter    = data["pixel_gaussian_counter"].astype(np.float32)
        counter    = np.maximum(counter, 1.0)                      # 防止除以 0
        sigma_full = uncert_map / counter                          # [H0,W0]

        # 6.3 将 sigma_full 裁剪到与 RGB 同尺寸
        h_r, w_r = rgb_pred.shape[:2]
        sigma_c  = sigma_full[:h_r, :w_r]  # 假设 uncertainty 分辨率 ≥ 渲染分辨率

        # 6.4 计算 RGB 误差 map (平均三个通道)
        err_rgb = np.abs(rgb_pred - rgb_gt).mean(axis=2)  # [H,W]

        # 6.5 计算 RGB AUSE 及 ΔMAE 曲线
        ause_val, dcurve = ause_and_curve(err_rgb, sigma_c)
        accum["ause"].append(ause_val)
        curve_rgb += dcurve

        # 6.6 计算 ΔMAE / ΔPSNR / ρ / AUROC
        N = sigma_c.size
        k = int(REMOVE_FRAC * N)
        if k > 0:
            thr = np.partition(sigma_c.flatten(), -k)[-k]
        else:
            thr = sigma_c.max() + 1.0
        mask_keep = (sigma_c < thr)

        mae_all = err_rgb.mean()
        mse_all = (err_rgb**2).mean()
        if mask_keep.any():
            mae_keep = err_rgb[mask_keep].mean()
            mse_keep = (err_rgb[mask_keep]**2).mean()
        else:
            mae_keep = mae_all
            mse_keep = mse_all

        accum["dmae"].append(mae_keep - mae_all)
        accum["dpsnr"].append(-10 * math.log10(mse_keep + 1e-8) + 10 * math.log10(mse_all + 1e-8))
        accum["rho"].append(spearmanr(err_rgb.flatten(), sigma_c.flatten()))
        thr_err = np.partition(err_rgb.flatten(), -int(0.1 * N))[-int(0.1 * N)]
        labels_err = (err_rgb.flatten() >= thr_err).astype(np.uint8)
        accum["auc"].append(roc_auc(sigma_c.flatten(), labels_err))

        # 6.7 写入 CSV (仅 RGB 部分，Depth 部分留空)
        if save_csv:
            writer.writerow([
                idx,
                ause_val,
                accum["dmae"][-1],
                accum["dpsnr"][-1],
                accum["rho"][-1],
                accum["auc"][-1],
                "", "", ""
            ])

    # ================================
    #  7. Depth-σ 分支 (线性 σ + min–max 归一化 → ause_and_curve)
    # ================================
    for idx in common_dep:
        # 7.1 读取渲染深度 & 深度真值
        d_pred      = read_depth_npy(depth_map[idx])                  # [H1,W1]
        depth_gt_fp = depth_gt_map[idx]                                # 例如 root/depth_images/depth_gt_42.npy
        d_gt        = read_depth_gt(depth_gt_fp)                       # [H2,W2]
        d_pred, d_gt = crop_common(d_pred, d_gt)                       # 裁剪到相同尺寸 (H, W)

        # 7.2 从 .npz 加载不确定性
        data        = np.load(npz_map[idx])
        uncert_map  = data["uncertainty_map"].astype(np.float32)      # [H0,W0]
        counter     = data["pixel_gaussian_counter"].astype(np.float32)
        counter     = np.maximum(counter, 1.0)
        sigma_full  = uncert_map / counter                             # [H0,W0]

        # 7.3 将 sigma_full 裁剪到深度尺寸
        h_d, w_d    = d_pred.shape
        sigma_crop  = sigma_full[:h_d, :w_d]  # 假设 uncertainty 分辨率 ≥ 深度分辨率

        # 7.4 计算深度误差 map
        err_map = np.abs(d_pred - d_gt)    # [h_d, w_d]

        # 7.5 对 err_map 与 sigma_crop 做 min–max 归一化
        # err_norm ∈ [0,1], sigma_norm ∈ [0,1]
        e_min, e_max = err_map.min(), err_map.max()
        s_min, s_max = sigma_crop.min(), sigma_crop.max()
        if e_max - e_min < EPSILON:
            err_norm = np.zeros_like(err_map)
        else:
            err_norm = (err_map - e_min) / (e_max - e_min)
        if s_max - s_min < EPSILON:
            sigma_norm = np.zeros_like(sigma_crop)
        else:
            sigma_norm = (sigma_crop - s_min) / (s_max - s_min)

        # 7.6 直接调用 ause_and_curve (线性 sigma_norm，err_norm)
        ause_d_val, _ = ause_and_curve(err_norm, sigma_norm)
        accum["ause_d"].append(ause_d_val)

        # 7.7 计算 Depth‐Spearman ρ_D 与 Depth‐AUROC_D
        err_vec = err_norm.flatten()      # 归一化后
        unc_vec = sigma_norm.flatten()    # 归一化后
        rho_d   = spearmanr(err_vec, unc_vec)
        thr_d   = np.partition(err_vec, -int(0.1 * err_vec.size))[-int(0.1 * err_vec.size)]
        labels_d = (err_vec >= thr_d).astype(np.uint8)
        auc_d   = roc_auc(unc_vec, labels_d)
        accum["rho_d"].append(rho_d)
        accum["auc_d"].append(auc_d)

        # 7.8 写入 CSV (仅 Depth 部分，RGB 部分留空)
        if save_csv:
            writer.writerow([
                idx,
                "", "", "", "", "",
                ause_d_val,
                rho_d,
                auc_d
            ])

    # ================================
    #  8. 最终汇总 & 绘图
    # ================================
    # RGB-σ 指标
    print("=== RGB-σ metrics ===")
    print(f"AUSE      : {np.mean(accum['ause']):.4f}")
    print(f"ΔMAE      : {np.mean(accum['dmae']):.4f}")
    print(f"ΔPSNR     : {np.mean(accum['dpsnr']):.2f} dB")
    print(f"Spearman ρ: {np.mean(accum['rho']):.4f}")
    print(f"AUROC     : {np.mean(accum['auc']):.4f}")

    # Depth-σ 指标
    print("=== Depth-σ metrics ===")
    print(f"AUSE_D    : {np.mean(accum['ause_d']):.4f}")
    print(f"Spearman ρD: {np.mean(accum['rho_d']):.4f}")
    print(f"AUROC_D   : {np.mean(accum['auc_d']):.4f}")

    # 绘制 RGB-ΔMAE Sparsification 曲线
    curve_rgb /= len(common_rgb)
    fracs = np.linspace(0, 1, AUSE_STEPS+1)[1:]
    plt.figure(figsize=(5,4), dpi=120)
    plt.plot(fracs, curve_rgb, lw=2)
    plt.xlabel("Fraction removed (σ-sorted)")
    plt.ylabel("ΔMAE")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

    if save_csv:
        csv_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("3DGS UQ evaluation (Depth-AUSE 做 min–max 归一化)")
    parser.add_argument(
        "--root", required=True,
        help="…/ours_<iter> 目录路径，例如 data/LF/basket/output/train/ours_3000"
    )
    parser.add_argument(
        "--save_csv", action="store_true",
        help="是否输出 uq_eval/metrics.csv"
    )
    args = parser.parse_args()

    main(Path(args.root), args.save_csv)
