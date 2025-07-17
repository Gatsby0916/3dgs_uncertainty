#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2

def print_threshold_stats(mask: np.ndarray, eps: float):
    """打印掩码在 eps 阈值下的像素统计信息。"""
    total = mask.size
    keep = np.count_nonzero(mask >= eps)
    tiny = np.count_nonzero((mask > 0) & (mask < eps))
    print(f"threshold eps = {eps}")
    print(f"keep pixels   = {keep:,d}  ({keep/total:.2%})")
    print(f"tiny pixels   = {tiny:,d}  ({tiny/total:.2%})")
    print(f"min / max     = {mask.min():.3g} / {mask.max():.3g}")

def normalize_to_uint8(mask: np.ndarray) -> np.ndarray:
    """归一化浮点掩码到 [0,255] 的 uint8 图像。"""
    mn, mx = mask.min(), mask.max()
    if mx > mn:
        norm = (mask - mn) / (mx - mn)
    else:
        norm = mask - mn
    return (norm * 255).astype(np.uint8)

def visualize_image(win_name: str, img: np.ndarray):
    """显示图像并等待按键，然后关闭窗口。"""
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cmap_code(name: str) -> int:
    """将 cmap 名称映射到 OpenCV 的 COLORMAP_* 常量。"""
    mapping = {
        'jet':       cv2.COLORMAP_JET,
        'viridis':   cv2.COLORMAP_VIRIDIS,
        'hot':       cv2.COLORMAP_HOT,
        'rainbow':   cv2.COLORMAP_RAINBOW,
        'plasma':    cv2.COLORMAP_PLASMA,
    }
    return mapping.get(name.lower(), cv2.COLORMAP_JET)

def handle_single_file(path: str, eps: float, cmap_name: str):
    mask = np.load(path).astype(np.float32).squeeze()
    if mask.ndim != 2:
        print(f"[错误] 加载后维度为 {mask.shape}，非二维掩码，无法处理。")
        return
    print_threshold_stats(mask, eps)
    gray = normalize_to_uint8(mask)
    color = cv2.applyColorMap(gray, cmap_code(cmap_name))
    visualize_image(f"ColorMap ({cmap_name})", color)
    # 二值化展示
    binary = (mask >= eps).astype(np.uint8) * 255
    visualize_image(f"Binary (>= {eps})", binary)

def handle_directory(dir_path: str, num: int, eps: float, cmap_name: str):
    files = sorted(f for f in os.listdir(dir_path) if f.lower().endswith('.npy'))
    for fname in files[:num]:
        full = os.path.join(dir_path, fname)
        print(f"\n=== {fname} ===")
        handle_single_file(full, eps, cmap_name)

def main():
    p = argparse.ArgumentParser(
        description="可视化 .npy 掩码并输出阈值统计（支持单文件或目录模式）。"
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--mask", type=str,
                     help="单个 .npy 掩码文件路径")
    grp.add_argument("--dir",   type=str,
                     help="包含 .npy 掩码文件的目录路径")
    p.add_argument("--eps",   type=float, default=0.9,
                   help="阈值 eps（默认 0.9）")
    p.add_argument("--num",   type=int,   default=5,
                   help="目录模式下展示前 N 个掩码（默认 5）")
    p.add_argument("--cmap",  type=str,   default='jet',
                   choices=['jet','viridis','hot','rainbow','plasma'],
                   help="伪彩色映射名称（默认 jet）")
    args = p.parse_args()

    if args.mask:
        handle_single_file(args.mask, args.eps, args.cmap)
    else:
        handle_directory(args.dir, args.num, args.eps, args.cmap)

if __name__ == "__main__":
    main()
