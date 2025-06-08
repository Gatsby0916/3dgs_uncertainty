#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse

import numpy as np
import torch
from PIL import Image

def np_depth_to_image(depth_np):
    """
    将浮点深度数组归一化到 [0,255] 并转为 uint8 灰度图
    """
    d_min, d_max = np.nanmin(depth_np), np.nanmax(depth_np)
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth_np, dtype=np.uint8)
    else:
        norm = (depth_np - d_min) / (d_max - d_min)  # 归一化到 [0,1]
        norm = (norm * 255.0).astype(np.uint8)
    return norm

def main(args):
    # 1. 构造深度文件列表
    pattern = os.path.join(args.dataset_dir, 'depth_*.npy')
    depth_files = sorted(glob.glob(pattern))
    if not depth_files:
        print(f"未在 `{args.dataset_dir}` 下找到匹配 `depth_*.npy` 的文件。")
        return

    # 2. 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 遍历每个文件
    for path in depth_files:
        # 3.1 加载 numpy 数组
        depth_np = np.load(path)  # shape = (H, W)
        # 3.2 转为 torch.Tensor（可选）
        depth_torch = torch.from_numpy(depth_np)
        # 3.3 打印分辨率
        H, W = depth_np.shape
        print(f"[Loaded] {os.path.basename(path)}  →  分辨率: {W}×{H}")

        # 3.4 转为灰度图并保存
        img_arr = np_depth_to_image(depth_np)
        img = Image.fromarray(img_arr, mode='L')  # L 模式：灰度图
        save_name = os.path.splitext(os.path.basename(path))[0] + '.png'
        out_path = os.path.join(args.output_dir, save_name)
        img.save(out_path)
        print(f"         已保存灰度图: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="从 LF 数据集的 depth_*.npy 文件提取深度图并保存为 PNG"
    )
    parser.add_argument(
        '--dataset_dir', '-d',
        type=str, required=True,
        help="LF 场景目录，包含 depth_*.npy 文件"
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str, default='./depth_images',
        help="保存生成 PNG 深度图的目录"
    )
    args = parser.parse_args()
    main(args)
