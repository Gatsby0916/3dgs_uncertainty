#!/usr/bin/env python3
"""
将.npy格式的概率掩码转换为PNG图像
使用viridis颜色映射，只显示概率大于0.5的区域
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
import glob
from pathlib import Path
import argparse

def convert_mask_to_png(npy_path, output_path, threshold=0.5):
    """
    将.npy概率掩码转换为PNG图像
    
    Args:
        npy_path: .npy文件路径
        output_path: 输出PNG文件路径
        threshold: 概率阈值，默认0.5
    """
    # 加载.npy文件
    prob_mask = np.load(npy_path)
    
    # 去掉多余的维度 (1, 1, H, W) -> (H, W)
    prob_mask = prob_mask.squeeze()
    
    # 创建掩码：只保留概率大于阈值的区域
    mask = prob_mask > threshold
    
    # 创建输出图像，背景为黑色
    output_image = np.zeros_like(prob_mask)
    output_image[mask] = prob_mask[mask]
    
    # 使用viridis颜色映射
    plt.figure(figsize=(prob_mask.shape[1]/100, prob_mask.shape[0]/100), dpi=100)
    plt.imshow(output_image, cmap='viridis', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # 保存为PNG
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    print(f"转换完成: {os.path.basename(npy_path)} -> {os.path.basename(output_path)}")

def process_dataset(dataset_name):
    """处理单个数据集的所有mask文件"""
    mask_dir = f"/home/haiyi/3dgs_uncertainty/LF/ours/{dataset_name}/mask"
    output_dir = f"/home/haiyi/3dgs_uncertainty/LF/ours/{dataset_name}/mask_png"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有.npy文件
    npy_files = glob.glob(os.path.join(mask_dir, "*.npy"))
    npy_files.sort()
    
    print(f"\n处理数据集: {dataset_name}")
    print(f"找到 {len(npy_files)} 个.npy文件")
    
    for npy_file in npy_files:
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(npy_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.png")
        
        # 转换文件
        convert_mask_to_png(npy_file, output_file)
    
    print(f"数据集 {dataset_name} 处理完成，共 {len(npy_files)} 个文件")

def main():
    parser = argparse.ArgumentParser(description='转换概率掩码为PNG图像')
    parser.add_argument('--dataset', type=str, choices=['africa', 'basket', 'statue', 'torch', 'all'], 
                        default='all', help='要处理的数据集')
    parser.add_argument('--threshold', type=float, default=0.5, help='概率阈值')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        datasets = ['africa', 'basket', 'statue', 'torch']
    else:
        datasets = [args.dataset]
    
    print("=== 概率掩码转PNG工具 ===")
    print(f"阈值: {args.threshold}")
    print(f"颜色映射: viridis")
    
    for dataset in datasets:
        process_dataset(dataset)
    
    print("\n✅ 所有转换完成！")

if __name__ == "__main__":
    main()
