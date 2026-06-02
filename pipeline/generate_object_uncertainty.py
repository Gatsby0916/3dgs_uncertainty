#!/usr/bin/env python3
"""
批量生成Object Uncertainty图像
使用mask增强uncertainty显示，专为视频制作设计
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from pathlib import Path
import argparse

def load_uncertainty_from_npz(npz_path):
    """从NPZ文件加载不确定性数据"""
    try:
        data = np.load(npz_path)
        if 'uncertainty_map' in data:
            return data['uncertainty_map']
        elif 'uncertainty' in data:
            return data['uncertainty']
        else:
            # 取第一个可用的数组
            keys = list(data.keys())
            return data[keys[0]]
    except Exception as e:
        print(f"加载NPZ文件失败 {npz_path}: {e}")
        return None

def load_mask(mask_path):
    """从NPY文件加载mask数据"""
    try:
        mask = np.load(mask_path)
        # 去掉多余维度 (1, 1, H, W) -> (H, W)
        if mask.ndim > 2:
            mask = mask.squeeze()
        return mask
    except Exception as e:
        print(f"加载mask文件失败 {mask_path}: {e}")
        return None

def enhance_uncertainty_with_mask(uncertainty, mask, enhancement_factor=5.0, mask_threshold=0.1, global_brightness=1.2):
    """
    使用mask增强uncertainty显示
    
    Args:
        uncertainty: 不确定性图 (H, W)
        mask: 对象mask (H, W)，值在[0,1]范围
        enhancement_factor: 增强因子，object区域的放大倍数
        mask_threshold: mask阈值，高于此值的区域被认为是object
        global_brightness: 全局亮度控制
    """
    # 创建object区域的mask
    object_mask = mask > mask_threshold
    
    # 增强object区域的uncertainty
    enhanced = uncertainty.copy()
    enhanced[object_mask] = enhanced[object_mask] * enhancement_factor
    
    # 应用全局亮度调整 - 增加背景亮度
    enhanced = enhanced * global_brightness
    
    # 确保值在合理范围内
    enhanced = np.clip(enhanced, 0, enhanced.max())
    
    return enhanced

def save_clean_image(data, save_path, colormap='viridis', dpi=300, vmin=None, vmax=None):
    """保存纯净的图像，无边框、标题、colorbar"""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    
    plt.figure(figsize=(data.shape[1]/dpi, data.shape[0]/dpi), dpi=dpi)
    plt.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

def get_dataset_config(dataset_name):
    """获取数据集特定的配置"""
    configs = {
        'africa': {
            'uncertainty_dir': 'output/train/ours_4000/uncertainty_npz',
            'mask_pattern': lambda frame_id: f"africa{frame_id}_prob.npy",
            'uncertainty_pattern': lambda frame_id: f"africa{frame_id}.npz"
        },
        'basket': {
            'uncertainty_dir': 'output/train/ours_4000/uncertainty_npz',
            'mask_pattern': lambda frame_id: f"basket_{frame_id}_Cropped_prob.npy",
            'uncertainty_pattern': lambda frame_id: f"basket_{frame_id}_Cropped.npz"
        },
        'statue': {
            'uncertainty_dir': 'output/train/ours_4000/uncertainty_npz',
            'mask_pattern': lambda frame_id: f"statue_alex{frame_id}_prob.npy",
            'uncertainty_pattern': lambda frame_id: f"statue_alex{frame_id}.npz"
        },
        'torch': {
            'uncertainty_dir': 'output/train/ours_4000/uncertainty_npz',
            'mask_pattern': lambda frame_id: f"roof{frame_id}_prob.npy",
            'uncertainty_pattern': lambda frame_id: f"roof{frame_id}.npz"
        }
    }
    return configs.get(dataset_name)

def extract_frame_id(filename, dataset_name):
    """从文件名中提取帧ID，根据数据集调整提取规则"""
    import re
    
    if dataset_name == 'africa':
        # africa03200.npz -> 03200
        match = re.search(r'africa(\d{5})', filename)
    elif dataset_name == 'basket':
        # basket_00001_Cropped.npz -> 00001
        match = re.search(r'basket_(\d{5})_Cropped', filename)
    elif dataset_name == 'statue':
        # statue_alex01250.npz -> 01250
        match = re.search(r'statue_alex(\d{5})', filename)
    elif dataset_name == 'torch':
        # roof00060.npz -> 00060
        match = re.search(r'roof(\d{5})', filename)
    else:
        # 通用模式
        match = re.search(r'(\d{5})', filename)
    
    if match:
        return match.group(1)
    return None

def process_dataset(dataset_name, root, enhancement_factor=5.0, colormap='viridis'):
    """处理单个数据集的所有文件"""
    print(f"\n处理数据集: {dataset_name}")

    # 获取数据集配置
    config = get_dataset_config(dataset_name)
    if not config:
        print(f"错误: 不支持的数据集 {dataset_name}")
        return 0

    # 设置路径
    base_dir = os.path.join(root, dataset_name)
    uncertainty_dir = os.path.join(base_dir, config['uncertainty_dir'])
    mask_dir = os.path.join(base_dir, "mask")
    output_dir = os.path.join(base_dir, "object_uncertainty_png")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查目录是否存在
    if not os.path.exists(uncertainty_dir):
        print(f"警告: 不确定性目录不存在 {uncertainty_dir}")
        return 0
    if not os.path.exists(mask_dir):
        print(f"警告: mask目录不存在 {mask_dir}")
        return 0
    
    # 获取所有uncertainty文件
    uncertainty_files = glob.glob(os.path.join(uncertainty_dir, "*.npz"))
    uncertainty_files.sort()
    
    processed_count = 0
    failed_count = 0
    
    for unc_file in uncertainty_files:
        # 提取帧ID
        frame_id = extract_frame_id(os.path.basename(unc_file), dataset_name)
        if not frame_id:
            print(f"无法提取帧ID: {unc_file}")
            failed_count += 1
            continue
        
        # 生成对应的mask文件路径
        mask_filename = config['mask_pattern'](frame_id)
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"找不到对应的mask文件: {mask_path}")
            failed_count += 1
            continue
        
        # 加载数据
        uncertainty = load_uncertainty_from_npz(unc_file)
        mask = load_mask(mask_path)
        
        if uncertainty is None or mask is None:
            print(f"加载数据失败: {unc_file}")
            failed_count += 1
            continue
        
        # 检查尺寸匹配
        if uncertainty.shape != mask.shape:
            print(f"尺寸不匹配: uncertainty {uncertainty.shape} vs mask {mask.shape}")
            failed_count += 1
            continue
        
        # 增强uncertainty
        enhanced = enhance_uncertainty_with_mask(uncertainty, mask, enhancement_factor)
        
        # 生成输出文件名
        if dataset_name == 'africa':
            output_filename = f"africa{frame_id}_object_uncertainty.png"
        elif dataset_name == 'basket':
            output_filename = f"basket_{frame_id}_object_uncertainty.png"
        elif dataset_name == 'statue':
            output_filename = f"statue_alex{frame_id}_object_uncertainty.png"
        elif dataset_name == 'torch':
            output_filename = f"roof{frame_id}_object_uncertainty.png"
        else:
            output_filename = f"{dataset_name}{frame_id}_object_uncertainty.png"
            
        output_path = os.path.join(output_dir, output_filename)
        save_clean_image(enhanced, output_path, colormap=colormap)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"已处理 {processed_count} 个文件...")
    
    if failed_count > 0:
        print(f"数据集 {dataset_name} 处理完成，成功 {processed_count} 个，失败 {failed_count} 个")
    else:
        print(f"数据集 {dataset_name} 处理完成，共处理 {processed_count} 个文件")
    return processed_count

def main():
    parser = argparse.ArgumentParser(description='生成Object Uncertainty图像')
    parser.add_argument('--dataset', type=str, choices=['africa', 'basket', 'statue', 'torch', 'all'], 
                        default='all', help='要处理的数据集')
    parser.add_argument('--enhancement-factor', type=float, default=2.5, 
                        help='Object区域的增强因子')
    parser.add_argument('--colormap', type=str, default='viridis',
                        help='颜色映射')
    parser.add_argument('--root', type=str, required=True,
                        help='dataset root, contains <dataset>/{output,mask}')

    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = ['africa', 'basket', 'statue', 'torch']
    else:
        datasets = [args.dataset]

    print("=== Object Uncertainty 生成工具 ===")
    print(f"增强因子: {args.enhancement_factor}")
    print(f"颜色映射: {args.colormap}")
    print(f"根目录: {args.root}")

    total_processed = 0
    for dataset in datasets:
        count = process_dataset(dataset, args.root, args.enhancement_factor, args.colormap)
        total_processed += count
    
    print(f"\n✅ 处理完成！总共处理了 {total_processed} 个文件")

if __name__ == "__main__":
    main()
