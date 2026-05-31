#!/usr/bin/env python3
"""
正确的不确定性增强脚本
让mask中亮的部分在最终可视化中更亮
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_uncertainty_from_npz(npz_path):
    """从NPZ文件加载不确定性数据"""
    data = np.load(npz_path)
    return data['uncertainty_map']

def load_mask(mask_path):
    """加载mask数据"""
    mask = np.load(mask_path)
    # mask的形状是(1, 1, H, W)，需要squeeze到(H, W)
    if len(mask.shape) == 4:
        mask = mask.squeeze()
    # 确保mask在0-1范围内
    if mask.max() > 1.0:
        mask = mask / mask.max()
    return mask

def enhance_uncertainty_with_mask(uncertainty, mask, enhancement_factor=2.0):
    """
    使用mask增强不确定性：mask中值越高的地方，不确定性增强越多
    """
    # 方法：在mask亮的区域，将不确定性值乘以更大的系数
    # mask值范围是0-1，我们希望mask=1的地方增强最多，mask=0的地方不增强
    enhancement_multiplier = 1.0 + mask * (enhancement_factor - 1.0)
    enhanced_uncertainty = uncertainty * enhancement_multiplier
    return enhanced_uncertainty

def save_clean_image(data, save_path, colormap='viridis', dpi=300):
    """保存纯净的图像，无边框、标题、colorbar"""
    fig, ax = plt.subplots(figsize=(data.shape[1]/100, data.shape[0]/100), dpi=dpi)
    ax.imshow(data, cmap=colormap)
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(data.shape[0], 0)
    ax.axis('off')
    
    # 去除所有边距和边框
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    # 保存图像
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # 文件路径 - 修正匹配关系
    # 根据camera位置分析：
    # basket_01601 (右侧位置) 应该对应 basket_00161 (左侧位置) - 这可能是编号规则的差异
    # basket_01441 (右侧位置) 应该对应 basket_00141 (左侧位置) 
    files = {
        'uncertainty1_npz': 'LF/basket/output/test/ours_4000/uncertainty_npz/basket_01601_Cropped.npz',
        'uncertainty2_npz': 'LF/basket/output/test/ours_4000/uncertainty_npz/basket_01441_Cropped.npz',
        'mask1': 'LF/basket/mask/basket_00161_Cropped_prob.npy',  # 修正：使用00161而不是01601
        'mask2': 'LF/basket/mask/basket_00141_Cropped_prob.npy'   # 修正：使用00141而不是01441
    }
    
    # 检查文件是否存在
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"错误: 文件不存在 {name}: {path}")
            return
    
    # 创建输出目录
    output_dir = 'uncertainty_enhanced_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("加载数据...")
    
    # 加载数据
    unc1 = load_uncertainty_from_npz(files['uncertainty1_npz'])
    unc2 = load_uncertainty_from_npz(files['uncertainty2_npz'])
    mask1 = load_mask(files['mask1'])
    mask2 = load_mask(files['mask2'])
    
    print(f"数据统计:")
    print(f"  Uncertainty 1: shape={unc1.shape}, range=[{unc1.min():.6f}, {unc1.max():.6f}]")
    print(f"  Uncertainty 2: shape={unc2.shape}, range=[{unc2.min():.6f}, {unc2.max():.6f}]")
    print(f"  Mask 1: shape={mask1.shape}, range=[{mask1.min():.6f}, {mask1.max():.6f}]")
    print(f"  Mask 2: shape={mask2.shape}, range=[{mask2.min():.6f}, {mask2.max():.6f}]")
    
    # 先保存原始的不确定性可视化作为对比
    print("保存原始不确定性图像...")
    save_clean_image(unc1, f'{output_dir}/original_unc1.png', 'viridis')
    save_clean_image(unc2, f'{output_dir}/original_unc2.png', 'viridis')
    save_clean_image(unc1, f'{output_dir}/original_unc1_plasma.png', 'plasma')
    save_clean_image(unc2, f'{output_dir}/original_unc2_plasma.png', 'plasma')
    
    # 也保存mask的可视化
    print("保存mask可视化...")
    save_clean_image(mask1, f'{output_dir}/mask1_visualization.png', 'gray')
    save_clean_image(mask2, f'{output_dir}/mask2_visualization.png', 'gray')
    
    # 测试不同的增强系数
    enhancement_factors = [1.5, 2.0, 3.0, 5.0]
    
    for factor in enhancement_factors:
        print(f"处理增强系数 {factor}...")
        
        # 增强不确定性
        enhanced_unc1 = enhance_uncertainty_with_mask(unc1, mask1, factor)
        enhanced_unc2 = enhance_uncertainty_with_mask(unc2, mask2, factor)
        
        print(f"  增强后 Uncertainty 1: range=[{enhanced_unc1.min():.6f}, {enhanced_unc1.max():.6f}]")
        print(f"  增强后 Uncertainty 2: range=[{enhanced_unc2.min():.6f}, {enhanced_unc2.max():.6f}]")
        
        # 保存增强后的图像
        save_clean_image(enhanced_unc1, f'{output_dir}/enhanced_unc1_factor{factor}.png', 'viridis')
        save_clean_image(enhanced_unc2, f'{output_dir}/enhanced_unc2_factor{factor}.png', 'viridis')
        save_clean_image(enhanced_unc1, f'{output_dir}/enhanced_unc1_factor{factor}_plasma.png', 'plasma')
        save_clean_image(enhanced_unc2, f'{output_dir}/enhanced_unc2_factor{factor}_plasma.png', 'plasma')
        
        # 创建混合图像
        blended = 0.5 * enhanced_unc1 + 0.5 * enhanced_unc2
        save_clean_image(blended, f'{output_dir}/blended_factor{factor}.png', 'viridis')
        save_clean_image(blended, f'{output_dir}/blended_factor{factor}_plasma.png', 'plasma')
    
    print(f"✓ 所有结果已保存到: {output_dir}/")
    print("处理完成!")
    
    # 打印一些额外的统计信息
    print("\n=== 数据分析 ===")
    print(f"Mask1中非零值的比例: {(mask1 > 0.1).sum() / mask1.size * 100:.1f}%")
    print(f"Mask2中非零值的比例: {(mask2 > 0.1).sum() / mask2.size * 100:.1f}%")
    print(f"原始不确定性1的非零值比例: {(unc1 > unc1.max()*0.1).sum() / unc1.size * 100:.1f}%")
    print(f"原始不确定性2的非零值比例: {(unc2 > unc2.max()*0.1).sum() / unc2.size * 100:.1f}%")

if __name__ == "__main__":
    main()
