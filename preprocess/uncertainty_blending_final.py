import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

def load_uncertainty_from_png(png_path):
    """从PNG文件加载不确定性数据"""
    from PIL import Image
    
    img = Image.open(png_path)
    img_array = np.array(img)
    
    print(f"从 {png_path} 加载不确定性数据，原始形状: {img_array.shape}")
    
    # 如果是彩色图像，转换为灰度
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 3:  # RGB
            uncertainty = np.mean(img_array, axis=2)
        elif img_array.shape[2] == 4:  # RGBA
            uncertainty = np.mean(img_array[:,:,:3], axis=2)
        else:
            uncertainty = img_array[:,:,0]
    else:
        uncertainty = img_array
    
    # 归一化到0-1范围
    uncertainty = uncertainty.astype(np.float32)
    uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    
    print(f"处理后的不确定性形状: {uncertainty.shape}")
    print(f"归一化后数值范围: {uncertainty.min():.6f} - {uncertainty.max():.6f}")
    
    return uncertainty

def load_mask(mask_path):
    """从NPY文件加载mask数据"""
    mask = np.load(mask_path)
    print(f"从 {mask_path} 加载mask数据，原始形状: {mask.shape}")
    
    # 如果mask有多余的维度，压缩掉
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    print(f"处理后的mask形状: {mask.shape}")
    print(f"Mask数值范围: {mask.min()} - {mask.max()}")
    print(f"Mask覆盖率: {(mask > 0).sum() / mask.size * 100:.1f}%")
    
    return mask

def enhance_uncertainty_with_mask(uncertainty, mask, enhancement_factor=2.0, brightness_boost=1.5, mask_threshold=0.1, global_brightness=0.8):
    """
    使用mask调整不确定性可视化
    
    参数:
    - uncertainty: 不确定性数组
    - mask: mask数组 (0表示背景，>0表示前景)
    - enhancement_factor: mask区域的调整倍数（<1为降低，>1为增强）
    - brightness_boost: mask区域的额外亮度调整倍数
    - mask_threshold: mask阈值，只有大于此值的区域才被认为是前景
    - global_brightness: 全局亮度调整倍数
    """
    # 确保形状匹配
    if uncertainty.shape != mask.shape:
        print(f"警告: 不确定性形状 {uncertainty.shape} 与mask形状 {mask.shape} 不匹配")
        return uncertainty
    
    # 创建调整后的不确定性副本
    enhanced = uncertainty.copy()
    
    # 先应用全局亮度调整
    enhanced = enhanced * global_brightness
    
    # 对mask区域进行调整（降低不确定性值）
    mask_region = mask > mask_threshold
    if mask_region.any():
        # 对mask区域应用降低因子，使其不确定性更低
        mask_adjustment = enhancement_factor * brightness_boost
        enhanced[mask_region] = uncertainty[mask_region] * global_brightness * mask_adjustment
        
        print(f"使用mask阈值: {mask_threshold}")
        print(f"全局亮度调整: {global_brightness:.1f}")
        print(f"调整了 {mask_region.sum()} 个像素点 ({mask_region.sum()/mask.size*100:.1f}%)")
        print(f"原始不确定性范围: {uncertainty.min():.6f} - {uncertainty.max():.6f}")
        print(f"调整后范围: {enhanced.min():.6f} - {enhanced.max():.6f}")
        print(f"背景区域亮度: {global_brightness:.1f}倍，mask区域调整: {global_brightness * mask_adjustment:.2f}倍")
    else:
        print(f"警告: 使用阈值 {mask_threshold} 后没有找到mask区域")
        print(f"应用全局亮度调整: {global_brightness:.1f}")
    
    return enhanced

def save_clean_image(data, output_path, colormap='viridis', dpi=150, vmin=0.0, vmax=1.0):
    """保存干净的图像（无坐标轴、标题等），保持原始宽高比，固定colormap范围"""
    # 计算数据的宽高比
    height, width = data.shape
    aspect_ratio = width / height
    
    # 根据宽高比设置figsize，保持相同的高度
    fig_height = 6
    fig_width = fig_height * aspect_ratio
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # 使用给定的colormap，保持原始宽高比，固定colormap范围
    plt.imshow(data, cmap=colormap, aspect='equal', vmin=vmin, vmax=vmax)
    
    # 移除所有UI元素
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])  # 填满整个图像
    
    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()
    
    print(f"已保存图像到: {output_path} (尺寸: {width}x{height}, 宽高比: {aspect_ratio:.3f}, colormap范围: {vmin}-{vmax})")

def main():
    # ====== 用户可调整参数 ======
    # 整体降低幅度：控制所有区域的亮度（0.0-1.0，越小越暗）
    GLOBAL_BRIGHTNESS = 0.5
    
    # 篮子增强幅度：控制篮子相对于背景的突出程度（>1.0为增强，<1.0为降低）
    BASKET_ENHANCEMENT = 3
    
    # mask阈值：控制哪些区域被认为是篮子（通常保持0.1即可）
    MASK_THRESHOLD = 0.1
    # ===========================
    
    # 使用PNG文件读取正确的不确定性数据
    uncertainty_file = "LF/basket/output/test/ours_4000/uncertainty/basket_01601_Cropped.png"
    mask_file = "LF/basket/mask/basket_01601_Cropped_prob.npy"
    
    # 检查文件是否存在
    if not os.path.exists(uncertainty_file):
        print(f"错误: 找不到不确定性文件 {uncertainty_file}")
        return
    if not os.path.exists(mask_file):
        print(f"错误: 找不到mask文件 {mask_file}")
        return
    
    # 加载数据
    print("加载不确定性数据...")
    uncertainty = load_uncertainty_from_png(uncertainty_file)
    
    print("\n加载mask数据...")
    mask = load_mask(mask_file)
    
    if uncertainty is None or mask is None:
        print("数据加载失败")
        return
    
    # 创建输出目录
    output_dir = "final_enhanced_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用用户设置的参数
    enhancement_factor = BASKET_ENHANCEMENT  # 篮子增强幅度
    brightness_boost = 1.0    # 固定为1.0，简化控制
    mask_threshold = MASK_THRESHOLD      # mask阈值
    global_brightness = GLOBAL_BRIGHTNESS   # 整体亮度
    
    print(f"\n当前参数设置:")
    print(f"  - 整体亮度: {global_brightness} (原始的{global_brightness*100:.0f}%)")
    print(f"  - 篮子增强: {enhancement_factor} (比背景高{(enhancement_factor-1)*100:.0f}%)")
    print(f"  - mask阈值: {mask_threshold}")
    
    print(f"\n应用调整...")
    enhanced_uncertainty = enhance_uncertainty_with_mask(
        uncertainty, mask, enhancement_factor, brightness_boost, mask_threshold, global_brightness
    )
    
    # 保存不同colormap的结果，固定colormap范围以显示真实的降低效果
    colormaps = ['viridis', 'plasma', 'hot', 'cool']
    
    # 固定colormap范围为原始的0-1，这样降低后的值会显示为更暗的颜色
    for cmap in colormaps:
        output_path = os.path.join(output_dir, f"enhanced_basket_01601_{cmap}.png")
        save_clean_image(enhanced_uncertainty, output_path, colormap=cmap, vmin=0.0, vmax=1.0)
    
    print(f"\n所有增强图像已保存到 {output_dir}/ 目录")
    print("生成的文件:")
    for cmap in colormaps:
        filename = f"enhanced_basket_01601_{cmap}.png"
        print(f"  - {filename}")

if __name__ == "__main__":
    main()
