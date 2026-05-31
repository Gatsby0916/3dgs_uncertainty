#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
masked_metrics.py - 基于mask计算object区域的重建质量指标

功能：
1. 使用mask过滤GT和rendered图像
2. 只在object区域计算PSNR、SSIM、LPIPS
3. 支持批量处理多个数据集
4. 生成详细的评估报告
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from lpipsPyTorch.modules import lpips
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在3dgs_uncertainty目录下运行此脚本")
    sys.exit(1)


class MaskedMetrics:
    """基于mask的重建质量评估器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net_type='vgg').to(device)
        self.to_tensor = transforms.ToTensor()
        
    def load_image(self, image_path):
        """加载图像并转换为tensor"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.to_tensor(image).to(self.device)
        except Exception as e:
            print(f"❌ 加载图像失败 {image_path}: {e}")
            return None
    
    def load_mask(self, mask_path):
        """加载mask文件"""
        try:
            mask_path = Path(mask_path)
            
            if mask_path.suffix == '.npy':
                # 加载.npy文件
                mask = np.load(mask_path)
                # 如果是概率图，转换为二值mask
                if mask.max() <= 1.0:
                    mask = (mask > 0.5).astype(np.float32)
                else:
                    mask = (mask > 127).astype(np.float32)
            else:
                # 加载图像文件
                mask = np.array(Image.open(mask_path).convert('L'))
                mask = mask.astype(np.float32)
            
            mask_tensor = torch.from_numpy(mask).float().to(self.device)
            
            # 确保mask是2D的
            if len(mask_tensor.shape) == 3:
                mask_tensor = mask_tensor.mean(dim=0)
                
            return mask_tensor
            
        except Exception as e:
            print(f"❌ 加载mask失败 {mask_path}: {e}")
            return None
    
    def apply_mask(self, image_tensor, mask_tensor):
        """将mask应用到图像上"""
        # 确保mask的尺寸匹配
        if mask_tensor.shape != image_tensor.shape[-2:]:
            # 调整mask尺寸 - 确保维度正确
            if len(mask_tensor.shape) == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
            elif len(mask_tensor.shape) == 3:
                mask_tensor = mask_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor,
                size=image_tensor.shape[-2:],
                mode='nearest'
            ).squeeze()
        
        # 将mask扩展到3个通道
        if len(mask_tensor.shape) == 2:
            mask_3d = mask_tensor.unsqueeze(0).expand_as(image_tensor)
        else:
            mask_3d = mask_tensor.expand_as(image_tensor)
        
        # 应用mask
        masked_image = image_tensor * mask_3d
        
        return masked_image, mask_3d
    
    def compute_masked_psnr(self, img1, img2, mask):
        """计算mask区域的PSNR"""
        # 只在mask区域计算
        valid_pixels = mask > 0.5
        
        if valid_pixels.sum() == 0:
            return float('nan')
        
        # 提取有效像素
        img1_masked = img1[valid_pixels]
        img2_masked = img2[valid_pixels]
        
        # 计算MSE
        mse = torch.mean((img1_masked - img2_masked) ** 2)
        
        if mse == 0:
            return float('inf')
        
        psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr_value.item()
    
    def compute_masked_ssim(self, img1, img2, mask):
        """计算mask区域的SSIM"""
        # 应用mask
        img1_masked = img1 * mask
        img2_masked = img2 * mask
        
        # 使用项目中的ssim函数
        ssim_value = ssim(img1_masked, img2_masked)
        return ssim_value.item()
    
    def compute_masked_lpips(self, img1, img2, mask):
        """计算mask区域的LPIPS"""
        # 应用mask
        img1_masked = img1 * mask
        img2_masked = img2 * mask
        
        # 添加batch维度
        img1_batch = img1_masked.unsqueeze(0)
        img2_batch = img2_masked.unsqueeze(0)
        
        # 计算LPIPS
        lpips_value = self.lpips_fn(img1_batch, img2_batch)
        return lpips_value.item()

    def save_masked_images(self, gt_img, render_img, mask, output_dir, image_name):
        """保存过滤后的GT和render图像"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 应用mask
        gt_masked, mask_3d = self.apply_mask(gt_img, mask)
        render_masked, _ = self.apply_mask(render_img, mask)
        
        # 转换为PIL图像并保存
        to_pil = transforms.ToPILImage()
        
        # 保存过滤后的GT
        gt_pil = to_pil(gt_masked.cpu())
        gt_path = output_dir / f"{image_name}_gt_masked.png"
        gt_pil.save(gt_path)
        
        # 保存过滤后的render
        render_pil = to_pil(render_masked.cpu())
        render_path = output_dir / f"{image_name}_render_masked.png"
        render_pil.save(render_path)
        
        # 保存mask（可视化用）
        # 确保mask是2D的以便保存
        mask_for_save = mask.cpu()
        if len(mask_for_save.shape) > 2:
            mask_for_save = mask_for_save.squeeze()
        if len(mask_for_save.shape) == 2:
            mask_for_save = mask_for_save.unsqueeze(0)  # 添加通道维度
        mask_pil = to_pil(mask_for_save)
        mask_path = output_dir / f"{image_name}_mask.png"
        mask_pil.save(mask_path)
        
        return {
            'gt_masked': str(gt_path),
            'render_masked': str(render_path),
            'mask': str(mask_path)
        }
    
    def find_matching_files(self, gt_dir, render_dir, mask_dir):
        """找到匹配的GT、render和mask文件（文件名大小写不敏感）"""
        gt_files = list(Path(gt_dir).glob('*.png')) + list(Path(gt_dir).glob('*.jpg'))
        matches = []

        mask_index = {p.name.lower(): p for p in Path(mask_dir).iterdir() if p.is_file()}

        def find_mask(stem):
            for ext in ('.npy', '.png', '.jpg'):
                for cand in (f"{stem}{ext}", f"{stem}_prob{ext}"):
                    hit = mask_index.get(cand.lower())
                    if hit is not None:
                        return hit
            return None

        for gt_file in gt_files:
            stem = gt_file.stem

            render_file = None
            for ext in ['.png', '.jpg']:
                render_path = Path(render_dir) / f"{stem}{ext}"
                if render_path.exists():
                    render_file = render_path
                    break

            if render_file is None:
                print(f"⚠️  未找到render文件: {stem}")
                continue

            mask_file = find_mask(stem)
            if mask_file is None:
                print(f"⚠️  未找到mask文件: {stem} (尝试了 {stem}.npy, {stem}_prob.npy 等)")
                continue
            
            matches.append({
                'gt': gt_file,
                'render': render_file,
                'mask': mask_file,
                'name': stem
            })
        
        return matches
    
    def evaluate_final_test(self, output_path, save_images=True, output_dir=None, mask_dir_override=None):
        """评估最终test结果（如 /hdd/gatsbyli/3dgs_output/stump/test/ours_29999）"""
        output_path = Path(output_path)
        
        # 从路径中提取信息
        dataset_name = output_path.parent.parent.name  # stump
        iteration = int(output_path.name.split('_')[-1])  # 29999
        
        print(f"📊 评估最终test结果:")
        print(f"   数据集: {dataset_name}")
        print(f"   Iteration: {iteration}")
        print(f"   Render目录: {output_path}")
        
        # 检查test结果目录结构
        render_dir = output_path / "renders"
        gt_dir = output_path / "gt"
        
        if mask_dir_override:
            mask_dir = Path(mask_dir_override)
            if not mask_dir.exists():
                print(f"❌ 指定的mask目录不存在: {mask_dir}")
                # 尝试从data目录找
                mask_dir = Path("data") / dataset_name / "mask"
        else:
            # 默认逻辑: 尝试从data目录找
            mask_dir = Path("data") / dataset_name / "mask"
            if not mask_dir.exists():
                 # 之前的逻辑回退
                if "3dgs_output" in str(output_path):
                    data_dir = Path("data") / dataset_name
                else:
                    data_dir = output_path.parent.parent.parent
                mask_dir = data_dir / "mask"

        # 检查目录是否存在
        if not render_dir.exists():
            print(f"❌ render目录不存在: {render_dir}")
            return None
        
        if not gt_dir.exists():
            print(f"❌ GT目录不存在: {gt_dir}")
            return None
        
        if not mask_dir.exists():
            print(f"❌ mask目录不存在: {mask_dir}")
            return None
        
        # 查找匹配的文件
        matches = self.find_matching_files(gt_dir, render_dir, mask_dir)
        
        if not matches:
            print(f"❌ 未找到匹配的文件")
            return None
        
        print(f"📁 找到 {len(matches)} 个匹配的文件")
        
        # 设置输出目录
        if save_images:
            if output_dir is None:
                output_dir = Path(f"masked_results_{dataset_name}_{iteration}")
            else:
                output_dir = Path(output_dir)
            
            print(f"💾 过滤后的图像将保存到: {output_dir}")
        
        # 计算指标
        results = {
            'dataset': dataset_name,
            'iteration': iteration,
            'render_path': str(render_dir),
            'gt_path': str(gt_dir),
            'mask_path': str(mask_dir),
            'total_images': len(matches),
            'metrics': {
                'PSNR': [],
                'SSIM': [],
                'LPIPS': []
            },
            'per_image': {},
            'saved_images': {} if save_images else None
        }
        
        for match in tqdm(matches, desc="计算指标并保存图像"):
            # 加载图像和mask
            gt_img = self.load_image(match['gt'])
            render_img = self.load_image(match['render'])
            mask = self.load_mask(match['mask'])
            
            if gt_img is None or render_img is None or mask is None:
                continue
            
            # 应用mask
            gt_masked, mask_3d = self.apply_mask(gt_img, mask)
            render_masked, _ = self.apply_mask(render_img, mask)
            
            # 计算指标
            psnr_val = self.compute_masked_psnr(gt_masked, render_masked, mask_3d)
            ssim_val = self.compute_masked_ssim(gt_masked, render_masked, mask_3d)
            lpips_val = self.compute_masked_lpips(gt_masked, render_masked, mask_3d)
            
            # 记录结果
            if not np.isnan(psnr_val):
                results['metrics']['PSNR'].append(psnr_val)
            if not np.isnan(ssim_val):
                results['metrics']['SSIM'].append(ssim_val)
            if not np.isnan(lpips_val):
                results['metrics']['LPIPS'].append(lpips_val)
            
            results['per_image'][match['name']] = {
                'PSNR': psnr_val,
                'SSIM': ssim_val,
                'LPIPS': lpips_val
            }
            
            # 保存过滤后的图像
            if save_images:
                saved_paths = self.save_masked_images(
                    gt_img, render_img, mask, output_dir, match['name']
                )
                results['saved_images'][match['name']] = saved_paths
        
        # 计算平均值
        summary = {}
        for metric, values in results['metrics'].items():
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
                summary[f'{metric}_count'] = len(values)
            else:
                summary[f'{metric}_mean'] = float('nan')
                summary[f'{metric}_std'] = float('nan')
                summary[f'{metric}_min'] = float('nan')
                summary[f'{metric}_max'] = float('nan')
                summary[f'{metric}_count'] = 0
        
        results['summary'] = summary
        
        # 打印结果
        print(f"\n📈 {dataset_name} (iter {iteration}) 最终评估结果 (仅object区域):")
        print(f"   有效图像数: {summary['PSNR_count']}/{results['total_images']}")
        print(f"   PSNR: {summary['PSNR_mean']:.4f} ± {summary['PSNR_std']:.4f}")
        print(f"   SSIM: {summary['SSIM_mean']:.4f} ± {summary['SSIM_std']:.4f}")
        print(f"   LPIPS: {summary['LPIPS_mean']:.4f} ± {summary['LPIPS_std']:.4f}")
        
        if save_images:
            print(f"   过滤后图像已保存到: {output_dir}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="评估最终test结果的object区域重建质量")
    parser.add_argument('render_paths', nargs='+', 
                        help='最终render结果路径（如 /hdd/gatsbyli/3dgs_output/stump/test/ours_29999）')
    parser.add_argument('--output-dir', help='保存过滤后图像的目录')
    parser.add_argument('--no-save-images', action='store_true', help='不保存过滤后的图像')
    parser.add_argument('--output-json', help='输出结果JSON文件路径')
    parser.add_argument('--device', default='cuda', help='计算设备')
    parser.add_argument('--mask-dir', help='指定mask目录 (默认: data/<dataset>/mask)')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = MaskedMetrics(device=args.device)
    
    all_results = []
    save_images = not args.no_save_images
    
    # 评估每个render路径
    for render_path in args.render_paths:
        try:
            print(f"\n{'='*60}")
            result = evaluator.evaluate_final_test(
                render_path, 
                save_images=save_images,
                output_dir=args.output_dir,
                mask_dir_override=args.mask_dir
            )
            
            if result:
                all_results.append(result)
            
        except Exception as e:
            print(f"❌ 评估 {render_path} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("❌ 没有成功评估的结果")
        return
    
    # 计算总体统计
    overall_stats = {
        'evaluation_type': 'final_test_masked',
        'total_datasets': len(all_results),
        'total_images': sum(r['total_images'] for r in all_results),
        'results': all_results
    }
    
    # 计算所有数据集的平均指标
    all_psnr = []
    all_ssim = []
    all_lpips = []
    
    for result in all_results:
        all_psnr.extend(result['metrics']['PSNR'])
        all_ssim.extend(result['metrics']['SSIM'])
        all_lpips.extend(result['metrics']['LPIPS'])
    
    if all_psnr:
        overall_stats['overall_metrics'] = {
            'PSNR_mean': np.mean(all_psnr),
            'PSNR_std': np.std(all_psnr),
            'SSIM_mean': np.mean(all_ssim),
            'SSIM_std': np.std(all_ssim),
            'LPIPS_mean': np.mean(all_lpips),
            'LPIPS_std': np.std(all_lpips),
            'total_valid_images': len(all_psnr)
        }
        
        print(f"\n{'='*60}")
        print(f"🎯 总体最终评估结果 (仅object区域):")
        print(f"   数据集数量: {len(all_results)}")
        print(f"   总图像数: {sum(r['total_images'] for r in all_results)}")
        print(f"   有效图像数: {len(all_psnr)}")
        print(f"   PSNR: {overall_stats['overall_metrics']['PSNR_mean']:.4f} ± {overall_stats['overall_metrics']['PSNR_std']:.4f}")
        print(f"   SSIM: {overall_stats['overall_metrics']['SSIM_mean']:.4f} ± {overall_stats['overall_metrics']['SSIM_std']:.4f}")
        print(f"   LPIPS: {overall_stats['overall_metrics']['LPIPS_mean']:.4f} ± {overall_stats['overall_metrics']['LPIPS_std']:.4f}")
        print(f"{'='*60}")
    
    # 保存结果
    if args.output_json:
        output_path = Path(args.output_json)
    else:
        output_path = Path("final_masked_metrics_results.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 结果已保存到: {output_path}")
    
    if save_images:
        print(f"🖼️  过滤后的图像已保存到各自的输出目录")


if __name__ == "__main__":
    main()
