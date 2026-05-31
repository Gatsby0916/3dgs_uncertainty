#!/usr/bin/env python3
"""
combined_metrics.py - 计算现有结果和新数据的综合平均值
"""

import json
import glob
from pathlib import Path

def calculate_combined_metrics():
    """计算现有结果和新提供数据的综合平均值"""
    
    print("🎯 综合Metrics计算")
    print("=" * 60)
    
    # 1. 收集现有数据集结果
    results_files = glob.glob("data/*/output/results.json")
    existing_results = []
    
    print("📊 现有数据集结果:")
    for results_file in sorted(results_files):
        dataset_name = Path(results_file).parent.parent.name
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            for method, metrics in data.items():
                result = {
                    'dataset': dataset_name,
                    'psnr': metrics.get('PSNR', 0),
                    'ssim': metrics.get('SSIM', 0),
                    'lpips': metrics.get('LPIPS', 0)
                }
                existing_results.append(result)
                print(f"   {dataset_name:12} | PSNR: {result['psnr']:8.4f} | SSIM: {result['ssim']:.4f} | LPIPS: {result['lpips']:.4f}")
        
        except Exception as e:
            print(f"❌ {dataset_name}: 读取失败 - {e}")
    
    # 2. 添加新提供的数据
    print("\n📋 新提供的数据:")
    new_data = [
        {'dataset': 'new_1', 'psnr': 20.3661, 'ssim': 0.5425, 'lpips': 0.3239},
        {'dataset': 'new_2', 'psnr': 16.1671, 'ssim': 0.3945, 'lpips': 0.4026},
        {'dataset': 'new_3', 'psnr': 25.4884, 'ssim': 0.8384, 'lpips': 0.1184},
        {'dataset': 'new_4', 'psnr': 22.5408, 'ssim': 0.7992, 'lpips': 0.1883},
        {'dataset': 'new_5', 'psnr': 21.4569, 'ssim': 0.5766, 'lpips': 0.3237},
        {'dataset': 'new_6', 'psnr': 17.8155, 'ssim': 0.4970, 'lpips': 0.3817}
    ]
    
    for result in new_data:
        print(f"   {result['dataset']:12} | PSNR: {result['psnr']:8.4f} | SSIM: {result['ssim']:.4f} | LPIPS: {result['lpips']:.4f}")
    
    # 3. 验证新数据的平均值
    print("\n📐 验证新数据平均值:")
    new_avg_psnr = sum(r['psnr'] for r in new_data) / len(new_data)
    new_avg_ssim = sum(r['ssim'] for r in new_data) / len(new_data)
    new_avg_lpips = sum(r['lpips'] for r in new_data) / len(new_data)
    
    print(f"   新数据平均   | PSNR: {new_avg_psnr:8.4f} | SSIM: {new_avg_ssim:.4f} | LPIPS: {new_avg_lpips:.4f}")
    print(f"   您计算的平均 | PSNR: {20.6391:8.4f} | SSIM: {0.6080:.4f} | LPIPS: {0.2898:.4f}")
    
    # 4. 计算现有数据的平均值
    if existing_results:
        print("\n📊 现有数据平均值:")
        existing_avg_psnr = sum(r['psnr'] for r in existing_results) / len(existing_results)
        existing_avg_ssim = sum(r['ssim'] for r in existing_results) / len(existing_results)
        existing_avg_lpips = sum(r['lpips'] for r in existing_results) / len(existing_results)
        
        print(f"   现有数据平均 | PSNR: {existing_avg_psnr:8.4f} | SSIM: {existing_avg_ssim:.4f} | LPIPS: {existing_avg_lpips:.4f}")
    
    # 5. 计算综合平均值
    all_results = existing_results + new_data
    
    print("\n" + "=" * 60)
    print("🎯 综合平均值计算:")
    
    combined_avg_psnr = sum(r['psnr'] for r in all_results) / len(all_results)
    combined_avg_ssim = sum(r['ssim'] for r in all_results) / len(all_results)
    combined_avg_lpips = sum(r['lpips'] for r in all_results) / len(all_results)
    
    print(f"📈 总数据集数量: {len(all_results)} (现有: {len(existing_results)}, 新增: {len(new_data)})")
    print(f"📊 综合平均值   | PSNR: {combined_avg_psnr:8.4f} | SSIM: {combined_avg_ssim:.4f} | LPIPS: {combined_avg_lpips:.4f}")
    
    # 6. 找到最佳结果
    print("\n🏆 所有数据中的最佳结果:")
    best_psnr = max(all_results, key=lambda x: x['psnr'])
    best_ssim = max(all_results, key=lambda x: x['ssim'])
    best_lpips = min(all_results, key=lambda x: x['lpips'])  # LPIPS越小越好
    
    print(f"   最高PSNR:  {best_psnr['dataset']} ({best_psnr['psnr']:.4f})")
    print(f"   最高SSIM:  {best_ssim['dataset']} ({best_ssim['ssim']:.4f})")
    print(f"   最低LPIPS: {best_lpips['dataset']} ({best_lpips['lpips']:.4f})")
    
    return {
        'combined_avg_psnr': combined_avg_psnr,
        'combined_avg_ssim': combined_avg_ssim,
        'combined_avg_lpips': combined_avg_lpips,
        'total_datasets': len(all_results),
        'existing_count': len(existing_results),
        'new_count': len(new_data)
    }

if __name__ == "__main__":
    calculate_combined_metrics()
