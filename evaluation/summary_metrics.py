#!/usr/bin/env python3
"""
summary_metrics.py - 汇总所有数据集的metrics结果
"""

import json
import glob
from pathlib import Path

def collect_all_results():
    """收集所有数据集的结果"""
    results_files = glob.glob("data/*/output/results.json")
    
    print("🎯 NBV Pipeline Results Summary")
    print("=" * 60)
    
    all_results = []
    
    for results_file in sorted(results_files):
        dataset_name = Path(results_file).parent.parent.name
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            for method, metrics in data.items():
                result = {
                    'dataset': dataset_name,
                    'method': method,
                    'psnr': metrics.get('PSNR', 0),
                    'ssim': metrics.get('SSIM', 0),
                    'lpips': metrics.get('LPIPS', 0)
                }
                all_results.append(result)
                
                print(f"📊 {dataset_name:12} | PSNR: {result['psnr']:8.4f} | SSIM: {result['ssim']:.4f} | LPIPS: {result['lpips']:.4f}")
        
        except Exception as e:
            print(f"❌ {dataset_name}: 读取失败 - {e}")
    
    if all_results:
        print("-" * 60)
        
        # 计算总体平均值
        avg_psnr = sum(r['psnr'] for r in all_results) / len(all_results)
        avg_ssim = sum(r['ssim'] for r in all_results) / len(all_results)
        avg_lpips = sum(r['lpips'] for r in all_results) / len(all_results)
        
        print(f"📈 总体平均    | PSNR: {avg_psnr:8.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
        print(f"📋 数据集数量: {len(all_results)}")
        
        # 找到最佳和最差结果
        best_psnr = max(all_results, key=lambda x: x['psnr'])
        worst_psnr = min(all_results, key=lambda x: x['psnr'])
        best_ssim = max(all_results, key=lambda x: x['ssim'])
        best_lpips = min(all_results, key=lambda x: x['lpips'])  # LPIPS越小越好
        
        print("\n🏆 最佳结果:")
        print(f"   最高PSNR:  {best_psnr['dataset']} ({best_psnr['psnr']:.4f})")
        print(f"   最高SSIM:  {best_ssim['dataset']} ({best_ssim['ssim']:.4f})")
        print(f"   最低LPIPS: {best_lpips['dataset']} ({best_lpips['lpips']:.4f})")
        
        print(f"\n📝 配置信息:")
        print(f"   • patch_size = 4")
        print(f"   • NBV 触发点: 16个")
        print(f"   • 总训练迭代: 30000")
        
        return {
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim, 
            'avg_lpips': avg_lpips,
            'dataset_count': len(all_results),
            'results': all_results
        }
    else:
        print("❌ 没有找到任何结果文件")
        return None

if __name__ == "__main__":
    collect_all_results()
