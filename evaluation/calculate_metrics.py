#!/usr/bin/env python3
"""
calculate_metrics.py - 计算results.json中的平均PSNR、SSIM、LPIPS
"""

import json
import sys
from pathlib import Path

def calculate_average_metrics(results_file):
    """计算指定results.json文件的平均metrics"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"📊 分析文件: {results_file}")
        print("=" * 50)
        
        all_psnr = []
        all_ssim = []
        all_lpips = []
        
        for method, metrics in results.items():
            psnr = metrics.get('PSNR', 0)
            ssim = metrics.get('SSIM', 0)
            lpips = metrics.get('LPIPS', 0)
            
            print(f"方法: {method}")
            print(f"  PSNR:  {psnr:.4f}")
            print(f"  SSIM:  {ssim:.4f}")
            print(f"  LPIPS: {lpips:.4f}")
            print()
            
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(lpips)
        
        if all_psnr:
            avg_psnr = sum(all_psnr) / len(all_psnr)
            avg_ssim = sum(all_ssim) / len(all_ssim)
            avg_lpips = sum(all_lpips) / len(all_lpips)
            
            print("📈 平均结果:")
            print(f"  平均 PSNR:  {avg_psnr:.4f}")
            print(f"  平均 SSIM:  {avg_ssim:.4f}")
            print(f"  平均 LPIPS: {avg_lpips:.4f}")
            print()
            
            # 提供一个简洁的摘要
            print("📋 摘要:")
            print(f"PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
            
            return {
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'avg_lpips': avg_lpips,
                'method_count': len(all_psnr)
            }
        else:
            print("❌ 没有找到有效的metrics数据")
            return None
            
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None

def main():
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = "data/bonsai/output/results.json"
    
    if not Path(results_file).exists():
        print(f"❌ 文件不存在: {results_file}")
        sys.exit(1)
    
    calculate_average_metrics(results_file)

if __name__ == "__main__":
    main()
