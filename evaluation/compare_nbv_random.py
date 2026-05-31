#!/usr/bin/env python3
"""
compare_nbv_random.py - 对比NBV和随机选择的结果
"""

import json
import glob
from pathlib import Path

def compare_methods():
    """对比NBV和随机选择方法的结果"""
    
    print("🔬 NBV vs Random 方法对比分析")
    print("=" * 70)
    
    # 收集NBV结果
    nbv_files = glob.glob("results/*/results.json")
    nbv_results = {}
    
    print("📊 NBV结果:")
    for results_file in sorted(nbv_files):
        dataset_name = Path(results_file).parent.name
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            for method, metrics in data.items():
                nbv_results[dataset_name] = {
                    'psnr': metrics.get('PSNR', 0),
                    'ssim': metrics.get('SSIM', 0),
                    'lpips': metrics.get('LPIPS', 0)
                }
                print(f"   {dataset_name:12} | PSNR: {nbv_results[dataset_name]['psnr']:8.4f} | SSIM: {nbv_results[dataset_name]['ssim']:.4f} | LPIPS: {nbv_results[dataset_name]['lpips']:.4f}")
        
        except Exception as e:
            print(f"❌ {dataset_name}: NBV结果读取失败 - {e}")
    
    # 收集Random结果
    random_files = glob.glob("random_results/*/results.json")
    random_results = {}
    
    print("\n🎲 Random结果:")
    for results_file in sorted(random_files):
        dataset_name = Path(results_file).parent.name
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            for method, metrics in data.items():
                random_results[dataset_name] = {
                    'psnr': metrics.get('PSNR', 0),
                    'ssim': metrics.get('SSIM', 0),
                    'lpips': metrics.get('LPIPS', 0)
                }
                print(f"   {dataset_name:12} | PSNR: {random_results[dataset_name]['psnr']:8.4f} | SSIM: {random_results[dataset_name]['ssim']:.4f} | LPIPS: {random_results[dataset_name]['lpips']:.4f}")
        
        except Exception as e:
            print(f"❌ {dataset_name}: Random结果读取失败 - {e}")
    
    # 对比分析
    print("\n" + "=" * 70)
    print("📈 逐数据集对比:")
    print(f"{'数据集':<12} | {'方法':<8} | {'PSNR':<8} | {'SSIM':<8} | {'LPIPS':<8} | {'优势'}")
    print("-" * 70)
    
    common_datasets = set(nbv_results.keys()) & set(random_results.keys())
    
    nbv_wins = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    random_wins = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    
    for dataset in sorted(common_datasets):
        nbv = nbv_results[dataset]
        rand = random_results[dataset]
        
        # 确定每个指标的优势
        psnr_winner = "NBV" if nbv['psnr'] > rand['psnr'] else "Random"
        ssim_winner = "NBV" if nbv['ssim'] > rand['ssim'] else "Random"
        lpips_winner = "NBV" if nbv['lpips'] < rand['lpips'] else "Random"  # LPIPS越小越好
        
        # 统计获胜次数
        if psnr_winner == "NBV": nbv_wins['psnr'] += 1
        else: random_wins['psnr'] += 1
        
        if ssim_winner == "NBV": nbv_wins['ssim'] += 1
        else: random_wins['ssim'] += 1
        
        if lpips_winner == "NBV": nbv_wins['lpips'] += 1
        else: random_wins['lpips'] += 1
        
        # 显示对比
        print(f"{dataset:<12} | {'NBV':<8} | {nbv['psnr']:<8.4f} | {nbv['ssim']:<8.4f} | {nbv['lpips']:<8.4f} |")
        print(f"{'':<12} | {'Random':<8} | {rand['psnr']:<8.4f} | {rand['ssim']:<8.4f} | {rand['lpips']:<8.4f} | PSNR:{psnr_winner}, SSIM:{ssim_winner}, LPIPS:{lpips_winner}")
        print("-" * 70)
    
    # 总体统计
    if common_datasets:
        print("📊 总体平均值对比:")
        
        # NBV平均值
        nbv_avg_psnr = sum(nbv_results[d]['psnr'] for d in common_datasets) / len(common_datasets)
        nbv_avg_ssim = sum(nbv_results[d]['ssim'] for d in common_datasets) / len(common_datasets)
        nbv_avg_lpips = sum(nbv_results[d]['lpips'] for d in common_datasets) / len(common_datasets)
        
        # Random平均值
        rand_avg_psnr = sum(random_results[d]['psnr'] for d in common_datasets) / len(common_datasets)
        rand_avg_ssim = sum(random_results[d]['ssim'] for d in common_datasets) / len(common_datasets)
        rand_avg_lpips = sum(random_results[d]['lpips'] for d in common_datasets) / len(common_datasets)
        
        print(f"NBV 平均      | PSNR: {nbv_avg_psnr:8.4f} | SSIM: {nbv_avg_ssim:.4f} | LPIPS: {nbv_avg_lpips:.4f}")
        print(f"Random 平均   | PSNR: {rand_avg_psnr:8.4f} | SSIM: {rand_avg_ssim:.4f} | LPIPS: {rand_avg_lpips:.4f}")
        
        # 计算改进百分比
        psnr_improvement = ((nbv_avg_psnr - rand_avg_psnr) / rand_avg_psnr) * 100
        ssim_improvement = ((nbv_avg_ssim - rand_avg_ssim) / rand_avg_ssim) * 100
        lpips_improvement = ((rand_avg_lpips - nbv_avg_lpips) / rand_avg_lpips) * 100  # LPIPS越小越好
        
        print(f"\n📈 NBV相对Random的改进:")
        print(f"PSNR: {psnr_improvement:+6.2f}%")
        print(f"SSIM: {ssim_improvement:+6.2f}%") 
        print(f"LPIPS: {lpips_improvement:+6.2f}%")
        
        print(f"\n🏆 获胜统计 (共{len(common_datasets)}个数据集):")
        print(f"PSNR:  NBV {nbv_wins['psnr']}, Random {random_wins['psnr']}")
        print(f"SSIM:  NBV {nbv_wins['ssim']}, Random {random_wins['ssim']}")
        print(f"LPIPS: NBV {nbv_wins['lpips']}, Random {random_wins['lpips']}")
        
        # 总体结论
        total_nbv_wins = sum(nbv_wins.values())
        total_random_wins = sum(random_wins.values())
        
        print(f"\n🎯 总体结论:")
        print(f"NBV获胜次数: {total_nbv_wins}/{len(common_datasets)*3}")
        print(f"Random获胜次数: {total_random_wins}/{len(common_datasets)*3}")
        
        if total_nbv_wins > total_random_wins:
            print("🎉 NBV方法总体表现更优!")
        elif total_random_wins > total_nbv_wins:
            print("😮 Random方法总体表现更优!")
        else:
            print("🤝 两种方法表现相当!")
            
    else:
        print("❌ 没有找到可对比的数据集")

if __name__ == "__main__":
    compare_methods()
