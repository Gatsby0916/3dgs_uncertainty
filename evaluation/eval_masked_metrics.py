#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_masked_metrics.sh - 批量评估最终test结果的脚本

使用示例：
./eval_masked_metrics.sh
"""

import subprocess
import sys
from pathlib import Path

def find_final_test_results(base_dir="data"):
    """查找所有最终test结果目录"""
    base_path = Path(base_dir)
    test_results = []
    
    if not base_path.exists():
        print(f"❌ 基础目录不存在: {base_dir}")
        return []
    
    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir():
            # 检查NBV结果 (output/test/)
            nbv_test_dir = dataset_dir / "output" / "test"
            if nbv_test_dir.exists():
                ours_dirs = list(nbv_test_dir.glob("ours_*"))
                if ours_dirs:
                    iteration_dirs = []
                    for d in ours_dirs:
                        try:
                            iter_num = int(d.name.split('_')[-1])
                            iteration_dirs.append((iter_num, d))
                        except ValueError:
                            continue
                    
                    if iteration_dirs:
                        iteration_dirs.sort(key=lambda x: x[0], reverse=True)
                        _, latest_dir = iteration_dirs[0]
                        test_results.append(str(latest_dir))
                        print(f"✅ 找到NBV结果: {latest_dir}")
            
            # 检查Random结果 (random_output/test/)
            random_test_dir = dataset_dir / "random_output" / "test"
            if random_test_dir.exists():
                ours_dirs = list(random_test_dir.glob("ours_*"))
                if ours_dirs:
                    iteration_dirs = []
                    for d in ours_dirs:
                        try:
                            iter_num = int(d.name.split('_')[-1])
                            iteration_dirs.append((iter_num, d))
                        except ValueError:
                            continue
                    
                    if iteration_dirs:
                        iteration_dirs.sort(key=lambda x: x[0], reverse=True)
                        _, latest_dir = iteration_dirs[0]
                        test_results.append(str(latest_dir))
                        print(f"✅ 找到Random结果: {latest_dir}")
    
    return test_results

def main():
    print("🔍 查找最终test结果...")
    
    # 查找所有test结果
    test_results = find_final_test_results()
    
    if not test_results:
        print("❌ 未找到任何test结果")
        return
    
    print(f"📁 找到 {len(test_results)} 个test结果目录")
    
    # 构建评估命令
    cmd = [
        "python", "evaluation/masked_metrics.py"
    ] + test_results + [
        "--output-json", "final_masked_evaluation.json"
    ]
    
    print("🚀 开始评估...")
    print(f"命令: {' '.join(cmd)}")
    
    # 执行评估
    try:
        subprocess.run(cmd, check=True)
        print("✅ 评估完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 评估失败: {e}")
        return
    
    print("\n📊 查看结果:")
    print("   - JSON结果: final_masked_evaluation.json")
    print("   - 过滤后图像: masked_results_*/ 目录")

if __name__ == "__main__":
    main()
