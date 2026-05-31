#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_pipeline.py - 统一的3DGS不确定性NBV pipeline

功能：
1. 支持多个数据集连续运行
2. patch_size=4的不确定性计算
3. 自动保存每个数据集的metrics结果
4. 完整的Fisher NBV pipeline
"""

import os
import sys
import time
import subprocess
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime


def log_message(message):
    """记录日志信息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    
    # 同时写入日志文件
    with open("pipeline_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def sh(cmd: str):
    """执行shell命令"""
    log_message(f"执行命令: {cmd}")
    # 使用实时输出，不捕获输出，让用户看到进度条
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        log_message(f"❌ 命令执行失败: {cmd}")
        log_message(f"返回码: {result.returncode}")
        return False
    return True


def flags_from_opt(opt: dict) -> list[str]:
    """将配置字典转换为命令行参数"""
    out = []
    for k, v in opt.items():
        if v is None or v == "":
            continue
        key = "--" + k
        if isinstance(v, bool):
            if v:
                out.append(key)
        else:
            out += [key, str(v)]
    return out


def setup_environment():
    """设置环境变量"""
    if "TMPDIR" not in os.environ or not Path(os.environ["TMPDIR"]).is_dir():
        tmp_path = Path.home() / "tmp"
        tmp_path.mkdir(exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_path)


def check_dataset_exists(dataset_path):
    """检查数据集是否存在"""
    path = Path(dataset_path)
    if not path.exists():
        log_message(f"⚠️ 数据集 {dataset_path} 不存在")
        return False
    
    # 检查必要的文件
    sparse_dir = path / "sparse" / "0"
    if not sparse_dir.exists():
        log_message(f"⚠️ 数据集 {dataset_path} 缺少sparse目录")
        return False
    
    return True


def save_metrics_results(dataset_path, output_dir):
    """保存metrics结果到数据集特定的文件"""
    try:
        # 读取metrics结果
        results_file = Path(output_dir) / "results.json"
        per_view_file = Path(output_dir) / "per_view.json"
        
        if results_file.exists():
            # 创建数据集专用的结果目录
            dataset_name = Path(dataset_path).name
            results_dir = Path("results") / dataset_name
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制结果文件
            shutil.copy2(results_file, results_dir / "results.json")
            if per_view_file.exists():
                shutil.copy2(per_view_file, results_dir / "per_view.json")
            
            # 读取并打印结果
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            log_message(f"✅ {dataset_name} 评估结果:")
            for method, metrics in results.items():
                log_message(f"  方法: {method}")
                log_message(f"    PSNR: {metrics.get('PSNR', 'N/A'):.4f}")
                log_message(f"    SSIM: {metrics.get('SSIM', 'N/A'):.4f}")
                log_message(f"    LPIPS: {metrics.get('LPIPS', 'N/A'):.4f}")
            
            return True
        else:
            log_message(f"⚠️ 未找到评估结果文件: {results_file}")
            return False
            
    except Exception as e:
        log_message(f"❌ 保存metrics结果失败: {str(e)}")
        return False


def run_fisher_pipeline(dataset_path, config):
    """运行单个数据集的Fisher NBV pipeline"""
    log_message(f"🚀 开始处理数据集: {dataset_path}")
    
    # 设置路径
    output = f"{dataset_path}/output"
    mask_dir = f"{dataset_path}/mask"
    split_txt = Path(dataset_path) / "train_split.txt"
    
    # 获取配置参数
    opt_dict = config.get("opt_params", {})
    opt_flags = flags_from_opt(opt_dict)
    densify_until = config.get("densify_until_iter") or opt_dict.get("densify_until_iter", 2000)
    
    # 1. 首先生成初始的4个视图 (使用gen_split.py)
    log_message(f"🎯 生成初始4个视图...")
    if not sh(f"python pipeline/gen_split.py {dataset_path}"):
        log_message(f"❌ 初始视图生成失败")
        return False
    log_message(f"✅ 初始视图生成完成: {split_txt}")
    
    prev_end_iter = 0
    first_seg = True
    recent_ids = []
    
    try:
        # NBV循环
        for step in config["nbv_schedule"]:
            end_iter = step["iter"]
            base_iter = prev_end_iter
            seg_iters = end_iter - base_iter
            save_id = end_iter
            
            log_message(f"NBV步骤: base_iter={base_iter}, end_iter={end_iter}, seg_iters={seg_iters}")
            
            # 1. 训练
            # 注意：训练循环是0到seg_iters-1，所以最大global_iter是base_iter+seg_iters-1
            # 为了确保保存，我们需要将save_id设置为实际能达到的最大值
            actual_save_id = base_iter + seg_iters - 1
            cmd = [
                "python", "train.py",
                "-s", dataset_path,
                "-m", output,
                "--iterations_per_segment", str(seg_iters),
                "--base_iter", str(base_iter),
                "--train_split", str(split_txt),
                "--save_iterations", str(actual_save_id),
                "--checkpoint_iterations", str(actual_save_id),
                "--eval",
                "--densify_until_iter", str(densify_until),
                "--sh_up_every", str(config.get("sh_up_every", 5000)),
                "--sh_up_after", str(config.get("sh_up_after", 1000)),
                "--min_opacity", str(config.get("min_opacity", 0.005))
            ] + opt_flags
            
            if not first_seg:
                cmd += ["--start_checkpoint", f"{output}/chkpnt{prev_end_iter - 1}.pth"]
            
            if not sh(" ".join(cmd)):
                return False
            first_seg = False
            
            # 2. 渲染不确定性 (使用patch_size=4)
            if not sh(
                f"python render.py "
                f"-m {output} "
                f"--iteration {actual_save_id} "
                f"--uncertainty_mode "
                f"--patch_size {config['uncert_mode']['patch_size']} "
                f"--mask-dir {mask_dir} "
                f"--skip_test"
            ):
                return False
            
            # 3. NBV打分
            if not sh(
                f"python pipeline/object_nbv_score.py "
                f"--uncert-dir {output}/train/ours_{actual_save_id}/uncertainty_npz "
                f"--mask-dir {mask_dir} "
                f"--thr {config['uncert_mode']['thr']} "
                f"--mode {config['uncert_mode']['mode']} "
                f"--out-json {output}/score_{actual_save_id}.json "
                f"--train-split {split_txt}"
            ):
                return False
            
            # 4. 清理旧文件 (保留最近2个)
            recent_ids.append(actual_save_id)
            if len(recent_ids) > 2:
                old_id = recent_ids.pop(0)
                for p in [
                    Path(output) / "train" / f"ours_{old_id}",
                    Path(output) / "point_cloud" / f"iteration_{old_id}",
                    Path(output) / f"chkpnt{old_id}.pth",
                    Path(output) / f"score_{old_id}.json",
                ]:
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    elif p.is_file():
                        p.unlink(missing_ok=True)
            
            prev_end_iter = end_iter
        
        # 最终训练
        total_iter = config["train"]["total_iterations"]
        base_iter = prev_end_iter
        seg_iters = total_iter - base_iter
        final_save_id = total_iter - 1  # 实际能达到的最大iteration
        
        log_message(f"最终训练: base_iter={base_iter}, total_iter={total_iter}, seg_iters={seg_iters}")
        
        final_cmd = [
            "python", "train.py",
            "-s", dataset_path, "-m", output,
            "--iterations_per_segment", str(seg_iters),
            "--base_iter", str(base_iter),
            "--start_checkpoint", f"{output}/chkpnt{prev_end_iter - 1}.pth",
            "--train_split", str(split_txt),
            "--save_iterations", str(final_save_id),
            "--checkpoint_iterations", str(final_save_id),
            "--sh_up_every", str(config.get("sh_up_every", 5000)),
            "--sh_up_after", str(config.get("sh_up_after", 1000)),
            "--min_opacity", str(config.get("min_opacity", 0.005)),
            "--eval",
            "--densify_until_iter", str(densify_until)
        ] + opt_flags
        
        if not sh(" ".join(final_cmd)):
            return False
        
        # 最终渲染和评估
        if not sh(f"python render.py -m {output} --iteration {final_save_id}"):
            return False
        
        log_message("=== 开始最终评估 ===")
        if not sh(f"python metrics.py -m {output} --split test"):
            return False
        
        # 保存结果
        save_metrics_results(dataset_path, output)
        
        log_message(f"✅ 数据集 {dataset_path} 处理完成")
        return True
        
    except Exception as e:
        log_message(f"❌ 数据集 {dataset_path} 处理出错: {str(e)}")
        return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python unified_pipeline.py pipeline_config.yml")
        sys.exit(1)
    
    # 设置环境
    setup_environment()
    
    # 读取配置
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        log_message(f"❌ 配置文件不存在: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取数据集列表
    datasets = config.get("datasets", [])
    if not datasets:
        log_message("❌ 配置文件中未找到数据集列表")
        sys.exit(1)
    
    log_message("🎯 开始批量处理数据集")
    log_message(f"计划处理的数据集: {datasets}")
    log_message(f"不确定性patch_size: {config['uncert_mode']['patch_size']}")
    
    total_start_time = time.time()
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    # 创建结果目录
    Path("results").mkdir(exist_ok=True)
    
    for i, dataset in enumerate(datasets, 1):
        log_message(f"📊 进度: {i}/{len(datasets)} - {dataset}")
        
        # 检查数据集是否存在
        if not check_dataset_exists(dataset):
            skipped_count += 1
            continue
        
        # 运行pipeline
        dataset_start_time = time.time()
        success = run_fisher_pipeline(dataset, config)
        dataset_end_time = time.time()
        dataset_elapsed = (dataset_end_time - dataset_start_time) / 3600
        
        if success:
            successful_count += 1
            log_message(f"✅ 数据集 {dataset} 完成，耗时: {dataset_elapsed:.2f}小时")
        else:
            failed_count += 1
            log_message(f"❌ 数据集 {dataset} 失败，耗时: {dataset_elapsed:.2f}小时")
        
        # 休息一下再处理下一个数据集
        if i < len(datasets):
            log_message("💤 休息30秒后继续...")
            time.sleep(30)
    
    # 最终统计
    total_end_time = time.time()
    total_elapsed = (total_end_time - total_start_time) / 3600
    
    log_message("🏁 所有数据集处理完成!")
    log_message(f"📈 统计结果:")
    log_message(f"   ✅ 成功: {successful_count}")
    log_message(f"   ❌ 失败: {failed_count}")
    log_message(f"   ⏭️  跳过: {skipped_count}")
    log_message(f"   ⏱️  总耗时: {total_elapsed:.2f}小时")
    
    if failed_count > 0:
        log_message("⚠️ 有数据集处理失败，请检查日志")
        sys.exit(1)
    else:
        log_message("🎉 所有数据集处理成功!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("❌ 用户中断了处理")
        sys.exit(1)
    except Exception as e:
        log_message(f"❌ 出现未预期错误: {str(e)}")
        sys.exit(1)
