#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_pipeline.py - 随机视图选择的3DGS pipeline

功能：
1. 支持多个数据集连续运行
2. 随机选择视图进行添加（对比NBV方法）
3. 简化流程：训练 → 随机添加视图 → 继续训练
4. 最终统一渲染和评估
"""

import os
import sys
import time
import subprocess
import yaml
import json
import shutil
import random
from pathlib import Path
from datetime import datetime


def log_message(message):
    """记录日志信息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    
    # 同时写入日志文件
    with open("random_pipeline_log.txt", "a", encoding="utf-8") as f:
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


def get_all_training_images(dataset_path):
    """获取数据集中所有训练图像的列表"""
    try:
        # 直接读取images目录下的图像文件
        images_folder = Path(dataset_path) / "images"
        if images_folder.exists():
            # 支持常见的图像格式
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(images_folder.glob(f"*{ext}")))
            
            image_names = [img.name for img in sorted(image_files)]
            log_message(f"📸 发现 {len(image_names)} 张训练图像")
            return image_names
        else:
            log_message(f"❌ 未找到images目录: {images_folder}")
            return []
    except Exception as e:
        log_message(f"❌ 读取训练图像失败: {e}")
        return []


def add_random_view(split_txt_path, all_images, used_images):
    """随机添加一个新视图到train_split.txt"""
    try:
        # 找到未使用的图像
        available_images = [img for img in all_images if img not in used_images]
        
        if not available_images:
            log_message("⚠️ 没有更多可用的图像了")
            return False
        
        # 随机选择一个图像
        selected_image = random.choice(available_images)
        image_stem = Path(selected_image).stem
        
        # 添加到train_split.txt
        with open(split_txt_path, 'a') as f:
            f.write(f"{image_stem}\n")
        
        used_images.add(selected_image)
        log_message(f"🎲 随机添加视图: {selected_image} (剩余可选: {len(available_images)-1})")
        
        return True
        
    except Exception as e:
        log_message(f"❌ 添加随机视图失败: {e}")
        return False


def save_metrics_results(dataset_path, output_dir):
    """保存metrics结果到数据集特定的文件"""
    try:
        # 读取metrics结果
        results_file = Path(output_dir) / "results.json"
        per_view_file = Path(output_dir) / "per_view.json"
        
        if results_file.exists():
            # 创建数据集专用的结果目录
            dataset_name = Path(dataset_path).name
            results_dir = Path("random_results") / dataset_name
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制结果文件
            shutil.copy2(results_file, results_dir / "results.json")
            if per_view_file.exists():
                shutil.copy2(per_view_file, results_dir / "per_view.json")
            
            # 读取并打印结果
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            log_message(f"✅ {dataset_name} 随机选择评估结果:")
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


def run_random_pipeline(dataset_path, config):
    """运行单个数据集的随机视图选择pipeline"""
    log_message(f"🎲 开始处理数据集(随机选择): {dataset_path}")
    
    # 设置路径
    output = f"{dataset_path}/random_output"
    split_txt = Path(dataset_path) / "train_split_random.txt"
    
    # 获取配置参数
    opt_dict = config.get("opt_params", {})
    opt_flags = flags_from_opt(opt_dict)
    densify_until = config.get("densify_until_iter") or opt_dict.get("densify_until_iter", 2000)
    
    # 1. 首先生成初始的4个视图 (重用gen_split.py)
    log_message(f"🎯 生成初始4个视图...")
    if not sh(f"python pipeline/gen_split.py {dataset_path}"):
        log_message(f"❌ 初始视图生成失败")
        return False
    
    # 复制到随机版本
    original_split = Path(dataset_path) / "train_split.txt"
    shutil.copy2(original_split, split_txt)
    log_message(f"✅ 初始视图生成完成: {split_txt}")
    
    # 2. 获取所有可用图像（用于随机选择）
    all_images = get_all_training_images(dataset_path)
    if not all_images:
        log_message("❌ 无法获取训练图像列表")
        return False
    
    # 3. 读取已使用的图像
    used_images = set()
    try:
        with open(split_txt, 'r') as f:
            stems = [line.strip() for line in f.readlines() if line.strip()]
        # 根据stem找到完整的图像名
        for stem in stems:
            for img in all_images:
                if Path(img).stem == stem:
                    used_images.add(img)
                    break
        log_message(f"📊 当前已使用 {len(used_images)} 张图像")
    except Exception as e:
        log_message(f"❌ 读取已使用图像失败: {e}")
        return False
    
    prev_end_iter = 0
    first_seg = True
    
    try:
        # 随机视图添加循环
        for step in config["random_schedule"]:
            end_iter = step["iter"]
            base_iter = prev_end_iter
            seg_iters = end_iter - base_iter
            save_id = end_iter - 1  # 修正保存ID
            
            log_message(f"🎲 随机步骤: base_iter={base_iter}, end_iter={end_iter}, seg_iters={seg_iters}")
            
            # 1. 训练
            cmd = [
                "python", "train.py",
                "-s", dataset_path,
                "-m", output,
                "--iterations_per_segment", str(seg_iters),
                "--base_iter", str(base_iter),
                "--train_split", str(split_txt),
                "--save_iterations", str(save_id),
                "--checkpoint_iterations", str(save_id),
                "--eval",
                "--densify_until_iter", str(densify_until),
                "--sh_up_every", str(config.get("sh_up_every", 5000)),
                "--sh_up_after", str(config.get("sh_up_after", 1000)),
                "--min_opacity", str(config.get("min_opacity", 0.005))
            ] + opt_flags
            
            if not first_seg:
                prev_save_id = prev_end_iter - 1
                cmd += ["--start_checkpoint", f"{output}/chkpnt{prev_save_id}.pth"]
            
            if not sh(" ".join(cmd)):
                return False
            first_seg = False
            
            # 2. 随机添加新视图（如果需要）
            if step.get("add", 0) > 0:
                if not add_random_view(split_txt, all_images, used_images):
                    log_message("⚠️ 无法添加更多视图，继续训练")
            
            prev_end_iter = end_iter
        
        # 最终训练
        total_iter = config["train"]["total_iterations"]
        base_iter = prev_end_iter
        seg_iters = total_iter - base_iter
        save_id = total_iter - 1  # 修正保存ID
        
        log_message(f"🏁 最终训练: base_iter={base_iter}, total_iter={total_iter}, seg_iters={seg_iters}")
        
        final_cmd = [
            "python", "train.py",
            "-s", dataset_path, "-m", output,
            "--iterations_per_segment", str(seg_iters),
            "--base_iter", str(base_iter),
            "--start_checkpoint", f"{output}/chkpnt{prev_end_iter-1}.pth",
            "--train_split", str(split_txt),
            "--save_iterations", str(save_id),
            "--checkpoint_iterations", str(save_id),
            "--sh_up_every", str(config.get("sh_up_every", 5000)),
            "--sh_up_after", str(config.get("sh_up_after", 1000)),
            "--min_opacity", str(config.get("min_opacity", 0.005)),
            "--eval",
            "--densify_until_iter", str(densify_until)
        ] + opt_flags
        
        if not sh(" ".join(final_cmd)):
            return False
        
        # 最终渲染和评估
        if not sh(f"python render.py -m {output} --iteration {save_id}"):
            return False
        
        log_message("=== 开始最终评估(随机选择) ===")
        if not sh(f"python metrics.py -m {output} --split test"):
            return False
        
        # 保存结果
        save_metrics_results(dataset_path, output)
        
        log_message(f"✅ 数据集 {dataset_path} 随机选择处理完成")
        return True
        
    except Exception as e:
        log_message(f"❌ 数据集 {dataset_path} 处理出错: {str(e)}")
        return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python random_pipeline.py random_pipeline_config.yml")
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
    
    log_message("🎲 开始批量处理数据集(随机视图选择)")
    log_message(f"计划处理的数据集: {datasets}")
    log_message(f"随机添加schedule: {len(config['random_schedule'])}个触发点")
    
    total_start_time = time.time()
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    # 创建结果目录
    Path("random_results").mkdir(exist_ok=True)
    
    for i, dataset in enumerate(datasets, 1):
        log_message(f"📊 进度: {i}/{len(datasets)} - {dataset}")
        
        # 检查数据集是否存在
        if not check_dataset_exists(dataset):
            skipped_count += 1
            continue
        
        # 运行pipeline
        dataset_start_time = time.time()
        success = run_random_pipeline(dataset, config)
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
    
    log_message("🏁 所有数据集处理完成!(随机选择)")
    log_message(f"📈 统计结果:")
    log_message(f"   ✅ 成功: {successful_count}")
    log_message(f"   ❌ 失败: {failed_count}")
    log_message(f"   ⏭️  跳过: {skipped_count}")
    log_message(f"   ⏱️  总耗时: {total_elapsed:.2f}小时")
    
    if failed_count > 0:
        log_message("⚠️ 有数据集处理失败，请检查日志")
        sys.exit(1)
    else:
        log_message("🎉 所有数据集处理成功!(随机选择)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("❌ 用户中断了处理")
        sys.exit(1)
    except Exception as e:
        log_message(f"❌ 出现未预期错误: {str(e)}")
        sys.exit(1)
