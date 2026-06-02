#!/bin/bash
# eval_masked_metrics.sh - 批量评估masked metrics的便捷脚本

cd "$(dirname "$0")/.." || exit 1

echo "🎯 基于mask的重建质量评估"
echo "============================"

# 检查CUDA是否可用
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "❌ PyTorch环境检查失败"
    exit 1
fi

# 设置默认参数
ITERATION=""
OUTPUT_DIR="masked_results"
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --iteration)
            ITERATION="--iteration $2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --iteration N     指定iteration（默认自动查找）"
            echo "  --output-dir DIR  输出目录（默认masked_results）"
            echo "  --device DEVICE   计算设备（默认cuda）"
            echo "  --help           显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                          # 评估所有数据集"
            echo "  $0 --iteration 30000        # 指定iteration"
            echo "  $0 --output-dir my_results  # 指定输出目录"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 查找数据集目录
echo "🔍 查找数据集..."
DATASETS=()

# 检查HDD输出目录
if [[ -d "/hdd/gatsbyli/3dgs_output" ]]; then
    echo "📁 检查HDD输出目录: /hdd/gatsbyli/3dgs_output"
    for dataset_dir in /hdd/gatsbyli/3dgs_output/*/; do
        if [[ -d "$dataset_dir" ]]; then
            dataset_name=$(basename "$dataset_dir")
            # 检查是否有对应的原始数据集
            if [[ -d "data/$dataset_name" ]]; then
                echo "   ✅ 找到数据集: $dataset_name"
                DATASETS+=("$dataset_dir")
            else
                echo "   ⚠️  $dataset_name: 缺少原始数据集目录"
            fi
        fi
    done
fi

# 检查本地数据集目录
if [[ -d "data" ]]; then
    echo "📁 检查本地数据集目录: data/"
    for dataset_dir in data/*/; do
        if [[ -d "$dataset_dir" && -d "${dataset_dir}test" ]]; then
            dataset_name=$(basename "$dataset_dir")
            echo "   ✅ 找到数据集: $dataset_name"
            DATASETS+=("$dataset_dir")
        fi
    done
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "❌ 未找到可评估的数据集"
    echo "💡 请确保："
    echo "   1. 数据集已完成训练"
    echo "   2. 存在test/ours_X目录"
    echo "   3. 存在images/和mask/目录"
    exit 1
fi

echo ""
echo "📊 准备评估 ${#DATASETS[@]} 个数据集:"
for dataset in "${DATASETS[@]}"; do
    echo "   - $(basename "$dataset")"
done

echo ""
echo "🚀 开始评估..."
echo "⏰ 开始时间: $(date)"

# 运行评估
python evaluation/masked_metrics.py \
    "${DATASETS[@]}" \
    $ITERATION \
    --output "$OUTPUT_DIR/masked_metrics_results.json" \
    --device "$DEVICE"

# 检查结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ 评估完成!"
    echo "📄 结果文件: $OUTPUT_DIR/masked_metrics_results.json"
    echo "⏰ 完成时间: $(date)"
    
    # 显示简要结果
    if [[ -f "$OUTPUT_DIR/masked_metrics_results.json" ]]; then
        echo ""
        echo "📈 简要结果:"
        python -c "
import json
try:
    with open('$OUTPUT_DIR/masked_metrics_results.json', 'r') as f:
        data = json.load(f)
    
    if 'overall_metrics' in data:
        om = data['overall_metrics']
        print(f'   整体PSNR: {om[\"PSNR_mean\"]:.4f} ± {om[\"PSNR_std\"]:.4f}')
        print(f'   整体SSIM: {om[\"SSIM_mean\"]:.4f} ± {om[\"SSIM_std\"]:.4f}')
        print(f'   整体LPIPS: {om[\"LPIPS_mean\"]:.4f} ± {om[\"LPIPS_std\"]:.4f}')
    
    print(f'   数据集数量: {data[\"total_datasets\"]}')
    print(f'   总图像数: {data[\"total_images\"]}')
    
except Exception as e:
    print(f'   无法解析结果文件: {e}')
"
    fi
else
    echo "❌ 评估失败"
    exit 1
fi
