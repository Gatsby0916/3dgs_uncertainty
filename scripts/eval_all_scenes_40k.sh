#!/bin/bash
# eval_all_scenes_40k.sh
# 批量评估脚本：评估 40k iteration (30 views) 的结果
# 场景: africa, basket, statue, torch

cd "$(dirname "$0")/.." || exit 1

SCENES="africa basket statue torch"
PYTHON_EXE="/home/haiyi/miniconda/envs/3dgs/bin/python"

# 结果汇总目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="final_evaluation_40k_$TIMESTAMP"
mkdir -p "$RESULT_DIR"

echo "=================================================="
echo "📊 开始批量评估 Active Learning 结果 (Iter 40000)"
echo "📆 时间戳: $TIMESTAMP"
echo "📂 结果保存: $RESULT_DIR"
echo "=================================================="

for SCENE in $SCENES; do
    echo ""
    echo "▶️  评估场景: $SCENE"
    
    # 定义路径
    DATASET_ROOT="LF/ours/$SCENE"
    OUTPUT_DIR="$DATASET_ROOT/output"
    RENDER_PATH="$OUTPUT_DIR/test/ours_40000"
    MASK_GT_DIR="$DATASET_ROOT/mask_gt"
    
    # 检查路径是否存在
    if [ ! -d "$RENDER_PATH" ]; then
        echo "❌ 找不到 Render 路径: $RENDER_PATH"
        continue
    fi
    
    if [ ! -d "$MASK_GT_DIR" ]; then
        # 尝试fallback
        MASK_GT_DIR="$DATASET_ROOT/mask"
    fi

    # 运行评估
    # 使用正确的python环境
    $PYTHON_EXE evaluation/masked_metrics.py "$RENDER_PATH" \
        --mask-dir "$MASK_GT_DIR" \
        --output-json "$RESULT_DIR/${SCENE}_metrics.json" \
        --no-save-images
        
    if [ $? -eq 0 ]; then
        echo "✅ $SCENE 评估完成。"
    else
        echo "❌ $SCENE 评估失败。"
    fi
done

echo ""
echo "=================================================="
echo "🏁 评估结束"
echo "=================================================="
echo "结果已保存在 $RESULT_DIR 目录中。"

# 简单的结果汇总打印
echo "汇总结果 (PSNR Mean):"
grep "PSNR_mean" "$RESULT_DIR"/*.json 2>/dev/null
