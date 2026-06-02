#!/bin/bash
# eval_all_scenes.sh
# 批量评估脚本：使用 masked_metrics.py 评估 LF 数据集 Active Learning 实验结果
# 场景: africa, basket, statue, torch
# 评估目标: test set, 30000 iteration
# Mask来源: mask_gt (原始 GT mask)

cd "$(dirname "$0")/.." || exit 1

SCENES="africa basket statue torch"
PYTHON_EXE="${PYTHON_EXE:-python}"

# 结果汇总目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="final_evaluation_$TIMESTAMP"
mkdir -p "$RESULT_DIR"

echo "=================================================="
echo "📊 开始批量评估 Active Learning 结果"
echo "📆 时间戳: $TIMESTAMP"
echo "📂 结果保存: $RESULT_DIR"
echo "=================================================="

for SCENE in $SCENES; do
    echo ""
    echo "▶️  评估场景: $SCENE"
    
    # 定义路径
    DATASET_ROOT="LF/ours/$SCENE"
    OUTPUT_DIR="$DATASET_ROOT/output"
    RENDER_PATH="$OUTPUT_DIR/test/ours_30000"
    MASK_GT_DIR="$DATASET_ROOT/mask_gt"
    
    # 检查路径是否存在
    if [ ! -d "$RENDER_PATH" ]; then
        echo "❌ 找不到 Render 路径: $RENDER_PATH"
        echo "   (可能尚未完成训练或路径错误)"
        continue
    fi
    
    if [ ! -d "$MASK_GT_DIR" ]; then
        # 尝试fallback到mask (如果在training script里没有做mask_gt备份的话)
        MASK_GT_DIR="$DATASET_ROOT/mask"
        echo "⚠️  未找到 mask_gt，使用 mask 目录: $MASK_GT_DIR"
    fi

    # 运行评估
    # 注意: 我们不需要 --save-images 除非想看 debug
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

# 最后运行汇总脚本 (如果有) 或者简单的cat
echo ""
echo "=================================================="
echo "🏁 评估结束"
echo "=================================================="
echo "结果已保存在 $RESULT_DIR 目录中。"

# 简单的结果汇总打印
echo "汇总结果:"
grep "PSNR" "$RESULT_DIR"/*.json 2>/dev/null
