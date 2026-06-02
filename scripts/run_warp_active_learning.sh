#!/bin/bash
# run_warp_active_learning.sh
# 激活 Active Learning 实验流程（Warp Mask + OUGS）
# 用法: bash run_warp_active_learning.sh [SCENE] [DEVICE]
# 默认场景: basket

cd "$(dirname "$0")/.." || exit 1

SCENE=${1:-basket}
DEVICE=${2:-cuda}
PYTHON_EXE="${PYTHON_EXE:-python}"

# 硬编码路径，可根据需要修改
DATASET_ROOT="LF/ours/$SCENE"
OUTPUT_DIR="$DATASET_ROOT/output"
MASK_DIR="$DATASET_ROOT/mask"
MASK_GT_DIR="$DATASET_ROOT/mask_gt"
SPLIT_TXT="$DATASET_ROOT/train_split.txt"

# 实验参数
PATCH_SIZE=4
TOTAL_ITER=30000
MAX_VIEWS=20
# FisherRF NBV Schedule: 初步训练400步，然后在这个基础上加view
# Schedule: [400, 900, 1500, 2200, 3000, 3900, 4900, 6000, 7200, 8500, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000]
SCHEDULE=(400 900 1500 2200 3000 3900 4900 6000 7200 8500 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000)

echo "=== 开始 Active Learning 实验: $SCENE ==="
echo "=== 使用 Warp Mask 自我优化 ==="
echo "=== 设备: $DEVICE ==="

# 0. 准备环境 & 数据备份
# 因为我们会覆盖 mask/ 文件夹中的文件，必须先备份原始 GT
if [ -d "$MASK_DIR" ]; then
    if [ ! -d "$MASK_GT_DIR" ]; then
        echo "正在备份 GT Mask 到 mask_gt/..."
        cp -r "$MASK_DIR" "$MASK_GT_DIR"
    else
        echo "GT Mask 备份已存在: $MASK_GT_DIR"
    fi
    # 恢复环境：每次运行前，理论上应该重置 mask/ 为 GT（对于初始的4个view）
    # 但是，更严谨的做法是：只保留初始4个的 mask，或者让 update 脚本去覆盖
    # 为了简单，我们先还原全部 mask，确保初始4个 view 能读到 mask。
    # 后续 update 脚本会覆盖选中的 view 的 mask。
    cp -r "$MASK_GT_DIR/"* "$MASK_DIR/"
else
    echo "错误：找不到 mask 目录 $MASK_DIR"
    exit 1
fi

# 1. 生成初始 Split (4 Views)
echo "生成初始训练集 (4 Views)..."
$PYTHON_EXE pipeline/gen_split.py "$DATASET_ROOT"

# 清理旧的 Output (可选)
# rm -rf "$OUTPUT_DIR"

PREV_ITER=0
PREV_CKPT=""

# 2. 循环训练
for ITER in "${SCHEDULE[@]}"; do
    SEG_ITERS=$((ITER - PREV_ITER))
    
    echo "------------------------------------------------"
    echo "阶段目标 Iteration: $ITER (新增训练步骤: $SEG_ITERS)"
    echo "------------------------------------------------"
    
    # A. 训练 3DGS
    echo "正在训练..."
    CMD="$PYTHON_EXE train.py -s $DATASET_ROOT -m $OUTPUT_DIR \
        --iterations_per_segment $SEG_ITERS \
        --base_iter $PREV_ITER \
        --train_split $SPLIT_TXT \
        --save_iterations $ITER \
        --checkpoint_iterations $ITER \
        --eval \
        --densify_until_iter 2000 \
        --sh_up_every 5000 \
        --sh_up_after 1000 \
        --min_opacity 0.005"
        
    if [ -n "$PREV_CKPT" ]; then
        CMD="$CMD --start_checkpoint $PREV_CKPT"
    fi
    
    # 执行训练
    $CMD
    if [ $? -ne 0 ]; then echo "训练失败!"; exit 1; fi
    
    # 检查是否是最后一步
    if [ "$ITER" -ge "$TOTAL_ITER" ]; then
        echo "达到最大训练步数，停止 NBV 选择。"
        # 注意：这里我们不需要break，因为循环本身就是由Schedule控制的
        # 但是如果Schedule包含30000，且我们不想在30000做选择，可以在下面控制
    fi
    
    # 检查是否已达到最大视图数
    VIEW_COUNT=$(grep -cve '^\s*$' "$SPLIT_TXT")
    # 如果文件末尾没有换行符，wc -l 可能少算一行；grep -c 行数更准
    # 或者用 python calc? simple wc -l usually fine for generated files.
    # gen_split writes with newlines.
    VIEW_COUNT=$(wc -l < "$SPLIT_TXT")
    echo "当前视图数量: $VIEW_COUNT / $MAX_VIEWS"
    
    if [ "$VIEW_COUNT" -ge "$MAX_VIEWS" ]; then
        echo "⚠️ 已达到最大视图数 $MAX_VIEWS，跳过 NBV 选择 (Render & Update)。"
        PREV_ITER=$ITER
        PREV_CKPT="$OUTPUT_DIR/chkpnt$ITER.pth"
        continue
    fi
    
    # B. 渲染 Depth & Uncertainty
    # 注意: 我们需要渲染 'ours_ITER' 文件夹
    echo "正在渲染 Depth & Uncertainty (Patch Size $PATCH_SIZE)..."
    $PYTHON_EXE render.py -m "$OUTPUT_DIR" \
        --iteration $ITER \
        --uncertainty_mode \
        --patch_size $PATCH_SIZE \
        --mask-dir "$MASK_DIR" \
        --skip_test
        
    if [ $? -ne 0 ]; then echo "渲染失败!"; exit 1; fi
    
    # C. NBV Selection & Mask Update (Warp)
    # 调用 warp_nbv_update.py
    # 深度路径通常在 output/train/ours_ITER/depth
    # 不确定性路径在 output/train/ours_ITER/uncertainty_npz
    DEPTH_PATH="$OUTPUT_DIR/train/ours_$ITER/depth"
    UNCERT_PATH="$OUTPUT_DIR/train/ours_$ITER/uncertainty_npz"
    SCORE_JSON="$OUTPUT_DIR/score_$ITER.json"
    
    echo "正在执行 NBV 选择与 Mask 更新 (Warping)..."
    $PYTHON_EXE pipeline/warp_nbv_update.py \
        --dataset-path "$DATASET_ROOT" \
        --model-output-path "$OUTPUT_DIR" \
        --uncert-dir "$UNCERT_PATH" \
        --depth-dir "$DEPTH_PATH" \
        --mask-dir "$MASK_DIR" \
        --train-split "$SPLIT_TXT" \
        --out-score "$SCORE_JSON" \
        --device "$DEVICE"
        
    if [ $? -ne 0 ]; then echo "NBV 更新失败!"; exit 1; fi
    
    # 更新变量用于下一轮
    PREV_ITER=$ITER
    PREV_CKPT="$OUTPUT_DIR/chkpnt$ITER.pth"
    
    # D. 清理旧文件 (可选，防止磁盘爆满)
    # ...
    
done

# 3. 最终评估
echo "=== 训练完成，开始最终评估 ==="
$PYTHON_EXE render.py -m "$OUTPUT_DIR" --iteration "$TOTAL_ITER"

# $PYTHON_EXE metrics.py -m "$OUTPUT_DIR" --split test
echo "⚠️ 已跳过单场景 Metrics 计算，请运行 unified masked evaluation 脚本进行批量评估。"

echo "=== 实验结束 ==="
