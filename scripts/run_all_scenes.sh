#!/bin/bash
# run_all_scenes.sh
# 批量运行 LF 数据集 Active Learning 实验
# 场景: africa, basket, statue, torch
# 限制: 20 Views
# 流程: Init -> Active Loop (max 20) -> Train to 30k -> Final Render

cd "$(dirname "$0")/.." || exit 1

SCENES="africa basket statue torch"
DEVICE=${1:-cuda}

echo "=================================================="
echo "🚀 开始批量实验: $SCENES"
echo "=================================================="

for SCENE in $SCENES; do
    echo ""
    echo "##################################################"
    echo "▶️  处理场景: $SCENE"
    echo "##################################################"
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 运行单个场景脚本
    # 注意: run_warp_active_learning.sh 内部有错误检查，如果失败会退出
    # 我们这里使用 source 还是 bash? bash 新进程 safer.
    bash scripts/run_warp_active_learning.sh "$SCENE" "$DEVICE"
    
    RET_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $RET_CODE -eq 0 ]; then
        echo "✅ 场景 $SCENE 完成! 耗时: ${DURATION}秒"
    else
        echo "❌ 场景 $SCENE 失败! 耗时: ${DURATION}秒"
        # 我们可以选择是否继续下个场景。通常这里选择继续，以便获得尽可能多的结果。
        echo "⚠️  将在 10 秒后继续下一个场景..."
        sleep 10
    fi
    
    # 清理一下显存? (Bash脚本结束进程后通常会自动释放)
done

echo ""
echo "=================================================="
echo "🏁 所有场景处理尝试完成"
echo "=================================================="
echo "请运行 eval_all_scenes.sh 进行统一评估。"
