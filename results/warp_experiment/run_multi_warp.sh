#!/bin/bash
set -e

# Scenes to process
SCENES=("basket" "statue" "torch" "africa")
TARGET_VIEW=10
SOURCES="0 1 2 3"

# Python interpreter
PYTHON="/home/haiyi/miniconda/envs/3dgs/bin/python"

for SCENE in "${SCENES[@]}"; do
    echo "========================================"
    echo "Processing Multi-View Warp (0,1,2,3 -> $TARGET_VIEW) for $SCENE"
    echo "========================================"
    
    $PYTHON results/warp_experiment/warp_masks.py \
        --source_path LF/ours/$SCENE \
        --model_output LF/ours/$SCENE/output \
        --colmap_path LF/ours/$SCENE \
        --src_views $SOURCES \
        --tgt_view $TARGET_VIEW \
        --output_vis results/warp_experiment/${SCENE}_multi.png \
        --save_numpy
done

echo "Done! All experiments run."
