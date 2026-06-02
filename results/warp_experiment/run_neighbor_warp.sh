#!/bin/bash
set -e

# Scenes to process
SCENES=("basket" "statue" "torch" "africa")

# Python interpreter
PYTHON="/home/haiyi/miniconda/envs/3dgs/bin/python"

for SCENE in "${SCENES[@]}"; do
    echo "========================================"
    echo "Processing Neighbor Warp (0->1) for $SCENE"
    echo "========================================"
    
    $PYTHON results/warp_experiment/warp_masks.py \
        --source_path LF/ours/$SCENE \
        --model_output LF/ours/$SCENE/output \
        --colmap_path LF/ours/$SCENE \
        --src_views 0 \
        --tgt_view 1 \
        --output_vis results/warp_experiment/${SCENE}_neighbor.png \
        --save_numpy
done

echo "Done! All experiments run."
