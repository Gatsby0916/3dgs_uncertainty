#!/bin/bash
set -e

# Scenes to process
SCENES=("basket" "statue" "torch" "africa")

# Python interpreter
PYTHON="/home/haiyi/miniconda/envs/3dgs/bin/python"

for SCENE in "${SCENES[@]}"; do
    echo "========================================"
    echo "Running Mask Quality Curve (Random) for $SCENE"
    echo "========================================"
    
    $PYTHON results/warp_experiment/experiment_mask_quality_curve.py --scene $SCENE
done

echo "Done! All experiments run."
