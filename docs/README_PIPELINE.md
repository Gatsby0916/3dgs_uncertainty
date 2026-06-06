# OUGS Pipeline — Walk-through

End-to-end guide for the active view selection (NBV) pipeline and the random-baseline pipeline.

## File Overview

| File | Purpose |
| :--- | :--- |
| `unified_pipeline.py` | Main NBV pipeline — initial views → train → render uncertainty → score → add best view, repeat |
| `random_pipeline.py` | Random-baseline pipeline — same loop but selects next view at random |
| `pipeline_config.yml` | Config for the NBV pipeline |
| `random_pipeline_config.yml` | Config for the random-baseline pipeline |
| `gen_split.py` | Generates the initial training/candidate split (FPS-based seed views) |
| `generate_object_uncertainty.py` | Renders per-view uncertainty maps for candidate views |
| `object_nbv_score.py` | Scores candidates using object-aware (mask-gated) uncertainty aggregation |
| `warp_nbv_update.py` | Optional warp-based view update utility |
| `run_random.py` | Convenience wrapper to launch the random pipeline |

## Quick Start

### NBV pipeline
```bash
# Edit pipeline_config.yml to set your scene path under `datasets:`
python pipeline/unified_pipeline.py pipeline/pipeline_config.yml
```

### Random baseline
```bash
python pipeline/random_pipeline.py pipeline/random_pipeline_config.yml
```

## Key Config Knobs (`pipeline_config.yml`)

| Key | Default | Meaning |
| :--- | :---: | :--- |
| `init_pick` | `4` | Number of seed views (view 0 + FPS) |
| `train.total_iterations` | `30000` | Total training budget (FisherRF-aligned) |
| `nbv_schedule` | 16 entries | Iteration milestones at which one new view is added |
| `uncert_mode.patch_size` | `8` | Spatial patch size for uncertainty scoring |
| `uncert_mode.thr` | `0.3` | Foreground-probability threshold for mask gating |
| `uncert_mode.mode` | `mean` | Score aggregation: `sum` / `mean` / `max` / `pXX` |
| `opt_params.densify_until_iter` | `1000` | Stop densification after this iteration |

## NBV Loop Logic

```
1. Pick init_pick seed views (view 0 + FPS-selected)
2. Train for iterations up to first nbv_schedule milestone
3. Render uncertainty over all remaining candidate views
   (generate_object_uncertainty.py, mask-gated)
4. Score candidates → pick highest (object_nbv_score.py)
5. Add best view to training set
6. Resume training to next milestone
7. Repeat 3–6 for each entry in nbv_schedule
8. Final training run to total_iterations
```

## Foreground Mask

Each scene directory should contain a `mask/` subdirectory with per-image binary or probability masks aligned to the corresponding training images. The mask gates the uncertainty so background variance does not bias view selection.

If no mask is provided, the pipeline falls back to full-image uncertainty (equivalent to scene-level baselines).

## Output Structure

```
<output_dir>/
├── point_cloud/iteration_<N>/point_cloud.ply
├── train/ours_<N>/
│   ├── renders/          # RGB predictions
│   ├── uncertainty/      # Viridis heatmaps
│   └── uncertainty_npz/  # Raw per-pixel variance (.npz)
└── test/ours_<N>/
    └── renders/
```

## Evaluation

```bash
python evaluation/masked_metrics.py <output_dir>/test/ours_<iter> \
       --mask-dir <scene>/mask \
       --output-json results.json
```

Computes object-masked PSNR / SSIM / LPIPS and the AUSE calibration metric.
See [`evaluation/`](../evaluation/) for details.
