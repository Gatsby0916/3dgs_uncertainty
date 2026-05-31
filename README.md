# OUGS: Active View Selection via Object-aware Uncertainty Estimation in 3DGS

> **Eurographics 2026** · *Computer Graphics Forum*
>
> Haiyi Li, Qi Chen, Denis Kalkofen, Hsiang-Ting Chen
> The University of Adelaide · Graz University of Technology

[Paper (EG Digital Library)](https://diglib.eg.org/handle/10.1111/cgf70363) · [DOI: 10.1111/cgf.70363](https://doi.org/10.1111/cgf.70363)

---

![Teaser](assets/teaser.png)

OUGS is an **object-aware** active-view-selection framework for 3D Gaussian Splatting. Scene-level uncertainty in 3DGS is dominated by background clutter, which biases next-best-view (NBV) selection *away* from the object of interest. OUGS derives uncertainty directly from the physical parameters of each Gaussian via a Jacobian-propagated pixel covariance, then modulates it with a semantic foreground mask so that NBV scoring focuses on the object.

## Highlights

- **Object-centric NBV for 3DGS** — the first method to explicitly address the background-bias problem in active 3DGS reconstruction.
- **Physically-grounded uncertainty** — derived from each Gaussian's explicit parameters (position, scale, rotation, opacity, SH colour) and propagated to pixels through the rendering Jacobian.
- **Efficient online Fisher** — a diagonal Fisher Information Matrix updated by an EMA of squared gradients with a decaying schedule, folded into the standard 3DGS training loop.

---

## Method

![Method overview](assets/method.png)

The diagonal parameter covariance is propagated through the rendering Jacobian `J_u` into a per-pixel colour covariance and modulated by the (squared) foreground probability:

```
Σ_{C,k}(u) = M_k(u)^2 · J_u (diag(I_t) + λI)^{-1} J_u^T
```

Aggregating this variance over the object mask scores each candidate view; the highest-scoring unobserved view is added to the training set.

---

## Installation

```bash
conda create -n 3dgs python=3.10
conda activate 3dgs

git submodule update --init --recursive
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim          # optional, faster SSIM
```

Environment is identical to the upstream 3DGS codebase.

---

## Usage

### Train (with Fisher-EMA covariance)

```bash
python train.py -s <path/to/scene> -m <output_dir>
```

The checkpoint contains the usual 3DGS state plus per-parameter Fisher / covariance buffers.

### Render with uncertainty

```bash
python render.py -m <output_dir> --uncertainty_mode --patch_size 8
```

Produces, under `<output_dir>/{train,test}/ours_<iter>/`:

- `renders/` — RGB predictions
- `uncertainty/` — viridis uncertainty heatmaps
- `uncertainty_npz/` — raw per-pixel variance (used by the NBV scorer)
- `depth/` — depth maps

### Active view selection pipeline

End-to-end NBV loop (initial-view picking → train segment → render uncertainty → score → append view):

```bash
python pipeline/unified_pipeline.py pipeline/pipeline_config.yml
```

Tunable knobs in [`pipeline/pipeline_config.yml`](pipeline/pipeline_config.yml):

- `nbv_schedule` — iteration milestones at which a new view is added
- `uncert_mode.patch_size`, `uncert_mode.thr`, `uncert_mode.mode` — patch-level scoring controls
- `train.total_iterations` — final training budget (30k by default, FisherRF-aligned)

The random-view baseline lives at [`pipeline/random_pipeline.py`](pipeline/random_pipeline.py).

### Evaluation

```bash
python evaluation/masked_metrics.py -m <output_dir> --split test
```

Object-masked PSNR/SSIM/LPIPS plus the AUSE uncertainty-calibration metric are implemented under [`evaluation/`](evaluation/).

---

## Results

![Active view selection results](assets/results.png)

OUGS substantially outperforms ActiveNeRF, FisherRF, Bayes' Rays and GauSS-MI at object-level reconstruction across Mip-NeRF 360, Light-Field, and Tanks-and-Temples scenes. See the [paper](https://doi.org/10.1111/cgf.70363) for full quantitative tables.

---

## Citation

```bibtex
@article{10.1111:cgf.70363,
  journal   = {Computer Graphics Forum},
  title     = {{OUGS: Active View Selection via Object-aware Uncertainty Estimation in 3DGS}},
  author    = {Li, Haiyi and Chen, Qi and Kalkofen, Denis and Chen, Hsiang-Ting},
  year      = {2026},
  publisher = {The Eurographics Association and John Wiley \& Sons Ltd.},
  ISSN      = {1467-8659},
  DOI       = {10.1111/cgf.70363}
}
```

Please also cite the original 3D Gaussian Splatting paper:

```bibtex
@inproceedings{kerbl23gaussians,
  title     = {{3D} Gaussian Splatting for Real-Time Radiance Field Rendering},
  author    = {Kerbl, Bernhard and Kopanas, Georgios and Drettakis, George},
  booktitle = {SIGGRAPH Asia},
  year      = 2023
}
```

---

## License & Acknowledgments

Non-commercial research use — see [`LICENSE.md`](LICENSE.md).

This codebase extends the official [3D Gaussian Splatting implementation](https://github.com/graphdeco-inria/gaussian-splatting). We thank the original authors.
