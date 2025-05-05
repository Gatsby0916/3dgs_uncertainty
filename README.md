# Uncertainty estimation on 3D-Gaussian-Splatting 

This fork adds **Fisher‑information–based uncertainty estimation** on top of the official 3‑D Gaussian Splatting (3DGS) codebase.
All datasets, environment requirements and training commands remain **100 % back‑compatible** with the original repo – you can simply swap the Python package.

---

## 1 · Quick start

### 1.1 Install (identical to upstream)

```bash
conda create -n 3dgs python=3.10
conda activate 3dgs
pip install -r requirements.txt        # identical `requirements.txt`
pip install -e .                       # build CUDA rasteriser
```
The enviornment setting is just the same as the **Original** 3DGS.
### 1.2 Train (identical)

```bash
python train.py -s <path_to_dataset>
# e.g.
python train.py -s data/tandt/train
```

A fully‑trained scene is saved under
`output/<scene_id>/` (same structure as upstream).

---

## 2 · Render **with uncertainty**

```bash
python render.py \
       -m output/<scene_id> \
       --uncertainty_mode \
       --patch_size 8 \
       --top_k 15000
```

| switch                | default      | description                                                                                                                                                                                                                               |
| --------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-m` / `--model_path` | **required** | Path to the saved model folder produced by `train.py`.                                                                                                                                                                                    |
| `--uncertainty_mode`  | *off*        | Turn on Fisher‑information propagation and output extra heat‑maps & a text log.                                                                                                                                                           |
| `--patch_size`        | `8`          | Size (in pixels) of the square patches used for first‑pass variance scanning.<br>Smaller → finer localisation but slower; larger → coarser but faster.                                                                                    |
| `--top_k`             | `15000`      | After the fast scan, only the **top‑K most variant patches** enter the expensive Fisher back‑prop step.<br>Reducing `top_k` speeds up rendering but may miss rare artefacts; increasing gives more accurate maps at the cost of GPU time. |

eg. --patch_size 4 --top_k 30000/--patch_size 6 --top_k 20000
> Other CLI flags (`--skip_train`, `--iteration`, …) behave exactly as in the
> original repo.

---

## 3 · Outputs

```
output/<scene_id>/<split>/ours_<iter>/
│
├─ renders/        # model predictions (RGB)
├─ gt/             # ground‑truth images
├─ uncertainty/    # 3‑channel cividis heat‑maps
│
└─ gaussian_uncertainty_record.txt
```

`gaussian_uncertainty_record.txt`

```
[View 000]
  patch_idx     = 9828
  patch_coords  = (160:164, 112:116)
  gaussian_idx  = 124578
```

* **patch\_idx / coords** – the highest‑uncertainty patch in that view.
* **gaussian\_idx** – the global index of the single Gaussian contributing the
  most to that patch’s uncertainty.
  Use this ID to visualise or remove the culprit point.

---

## 4 · Advanced / debug knobs

Open **`gaussian_renderer/__init__.py`** and tweak:

```python
# master switches
DEBUG = True               # prints Σ‑clipping & Fisher stats
MAX_CLIP = 30.0            # global σ upper‑bound
USE_FISHER_SIGMA = True    # turn off to ignore Σ and view only grad‑norm maps
```

* `DEBUG=True` prints

  * Σ‑clip statistics every frame,
  * `[DBG] ✓ / ✘ …` lines that show candidate Gaussians searched & the selected index.
* Tighten `MAX_CLIP` to suppress very large Fisher variances.
* Other heuristics (`K_COLOR`, `gaussian_search_tol`, …) are exposed as kwargs inside `estimate_uncertainty()`.


## 5 · Citation

Please also cite the original 3‑D Gaussian Splatting paper:

```bibtex
@inproceedings{kerbl23gaussians,
  title     = {{3D} Gaussian Splatting for Real‑Time Radiance Field Rendering},
  author    = {Kerbl, Bernhard and Kopanas, Georgios and Drettakis, George},
  booktitle = SIGGRAPH Asia,
  year      = 2023
}
```

and ***our uncertainty extension (to appear)***.

Enjoy probing what your Gaussians “don’t know”!
Contributions & bug‑reports are welcome.
