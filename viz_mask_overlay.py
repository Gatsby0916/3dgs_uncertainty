#!/usr/bin/env python3
# ------------------------------------------------------------
# 可视化概率图（turbo colormap 固定）
# ------------------------------------------------------------
import argparse, numpy as np, cv2, matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

# 固定 colormap
CMAP_NAME = "turbo"

def load_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main(img_path, mask_path, alpha, thr):
    rgb  = load_img(img_path)
    mask = np.load(mask_path)

    if thr > 0:
        mask = np.where(mask >= thr, mask, 0.0)

    cmap  = cm.get_cmap(CMAP_NAME)
    heat  = (cmap(mask)[..., :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(rgb);     ax[0].set_title("RGB")
    ax[1].imshow(mask, cmap=CMAP_NAME); ax[1].set_title("Prob map")
    ax[2].imshow(overlay); ax[2].set_title("Overlay")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Visualize mask overlay (turbo cmap)")
    ap.add_argument("--img",  required=True, help="RGB image")
    ap.add_argument("--mask", required=True, help=".npy probability map")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="overlay alpha (0-1)")
    ap.add_argument("--thr", type=float, default=0.0,
                    help="probability threshold")
    args = ap.parse_args()
    main(Path(args.img), Path(args.mask), args.alpha, args.thr)
