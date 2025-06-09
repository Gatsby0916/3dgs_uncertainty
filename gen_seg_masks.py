#!/usr/bin/env python3
# ------------------------------------------------------------
# 生成 .npy 前景概率图 (U²-Net)
#   · 自动扫描常见图像后缀，不再只限 .png
#   · batch = 4, device 固定 cuda:0（可改）
# ------------------------------------------------------------
import os, argparse, glob
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ---------- 固定配置 ----------
BATCH   = 4
DEVICE  = "cuda:0"
MODEL_URL = "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth"
DEFAULT_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
# --------------------------------

# ---------- 下载并加载模型 ----------
# ---------- 下载并加载模型 ----------
from models.u2net import U2NET   # 已将 models/ 放在工程根目录

SEARCH_CKPTS = [
    Path("u2net.pth"),
    Path("models/u2net.pth"),
    Path("models/u2net/u2net.pth"),
]

def load_u2net():
    # 1) 在本地若干默认路径里查找权重
    ckpt_path = None
    for p in SEARCH_CKPTS:
        if p.exists():
            ckpt_path = p
            break

    # 2) 若全都不存在再尝试下载
    if ckpt_path is None:
        import urllib.request, ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        ckpt_path = SEARCH_CKPTS[0]    # 默认保存成 ./u2net.pth
        print("[u2net] checkpoint not found locally, downloading…")
        urllib.request.urlretrieve(MODEL_URL, str(ckpt_path))

    # 3) 加载模型
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return net.to(DEVICE).eval()
# ------------------------------------


to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
])

@torch.no_grad()
def predict_masks(model, imgs):
    """imgs: list[PIL.Image]; return list[np.ndarray] (H,W) float32"""
    batch = torch.stack([to_tensor(im) for im in imgs]).to(DEVICE)
    d1, *_ = model(batch)                  # U²-Net 多尺度，d1 分辨率最高
    d1 = torch.sigmoid(F.interpolate(d1, size=imgs[0].size[::-1],
                                     mode="bilinear", align_corners=False))
    return list(d1.squeeze(1).cpu().numpy())

# ---------- 主流程 ----------
def run(img_dir, out_dir, exts):
    img_dir, out_dir = Path(img_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = [e.lower() for e in exts]

    # 收集所有匹配图像
    img_paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    img_paths.sort()
    if not img_paths:
        raise RuntimeError(
            f"No image files ({', '.join(exts)}) found in {img_dir}"
        )

    model = load_u2net()

    for i in tqdm(range(0, len(img_paths), BATCH), desc="saliency"):
        paths = img_paths[i:i+BATCH]
        ims   = [Image.open(p).convert("RGB") for p in paths]
        masks = predict_masks(model, ims)
        for p, m in zip(paths, masks):
            np.save(out_dir / (p.stem + p.suffix + ".npy"),
                    m.astype(np.float32))
    print(f"[done] {len(img_paths)} masks saved → {out_dir}")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Generate .npy foreground masks (U²-Net)")
    ap.add_argument("--img_dir", required=True,
                    help="folder containing RGB images")
    ap.add_argument("--out_dir", required=True,
                    help="where to save .npy masks")
    ap.add_argument("--exts", default=",".join(DEFAULT_EXTS),
                    help="comma-separated list of image extensions to search")
    args = ap.parse_args()
    exts = [e.strip() if e.startswith(".") else "."+e.strip()
            for e in args.exts.split(",") if e.strip()]
    run(args.img_dir, args.out_dir, exts)
