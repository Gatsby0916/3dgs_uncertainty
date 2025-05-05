# check_restore_plus.py
import argparse, json, sys, math, re
from pathlib import Path
from types   import SimpleNamespace
from argparse import ArgumentParser

import torch
from scene.gaussian_model import GaussianModel
from scene import Scene
from arguments import ModelParams

# ---------- util ----------
def fmt(t: torch.Tensor):
    return (f"shape={tuple(t.shape)}  mean={t.mean():.4e}  "
            f"std={t.std():.4e}  min={t.min():.4e}  max={t.max():.4e}")

def banner(msg):
    print("\n" + "-"*10, msg, "-"*10)

# ---------- parse arg ----------
ap = argparse.ArgumentParser()
ap.add_argument("model_dir", nargs="?",
                help="路径类似 output/xxxxxx-x (缺省则用修改时间最新的)")
args = ap.parse_args()

out_root = Path("output")
if args.model_dir:
    model_dir = Path(args.model_dir)
else:
    # 最近修改的 output/*
    model_dir = max(out_root.glob("*"), key=lambda p: p.stat().st_mtime)

print(f"▶ Loading  {model_dir}")

# ---------- 读 cfg_args ----------
cfg_txt = (model_dir / "cfg_args").read_text().strip()
cfg_dict = eval(cfg_txt, {"Namespace": lambda **kw: kw})  # Namespace(...) → dict
cfg_ns   = SimpleNamespace(**cfg_dict)

# ---------- 还原 dataset / scene ----------
dummy_parser = ArgumentParser()
mp_obj  = ModelParams(dummy_parser)
dataset = mp_obj.extract(cfg_ns)              # ✔ 在这一步应用 cfg
gauss   = GaussianModel(dataset.sh_degree)
scene   = Scene(dataset, gauss, load_iteration=-1, shuffle=False)

banner("BASIC INFO")
print("loaded_iter :", scene.loaded_iter)
print("total points:", gauss.get_xyz.shape[0])

# ---------- 主参数 ----------
param_map = {
    "xyz"          : gauss._xyz,
    "scaling (log)": gauss._scaling,
    "rotation"     : gauss._rotation,
    "opacity (log)": gauss._opacity,
    "features_dc"  : gauss._features_dc,
    "features_rest": gauss._features_rest
}
banner("PARAMETERS")
for k, t in param_map.items():
    if t.numel():
        print(f"{k:<14} {fmt(t)}")

# ---------- 协方差 ----------
cov_map = {
    "xyz_cov"         : gauss._xyz_cov,
    "scaling_cov"     : gauss._scaling_cov,
    "rotation_cov"    : gauss._rotation_cov,
    "opacity_cov"     : gauss._opacity_cov,
    "features_dc_cov" : gauss._features_dc_cov,
    "features_rest_cov":gauss._features_rest_cov
}
banner("COVARIANCES")
for k, t in cov_map.items():
    if t.numel():
        print(f"{k:<18} {fmt(t)}")

# ---------- Fisher buffer（若存在） ----------
fisher_map = {
    "xyz_fisher"      : getattr(gauss, "_xyz_fisher_buf", torch.empty(0)),
    "scaling_fisher"  : getattr(gauss, "_scaling_fisher_buf", torch.empty(0)),
    "rotation_fisher" : getattr(gauss, "_rotation_fisher_buf", torch.empty(0)),
    "opacity_fisher"  : getattr(gauss, "_opacity_fisher_buf", torch.empty(0)),
    "f_dc_fisher"     : getattr(gauss, "_f_dc_fisher_buf", torch.empty(0)),
    "f_rest_fisher"   : getattr(gauss, "_f_rest_fisher_buf", torch.empty(0)),
}
if any(t.numel() for t in fisher_map.values()):
    banner("FISHER EMA BUFFER")
    for k, t in fisher_map.items():
        if t.numel():
            print(f"{k:<18} {fmt(t)}")
else:
    banner("FISHER EMA BUFFER")
    print("（此模型尚未存储 Fisher 统计信息）")

print("\n✅  Done.")
