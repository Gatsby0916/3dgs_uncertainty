#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_split.py — 生成与FisherRF完全对齐的初始4‑view训练集

关键发现：
• FisherRF的readColmapSceneInfo按image_name排序所有相机
• 然后应用--eval规则：idx % 8 != 0 作为训练集
• active/schema.py的self.init_views=[0]选择原始训练集顺序的第0个（未shuffle）
• 后续3个视图用farthest-first选择

输出: <scene>/train_split.txt（与FisherRF选择的初始4个视图完全一致）
"""

import argparse, struct, random
from pathlib import Path
import numpy as np
import torch


# ---------- helpers ----------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


def read_images_binary(path: Path):
    with open(path, "rb") as f:
        n_img = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_img):
            _id      = struct.unpack("<I", f.read(4))[0]
            qvec     = np.array(struct.unpack("<4d", f.read(32)))
            tvec     = np.array(struct.unpack("<3d", f.read(24)))
            f.read(4)                                   # camera_id
            name_b = bytearray()
            while (c := f.read(1)) != b"\x00":
                name_b.extend(c)
            image_name = name_b.decode("utf-8")
            n_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_pts2d * 24)
            R = qvec2rotmat(qvec)
            C = -R.T.dot(tvec)
            yield torch.from_numpy(C.astype(np.float32)), image_name


# ---------- main ----------
def main(scene_dir: str):
    """与FisherRF完全对齐的初始视图选择"""
    scene = Path(scene_dir)
    
    # Step 1: 读取COLMAP数据
    img_bin = scene / "sparse" / "0" / "images.bin"
    if not img_bin.exists():
        raise FileNotFoundError(f"找不到 {img_bin}")

    # 读取所有图像并按name排序（与FisherRF的readColmapSceneInfo一致）
    items = list(read_images_binary(img_bin))
    if not items:
        raise RuntimeError("images.bin 为空")
    
    # 转换为 (center, name) 对并按name排序（关键：按image_name排序）
    camera_data = [(center, name) for center, name in items]
    camera_data.sort(key=lambda x: x[1])  # 按image_name排序，与FisherRF一致
    
    print(f"📊 总图像数: {len(camera_data)}")
    
    # Step 2: 应用--eval规则分离训练集（在排序后的列表上应用8-fold）
    # FisherRF的逻辑：在排序后应用 idx % 8 != 0
    train_items = [(center, name) for idx, (center, name) in enumerate(camera_data) if idx % 8 != 0]
    
    print(f"📊 训练集大小: {len(train_items)}")
    
    if len(train_items) < 4:
        raise ValueError("训练视图不足 4 张")
    
    # 提取训练集数据
    train_centers = torch.stack([center for center, _ in train_items])
    train_names = [name for _, name in train_items]
    
    # Step 3: 选择初始4个视图（与FisherRF完全一致）
    # 关键：FisherRF的schema中self.init_views=[0]选择原始训练集顺序的第0个
    K = 4
    seed_idx = 0  # FisherRF选择原始训练集顺序的第0个，不是shuffle后的第0个
    
    print(f"🎯 FisherRF对齐的视图选择:")
    print(f"   seed视图: {train_names[seed_idx]}")
    
    # 选择逻辑：原始第0个 + farthest-first选择剩余3个
    picked_centers = [train_centers[seed_idx]]
    selected_idxs = [seed_idx]
    
    # 候选视图（排除已选择的seed）
    candidates = list(range(len(train_items)))
    candidates.remove(seed_idx)

    for step in range(K - 1):
        max_dist = -1
        best_idx = -1
        
        for cand_idx in candidates:
            if cand_idx in selected_idxs:
                continue
            
            cand_center = train_centers[cand_idx]
            min_dist_to_selected = min(
                torch.norm(cand_center - sel_center).item()
                for sel_center in picked_centers
            )
            
            if min_dist_to_selected > max_dist:
                max_dist = min_dist_to_selected
                best_idx = cand_idx
        
        if best_idx == -1:
            break
            
        selected_idxs.append(best_idx)
        picked_centers.append(train_centers[best_idx])
        print(f"   第{step+2}个视图: {train_names[best_idx]} (距离={max_dist:.3f})")
    
    # 输出结果
    selected_names = [train_names[i] for i in selected_idxs]
    stems = [Path(name).stem for name in selected_names]  # 始终输出stem（不带扩展名）
    
    out = scene / "train_split.txt"
    out.write_text("\n".join(stems) + "\n", encoding="utf-8")
    
    print(f"✅ 生成FisherRF对齐版本: {out}")
    print("   选中训练集索引:", selected_idxs)
    print("   文件名stems   :", stems)


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="生成与FisherRF完全对齐的4张训练视图split"
    )
    parser.add_argument("scene_dir", help="场景根目录，如 data/bicycle")
    main(parser.parse_args().scene_dir)
