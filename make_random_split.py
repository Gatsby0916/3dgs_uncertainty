#!/usr/bin/env python3
"""
make_random_split.py
────────────────────────────────────────────────────────────
从【排序后的前 100 张视图】里随机选 N 张写入 train_split.txt，
其余（仍然仅限前 100）写入 candidate_split.txt。
"""
import argparse, random, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True,
                    help="图像目录，如 data/LF/basket/images")
    ap.add_argument("--num-train", type=int, default=4,
                    help="随机挑选的训练视图数量 (默认 4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pool", type=int, default=100,
                    help="只取排序后的前 max-pool 张参与抽样 (默认 100)")
    ap.add_argument("--train-out", default="train_split.txt",
                    help="输出: 训练 split 路径")
    ap.add_argument("--cand-out",  default="candidate_split.txt",
                    help="输出: 候选 split 路径")
    args = ap.parse_args()

    random.seed(args.seed)

    # 1. 收集并按文件名排序
    imgs = sorted(pathlib.Path(args.img_dir).glob("*.png"))
    if not imgs:
        imgs = sorted(pathlib.Path(args.img_dir).glob("*.jpg"))
    assert imgs, f"{args.img_dir} 中未找到 png/jpg"

    # 2. 仅保留前 max_pool 张
    pool_imgs = imgs[:args.max_pool]
    stems_all = [p.stem for p in pool_imgs]
    pool_size = len(stems_all)

    assert pool_size >= args.num_train, (
        f"候选池不足 {args.num_train} 张 (实际 {pool_size})")

    # 3. 随机采样
    train_stems = random.sample(stems_all, args.num_train)
    cand_stems  = sorted(set(stems_all) - set(train_stems))

    # 4. 写文件
    pathlib.Path(args.train_out).write_text("\n".join(train_stems) + "\n")
    pathlib.Path(args.cand_out ).write_text("\n".join(cand_stems)  + ("\n" if cand_stems else ""))

    print(f"✓  写入 {args.train_out}  共 {len(train_stems)} 行")
    print(f"✓  写入 {args.cand_out}   共 {len(cand_stems)} 行")
    print(f"(仅从排序后的前 {args.max_pool} 张中抽样)")
    print("train_stems =", train_stems)

if __name__ == "__main__":
    main()
