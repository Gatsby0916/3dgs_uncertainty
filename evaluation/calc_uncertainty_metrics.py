
import os
import sys
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import json
import matplotlib.pyplot as plt

def load_image(path):
    img = Image.open(path)
    return np.array(img).astype(np.float32) / 255.0

def load_npy(path):
    return np.load(path)


def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def mse(pred, gt):
    return np.mean((pred - gt) ** 2)

def delta_mae(mae_list, uncertainty_list, delta=0.1):
    # 计算不确定性分桶下的MAE
    bins = np.arange(0, 1+delta, delta)
    bin_mae = []
    for i in range(len(bins)-1):
        idx = (uncertainty_list >= bins[i]) & (uncertainty_list < bins[i+1])
        if np.sum(idx) > 0:
            bin_mae.append(np.mean(mae_list[idx]))
        else:
            bin_mae.append(None)
    return bins, bin_mae

def ause(mae_list, uncertainty_list):
    # 按不确定性排序，计算AUSE
    sort_idx = np.argsort(uncertainty_list)
    sorted_mae = np.array(mae_list)[sort_idx]
    return np.mean(sorted_mae)

def ause_mse(mse_list, uncertainty_list):
    sort_idx = np.argsort(uncertainty_list)
    sorted_mse = np.array(mse_list)[sort_idx]
    return np.mean(sorted_mse)

def main():
    parser = argparse.ArgumentParser(description='Uncertainty metrics evaluation')
    parser.add_argument('base_dir', type=str, nargs='?', default='LF/torch/output/test/ours_7000',
                        help='Base directory containing gt, renders, uncertainty, uncertainty_npz')
    parser.add_argument('--npz_dir', type=str, default=None,
                        help='Optional: directory or file for uncertainty_npz (default: <base_dir>/uncertainty_npz)')
    args = parser.parse_args()

    base_dir = args.base_dir
    gt_dir = os.path.join(base_dir, 'gt')
    render_dir = os.path.join(base_dir, 'renders')
    uncertainty_dir = os.path.join(base_dir, 'uncertainty')
    # 文件名对齐，只处理三者都存在的样本
    exts = ['.png', '.jpg', '.jpeg', '.npy']
    def valid_file(f):
        return any(f.lower().endswith(e) for e in exts)
    gt_files_all = sorted([f for f in glob(os.path.join(gt_dir, '*')) if valid_file(f)])
    all_render_files = sorted(glob(os.path.join(render_dir, '*')))
    render_files_all = sorted([f for f in all_render_files if valid_file(f)])
    uncertainty_files_all = sorted([f for f in glob(os.path.join(uncertainty_dir, '*')) if valid_file(f)])
    def clean_name(f):
        return os.path.basename(f).strip().lower()
    gt_names = set([clean_name(f) for f in gt_files_all])
    render_names = set([clean_name(f) for f in render_files_all])
    unc_names = set([clean_name(f) for f in uncertainty_files_all])
    common_names = gt_names & render_names & unc_names
    missing_in_gt = (render_names & unc_names) - gt_names
    missing_in_render = (gt_names & unc_names) - render_names
    missing_in_unc = (gt_names & render_names) - unc_names
    # 如需保留缺失提示可取消注释
    # if missing_in_gt:
    #     print(f"缺失于gt: {sorted(missing_in_gt)}")
    # if missing_in_render:
    #     print(f"缺失于render: {sorted(missing_in_render)}")
    # if missing_in_unc:
    #     print(f"缺失于uncertainty: {sorted(missing_in_unc)}")
    if not common_names:
        return
    gt_files = [os.path.join(gt_dir, n) for n in sorted(common_names)]
    render_files = [os.path.join(render_dir, n) for n in sorted(common_names)]
    uncertainty_files = [os.path.join(uncertainty_dir, n) for n in sorted(common_names)]

    # 计算uncertainty降序mask曲线（selective risk curve）
    # 将所有像素的mae、mse和uncertainty拼接到一起
    all_mae_pix = []
    all_mse_pix = []
    all_unc_pix = []
    for gt_path, render_path, unc_path in zip(gt_files, render_files, uncertainty_files):
        gt = load_image(gt_path)
        render = load_image(render_path)
        unc = load_npy(unc_path) if unc_path.endswith('.npy') else load_image(unc_path)
        # 保证三者shape一致
        if gt.shape != render.shape or gt.shape != unc.shape:
            min_shape = np.minimum(np.minimum(gt.shape, render.shape), unc.shape)
            gt = gt[:min_shape[0], :min_shape[1], ...]
            render = render[:min_shape[0], :min_shape[1], ...]
            unc = unc[:min_shape[0], :min_shape[1], ...]
        # 只对前3通道
        if gt.ndim == 3:
            abs_err = np.mean(np.abs(render - gt), axis=2)
            sq_err = np.mean((render - gt) ** 2, axis=2)
        else:
            abs_err = np.abs(render - gt)
            sq_err = (render - gt) ** 2
        mae_flat = abs_err.flatten()
        mse_flat = sq_err.flatten()
        unc_flat = unc.flatten()
        min_len = min(len(mae_flat), len(unc_flat), len(mse_flat))
        all_mae_pix.append(mae_flat[:min_len])
        all_mse_pix.append(mse_flat[:min_len])
        all_unc_pix.append(unc_flat[:min_len])
    all_mae_pix = np.concatenate(all_mae_pix)
    all_mse_pix = np.concatenate(all_mse_pix)
    all_unc_pix = np.concatenate(all_unc_pix)
    # 按uncertainty降序排序
    sort_idx = np.argsort(-all_unc_pix)
    sorted_mae = all_mae_pix[sort_idx]
    sorted_mse = all_mse_pix[sort_idx]
    # 计算全体像素MAE和MSE
    global_mae = float(np.mean(sorted_mae))
    global_mse = float(np.mean(sorted_mse))
    # 计算不同保留比例下的MAE和MSE（横坐标为去除像素比例）
    keep_ratios = np.linspace(1.0, 0.0, 101)  # 0~1, 101个点
    removed_ratios = 1 - keep_ratios
    selective_mae = []
    selective_mse = []
    n_pix = len(sorted_mae)
    for r in keep_ratios:
        k = int(n_pix * r)
        if k > 0:
            selective_mae.append(float(np.mean(sorted_mae[-k:])))
            selective_mse.append(float(np.mean(sorted_mse[-k:])))
        else:
            selective_mae.append(None)
            selective_mse.append(None)
    # 绘制selective risk curve (MAE/MSE)
    plt.figure(figsize=(8,5))
    plt.plot(removed_ratios, selective_mae, marker='o', label='MAE')
    plt.plot(removed_ratios, selective_mse, marker='s', label='MSE')
    plt.xlabel('Removed Percentage of Pixels')
    plt.ylabel('Error (MAE / MSE)')
    plt.title('Selective Error Curve (Ours)')
    plt.legend()
    plt.grid(True)
    plt.savefig('selective_mae_mse_curve.png')
    plt.close()
    # 文件名对齐，只处理三者都存在的样本
    exts = ['.png', '.jpg', '.jpeg', '.npy']
    def valid_file(f):
        return any(f.lower().endswith(e) for e in exts)
    gt_files_all = sorted([f for f in glob(os.path.join(gt_dir, '*')) if valid_file(f)])
    all_render_files = sorted(glob(os.path.join(render_dir, '*')))
    render_files_all = sorted([f for f in all_render_files if valid_file(f)])
    uncertainty_files_all = sorted([f for f in glob(os.path.join(uncertainty_dir, '*')) if valid_file(f)])
    def clean_name(f):
        return os.path.basename(f).strip().lower()
    gt_names = set([clean_name(f) for f in gt_files_all])
    render_names = set([clean_name(f) for f in render_files_all])
    unc_names = set([clean_name(f) for f in uncertainty_files_all])
    common_names = gt_names & render_names & unc_names
    missing_in_gt = (render_names & unc_names) - gt_names
    missing_in_render = (gt_names & unc_names) - render_names
    missing_in_unc = (gt_names & render_names) - unc_names
    # 如需保留缺失提示可取消注释
    # if missing_in_gt:
    #     print(f"缺失于gt: {sorted(missing_in_gt)}")
    # if missing_in_render:
    #     print(f"缺失于render: {sorted(missing_in_render)}")
    # if missing_in_unc:
    #     print(f"缺失于uncertainty: {sorted(missing_in_unc)}")
    if not common_names:
        return
    gt_files = [os.path.join(gt_dir, n) for n in sorted(common_names)]
    render_files = [os.path.join(render_dir, n) for n in sorted(common_names)]
    uncertainty_files = [os.path.join(uncertainty_dir, n) for n in sorted(common_names)]
    mae_list = []
    mse_list = []
    uncertainty_list = []
    for gt_path, render_path, unc_path in tqdm(zip(gt_files, render_files, uncertainty_files), total=len(gt_files)):
        gt = load_image(gt_path)
        render = load_image(render_path)
        unc = load_npy(unc_path) if unc_path.endswith('.npy') else load_image(unc_path)
        m = mae(render, gt)
        s = mse(render, gt)
        u = np.mean(unc)
        mae_list.append(m)
        mse_list.append(s)
        uncertainty_list.append(u)
    mae_list = np.array(mae_list)
    mse_list = np.array(mse_list)
    uncertainty_list = np.array(uncertainty_list)
    ause_score = ause(mae_list, uncertainty_list)
    ause_mse_score = ause_mse(mse_list, uncertainty_list)
    bins, bin_mae = delta_mae(mae_list, uncertainty_list)
    bins_mse, bin_mse = delta_mae(mse_list, uncertainty_list)
    # 处理uncertainty_npz
    if args.npz_dir is not None:
        npz_dir = args.npz_dir
    else:
        npz_dir = os.path.join(base_dir, 'uncertainty_npz')
    npz_ause = None
    npz_bins, npz_bin_mae = None, None
    if os.path.isdir(npz_dir):
        npz_files = sorted([f for f in glob(os.path.join(npz_dir, '*.npz'))])
        if npz_files:
            for npz_path in npz_files:
                npz = np.load(npz_path)
                npz_mae = npz['mae'] if 'mae' in npz else None
                npz_unc = npz['uncertainty'] if 'uncertainty' in npz else None
                if npz_mae is not None and npz_unc is not None:
                    npz_ause = ause(npz_mae, npz_unc)
                    npz_bins, npz_bin_mae = delta_mae(npz_mae, npz_unc)
                    break  # 只处理第一个有效的npz文件
    elif os.path.isfile(npz_dir) and npz_dir.endswith('.npz'):
        npz = np.load(npz_dir)
        npz_mae = npz['mae'] if 'mae' in npz else None
        npz_unc = npz['uncertainty'] if 'uncertainty' in npz else None
        if npz_mae is not None and npz_unc is not None:
            npz_ause = ause(npz_mae, npz_unc)
            npz_bins, npz_bin_mae = delta_mae(npz_mae, npz_unc)
    # 格式化分桶区间为一位小数
    def format_bin_keys(bins, values):
        return {f"{b:.1f}": (float(m) if m is not None else None) for b, m in zip(bins, values)}

    result = {
        'AUSE_MAE': float(ause_score),
        'AUSE_MSE': float(ause_mse_score),
        'delta_MAE': format_bin_keys(bins, bin_mae),
        'delta_MSE': format_bin_keys(bins_mse, bin_mse),
        'npz_AUSE': float(npz_ause) if npz_ause is not None else None,
        'npz_delta_MAE': format_bin_keys(npz_bins, npz_bin_mae) if npz_bins is not None else None
    }
    print('===== AUSE (MAE) =====')
    print(ause_score)
    print('===== AUSE (MSE) =====')
    print(ause_mse_score)
    print('===== delta_MAE (per bin) =====')
    print(format_bin_keys(bins, bin_mae))
    print('===== delta_MSE (per bin) =====')
    print(format_bin_keys(bins_mse, bin_mse))
    print(json.dumps(result, indent=2))
    with open('uncertainty_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)

    # 绘制delta_MAE变化曲线
    # 只绘制主delta_MAE
    bin_labels = [f"{b:.1f}" for b in bins[:-1]]
    y = [float(m) if m is not None else None for m in bin_mae]
    plt.figure(figsize=(8,5))
    plt.plot(bin_labels, y, marker='o')
    plt.xlabel('Uncertainty bin')
    plt.ylabel('MAE')
    plt.title('delta_MAE vs Uncertainty')
    plt.grid(True)
    plt.savefig('delta_mae_curve.png')
    plt.close()

if __name__ == '__main__':
    main()
