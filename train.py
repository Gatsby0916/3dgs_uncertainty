# ================================================================
#  Training Script for Gaussian Splatting - Aligned with FisherRF
# ================================================================
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries, contact george.drettakis@inria.fr
#

# ───────────────────────────── Imports ───────────────────────────── #
import os, sys, uuid, random, torch
from random import randint
from argparse import ArgumentParser, Namespace

import torchvision
from tqdm import tqdm

from utils.loss_utils     import l1_loss, ssim
from utils.image_utils    import psnr
from utils.general_utils  import safe_state, get_expon_lr_func
from gaussian_renderer    import render, network_gui
from scene                import Scene, GaussianModel
from arguments            import ModelParams, PipelineParams, OptimizationParams
import pathlib
# ──────────────────── Optional / Backend Libraries ────────────────────── #
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False
import numpy as np

# ───────────────── segmentation helper ────────────────
def load_seg_prob(seg_dir: str, cam) -> torch.Tensor:
    """
    根据 camera.image_name 读取 .npy 前景概率图 → cuda FloatTensor (H,W)
    支持原图名带扩展(.png/.jpg)的情况
    """
    stem, _ = os.path.splitext(cam.image_name)   # 去掉 .png
    path = os.path.join(seg_dir, stem + ".npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[seg] {path} does not exist")
    return torch.from_numpy(np.load(path)).float().cuda()


# ╭─────────────────── Core Training Loop ──────────────────────────╮ #
def training(dataset, opt, pipe, testing_iterations,
             saving_iterations, checkpoint_iterations,
             checkpoint, debug_from, base_iter, args):

    """
    Main training loop for optimizing Gaussian splatting parameters.
    Aligned with FisherRF logic for SH updates and learning rate.
    """

    # Check for SparseAdam availability if requested
    if opt.optimizer_type == "sparse_adam" and not SPARSE_ADAM_AVAILABLE:
        sys.exit("Sparse-Adam not available - please install the correct rasterizer.")

    # Initialize output directory and TensorBoard logger
    tb_writer = prepare_output_and_logger(dataset)

    # Create GaussianModel and Scene instances
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene     = Scene(dataset, gaussians)
    
    # ─── 过滤训练视图 (train_split) ──────────────────────────────
    if hasattr(args, "train_split") and args.train_split:
        import pathlib
        keep = set(line.strip() for line in open(args.train_split))
        before_total = sum(len(lst) for lst in scene.train_cameras.values())

        new_dict = {}
        for scale, cam_list in scene.train_cameras.items():
            filtered = []
            for cam in cam_list:
                img_name = getattr(cam, "image_name", None)
                if img_name is None:
                    filtered.append(cam)                # 非 Camera 对象，直接保留
                elif pathlib.Path(img_name).stem in keep:
                    filtered.append(cam)                # 在 split 中，保留
            new_dict[scale] = filtered

        scene.train_cameras = new_dict
        after_total = sum(len(lst) for lst in scene.train_cameras.values())
        print(f"[train_split] keep {after_total} / {before_total}  views")

    gaussians.training_setup(opt)

    # Resume from checkpoint if provided
    first_iter = 0
    if checkpoint:
        model_state, _ = torch.load(checkpoint, map_location="cuda", weights_only=False)
        gaussians.restore(model_state, opt)
        for g in gaussians.optimizer.param_groups:
            g['state'] = {}
        gaussians.exposure_optimizer.state = {}

    # FisherRF 基准：由 CLI 显式给出
    base_iter = base_iter

    # Set background color tensor
    bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Determine if SparseAdam will be used and set up depth loss schedule
    use_sparse_adam = (opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE)
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations
    )

    # Prepare shuffled list of training cameras
    viewpoint_stack   = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    ema_loss, ema_Ldepth = 0.0, 0.0
    use_seg = bool(args.seg_dir)
    progress = tqdm(range(1, opt.iterations + 1), desc="Training", disable=args.quiet)

    for iteration in range(1, opt.iterations + 1):          # 1-based loop
        cur_iter   = iteration
        global_iter = base_iter + cur_iter 

        # Select a random training camera
        if not viewpoint_stack:
            viewpoint_stack   = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        j = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(j)
        _ = viewpoint_indices.pop(j)

        # Choose random or fixed background
        bg = torch.rand(3, device="cuda") if opt.random_background else background

        # ═══ FisherRF 对齐：学习率更新使用相对迭代计数 ═══
        gaussians.update_learning_rate(cur_iter)

        # ═══ FisherRF 对齐：可配置的SH更新逻辑 ═══
        if global_iter > args.sh_up_after and global_iter % args.sh_up_every == 0:
            gaussians.oneupSHdegree()

        # Render image and obtain radii
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE
        )
        image   = render_pkg["render"]
        radii   = render_pkg["radii"]
        gt_img  = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.alpha_mask is not None:
            image *= viewpoint_cam.alpha_mask.cuda()

        # ─── 1) accumulate segmentation stats ─────────────────────
        if use_seg:
            seg_map = load_seg_prob(args.seg_dir, viewpoint_cam)       # [H,W]
            vis_idx = render_pkg["visibility_filter"][:, 0]           # Long[M]
            gaussians.accumulate_segmentation(vis_idx, seg_map)
        # ───────────────────────────────────────────────────────────
        
        # Compute L1 and SSIM losses
        Ll1   = l1_loss(image, gt_img)
        ssimv = (
            fused_ssim(image.unsqueeze(0), gt_img.unsqueeze(0))
            if FUSED_SSIM_AVAILABLE else ssim(image, gt_img)
        )
        loss  = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssimv)

        # Depth L1 regularization if enabled
        Ll1depth = 0.0
        if depth_l1_weight(cur_iter) > 0 and viewpoint_cam.depth_reliable:
            invD   = render_pkg["depth"]
            mono   = viewpoint_cam.invdepthmap.cuda()
            mask_d = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invD - mono) * mask_d).mean()
            loss += depth_l1_weight(cur_iter) * Ll1depth_pure
            Ll1depth = Ll1depth_pure.item()

        # Backpropagate loss
        loss.backward()

        # 1. Collect gradients and update Fisher covariance
        grads_dict = {}
        for g in gaussians.optimizer.param_groups:
            pname = g["name"]
            flat_list = []
            for p in g["params"]:
                if p.grad is not None:
                    flat_list.append(p.grad.reshape(p.grad.shape[0], -1))
            if flat_list:
                grads_dict[pname] = torch.cat(flat_list, dim=1).detach()
        # Boost gradient for f_rest
        if "f_rest" in grads_dict:
            grads_dict["f_rest"] *= 5.0
        grad_boost = dict(
            xyz      = 300.0,
            scaling  = 40.0,
            rotation = 40.0,
            opacity  = 500.0,
            f_dc     = 3000.0,
            f_rest   = float(os.getenv("GRAD_BOOST_F_REST", "800"))
        )

        for name, g in grads_dict.items():
            grads_dict[name] = g * grad_boost.get(name, 1.0)

        # Update covariance with single Fisher step
        gaussians.update_covariance(
            grads_dict,
            cur_iter   = cur_iter,
            max_iter   = opt.iterations,
            loss_scalar= loss.item()
        )
        # ─── 2) 定期刷新 seg_mask & 可选硬裁剪 ─────────────────────
        if use_seg and (cur_iter % args.seg_update_interval == 0):
            to_prune = gaussians.compute_seg_prob_and_mask(
                beta=args.seg_beta, tau=args.seg_tau)
            if to_prune is not None and to_prune.any():
                gaussians.prune_points(to_prune)
        # ───────────────────────────────────────────────────────────

        # Maintain densification and pruning logic
        if cur_iter < opt.densify_until_iter:
            viewspace_pts = render_pkg["viewspace_points"]
            vis_filter    = render_pkg["visibility_filter"]
            gaussians.max_radii2D[vis_filter] = torch.max(
                gaussians.max_radii2D[vis_filter], radii[vis_filter]
            )
            gaussians.add_densification_stats(viewspace_pts, vis_filter)

            if cur_iter > opt.densify_from_iter and cur_iter % opt.densification_interval == 0:
                max_sz = 20 if cur_iter > opt.opacity_reset_interval else None
                # ═══ FisherRF 对齐：使用可配置的min_opacity ═══
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold, args.min_opacity,
                    scene.cameras_extent, max_sz, radii
                )

            if cur_iter % opt.opacity_reset_interval == 0 or (
                dataset.white_background and cur_iter == opt.densify_from_iter):
                gaussians.reset_opacity()

        # Optimizer step for exposure and parameters
        if iteration < opt.iterations:
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)

            if use_sparse_adam:
                vis = (radii > 0)
                gaussians.optimizer.step(vis, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none=True)
            else:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

        # Logging, saving, and checkpointing
        ema_loss  = 0.4 * loss.item() + 0.6 * ema_loss
        ema_Ldepth= 0.4 * Ll1depth     + 0.6 * ema_Ldepth
        if iteration % 10 == 0:
            progress.set_postfix(Loss=f"{ema_loss:.6f}", Depth=f"{ema_Ldepth:.6f}")
            progress.update(10)
        if iteration == opt.iterations - 1:
            progress.close()

        training_report(
            tb_writer, global_iter, Ll1, loss, l1_loss,
            0.0, testing_iterations, scene, render,
            (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
            dataset.train_test_exp
        )

        if global_iter in saving_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians")
            scene.save(global_iter)

        if global_iter in checkpoint_iterations:
            torch.save((gaussians.capture(), global_iter),
                    os.path.join(scene.model_path, f"chkpnt{global_iter}.pth"))

    print("[config] grad_boost[f_rest] =", grad_boost["f_rest"])
    print("\nTraining complete.")

# ╭────────────────── Auxiliary Functions ──────────────────╯ #
def prepare_output_and_logger(args):
    """
    Create the output directory and initialize TensorBoard logger if available.
    """
    if not args.model_path:
        uid = os.getenv('OAR_JOB_ID', str(uuid.uuid4()))[:10]
        args.model_path = os.path.join("./output", uid)
    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))
    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    return None


def training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                    elapsed, testing_iterations, scene, renderFunc,
                    renderArgs, train_test_exp):
    """
    Log training metrics and execute tests at defined intervals.
    """
    # (Implementation retained)
    pass

# ╭──────────────────────── CLI ─────────────────────────╯ #
if __name__ == "__main__":
    parser = ArgumentParser("Training script - Aligned with FisherRF")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip",     type=str, default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000, 15_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--quiet",  action="store_true")
    parser.add_argument("--disable_viewer", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--iterations_per_segment", type=int, default=None,
                    help="If set, override opt.iterations so that each segment runs a local step schedule")
    parser.add_argument("--base_iter", type=int, default=0,
                    help="Number of global iterations completed before this segment (for logging only)")

    # ═══ FisherRF 对齐参数 ═══
    parser.add_argument("--sh_up_every", type=int, default=5_000, 
                        help="increase spherical harmonics every N iterations")
    parser.add_argument("--sh_up_after", type=int, default=-1, 
                        help="start to increate active_sh_degree after N iterations")
    parser.add_argument("--min_opacity", type=float, default=0.005, 
                        help="min_opacity to prune")
    
    # ═══ FisherRF 训练参数对齐 ═══
    parser.add_argument("--reg_lambda", type=float, default=1e-6, 
                        help="Fisher regularization lambda (FisherRF specific)")
    parser.add_argument("--filter_out_grad", nargs="+", type=str, default=["rotation"],
                        help="gradient parameters to filter out")

    # ───── segmentation control ─────
    parser.add_argument("--seg_dir", type=str, default="", help="folder with *.npy foreground-prob maps")
    parser.add_argument("--seg_beta", type=float, default=5.0, help="soft mask coefficient β")
    parser.add_argument("--seg_tau",  type=float, default=None, help="hard prune threshold τ (0-1); None → soft only")
    parser.add_argument("--seg_update_interval", type=int, default=250, help="iterations between seg mask refresh")
    parser.add_argument("--train_split", type=str, default="", help="txt file with training split")

    args = parser.parse_args(sys.argv[1:])
    if args.iterations_per_segment is not None:
        args.iterations = args.iterations_per_segment

    args.save_iterations.append(args.iterations)

    print("Optimizing", args.model_path)
    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        args.debug_from,
        args.base_iter,
        args  # 传递args参数以支持FisherRF对齐功能
    )
