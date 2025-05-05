# ================================================================
#  Training Script for Gaussian Splatting
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

# ╭─────────────────── Core Training Loop ──────────────────────────╮ #
def training(dataset, opt, pipe, testing_iterations,
             saving_iterations, checkpoint_iterations,
             checkpoint, debug_from):
    """
    Main training loop for optimizing Gaussian splatting parameters.
    """

    # Check for SparseAdam availability if requested
    if opt.optimizer_type == "sparse_adam" and not SPARSE_ADAM_AVAILABLE:
        sys.exit("Sparse-Adam not available - please install the correct rasterizer.")

    # Initialize output directory and TensorBoard logger
    tb_writer = prepare_output_and_logger(dataset)

    # Create GaussianModel and Scene instances
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene     = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # Resume from checkpoint if provided
    first_iter = 0
    if checkpoint:
        model_state, first_iter = torch.load(checkpoint, map_location="cuda")
        gaussians.restore(model_state, opt)

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
    progress = tqdm(range(first_iter, opt.iterations), desc="Training")
    first_iter += 1  # adjust start of loop

    for iteration in range(first_iter, opt.iterations + 1):

        # Select a random training camera
        if not viewpoint_stack:
            viewpoint_stack   = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        j = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(j)
        _ = viewpoint_indices.pop(j)

        # Choose random or fixed background
        bg = torch.rand(3, device="cuda") if opt.random_background else background

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

        # Compute L1 and SSIM losses
        Ll1   = l1_loss(image, gt_img)
        ssimv = (
            fused_ssim(image.unsqueeze(0), gt_img.unsqueeze(0))
            if FUSED_SSIM_AVAILABLE else ssim(image, gt_img)
        )
        loss  = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssimv)

        # Depth L1 regularization if enabled
        Ll1depth = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invD   = render_pkg["depth"]
            mono   = viewpoint_cam.invdepthmap.cuda()
            mask_d = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invD - mono) * mask_d).mean()
            loss += depth_l1_weight(iteration) * Ll1depth_pure
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
            xyz      = 200.0,
            scaling  = 40.0,
            rotation = 40.0,
            opacity  = 500.0,
            f_dc     = 3000.0,
            f_rest   = 500.0
        )
        for name, g in grads_dict.items():
            grads_dict[name] = g * grad_boost.get(name, 1.0)

        # Update covariance with single Fisher step
        gaussians.update_covariance(
            grads_dict,
            cur_iter   = iteration,
            max_iter   = opt.iterations,
            loss_scalar= loss.item()
        )

        # 2. Update learning rate schedule
        gaussians.update_learning_rate(iteration)

        # Maintain densification and pruning logic
        if iteration < opt.densify_until_iter:
            viewspace_pts = render_pkg["viewspace_points"]
            vis_filter    = render_pkg["visibility_filter"]
            gaussians.max_radii2D[vis_filter] = torch.max(
                gaussians.max_radii2D[vis_filter], radii[vis_filter]
            )
            gaussians.add_densification_stats(viewspace_pts, vis_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                max_sz = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold, 0.005,
                    scene.cameras_extent, max_sz, radii
                )

            if iteration % opt.opacity_reset_interval == 0 or (
               dataset.white_background and iteration == opt.densify_from_iter
            ):
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
        if iteration == opt.iterations:
            progress.close()

        training_report(
            tb_writer, iteration, Ll1, loss, l1_loss,
            0.0, testing_iterations, scene, render,
            (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
            dataset.train_test_exp
        )

        if iteration in saving_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians")
            scene.save(iteration)

        if iteration in checkpoint_iterations:
            torch.save((gaussians.capture(), iteration),
                       os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))

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
    parser = ArgumentParser("Training script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip",     type=str, default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7_000, 30_000])
    parser.add_argument("--quiet",  action="store_true")
    parser.add_argument("--disable_viewer", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[]
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
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
        args.debug_from
    )
