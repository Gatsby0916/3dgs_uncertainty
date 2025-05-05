import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, estimate_uncertainty  
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline,
               background, train_test_exp, separate_sh,
               uncertainty_mode=False, patch_size=8, top_k=15000):

    # -------------------
    render_path       = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path          = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    uncertainty_path  = os.path.join(model_path, name, f"ours_{iteration}", "uncertainty") \
                        if uncertainty_mode else None

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path,    exist_ok=True)
    if uncertainty_mode:
        makedirs(uncertainty_path, exist_ok=True)

    record_file = (os.path.join(model_path, name, f"ours_{iteration}",
                                "gaussian_uncertainty_record.txt")
                   if uncertainty_mode else None)
    if record_file:
        with open(record_file, "w") as f:
            f.write("Gaussian-Uncertainty Record\n")

    # ---------- main loop ----------
    for idx, view in enumerate(tqdm(views, desc="render process")):

        if uncertainty_mode:
            # ----- uncertainty render -----
            with torch.enable_grad():
                out = estimate_uncertainty(
                    viewpoint_camera = view,
                    pc        = gaussians,
                    pipe      = pipeline,
                    bg_color  = background,
                    scaling_modifier = 1.0,
                    separate_sh      = separate_sh,
                    use_trained_exp  = train_test_exp,
                    patch_size       = patch_size,
                    top_k           = top_k
                )

            rendering        = out["render"]
            uncertainty      = out["uncertainty"]
            max_patch_coords = out["max_patch_coords"]
            max_patch_idx    = out["max_patch_idx"]      # ← 新字段
            max_gauss_idx    = out["max_gaussian_idx"]   # ← 新字段

            if max_patch_coords is not None:
                y0, y1, x0, x1 = max_patch_coords
                print(f"[View {idx:03d}] worst-patch #{max_patch_idx} "
                      f"coords=({y0}:{y1}, {x0}:{x1})  "
                      f"worst-gaussian #{max_gauss_idx}")

                if record_file:
                    with open(record_file, "a") as f:
                        f.write(f"[View {idx:03d}]\n")
                        f.write(f"  patch_idx     = {max_patch_idx}\n")
                        f.write(f"  patch_coords  = ({y0}:{y1}, {x0}:{x1})\n")
                        f.write(f"  gaussian_idx  = {max_gauss_idx}\n\n")


        else:
            # ----- normal render -----
            with torch.no_grad():
                out        = render(view, gaussians, pipeline, background,
                                    use_trained_exp=train_test_exp,
                                    separate_sh=separate_sh)
            rendering  = out["render"]
            uncertainty = None

        # ---------- GT & clip----------
        gt = view.original_image[0:3]
        if train_test_exp:
            rendering   = rendering[..., rendering.shape[-1] // 2:]
            gt          = gt[..., gt.shape[-1] // 2:]
            if uncertainty is not None:
                uncertainty = uncertainty[..., uncertainty.shape[-1] // 2:]

        torchvision.utils.save_image(rendering.clamp(0, 1),
                                     os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt.clamp(0, 1),
                                     os.path.join(gts_path, f"{idx:05d}.png"))
        if uncertainty is not None:
            torchvision.utils.save_image(uncertainty.clamp(0, 1),
                                         os.path.join(uncertainty_path, f"{idx:05d}.png"))
        del rendering, gt
        if uncertainty is not None:
            del uncertainty
        torch.cuda.empty_cache()

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, separate_sh: bool, *, uncertainty_mode: bool, patch_size: int, top_k: int):
    """
    Render the training and testing view sets.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Set background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Render training and testing sets
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                       gaussians, pipeline, background, dataset.train_test_exp,
                       separate_sh, uncertainty_mode, patch_size=patch_size, top_k=top_k)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                       gaussians, pipeline, background, dataset.train_test_exp,
                       separate_sh, uncertainty_mode, patch_size=patch_size, top_k=top_k)


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = ArgumentParser(description="Test script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int, help="Number of iterations")
    parser.add_argument("--skip_train", action="store_true", help="Skip training set rendering")
    parser.add_argument("--skip_test", action="store_true", help="Skip testing set rendering")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode: do not print detailed info")
    parser.add_argument("--uncertainty_mode", action="store_true", help="Enable uncertainty rendering")
    parser.add_argument("--patch_size", type=int, default=8, help="Square patch size (pixels) used for uncertainty pooling")
    parser.add_argument("--top_k", type=int, default=15000, help="Number of highest-score patches kept for gradient back-prop")
    args = get_combined_args(parser)
    print("Render model path: " + args.model_path)

    # Initialize system state (random seed etc.)
    safe_state(args.quiet)

    # Execute rendering
    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,  
        uncertainty_mode=args.uncertainty_mode,
        patch_size=args.patch_size,
        top_k=args.top_k
    )
