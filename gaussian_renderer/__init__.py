# ================================================================
#  Unified Gaussian Splatting Utilities
#  ––– Render • Gradient Render • Fisher-Info Uncertainty
# ================================================================
# Copyright (C) 2023, Inria – GRAPHDECO research group
# Non-commercial research use only – see LICENSE.md
# Contact: george.drettakis@inria.fr
# ---------------------------------------------------------------
# This script integrates:
#   (1) The original render.py (lightweight version)
#   (2) uncertainty_utils.py (enhanced version)
# ---------------------------------------------------------------
# Dependencies: PyTorch ≥1.13 • diff_gaussian_rasterization • numpy • matplotlib
# ================================================================

from __future__ import annotations
import math, torch, torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, GaussianRasterizer
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils       import eval_sh

# ---------------- global debug switches ----------------
DEBUG             = False          # master print switch
USE_FISHER_SIGMA  = True           # enable Fisher sigma usage
K_COLOR           = 8.0            # color weight for Fisher Sigma
# Fisher Sigma clipping upper bound; None disables clipping
MAX_CLIP: float | None = 100.0
# -------------------------------------------------------
VERBOSE_RENDER_STATS = True        # enable verbose render stats
_first_verbose_call  = True        # internal guard for one-time banner

PARAM_NAMES = ["xyz", "opacity", "scaling", "rotation", "f_dc", "f_rest"]

# =========================================================
#  I.  Basic Rendering (No Gradient) — Based on original render.py
# =========================================================
def render(viewpoint_camera,
           pc: GaussianModel,
           pipe,
           bg_color: torch.Tensor,
           scaling_modifier: float = 1.0,
           separate_sh: bool = False,
           override_color=None,
           use_trained_exp: bool = False):
    """
    Non-gradient rendering: project 3D Gaussian point cloud into a 2D image.
    """

    # 1) Reserve grad-enabled screenspace tensor
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype,
        requires_grad=True, device="cuda"
    )
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    # 2) Rasterizer settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 3) Gather Gaussian attributes
    means3D   = pc.get_xyz
    means2D   = screenspace_points
    opacity   = pc.get_opacity
    scales    = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_pre = pc.get_covariance(scaling_modifier) if pipe.compute_cov3D_python else None

    # 4) Handle colour / SH
    dc = shs = colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = torch.clamp_min(
                eval_sh(pc.active_sh_degree, shs_view, dir_pp) + 0.5, 0.0
            )
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # 5) Rasterize  ➜  获取 rendered_image / radii / depth_image🔹
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D, means2D,
            dc=dc, shs=shs, colors_precomp=colors_precomp,
            opacities=opacity, scales=scales, rotations=rotations,
            cov3D_precomp=cov3D_pre
        )
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D, means2D,
            shs=shs, colors_precomp=colors_precomp,
            opacities=opacity, scales=scales, rotations=rotations,
            cov3D_precomp=cov3D_pre
        )

    # 6) Optional exposure
    if use_trained_exp:
        exp = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            rendered_image.permute(1, 2, 0)
            .matmul(exp[:3, :3])
            .permute(2, 0, 1)
            + exp[:3, 3, None, None]
        )

    rendered_image = rendered_image.clamp(0, 1)

    # 7) Return dict —— ★确保包含 depth_image🔹
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,          # 🔹关键：深度图
    }


# =========================================================
#  II.  Fisher Information Utilities
# =========================================================
def _sanitize_cov_diag(
        diag          : torch.Tensor,
        max_percentile: float = 95.0,
        min_floor     : float = 1e-4,
        sample_cap    : int   = 10_000_000
    ):
    """
    Soft-clip diagonal covariance to avoid extreme values causing numerical issues.
    """
    if diag.numel() == 0:
        return diag

    if DEBUG:
        print(f"[Sigma-clip] raw min={diag.min():.4e}, mean={diag.mean():.4e}, max={diag.max():.4e}")

    need_sample = diag.numel() > sample_cap
    try:
        if need_sample:
            step   = math.ceil(diag.numel() / sample_cap)
            sample = diag.flatten()[::step]
            high   = torch.quantile(sample, max_percentile / 100.0)
        else:
            high   = torch.quantile(diag, max_percentile / 100.0)
    except RuntimeError:
        # Fallback to CPU if GPU quantile fails
        flat = diag.cpu().flatten()
        if need_sample:
            step = math.ceil(flat.numel() / sample_cap)
            flat = flat[::step]
        high = torch.quantile(flat, max_percentile / 100.0).to(diag.device)

    if MAX_CLIP is not None:
        high = min(high, torch.tensor(MAX_CLIP, device=diag.device))
    if high <= min_floor:
        diag.clamp_(min_floor)
    else:
        diag.clamp_(min_floor, high)

    if DEBUG:
        print(f"[Sigma-clip] clipped min={diag.min():.4e}, mean={diag.mean():.4e}, max={diag.max():.4e}")
    return diag


def get_cov_flat_dict(pc: GaussianModel,
                      batch_size: int = 2000,
                      max_percentile: float = 95.0,
                      min_floor: float = 1e-4):
    """
    Flatten Fisher-EMA covariance per parameter category into a dict {name: [N, D]}.
    """
    cov_dict = pc.get_covariance_dict(batch_size=batch_size)
    cov_flat = {}
    for name, cov in cov_dict.items():
        if cov.numel() == 0:
            continue
        diag = cov.diagonal(-2, -1).flatten(1) if cov.dim() == 3 else cov
        if USE_FISHER_SIGMA:
            diag = _sanitize_cov_diag(diag.detach(), max_percentile, min_floor)
        cov_flat[name] = diag
    return cov_flat


def get_params_for_grad(pc: GaussianModel, requires_grad: bool = True):
    """
    Collect differentiable parameters for use in gradient rendering.
    """
    return dict(
        xyz           = pc.get_xyz.clone().detach().requires_grad_(requires_grad),
        opacity       = pc.get_opacity.clone().detach().requires_grad_(requires_grad),
        scaling       = pc.get_scaling.clone().detach().requires_grad_(requires_grad),
        rotation      = pc.get_rotation.clone().detach().requires_grad_(requires_grad),
        f_dc          = pc.get_features_dc.clone().detach().requires_grad_(requires_grad),
        f_rest        = pc.get_features_rest.clone().detach().requires_grad_(requires_grad)
    )

# =========================================================
#  III.  Reusable Math / Projection Tools
# =========================================================
def quat_to_rotmat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices. Input shape (N,4) [w,x,y,z], output (N,3,3).
    """
    w, x, y, z = quaternions.unbind(-1)
    N = quaternions.shape[0]
    rot = torch.zeros((N, 3, 3), dtype=quaternions.dtype, device=quaternions.device)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot[:, 0, 0] = 1 - 2 * (yy + zz)
    rot[:, 0, 1] = 2 * (xy - wz)
    rot[:, 0, 2] = 2 * (xz + wy)
    rot[:, 1, 0] = 2 * (xy + wz)
    rot[:, 1, 1] = 1 - 2 * (xx + zz)
    rot[:, 1, 2] = 2 * (yz - wx)
    rot[:, 2, 0] = 2 * (xz - wy)
    rot[:, 2, 1] = 2 * (yz + wx)
    rot[:, 2, 2] = 1 - 2 * (xx + yy)
    return rot


def project_xyz_to_pixels(xyz: torch.Tensor,
                          viewpoint_camera,
                          scales: torch.Tensor,
                          rotations: torch.Tensor,
                          batch_size: int = 2000):
    """
    Project 3D points to pixel coordinates and approximate 2D covariance.
    Returns:
        pixel_coords : (N,2) [y,x]
        cov2D        : (N,2,2)
    """
    N = xyz.shape[0]
    pixel_coords, cov2D = [], []

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        bs = e - s
        if bs <= 0:
            continue

        batch_xyz = xyz[s:e]
        batch_scales = scales[s:e]
        batch_rot = rotations[s:e]

        # 1) World to camera to NDC
        xyz_h = torch.cat([batch_xyz, torch.ones((bs, 1), device=xyz.device)], dim=1)
        cam = (viewpoint_camera.world_view_transform @ xyz_h.T).T
        clip = (viewpoint_camera.full_proj_transform @ cam.T).T
        w = torch.where(clip[:, 3:4].abs() < 1e-6,
                        torch.full_like(clip[:, 3:4], 1e-6),
                        clip[:, 3:4])
        ndc = torch.clamp(clip[:, :3] / w, -1.0, 1.0)

        H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
        x_pix = ((ndc[:, 0] + 1) / 2) * (W - 1)
        y_pix = ((1 - (ndc[:, 1] + 1) / 2)) * (H - 1)

        pix = torch.stack([y_pix, x_pix], dim=1)
        pix[:, 0].clamp_(0, H - 1)
        pix[:, 1].clamp_(0, W - 1)
        pixel_coords.append(pix)

        # 2) Covariance approximation
        cov3D = torch.diag_embed(batch_scales ** 2)
        rotmat = quat_to_rotmat(batch_rot)
        cov3D = rotmat @ cov3D @ rotmat.transpose(1, 2)

        proj3 = viewpoint_camera.full_proj_transform[:3, :3]
        view3 = viewpoint_camera.world_view_transform[:3, :3]
        jac = proj3 @ view3
        cov2D_batch = torch.zeros((bs, 2, 2), device=xyz.device)
        cov2D_batch[:] = (jac @ cov3D @ jac.T)[:, :2, :2]
        cov2D.append(cov2D_batch)

    return torch.cat(pixel_coords), torch.cat(cov2D)

# =========================================================
#  IV.  Gradient Rendering (Differentiable)
# =========================================================
def render_with_grad(viewpoint_camera,
                     xyz, opacity, scaling, rotation,
                     features_dc, features_rest,
                     *, pc: GaussianModel, pipe, bg_color,
                     scaling_modifier: float = 1.0,
                     separate_sh: bool = False,
                     override_color=None,
                     use_trained_exp=False):
    """
    Differentiable rendering: accept external gradient-enabled parameters and retain computation graph for uncertainty analysis.
    """
    screenspace_points = torch.zeros_like(
        xyz, dtype=xyz.dtype, device="cuda", requires_grad=True
    )

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height   = int(viewpoint_camera.image_height),
        image_width    = int(viewpoint_camera.image_width),
        tanfovx        = tanfovx,
        tanfovy        = tanfovy,
        bg             = bg_color,
        scale_modifier = scaling_modifier,
        viewmatrix     = viewpoint_camera.world_view_transform,
        projmatrix     = viewpoint_camera.full_proj_transform,
        sh_degree      = pc.active_sh_degree,
        campos         = viewpoint_camera.camera_center,
        prefiltered    = False,
        debug          = False,
        antialiasing   = False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    cov3D_precomp = (
        torch.diag_embed(scaling ** 2) if pipe.compute_cov3D_python else None
    )

    # Color / SH handling same as non-gradient version
    dc = shs = colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = torch.cat([features_dc, features_rest], dim=1)
            shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.active_sh_degree + 1)**2)
            dir_pp = xyz - viewpoint_camera.camera_center.repeat(xyz.shape[0], 1)
            dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = torch.clamp_min(
                eval_sh(pc.active_sh_degree, shs_view, dir_pp) + 0.5, 0.0
            ).requires_grad_(True)
        else:
            if separate_sh:
                dc, shs = features_dc, features_rest
            else:
                shs = torch.cat([features_dc, features_rest], dim=1)
    else:
        colors_precomp = override_color

    # Render
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            xyz, screenspace_points,
            dc=dc, shs=shs, colors_precomp=colors_precomp,
            opacities=opacity, scales=scaling, rotations=rotation,
            cov3D_precomp=cov3D_precomp
        )
    else:
        rendered_image, radii, depth_image = rasterizer(
            xyz, screenspace_points,
            shs=shs, colors_precomp=colors_precomp,
            opacities=opacity, scales=scaling, rotations=rotation,
            cov3D_precomp=cov3D_precomp
        )

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            rendered_image.permute(1, 2, 0)
            .matmul(exposure[:3, :3])
            .permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )

    return {
        "render"           : rendered_image.clamp(0, 1),
        "viewspace_points" : screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii"            : radii,
        "depth"            : depth_image
    }

# =========================================================
#  V.  Uncertainty Estimation (Patch / Top-K Acceleration)
# =========================================================
def estimate_uncertainty(viewpoint_camera,
                         pc                  : GaussianModel,
                         pipe,
                         bg_color            : torch.Tensor,
                         scaling_modifier    : float = 1.0,
                         separate_sh         : bool = False,
                         override_color      = None,
                         use_trained_exp     : bool = False,
                         return_raw          : bool = False,
                         patch_size          : int = 4,
                         K_COLOR             : float = 5.0,
                         cov_flat_dict       : dict | None = None,
                         top_k               : int = 20000,
                         return_gaussian     : bool = False,
                         gaussian_search_tol : int = 2,
                         *, DEBUG: bool = True):
    """
    Estimate pixel/patch uncertainty using first-order Taylor expansion
    and diagonal Fisher covariance. First scan gray-scale variance to
    select top-K patches, then backpropagate gradients for acceleration.

    Args
    ----
    viewpoint_camera     : Camera object
    pc                   : GaussianModel instance
    pipe                 : config wrapper with debug, antialiasing, etc.
    bg_color             : (3,) GPU tensor in [0,1]
    scaling_modifier     : float, uniform scale factor
    separate_sh          : bool, whether to separate DC & residual SH
    override_color       : Tensor or None, if given ignore SH features
    use_trained_exp      : bool, apply exposure transform as trained
    return_raw           : bool, return raw uncertainty heatmap
    patch_size           : int, side length of square patch (pixels)
    K_COLOR              : float, weight for color parameters
    cov_flat_dict        : optional cached Fisher covariance dict
    top_k                : int, number of highest-variance patches to process
    return_gaussian      : bool, include index of single worst Gaussian
    gaussian_search_tol  : int, pixel tol when localizing worst Gaussian
    DEBUG                : bool, print debug info

    Returns
    -------
    dict with keys:
        "render"           : rendered image tensor
        "uncertainty"      : RGB heatmap tensor
        "max_patch_idx"    : index of worst patch
        "max_patch_coords" : (y0,y1,x0,x1) of that patch
        "max_gaussian_idx" : (optional) index of worst Gaussian
        "uncertainty_raw"  : (optional) raw grayscale uncertainty
    """

    global _first_verbose_call

    # ———— patch 全量开关 ————
    # If top_k ≤ 0 or not specified, use ALL patches (no top-K filtering)
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)
    patch_H = math.ceil(H / patch_size)
    patch_W = math.ceil(W / patch_size)
    if top_k is None or top_k <= 0:
        top_k = patch_H * patch_W
    # ————————————————————

    if VERBOSE_RENDER_STATS and _first_verbose_call:
        _first_verbose_call = False
        print("\n══════════  Uncertainty-render verbose stats  ══════════")
        print(f"patch_size        = {patch_size}")
        print(f"top_k patches     = {top_k}")
        print(f"K_COLOR weight    = {K_COLOR}")
        print(f"USE_FISHER_SIGMA  = {USE_FISHER_SIGMA}")
        print(f"MAX_CLIP          = {MAX_CLIP}")
        print("════════════════════════════════════════════════════════\n")

    # 1. Differentiable render to get gradients later
    params = get_params_for_grad(pc, requires_grad=True)
    render_dict = render_with_grad(
        viewpoint_camera,
        params["xyz"], params["opacity"],
        params["scaling"], params["rotation"],
        params["f_dc"], params["f_rest"],
        pc=pc, pipe=pipe, bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        separate_sh=separate_sh,
        override_color=override_color,
        use_trained_exp=use_trained_exp
    )
    rendered_image = render_dict["render"]  # [3, H, W]

    # 2. Gather or reuse Fisher covariance
    if cov_flat_dict is None:
        if DEBUG: print("\n[DEBUG] collecting Fisher sigma…")
        cov_flat_dict = get_cov_flat_dict(pc)
    elif DEBUG:
        print("\n[DEBUG] using cached Fisher sigma")

    # 3. Compute patch-level variance to rank patches
    gray          = rendered_image.mean(0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
    patch_mean    = F.avg_pool2d(gray, patch_size, patch_size)
    patch_sqmean  = F.avg_pool2d(gray**2, patch_size, patch_size)
    patch_var     = (patch_sqmean - patch_mean**2).squeeze()
    rows, cols    = patch_var.shape
    patch_flat    = patch_var.flatten()
    flat_mean     = patch_mean.squeeze().flatten()
    score         = patch_flat + 0.05 / (flat_mean + 1e-4)

    # 4. Select top-K patches (or all, if top_k = total)
    if top_k < score.numel():
        top_idx     = torch.topk(score, k=top_k, sorted=False).indices
        active_mask = torch.zeros_like(patch_flat, dtype=torch.bool)
        active_mask[top_idx] = True
    else:
        active_mask = torch.ones_like(patch_flat, dtype=torch.bool)

    # 5. Accumulate Taylor-Fisher uncertainty on each selected patch
    patch_unc = torch.zeros_like(patch_flat)
    for flat_idx in active_mask.nonzero(as_tuple=True)[0]:
        i, j = divmod(flat_idx.item(), cols)
        y0, y1 = i * patch_size, min((i+1)*patch_size, H)
        x0, x1 = j * patch_size, min((j+1)*patch_size, W)

        patch_sum = rendered_image[:, y0:y1, x0:x1].sum()
        grads = torch.autograd.grad(
            patch_sum, [params[n] for n in PARAM_NAMES],
            retain_graph=True, allow_unused=True
        )

        u = 0.0
        for g, name in zip(grads, PARAM_NAMES):
            if g is None or name not in cov_flat_dict:
                continue
            grad_flat = g.view(g.shape[0], -1)
            cov_flat  = cov_flat_dict[name]
            D = min(grad_flat.shape[1], cov_flat.shape[1])
            if D == 0:
                continue
            weight = K_COLOR if name in ("f_dc", "f_rest") else 1.0
            u += weight * ((grad_flat[:,:D]**2) * cov_flat[:,:D]).sum(1).mean()
        patch_unc[flat_idx] = u

    # 6. Reconstruct full-resolution uncertainty map & colormap
    uncertainty_img = torch.zeros((H, W), device=rendered_image.device)
    k = 0
    for i in range(rows):
        for j in range(cols):
            y0, y1 = i*patch_size, min((i+1)*patch_size, H)
            x0, x1 = j*patch_size, min((j+1)*patch_size, W)
            uncertainty_img[y0:y1, x0:x1] = patch_unc[k]
            k += 1

    # # 7. Identify worst patch & optionally worst Gaussian
    # worst_idx = patch_unc.argmax().item()
    # wi, wj   = divmod(worst_idx, cols)
    # wy0, wy1 = wi*patch_size, min((wi+1)*patch_size, H)
    # wx0, wx1 = wj*patch_size, min((wj+1)*patch_size, W)

    # max_gaussian_idx = None
    # try:
    #     max_gaussian_idx, _ = find_max_uncertainty_gaussian_in_patch(
    #         viewpoint_camera, pc, pipe, bg_color,
    #         (wy0, wy1, wx0, wx1),
    #         cov_flat_dict, K_COLOR,
    #         separate_sh, override_color, use_trained_exp,
    #         patch_size,
    #         gaussian_search_tol=max(patch_size*2, 24)
    #     )
    # except RuntimeError:
    #     pass

    norm_u = (uncertainty_img - uncertainty_img.min()) / \
             (uncertainty_img.max() - uncertainty_img.min() + 1e-8)
    uncertainty_rgb = torch.from_numpy(
        cm.cividis(norm_u.cpu().numpy())[...,:3]
    ).permute(2,0,1).to(rendered_image.device)

    # # 8. Highlight worst patch region
    # hi_color = torch.tensor([0.9,0.2,0.6], device=rendered_image.device).view(3,1,1)
    # uncertainty_rgb[:, wy0:wy1, wx0:wx1] = (
    #     0.7 * uncertainty_rgb[:, wy0:wy1, wx0:wx1] + 0.3 * hi_color
    # )

    # if DEBUG:
    #     print(f"Worst patch idx={worst_idx}, uncertainty={patch_unc[worst_idx]:.3e}, coords=({wy0}:{wy1},{wx0}:{wx1})")

    result = {
        "render"           : rendered_image.detach(),
        "uncertainty"      : uncertainty_rgb,
        # "max_patch_idx"    : worst_idx,
        # "max_patch_coords" : (wy0, wy1, wx0, wx1),
        # "max_gaussian_idx" : max_gaussian_idx
    }
    if return_raw:
        result["uncertainty_raw"] = uncertainty_img

    return result


# =========================================================
#  VI.  Single Gaussian Localization (within worst patch)
# =========================================================
def find_max_uncertainty_gaussian_in_patch(
        viewpoint_camera, pc, pipe, bg_color,
        patch_coords, cov_flat_dict,
        K_COLOR: float = K_COLOR,
        separate_sh: bool = False,
        override_color = None,
        use_trained_exp: bool = False,
        patch_size: int = 10,
        gaussian_search_tol: int = 6
    ):
    """
    Identify the single Gaussian most responsible for uncertainty within a specified image patch.
    """
    # 1) Perform differentiable render for the patch
    params = get_params_for_grad(pc, requires_grad=True)
    rd = render_with_grad(
        viewpoint_camera,
        params["xyz"], params["opacity"], params["scaling"],
        params["rotation"], params["f_dc"], params["f_rest"],
        pc=pc, pipe=pipe, bg_color=bg_color,
        separate_sh=separate_sh, override_color=override_color,
        use_trained_exp=use_trained_exp
    )

    max_y0, max_y1, max_x0, max_x1 = patch_coords

    # 2) Filter visible Gaussians from render
    vis_idx = rd["visibility_filter"][..., 0]
    xyz_v = params["xyz"][vis_idx]
    sc_v = params["scaling"][vis_idx]
    rot_v = params["rotation"][vis_idx]

    pix_v, _ = project_xyz_to_pixels(xyz_v, viewpoint_camera, sc_v, rot_v)

    # 3) Select candidates within patch ± tolerance
    radii_v = rd["radii"][vis_idx]
    tol = max(gaussian_search_tol, radii_v.max() * 2.5)
    mask = (
        (pix_v[:, 0] >= max_y0 - tol) & (pix_v[:, 0] < max_y1 + tol) &
        (pix_v[:, 1] >= max_x0 - tol) & (pix_v[:, 1] < max_x1 + tol)
    )
    candidates = mask.nonzero(as_tuple=True)[0]
    if candidates.numel() == 0:
        print(f"[DBG] No candidates found (tol={tol}) for patch {patch_coords}")
        return None, None

    cand_idx = vis_idx[candidates]

    # 4) Compute Fisher-gradient contributions for each candidate
    grads = torch.autograd.grad(
        rd["render"][..., max_y0:max_y1, max_x0:max_x1].sum(),
        [params[n] for n in PARAM_NAMES],
        retain_graph=True, allow_unused=True
    )
    contrib = None
    for g, name in zip(grads, PARAM_NAMES):
        if g is None or name not in cov_flat_dict:
            continue
        g_flat = g.view(g.shape[0], -1)[cand_idx]
        v_flat = cov_flat_dict[name][cand_idx]
        D = min(g_flat.shape[1], v_flat.shape[1])
        if D == 0:
            continue
        weight = K_COLOR if name in ("f_dc", "f_rest") else 1.0
        c = (weight * (g_flat[:, :D] ** 2 * v_flat[:, :D]).sum(1))
        contrib = c if contrib is None else contrib + c

    best_local = contrib.argmax()
    best_idx = cand_idx[best_local].item()
    print(f"[DBG] {candidates.numel()} candidates evaluated, best={best_idx}, uncertainty={contrib[best_local]:.3e}")
    return best_idx, contrib