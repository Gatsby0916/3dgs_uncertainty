# ================================================================
#  scene/gaussian_model.py
#  (c) 2024  –  workspace-2  整合版
# ================================================================
#
#  - 兼容 v1 / v2 二进制快照          (restore)
#  - Fisher-EMA 协方差在线估计        (update_covariance)
#  - Σ 软-clip（p-percentile）/ floor  (_post_restore_sanitize_cov)
#  - 超大张量安全 quantile            (_safe_percentile)
#
#  其余 Dense / Sparse-Adam、densify / prune 等原逻辑保持不变
# ---------------------------------------------------------------

from __future__ import annotations
import os, json, logging, math
from   typing import Dict

import torch, numpy as np
import torch.nn.functional as F
from   torch import nn

from utils.general_utils  import (inverse_sigmoid, get_expon_lr_func,
                                  build_rotation, strip_symmetric,
                                  build_scaling_rotation)
from utils.system_utils   import mkdir_p
from utils.sh_utils       import RGB2SH
from simple_knn._C        import distCUDA2
from utils.graphics_utils import BasicPointCloud

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except ImportError:
    SparseGaussianAdam = None           # fallback to dense Adam


# ─────────────────── Fisher-EMA 全局表 ──────────────────── #
FISHER_CFG = {
    "xyz":      (1e-4, 1e2),
    "scaling":  (1e-4, 1e2),
    "rotation": (1e-4, 1e2),
    "f_dc":     (1e-4, 1e2),
    "opacity":  (1e-5,   30.0),
    "f_rest":   (5e-6,   30.0),
}
DEFAULT_EPS, DEFAULT_CEIL = 1e-4, 1e2
# --------------------------------------------------------- #


# ╭──────────────────────────────────────────────────────────╮
# │  utils:  超大张量安全 percentile                        │
# ╰──────────────────────────────────────────────────────────╯
def _safe_percentile(t: torch.Tensor,
                     q: float,
                     max_sample: int = 8_000_000) -> torch.Tensor:
    """
    等价于  torch.quantile(t, q/100)；当 t 太大时先随机抽样，
    避免 thrust::sort 在 CPU / CUDA 两侧触发 `tensor is too large`.
    """
    try:                                # 先尝试原生
        return torch.quantile(t, q / 100.0)
    except RuntimeError:
        flat = t.flatten()
        if flat.numel() > max_sample:
            idx    = torch.randint(0, flat.numel(), (max_sample,),
                                    device=flat.device)
            sample = flat[idx].cpu().numpy()
        else:
            sample = flat.cpu().numpy()
        pct = np.percentile(sample, q, interpolation="nearest")
        return torch.as_tensor(pct, dtype=t.dtype, device=t.device)


# ╭──────────────────────────────────────────────────────────╮
# │  GaussianModel                                          │
# ╰──────────────────────────────────────────────────────────╯
class GaussianModel:

    # ============== 基础工具函数 ==============
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            return strip_symmetric(L @ L.transpose(1, 2))

        self.scaling_activation         = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation      = build_covariance_from_scaling_rotation
        self.opacity_activation         = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation        = torch.nn.functional.normalize

    # ============== ctor ==============
    def __init__(self, sh_degree: int, optimizer_type: str = "default"):
        super().__init__()
        self.max_sh_degree, self.active_sh_degree = sh_degree, 0
        self.optimizer_type = optimizer_type

        # --- Learnt params ---
        self._xyz = self._features_dc = self._features_rest = torch.empty(0)
        self._scaling = self._rotation = self._opacity     = torch.empty(0)

        # --- runtime ---
        self.max_radii2D = self.xyz_gradient_accum = self.denom = torch.empty(0)
        self.optimizer = self.exposure_optimizer = None
        self.percent_dense = self.spatial_lr_scale = 0

        # --- Σ & Fisher ---
        self._xyz_cov = self._scaling_cov = self._rotation_cov = torch.empty(0)
        self._features_dc_cov = self._features_rest_cov = torch.empty(0)
        self._opacity_cov = torch.empty(0)

        self._xyz_fisher_buf = self._scaling_fisher_buf = self._rotation_fisher_buf = torch.empty(0)
        self._f_dc_fisher_buf = self._f_rest_fisher_buf = torch.empty(0)
        self._opacity_fisher_buf = torch.empty(0)

        self.setup_functions()

    # ============== snapshot ==============
    def capture(self):
        """把所有需要持久化的内容打成 tuple"""
        return (
            self.active_sh_degree, self.spatial_lr_scale,
            self._xyz, self._features_dc, self._features_rest,
            self._scaling, self._rotation, self._opacity, self._exposure,
            self.max_radii2D, self.xyz_gradient_accum, self.denom,
            self.optimizer.state_dict(),

            self._xyz_cov, self._opacity_cov, self._scaling_cov,
            self._rotation_cov, self._features_dc_cov, self._features_rest_cov,

            self._xyz_fisher_buf, self._opacity_fisher_buf,
            self._scaling_fisher_buf, self._rotation_fisher_buf,
            self._f_dc_fisher_buf, self._f_rest_fisher_buf,
        )

    # ============== restore ==============
    def restore(self, packed, training_args,
                *, clip_percentile: float = 99.0, min_floor: float = 1e-4):
        """
        支持两种快照格式：
          • v1  (24 fields)：无 _exposure，spatial_lr_scale 在第 12 位
          • v2  (25 fields)：有 _exposure，spatial_lr_scale 在第  2 位
        恢复后对协方差做 soft-clip，避免 quantile 超大报错。
        """
        ver = len(packed)
        if ver not in (24, 25):
            raise ValueError(f"[restore] Unknown snapshot length {ver}")

        if ver == 24:                          # —— old format → map →
            logging.warning("⚠  old snapshot detected – converting")
            (
                active_sh_degree,
                _xyz, _f_dc, _f_rest, _scaling, _rotation, _opacity,
                max_r2d, xyz_accum, denom,
                opt_state_dict,
                spatial_lr_scale,
                _xyz_cov, _opacity_cov, _scaling_cov,
                _rotation_cov, _f_dc_cov, _f_rest_cov,
                _xyz_fisher, _opacity_fisher, _scaling_fisher,
                _rotation_fisher, _f_dc_fisher, _f_rest_fisher,
            ) = packed
            _exposure = torch.eye(3, 4, device="cuda", dtype=torch.float)[None]

        else:                                  # —— new format
            (
                active_sh_degree, spatial_lr_scale,
                _xyz, _f_dc, _f_rest, _scaling, _rotation, _opacity, _exposure,
                max_r2d, xyz_accum, denom,
                opt_state_dict,
                _xyz_cov, _opacity_cov, _scaling_cov,
                _rotation_cov, _f_dc_cov, _f_rest_cov,
                _xyz_fisher, _opacity_fisher, _scaling_fisher,
                _rotation_fisher, _f_dc_fisher, _f_rest_fisher,
            ) = packed

        # ---- 回填到对象 ----
        self.active_sh_degree, self.spatial_lr_scale = active_sh_degree, spatial_lr_scale
        self._xyz, self._features_dc, self._features_rest = _xyz, _f_dc, _f_rest
        self._scaling, self._rotation, self._opacity     = _scaling, _rotation, _opacity
        self._exposure                                    = _exposure

        self.max_radii2D, self.xyz_gradient_accum, self.denom = max_r2d, xyz_accum, denom

        self._xyz_cov, self._opacity_cov             = _xyz_cov, _opacity_cov
        self._scaling_cov, self._rotation_cov        = _scaling_cov, _rotation_cov
        self._features_dc_cov, self._features_rest_cov = _f_dc_cov, _f_rest_cov

        self._xyz_fisher_buf, self._opacity_fisher_buf       = _xyz_fisher, _opacity_fisher
        self._scaling_fisher_buf, self._rotation_fisher_buf  = _scaling_fisher, _rotation_fisher
        self._f_dc_fisher_buf,  self._f_rest_fisher_buf      = _f_dc_fisher, _f_rest_fisher

        # ---- 重建优化器 ----
        self.training_setup(training_args)
        if opt_state_dict:
            self.optimizer.load_state_dict(opt_state_dict)

        # ---- Σ soft-clip ----
        self._post_restore_sanitize_cov(clip_percentile = 100.0)
        logging.info(f"[restore] Σ clamped to ≤ {clip_percentile}%  & ≥ {min_floor}")

    # --------------- Σ clip helper ---------------
    def _post_restore_sanitize_cov(self,
                                   clip_percentile: float = 95.0,
                                   min_floor: float = 1e-4):
        cov_attrs = ["_xyz_cov", "_scaling_cov", "_rotation_cov",
                     "_opacity_cov", "_features_dc_cov", "_features_rest_cov"]
        for attr in cov_attrs:
            cov = getattr(self, attr)
            if cov.numel() == 0:
                continue
            diag = cov if cov.dim() == 2 else cov.diagonal(dim1=-2, dim2=-1)
            diag.clamp_(min=min_floor)

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        features = torch.zeros(
            (fused_point_cloud.shape[0], 3, (self.max_sh_degree+1)**2),
            device="cuda", dtype=torch.float
        )
        features[:, :3, 0] = fused_color
        # rest channel = 0
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation:", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1,3)
        rots = torch.zeros((fused_point_cloud.shape[0],4), device="cuda", dtype=torch.float)
        rots[:, 0] = 1.0
        opacities = self.inverse_opacity_activation(
            0.1*torch.ones((fused_point_cloud.shape[0],1),device="cuda",dtype=torch.float)
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1,2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :,1:].transpose(1,2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx,cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3,4,device="cuda",dtype=torch.float)[None].repeat(len(cam_infos),1,1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        # ---- init covariance ----
        N = self._xyz.shape[0]
        # xyz => [N,3,3]
        self._xyz_cov = (
            dist2[:,None,None]*torch.eye(3,device="cuda")[None,:,:]
        ).detach()
        # f_dc => [N,3,3]
        color_var = torch.var(fused_color, dim=0, keepdim=True)
        self._features_dc_cov = (
            color_var*torch.eye(3,device="cuda")[None,:,:]
        ).expand(N,3,3).detach()

        # f_rest => [N, 45] (只存对角)
        C = self._features_rest.shape[1]  # 15
        D_ = self._features_rest.shape[2] # 3
        D_total = C * D_
        self._features_rest_cov = torch.full((N,D_total), 0.1, device="cuda").detach()

        # scaling => [N,3,3]
        self._scaling_cov = (
            dist2[:,None,None]*0.1*torch.eye(3,device="cuda")[None,:,:]
        ).detach()
        # rotation => [N,4,4]
        self._rotation_cov = (
            0.1*torch.eye(4,device="cuda")[None,:,:].repeat(N,1,1)
        ).detach()
        # opacity => [N,1]
        op_var = torch.var(opacities, dim=0, keepdim=True)
        self._opacity_cov = (op_var*0.1*torch.ones_like(self._opacity)).detach()
            # 下面为Fisher buffer赋零值:
        self._xyz_fisher_buf = torch.zeros((N,3), device="cuda")  # Nx3
        self._f_dc_fisher_buf = torch.zeros((N,3), device="cuda")
        self._scaling_fisher_buf = torch.zeros((N,3), device="cuda")
        self._rotation_fisher_buf= torch.zeros((N,4), device="cuda")
        self._opacity_fisher_buf = torch.zeros((N,1), device="cuda")
        # f_rest => Nx45:
        C = self._features_rest.shape[1]  # 15
        D_ = self._features_rest.shape[2] # 3
        D_total = C*D_
        self._f_rest_fisher_buf = torch.zeros((N,D_total), device="cuda")

        logging.info(
            f"Initial cov - xyz: min {self._xyz_cov.min():.6f}, max {self._xyz_cov.max():.6f}"
        )
        logging.info(
            f"Initial cov - scaling: min {self._scaling_cov.min():.6f}, max {self._scaling_cov.max():.6f}"
        )
        logging.info(
            f"Initial cov - rotation: min {self._rotation_cov.min():.6f}, max {self._rotation_cov.max():.6f}"
        )
        logging.info(
            f"Initial cov - features_dc: min {self._features_dc_cov.min():.6f}, max {self._features_dc_cov.max():.6f}"
        )
        logging.info(
            f"Initial cov - features_rest: min {self._features_rest_cov.min():.6f}, max {self._features_rest_cov.max():.6f}"
        )
        logging.info(
            f"Initial cov - opacity: min {self._opacity_cov.min():.6f}, max {self._opacity_cov.max():.6f}"
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0],1), device="cuda"
        )
        self.denom = torch.zeros(
            (self.get_xyz.shape[0],1), device="cuda"
        )

        param_list = [
            {'params':[self._xyz],          'lr': training_args.position_lr_init*self.spatial_lr_scale, "name":"xyz"},
            {'params':[self._features_dc],  'lr': training_args.feature_lr,      "name":"f_dc"},

            # 原先是 feature_lr/20.0, 这里改为 /5.0 或 /2.0, 
            # 甚至也可以直接 = training_args.feature_lr, 视实际需求而定
            {'params':[self._features_rest],'lr': training_args.feature_lr/5.0,   "name":"f_rest"},

            # opacity 原先= training_args.opacity_lr; 可以让它多乘一个系数, 例如 2.0
            {'params':[self._opacity],      'lr': training_args.opacity_lr * 2.0, "name":"opacity"},

            {'params':[self._scaling],      'lr': training_args.scaling_lr,      "name":"scaling"},
            {'params':[self._rotation],     'lr': training_args.rotation_lr,     "name":"rotation"}
        ]


        if self.optimizer_type=="default":
            self.optimizer = torch.optim.Adam(param_list, lr=0.0, eps=1e-15)
        elif self.optimizer_type=="sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(param_list, lr=0.0, eps=1e-15)
            except:
                self.optimizer = torch.optim.Adam(param_list, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )
        self.exposure_scheduler_args = get_expon_lr_func(
            training_args.exposure_lr_init,
            training_args.exposure_lr_final,
            lr_delay_steps=training_args.exposure_lr_delay_steps,
            lr_delay_mult=training_args.exposure_lr_delay_mult,
            max_steps=training_args.iterations
        )

    # ==============================================================
    #  Fisher-EMA & 对角协方差  ——  稀疏触发版
    #  --------------------------------------------------------------
    #  * 如果最近一次 mini-batch 的重建损失 (ℓ) 与上次相比
    #    没有“恶化” ≥ 2 % ，则本 step 直接 early-return；
    #    Fisher Buffer 与 Σ 均沿用上一帧，省掉一次 O(N) 运算。
    # ==============================================================
    @torch.no_grad()
    def update_covariance(self,
                          grads_dict: dict[str, torch.Tensor],
                          * ,               # 仅允许关键字参数
                          cur_iter: int,
                          max_iter: int,
                          loss_scalar: float | None = None,
                          var_floor: float = 1e-6,
                          max_pts_per_group: int = 300_000):
        """
        Args
        ----
        grads_dict : 由 train.py 采集好的梯度张量，维度统一为  [N, d]
        cur_iter   : 当前迭代数（打印 α 动态信息时用）
        max_iter   : 总迭代数（用于调节 α, var-ceil）
        loss_scalar: 本 mini-batch 的标量重建损失；若 None 则始终更新
        var_floor  : Σ 的最小值，防止除 0
        max_pts_per_group : 大场景一次可能上 M 个点，可拆批
        """

        # ---------- 0. Early-Exit : 稳定阶段跳过 Fisher 更新 ----------
        if loss_scalar is not None:
            if hasattr(self, "_prev_loss_scalar"):
                # 若损失没有“恶化” ≥2 % ，直接返回 (保持上一帧 Σ)
                if loss_scalar < self._prev_loss_scalar * 1.02:
                    return
            # 记录本次 loss
            self._prev_loss_scalar = loss_scalar

        # ---------- 1. 计算动态 α、动态 var-ceil ------------------------
        # α 与 var-ceil 都随训练推进而递减
        prog = cur_iter / max_iter
        alpha_dyn = 0.9 * (1.0 - 0.5 * prog)          # 0.9 → 0.45

        if cur_iter % 2000 == 0:                       # 打印一次
            print(f"[Fisher-EMA] iter {cur_iter}: α={alpha_dyn:.3f}, "
                  f"var_ceil={var_ceil_dyn:.1f}")

        # ---------- 2. 逐 param-group 更新 Fisher & Σ ---------------
        for pname, grad in grads_dict.items():
            if (grad is None) or (grad.numel() == 0):
                continue

            eps, _ = FISHER_CFG.get(pname, (DEFAULT_EPS, DEFAULT_CEIL))

            # -- 2.1 grad²，可能需要分批，避免显存爆 --
            N, D = grad.shape
            buf_name = {
                "xyz":      "_xyz_fisher_buf",
                "scaling":  "_scaling_fisher_buf",
                "rotation": "_rotation_fisher_buf",
                "opacity":  "_opacity_fisher_buf",
                "f_dc":     "_f_dc_fisher_buf",
                "f_rest":   "_f_rest_fisher_buf",
            }[pname]

            fisher_buf = getattr(self, buf_name)
            if fisher_buf.numel() == 0:
                # 第一次初始化
                fisher_buf = torch.zeros_like(grad)

            # 分块更新，防显存顶不住
            for start in range(0, N, max_pts_per_group):
                end = min(start + max_pts_per_group, N)
                g_chunk = grad[start:end]              # [M, D]
                buf_chunk = fisher_buf[start:end]

                buf_chunk.mul_(alpha_dyn) \
                         .add_(g_chunk.square(), alpha=1.0 - alpha_dyn)

            setattr(self, buf_name, fisher_buf)

            # -- 2.2 由 Fisher 推 Σ，并 clamp --
            var = torch.reciprocal(fisher_buf + eps)   # 1/(F+ε)
            var.clamp_(min=var_floor)

            cov_name = {
                "xyz":      "_xyz_cov",
                "scaling":  "_scaling_cov",
                "rotation": "_rotation_cov",
                "opacity":  "_opacity_cov",
                "f_dc":     "_features_dc_cov",
                "f_rest":   "_features_rest_cov",
            }[pname]

            if pname in ("f_rest", "opacity"):         # 已经是向量
                setattr(self, cov_name, var)
            else:                                     # 转回对角矩阵
                eye = torch.eye(D, device=var.device).unsqueeze(0)
                setattr(self, cov_name, var.unsqueeze(-1) * eye)

    def update_learning_rate(self, iteration):
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"]=="xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def get_covariance_dict(self, batch_size=5000):
        return {
            "xyz":      self._xyz_cov,
            "opacity":  self._opacity_cov,
            "scaling":  self._scaling_cov,
            "rotation": self._rotation_cov,
            "f_dc":     self._features_dc_cov,
            "f_rest":   self._features_rest_cov
        }

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def get_covariance(self, scaling_modifier=1):
        # 用于渲染: shape [N,3,3], purely geometry transform
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree<self.max_sh_degree:
            self.active_sh_degree+=1

    def construct_list_of_attributes(self):
        l = ['x','y','z','nx','ny','nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute,'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz,normals,f_dc,f_rest,opacities,scale,rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements,'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        # 略。可与原逻辑相同
        ...

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors={}
        for group in self.optimizer.param_groups:
            if group["name"]==name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        优化器层面裁剪: 先更新 param, 再更新状态
        """
        optimizable_tensors={}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            param_len = group["params"][0].shape[0]
            mask_len = mask.shape[0]
            if mask_len!=param_len:
                logging.warning(f"[prune] mismatch param_len={param_len}, mask_len={mask_len}, do min-len cut.")
                mask = mask[:param_len]

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]

            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            if stored_state is not None:
                self.optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        同步裁剪 param, cov, fisher_buf, 确保 shape 一致
        """
        n_points = self._xyz.shape[0]
        if mask.shape[0] != n_points:
            logging.warning(f"[prune_points] shape mismatch: param pts={n_points}, mask={mask.shape[0]}")
            mask = mask[:n_points]

        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        oldN = self._xyz_cov.shape[0]
        newN = valid_points_mask.sum().item()
        if newN < oldN:
            # --- 1) Cov 的裁剪 ---
            self._xyz_cov          = self._xyz_cov[valid_points_mask]
            self._opacity_cov      = self._opacity_cov[valid_points_mask]
            self._scaling_cov      = self._scaling_cov[valid_points_mask]
            self._rotation_cov     = self._rotation_cov[valid_points_mask]
            self._features_dc_cov  = self._features_dc_cov[valid_points_mask]
            self._features_rest_cov= self._features_rest_cov[valid_points_mask]

            # --- 2) Fisher buf 的裁剪 (新增) ---
            self._xyz_fisher_buf      = self._xyz_fisher_buf[valid_points_mask]
            self._opacity_fisher_buf  = self._opacity_fisher_buf[valid_points_mask]
            self._scaling_fisher_buf  = self._scaling_fisher_buf[valid_points_mask]
            self._rotation_fisher_buf = self._rotation_fisher_buf[valid_points_mask]
            self._f_dc_fisher_buf     = self._f_dc_fisher_buf[valid_points_mask]
            self._f_rest_fisher_buf   = self._f_rest_fisher_buf[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if hasattr(self, "tmp_radii"):
            if self.tmp_radii.shape[0] >= valid_points_mask.shape[0]:
                self.tmp_radii = self.tmp_radii[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        为 densify 新增点: cat 到 param, 同时 cat 到 optimizer.state
        """
        optimizable_tensors={}
        for group in self.optimizer.param_groups:
            assert len(group["params"])==1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                old_param = group["params"][0]
                del self.optimizer.state[old_param]
                group["params"][0] = nn.Parameter(
                    torch.cat((old_param, extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_covariances(self, xyz_count, new_count):
        """
        为 cov + fisher_buf 同步扩展, 给新点赋予一定的初始 cov (这里简单给 0.1).
        同时也给 fisher_buf 扩展 0.0.
        """
        device = self._xyz_cov.device
        # ----- Cov expansions -----
        new_xyz_cov = 0.1 * torch.eye(3, device=device).unsqueeze(0).repeat(new_count,1,1)
        self._xyz_cov = torch.cat((self._xyz_cov, new_xyz_cov), dim=0)

        new_fdc_cov = 0.1*torch.eye(3, device=device).unsqueeze(0).repeat(new_count,1,1)
        self._features_dc_cov = torch.cat((self._features_dc_cov, new_fdc_cov), dim=0)

        D_ = self._features_rest_cov.shape[1]
        new_frest_cov = 0.1*torch.ones((new_count, D_), device=device)
        self._features_rest_cov = torch.cat((self._features_rest_cov, new_frest_cov), dim=0)

        new_scaling_cov = 0.1*torch.eye(3, device=device).unsqueeze(0).repeat(new_count,1,1)
        self._scaling_cov = torch.cat((self._scaling_cov, new_scaling_cov), dim=0)

        new_rotation_cov = 0.1*torch.eye(4, device=device).unsqueeze(0).repeat(new_count,1,1)
        self._rotation_cov = torch.cat((self._rotation_cov, new_rotation_cov), dim=0)

        new_opacity_cov = 0.1*torch.ones((new_count,1), device=device)
        self._opacity_cov = torch.cat((self._opacity_cov, new_opacity_cov), dim=0)

        # ----- Fisher buf expansions (新增) -----
        new_xyz_fish   = torch.zeros((new_count, 3), device=device)
        self._xyz_fisher_buf = torch.cat((self._xyz_fisher_buf, new_xyz_fish), dim=0)

        new_fdc_fish   = torch.zeros((new_count, 3), device=device)
        self._f_dc_fisher_buf = torch.cat((self._f_dc_fisher_buf, new_fdc_fish), dim=0)

        new_frest_fish = torch.zeros((new_count, D_), device=device)
        self._f_rest_fisher_buf = torch.cat((self._f_rest_fisher_buf, new_frest_fish), dim=0)

        new_scaling_fish = torch.zeros((new_count,3), device=device)
        self._scaling_fisher_buf = torch.cat((self._scaling_fisher_buf, new_scaling_fish), dim=0)

        new_rotation_fish = torch.zeros((new_count,4), device=device)
        self._rotation_fisher_buf = torch.cat((self._rotation_fisher_buf, new_rotation_fish), dim=0)

        new_opacity_fish = torch.zeros((new_count,1), device=device)
        self._opacity_fisher_buf = torch.cat((self._opacity_fisher_buf, new_opacity_fish), dim=0)

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation, new_tmp_radii):
        """
        在 densify 后对 param + cov 同步扩容
        """
        # 1) 先对 param cat
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation
        }
        old_count = self.get_xyz.shape[0]
        new_count = new_xyz.shape[0]

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 2) 同步扩展 covariance
        self.cat_covariances(old_count, new_count)

        # 3) 同步更新 tmp_radii & grad accum
        if not hasattr(self,"tmp_radii"):
            self.tmp_radii = torch.zeros((old_count), device="cuda")
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0],1),device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0],1),device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]),device="cuda")

        # clamp
        with torch.no_grad():
            self._xyz.data.clamp_(-1e5,1e5)
            self._scaling.data.clamp_(-10,10)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points),device="cuda")
        if grads.numel()>0:
            padded_grad[:grads.shape[0]] = grads.squeeze()

        selected_pts_mask = (padded_grad>=grad_threshold)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values>self.percent_dense*scene_extent
        )
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0),3),device="cuda")
        samples = torch.normal(mean=means,std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N,1)

        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N,1)/(0.8*N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest= self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacity, new_scaling, new_rotation, new_tmp_radii)
        # prune
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N*selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)

        with torch.no_grad():
            self._xyz.data.clamp_(-1e5,1e5)
            self._scaling.data.clamp_(-10,10)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        selected_pts_mask = torch.zeros(n_init_points,dtype=torch.bool, device="cuda")
        if grads.numel()>0:
            selected_pts_mask = (torch.norm(grads,dim=-1)>=grad_threshold)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling,dim=1).values<=self.percent_dense*scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest= self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation= self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz,new_features_dc,new_features_rest,
                                   new_opacities,new_scaling,new_rotation,new_tmp_radii)
        with torch.no_grad():
            self._xyz.data.clamp_(-1e5,1e5)
            self._scaling.data.clamp_(-10,10)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum/self.denom
        grads[grads.isnan()]=0.0
        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity<min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = (self.max_radii2D>max_screen_size)
            big_points_ws = (self.get_scaling.max(dim=1).values>0.1*extent)
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask,big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        self.tmp_radii=None
        torch.cuda.empty_cache()
        with torch.no_grad():
            self._xyz.data.clamp_(-1e5,1e5)
            self._scaling.data.clamp_(-10,10)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if viewspace_point_tensor.grad is not None:
            self.xyz_gradient_accum[update_filter]+=torch.norm(viewspace_point_tensor.grad[update_filter,:2],
                                                               dim=-1,keepdim=True)
            self.denom[update_filter]+=1
        with torch.no_grad():
            self._xyz.data.clamp_(-1e5,1e5)
            self._scaling.data.clamp_(-10,10)

    @property
    def get_parameters(self):
        tlist = []
        for t_ in [
            self._xyz,self._features_dc,self._features_rest,
            self._scaling,self._rotation,self._opacity
        ]:
            if t_.numel()>0:
                t_leaf = t_.detach().clone().requires_grad_(True)
                tlist.append(t_leaf.reshape(-1))
        return torch.cat(tlist) if tlist else torch.tensor([],requires_grad=True)

    @property
    def get_param_covariance(self):
        cov_list=[]
        # xyz_cov => [N,3,3] => diagonal => [N,3]
        # ...
        if self._xyz_cov.numel()>0:
            cov_list.append(self._xyz_cov.diagonal(dim1=-2,dim2=-1).reshape(-1))
        if self._opacity_cov.numel()>0:
            cov_list.append(self._opacity_cov.reshape(-1))
        if self._scaling_cov.numel()>0:
            cov_list.append(self._scaling_cov.diagonal(dim1=-2,dim2=-1).reshape(-1))
        if self._rotation_cov.numel()>0:
            cov_list.append(self._rotation_cov.diagonal(dim1=-2,dim2=-1).reshape(-1))
        if self._features_dc_cov.numel()>0:
            cov_list.append(self._features_dc_cov.diagonal(dim1=-2,dim2=-1).reshape(-1))
        if self._features_rest_cov.numel()>0:
            cov_list.append(self._features_rest_cov.reshape(-1))

        if cov_list:
            return torch.cat(cov_list)
        else:
            return torch.tensor([], requires_grad=True)