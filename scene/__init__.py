#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import os, random, json, logging, types, torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    """
    Helper that owns – and knows how to (de)serialize – the training / test
    cameras **and** the GaussianModel for a single scene.
    """

    gaussians: GaussianModel

    # --------------------------------------------------------------------- #
    #                               ctor                                    #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        self.model_path   = args.model_path
        self.gaussians    = gaussians
        self.loaded_iter  = None        # ← after we find which iter to load

        # ── 0. 解析当前要加载的迭代 ────────────────────────────────────────
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        # ── 1. 解析数据集（Colmap / Blender） ────────────────────────────
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.depths,
                args.eval,
                args.train_test_exp,
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.depths,
                args.eval,
            )
        else:
            raise RuntimeError("Could not recognize scene type!")

        # ── 2. 把原始 input.ply & cameras.json 复制 / 写到 output 目录 ────
        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dst:
                dst.write(src.read())

            camlist = (scene_info.test_cameras or []) + (
                scene_info.train_cameras or []
            )
            json.dump(
                [camera_to_JSON(i, cam) for i, cam in enumerate(camlist)],
                open(os.path.join(self.model_path, "cameras.json"), "w"),
            )

        # ── 3. 打乱相机顺序（可复现的随机） ───────────────────────────────
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # ── 4. 缩放 (multi-res) 相机列表缓存 ───────────────────────────────
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_cameras, self.test_cameras = {}, {}
        for s in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[s] = cameraList_from_camInfos(
                scene_info.train_cameras, s, args, scene_info.is_nerf_synthetic, False
            )
            print("Loading Test Cameras")
            self.test_cameras[s] = cameraList_from_camInfos(
                scene_info.test_cameras, s, args, scene_info.is_nerf_synthetic, True
            )

        # ── 5. 载入 / 初始化 Gaussian 模型 ───────────────────────────────
        if self.loaded_iter:           # ★ 优先尝试二进制快照 ★
            state_path = os.path.join(
                self.model_path, "point_cloud", f"iteration_{self.loaded_iter}", "state.pth"
            )
            if os.path.isfile(state_path):
                logging.info(f"✔  Found binary checkpoint: {state_path}")
                ckpt_tuple = torch.load(state_path, map_location="cuda")

                dummy_opt = types.SimpleNamespace(
                    optimizer_type="default",
                    iterations             = 0,
                    position_lr_init       = 0.0,
                    position_lr_final      = 0.0,
                    position_lr_delay_mult = 1.0,
                    position_lr_max_steps  = 1,
                    feature_lr             = 0.0,
                    opacity_lr             = 0.0,
                    scaling_lr             = 0.0,
                    rotation_lr            = 0.0,
                    exposure_lr_init       = 0.0,
                    exposure_lr_final      = 0.0,
                    exposure_lr_delay_steps= 0,
                    exposure_lr_delay_mult = 1.0,
                    percent_dense          = 0.0,
                )
                self.gaussians.restore(ckpt_tuple, dummy_opt)
            else:
                ply_path = os.path.join(
                    self.model_path,
                    "point_cloud",
                    f"iteration_{self.loaded_iter}",
                    "point_cloud.ply",
                )
                logging.warning(f"⚠  state.pth not found, fallback to {ply_path}")
                self.gaussians.load_ply(ply_path, args.train_test_exp)
        else:
            # fresh start
            self.gaussians.create_from_pcd(
                scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent
            )

    # ------------------------------------------------------------------ #
    #                          🔖  I/O util                               #
    # ------------------------------------------------------------------ #
    def save(self, iteration: int):
        pc_dir = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        os.makedirs(pc_dir, exist_ok=True)

        # 1) PLY
        self.gaussians.save_ply(os.path.join(pc_dir, "point_cloud.ply"))

        # 2) 完整二进制快照
        torch.save(self.gaussians.capture(), os.path.join(pc_dir, "state.pth"))

        # 3) Exposure
        exposure = {
            k: self.gaussians.get_exposure_from_name(k).detach().cpu().numpy().tolist()
            for k in self.gaussians.exposure_mapping
        }
        json.dump(exposure, open(os.path.join(self.model_path, "exposure.json"), "w"), indent=2)

    # ------------------------------------------------------------------ #
    #                        🔖  tiny helpers                             #
    # ------------------------------------------------------------------ #
    def getTrainCameras(self, scale: float = 1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale: float = 1.0):
        return self.test_cameras[scale]
