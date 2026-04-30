"""
longstream/data/stream_feeder.py
--------------------------------
基于生成器的帧馈送器，模拟实时摄像头按 FPS 逐帧吐帧。
支持来源：image_dir | video | npz | generalizable 数据目录。
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torch
import torchvision.transforms as tvf

# ── 图像归一化（与 DUSt3R / load_images_for_eval 完全一致）
# ImgNorm: ToTensor([0,1]) + Normalize(0.5,0.5) → [-1,1]
# _load_images 还会做 (x+1)/2 → [0,1]，_preprocess_single_image 也复现此步骤
_ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

_SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


@dataclass
class FramePacket:
    """单帧数据包：携带 CPU 数组和 CPU 张量，禁止含 GPU Tensor。"""

    frame_index: int
    rgb: np.ndarray               # uint8, HWC, CPU
    image_tensor: torch.Tensor    # [1, 1, C, H, W], CPU float（归一化）
    image_path: Optional[str]     # 原始文件路径，视频/npz 时可为 None
    gps_xyz: Optional[np.ndarray] # [3,] float32，GPS 位置（世界坐标或 ECEF），无则 None
    gt_pose: Optional[np.ndarray] # [4, 4] float32 w2c，无则 None


# ═══════════════════════════════════════════════════════════════════════════
#  内部工具函数
# ═══════════════════════════════════════════════════════════════════════════

def _pose_to_camera_center(w2c: np.ndarray) -> np.ndarray:
    """
    从 w2c [4, 4] 或 [3, 4] 矩阵提取相机中心（世界坐标）[3,] float32。

    公式: center = -R^T @ t
    严禁直接取 w2c[:3, 3]（那是相机坐标系下的 t，不是世界坐标）。

    与 dataloader.py L553-L556 的计算方式一致：
        gps_xyz = -np.einsum('nji,nj->ni', R, t)
    """
    R = w2c[:3, :3].astype(np.float64)
    t = w2c[:3, 3].astype(np.float64)
    return -(R.T @ t).astype(np.float32)


def _preprocess_single_image(
    pil_img: PIL.Image.Image,
    size: int = 518,
    patch_size: int = 14,
    crop: bool = False,
) -> torch.Tensor:
    """
    与 LongStreamDataLoader._load_images + load_images_for_eval 完全一致的预处理。
    返回 [1, 1, C, H, W] CPU float Tensor，值域 [0, 1]。

    流程：
      1. 长边缩放到 size
      2. 中心 crop/resize 对齐 patch_size 倍数
      3. ImgNorm → [-1, 1]
      4. (x+1)/2 → [0, 1]  （与 _load_images 中的 (images+1)/2 一致）
    """
    img = pil_img.convert("RGB")
    # Step 1: resize long edge to size
    S = max(img.size)
    interp_down = PIL.Image.LANCZOS
    interp_up = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * size / S)) for x in img.size)
    img = img.resize(new_size, interp_down if S > size else interp_up)
    # Step 2: align to patch_size
    W, H = img.size
    cx, cy = W // 2, H // 2
    halfw = ((2 * cx) // patch_size) * (patch_size // 2)
    halfh = ((2 * cy) // patch_size) * (patch_size // 2)
    # 对齐 image.py L211：方形图像按 4:3 处理（如果不是正方形 crop）
    if size != 224 and W == H:
        halfh = int(3 * halfw / 4)
    halfw = max(halfw, patch_size // 2)
    halfh = max(halfh, patch_size // 2)
    if crop:
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    else:
        img = img.resize((2 * halfw, 2 * halfh), PIL.Image.LANCZOS)
    # Step 3+4: ImgNorm then rescale to [0,1]
    tensor = _ImgNorm(img)          # [C, H, W], [-1, 1]
    tensor = (tensor + 1.0) / 2.0  # [C, H, W], [0, 1]
    return tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]


def _read_npz_compatible(path: str) -> Tuple[
    Optional[np.ndarray],   # images: [N, H, W, 3] uint8
    Optional[np.ndarray],   # poses:  [N, 4, 4] float32
    Optional[np.ndarray],   # gps_xyz: [N, 3] float32
]:
    """
    兼容多种 key 命名规范的 NPZ 读取器。

    RGB key 尝试顺序:  images → rgb → rgbs
    pose key 优先:     pred_cam_T → cam_T → w2c → extri → poses → pred_poses
    points/depth/conf 在此函数内不读取（由上层按需处理）。
    返回 None 表示该字段在文件中不存在。
    """
    data = np.load(path, allow_pickle=True)

    # ── RGB ──────────────────────────────────────────────────────────────
    images_np: Optional[np.ndarray] = None
    for key in ("images", "rgb", "rgbs"):
        if key in data:
            arr = data[key]
            # 统一到 uint8 HWC 格式
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            if arr.ndim == 4 and arr.shape[-1] == 3:
                images_np = arr               # [N, H, W, 3]
            elif arr.ndim == 4 and arr.shape[1] == 3:
                images_np = arr.transpose(0, 2, 3, 1).astype(np.uint8)  # CHW→HWC
            elif arr.ndim == 3 and arr.shape[-1] == 3:
                images_np = arr[np.newaxis]   # 单帧
            break

    # ── Pose ──────────────────────────────────────────────────────────────
    poses_np: Optional[np.ndarray] = None
    for key in ("pred_cam_T", "cam_T", "w2c", "extri", "poses", "pred_poses"):
        if key in data:
            arr = np.asarray(data[key], dtype=np.float32)
            # 补充到 [N, 4, 4]
            if arr.ndim == 3 and arr.shape[-2:] == (3, 4):
                N = arr.shape[0]
                homo = np.zeros((N, 4, 4), dtype=np.float32)
                homo[:, :3, :] = arr
                homo[:, 3, 3] = 1.0
                arr = homo
            if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                poses_np = arr
            break

    # ── GPS ───────────────────────────────────────────────────────────────
    gps_np: Optional[np.ndarray] = None
    for key in ("gps_xyz", "gps", "position"):
        if key in data:
            gps_np = np.asarray(data[key], dtype=np.float32)
            if gps_np.ndim == 1:
                gps_np = gps_np[np.newaxis]
            break

    return images_np, poses_np, gps_np


# ═══════════════════════════════════════════════════════════════════════════
#  StreamFeeder 主类
# ═══════════════════════════════════════════════════════════════════════════

class StreamFeeder:
    """
    帧生成器，逐帧 yield FramePacket，可控制目标 FPS。

    Parameters
    ----------
    source : str
        输入来源路径。根据 ``source_type`` 解释：
        - ``image_dir``:      图片目录路径
        - ``video``:          视频文件路径
        - ``npz``:            .npz 文件路径
        - ``generalizable``:  generalizable 数据目录
    source_type : str
        ``image_dir`` | ``video`` | ``npz`` | ``generalizable``
    fps : float
        目标帧率。若为 0 或 None，则尽快推帧（不限速）。
    size : int
        模型输入长边像素数（预处理时用）。
    patch_size : int
        Patch 尺寸，用于对齐分辨率。
    max_frames : int | None
        最多读取帧数，None 表示全部。
    camera : str | None
        generalizable 模式下的子摄像头名称。
    crop : bool
        是否使用 center-crop（True）还是 resize（False，默认）对齐 patch_size。
        与 LongStreamDataLoader.crop 语义相同。
    data_cfg : dict | None
        generalizable 模式下的完整 data 配置，用于复用 LongStreamDataLoader
        的路径解析逻辑（data_roots_file / seq_list / camera 等）。
        若为 None，则直接以 source 作为场景根目录。
    """

    def __init__(
        self,
        source: str,
        source_type: str = "image_dir",
        fps: float = 0.0,
        size: int = 518,
        patch_size: int = 14,
        max_frames: Optional[int] = None,
        camera: Optional[str] = None,
        crop: bool = False,
        data_cfg: Optional[dict] = None,
    ) -> None:
        self.source = source
        self.source_type = source_type
        self.fps = float(fps) if fps else 0.0
        self.size = size
        self.patch_size = patch_size
        self.max_frames = max_frames
        self.camera = camera
        self.crop = crop
        self._data_cfg = data_cfg

    # ── 对外接口 ──────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[FramePacket]:
        return self._generate()

    def _generate(self) -> Generator[FramePacket, None, None]:
        target_dt = (1.0 / self.fps) if self.fps > 0 else 0.0

        if self.source_type == "image_dir":
            raw_iter = self._iter_image_dir()
        elif self.source_type == "video":
            raw_iter = self._iter_video()
        elif self.source_type == "npz":
            raw_iter = self._iter_npz()
        elif self.source_type == "generalizable":
            raw_iter = self._iter_generalizable()
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type!r}")

        count = 0
        t_last = time.monotonic()
        for packet in raw_iter:
            if self.max_frames is not None and count >= self.max_frames:
                break

            if target_dt > 0:
                elapsed = time.monotonic() - t_last
                remaining = target_dt - elapsed
                if remaining > 0.001:
                    time.sleep(remaining)
            t_last = time.monotonic()

            yield packet
            count += 1

    # ── 数据源适配器 ──────────────────────────────────────────────────────

    def _iter_image_dir(self) -> Generator[FramePacket, None, None]:
        """遍历图片目录，按文件名排序。使用 PIL.Image.open 以支持 Unicode 路径。"""
        paths: List[str] = []
        for ext in _SUPPORTED_IMG_EXTS:
            paths.extend(glob.glob(os.path.join(self.source, f"*{ext}")))
            paths.extend(glob.glob(os.path.join(self.source, f"*{ext.upper()}")))
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(f"No images found in {self.source!r}")

        for idx, path in enumerate(paths):
            pil_img = exif_transpose(PIL.Image.open(path)).convert("RGB")
            rgb = np.array(pil_img, dtype=np.uint8)  # HWC uint8
            tensor = _preprocess_single_image(pil_img, self.size, self.patch_size, self.crop)
            yield FramePacket(
                frame_index=idx,
                rgb=rgb,
                image_tensor=tensor,
                image_path=path,
                gps_xyz=None,
                gt_pose=None,
            )

    def _iter_video(self) -> Generator[FramePacket, None, None]:
        """逐帧读取视频文件（cv2.VideoCapture）。"""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.source!r}")
        idx = 0
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = PIL.Image.fromarray(rgb)
                tensor = _preprocess_single_image(pil_img, self.size, self.patch_size, self.crop)
                yield FramePacket(
                    frame_index=idx,
                    rgb=rgb,
                    image_tensor=tensor,
                    image_path=None,
                    gps_xyz=None,
                    gt_pose=None,
                )
                idx += 1
        finally:
            cap.release()

    def _iter_npz(self) -> Generator[FramePacket, None, None]:
        """读取 .npz 文件，支持多种 key 命名（兼容 _read_npz_compatible）。"""
        images_np, poses_np, gps_np = _read_npz_compatible(self.source)
        if images_np is None:
            raise ValueError(
                f"NPZ 文件 {self.source!r} 中找不到有效 RGB 数据"
                " (尝试过 keys: images, rgb, rgbs)"
            )
        N = len(images_np)
        for idx in range(N):
            rgb = images_np[idx]  # [H, W, 3] uint8
            pil_img = PIL.Image.fromarray(rgb)
            tensor = _preprocess_single_image(pil_img, self.size, self.patch_size, self.crop)
            gps = gps_np[idx] if (gps_np is not None and idx < len(gps_np)) else None
            pose = poses_np[idx] if (poses_np is not None and idx < len(poses_np)) else None
            # TTO 需要 gps_xyz；若 NPZ 无 GPS 但有 GT 位姿，就从位姿提取相机中心作为虚拟 GPS
            if gps is None and pose is not None:
                gps = _pose_to_camera_center(pose)
            yield FramePacket(
                frame_index=idx,
                rgb=rgb,
                image_tensor=tensor,
                image_path=None,
                gps_xyz=gps,
                gt_pose=pose,
            )

    def _iter_generalizable(self) -> Generator[FramePacket, None, None]:
        """
        遍历 generalizable 格式数据目录。

        若构造时传入了 ``data_cfg``，则使用 LongStreamDataLoader.iter_sequence_infos()
        以复用其完整路径解析逻辑（data_roots_file / seq_list / camera 等）。
        否则以 self.source 作为单个场景根目录直接扫描。

        使用 PIL.Image.open 读取图片（天然支持 Unicode 路径）。
        """
        if self._data_cfg is not None:
            yield from self._iter_generalizable_via_loader()
            return

        # ── 直接扫描单个场景目录 ────────────────────────────────────────
        img_exts = list(_SUPPORTED_IMG_EXTS)
        if self.camera:
            img_dir_candidates = [
                os.path.join(self.source, "images", self.camera),
                os.path.join(self.source, self.camera),
            ]
        else:
            img_dir_candidates = [
                os.path.join(self.source, "images"),
                self.source,
            ]

        img_dir = None
        for candidate in img_dir_candidates:
            if os.path.isdir(candidate):
                img_dir = candidate
                break
        if img_dir is None:
            raise FileNotFoundError(
                f"Cannot locate image directory under {self.source!r}"
            )

        paths: List[str] = []
        for ext in img_exts:
            paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
            paths.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))
        paths = sorted(set(paths))
        if not paths:
            raise FileNotFoundError(f"No images found in {img_dir!r}")

        gps_file = os.path.join(self.source, "gps_xyz.npy")
        gps_all: Optional[np.ndarray] = None
        if os.path.isfile(gps_file):
            gps_all = np.load(gps_file).astype(np.float32)

        poses_all: Optional[np.ndarray] = None
        for pose_candidate in [
            os.path.join(self.source, "gt_poses.npy"),
            os.path.join(self.source, "cameras",
                         self.camera or "00", "gt_poses.npy"),
        ]:
            if os.path.isfile(pose_candidate):
                arr = np.load(pose_candidate).astype(np.float32)
                if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                    poses_all = arr
                break

        for idx, path in enumerate(paths):
            pil_img = exif_transpose(PIL.Image.open(path)).convert("RGB")
            rgb = np.array(pil_img, dtype=np.uint8)
            tensor = _preprocess_single_image(pil_img, self.size, self.patch_size, self.crop)
            gps = gps_all[idx] if (gps_all is not None and idx < len(gps_all)) else None
            pose = poses_all[idx] if (poses_all is not None and idx < len(poses_all)) else None
            # TTO 需要 gps_xyz；若目录无 GPS 但有 GT 位姿，就从位姿提取相机中心作为虚拟 GPS
            if gps is None and pose is not None:
                gps = _pose_to_camera_center(pose)
            yield FramePacket(
                frame_index=idx,
                rgb=rgb,
                image_tensor=tensor,
                image_path=path,
                gps_xyz=gps,
                gt_pose=pose,
            )

    def _iter_generalizable_via_loader(self) -> Generator[FramePacket, None, None]:
        """
        借助 LongStreamDataLoader.iter_sequence_infos() 解析路径列表，
        逐帧逐序列 yield FramePacket（全局 frame_index 跨序列累加）。
        仅读取单个序列（seq_list[0]）以匹配实时单流语义。
        """
        from longstream.data.dataloader import LongStreamDataLoader

        loader = LongStreamDataLoader(self._data_cfg)
        global_idx = 0
        for seq_info in loader.iter_sequence_infos():
            scene_root = seq_info.scene_root

            # GPS
            gps_all: Optional[np.ndarray] = None
            gps_file = os.path.join(scene_root, "gps_xyz.npy")
            if os.path.isfile(gps_file):
                gps_all = np.load(gps_file).astype(np.float32)

            # GT poses
            poses_all: Optional[np.ndarray] = None
            for pose_candidate in [
                os.path.join(scene_root, "gt_poses.npy"),
                os.path.join(scene_root, "cameras",
                             seq_info.camera or "00", "gt_poses.npy"),
            ]:
                if os.path.isfile(pose_candidate):
                    arr = np.load(pose_candidate).astype(np.float32)
                    if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                        poses_all = arr
                    break

            local_idx = 0
            for path in seq_info.image_paths:
                pil_img = exif_transpose(PIL.Image.open(path)).convert("RGB")
                rgb = np.array(pil_img, dtype=np.uint8)
                tensor = _preprocess_single_image(pil_img, self.size, self.patch_size, self.crop)
                gps = gps_all[local_idx] if (gps_all is not None and local_idx < len(gps_all)) else None
                pose = poses_all[local_idx] if (poses_all is not None and local_idx < len(poses_all)) else None
                # TTO 需要 gps_xyz；若无 GPS 但有 GT 位姿，就从位姿提取相机中心作为虚拟 GPS
                if gps is None and pose is not None:
                    gps = _pose_to_camera_center(pose)
                yield FramePacket(
                    frame_index=global_idx,
                    rgb=rgb,
                    image_tensor=tensor,
                    image_path=path,
                    gps_xyz=gps,
                    gt_pose=pose,
                )
                local_idx += 1
                global_idx += 1
