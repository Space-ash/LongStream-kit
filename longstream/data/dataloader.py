import os
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional, Tuple

import cv2
import numpy as np
import torch

from longstream.utils.vendor.dust3r.utils.image import load_images_for_eval
from longstream.utils.filter import is_high_quality
from longstream.utils.gt_pose import resolve_gt_poses, subset_pose_array

dataset_metadata: Dict[str, Dict[str, Any]] = {
    "davis": {
        "img_path": "data/davis/DAVIS/JPEGImages/480p",
        "mask_path": "data/davis/DAVIS/masked_images/480p",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,
    },
    "kitti": {
        "img_path": "data/kitti/sequences",
        "anno_path": "data/kitti/poses",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "image_2"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            anno_path, f"{seq}.txt"
        )
        if os.path.exists(os.path.join(anno_path, f"{seq}.txt"))
        else None,
        "traj_format": "kitti",
        "seq_list": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "bonn": {
        "img_path": "data/bonn/rgbd_bonn_dataset",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "rgb_110"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "groundtruth_110.txt"
        ),
        "traj_format": "tum",
        "seq_list": ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "nyu": {
        "img_path": "data/nyu-v2/val/nyu_images",
        "mask_path": None,
        "process_func": None,
    },
    "scannet": {
        "img_path": "data/scannetv2",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_90.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "tum": {
        "img_path": "data/tum",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "rgb_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "groundtruth_90.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "sintel": {
        "img_path": "data/sintel/training/final",
        "anno_path": "data/sintel/training/camdata_left",
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": None,
        "seq_list": [
            "alley_2",
            "ambush_4",
            "ambush_5",
            "ambush_6",
            "cave_2",
            "cave_4",
            "market_2",
            "market_5",
            "market_6",
            "shaman_3",
            "sleeping_1",
            "sleeping_2",
            "temple_2",
            "temple_3",
        ],
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "waymo": {
        "img_path": "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/tao02.xie/datasets/scatt3r_evaluation/waymo_open_dataset_v1_4_3",
        "anno_path": None,
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path,
            seq.split("_cam")[0] if "_cam" in seq else seq,
            "images",
            seq.split("_cam")[1] if "_cam" in seq else "00",
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path,
            seq.split("_cam")[0] if "_cam" in seq else seq,
            "cameras",
            seq.split("_cam")[1] if "_cam" in seq else "00",
            "extri.yml",
        ),
        "traj_format": "waymo",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
}


@dataclass
class LongStreamSequenceInfo:
    name: str
    scene_root: str
    image_dir: str
    image_paths: List[str]
    camera: Optional[str]


class LongStreamSequence:
    def __init__(
        self,
        name: str,
        images: torch.Tensor,
        image_paths: List[str],
        scene_root: Optional[str] = None,
        image_dir: Optional[str] = None,
        camera: Optional[str] = None,
        gt_poses: Optional[np.ndarray] = None,
        original_frame_indices: Optional[List[int]] = None,
        gt_poses_source: str = "none",
        gps_xyz: Optional[np.ndarray] = None,
    ):
        self.name = name
        self.images = images
        self.image_paths = image_paths
        self.scene_root = scene_root
        self.image_dir = image_dir
        self.camera = camera
        # GT poses as [S, 4, 4] float32 array (world-to-camera, first frame = identity).
        self.gt_poses: Optional[np.ndarray] = gt_poses
        # 筛帧后保留帧的原始索引（未筛帧时为 None表示顺序完整）
        self.original_frame_indices: Optional[List[int]] = original_frame_indices
        # GT 位姿来源标记: "camera_yml" | "npy" | "none"
        self.gt_poses_source: str = gt_poses_source
        # 从 GT w2c 位姿提取的相机中心 [S, 3]（世界坐标系），作为虚拟 GPS 信号。
        # 计算方式: center = -R^T @ t，严格禁止直接使用 [:3, 3]。
        self.gps_xyz: Optional[np.ndarray] = gps_xyz


def _read_list_file(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = []
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            lines.append(line)
    return lines


def _is_generalizable_scene_root(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "images"))


def _direct_image_files(dir_path: str) -> List[str]:
    filelist = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    if not filelist:
        filelist = sorted(glob.glob(os.path.join(dir_path, "*.jpg")))
    if not filelist:
        filelist = sorted(glob.glob(os.path.join(dir_path, "*.jpeg")))
    return filelist


class LongStreamDataLoader:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.dataset = cfg.get("dataset", None)
        meta = dataset_metadata.get(self.dataset, {})
        self.img_path = cfg.get("img_path", meta.get("img_path"))
        self.mask_path = cfg.get("mask_path", meta.get("mask_path"))
        self.dir_path_func = meta.get("dir_path_func", lambda p, s: os.path.join(p, s))
        self.mask_path_seq_func = meta.get("mask_path_seq_func", lambda p, s: None)
        self.full_seq = bool(cfg.get("full_seq", meta.get("full_seq", True)))
        self.seq_list = cfg.get("seq_list", None)
        self.stride = int(cfg.get("stride", 1))
        self.max_frames = cfg.get("max_frames", None)
        self.size = int(cfg.get("size", 518))
        self.crop = bool(cfg.get("crop", False))
        self.patch_size = int(cfg.get("patch_size", 14))
        self.format = cfg.get("format", "auto")
        self.data_roots_file = cfg.get("data_roots_file", None)
        self.split = cfg.get("split", None)
        self.camera = cfg.get("camera", None)

        # --- GT 位姿来源 ---
        # gt_source: "camera_yml" | "npy" | "auto"  优先 cameras/extri.yml，退回 gt_poses.npy
        self.gt_source: str = str(cfg.get("gt_source", "auto"))
        self.gt_poses_file: Optional[str] = cfg.get("gt_poses_file", "gt_poses.npy")

        # --- 帧质量过滤 ---
        filter_cfg = cfg.get("filter", {})
        self.filter_enabled: bool = bool(
            filter_cfg.get(
                "frame_filter_enabled",
                filter_cfg.get("enabled", False),
            )
        )
        self.filter_blur_threshold: float = float(
            filter_cfg.get("blur_threshold", 100.0)
        )
        self.filter_motion_threshold: float = float(
            filter_cfg.get("motion_threshold", 0.02)
        )

    def _infer_format(self) -> str:
        if self.format in ["relpose", "generalizable"]:
            return self.format
        if self.img_path is None:
            return "relpose"
        if _is_generalizable_scene_root(self.img_path):
            return "generalizable"
        default_list = self.data_roots_file or "data_roots.txt"
        if os.path.exists(os.path.join(self.img_path, default_list)):
            return "generalizable"
        return "relpose"

    def _resolve_seq_list_generalizable(self) -> List[str]:
        if self.seq_list is not None:
            return list(self.seq_list)
        if self.img_path is None or not os.path.isdir(self.img_path):
            return []

        if _is_generalizable_scene_root(self.img_path):
            return [self.img_path]

        candidates = []
        if isinstance(self.data_roots_file, str) and self.data_roots_file:
            candidates.append(self.data_roots_file)
        if isinstance(self.split, str) and self.split:
            split_name = self.split.lower()
            if split_name in ["val", "valid", "validate"]:
                split_name = "validate"
            candidates.append(f"{split_name}_data_roots.txt")
        candidates.append("data_roots.txt")
        candidates.append("train_data_roots.txt")
        candidates.append("validate_data_roots.txt")

        for fname in candidates:
            path = os.path.join(self.img_path, fname)
            if os.path.exists(path):
                return _read_list_file(path)

        img_dirs = sorted(
            glob.glob(os.path.join(self.img_path, "**", "images"), recursive=True)
        )
        scene_roots = [os.path.dirname(p) for p in img_dirs]

        rels = []
        for p in scene_roots:
            try:
                rels.append(os.path.relpath(p, self.img_path))
            except ValueError:
                rels.append(p)
        return sorted(set(rels))

    def _resolve_seq_list_relpose(self) -> List[str]:
        if self.seq_list is not None:
            return list(self.seq_list)
        meta = dataset_metadata.get(self.dataset, {})
        if self.full_seq:
            if self.img_path is None or not os.path.isdir(self.img_path):
                return []
            seqs = [
                s
                for s in os.listdir(self.img_path)
                if os.path.isdir(os.path.join(self.img_path, s))
            ]
            return sorted(seqs)
        seqs = meta.get("seq_list", []) or []
        return list(seqs)

    def _resolve_seq_list(self) -> List[str]:
        fmt = self._infer_format()
        if fmt == "generalizable":
            return self._resolve_seq_list_generalizable()
        return self._resolve_seq_list_relpose()

    def _resolve_scene_root(self, seq_entry: str) -> Tuple[str, str]:
        if os.path.isabs(seq_entry):
            scene_root = seq_entry
            name = os.path.basename(os.path.normpath(seq_entry))
        else:
            # Always join with img_path for relative paths, even if they
            # contain path separators (e.g. "Scene01/clone").
            scene_root = os.path.join(self.img_path, seq_entry)
            name = seq_entry.replace(os.path.sep, "_")
        return name, scene_root

    def _resolve_image_dir_generalizable(self, scene_root: str) -> Optional[str]:
        images_root = os.path.join(scene_root, "images")
        if not os.path.isdir(images_root):
            return None

        if isinstance(self.camera, str) and self.camera:
            cam_dir = os.path.join(images_root, self.camera)
            if os.path.isdir(cam_dir):
                return cam_dir

        if _direct_image_files(images_root):
            return images_root

        cams = [
            d
            for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d))
        ]
        if not cams:
            return None
        cams = sorted(cams)

        frame_dirs = []
        for name in cams:
            child_dir = os.path.join(images_root, name)
            child_images = _direct_image_files(child_dir)
            if child_images:
                frame_dirs.append((name, len(child_images)))

        if (
            len(cams) > 10
            and len(frame_dirs) == len(cams)
            and max(count for _, count in frame_dirs) == 1
        ):
            return images_root

        return os.path.join(images_root, cams[0])

    def _camera_from_image_dir(self, image_dir: str) -> Optional[str]:
        parent = os.path.basename(os.path.dirname(image_dir))
        if parent != "images":
            return None
        return os.path.basename(image_dir)

    def _collect_filelist(self, dir_path: str) -> List[str]:
        filelist = _direct_image_files(dir_path)
        if not filelist:
            nested = []
            child_dirs = sorted(
                d for d in glob.glob(os.path.join(dir_path, "*")) if os.path.isdir(d)
            )
            for child_dir in child_dirs:
                child_images = _direct_image_files(child_dir)
                if child_images:
                    nested.append(child_images[0])
            filelist = nested
        if self.stride > 1:
            filelist = filelist[:: self.stride]
        if self.max_frames is not None:
            filelist = filelist[: self.max_frames]
        return filelist

    def _load_images(self, filelist: List[str]) -> torch.Tensor:
        views = load_images_for_eval(
            filelist,
            size=self.size,
            verbose=False,
            crop=self.crop,
            patch_size=self.patch_size,
        )
        imgs = torch.cat([view["img"] for view in views], dim=0)
        images = imgs.unsqueeze(0)
        images = (images + 1.0) / 2.0
        return images

    def iter_sequence_infos(self) -> Iterator[LongStreamSequenceInfo]:
        fmt = self._infer_format()
        seqs = self._resolve_seq_list()
        for seq_entry in seqs:
            if fmt == "generalizable":
                seq, scene_root = self._resolve_scene_root(seq_entry)
                dir_path = self._resolve_image_dir_generalizable(scene_root)
                if dir_path is None or not os.path.isdir(dir_path):
                    continue
                camera = self._camera_from_image_dir(dir_path)
            else:
                seq = seq_entry
                scene_root = os.path.join(self.img_path, seq)
                dir_path = self.dir_path_func(self.img_path, seq)
                if not os.path.isdir(dir_path):
                    continue
                camera = None

            filelist = self._collect_filelist(dir_path)
            if not filelist:
                continue
            yield LongStreamSequenceInfo(
                name=seq,
                scene_root=scene_root,
                image_dir=dir_path,
                image_paths=filelist,
                camera=camera,
            )

    def _filter_image_paths(
        self, image_paths: List[str]
    ) -> Tuple[List[str], Optional[List[int]]]:
        """
        当帧质量过滤开启时，逐帧评估 is_high_quality 并返回过滤后的路径列表，
        同时返回被保留帧的原始索引列表（用于对齐 gt_poses）。
        未开启过滤时直接返回原列表和 None。
        """
        if not self.filter_enabled:
            return image_paths, None

        kept_paths: List[str] = []
        kept_indices: List[int] = []
        prev_img: Optional[np.ndarray] = None

        for idx, path in enumerate(image_paths):
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if is_high_quality(
                img,
                prev_img,
                blur_threshold=self.filter_blur_threshold,
                motion_threshold=self.filter_motion_threshold,
            ):
                kept_paths.append(path)
                kept_indices.append(idx)
                prev_img = img

        n_orig = len(image_paths)
        n_kept = len(kept_paths)
        if n_kept < n_orig:
            print(
                f"[longstream][filter] 质量过滤：保留 {n_kept}/{n_orig} 帧",
                flush=True,
            )
        return kept_paths, kept_indices

    def _resolve_gt_poses(
        self,
        scene_root: Optional[str],
        camera: Optional[str],
        kept_indices: Optional[List[int]],
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        统一的 GT 位姿加载入口。

        优先从 cameras/<cam>/extri.yml 读取，没有时退回 gt_poses.npy。
        若 kept_indices 不为 None，则按索引取对应帧的位姿。

        Returns:
            (poses, source_tag)
        """
        if scene_root is None:
            return None, "none"

        npy_name = self.gt_poses_file if self.gt_poses_file else "gt_poses.npy"
        poses, source_tag = resolve_gt_poses(
            scene_root,
            camera=camera,
            gt_source=self.gt_source,
            npy_name=npy_name,
        )
        if poses is None:
            return None, "none"

        if kept_indices is not None:
            poses = subset_pose_array(poses, kept_indices)
            if len(poses) == 0:
                return None, "none"

        print(
            f"[longstream][gt_poses] 已加载 {len(poses)} 帧位姿 (source={source_tag})",
            flush=True,
        )
        return poses, source_tag

    def __iter__(self) -> Iterator[LongStreamSequence]:
        for info in self.iter_sequence_infos():
            print(
                f"[longstream] loading sequence {info.name}: {len(info.image_paths)} frames",
                flush=True,
            )
            # 帧质量过滤（可选）—— 筛帧和 GT 子采样一次完成
            filtered_paths, kept_indices = self._filter_image_paths(info.image_paths)
            # GT 位姿加载（统一入口，带筛帧子采样）
            gt_poses, gt_source = self._resolve_gt_poses(
                info.scene_root, info.camera, kept_indices
            )

            images = self._load_images(filtered_paths)
            print(
                f"[longstream] loaded sequence {info.name}: {tuple(images.shape)}",
                flush=True,
            )

            # 从 GT w2c 位姿提取虚拟 GPS 相机中心坐标 [S, 3]。
            # 数学推导：w2c 满足 p_cam = R @ p_world + t，
            # 故相机中心 = -R^T @ t（严禁直接取 [:3, 3]）。
            gps_xyz: Optional[np.ndarray] = None
            if gt_poses is not None and len(gt_poses) > 0:
                R = gt_poses[:, :3, :3].astype(np.float64)  # [S, 3, 3]
                t = gt_poses[:, :3, 3].astype(np.float64)   # [S, 3]
                # center_i = -(R_i^T @ t_i)，向量化实现
                gps_xyz = -np.einsum('nji,nj->ni', R, t).astype(np.float32)
                print(
                    f"[longstream][gps] 已从 GT 位姿提取 {len(gps_xyz)} 帧相机中心",
                    flush=True,
                )

            yield LongStreamSequence(
                info.name,
                images,
                filtered_paths,
                scene_root=info.scene_root,
                image_dir=info.image_dir,
                camera=info.camera,
                gt_poses=gt_poses,
                original_frame_indices=kept_indices,
                gt_poses_source=gt_source,
                gps_xyz=gps_xyz,
            )
