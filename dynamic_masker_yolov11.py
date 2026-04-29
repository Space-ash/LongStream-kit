import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# YOLO 和跟踪器依赖可选：当 use_yolo=False 时不需要安装这两个库
try:
    from sort import Sort
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

from longstream.preprocess import (
    DepthAnythingV2Runner,
    default_preprocess_config_path,
    load_preprocess_config,
    prepare_images_to_generalizable,
    prepare_video_to_generalizable,
)
from longstream.preprocess.config import deep_update


DEFAULT_DYNAMIC_CLASS_IDS = (0, 1, 2, 3, 5, 7)


def _natural_sort_key(path: str):
    stem = Path(path).stem
    parts = []
    token = ""
    is_digit = None
    for ch in stem:
        curr_digit = ch.isdigit()
        if is_digit is None or curr_digit == is_digit:
            token += ch
        else:
            parts.append((0, int(token)) if is_digit else (1, token.lower()))
            token = ch
        is_digit = curr_digit
    if token:
        parts.append((0, int(token)) if is_digit else (1, token.lower()))
    return parts, Path(path).name.lower()


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _remove_path(path: Path):
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray):
    ext = path.suffix.lower()
    if ext == ".png":
        ok, buf = cv2.imencode(ext, image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif ext in {".jpg", ".jpeg"}:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        raise ValueError(f"unsupported image extension: {path.suffix}")
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    _ensure_dir(path.parent)
    buf.tofile(str(path))


def _resize_mask(mask: np.ndarray, image_shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape_hw
    if mask.shape[:2] == (h, w):
        return mask.astype(np.uint8, copy=False)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def _resize_image(image: np.ndarray, image_shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape_hw
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def _ensure_even_size(width: int, height: int) -> Tuple[int, int]:
    width = max(2, int(width))
    height = max(2, int(height))
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1
    return max(2, width), max(2, height)


def _prepare_preview_frame(frame: np.ndarray, max_side: int = 4096) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    target_w, target_h = _ensure_even_size(target_w, target_h)
    if (target_h, target_w) != (h, w):
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    elif w % 2 != 0 or h % 2 != 0:
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return frame


def _copy_scene_metadata(src_scene_root: Path, dst_scene_root: Path):
    for subdir in ["cameras", "depths", "masks", "vis_depths"]:
        src = src_scene_root / subdir
        dst = dst_scene_root / subdir
        if src.exists() and src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _copy_prepared_scene(
    src_root: Path, dst_root: Path, scene_name: str, overwrite: bool = True
):
    src_scene_root = src_root / scene_name
    dst_scene_root = dst_root / scene_name
    if not src_scene_root.exists():
        raise FileNotFoundError(src_scene_root)
    if dst_scene_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"target scene already exists: {dst_scene_root}. "
                "Enable overwrite to replace it."
            )
        shutil.rmtree(dst_scene_root)
    shutil.copytree(src_scene_root, dst_scene_root)
    _ensure_dir(dst_root)
    with open(dst_root / "data_roots.txt", "w", encoding="utf-8") as f:
        f.write(f"{scene_name}\n")


@dataclass
class InstanceSegmentation:
    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray
    area: int
    bbox_xyxy: Tuple[float, float, float, float]


@dataclass
class FrameSegmentation:
    instances: List[InstanceSegmentation]
    semantic_mask: np.ndarray


@dataclass
class MotionDecision:
    instance_index: int
    class_id: int
    class_name: str
    confidence: float
    area: int
    bbox_xyxy: Tuple[float, float, float, float]
    sampled_points: int
    median_error: float
    mean_error: float
    dynamic_ratio: float
    median_depth_error: float
    mean_depth_error: float
    depth_dynamic_ratio: float
    depth_error_threshold_value: float
    median_flow_magnitude: float
    is_dynamic: bool
    reason: str
    raw_is_dynamic: Optional[bool] = None
    track_id: Optional[int] = None
    temporal_vote_dynamic_count: int = 0
    temporal_vote_total_count: int = 0
    temporal_dynamic_fraction: float = 0.0
    temporal_window_size: int = 1
    temporal_refined: bool = False
    flow_variance: float = 0.0
    ego_motion_factor: float = 1.0
    adaptive_pixel_threshold: float = 0.0
    adaptive_depth_threshold: float = 0.0
    scale_correction: float = 1.0


@dataclass
class AdaptiveThresholdDebug:
    bg_flow_magnitude: float
    ego_motion_factor: float
    base_pixel_threshold: float
    adaptive_pixel_threshold: float
    base_depth_error_threshold: float
    depth_error_min: float
    scale_correction: float


@dataclass
class GeometryDebug:
    model: str
    valid: bool
    matched_points: int
    inlier_points: int
    inlier_ratio: float
    median_reprojection_error: float = 0.0
    mean_reprojection_error: float = 0.0
    scale_correction: float = 1.0
    bg_flow_magnitude: float = 0.0


@dataclass
class FramePairResult:
    original_image: np.ndarray
    instances: List[InstanceSegmentation]
    masked_image: np.ndarray
    dynamic_mask: np.ndarray
    semantic_mask: np.ndarray
    static_mask: np.ndarray
    overlay_image: np.ndarray
    comparison_image: np.ndarray
    object_decisions: List[MotionDecision] = field(default_factory=list)
    geometry_model: str = "pnp_depth"
    geometry_matrix: Optional[np.ndarray] = None
    geometry_debug: Optional[GeometryDebug] = None
    adaptive_threshold_debug: Optional[AdaptiveThresholdDebug] = None


class SemanticFlowDynamicMasker:
    def __init__(
        self,
        model_path: str = "./model/yolo11n-seg.pt",
        dynamic_class_ids: Sequence[int] = DEFAULT_DYNAMIC_CLASS_IDS,
        depth_repo_path: str = "Depth-Anything-V2-main",
        depth_checkpoint_path: str = "model/depth_anything_v2_vits.pth",
        depth_encoder: str = "vits",
        depth_input_size: int = 518,
        depth_device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.5,
        max_corners: int = 500,
        lk_win_size: Tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
        pixel_error_threshold: float = 4.0,
        dynamic_ratio_threshold: float = 0.6,
        min_motion_magnitude: float = 2.0,
        min_mask_area: int = 80,
        max_mask_points: int = 60,
        pnp_reprojection_error: float = 4.0,
        pnp_iterations_count: int = 100,
        pnp_confidence: float = 0.999,
        depth_error_threshold: float = 0.1,
        depth_error_min: float = 0.05,
        virtual_focal_scale: float = 1.2,
        depth_eps: float = 1e-3,
        depth_floor_percentile: float = 1.0,
        mask_core_kernel_size: int = 0,
        mask_core_iterations: int = 1,
        dilation_kernel_size: int = 5,
        dilation_iterations: int = 2,
        temporal_consistency_enabled: bool = True,
        temporal_window_size: int = 7,
        temporal_vote_ratio_threshold: float = 0.6,
        temporal_min_track_length: int = 3,
        temporal_match_iou_threshold: float = 0.2,
        temporal_max_center_distance_ratio: float = 0.75,
        sort_max_age: int = 15,
        sort_min_hits: int = 3,
        sort_iou_threshold: float = 0.3,
        non_rigid_flow_variance_threshold: float = 5.0,
        fallback_keep_semantic_if_geometry_fails: bool = True,
        random_seed: int = 0,
        use_yolo: bool = False,  # 屏蔽 YOLO 动态物体识别，仅保留几何一致性判断
    ):
        # --- YOLO 及追踪器（可选） ---
        self.use_yolo = bool(use_yolo)
        if self.use_yolo:
            if not _YOLO_AVAILABLE:
                raise ImportError(
                    "use_yolo=True 但未安装 ultralytics/sort，"
                    "请安装依赖或设置 use_yolo=False"
                )
            self.yolo_model = YOLO(model_path)
            self.tracker = Sort(
                max_age=int(sort_max_age),
                min_hits=int(sort_min_hits),
                iou_threshold=float(sort_iou_threshold),
            )
            self.class_names = self.yolo_model.names
        else:
            self.yolo_model = None
            self.tracker = None
            self.class_names = {}
        self.depth_runner = DepthAnythingV2Runner(
            repo_path=depth_repo_path,
            checkpoint_path=depth_checkpoint_path,
            encoder=depth_encoder,
            input_size=depth_input_size,
            device=depth_device,
            grayscale=True,
            save_raw_npy=False,
            save_visualization=False,
            save_comparison=False,
        )
        self.dynamic_class_ids = tuple(int(x) for x in dynamic_class_ids)
        self.geometry_model = "pnp_depth"
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_corners = int(max_corners)
        self.lk_win_size = tuple(int(x) for x in lk_win_size)
        self.lk_max_level = int(lk_max_level)
        self.pixel_error_threshold = float(pixel_error_threshold)
        self.dynamic_ratio_threshold = float(dynamic_ratio_threshold)
        self.min_motion_magnitude = float(min_motion_magnitude)
        self.min_mask_area = int(min_mask_area)
        self.max_mask_points = int(max_mask_points)
        self.pnp_reprojection_error = float(pnp_reprojection_error)
        self.pnp_iterations_count = int(pnp_iterations_count)
        self.pnp_confidence = float(pnp_confidence)
        self.depth_error_threshold = float(depth_error_threshold)
        self.depth_error_min = float(depth_error_min)
        self.virtual_focal_scale = float(virtual_focal_scale)
        self.depth_eps = float(depth_eps)
        self.depth_floor_percentile = float(depth_floor_percentile)
        self.mask_core_kernel_size = int(mask_core_kernel_size)
        self.mask_core_iterations = int(mask_core_iterations)
        self.dilation_kernel_size = int(dilation_kernel_size)
        self.dilation_iterations = int(dilation_iterations)
        self.temporal_consistency_enabled = bool(temporal_consistency_enabled)
        self.temporal_window_size = max(1, int(temporal_window_size))
        self.temporal_vote_ratio_threshold = float(temporal_vote_ratio_threshold)
        self.temporal_min_track_length = max(1, int(temporal_min_track_length))
        self.temporal_match_iou_threshold = float(temporal_match_iou_threshold)
        self.temporal_max_center_distance_ratio = float(
            temporal_max_center_distance_ratio
        )
        self.sort_max_age = int(sort_max_age)
        self.sort_min_hits = int(sort_min_hits)
        self.sort_iou_threshold = float(sort_iou_threshold)
        self.non_rigid_flow_variance_threshold = float(
            non_rigid_flow_variance_threshold
        )
        self.fallback_keep_semantic_if_geometry_fails = bool(
            fallback_keep_semantic_if_geometry_fails
        )
        self.rng = np.random.default_rng(random_seed)
        if self.use_yolo:
            self.class_names = self.yolo_model.names

    @classmethod
    def from_preprocess_config(cls, preprocess_cfg: Dict):
        dynamic_cfg = dict(preprocess_cfg.get("dynamic_filter", {}))
        depth_cfg = dict(preprocess_cfg.get("depth_anything_v2", {}))
        model_path = dynamic_cfg.pop("model_path", "./model/yolo11n-seg.pt")
        dynamic_class_ids = dynamic_cfg.pop(
            "dynamic_class_ids", DEFAULT_DYNAMIC_CLASS_IDS
        )
        if "repo_path" in depth_cfg and "depth_repo_path" not in dynamic_cfg:
            dynamic_cfg["depth_repo_path"] = depth_cfg.get("repo_path")
        if "checkpoint_path" in depth_cfg and "depth_checkpoint_path" not in dynamic_cfg:
            dynamic_cfg["depth_checkpoint_path"] = depth_cfg.get("checkpoint_path")
        if "encoder" in depth_cfg and "depth_encoder" not in dynamic_cfg:
            dynamic_cfg["depth_encoder"] = depth_cfg.get("encoder")
        if "input_size" in depth_cfg and "depth_input_size" not in dynamic_cfg:
            dynamic_cfg["depth_input_size"] = depth_cfg.get("input_size")
        if "device" in depth_cfg and "depth_device" not in dynamic_cfg:
            dynamic_cfg["depth_device"] = depth_cfg.get("device")
        dynamic_cfg.pop("enabled", None)
        dynamic_cfg.pop("backend", None)
        dynamic_cfg.pop("save_preview_video", None)
        dynamic_cfg.pop("preview_fps", None)
        dynamic_cfg.pop("geometry_model", None)
        if (
            "fallback_keep_semantic_if_geometry_fails" not in dynamic_cfg
            and "fallback_keep_semantic_if_homography_fails" in dynamic_cfg
        ):
            dynamic_cfg["fallback_keep_semantic_if_geometry_fails"] = dynamic_cfg.pop(
                "fallback_keep_semantic_if_homography_fails"
            )
        return cls(
            model_path=model_path,
            dynamic_class_ids=dynamic_class_ids,
            **dynamic_cfg,
        )

    def _segment_frames(self, frames: Sequence[np.ndarray]) -> List[FrameSegmentation]:
        """YOLO 实例分割。use_yolo=False 时返回空分割结果（仅几何一致性校验仍正常运行）。"""
        if not self.use_yolo:
            return [
                FrameSegmentation(
                    instances=[],
                    semantic_mask=np.zeros(f.shape[:2], dtype=np.uint8),
                )
                for f in frames
            ]

        results = self.yolo_model(
            list(frames),
            classes=list(self.dynamic_class_ids),
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        segmentations: List[FrameSegmentation] = []
        for frame, result in zip(frames, results):
            h, w = frame.shape[:2]
            instances: List[InstanceSegmentation] = []
            semantic_mask = np.zeros((h, w), dtype=np.uint8)

            boxes = result.boxes
            masks = result.masks
            if boxes is not None and masks is not None and len(boxes) > 0:
                mask_data = masks.data.detach().cpu().numpy()
                cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
                confs = boxes.conf.detach().cpu().numpy().astype(float)
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)

                for idx in range(len(cls_ids)):
                    mask = (mask_data[idx] > 0.5).astype(np.uint8)
                    mask = _resize_mask(mask, (h, w))
                    area = int(mask.sum())
                    if area <= 0:
                        continue
                    class_id = int(cls_ids[idx])
                    class_name = str(self.class_names.get(class_id, class_id))
                    instances.append(
                        InstanceSegmentation(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(confs[idx]),
                            mask=mask,
                            area=area,
                            bbox_xyxy=tuple(float(v) for v in xyxy[idx]),
                        )
                    )
                    semantic_mask = np.maximum(semantic_mask, mask)

            segmentations.append(
                FrameSegmentation(instances=instances, semantic_mask=semantic_mask * 255)
            )
        return segmentations

    def _reset_tracker(self) -> None:
        if not self.use_yolo or self.tracker is None:
            return
        self.tracker = Sort(
            max_age=self.sort_max_age,
            min_hits=self.sort_min_hits,
            iou_threshold=self.sort_iou_threshold,
        )

    def _depth_to_pseudo_metric(self, depth: np.ndarray) -> np.ndarray:
        depth = depth.astype(np.float32, copy=False)
        positive = depth[np.isfinite(depth) & (depth > 0)]
        if positive.size == 0:
            return np.ones_like(depth, dtype=np.float32)
        floor = float(np.percentile(positive, self.depth_floor_percentile))
        floor = max(floor, self.depth_eps)
        pseudo = 1.0 / np.maximum(depth, floor)
        scale = float(np.median(pseudo[np.isfinite(pseudo) & (pseudo > 0)]))
        if scale > 1e-6:
            pseudo = pseudo / scale
        return pseudo.astype(np.float32, copy=False)

    def _virtual_camera_matrix(self, image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        focal = self.virtual_focal_scale * float(max(h, w))
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        return np.array(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _sample_depth(self, depth: np.ndarray, points: np.ndarray) -> np.ndarray:
        h, w = depth.shape[:2]
        x = np.clip(points[:, 0], 0.0, w - 1.0)
        y = np.clip(points[:, 1], 0.0, h - 1.0)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        wx = x - x0
        wy = y - y0
        top = depth[y0, x0] * (1.0 - wx) + depth[y0, x1] * wx
        bottom = depth[y1, x0] * (1.0 - wx) + depth[y1, x1] * wx
        return (top * (1.0 - wy) + bottom * wy).astype(np.float32, copy=False)

    def _unproject_points(
        self, points: np.ndarray, depth: np.ndarray, camera_matrix: np.ndarray
    ) -> np.ndarray:
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        z = depth.astype(np.float32, copy=False)
        x = ((points[:, 0] - cx) / fx) * z
        y = ((points[:, 1] - cy) / fy) * z
        return np.stack([x, y, z], axis=1).astype(np.float32, copy=False)

    def _estimate_background_geometry(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_depth: np.ndarray,
        curr_depth: np.ndarray,
        prev_semantic_mask: np.ndarray,
        curr_semantic_mask: np.ndarray,
    ) -> Tuple[
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]], GeometryDebug
    ]:
        prev_bg = cv2.bitwise_not(prev_semantic_mask)
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=8,
            mask=prev_bg,
        )
        if prev_pts is None or len(prev_pts) < 6:
            return None, GeometryDebug(self.geometry_model, False, 0, 0, 0.0)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if curr_pts is None or status is None:
            return None, GeometryDebug(self.geometry_model, False, 0, 0, 0.0)

        valid = status.reshape(-1) == 1
        prev_xy = prev_pts.reshape(-1, 2)[valid]
        curr_xy = curr_pts.reshape(-1, 2)[valid]
        if len(prev_xy) < 6:
            return None, GeometryDebug(self.geometry_model, False, int(len(prev_xy)), 0, 0.0)

        h, w = curr_gray.shape[:2]
        keep = (
            np.isfinite(prev_xy[:, 0])
            & np.isfinite(prev_xy[:, 1])
            & np.isfinite(curr_xy[:, 0])
            & np.isfinite(curr_xy[:, 1])
            & (curr_xy[:, 0] >= 0)
            & (curr_xy[:, 0] < w)
            & (curr_xy[:, 1] >= 0)
            & (curr_xy[:, 1] < h)
        )
        if curr_semantic_mask is not None:
            xi = np.clip(curr_xy[:, 0].astype(np.int32), 0, w - 1)
            yi = np.clip(curr_xy[:, 1].astype(np.int32), 0, h - 1)
            keep &= curr_semantic_mask[yi, xi] == 0
        prev_xy = prev_xy[keep]
        curr_xy = curr_xy[keep]
        if len(prev_xy) < 6:
            return None, GeometryDebug(self.geometry_model, False, int(len(prev_xy)), 0, 0.0)

        bg_displacements = np.linalg.norm(curr_xy - prev_xy, axis=1)
        bg_flow_magnitude = float(np.median(bg_displacements)) if len(bg_displacements) > 0 else 0.0

        depth_z = self._sample_depth(prev_depth, prev_xy)
        keep = np.isfinite(depth_z) & (depth_z > self.depth_eps)
        prev_xy = prev_xy[keep]
        curr_xy = curr_xy[keep]
        depth_z = depth_z[keep]
        if len(prev_xy) < 6:
            return None, GeometryDebug(self.geometry_model, False, int(len(prev_xy)), 0, 0.0)

        camera_matrix = self._virtual_camera_matrix(prev_gray.shape[:2])
        object_points = self._unproject_points(prev_xy, depth_z, camera_matrix)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            curr_xy,
            camera_matrix,
            None,
            iterationsCount=self.pnp_iterations_count,
            reprojectionError=self.pnp_reprojection_error,
            confidence=self.pnp_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or rvec is None or tvec is None or inliers is None:
            return None, GeometryDebug(self.geometry_model, False, int(len(prev_xy)), 0, 0.0)

        inlier_idx = inliers.reshape(-1)
        inlier_count = int(len(inlier_idx))
        if inlier_count < 6:
            return None, GeometryDebug(self.geometry_model, False, int(len(prev_xy)), inlier_count, 0.0)

        try:
            refined_ok, refined_rvec, refined_tvec = cv2.solvePnP(
                object_points[inlier_idx],
                curr_xy[inlier_idx],
                camera_matrix,
                None,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if refined_ok:
                rvec = refined_rvec
                tvec = refined_tvec
        except cv2.error:
            pass

        reproj, _ = cv2.projectPoints(object_points[inlier_idx], rvec, tvec, camera_matrix, None)
        reproj = reproj.reshape(-1, 2)
        reproj_errors = np.linalg.norm(reproj - curr_xy[inlier_idx], axis=1)
        bg_prev_z = depth_z[inlier_idx]
        bg_curr_z = self._sample_depth(curr_depth, curr_xy[inlier_idx])
        valid_scale = (
            np.isfinite(bg_prev_z)
            & np.isfinite(bg_curr_z)
            & (bg_prev_z > self.depth_eps)
            & (bg_curr_z > self.depth_eps)
        )
        if np.any(valid_scale):
            scale_correction = float(
                np.median(bg_curr_z[valid_scale] / bg_prev_z[valid_scale])
            )
            if not np.isfinite(scale_correction) or scale_correction <= self.depth_eps:
                scale_correction = 1.0
        else:
            scale_correction = 1.0
        matched_points = int(len(prev_xy))
        inlier_ratio = float(inlier_count / max(matched_points, 1))
        pose_matrix = np.eye(4, dtype=np.float32)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        pose_matrix[:3, :3] = rotation_matrix.astype(np.float32, copy=False)
        pose_matrix[:3, 3] = tvec.reshape(3).astype(np.float32, copy=False)
        return (
            (
                pose_matrix,
                rvec.astype(np.float32, copy=False),
                camera_matrix,
                float(scale_correction),
                float(bg_flow_magnitude),
            ),
            GeometryDebug(
                self.geometry_model,
                True,
                matched_points,
                inlier_count,
                inlier_ratio,
                median_reprojection_error=float(np.median(reproj_errors)),
                mean_reprojection_error=float(np.mean(reproj_errors)),
                scale_correction=float(scale_correction),
                bg_flow_magnitude=float(bg_flow_magnitude),
            ),
        )

    def _sample_mask_points(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sample_mask = mask.astype(np.uint8, copy=True)
        if self.mask_core_kernel_size > 1 and self.mask_core_iterations > 0:
            kernel = np.ones(
                (self.mask_core_kernel_size, self.mask_core_kernel_size), dtype=np.uint8
            )
            eroded = cv2.erode(sample_mask, kernel, iterations=self.mask_core_iterations)
            if int(eroded.sum()) > 0:
                sample_mask = eroded
        y_coords, x_coords = np.where(sample_mask > 0)
        if len(x_coords) == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        if len(x_coords) > self.max_mask_points:
            indices = self.rng.choice(len(x_coords), size=self.max_mask_points, replace=False)
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        return y_coords.astype(np.int32), x_coords.astype(np.int32)

    def _compute_ego_motion_factor(self, bg_flow_magnitude: float) -> float:
        return max(1.0, 1.0 + 0.01 * float(bg_flow_magnitude))

    def _classify_instance_motion(
        self,
        instance_index: int,
        instance: InstanceSegmentation,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_depth: np.ndarray,
        curr_depth: np.ndarray,
        pose_state: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]],
        track_id: Optional[int] = None,
    ) -> MotionDecision:
        if instance.area < self.min_mask_area:
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=0,
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_depth_error=0.0,
                mean_depth_error=0.0,
                depth_dynamic_ratio=0.0,
                depth_error_threshold_value=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=False,
                reason="mask_too_small",
                raw_is_dynamic=False,
                track_id=track_id,
            )

        if pose_state is None:
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=0,
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_depth_error=0.0,
                mean_depth_error=0.0,
                depth_dynamic_ratio=0.0,
                depth_error_threshold_value=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=not self.fallback_keep_semantic_if_geometry_fails,
                reason="pnp_depth_unavailable",
                raw_is_dynamic=not self.fallback_keep_semantic_if_geometry_fails,
                track_id=track_id,
            )

        y_coords, x_coords = self._sample_mask_points(instance.mask)
        if len(x_coords) < 6:
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=int(len(x_coords)),
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_depth_error=0.0,
                mean_depth_error=0.0,
                depth_dynamic_ratio=0.0,
                depth_error_threshold_value=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=False,
                reason="not_enough_points",
                raw_is_dynamic=False,
                track_id=track_id,
            )

        curr_pts = np.stack(
            [x_coords.astype(np.float32), y_coords.astype(np.float32)], axis=1
        )
        prev_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray,
            prev_gray,
            curr_pts.reshape(-1, 1, 2),
            None,
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if prev_pts is None or status is None:
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=0,
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_depth_error=0.0,
                mean_depth_error=0.0,
                depth_dynamic_ratio=0.0,
                depth_error_threshold_value=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=False,
                reason="lk_tracking_failed",
                raw_is_dynamic=False,
                track_id=track_id,
            )

        valid = status.reshape(-1) == 1
        curr_pts = curr_pts[valid]
        prev_pts = prev_pts.reshape(-1, 2)[valid]
        flow_magnitude = np.linalg.norm(curr_pts - prev_pts, axis=1)
        h, w = instance.mask.shape[:2]
        valid = (
            np.isfinite(prev_pts[:, 0])
            & np.isfinite(prev_pts[:, 1])
            & np.isfinite(curr_pts[:, 0])
            & np.isfinite(curr_pts[:, 1])
            & (prev_pts[:, 0] >= 0.0)
            & (prev_pts[:, 0] < w)
            & (prev_pts[:, 1] >= 0.0)
            & (prev_pts[:, 1] < h)
        )
        if not np.any(valid):
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=0,
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_depth_error=0.0,
                mean_depth_error=0.0,
                depth_dynamic_ratio=0.0,
                depth_error_threshold_value=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=False,
                reason="no_valid_correspondence_points",
                raw_is_dynamic=False,
                track_id=track_id,
            )
        curr_pts = curr_pts[valid]
        prev_pts = prev_pts[valid]
        flow_magnitude = flow_magnitude[valid]

        pose_matrix, rvec, camera_matrix, scale_correction, bg_flow_magnitude = pose_state
        depth_z = self._sample_depth(prev_depth, prev_pts)
        valid = np.isfinite(depth_z) & (depth_z > self.depth_eps)
        if not np.any(valid):
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=0,
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_depth_error=0.0,
                mean_depth_error=0.0,
                depth_dynamic_ratio=0.0,
                depth_error_threshold_value=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=False,
                reason="invalid_depth_samples",
                raw_is_dynamic=False,
                track_id=track_id,
            )
        curr_pts = curr_pts[valid]
        prev_pts = prev_pts[valid]
        flow_magnitude = flow_magnitude[valid]
        depth_z = depth_z[valid]

        object_points = self._unproject_points(prev_pts, depth_z, camera_matrix)
        tvec = pose_matrix[:3, 3].reshape(3, 1)
        rotation_matrix = pose_matrix[:3, :3]
        pred_pts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
        pred_pts = pred_pts.reshape(-1, 2)
        errors_2d = np.linalg.norm(curr_pts - pred_pts, axis=1)
        pred_3d_pts = (rotation_matrix @ object_points.T).T + tvec.reshape(1, 3)
        pred_3d_z = pred_3d_pts[:, 2]
        curr_3d_z = self._sample_depth(curr_depth, curr_pts)
        if scale_correction > self.depth_eps:
            curr_3d_z = curr_3d_z / float(scale_correction)
        valid = np.isfinite(curr_3d_z) & (curr_3d_z > self.depth_eps) & np.isfinite(pred_3d_z)
        if np.any(valid):
            errors_2d = errors_2d[valid]
            flow_magnitude = flow_magnitude[valid]
            curr_3d_z = curr_3d_z[valid]
            pred_3d_z = pred_3d_z[valid]
            errors_3d_z = np.abs(curr_3d_z - pred_3d_z)
        else:
            errors_3d_z = np.zeros(0, dtype=np.float32)

        if errors_2d.size == 0:
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=0,
                median_error=0.0,
                mean_error=0.0,
                dynamic_ratio=0.0,
                median_depth_error=0.0,
                mean_depth_error=0.0,
                depth_dynamic_ratio=0.0,
                depth_error_threshold_value=0.0,
                median_flow_magnitude=0.0,
                is_dynamic=False,
                reason="no_valid_depth_consistency_points",
                raw_is_dynamic=False,
                track_id=track_id,
            )

        median_error = float(np.median(errors_2d))
        mean_error = float(np.mean(errors_2d))
        dynamic_ratio = float(np.mean(errors_2d > self.pixel_error_threshold))
        if errors_3d_z.size > 0:
            depth_error_threshold_value = max(
                self.depth_error_min,
                self.depth_error_threshold * float(np.median(np.abs(pred_3d_z))),
            )
            median_depth_error = float(np.median(errors_3d_z))
            mean_depth_error = float(np.mean(errors_3d_z))
            depth_dynamic_ratio = float(
                np.mean(errors_3d_z > depth_error_threshold_value)
            )
        else:
            depth_error_threshold_value = max(self.depth_error_min, self.depth_error_threshold)
            median_depth_error = 0.0
            mean_depth_error = 0.0
            depth_dynamic_ratio = 0.0
        median_flow_magnitude = float(np.median(flow_magnitude))
        flow_vectors = curr_pts - prev_pts
        flow_variance = float(
            np.var(flow_vectors[:, 0]) + np.var(flow_vectors[:, 1])
        )
        ego_motion_factor = self._compute_ego_motion_factor(bg_flow_magnitude)
        adaptive_pixel_threshold = self.pixel_error_threshold * ego_motion_factor
        adaptive_depth_threshold = depth_error_threshold_value * ego_motion_factor

        if (
            instance.class_id in [0, 1, 3]
            and flow_variance > self.non_rigid_flow_variance_threshold
        ):
            return MotionDecision(
                instance_index=instance_index,
                class_id=instance.class_id,
                class_name=instance.class_name,
                confidence=instance.confidence,
                area=instance.area,
                bbox_xyxy=instance.bbox_xyxy,
                sampled_points=int(len(errors_2d)),
                median_error=median_error,
                mean_error=mean_error,
                dynamic_ratio=dynamic_ratio,
                median_depth_error=median_depth_error,
                mean_depth_error=mean_depth_error,
                depth_dynamic_ratio=depth_dynamic_ratio,
                depth_error_threshold_value=adaptive_depth_threshold,
                median_flow_magnitude=median_flow_magnitude,
                is_dynamic=True,
                reason="non_rigid_motion_variance",
                raw_is_dynamic=True,
                track_id=track_id,
                flow_variance=flow_variance,
                ego_motion_factor=ego_motion_factor,
                adaptive_pixel_threshold=adaptive_pixel_threshold,
                adaptive_depth_threshold=adaptive_depth_threshold,
                scale_correction=float(scale_correction),
            )

        error_2d_dynamic = (
            median_error > adaptive_pixel_threshold
            or dynamic_ratio > self.dynamic_ratio_threshold
        )
        error_3d_dynamic = (
            median_depth_error > adaptive_depth_threshold
            or depth_dynamic_ratio > self.dynamic_ratio_threshold
        )
        motion_significant = median_flow_magnitude > self.min_motion_magnitude
        is_dynamic = bool(error_3d_dynamic or (error_2d_dynamic and motion_significant))
        if error_3d_dynamic:
            reason = "depth_inconsistent_dynamic"
        elif is_dynamic:
            reason = "reprojection_dynamic"
        else:
            reason = "background_consistent"

        return MotionDecision(
            instance_index=instance_index,
            class_id=instance.class_id,
            class_name=instance.class_name,
            confidence=instance.confidence,
            area=instance.area,
            bbox_xyxy=instance.bbox_xyxy,
            sampled_points=int(len(errors_2d)),
            median_error=median_error,
            mean_error=mean_error,
            dynamic_ratio=dynamic_ratio,
            median_depth_error=median_depth_error,
            mean_depth_error=mean_depth_error,
            depth_dynamic_ratio=depth_dynamic_ratio,
            depth_error_threshold_value=adaptive_depth_threshold,
            median_flow_magnitude=median_flow_magnitude,
            is_dynamic=is_dynamic,
            reason=reason,
            raw_is_dynamic=is_dynamic,
            track_id=track_id,
            flow_variance=flow_variance,
            ego_motion_factor=ego_motion_factor,
            adaptive_pixel_threshold=adaptive_pixel_threshold,
            adaptive_depth_threshold=adaptive_depth_threshold,
            scale_correction=float(scale_correction),
        )

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.dilation_kernel_size <= 1 or self.dilation_iterations <= 0:
            return mask
        kernel = np.ones(
            (self.dilation_kernel_size, self.dilation_kernel_size), dtype=np.uint8
        )
        return cv2.dilate(mask, kernel, iterations=self.dilation_iterations)

    def _update_tracker_with_instances(
        self, instances: Sequence[InstanceSegmentation]
    ) -> Dict[int, int]:
        """use_yolo=False 时跳过这一步直接返回空字典。"""
        if not self.use_yolo or self.tracker is None:
            return {}

        if not instances:
            self.tracker.update(np.empty((0, 5), dtype=np.float32))
            return {}

        detections = np.array(
            [
                [
                    float(instance.bbox_xyxy[0]),
                    float(instance.bbox_xyxy[1]),
                    float(instance.bbox_xyxy[2]),
                    float(instance.bbox_xyxy[3]),
                    float(instance.confidence),
                ]
                for instance in instances
            ],
            dtype=np.float32,
        )
        tracks = self.tracker.update(detections)
        if tracks.size == 0:
            return {}

        track_map: Dict[int, int] = {}
        used_track_rows = set()
        for instance_idx, instance in enumerate(instances):
            best_row = None
            best_iou = -1.0
            for row_idx, track in enumerate(tracks):
                if row_idx in used_track_rows:
                    continue
                iou = self._bbox_iou(instance.bbox_xyxy, tuple(float(v) for v in track[:4]))
                if iou > best_iou:
                    best_iou = iou
                    best_row = row_idx
            if best_row is not None and best_iou >= self.sort_iou_threshold * 0.5:
                used_track_rows.add(best_row)
                track_map[instance_idx] = int(tracks[best_row, 4])
        return track_map

    def _bbox_iou(
        self,
        bbox_a: Tuple[float, float, float, float],
        bbox_b: Tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area
        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)

    def _bbox_center_and_diag(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, float]:
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        diag = float(np.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1)))
        return center, max(diag, 1.0)

    def _render_result_visuals(self, result: FramePairResult) -> None:
        dynamic_mask = np.zeros(result.original_image.shape[:2], dtype=np.uint8)
        static_mask = np.zeros(result.original_image.shape[:2], dtype=np.uint8)
        decision_by_index = {
            decision.instance_index: decision for decision in result.object_decisions
        }

        for idx, instance in enumerate(result.instances):
            decision = decision_by_index.get(idx)
            if decision is None:
                continue
            if decision.is_dynamic:
                dynamic_mask = np.maximum(dynamic_mask, instance.mask.astype(np.uint8) * 255)
            else:
                static_mask = np.maximum(static_mask, instance.mask.astype(np.uint8) * 255)

        dynamic_mask = self._dilate_mask(dynamic_mask)
        masked_image = result.original_image.copy()
        masked_image[dynamic_mask > 0] = 0
        overlay_image = self._build_overlay(result.original_image, dynamic_mask, static_mask)
        comparison_image = np.hstack([result.original_image, overlay_image, masked_image])

        result.dynamic_mask = dynamic_mask
        result.static_mask = static_mask
        result.masked_image = masked_image
        result.overlay_image = overlay_image
        result.comparison_image = comparison_image

    def _match_to_previous_track(
        self,
        prev_result: FramePairResult,
        curr_decision: MotionDecision,
        used_prev_indices: set,
    ) -> Optional[int]:
        curr_center, curr_diag = self._bbox_center_and_diag(curr_decision.bbox_xyxy)
        best_idx = None
        best_score = None

        for prev_decision in prev_result.object_decisions:
            prev_idx = prev_decision.instance_index
            if prev_idx in used_prev_indices:
                continue
            if prev_decision.class_id != curr_decision.class_id:
                continue

            iou = self._bbox_iou(prev_decision.bbox_xyxy, curr_decision.bbox_xyxy)
            prev_center, prev_diag = self._bbox_center_and_diag(prev_decision.bbox_xyxy)
            norm_dist = float(
                np.linalg.norm(curr_center - prev_center) / max(curr_diag, prev_diag, 1.0)
            )
            if (
                iou < self.temporal_match_iou_threshold
                and norm_dist > self.temporal_max_center_distance_ratio
            ):
                continue

            score = iou - 0.1 * norm_dist
            if best_score is None or score > best_score:
                best_score = score
                best_idx = prev_idx

        return best_idx

    def _assign_temporal_tracks(
        self, results: List[FramePairResult]
    ) -> Dict[int, List[Tuple[int, int]]]:
        tracks: Dict[int, List[Tuple[int, int]]] = {}
        for frame_idx, result in enumerate(results):
            for decision in result.object_decisions:
                if decision.track_id is None:
                    synthetic_id = -1 - (frame_idx * 10000 + decision.instance_index)
                    decision.track_id = synthetic_id
                tracks.setdefault(int(decision.track_id), []).append(
                    (frame_idx, decision.instance_index)
                )

        return tracks

    def _apply_temporal_consistency(self, results: List[FramePairResult]) -> None:
        for result in results:
            for decision in result.object_decisions:
                if decision.raw_is_dynamic is None:
                    decision.raw_is_dynamic = decision.is_dynamic

        if (
            not self.temporal_consistency_enabled
            or self.temporal_window_size <= 1
            or not results
        ):
            for result in results:
                self._render_result_visuals(result)
            return

        tracks = self._assign_temporal_tracks(results)
        radius = self.temporal_window_size // 2
        decision_lookup = {
            (frame_idx, decision.instance_index): decision
            for frame_idx, result in enumerate(results)
            for decision in result.object_decisions
        }

        for track in tracks.values():
            if not track:
                continue
            track = sorted(track, key=lambda item: item[0])
            for pos, key in enumerate(track):
                decision = decision_lookup[key]
                start = max(0, pos - radius)
                end = min(len(track), pos + radius + 1)
                window_keys = track[start:end]
                vote_total = len(window_keys)
                vote_dynamic = sum(
                    1
                    for item in window_keys
                    if bool(decision_lookup[item].raw_is_dynamic)
                )
                dynamic_fraction = float(vote_dynamic / max(vote_total, 1))
                refined_is_dynamic = bool(decision.raw_is_dynamic)
                window_depth_errors = [
                    decision_lookup[item].median_depth_error for item in window_keys
                ]
                window_depth_thresholds = [
                    decision_lookup[item].depth_error_threshold_value
                    for item in window_keys
                    if decision_lookup[item].depth_error_threshold_value > 0
                ]
                stable_depth_track = False
                if window_depth_errors and window_depth_thresholds:
                    stable_depth_track = (
                        float(np.median(window_depth_errors))
                        <= float(np.median(window_depth_thresholds))
                    )
                if vote_total >= self.temporal_min_track_length:
                    refined_is_dynamic = (
                        dynamic_fraction >= self.temporal_vote_ratio_threshold
                    )
                    if stable_depth_track:
                        refined_is_dynamic = False
                decision.temporal_vote_dynamic_count = vote_dynamic
                decision.temporal_vote_total_count = vote_total
                decision.temporal_dynamic_fraction = dynamic_fraction
                decision.temporal_window_size = self.temporal_window_size
                decision.temporal_refined = refined_is_dynamic != bool(
                    decision.raw_is_dynamic
                )
                if decision.temporal_refined:
                    decision.reason = (
                        f"temporal_vote_refined_to_"
                        f"{'dynamic' if refined_is_dynamic else 'static'}"
                    )
                decision.is_dynamic = refined_is_dynamic

        for result in results:
            self._render_result_visuals(result)

    def _build_overlay(
        self,
        image: np.ndarray,
        dynamic_mask: np.ndarray,
        static_mask: np.ndarray,
    ) -> np.ndarray:
        overlay = image.copy()
        dynamic_region = dynamic_mask > 0
        static_region = static_mask > 0
        overlay[static_region] = (
            overlay[static_region].astype(np.float32) * 0.5
            + np.array([0, 255, 0], dtype=np.float32) * 0.5
        ).astype(np.uint8)
        overlay[dynamic_region] = (
            overlay[dynamic_region].astype(np.float32) * 0.3
            + np.array([0, 0, 255], dtype=np.float32) * 0.7
        ).astype(np.uint8)
        return overlay

    def process_frame_pair(
        self, prev_img: np.ndarray, curr_img: np.ndarray
    ) -> FramePairResult:
        # Image datasets may contain mixed resolutions; LK optical flow requires
        # the two pyramid inputs to have identical sizes.
        prev_img = _resize_image(prev_img, curr_img.shape[:2])
        prev_seg, curr_seg = self._segment_frames([prev_img, curr_img])
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        prev_depth = self._depth_to_pseudo_metric(self.depth_runner.infer_image(prev_img))
        curr_depth = self._depth_to_pseudo_metric(self.depth_runner.infer_image(curr_img))

        pose_state, geometry_debug = self._estimate_background_geometry(
            prev_gray,
            curr_gray,
            prev_depth,
            curr_depth,
            prev_seg.semantic_mask,
            curr_seg.semantic_mask,
        )
        adaptive_threshold_debug = None
        if pose_state is not None:
            _, _, _, scale_correction, bg_flow_magnitude = pose_state
            ego_motion_factor = self._compute_ego_motion_factor(bg_flow_magnitude)
            adaptive_threshold_debug = AdaptiveThresholdDebug(
                bg_flow_magnitude=float(bg_flow_magnitude),
                ego_motion_factor=float(ego_motion_factor),
                base_pixel_threshold=float(self.pixel_error_threshold),
                adaptive_pixel_threshold=float(
                    self.pixel_error_threshold * ego_motion_factor
                ),
                base_depth_error_threshold=float(self.depth_error_threshold),
                depth_error_min=float(self.depth_error_min),
                scale_correction=float(scale_correction),
            )
        track_ids = self._update_tracker_with_instances(curr_seg.instances)

        object_decisions: List[MotionDecision] = []

        for idx, instance in enumerate(curr_seg.instances):
            decision = self._classify_instance_motion(
                idx,
                instance,
                prev_gray,
                curr_gray,
                prev_depth,
                curr_depth,
                pose_state,
                track_ids.get(idx),
            )
            object_decisions.append(decision)
        result = FramePairResult(
            original_image=curr_img,
            instances=curr_seg.instances,
            masked_image=np.zeros_like(curr_img),
            dynamic_mask=np.zeros(curr_img.shape[:2], dtype=np.uint8),
            semantic_mask=curr_seg.semantic_mask,
            static_mask=np.zeros(curr_img.shape[:2], dtype=np.uint8),
            overlay_image=np.zeros_like(curr_img),
            comparison_image=np.zeros((curr_img.shape[0], curr_img.shape[1] * 3, 3), dtype=np.uint8),
            object_decisions=object_decisions,
            geometry_model=self.geometry_model,
            geometry_matrix=pose_state[0] if pose_state is not None else None,
            geometry_debug=geometry_debug,
            adaptive_threshold_debug=adaptive_threshold_debug,
        )
        self._render_result_visuals(result)
        return result

    def _list_image_paths(self, image_dir: Path) -> List[Path]:
        paths = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        return sorted(paths, key=lambda p: _natural_sort_key(str(p)))

    def process_prepared_scene(
        self,
        prepared_root: str,
        masked_root: str,
        scene_name: str,
        camera_id: str = "00",
        save_preview_video: bool = True,
        preview_fps: float = 10.0,
    ) -> Dict:
        prepared_root_path = Path(prepared_root).expanduser().resolve()
        masked_root_path = Path(masked_root).expanduser().resolve()
        src_scene_root = prepared_root_path / scene_name
        src_image_dir = src_scene_root / "images" / camera_id
        if not src_image_dir.is_dir():
            raise FileNotFoundError(src_image_dir)

        final_scene_root = masked_root_path / scene_name
        in_place_output = prepared_root_path == masked_root_path
        dst_scene_root = (
            masked_root_path / f".{scene_name}__dynamic_mask_tmp"
            if in_place_output
            else final_scene_root
        )
        _remove_path(dst_scene_root)
        if not in_place_output:
            _remove_path(final_scene_root)
        dst_image_dir = dst_scene_root / "images" / camera_id
        debug_root = dst_scene_root / "dynamic_filter_debug" / camera_id
        final_debug_root = final_scene_root / "dynamic_filter_debug" / camera_id
        for path in [
            dst_image_dir,
            debug_root / "dynamic_masks",
            debug_root / "semantic_masks",
            debug_root / "overlays",
            debug_root / "comparisons",
            debug_root / "reports",
        ]:
            _ensure_dir(path)

        _copy_scene_metadata(src_scene_root, dst_scene_root)
        with open(masked_root_path / "data_roots.txt", "w", encoding="utf-8") as f:
            f.write(f"{scene_name}\n")

        image_paths = self._list_image_paths(src_image_dir)
        if not image_paths:
            raise RuntimeError(f"no images found in: {src_image_dir}")

        self._reset_tracker()
        first_image = _read_image(image_paths[0])
        first_seg = self._segment_frames([first_image])[0]
        self._update_tracker_with_instances(first_seg.instances)
        _write_image(dst_image_dir / image_paths[0].name, first_image)

        preview_writer = None
        preview_frame_size: Optional[Tuple[int, int]] = None
        preview_path = debug_root / "dynamic_filter_preview.mp4"
        final_preview_path = final_debug_root / "dynamic_filter_preview.mp4"
        summary_records = []

        def _write_preview(frame: np.ndarray):
            nonlocal preview_writer, preview_frame_size
            if not save_preview_video:
                return
            frame = _prepare_preview_frame(frame)
            if preview_writer is None:
                h, w = frame.shape[:2]
                preview_frame_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                preview_writer = cv2.VideoWriter(
                    str(preview_path), fourcc, float(preview_fps), (w, h)
                )
                if not preview_writer.isOpened():
                    raise RuntimeError(
                        "failed to initialize preview VideoWriter for "
                        f"{preview_path} with size {(w, h)}"
                    )
            elif preview_frame_size is not None and frame.shape[:2] != (
                preview_frame_size[1],
                preview_frame_size[0],
            ):
                frame = cv2.resize(
                    frame,
                    preview_frame_size,
                    interpolation=cv2.INTER_AREA,
                )
            preview_writer.write(frame)

        first_compare = np.hstack([first_image, first_image, first_image])
        _write_image(debug_root / "comparisons" / image_paths[0].name, first_compare)
        _write_preview(first_compare)
        summary_records.append(
            {
                "frame_name": image_paths[0].name,
                "is_reference_frame": True,
                "adaptive_threshold_debug": None,
                "object_decisions": [],
            }
        )

        prev_image = first_image
        frame_results: List[Tuple[Path, FramePairResult]] = []
        for image_path in image_paths[1:]:
            curr_image = _read_image(image_path)
            result = self.process_frame_pair(prev_image, curr_image)
            frame_results.append((image_path, result))
            prev_image = curr_image

        self._apply_temporal_consistency([result for _, result in frame_results])

        for image_path, result in frame_results:
            _write_image(dst_image_dir / image_path.name, result.masked_image)
            _write_image(
                debug_root / "dynamic_masks" / image_path.with_suffix(".png").name,
                result.dynamic_mask,
            )
            _write_image(
                debug_root / "semantic_masks" / image_path.with_suffix(".png").name,
                result.semantic_mask,
            )
            _write_image(debug_root / "overlays" / image_path.name, result.overlay_image)
            _write_image(
                debug_root / "comparisons" / image_path.name, result.comparison_image
            )
            _write_preview(result.comparison_image)

            report = {
                "frame_name": image_path.name,
                "is_reference_frame": False,
                "geometry_model": result.geometry_model,
                "geometry_debug": asdict(result.geometry_debug)
                if result.geometry_debug is not None
                else None,
                "adaptive_threshold_debug": asdict(result.adaptive_threshold_debug)
                if result.adaptive_threshold_debug is not None
                else None,
                "homography_debug": asdict(result.geometry_debug)
                if result.geometry_model == "homography"
                and result.geometry_debug is not None
                else None,
                "object_decisions": [asdict(item) for item in result.object_decisions],
            }
            with open(
                debug_root / "reports" / f"{image_path.stem}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(report, f, indent=2, ensure_ascii=True)
            summary_records.append(report)

        if preview_writer is not None:
            preview_writer.release()

        if in_place_output:
            _remove_path(final_scene_root)
            dst_scene_root.replace(final_scene_root)

        summary = {
            "prepared_root": str(prepared_root_path),
            "masked_root": str(masked_root_path),
            "scene_name": scene_name,
            "camera_id": camera_id,
            "num_frames": len(image_paths),
            "preview_video": str(final_preview_path) if save_preview_video else None,
            "geometry_model": self.geometry_model,
            "temporal_consistency_enabled": self.temporal_consistency_enabled,
            "temporal_window_size": self.temporal_window_size,
            "temporal_vote_ratio_threshold": self.temporal_vote_ratio_threshold,
            "base_pixel_threshold": self.pixel_error_threshold,
            "base_depth_error_threshold": self.depth_error_threshold,
            "depth_error_min": self.depth_error_min,
            "dynamic_classes": list(self.dynamic_class_ids),
            "reports": summary_records,
        }
        summary_path = (
            final_debug_root / "summary.json" if in_place_output else debug_root / "summary.json"
        )
        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        return summary

    def prepare_and_process_video(
        self,
        video_path: str,
        prepared_root: str,
        masked_root: str,
        scene_name: str = "runtime_scene",
        camera_id: str = "00",
        target_fps: float = 5.0,
        image_ext: str = "png",
        overwrite: bool = True,
    ) -> Dict:
        prepare_video_to_generalizable(
            video_path=video_path,
            prepared_root=prepared_root,
            scene_name=scene_name,
            camera_id=camera_id,
            target_fps=target_fps,
            image_ext=image_ext,
            overwrite=overwrite,
        )
        return self.process_prepared_scene(
            prepared_root=prepared_root,
            masked_root=masked_root,
            scene_name=scene_name,
            camera_id=camera_id,
        )

    def prepare_and_process_images(
        self,
        source_path: str,
        prepared_root: str,
        masked_root: str,
        scene_name: str = "runtime_scene",
        camera_id: str = "00",
        image_ext: str = "png",
        overwrite: bool = True,
        recursive: bool = False,
    ) -> Dict:
        prepare_images_to_generalizable(
            source_path=source_path,
            prepared_root=prepared_root,
            scene_name=scene_name,
            camera_id=camera_id,
            image_ext=image_ext,
            overwrite=overwrite,
            recursive=recursive,
        )
        return self.process_prepared_scene(
            prepared_root=prepared_root,
            masked_root=masked_root,
            scene_name=scene_name,
            camera_id=camera_id,
        )


DynamicMasker = SemanticFlowDynamicMasker


def run_preprocess_pipeline(preprocess_cfg: Dict) -> Dict:
    input_cfg = dict(preprocess_cfg.get("input", {}))
    io_cfg = dict(preprocess_cfg.get("io", {}))
    video_cfg = dict(preprocess_cfg.get("video", {}))
    dynamic_cfg = dict(preprocess_cfg.get("dynamic_filter", {}))

    input_mode = str(input_cfg.get("mode", "video")).lower()
    source_path = input_cfg.get("source_path", None)
    recursive_images = bool(input_cfg.get("recursive_images", False))

    scene_name = str(io_cfg.get("scene_name", "runtime_scene"))
    camera_id = str(io_cfg.get("camera_id", "00"))
    image_ext = str(io_cfg.get("image_ext", "png")).lower()
    overwrite = bool(io_cfg.get("overwrite", True))
    normalized_root = str(io_cfg.get("normalized_root", "prepared_inputs/preprocess/raw"))
    final_root = str(io_cfg.get("final_root", normalized_root))

    if input_mode in {"video", "images"} and not source_path:
        raise ValueError("input.source_path must be set for video/images preprocessing")

    if input_mode == "video":
        prepare_video_to_generalizable(
            video_path=source_path,
            prepared_root=normalized_root,
            scene_name=scene_name,
            camera_id=camera_id,
            target_fps=video_cfg.get("target_fps", 0.0),
            image_ext=image_ext,
            overwrite=overwrite,
        )
    elif input_mode == "images":
        prepare_images_to_generalizable(
            source_path=source_path,
            prepared_root=normalized_root,
            scene_name=scene_name,
            camera_id=camera_id,
            image_ext=image_ext,
            overwrite=overwrite,
            recursive=recursive_images,
        )
    elif input_mode != "prepared":
        raise ValueError(f"unsupported input.mode: {input_mode}")

    if bool(dynamic_cfg.get("enabled", False)):
        masker = SemanticFlowDynamicMasker.from_preprocess_config(preprocess_cfg)
        summary = masker.process_prepared_scene(
            prepared_root=normalized_root,
            masked_root=final_root,
            scene_name=scene_name,
            camera_id=camera_id,
            save_preview_video=bool(dynamic_cfg.get("save_preview_video", True)),
            preview_fps=float(dynamic_cfg.get("preview_fps", 10.0)),
        )
        summary["dynamic_filter_enabled"] = True
        summary["final_root"] = summary["masked_root"]
        return summary

    normalized_root_path = Path(normalized_root).expanduser().resolve()
    final_root_path = Path(final_root).expanduser().resolve()
    if normalized_root_path != final_root_path:
        _copy_prepared_scene(
            normalized_root_path,
            final_root_path,
            scene_name=scene_name,
            overwrite=overwrite,
        )

    summary = {
        "input_mode": input_mode,
        "dynamic_filter_enabled": False,
        "scene_name": scene_name,
        "camera_id": camera_id,
        "normalized_root": str(normalized_root_path),
        "final_root": str(final_root_path),
        "source_path": source_path,
    }
    summary_path = final_root_path / scene_name / "preprocess_summary.json"
    _ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_preprocess_config_path())
    parser.add_argument("--input-mode", choices=["video", "images", "prepared"], default=None)
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--normalized-root", default=None)
    parser.add_argument("--final-root", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--image-ext", choices=["png", "jpg", "jpeg"], default=None)
    parser.add_argument("--recursive-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable-dynamic-filter", action="store_true")
    parser.add_argument("--disable-dynamic-filter", action="store_true")
    args = parser.parse_args()

    preprocess_cfg = load_preprocess_config(args.config)
    overrides: Dict = {"input": {}, "io": {}, "video": {}, "dynamic_filter": {}}
    if args.input_mode is not None:
        overrides["input"]["mode"] = args.input_mode
    if args.source_path is not None:
        overrides["input"]["source_path"] = args.source_path
    if args.recursive_images:
        overrides["input"]["recursive_images"] = True
    if args.normalized_root is not None:
        overrides["io"]["normalized_root"] = args.normalized_root
    if args.final_root is not None:
        overrides["io"]["final_root"] = args.final_root
    if args.scene_name is not None:
        overrides["io"]["scene_name"] = args.scene_name
    if args.camera_id is not None:
        overrides["io"]["camera_id"] = args.camera_id
    if args.image_ext is not None:
        overrides["io"]["image_ext"] = args.image_ext
    if args.overwrite:
        overrides["io"]["overwrite"] = True
    if args.target_fps is not None:
        overrides["video"]["target_fps"] = args.target_fps
    if args.model_path is not None:
        overrides["dynamic_filter"]["model_path"] = args.model_path
    if args.enable_dynamic_filter:
        overrides["dynamic_filter"]["enabled"] = True
    if args.disable_dynamic_filter:
        overrides["dynamic_filter"]["enabled"] = False

    preprocess_cfg = deep_update(preprocess_cfg, overrides)
    summary = run_preprocess_pipeline(preprocess_cfg)

    final_root = summary.get("final_root", summary.get("masked_root"))
    print(
        "[dynamic-masker] done "
        f"scene={summary['scene_name']} final_root={final_root} "
        f"dynamic_filter_enabled={summary['dynamic_filter_enabled']}"
    )


if __name__ == "__main__":
    main()
