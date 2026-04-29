"""
GT 位姿统一读取 / 锚定 / 子采样模块。

把 preprocess_vkitti2.py、dataloader.py、infer.py 里分散的
GT 位姿读取 / 锚定到第一帧 / 筛帧后重新索引 逻辑收口到这里。

支持两种 GT 来源（优先级从高到低）：
  1. cameras/<cam>/extri.yml + intri.yml  （EasyVolcap 格式，主真值）
  2. gt_poses.npy                         （[N,4,4] w2c 缓存）
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ------------------------------------------------------------------ #
# 从 cameras/<cam>/extri.yml + intri.yml 读取 w2c
# ------------------------------------------------------------------ #

def load_w2c_from_camera_yml(
    scene_root: str,
    camera: Optional[str] = None,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    """
    从 EasyVolcap cameras/ 目录读取 w2c 外参和内参。

    查找顺序：
      1. <scene_root>/cameras/<camera>/extri.yml   （有 camera 时）
      2. <scene_root>/cameras/extri.yml            （无 camera / 退路）

    Returns:
        (extri_dict, intri_dict)   —— 均以帧名为 key，值为 4×4 / 3×3 ndarray。
        读取失败时返回 (None, None)。
    """
    extri_path: Optional[str] = None
    intri_path: Optional[str] = None

    if camera is not None:
        cam_dir = os.path.join(scene_root, "cameras", camera)
        candidate = os.path.join(cam_dir, "extri.yml")
        if os.path.isfile(candidate):
            extri_path = candidate
            intri_path = os.path.join(cam_dir, "intri.yml")

    if extri_path is None:
        candidate = os.path.join(scene_root, "cameras", "extri.yml")
        if os.path.isfile(candidate):
            extri_path = candidate
            intri_path = os.path.join(scene_root, "cameras", "intri.yml")

    if extri_path is None or not os.path.isfile(extri_path):
        return None, None

    extri_dict = _read_extri_yml(extri_path)
    intri_dict = _read_intri_yml(intri_path) if intri_path and os.path.isfile(intri_path) else {}
    if not extri_dict:
        return None, None
    return extri_dict, intri_dict


def _read_extri_yml(path: str) -> Dict[str, np.ndarray]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    if names_node.empty():
        fs.release()
        return {}
    names = [names_node.at(i).string() for i in range(names_node.size())]
    result: Dict[str, np.ndarray] = {}
    for name in names:
        rot = fs.getNode(f"Rot_{name}").mat()
        t = fs.getNode(f"T_{name}").mat()
        if rot is None or t is None:
            continue
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = np.asarray(rot, dtype=np.float64)
        mat[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        result[name] = mat
    fs.release()
    return result


def _read_intri_yml(path: str) -> Dict[str, np.ndarray]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    if names_node.empty():
        fs.release()
        return {}
    names = [names_node.at(i).string() for i in range(names_node.size())]
    result: Dict[str, np.ndarray] = {}
    for name in names:
        K = fs.getNode(f"K_{name}").mat()
        if K is None:
            continue
        result[name] = np.asarray(K, dtype=np.float64)
    fs.release()
    return result


# ------------------------------------------------------------------ #
# 从 gt_poses.npy 读取（缓存 / 退路）
# ------------------------------------------------------------------ #

def load_w2c_from_npy(
    scene_root: str,
    npy_name: str = "gt_poses.npy",
) -> Optional[np.ndarray]:
    """
    加载 <scene_root>/<npy_name> 作为 [N,4,4] w2c 位姿数组。
    文件不存在或加载失败时返回 None。
    """
    path = os.path.join(scene_root, npy_name)
    if not os.path.isfile(path):
        return None
    try:
        poses = np.load(path)
        return poses.astype(np.float32, copy=False)
    except Exception:
        return None


# ------------------------------------------------------------------ #
# 统一入口：自动选择 GT 来源
# ------------------------------------------------------------------ #

def resolve_gt_poses(
    scene_root: str,
    camera: Optional[str] = None,
    gt_source: str = "auto",
    npy_name: str = "gt_poses.npy",
) -> Tuple[Optional[np.ndarray], str]:
    """
    统一的 GT 位姿加载入口。

    Args:
        scene_root: 场景根目录。
        camera:     相机子目录名（如 ``"00"``）。
        gt_source:  ``"camera_yml"`` | ``"npy"`` | ``"auto"``。
        npy_name:   npy 缓存文件名。

    Returns:
        (poses, source_tag)
          - poses: [N,4,4] float32 w2c 数组，或 None。
          - source_tag: ``"camera_yml"`` | ``"npy"`` | ``"none"``。
    """
    if gt_source in ("camera_yml", "auto"):
        extri_dict, _ = load_w2c_from_camera_yml(scene_root, camera)
        if extri_dict:
            sorted_names = sorted(extri_dict.keys())
            poses = np.stack([extri_dict[n] for n in sorted_names], axis=0)
            # camera_yml 存储的是原始世界坐标系 w2c，必须锚定到第 0 帧，
            # 使其与 gt_poses.npy（已锚定）语义一致，否则下游尺度校正
            # 依赖的平移范数将绑定在任意世界原点上。
            poses = anchor_w2c_sequence(poses)
            return poses.astype(np.float32, copy=False), "camera_yml"
        if gt_source == "camera_yml":
            return None, "none"

    if gt_source in ("npy", "auto"):
        poses = load_w2c_from_npy(scene_root, npy_name)
        if poses is not None:
            return poses, "npy"

    return None, "none"


# ------------------------------------------------------------------ #
# 位姿锚定 / 相对化
# ------------------------------------------------------------------ #

def anchor_w2c_sequence(w2c_seq: np.ndarray) -> np.ndarray:
    """
    将 w2c 位姿序列锚定到第 0 帧为单位阵。

    给定 w2c 序列 T[0..N-1]（world-to-camera），
    对于 w2c 外参，相对化公式为::

        T_rel[i] = T[i] @ inv(T[0])

    使得 T_rel[0] = I, T_rel[i] 表示从第 0 帧相机坐标系到第 i 帧相机坐标系的变换。

    Args:
        w2c_seq: [N, 4, 4] world-to-camera 外参矩阵。

    Returns:
        [N, 4, 4] 相对化后的位姿序列（第 0 帧为 Identity）。
    """
    T0_inv = np.linalg.inv(w2c_seq[0])
    result = np.empty_like(w2c_seq)
    for i in range(len(w2c_seq)):
        result[i] = w2c_seq[i] @ T0_inv
    return result.astype(np.float32, copy=False)


# ------------------------------------------------------------------ #
# 子采样 / 帧索引筛选
# ------------------------------------------------------------------ #

def subset_pose_array(
    poses: np.ndarray,
    kept_indices: Sequence[int],
) -> np.ndarray:
    """
    按 kept_indices 从 [N,4,4] 位姿数组中取子集。

    Args:
        poses:        [N, 4, 4] 原始位姿序列。
        kept_indices: 保留的帧索引列表。

    Returns:
        [len(kept_indices), 4, 4] 子集位姿数组。
    """
    valid = [i for i in kept_indices if 0 <= i < len(poses)]
    if not valid:
        return np.empty((0, 4, 4), dtype=poses.dtype)
    return poses[valid].copy()


# ------------------------------------------------------------------ #
# 保存 gt_poses.npy 缓存
# ------------------------------------------------------------------ #

def save_gt_pose_npy(path: str, poses: np.ndarray) -> None:
    """将 [N,4,4] 位姿数组保存为 .npy 文件。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, poses.astype(np.float32, copy=False))


# ------------------------------------------------------------------ #
# 校验
# ------------------------------------------------------------------ #

def validate_pose_sequence(poses: np.ndarray, tol: float = 1e-3) -> bool:
    """
    检查位姿序列第 0 帧是否接近单位阵。

    Args:
        poses: [N, 4, 4]
        tol:   允许的最大 Frobenius 偏差。

    Returns:
        True 表示通过校验。
    """
    if len(poses) == 0:
        return False
    diff = np.linalg.norm(poses[0] - np.eye(4, dtype=poses.dtype))
    return float(diff) < tol
