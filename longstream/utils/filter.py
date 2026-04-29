"""
帧质量过滤模块：在推理前剔除模糊帧和冗余帧。

函数 `is_high_quality` 可在数据加载阶段逐帧调用，
从方案说明「优化1：输入预筛选」中提取。

独立评分函数 `blur_score` / `frame_diff_score` 可用于细粒度分析。
`filter_frame_sequence` 提供一步到位的批量筛选接口。
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


# ------------------------------------------------------------------ #
# 独立评分函数
# ------------------------------------------------------------------ #

def blur_score(img: np.ndarray) -> float:
    """
    计算图像的清晰度评分（拉普拉斯方差）。

    值越大表示越清晰；典型模糊帧 < 100。

    Args:
        img: (H, W, C) 或 (H, W) uint8 图像。

    Returns:
        拉普拉斯方差（float）。
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.astype(np.uint8, copy=False)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_diff_score(
    img: np.ndarray,
    prev_img: np.ndarray,
) -> float:
    """
    计算当前帧与前一帧的运动量评分（归一化均值绝对差，0~1）。

    值越大表示帧间差异越大；典型静止帧 < 0.02。

    Args:
        img:      当前帧 (H, W, C) uint8。
        prev_img: 前一帧 (H, W, C) uint8。

    Returns:
        归一化均值绝对差（float，范围 0~1）。
    """
    return float(
        np.mean(np.abs(img.astype(np.float32) - prev_img.astype(np.float32)))
    ) / 255.0


def is_high_quality(
    img: np.ndarray,
    prev_img: Optional[np.ndarray],
    blur_threshold: float = 100.0,
    motion_threshold: float = 0.02,
) -> bool:
    """
    判断当前帧是否满足质量要求。

    检测逻辑：
      1. 模糊检测：计算灰度图的拉普拉斯方差；低于 blur_threshold 则视为模糊帧。
      2. 冗余帧检测：与前一帧的均值绝对差（归一化到 0~1）；
                     低于 motion_threshold 则视为静止/重复帧。

    Args:
        img:              当前帧，形状 (H, W, C)，BGR 或 RGB，uint8。
        prev_img:         前一帧（相同形状），首帧时传入 None。
        blur_threshold:   拉普拉斯方差阈值；越大对清晰度要求越高。
        motion_threshold: 帧间差阈值（0-1）；越大对运动量要求越高。

    Returns:
        True 表示帧质量合格，可送入模型；False 表示应跳过该帧。
    """
    # ---- 1. 模糊检测 ----
    if blur_score(img) < blur_threshold:
        return False

    # ---- 2. 冗余帧检测 ----
    if prev_img is not None:
        if frame_diff_score(img, prev_img) < motion_threshold:
            return False

    return True


# ------------------------------------------------------------------ #
# 批量筛选接口
# ------------------------------------------------------------------ #

def filter_frame_sequence(
    image_paths: List[str],
    blur_threshold: float = 100.0,
    motion_threshold: float = 0.02,
) -> Tuple[List[str], List[int]]:
    """
    对图片路径列表做一次性质量筛选，返回过滤后路径和保留索引。

    Args:
        image_paths:      原始图片路径列表。
        blur_threshold:   拉普拉斯方差阈值。
        motion_threshold: 帧间差阈值（0-1）。

    Returns:
        (kept_paths, kept_indices)
    """
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
            blur_threshold=blur_threshold,
            motion_threshold=motion_threshold,
        ):
            kept_paths.append(path)
            kept_indices.append(idx)
            prev_img = img

    return kept_paths, kept_indices
