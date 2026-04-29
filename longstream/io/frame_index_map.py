"""
帧索引映射的保存与加载。

显式记录"推理第 i 帧对应原始序列第几帧"，
避免评估时靠隐式重复筛帧来猜测对齐关系。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


def save_frame_index_map(
    path: str,
    kept_indices: List[int],
    image_paths: Optional[List[str]] = None,
) -> None:
    """
    保存帧索引映射为 JSON 文件。

    格式::

        {
          "description": "...",
          "num_infer_frames": <int>,
          "mapping": [
            {"infer_idx": 0, "original_idx": 3, "image_path": "..."},
            ...
          ]
        }

    Args:
        path:           输出 JSON 路径。
        kept_indices:   保留帧的原始索引列表（推理第 i 帧 → 原始第 kept_indices[i] 帧）。
        image_paths:    （可选）推理所用的图片路径列表，长度应与 kept_indices 一致。
    """
    mapping = []
    for infer_idx, orig_idx in enumerate(kept_indices):
        entry: Dict[str, object] = {
            "infer_idx": infer_idx,
            "original_idx": orig_idx,
        }
        if image_paths is not None and infer_idx < len(image_paths):
            entry["image_path"] = image_paths[infer_idx]
        mapping.append(entry)

    data = {
        "description": "Frame index mapping: infer_idx -> original sequence index",
        "num_infer_frames": len(kept_indices),
        "mapping": mapping,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_frame_index_map(path: str) -> Optional[List[int]]:
    """
    加载帧索引映射，返回 kept_indices 列表。

    Returns:
        kept_indices: 推理第 i 帧对应的原始帧索引列表，或文件不存在时返回 None。
    """
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = data.get("mapping", [])
    # 按 infer_idx 排序并提取 original_idx
    mapping.sort(key=lambda x: x["infer_idx"])
    return [entry["original_idx"] for entry in mapping]
