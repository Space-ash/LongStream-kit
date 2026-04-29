#!/usr/bin/env python3
"""
将 vKITTI2 的 extrinsic.txt 解析为以第一帧为原点的相对绝对位姿，
并保存为 gt_poses.npy（形状 [N, 4, 4]）。

vKITTI2 extrinsic.txt 列顺序（首行为表头，跳过）：
    frame  cameraID  r11 r12 r13 t1  r21 r22 r23 t2  r31 r32 r33 t3
矩阵含义：world-to-camera 外参（相机外参矩阵 [R|t]）。

相对化公式（w2c 外参）：
    T_rel[i] = T[i] @ inv(T[0])
    使得 T_rel[0] = I，T_rel[i] 表示从第 0 帧相机坐标系到第 i 帧相机坐标系的变换。

用法：
    python scripts/preprocess_vkitti2.py \
        --extrinsic /path/to/vkitti2/Scene01/15-deg-left/extrinsic.txt \
        --output    /path/to/prepared_inputs/.../gt_poses.npy \
        [--camera_id 0]
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

# 把项目根目录加入 sys.path 以便导入 longstream 包
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from longstream.utils.gt_pose import anchor_w2c_sequence, save_gt_pose_npy, validate_pose_sequence


# ------------------------------------------------------------------ #
# 第一步：解析 vKITTI2 extrinsic.txt → 原始 w2c 矩阵
# ------------------------------------------------------------------ #

def parse_vkitti2_extrinsic_txt(
    txt_path: str, camera_id: int = 0
) -> Dict[int, np.ndarray]:
    """
    解析 vKITTI2 extrinsic.txt，返回 {frame_index: 4x4 原始 w2c} 字典。

    不做任何相对化处理，只负责读取原始 world-to-camera 外参。

    Args:
        txt_path:  extrinsic.txt 文件路径。
        camera_id: 仅处理该 ID 对应的相机（0=左相机，1=右相机）。

    Returns:
        {frame_idx: ndarray(4,4, float64)}
    """
    raw_data = np.loadtxt(txt_path, skiprows=1)
    if raw_data.ndim == 1:
        raw_data = raw_data[np.newaxis, :]

    poses: Dict[int, np.ndarray] = {}

    for row in raw_data:
        f_idx = int(row[0])
        cam_id = int(row[1])
        if cam_id != camera_id:
            continue

        v = row[2:14]
        T_w2c = np.array(
            [
                [v[0], v[1], v[2],  v[3]],
                [v[4], v[5], v[6],  v[7]],
                [v[8], v[9], v[10], v[11]],
                [0.0,  0.0,  0.0,   1.0],
            ],
            dtype=np.float64,
        )
        poses[f_idx] = T_w2c

    return poses


# ------------------------------------------------------------------ #
# 第二步：相对化（锚定到第 0 帧 = Identity）
# ------------------------------------------------------------------ #

def convert_w2c_to_first_frame_anchored(w2c_dict: Dict[int, np.ndarray]) -> np.ndarray:
    """
    将原始 w2c 字典按帧号升序排列，调用统一的 anchor_w2c_sequence 进行相对化。

    对于 w2c 外参，正确的相对化公式为：
        T_rel[i] = T[i] @ inv(T[0])

    Returns:
        [N, 4, 4] float32 相对化后的位姿数组。
    """
    sorted_keys = sorted(w2c_dict.keys())
    w2c_seq = np.stack([w2c_dict[k] for k in sorted_keys], axis=0)
    return anchor_w2c_sequence(w2c_seq)


# ------------------------------------------------------------------ #
# CLI 入口
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="预处理 vKITTI2 外参文件，生成以第一帧为原点的相对绝对位姿 .npy"
    )
    parser.add_argument(
        "--extrinsic", required=True, help="vKITTI2 extrinsic.txt 路径"
    )
    parser.add_argument(
        "--output", required=True, help="输出 gt_poses.npy 的保存路径"
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="要处理的相机 ID（0=左相机，1=右相机），默认 0",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.extrinsic):
        raise FileNotFoundError(f"外参文件不存在：{args.extrinsic}")

    # 1. 读取原始 w2c
    raw_poses = parse_vkitti2_extrinsic_txt(args.extrinsic, camera_id=args.camera_id)
    if not raw_poses:
        raise RuntimeError(
            f"在 {args.extrinsic} 中未找到 camera_id={args.camera_id} 的位姿数据"
        )

    # 2. 相对化（锚定到第 0 帧）
    poses_array = convert_w2c_to_first_frame_anchored(raw_poses)  # [N, 4, 4]

    # 3. 校验
    if not validate_pose_sequence(poses_array):
        print(
            f"[preprocess_vkitti2] 警告：第 0 帧偏离单位矩阵较大:\n{poses_array[0]}"
        )

    # 4. 保存
    save_gt_pose_npy(args.output, poses_array)
    print(
        f"[preprocess_vkitti2] 已保存 {len(poses_array)} 帧位姿至 {args.output}"
    )
    print(f"  数组形状: {poses_array.shape}，dtype: {poses_array.dtype}")
    print(f"  第 0 帧（应为单位矩阵）:\n{poses_array[0]}")


if __name__ == "__main__":
    main()
