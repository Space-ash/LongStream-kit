"""
KITTI Odometry -> LongStream generalizable 格式预处理脚本。

输入数据结构（KITTI 原始格式）：
  <kitti_root>/
    sequences/<seq>/
      image_2/  *.png    (左目灰度/彩色图像)
      calib.txt          (P0..P3)
      times.txt          (每帧时间戳，可选)
    poses/
      <seq>.txt          (每行 12 个数，3×4 c2w 矩阵，行优先)

输出数据结构（LongStream generalizable 格式）：
  <out_root>/
    data_roots.txt
    <seq>/
      images/
        02/
          000000.png
          ...
      cameras/
        02/
          extri.yml
          intri.yml
      gt_poses.npy        # [N,4,4] w2c，锚定到第 0 帧
      gt_poses_02.npy     # 同上（带相机后缀，保持与 vkitti2 管线一致）
      gps_xyz.npy         # [N,3] 相机中心（世界坐标）

用法：
  python scripts/kitti_odometry_to_generalizable.py \\
    --kitti_root  KITTI_Odometry/dataset \\
    --out_root    prepared_inputs/kitti_odometry \\
    --seqs        08 \\
    --camera      02 \\
    --copy                # 默认 symlink，加此参数改为复制

注意：
  - 本脚本只生成 GT 位姿与相机文件，不生成 GT depth。
  - depth/pointcloud 的 GT 指标需要 depths/ 目录，该目录在此场景中不可用。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ------------------------------------------------------------------ #
# Unicode-safe cv2 I/O helpers（解决 Windows 中文路径问题）
# ------------------------------------------------------------------ #

def _imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """cv2.imread 的 Unicode-safe 替代，使用 np.fromfile+cv2.imdecode。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


# 把项目根目录加入 sys.path 以便导入 longstream 包
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from longstream.utils.gt_pose import anchor_w2c_sequence, save_gt_pose_npy


# ------------------------------------------------------------------ #
# 辅助工具
# ------------------------------------------------------------------ #

def log(msg: str) -> None:
    print(msg, flush=True)


def _link_or_copy(src: str, dst: str, copy: bool = False) -> None:
    """跨平台 symlink / copy helper；目标已存在时跳过。"""
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(os.path.abspath(src), dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)


def _write_list(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for ln in lines:
            f.write(f"{ln}\n")


# ------------------------------------------------------------------ #
# calib.txt 解析
# ------------------------------------------------------------------ #

def parse_kitti_calib(calib_path: str) -> Dict[str, np.ndarray]:
    """
    解析 KITTI calib.txt，返回 P0..P3 以 key 为索引的字典。
    每个 value 形状为 (3, 4)。
    """
    result: Dict[str, np.ndarray] = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, vals = line.partition(":")
            key = key.strip()
            vals_arr = np.array([float(v) for v in vals.split()], dtype=np.float64)
            if vals_arr.size == 12:
                result[key] = vals_arr.reshape(3, 4)
    return result


# ------------------------------------------------------------------ #
# poses/<seq>.txt 解析
# ------------------------------------------------------------------ #

def load_kitti_poses(poses_path: str) -> np.ndarray:
    """
    读取 KITTI Odometry poses/<seq>.txt。

    KITTI 标准：每行 12 个数，表示 3×4 c2w 矩阵（行优先）。
    返回 [N, 4, 4] float64 的 w2c 矩阵数组。
    """
    raw = np.loadtxt(poses_path, dtype=np.float64)  # [N, 12]
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]

    N = raw.shape[0]
    w2c_seq = np.zeros((N, 4, 4), dtype=np.float64)
    for i, row in enumerate(raw):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :4] = row.reshape(3, 4)
        w2c_seq[i] = np.linalg.inv(c2w)
    return w2c_seq


# ------------------------------------------------------------------ #
# camera YAML 写入（格式与 vkitti2_to_generalizable.py 一致）
# ------------------------------------------------------------------ #

def _write_camera(camera_dict: Dict[str, dict], out_dir: str) -> None:
    """
    将相机参数字典写入 extri.yml + intri.yml。
    camera_dict 的 key 为帧名（如 "000000"），value 含 K / R / T / H / W。

    cv2.FileStorage 在 Windows 中文路径下无法正常打开文件，
    因此先写到 ASCII 临时目录，再用 shutil.move 移动到目标路径。
    """
    import tempfile
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(camera_dict.keys())

    tmp_dir = tempfile.mkdtemp(prefix="kitti_cam_")
    try:
        tmp_extri = os.path.join(tmp_dir, "extri.yml")
        tmp_intri = os.path.join(tmp_dir, "intri.yml")

        # extri.yml
        fs_extri = cv2.FileStorage(tmp_extri, cv2.FILE_STORAGE_WRITE)
        if hasattr(fs_extri, "isOpened") and not fs_extri.isOpened():
            raise RuntimeError(f"cv2.FileStorage failed to open: {tmp_extri}")
        fs_extri.startWriteStruct("names", cv2.FileNode_SEQ)
        for name in names:
            fs_extri.write("", name)
        fs_extri.endWriteStruct()
        for name in names:
            cam = camera_dict[name]
            fs_extri.write(f"Rot_{name}", np.asarray(cam["R"], dtype=np.float64))
            fs_extri.write(f"T_{name}",   np.asarray(cam["T"], dtype=np.float64))
        fs_extri.release()

        # intri.yml
        fs_intri = cv2.FileStorage(tmp_intri, cv2.FILE_STORAGE_WRITE)
        if hasattr(fs_intri, "isOpened") and not fs_intri.isOpened():
            raise RuntimeError(f"cv2.FileStorage failed to open: {tmp_intri}")
        fs_intri.startWriteStruct("names", cv2.FileNode_SEQ)
        for name in names:
            fs_intri.write("", name)
        fs_intri.endWriteStruct()
        for name in names:
            cam = camera_dict[name]
            fs_intri.write(f"K_{name}", np.asarray(cam["K"], dtype=np.float64))
            fs_intri.write(f"H_{name}", int(cam["H"]))
            fs_intri.write(f"W_{name}", int(cam["W"]))
        fs_intri.release()

        shutil.move(tmp_extri, os.path.join(out_dir, "extri.yml"))
        shutil.move(tmp_intri, os.path.join(out_dir, "intri.yml"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ------------------------------------------------------------------ #
# 单个 sequence 预处理
# ------------------------------------------------------------------ #

def process_sequence(
    kitti_root: str,
    out_root: str,
    seq: str,
    camera: str = "02",
    copy: bool = False,
) -> None:
    """
    将 KITTI Odometry 中的一个 sequence 转换为 generalizable 格式。

    Parameters
    ----------
    kitti_root : str
        KITTI dataset 根目录，包含 sequences/ 和 poses/ 子目录。
    out_root : str
        输出根目录（prepared_inputs/kitti_odometry）。
    seq : str
        序列编号字符串，例如 "08"。
    camera : str
        使用的相机编号，可以是 "2"、"02"、"3"、"03" 等。
        输出目录使用两位补零形式（如 "02"）；读取 KITTI 图片目录时
        同时尝试 image_<N> 和 image_<NN> 两种命名（KITTI 数据集存在两种）。
        使用的相机编号（KITTI image_<camera>），默认 "02"（左目彩色）。
    copy : bool
        True 表示复制图片，False 表示软链接。
    """
    # camera_out：两位补零形式，用于输出目录 images/<camera_out>、cameras/<camera_out>
    camera_out = str(camera).zfill(2)       # e.g. "02"
    # camera_idx：去掉前导零的整数字符串，用于 KITTI 原生 image_<N> 目录和 P<N> 键
    camera_idx = str(int(camera_out))       # e.g. "2"

    seq_dir    = os.path.join(kitti_root, "sequences", seq)
    poses_file = os.path.join(kitti_root, "poses", f"{seq}.txt")
    calib_file = os.path.join(seq_dir, "calib.txt")
    scene_out  = os.path.join(out_root, seq)

    # 同时尝试 image_2（KITTI 官方命名）和 image_02（部分重打包命名）
    _img_dir_candidates = [
        os.path.join(seq_dir, f"image_{camera_idx}"),
        os.path.join(seq_dir, f"image_{camera_out}"),
    ]
    img_src_dir = next((p for p in _img_dir_candidates if os.path.isdir(p)), None)

    # ------ 校验输入 ------
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"[kitti_odometry] sequence dir not found: {seq_dir}")
    if not os.path.isfile(poses_file):
        raise FileNotFoundError(f"[kitti_odometry] poses file not found: {poses_file}")
    if not os.path.isfile(calib_file):
        raise FileNotFoundError(f"[kitti_odometry] calib.txt not found: {calib_file}")
    if img_src_dir is None:
        raise FileNotFoundError(
            f"[kitti_odometry] image dir not found: tried "
            + ", ".join(_img_dir_candidates)
        )

    # ------ 加载 calib ------
    calib = parse_kitti_calib(calib_file)
    p_key = f"P{camera_idx}"  # KITTI calib 使用 P0/P1/P2/P3（无前导零）
    if p_key not in calib:
        raise KeyError(f"[kitti_odometry] {p_key} not found in calib.txt of seq {seq}")
    P = calib[p_key]           # (3, 4)
    K = P[:, :3].copy()        # (3, 3) 内参

    # ------ KITTI P 矩阵包含相机相对 cam0 的平移偏移 ------
    # P = K @ [I | t_cam_offset]，其中 t_cam_offset = K^{-1} @ P[:, 3]
    # KITTI poses 文件给出的是 cam0（P0）坐标系下的位姿，
    # 对于 cam2/cam3 需要叠加此偏移以得到正确的 w2c。
    t_cam_offset = np.linalg.solve(K, P[:, 3])  # (3,) cam0->cam_i 的平移（相机坐标系）
    T_cam_i_cam0 = np.eye(4, dtype=np.float64)
    T_cam_i_cam0[:3, 3] = t_cam_offset

    # ------ 加载 GT 位姿 ------
    log(f"[kitti_odometry] seq {seq}: loading poses from {poses_file}")
    w2c_seq_raw = load_kitti_poses(poses_file)   # [N, 4, 4] w2c（世界坐标系原点=第0帧，cam0 坐标系）
    N_poses = len(w2c_seq_raw)
    log(f"[kitti_odometry] seq {seq}: {N_poses} poses loaded")

    # 收集所有图片
    img_files = sorted([
        f for f in os.listdir(img_src_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    N_imgs = len(img_files)
    log(f"[kitti_odometry] seq {seq}: {N_imgs} images found in {img_src_dir}")

    if N_imgs == 0:
        raise RuntimeError(f"[kitti_odometry] no images found in {img_src_dir}")
    if N_imgs != N_poses:
        log(
            f"[kitti_odometry] WARNING: image count ({N_imgs}) != pose count ({N_poses}), "
            f"using min({N_imgs}, {N_poses}) frames"
        )
    N = min(N_imgs, N_poses)
    img_files = img_files[:N]
    w2c_seq_raw = w2c_seq_raw[:N]

    # ------ 读取一张图以获取 H / W ------
    sample_img = _imread_unicode(os.path.join(img_src_dir, img_files[0]))
    if sample_img is None:
        raise RuntimeError(f"[kitti_odometry] cannot read sample image: {img_files[0]}")
    H, W = sample_img.shape[:2]
    log(f"[kitti_odometry] seq {seq}: image size H={H}, W={W}")

    # ------ 软链接 / 复制图片 ------
    img_out_dir = os.path.join(scene_out, "images", camera_out)
    os.makedirs(img_out_dir, exist_ok=True)
    for tidx, fname in enumerate(img_files):
        # 输出名称：6 位零填充，去掉原扩展名再加上 .png
        stem = Path(fname).stem
        dst_name = f"{stem}.png"
        src_path = os.path.join(img_src_dir, fname)
        dst_path = os.path.join(img_out_dir, dst_name)
        _link_or_copy(src_path, dst_path, copy=copy)

    log(f"[kitti_odometry] seq {seq}: images linked/copied to {img_out_dir}")

    # ------ 构建 camera_dict（每帧一个 entry）------
    # 叠加 cam0 -> cam_i 的偏移（T_cam_i_cam0），将 cam0 位姿变换到 cam_i 坐标系
    # w2c_cam_i = T_cam_i_cam0 @ w2c_cam0
    w2c_seq_cam = np.einsum("ij,njk->nik", T_cam_i_cam0, w2c_seq_raw)  # [N, 4, 4]
    # 位姿锚定到第 0 帧（anchor 后相对坐标系）
    w2c_anchored = anchor_w2c_sequence(w2c_seq_cam)   # [N, 4, 4]

    camera_dict: Dict[str, dict] = {}
    for tidx, fname in enumerate(img_files):
        name = Path(fname).stem          # e.g. "000000"
        w2c = w2c_anchored[tidx]         # (4, 4)
        R = w2c[:3, :3].copy()           # (3, 3)
        T = w2c[:3, 3:].copy()           # (3, 1)
        camera_dict[name] = {
            "K": K.copy(),
            "R": R,
            "T": T,
            "H": H,
            "W": W,
        }

    # ------ 写入 cameras/<camera_out>/extri.yml + intri.yml ------
    cam_out_dir = os.path.join(scene_out, "cameras", camera_out)
    _write_camera(camera_dict, cam_out_dir)
    log(f"[kitti_odometry] seq {seq}: camera files written to {cam_out_dir}")

    # ------ 保存 gt_poses.npy / gt_poses_<camera_out>.npy ------
    save_gt_pose_npy(os.path.join(scene_out, "gt_poses.npy"),              w2c_anchored)
    save_gt_pose_npy(os.path.join(scene_out, f"gt_poses_{camera_out}.npy"), w2c_anchored)
    log(f"[kitti_odometry] seq {seq}: gt_poses.npy and gt_poses_{camera_out}.npy saved")

    # ------ 可选：保存 gps_xyz.npy（相机在世界坐标系中的位置）------
    # c2w = inv(w2c)，相机中心 = c2w[:3, 3] = -R.T @ t
    gps_xyz = np.stack([
        -(w2c[:3, :3].T @ w2c[:3, 3])
        for w2c in w2c_anchored
    ], axis=0)   # [N, 3]
    np.save(os.path.join(scene_out, "gps_xyz.npy"), gps_xyz.astype(np.float32))
    log(f"[kitti_odometry] seq {seq}: gps_xyz.npy saved (shape={gps_xyz.shape})")

    log(f"[kitti_odometry] seq {seq}: done. total {N} frames.")


# ------------------------------------------------------------------ #
# CLI 入口
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert KITTI Odometry sequences to LongStream generalizable format."
    )
    parser.add_argument(
        "--kitti_root",
        default="KITTI_Odometry/dataset",
        help="KITTI dataset root containing sequences/ and poses/ subdirectories.",
    )
    parser.add_argument(
        "--out_root",
        default="prepared_inputs/kitti_odometry",
        help="Output root directory (generalizable meta_root).",
    )
    parser.add_argument(
        "--seqs",
        nargs="+",
        default=["08"],
        help="Sequence IDs to process (e.g. 08 or 00 01 08).",
    )
    parser.add_argument(
        "--camera",
        default="02",
        help="Camera channel to use: 02=left color, 03=right color (default: 02).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy image files instead of creating symlinks.",
    )
    args = parser.parse_args()

    kitti_root = os.path.abspath(args.kitti_root)
    out_root   = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    for seq in args.seqs:
        process_sequence(
            kitti_root=kitti_root,
            out_root=out_root,
            seq=seq,
            camera=args.camera,
            copy=args.copy,
        )

    # 写 data_roots.txt
    roots_file = os.path.join(out_root, "data_roots.txt")
    _write_list(roots_file, args.seqs)
    log(f"[kitti_odometry] data_roots.txt written: {args.seqs}")
    log(f"[kitti_odometry] all done. output: {out_root}")


if __name__ == "__main__":
    main()
