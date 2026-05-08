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
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

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
    解析 KITTI calib.txt，返回以 key 为索引的字典。
    支持的条目：
      - P0..P3              形状 (3, 4)，投影矩阵
      - Tr / Tr_velo_to_cam 形状 (3, 4)，LiDAR->cam0 外参
      - R0_rect / R_rect    形状 (3, 3)，整流旋转矩阵
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
            elif vals_arr.size == 9:
                result[key] = vals_arr.reshape(3, 3)
    return result


def parse_kitti_raw_calib(path: str) -> Dict[str, np.ndarray]:
    """
    解析 KITTI 官方 raw 标定文件（calib_velo_to_cam.txt / calib_cam_to_cam.txt 等）。

    与 calib.txt 的区别：
      - 跳过 calib_time 等非数值行（try-except float 转换失败）
      - 支持任意长度数值行；3 值 -> (3,)，9 值 -> (3,3)，12 值 -> (3,4)，其余保留原形

    若文件不存在则静默返回空字典，由调用方决定如何处理缺失标定。
    """
    result: Dict[str, np.ndarray] = {}
    if not path or not os.path.isfile(path):
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, vals = line.split(":", 1)
            key = key.strip()
            try:
                arr = np.array([float(x) for x in vals.split()], dtype=np.float64)
            except ValueError:
                continue   # 跳过 calib_time 等文本行
            if arr.size == 0:
                continue
            if arr.size == 12:
                result[key] = arr.reshape(3, 4)
            elif arr.size == 9:
                result[key] = arr.reshape(3, 3)
            elif arr.size == 3:
                result[key] = arr.reshape(3)
            else:
                result[key] = arr
    return result


def load_kitti_calibration(
    seq_dir: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    加载序列标定参数，自动合并 calib.txt 与官方 raw 标定文件。

    查找顺序：
      1. <seq_dir>/calib.txt              — 必须存在，包含 P0-P3（及可选 Tr/R0_rect）
      2. <seq_dir>/calib_velo_to_cam.txt  — 若存在，从 R/T 合成 Tr_velo_to_cam
      3. <seq_dir>/calib_cam_to_cam.txt   — 若存在，从 R_rect_00 填充 R0_rect
      4. <seq_dir>/calib_imu_to_velo.txt  — 若存在，合并（备用）

    返回
    -----
    calib        : Dict[str, np.ndarray]
        合并后的标定字典，key 含 P0-P3、Tr_velo_to_cam（若可解析）、R0_rect（若可解析）
    calib_sources: Dict[str, str]
        各关键外参的来源描述，用于写入 depth_meta.json 的 calibration_source 字段
    """
    calib_file = os.path.join(seq_dir, "calib.txt")
    calib = parse_kitti_calib(calib_file)
    calib_sources: Dict[str, str] = {}

    cam_to_cam_path  = os.path.join(seq_dir, "calib_cam_to_cam.txt")
    velo_to_cam_path = os.path.join(seq_dir, "calib_velo_to_cam.txt")
    imu_to_velo_path = os.path.join(seq_dir, "calib_imu_to_velo.txt")

    cam_to_cam  = parse_kitti_raw_calib(cam_to_cam_path)
    velo_to_cam = parse_kitti_raw_calib(velo_to_cam_path)
    imu_to_velo = parse_kitti_raw_calib(imu_to_velo_path)

    # 将 raw 标定项以带前缀的 key 写入 calib，避免与 calib.txt 原有 key 冲突
    calib.update({f"cam_to_cam.{k}": v for k, v in cam_to_cam.items()})
    calib.update({f"velo_to_cam.{k}": v for k, v in velo_to_cam.items()})
    calib.update({f"imu_to_velo.{k}": v for k, v in imu_to_velo.items()})

    # 合成标准 Tr_velo_to_cam（若 calib.txt 无，但 calib_velo_to_cam.txt 有 R/T）
    if "Tr_velo_to_cam" not in calib and "Tr" not in calib:
        if "velo_to_cam.R" in calib and "velo_to_cam.T" in calib:
            Tr = np.zeros((3, 4), dtype=np.float64)
            Tr[:3, :3] = calib["velo_to_cam.R"].reshape(3, 3)
            Tr[:3, 3]  = calib["velo_to_cam.T"].reshape(3)
            calib["Tr_velo_to_cam"] = Tr
            calib_sources["velo_to_cam"] = "calib_velo_to_cam.txt:R/T"
    elif "Tr_velo_to_cam" in calib:
        calib_sources["velo_to_cam"] = "calib.txt:Tr_velo_to_cam"
    elif "Tr" in calib:
        calib_sources["velo_to_cam"] = "calib.txt:Tr"

    # 合成标准 R0_rect（若 calib.txt 无，但 calib_cam_to_cam.txt 有 R_rect_00）
    if "R0_rect" not in calib and "R_rect" not in calib:
        if "cam_to_cam.R_rect_00" in calib:
            calib["R0_rect"] = np.asarray(
                calib["cam_to_cam.R_rect_00"], dtype=np.float64
            ).reshape(3, 3)
            calib_sources["rectification"] = "calib_cam_to_cam.txt:R_rect_00"
    elif "R0_rect" in calib:
        calib_sources["rectification"] = "calib.txt:R0_rect"
    elif "R_rect" in calib:
        calib_sources["rectification"] = "calib.txt:R_rect"

    return calib, calib_sources


# ------------------------------------------------------------------ #
# LiDAR 辅助函数
# ------------------------------------------------------------------ #

def load_velodyne_bin(path: str) -> np.ndarray:
    """加载 KITTI velodyne .bin 文件，返回 Nx4 float32（x, y, z, intensity）。"""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def make_4x4(mat: np.ndarray) -> np.ndarray:
    """将 3x4 或 3x3 矩阵扩展为 4x4 齐次变换矩阵；4x4 直接返回副本。"""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat.copy()
    out = np.eye(4, dtype=np.float64)
    if mat.shape == (3, 4):
        out[:3, :] = mat
    elif mat.shape == (3, 3):
        out[:3, :3] = mat
    else:
        raise ValueError(f"[make_4x4] unsupported shape: {mat.shape}")
    return out


def resolve_velodyne_to_rect_cam0(calib: Dict[str, np.ndarray]) -> np.ndarray:
    """
    从 calib 字典中解析 T_rect0_velo = R0_rect @ Tr_velo_to_cam（均扩展为 4x4）。

    命名兼容：
      - KITTI Odometry  ：Tr + R0_rect
      - KITTI raw        ：Tr_velo_to_cam + R0_rect / R_rect

    R0_rect / R_rect 缺失时用 I 代替（等效不使用整流变换）。
    Tr / Tr_velo_to_cam 均缺失时抛出 KeyError，调用方应传 --depth_mode none。
    """
    tr = calib.get("Tr_velo_to_cam")
    if tr is None:
        tr = calib.get("Tr")
    if tr is None:
        raise KeyError(
            "[kitti_odometry] calib 中未找到 LiDAR 外参（Tr_velo_to_cam / Tr）。\n"
            "  calib.txt 可能只包含 P0-P3，需要以下任一方式提供 LiDAR 外参：\n"
            "  1. 在 calib.txt 同级目录放置 calib_velo_to_cam.txt（包含 R 和 T）\n"
            "  2. 在 calib.txt 中直接写入 Tr_velo_to_cam 或 Tr 行\n"
            "  3. 若该序列确实无 LiDAR，请显式设置 --depth_mode none"
        )
    Tr_4x4 = make_4x4(tr)
    r0 = calib.get("R0_rect")
    if r0 is None:
        r0 = calib.get("R_rect")
    R0_4x4 = make_4x4(r0) if r0 is not None else np.eye(4, dtype=np.float64)
    return R0_4x4 @ Tr_4x4


def project_velodyne_to_sparse_depth(
    bin_path: str,
    K: np.ndarray,
    T_cam_i_cam0: np.ndarray,
    T_rect0_velo: np.ndarray,
    H: int,
    W: int,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> np.ndarray:
    """
    将 velodyne 点云投影到相机 i 平面，生成稀疏深度图（float32）。

    投影流程：
      1. velo Nx4 -> 齐次坐标 Nx4
      2. p_rect0   = T_rect0_velo @ p_velo
      3. p_cam_i   = T_cam_i_cam0 @ p_rect0
      4. 深度范围过滤（min_depth <= z <= max_depth）
      5. u = fx*x/z + cx，v = fy*y/z + cy
      6. 边界过滤
      7. z-buffer（同一像素保留最近 z）

    Returns
    -------
    float32 depth map 形状 (H, W)，无效像素写 0.0，不做插值。
    """
    pts = load_velodyne_bin(bin_path)                                    # Nx4
    ones = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.concatenate([pts[:, :3].astype(np.float64), ones], axis=1)  # Nx4

    # 1. velo -> rect cam0
    p_rect0 = (T_rect0_velo @ pts_h.T).T                                 # Nx4
    # 2. rect cam0 -> cam_i
    p_cam_i = (T_cam_i_cam0 @ p_rect0.T).T                               # Nx4

    # 3. 深度过滤
    z = p_cam_i[:, 2]
    mask = (z >= min_depth) & (z <= max_depth)
    p_cam_i = p_cam_i[mask]
    z = z[mask]
    if len(z) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # 4. 投影
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x, y = p_cam_i[:, 0], p_cam_i[:, 1]
    u = np.round(fx * x / z + cx).astype(np.int32)
    v = np.round(fy * y / z + cy).astype(np.int32)

    # 5. 边界过滤
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]
    if len(z) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # 6. z-buffer：按 z 降序排列后写入，最近点（最小 z）最后覆盖
    order = np.argsort(z)[::-1]
    depth_map = np.zeros((H, W), dtype=np.float32)
    depth_map[v[order], u[order]] = z[order].astype(np.float32)
    return depth_map


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
# Unicode-safe EXR 写入
# ------------------------------------------------------------------ #

def _imwrite_exr_unicode(path: str, depth: np.ndarray) -> None:
    """
    Unicode-safe float32 EXR 写入。

    cv2.imwrite 在 Windows 中文路径下无法正常打开文件，
    先写到 ASCII 临时路径，再 shutil.move 到目标路径。
    """
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".exr")
    os.close(tmp_fd)
    try:
        ok = cv2.imwrite(tmp_path, depth.astype(np.float32))
        if not ok:
            raise RuntimeError(
                f"[kitti_odometry] cv2.imwrite EXR 失败: {path}\n"
                "  请确认 OpenCV 已编译 OpenEXR 支持（OPENCV_IO_ENABLE_OPENEXR=1）。"
            )
        shutil.move(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ------------------------------------------------------------------ #
# 离线 LiDAR 深度处理
# ------------------------------------------------------------------ #

def _process_lidar_depths(
    kitti_root: str,
    seq: str,
    seq_dir: str,
    scene_out: str,
    calib: Dict[str, np.ndarray],
    K: np.ndarray,
    T_cam_i_cam0: np.ndarray,
    img_files: List[str],
    camera_out: str,
    H: int,
    W: int,
    min_depth: float,
    max_depth: float,
    save_depth_mask: bool,
    overwrite_depths: bool,
    num_workers: int,
    calib_sources: Optional[Dict[str, str]] = None,
    allow_missing_lidar_calib: bool = False,
) -> None:
    """
    投影 velodyne 点云到相机视平面，保存稀疏深度图（float32 EXR）。

    输出：
      <scene_out>/depths/<camera_out>/<stem>.exr   — 与 images/<camera>/<stem>.png 同名
      <scene_out>/depth_meta.json                  — 描述稀疏 GT 属性（含标定来源）
      <scene_out>/depth_masks/<camera_out>/<stem>.png（可选，当 save_depth_mask=True）
    """
    import concurrent.futures

    velo_dir = os.path.join(seq_dir, "velodyne")
    if not os.path.isdir(velo_dir):
        msg = (
            f"[kitti_odometry] velodyne 目录不存在: {velo_dir}\n"
            f"  若该序列无 LiDAR 数据，请显式设置 --depth_mode none。"
        )
        if allow_missing_lidar_calib:
            log(f"[kitti_odometry] seq {seq}: WARN: {msg.splitlines()[0]}，跳过深度投影。")
            return
        raise FileNotFoundError(msg)

    # 解析 LiDAR -> rectified cam0 变换（此处会检查 Tr 是否存在）
    try:
        T_rect0_velo = resolve_velodyne_to_rect_cam0(calib)
    except KeyError as exc:
        if allow_missing_lidar_calib:
            log(f"[kitti_odometry] seq {seq}: WARN: {exc}，跳过深度投影。")
            return
        raise

    depth_out_dir = os.path.join(scene_out, "depths", camera_out)
    os.makedirs(depth_out_dir, exist_ok=True)

    mask_out_dir: Optional[str] = None
    if save_depth_mask:
        mask_out_dir = os.path.join(scene_out, "depth_masks", camera_out)
        os.makedirs(mask_out_dir, exist_ok=True)

    def _process_one(fname: str) -> Optional[str]:
        stem = Path(fname).stem
        exr_path = os.path.join(depth_out_dir, f"{stem}.exr")
        if not overwrite_depths and os.path.exists(exr_path):
            return None  # 已存在，跳过
        bin_path = os.path.join(velo_dir, f"{stem}.bin")
        if not os.path.isfile(bin_path):
            return f"[kitti_odometry] WARN: velodyne bin 不存在: {bin_path}"
        depth = project_velodyne_to_sparse_depth(
            bin_path=bin_path,
            K=K,
            T_cam_i_cam0=T_cam_i_cam0,
            T_rect0_velo=T_rect0_velo,
            H=H,
            W=W,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        _imwrite_exr_unicode(exr_path, depth)
        if save_depth_mask and mask_out_dir is not None:
            mask = (depth > 0).astype(np.uint8) * 255
            _, buf = cv2.imencode(".png", mask)
            np.array(buf).tofile(os.path.join(mask_out_dir, f"{stem}.png"))
        return None

    N = len(img_files)
    log(
        f"[kitti_odometry] seq {seq}: "
        f"投影 {N} 帧 LiDAR 深度图（workers={num_workers}）"
    )

    if num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_process_one, fname): fname for fname in img_files}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                warn = fut.result()
                if warn:
                    log(warn)
                done += 1
                if done % 500 == 0:
                    log(f"[kitti_odometry] seq {seq}: depth {done}/{N}")
    else:
        for i, fname in enumerate(img_files):
            warn = _process_one(fname)
            if warn:
                log(warn)
            if (i + 1) % 500 == 0:
                log(f"[kitti_odometry] seq {seq}: depth {i + 1}/{N}")

    # 写 depth_meta.json
    meta: Dict = {
        "depth_gt_type": "sparse_lidar",
        "source": "KITTI Odometry velodyne",
        "invalid_value": 0.0,
        "camera": camera_out,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "metric_scope": "valid_lidar_projected_pixels_only",
    }
    if calib_sources:
        meta["calibration_source"] = calib_sources
    meta_path = os.path.join(scene_out, "depth_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log(f"[kitti_odometry] seq {seq}: depth_meta.json → {meta_path}")
    log(f"[kitti_odometry] seq {seq}: 稀疏深度图已写入 → {depth_out_dir}")


# ------------------------------------------------------------------ #
# 单个 sequence 预处理
# ------------------------------------------------------------------ #

def process_sequence(
    kitti_root: str,
    out_root: str,
    seq: str,
    camera: str = "02",
    copy: bool = False,
    depth_mode: str = "sparse_lidar",
    min_depth: float = 0.1,
    max_depth: float = 80.0,
    save_depth_mask: bool = False,
    overwrite_depths: bool = False,
    num_workers: int = 1,
    allow_missing_lidar_calib: bool = False,
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
        同时尝试 image_<N> 和 image_<NN> 两种命名。
    copy : bool
        True 表示复制图片，False 表示软链接。
    depth_mode : str
        "sparse_lidar" 表示生成离线 LiDAR 深度图；
        "none" 表示跳过深度生成（该序列无 velodyne 或无 LiDAR 外参时使用）。
    min_depth : float
        投影深度下限（米）。
    max_depth : float
        投影深度上限（米）。
    save_depth_mask : bool
        True 表示额外保存深度有效像素掋码图（PNG）。
    overwrite_depths : bool
        True 表示强制重写已存在的 EXR 文件。
    num_workers : int
        深度投影的并行线程数。
    allow_missing_lidar_calib : bool
        True 表示当缺少 velodyne/ 或 LiDAR 外参时仅输出 WARNING、跳过深度生成，
        而不中断预处理（适用于混有缺己 LiDAR 的序列批）。
        默认 False（fail fast）。
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

    # ------ 加载 calib：自动合并 calib.txt 与周边 raw 标定文件 ------
    calib, calib_sources = load_kitti_calibration(seq_dir)
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

    # ------ 离线 LiDAR 深度投影 ------
    if depth_mode == "sparse_lidar":
        # 将相机的投影矩阵来源写入标定信息
        calib_sources["projection"] = f"calib.txt:P{camera_idx}"
        _process_lidar_depths(
            kitti_root=kitti_root,
            seq=seq,
            seq_dir=seq_dir,
            scene_out=scene_out,
            calib=calib,
            K=K,
            T_cam_i_cam0=T_cam_i_cam0,
            img_files=img_files,
            camera_out=camera_out,
            H=H,
            W=W,
            min_depth=min_depth,
            max_depth=max_depth,
            save_depth_mask=save_depth_mask,
            overwrite_depths=overwrite_depths,
            num_workers=num_workers,
            calib_sources=calib_sources,
            allow_missing_lidar_calib=allow_missing_lidar_calib,
        )
    elif depth_mode == "none":
        log(f"[kitti_odometry] seq {seq}: depth_mode=none，跳过 LiDAR 深度投影。")
    else:
        raise ValueError(
            f"[kitti_odometry] 未知 depth_mode: {depth_mode!r}，可选: sparse_lidar, none"
        )

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
    # ---- 离线 LiDAR 深度相关参数 ----
    parser.add_argument(
        "--depth_mode",
        default="sparse_lidar",
        choices=["sparse_lidar", "none"],
        help=(
            "LiDAR 深度生成模式：sparse_lidar（默认）生成稀疏深度图；"
            "若序列缺少 velodyne/ 目录或 calib 无 LiDAR 外参，请显式设置 none。"
        ),
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.1,
        help="投影深度下限（米），默认 0.1。",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=80.0,
        help="投影深度上限（米），默认 80.0。",
    )
    parser.add_argument(
        "--save_depth_mask",
        action="store_true",
        help="额外保存深度有效像素掩码图（PNG）到 depth_masks/<camera>/。",
    )
    parser.add_argument(
        "--overwrite_depths",
        action="store_true",
        help="强制重写已存在的 EXR 文件，默认跳过。",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="深度投影的并行线程数，默认 1（串行）。",
    )
    parser.add_argument(
        "--allow_missing_lidar_calib",
        action="store_true",
        help=(
            "当 velodyne/ 目录或 LiDAR 外参缺失时，仅输出 WARNING 并跳过深度生成，"
            "不中断预处理。默认为严格模式（fail fast）。"
        ),
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
            depth_mode=args.depth_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            save_depth_mask=args.save_depth_mask,
            overwrite_depths=args.overwrite_depths,
            num_workers=args.num_workers,
            allow_missing_lidar_calib=args.allow_missing_lidar_calib,
        )

    # 写 data_roots.txt
    roots_file = os.path.join(out_root, "data_roots.txt")
    _write_list(roots_file, args.seqs)
    log(f"[kitti_odometry] data_roots.txt written: {args.seqs}")
    log(f"[kitti_odometry] all done. output: {out_root}")


if __name__ == "__main__":
    main()
