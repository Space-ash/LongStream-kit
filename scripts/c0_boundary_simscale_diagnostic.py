"""
C0 Boundary Sim/Scale Diagnostic Tool
======================================
离线诊断脚本：验证"batch-local gauge/scale reset 引起的断层"能否通过
对后段 depth + camera center 同步缩放来缓解。

不修改原始推理结果，只输出实验性修正版到独立目录。

用法示例：
    python scripts/c0_boundary_simscale_diagnostic.py ^
        --input  outputs/drone_scene/drone_scene_200to270_new/drone_scene ^
        --output outputs/drone_scene/drone_scene_200to270_new_c0_simscale/drone_scene ^
        --gps    prepared_inputs/drone_scene/drone_scene/gps_xyz.npy ^
        --frame-start 200 ^
        --boundary 48 ^
        --mode depth

缩放模式（--mode）：
  depth      -- 边界前后深度中位数比值
  pose       -- 边界 GPS 步长 / 预测步长
  geom_mean  -- sqrt(s_depth * s_pose)
  manual     -- 直接用 --manual-scale 指定
"""

import argparse
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_w2c_txt(path: str):
    """
    读取 abs_pose.txt。
    格式：# w2c
          frame r00 r01 ... r22 tx ty tz
    返回：
        frames  np.ndarray [S]        (局部帧号, int)
        w2c     np.ndarray [S, 4, 4]  (float64)
    """
    frames = []
    mats = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            frame = int(vals[0])
            nums = list(map(float, vals[1:]))
            # nums = [r00..r22, tx, ty, tz]  共 12 个
            r = np.array(nums[:9]).reshape(3, 3)
            t = np.array(nums[9:12])
            mat = np.eye(4)
            mat[:3, :3] = r
            mat[:3, 3] = t
            frames.append(frame)
            mats.append(mat)
    frames = np.array(frames, dtype=np.int64)
    w2c = np.array(mats)          # [S, 4, 4]
    return frames, w2c


def save_w2c_txt(path: str, frames: np.ndarray, w2c: np.ndarray):
    """
    与 longstream/io/save_poses_txt.py 保持一致。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# w2c\n")
        for i, frame in enumerate(frames):
            mat = w2c[i]
            r = mat[:3, :3].reshape(-1)
            t = mat[:3, 3].reshape(-1)
            vals = [int(frame)] + r.tolist() + t.tolist()
            f.write(" ".join([str(v) for v in vals]) + "\n")


def load_intri_txt(path: str):
    """
    读取 intri.txt。
    格式：# fx fy cx cy
          frame fx fy cx cy
    返回：
        frames  np.ndarray [S]
        intri   np.ndarray [S, 3, 3]
    """
    frames = []
    mats = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            frame = int(vals[0])
            fx, fy, cx, cy = map(float, vals[1:5])
            K = np.array([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]])
            frames.append(frame)
            mats.append(K)
    return np.array(frames, dtype=np.int64), np.array(mats)


def save_intri_txt(path: str, frames: np.ndarray, intri: np.ndarray):
    """与 longstream/io/save_poses_txt.py 保持一致。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# fx fy cx cy\n")
        for i, frame in enumerate(frames):
            K = intri[i]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            f.write(f"{int(frame)} {fx} {fy} {cx} {cy}\n")


def load_depths(depth_dir: str):
    """
    读取 frame_*.npy，返回按帧号排序的 list[np.ndarray]。
    同时返回对应的文件名列表（方便后续按原名保存）。
    """
    files = sorted(
        [f for f in os.listdir(depth_dir) if f.endswith(".npy")],
        key=lambda x: int(x.replace("frame_", "").replace(".npy", ""))
    )
    depths = []
    for fn in files:
        d = np.load(os.path.join(depth_dir, fn))
        depths.append(d.astype(np.float64))
    return depths, files


# ---------------------------------------------------------------------------
# Camera geometry helpers
# ---------------------------------------------------------------------------

def camera_centers_from_w2c(w2c: np.ndarray) -> np.ndarray:
    """
    C = -R^T t
    w2c: [S, 4, 4]  →  centers: [S, 3]
    """
    R = w2c[:, :3, :3]                                 # [S, 3, 3]
    t = w2c[:, :3, 3]                                  # [S, 3]
    # R^T = R.transpose(0,2,1)
    centers = -np.einsum("nij,nj->ni", R.transpose(0, 2, 1), t)
    return centers


def w2c_from_centers_and_rotations(w2c_old: np.ndarray,
                                   centers_new: np.ndarray) -> np.ndarray:
    """
    Rotation 不变，根据新 camera center 重建 t = -R C。
    w2c_old:      [S, 4, 4]
    centers_new:  [S, 3]
    返回：        [S, 4, 4]
    """
    w2c_new = w2c_old.copy()
    R = w2c_old[:, :3, :3]                             # [S, 3, 3]
    t_new = -np.einsum("nij,nj->ni", R, centers_new)  # [S, 3]
    w2c_new[:, :3, 3] = t_new
    return w2c_new


# ---------------------------------------------------------------------------
# Depth statistics
# ---------------------------------------------------------------------------

def median_depth(depth: np.ndarray, mask=None) -> float:
    """只用正值 & 有限值的像素。"""
    v = depth[np.isfinite(depth) & (depth > 0)]
    if mask is not None:
        v = depth[np.isfinite(depth) & (depth > 0) & mask]
    if len(v) == 0:
        return float("nan")
    return float(np.median(v))


def window_median_depth(depths: list, indices) -> float:
    """多帧窗口合并后求中位数，避免单帧噪声。"""
    vals = []
    for i in indices:
        d = depths[i]
        v = d[np.isfinite(d) & (d > 0)]
        vals.append(v)
    if not vals:
        return float("nan")
    all_vals = np.concatenate(vals)
    return float(np.median(all_vals))


# ---------------------------------------------------------------------------
# Scale estimation
# ---------------------------------------------------------------------------

def estimate_scale(depths: list,
                   centers: np.ndarray,
                   gps_slice: np.ndarray,
                   boundary: int,
                   mode: str,
                   manual_scale: float = 1.0) -> dict:
    """
    估计缩放因子，同时收集所有诊断数值。

    参数：
        depths      list[np.ndarray]，局部帧号 0..S-1
        centers     [S, 3]，camera centers（world space）
        gps_slice   [S, 3]，对应的 GPS XYZ（与局部帧对齐）
        boundary    int，锚点帧局部索引 b
        mode        'depth' | 'pose' | 'geom_mean' | 'manual'
        manual_scale float（仅 mode='manual' 时使用）

    返回：dict，含 scale_used 及所有中间量
    """
    S = len(depths)
    b = boundary

    # --- depth 窗口 ---
    before_idx = list(range(max(0, b - 3), b + 1))      # b-3 .. b (含)
    after_idx  = list(range(b + 1, min(S, b + 5)))      # b+1 .. b+4

    depth_before = window_median_depth(depths, before_idx)
    depth_after  = window_median_depth(depths, after_idx)

    if depth_after > 0 and np.isfinite(depth_after):
        s_depth = depth_before / depth_after
    else:
        s_depth = float("nan")

    # --- pose 窗口 ---
    if b + 1 < S:
        pred_step = float(np.linalg.norm(centers[b + 1] - centers[b]))
    else:
        pred_step = float("nan")

    if gps_slice is not None and b + 1 < len(gps_slice):
        gps_step = float(np.linalg.norm(gps_slice[b + 1] - gps_slice[b]))
    else:
        gps_step = float("nan")

    if pred_step > 1e-9 and np.isfinite(pred_step) and np.isfinite(gps_step):
        s_pose = gps_step / pred_step
    else:
        s_pose = float("nan")

    # --- 选择使用的尺度 ---
    if mode == "depth":
        scale_used = s_depth
    elif mode == "pose":
        scale_used = s_pose
    elif mode == "geom_mean":
        if np.isfinite(s_depth) and np.isfinite(s_pose):
            scale_used = float(np.sqrt(s_depth * s_pose))
        else:
            scale_used = s_depth if np.isfinite(s_depth) else s_pose
    elif mode == "manual":
        scale_used = manual_scale
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not np.isfinite(scale_used):
        raise RuntimeError(
            f"Could not compute a finite scale for mode='{mode}'. "
            f"s_depth={s_depth:.4f}, s_pose={s_pose:.4f}. "
            "Check boundary index and GPS alignment."
        )

    report = {
        "boundary": b,
        "mode": mode,
        "before_window": before_idx,
        "after_window": after_idx,
        "depth_before": depth_before,
        "depth_after": depth_after,
        "s_depth": s_depth,
        "pred_step_before": pred_step,
        "gps_step": gps_step,
        "s_pose": s_pose,
        "scale_used": scale_used,
    }
    return report


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------

def unproject_depth_np(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    深度反投影到相机坐标系。
    depth: [H, W]  (正值为有效，NaN/<=0 视为无效)
    K:     [3, 3]
    返回：pts_cam [H, W, 3]，无效点 z<=0 或 nan。
    """
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)   # [H, W, 3]
    return pts_cam


def cam_to_world_points(pts_cam: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """
    把相机坐标系点云转到 world 坐标系，过滤无效点。
    pts_cam: [H, W, 3] 或 [N, 3]
    w2c:     [4, 4]
    返回：   [M, 3] float32，只含有效点
    """
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts = pts_cam.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
    pts = pts[valid]
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # world = R^T (pts - t)  ←→  R^T pts^T - R^T t
    world = (R.T @ (pts.T - t[:, None])).T
    return world.astype(np.float32)


def _maybe_downsample(pts: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points is None or pts.shape[0] <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    keep = rng.choice(pts.shape[0], size=max_points, replace=False)
    return pts[keep]


def save_ply_xyz(path: str, pts: np.ndarray):
    """
    保存无颜色 PLY（binary little endian）。
    使用项目内已有的 save_pointcloud 函数。
    """
    # 尝试用项目内函数
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from longstream.io.save_points import save_pointcloud
        save_pointcloud(path, pts, colors=None)
    except ImportError:
        # fallback: 手写 PLY
        _write_ply_binary(path, pts)


def _write_ply_binary(path: str, pts: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pts = pts.astype(np.float32)
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"end_header\n")
        pts.tofile(f)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(args):
    input_dir  = args.input
    output_dir = args.output
    boundary   = args.boundary

    # ------------------------------------------------------------------
    # 安全检查：不覆写已有输出（除非 --overwrite）
    # ------------------------------------------------------------------
    if os.path.exists(output_dir):
        if not args.overwrite:
            print(f"[ERROR] Output directory already exists: {output_dir}")
            print("        Pass --overwrite to allow overwriting.")
            sys.exit(1)
        else:
            print(f"[WARN]  Output directory exists, overwriting: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 加载输入数据
    # ------------------------------------------------------------------
    pose_path  = os.path.join(input_dir, "poses", "abs_pose.txt")
    intri_path = os.path.join(input_dir, "poses", "intri.txt")
    depth_dir  = os.path.join(input_dir, "depth", "dpt")

    print(f"[INFO] Loading poses:  {pose_path}")
    frames, w2c_old = load_w2c_txt(pose_path)

    print(f"[INFO] Loading intri:  {intri_path}")
    intri_frames, intri = load_intri_txt(intri_path)

    print(f"[INFO] Loading depths: {depth_dir}")
    depths_old, depth_files = load_depths(depth_dir)

    S = len(frames)
    print(f"[INFO] Total frames: {S}  (local 0..{S-1})")

    # ------------------------------------------------------------------
    # 加载 GPS（可选）
    # ------------------------------------------------------------------
    gps_slice = None
    if args.gps:
        print(f"[INFO] Loading GPS:    {args.gps}")
        gps_all = np.load(args.gps)
        fs = args.frame_start
        gps_slice = gps_all[fs: fs + S]
        print(f"[INFO] GPS slice: global[{fs}:{fs+S}]  shape={gps_slice.shape}")

    # ------------------------------------------------------------------
    # 验证边界索引
    # ------------------------------------------------------------------
    if boundary < 0 or boundary >= S - 1:
        print(f"[ERROR] boundary={boundary} out of range [0, {S-2}]")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 估计尺度
    # ------------------------------------------------------------------
    centers_old = camera_centers_from_w2c(w2c_old)   # [S, 3]

    print(f"[INFO] Estimating scale  mode={args.mode}  boundary={boundary}")
    report = estimate_scale(
        depths_old, centers_old, gps_slice,
        boundary=boundary,
        mode=args.mode,
        manual_scale=args.manual_scale,
    )
    s = report["scale_used"]
    print(f"[INFO] s_depth={report['s_depth']:.4f}  "
          f"s_pose={report['s_pose']:.4f}  "
          f"scale_used={s:.4f}")

    # ------------------------------------------------------------------
    # 核心校正
    # ------------------------------------------------------------------
    b = boundary
    anchor = centers_old[b].copy()

    # Camera centers：后段从 b+1 开始缩放
    centers_new = centers_old.copy()
    centers_new[b + 1:] = anchor + s * (centers_old[b + 1:] - anchor)

    # 深度：后段同步缩放
    depths_new = []
    for i, d in enumerate(depths_old):
        if i >= b + 1:
            depths_new.append(d * s)
        else:
            depths_new.append(d.copy())

    # 重建 w2c（rotation 不变）
    w2c_new = w2c_from_centers_and_rotations(w2c_old, centers_new)

    # ------------------------------------------------------------------
    # 诊断指标（校正前后对比）
    # ------------------------------------------------------------------
    pred_step_before = float(np.linalg.norm(centers_old[b + 1] - centers_old[b]))
    pred_step_after  = float(np.linalg.norm(centers_new[b + 1] - centers_new[b]))
    gps_step_val     = report["gps_step"]

    depth_b_before   = median_depth(depths_old[b])
    depth_next_before = median_depth(depths_old[b + 1])
    depth_ratio_before = (depth_next_before / depth_b_before
                          if depth_b_before > 0 else float("nan"))

    depth_b_after    = median_depth(depths_new[b])
    depth_next_after  = median_depth(depths_new[b + 1])
    depth_ratio_after = (depth_next_after / depth_b_after
                         if depth_b_after > 0 else float("nan"))

    report.update({
        "pred_step_before": pred_step_before,
        "pred_step_after":  pred_step_after,
        "gps_step_val":     gps_step_val,
        "depth_b_before":   depth_b_before,
        "depth_next_before": depth_next_before,
        "depth_ratio_before": depth_ratio_before,
        "depth_b_after":    depth_b_after,
        "depth_next_after": depth_next_after,
        "depth_ratio_after": depth_ratio_after,
    })

    print("\n[DIAG] ============ Boundary Diagnostic ============")
    print(f"       boundary frame (local):   {b}")
    print(f"       scale applied:             {s:.6f}")
    print(f"       pred_step  before: {pred_step_before:.6f}")
    print(f"       pred_step  after:  {pred_step_after:.6f}")
    if np.isfinite(gps_step_val):
        print(f"       gps_step:          {gps_step_val:.6f}")
    print(f"       depth_ratio before (b+1/b): {depth_ratio_before:.4f}")
    print(f"       depth_ratio after  (b+1/b): {depth_ratio_after:.4f}")
    print("[DIAG] ================================================\n")

    # ------------------------------------------------------------------
    # 保存 poses
    # ------------------------------------------------------------------
    out_pose_dir = os.path.join(output_dir, "poses")
    os.makedirs(out_pose_dir, exist_ok=True)

    save_w2c_txt(os.path.join(out_pose_dir, "abs_pose.txt"), frames, w2c_new)
    print(f"[SAVE] poses/abs_pose.txt")

    save_intri_txt(os.path.join(out_pose_dir, "intri.txt"), intri_frames, intri)
    print(f"[SAVE] poses/intri.txt")

    # ------------------------------------------------------------------
    # 保存 depth
    # ------------------------------------------------------------------
    out_depth_dir = os.path.join(output_dir, "depth", "dpt")
    os.makedirs(out_depth_dir, exist_ok=True)
    for fn, d in zip(depth_files, depths_new):
        out_path = os.path.join(out_depth_dir, fn)
        np.save(out_path, d.astype(np.float32))
    print(f"[SAVE] depth/dpt/  ({len(depth_files)} files)")

    # ------------------------------------------------------------------
    # 导出点云
    # ------------------------------------------------------------------
    out_pts_frame_dir = os.path.join(output_dir, "points", "dpt_unproj")
    os.makedirs(out_pts_frame_dir, exist_ok=True)

    all_pts_list = []
    rng = np.random.default_rng(42)

    for i, (fn, d_new, w2c_i, K_i) in enumerate(
            zip(depth_files, depths_new, w2c_new, intri)):
        pts_cam = unproject_depth_np(d_new, K_i)
        pts_world = cam_to_world_points(pts_cam, w2c_i)

        # 每帧下采样后保存
        pts_frame = _maybe_downsample(pts_world, args.max_frame_points)
        frame_ply = os.path.join(
            out_pts_frame_dir,
            fn.replace(".npy", ".ply")
        )
        save_ply_xyz(frame_ply, pts_frame)

        # 收集用于全局点云（先做粗下采样，减少内存压力）
        subsample_n = min(pts_world.shape[0],
                          max(1, args.max_full_points // S))
        if pts_world.shape[0] > subsample_n:
            keep = rng.choice(pts_world.shape[0], size=subsample_n, replace=False)
            all_pts_list.append(pts_world[keep])
        else:
            all_pts_list.append(pts_world)

    print(f"[SAVE] points/dpt_unproj/  ({len(depth_files)} ply files)")

    # 全局点云
    if all_pts_list:
        full_pts = np.concatenate(all_pts_list, axis=0)
        full_pts = _maybe_downsample(full_pts, args.max_full_points)

        full_ply_path = os.path.join(output_dir, "points", "dpt_unproj_full.ply")
        full_npy_path = os.path.join(output_dir, "points", "dpt_unproj_full.npy")
        save_ply_xyz(full_ply_path, full_pts)
        np.save(full_npy_path, full_pts)
        print(f"[SAVE] points/dpt_unproj_full.ply  ({full_pts.shape[0]} pts)")
        print(f"[SAVE] points/dpt_unproj_full.npy")

    # ------------------------------------------------------------------
    # 保存诊断报告
    # ------------------------------------------------------------------
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    report_path = os.path.join(diag_dir, "c0_report.json")

    # 把 numpy 类型转为 Python 原生类型，保证 JSON 序列化
    def _to_python(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    report_serializable = {k: _to_python(v) for k, v in report.items()}
    with open(report_path, "w") as f:
        json.dump(report_serializable, f, indent=2)
    print(f"[SAVE] diagnostics/c0_report.json")

    print(f"\n[DONE] Output written to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="C0 boundary sim/scale diagnostic – offline correction tool"
    )
    parser.add_argument(
        "--input", required=True,
        help="输入目录（含 poses/、depth/dpt/）",
    )
    parser.add_argument(
        "--output", required=True,
        help="输出目录（不能与 input 相同）",
    )
    parser.add_argument(
        "--gps", default=None,
        help="GPS XYZ npy 文件路径（用于 pose / geom_mean 模式）",
    )
    parser.add_argument(
        "--frame-start", type=int, default=0,
        help="第一帧对应全局 GPS 索引（默认 0）",
    )
    parser.add_argument(
        "--boundary", type=int, default=48,
        help="锚点帧的局部帧号（默认 48）",
    )
    parser.add_argument(
        "--mode",
        choices=["depth", "pose", "geom_mean", "manual"],
        default="depth",
        help="缩放估计模式（默认 depth）",
    )
    parser.add_argument(
        "--manual-scale", type=float, default=1.0,
        help="mode=manual 时使用的缩放因子",
    )
    parser.add_argument(
        "--max-frame-points", type=int, default=200_000,
        help="每帧点云最大点数（默认 200000）",
    )
    parser.add_argument(
        "--max-full-points", type=int, default=2_000_000,
        help="全局点云最大点数（默认 2000000）",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="允许覆写已有输出目录",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 防止 input == output
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("[ERROR] --input and --output must not be the same directory.")
        sys.exit(1)

    run(args)
