"""
longstream/core/pose_post_correction.py
----------------------------------------
GPS 分段 SE3 后处理：在非流式完整序列推理后，对 TTO 后模型输出位姿做分段 SE3 校正。

公开 API
--------
filter_gps_xyz(gps_xyz, *, enabled, method, ...) -> (np.ndarray, dict)
correct_poses_with_gps_segment_se3(extri_np, gps_xyz, ...) -> (np.ndarray, dict)
"""

from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  权重函数
# ═══════════════════════════════════════════════════════════════════════════

def _hann_weights(n: int) -> np.ndarray:
    """对称 Hann 窗权重，两端低、中间高，底部设 1e-3 下限确保首尾帧仍被校正。"""
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    w = np.hanning(n)  # 两端为 0，中间为 1
    return np.maximum(w, 1.0e-3)


def _linear_weights(n: int) -> np.ndarray:
    """梯形线性权重：首尾为 0.5，中间为 1。"""
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    half = max(1, n // 4)
    ramp = np.linspace(0.5, 1.0, half, endpoint=False)
    w[:half] = ramp
    w[n - half:] = ramp[::-1]
    return w


# ═══════════════════════════════════════════════════════════════════════════
#  内部工具
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_4x4(extri: np.ndarray) -> np.ndarray:
    """确保 extri 是 [S, 4, 4]。"""
    if extri.shape[-2:] == (4, 4):
        return extri
    S = extri.shape[0]
    out = np.zeros((S, 4, 4), dtype=extri.dtype)
    out[:, :3, :] = extri
    out[:, 3, 3] = 1.0
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  GPS 滤波工具函数
# ═══════════════════════════════════════════════════════════════════════════

def _make_odd(value: int, max_value: int) -> int:
    """将 value 限制在 [1, max_value] 并强制为奇数。"""
    value = int(value)
    value = max(1, min(value, max_value))
    if value % 2 == 0:
        value -= 1
    return max(1, value)


def _interp_invalid_xyz(xyz: np.ndarray):
    """
    对坐标序列中的 NaN/Inf 帧做线性插值修复。

    Returns (xyz_repaired, num_invalid)
    """
    xyz = np.asarray(xyz, dtype=np.float64).copy()
    valid = np.isfinite(xyz).all(axis=1)
    num_invalid = int((~valid).sum())

    if valid.all():
        return xyz, 0
    if valid.sum() == 0:
        return np.zeros_like(xyz), num_invalid

    idx = np.arange(len(xyz))
    for d in range(3):
        xyz[~valid, d] = np.interp(idx[~valid], idx[valid], xyz[valid, d])
    return xyz, num_invalid


def _repair_step_outliers(xyz: np.ndarray, max_step_m):
    """
    移除单帧异常大跳跃（阶跃异常值），用线性插值填回。

    Returns (xyz_repaired, num_outliers)
    """
    if max_step_m is None or max_step_m <= 0 or len(xyz) < 3:
        return xyz, 0

    xyz = xyz.copy()
    step = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
    outlier = np.zeros(len(xyz), dtype=bool)
    outlier[1:] = step > float(max_step_m)
    outlier[0] = False  # 首帧不判离群

    num_outliers = int(outlier.sum())
    if num_outliers == 0:
        return xyz, 0

    valid = ~outlier
    idx = np.arange(len(xyz))
    for d in range(3):
        xyz[outlier, d] = np.interp(idx[outlier], idx[valid], xyz[valid, d])
    return xyz, num_outliers


def _moving_average_xyz(xyz: np.ndarray, window: int) -> np.ndarray:
    """对 [N, 3] 轨迹逐维做滑动平均。"""
    window = max(1, int(window))
    if window <= 1:
        return xyz.copy()
    pad = window // 2
    padded = np.pad(xyz, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack([
        np.convolve(padded[:, d], kernel, mode="valid")[: len(xyz)]
        for d in range(3)
    ], axis=1)


def _kalman_cv_xyz(
    z: np.ndarray,
    *,
    process_noise: float = 1.0,
    measurement_noise: float = 4.0,
    dt: float = 1.0,
) -> np.ndarray:
    """
    匀速运动 Kalman 滤波（constant-velocity 模型）。

    状态向量 x = [px, py, pz, vx, vy, vz]^T。
    """
    n = len(z)
    if n == 0:
        return z.copy()

    x = np.zeros(6, dtype=np.float64)
    x[:3] = z[0]
    if n > 1:
        x[3:] = (z[1] - z[0]) / max(float(dt), 1e-6)

    F = np.eye(6, dtype=np.float64)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    H = np.zeros((3, 6), dtype=np.float64)
    H[:3, :3] = np.eye(3)

    Q = float(process_noise) * np.eye(6, dtype=np.float64)
    R = float(measurement_noise) * np.eye(3, dtype=np.float64)
    P = 100.0 * np.eye(6, dtype=np.float64)
    I6 = np.eye(6, dtype=np.float64)

    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        # 预测
        x = F @ x
        P = F @ P @ F.T + Q
        # 更新
        y = z[i] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (I6 - K @ H) @ P
        out[i] = x[:3]

    return out


def filter_gps_xyz(
    gps_xyz: np.ndarray,
    *,
    enabled: bool = False,
    method: str = "none",
    median_kernel: int = 5,
    savgol_window: int = 21,
    savgol_polyorder: int = 2,
    moving_average_window: int = 9,
    max_step_m = None,
    kalman_process_noise: float = 1.0,
    kalman_measurement_noise: float = 4.0,
    kalman_dt: float = 1.0,
):
    """
    可选 GPS 轨迹滤波。

    Parameters
    ----------
    gps_xyz : np.ndarray
        [N, 3] GPS 相机中心坐标。
    enabled : bool
        False 时直接返回原始数据副本（无任何修改）。
    method : str
        滤波方法：``"none"`` | ``"median"`` | ``"median_savgol"``
        | ``"moving_average"`` | ``"kalman_cv"``。

    Returns
    -------
    (filtered_gps, info_dict)
    """
    gps = np.asarray(gps_xyz, dtype=np.float64)
    info = {
        "enabled": bool(enabled),
        "method": method,
        "num_invalid_repaired": 0,
        "num_step_outliers_repaired": 0,
    }

    if not enabled or method in ("none", "", None):
        return gps.copy(), info

    # ── 预处理：插值 NaN/Inf，修复阶跃异常 ──────────────────────────────
    gps, n_invalid = _interp_invalid_xyz(gps)
    gps, n_outliers = _repair_step_outliers(gps, max_step_m)
    info["num_invalid_repaired"] = n_invalid
    info["num_step_outliers_repaired"] = n_outliers

    N = len(gps)
    out = gps.copy()

    if method == "median":
        try:
            from scipy.signal import medfilt
            k = _make_odd(median_kernel, N)
            out = np.stack([medfilt(gps[:, d], kernel_size=k) for d in range(3)], axis=1)
        except Exception:
            out = gps

    elif method == "median_savgol":
        # 先 median，再 Savitzky-Golay
        try:
            from scipy.signal import medfilt
            k = _make_odd(median_kernel, N)
            tmp = np.stack([medfilt(gps[:, d], kernel_size=k) for d in range(3)], axis=1)
        except Exception:
            tmp = gps
        try:
            from scipy.signal import savgol_filter
            w = _make_odd(savgol_window, N)
            p = min(int(savgol_polyorder), max(0, w - 2))
            if w >= p + 2 and w >= 3:
                out = savgol_filter(tmp, window_length=w, polyorder=p, axis=0, mode="interp")
            else:
                out = _moving_average_xyz(tmp, moving_average_window)
        except Exception:
            out = _moving_average_xyz(tmp, moving_average_window)

    elif method == "moving_average":
        out = _moving_average_xyz(gps, moving_average_window)

    elif method == "kalman_cv":
        out = _kalman_cv_xyz(
            gps,
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
            dt=kalman_dt,
        )

    else:
        print(f"[gps_filter] 未知方法 {method!r}，跳过滤波", flush=True)
        out = gps

    info.update({
        "median_kernel": median_kernel,
        "savgol_window": savgol_window,
        "savgol_polyorder": savgol_polyorder,
        "moving_average_window": moving_average_window,
        "max_step_m": max_step_m,
        "kalman_process_noise": kalman_process_noise,
        "kalman_measurement_noise": kalman_measurement_noise,
        "kalman_dt": kalman_dt,
    })
    return out, info


# ═══════════════════════════════════════════════════════════════════════════
#  核心函数
# ═══════════════════════════════════════════════════════════════════════════

def correct_poses_with_gps_segment_se3(
    extri_np: np.ndarray,
    gps_xyz: np.ndarray,
    *,
    segment_size: int = 160,
    overlap: int = 40,
    min_points: int = 8,
    blend: str = "hann",
    gps_filter_cfg = None,
) -> tuple:
    """
    GPS 分段 SE3 后处理。

    对推理输出的 w2c 位姿序列按段拟合 SE3（无尺度），
    再用 Hann 或线性权重在重叠区域融合校正结果，消除拼接缝。

    Parameters
    ----------
    extri_np : np.ndarray
        [S, 4, 4] 或 [S, 3, 4] world-to-camera 矩阵（float32/float64）。
    gps_xyz : np.ndarray
        [S, 3] GPS/相机中心参考坐标（世界坐标系）。
    segment_size : int
        每段帧数。
    overlap : int
        相邻段重叠帧数，用于混合消除拼接缝。
    min_points : int
        每段最少有效帧数，不足则跳过该段。
    blend : str
        混合权重策略：``"hann"`` 或 ``"linear"``。
    gps_filter_cfg : dict | None
        GPS 滤波配置，对应 YAML 中的 ``pose_post_correction.gps_filter``。
        None 或 {} 时等效于 enabled=false（不做滤波）。

    Returns
    -------
    corrected_extri : np.ndarray
        [S, 4, 4] float32，校正后的 w2c 位姿。
    metadata : dict
        包含 num_segments, num_valid_frames, mean_residual_before,
        mean_residual_after, segment_size, overlap, method, gps_filter 等字段。
    """
    from longstream.eval.metrics import similarity_align

    # ── 输入校验 ──────────────────────────────────────────────────────────
    extri_full = np.asarray(extri_np, dtype=np.float64)  # 保留完整副本以便后续追加尾帧
    gps_xyz = np.asarray(gps_xyz, dtype=np.float64)

    if extri_full.ndim != 3 or extri_full.shape[-1] != 4 or extri_full.shape[-2] not in (3, 4):
        raise ValueError(
            f"extri_np 应为 [S,3,4] 或 [S,4,4]，实际为 {extri_full.shape}"
        )
    if gps_xyz.ndim != 2 or gps_xyz.shape[1] != 3:
        raise ValueError(f"gps_xyz 应为 [S,3]，实际为 {gps_xyz.shape}")

    S_ext = extri_full.shape[0]
    S_gps = gps_xyz.shape[0]
    S = min(S_ext, S_gps)
    extri_np = extri_full[:S]   # 只在此局部副本上截断，extri_full 保持完整
    gps_xyz = gps_xyz[:S]

    # ── GPS 滤波（可选）──────────────────────────────────────────────────
    gps_filter_cfg = gps_filter_cfg or {}
    gps_used, gps_filter_info = filter_gps_xyz(gps_xyz, **gps_filter_cfg)

    # ── 提取预测相机中心 [S, 3] ──────────────────────────────────────────
    Rcw = extri_np[:, :3, :3]   # [S, 3, 3]
    tcw = extri_np[:, :3, 3]    # [S, 3]
    # C_pred_i = -Rcw_i^T @ tcw_i
    C_pred = -np.einsum("nij,nj->ni", np.transpose(Rcw, (0, 2, 1)), tcw)  # [S, 3]

    # ── 有效帧掩码（用于残差统计）────────────────────────────────────────
    valid_mask = np.isfinite(C_pred).all(axis=1) & np.isfinite(gps_used).all(axis=1)
    n_valid_total = int(valid_mask.sum())

    _skip_meta = {
        "num_segments": 0,
        "num_valid_frames": n_valid_total,
        "mean_residual_before": float("nan"),
        "mean_residual_after": float("nan"),
        "segment_size": segment_size,
        "overlap": overlap,
        "method": "segment_se3",
        "gps_filter": gps_filter_info,
    }

    if n_valid_total < min_points:
        extri_out = _ensure_4x4(extri_np).astype(np.float32)
        _skip_meta["note"] = "skipped: insufficient valid frames"
        return extri_out, _skip_meta

    resid_before = float(
        np.mean(np.linalg.norm(C_pred[valid_mask] - gps_used[valid_mask], axis=1))
    )

    # ── 分段索引 ─────────────────────────────────────────────────────────
    step = max(1, segment_size - overlap)
    last_start = max(0, S - segment_size)
    starts = list(range(0, last_start + 1, step))
    if last_start not in starts:
        starts.append(last_start)
    starts = sorted(set(starts))

    weight_func = _hann_weights if blend == "hann" else _linear_weights

    # frame_corrections[i] = [(weight, Q[3x3], u[3]) ...]
    frame_corrections: list = [[] for _ in range(S)]
    num_valid_segments = 0

    for start in starts:
        end = min(start + segment_size, S)
        seg_len = end - start
        if seg_len < min_points:
            continue

        src = C_pred[start:end]        # 预测中心
        dst = gps_used[start:end]      # GPS 参考（已滤波）
        seg_valid = (
            np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
        )
        if int(seg_valid.sum()) < min_points:
            continue

        # SE3 拟合：similarity_align with_scale=False
        _scale, Q, u = similarity_align(
            src[seg_valid], dst[seg_valid], with_scale=False
        )

        weights = weight_func(seg_len)
        for local_i, global_i in enumerate(range(start, end)):
            if seg_valid[local_i]:
                frame_corrections[global_i].append(
                    (float(weights[local_i]), Q.copy(), u.copy())
                )

        num_valid_segments += 1

    # ── 每帧融合 ─────────────────────────────────────────────────────────
    C_corr = C_pred.copy()
    Rcw_corr = Rcw.copy()
    corrected_count = 0

    for i in range(S):
        corrections = frame_corrections[i]
        if not corrections:
            continue

        total_w = sum(w for w, _Q, _u in corrections)
        if total_w <= 0.0:
            continue

        # 加权融合相机中心
        c_blend = np.zeros(3, dtype=np.float64)
        for w, Q_k, u_k in corrections:
            c_blend += (w / total_w) * (Q_k @ C_pred[i] + u_k)
        C_corr[i] = c_blend

        # 旋转：选权重最大的段的 Q 修正（避免旋转矩阵简单加权平均破坏正交性）
        best_Q = max(corrections, key=lambda x: x[0])[1]
        Rwc_corr_i = best_Q @ Rcw[i].T  # world←cam 旋转
        Rcw_corr[i] = Rwc_corr_i.T      # cam←world

        corrected_count += 1

    # ── 重建 tcw ─────────────────────────────────────────────────────────
    # tcw_i = -Rcw_i @ C_corr_i
    tcw_corr = -np.einsum("nij,nj->ni", Rcw_corr, C_corr)

    # ── 校正后残差 ───────────────────────────────────────────────────────
    resid_after = float(
        np.mean(np.linalg.norm(C_corr[valid_mask] - gps_used[valid_mask], axis=1))
    ) if corrected_count > 0 else resid_before

    # ── 构建输出 [S, 4, 4] ───────────────────────────────────────────────
    extri_out = np.zeros((S, 4, 4), dtype=np.float64)
    extri_out[:, :3, :3] = Rcw_corr
    extri_out[:, :3, 3] = tcw_corr
    extri_out[:, 3, 3] = 1.0

    # 如果原始输入比 GPS 长，把剩余帧从 extri_full 取并原样追加
    if S_ext > S:
        tail = _ensure_4x4(extri_full[S:S_ext].copy())
        extri_out = np.concatenate([extri_out, tail], axis=0)

    return extri_out.astype(np.float32), {
        "num_segments": num_valid_segments,
        "num_valid_frames": corrected_count,
        "mean_residual_before": resid_before,
        "mean_residual_after": resid_after,
        "segment_size": segment_size,
        "overlap": overlap,
        "method": "segment_se3",
        "gps_filter": gps_filter_info,
    }
