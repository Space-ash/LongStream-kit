"""
longstream/core/pose_post_correction.py
----------------------------------------
GPS 分段 SE3 后处理：在非流式完整序列推理后，对 TTO 后模型输出位姿做分段 SE3 校正。

公开 API
--------
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

    Returns
    -------
    corrected_extri : np.ndarray
        [S, 4, 4] float32，校正后的 w2c 位姿。
    metadata : dict
        包含 num_segments, num_valid_frames, mean_residual_before,
        mean_residual_after, segment_size, overlap, method 等字段。
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
    extri_np = extri_full[:S]   # 只在此局部副本上截断， extri_full 保持完整
    gps_xyz = gps_xyz[:S]

    # ── 提取预测相机中心 [S, 3] ──────────────────────────────────────────
    Rcw = extri_np[:, :3, :3]   # [S, 3, 3]
    tcw = extri_np[:, :3, 3]    # [S, 3]
    # C_pred_i = -Rcw_i^T @ tcw_i
    C_pred = -np.einsum("nij,nj->ni", np.transpose(Rcw, (0, 2, 1)), tcw)  # [S, 3]

    # ── 有效帧掩码（用于残差统计）────────────────────────────────────────
    valid_mask = np.isfinite(C_pred).all(axis=1) & np.isfinite(gps_xyz).all(axis=1)
    n_valid_total = int(valid_mask.sum())

    _skip_meta = {
        "num_segments": 0,
        "num_valid_frames": n_valid_total,
        "mean_residual_before": float("nan"),
        "mean_residual_after": float("nan"),
        "segment_size": segment_size,
        "overlap": overlap,
        "method": "segment_se3",
    }

    if n_valid_total < min_points:
        extri_out = _ensure_4x4(extri_np).astype(np.float32)
        _skip_meta["note"] = "skipped: insufficient valid frames"
        return extri_out, _skip_meta

    resid_before = float(
        np.mean(np.linalg.norm(C_pred[valid_mask] - gps_xyz[valid_mask], axis=1))
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

        src = C_pred[start:end]   # 预测中心
        dst = gps_xyz[start:end]  # GPS 参考
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
        np.mean(np.linalg.norm(C_corr[valid_mask] - gps_xyz[valid_mask], axis=1))
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
    }
