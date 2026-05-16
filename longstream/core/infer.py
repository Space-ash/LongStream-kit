import argparse
import json
import os
import random
import yaml
import cv2
import numpy as np
import torch
from PIL import Image

from longstream.core.model import LongStreamModel
from longstream.data.dataloader import LongStreamDataLoader
from longstream.streaming.keyframe_selector import KeyframeSelector
from longstream.streaming.refresh import run_batch_refresh, run_streaming_refresh
from longstream.utils.vendor.models.components.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from longstream.utils.vendor.models.components.utils.rotation import quat_to_mat
from longstream.utils.camera import compose_abs_from_rel
from longstream.utils.depth import colorize_depth, unproject_depth_to_points
from longstream.utils.sky_mask import compute_sky_mask
from longstream.io.save_points import save_pointcloud
from longstream.io.save_poses_txt import save_w2c_txt, save_intri_txt, save_rel_pose_txt
from longstream.io.save_images import save_image_sequence, save_video
from longstream.io.frame_index_map import save_frame_index_map
from longstream.core.pose_post_correction import correct_poses_with_gps_segment_se3
from longstream.utils.resource_monitor import CriticalOperationProfiler


def _to_uint8_rgb(images):
    imgs = images.detach().cpu().numpy()
    imgs = np.clip(imgs, 0.0, 1.0)
    imgs = (imgs * 255.0).astype(np.uint8)
    return imgs


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _apply_sky_mask(depth, mask):
    if mask is None:
        return depth
    m = (mask > 0).astype(np.float32)
    return depth * m


def _to_homogeneous_w2c(extri_np):
    """
    将预测位姿统一转换为齐次 4x4 w2c 矩阵序列。

    `pose_encoding_to_extri_intri` 当前返回的是 [S, 3, 4] 的 [R|t]，
    但后续 GT 校正、矩阵求逆、点云变换都假设使用 [S, 4, 4]。
    这里统一补最后一行 [0, 0, 0, 1]，避免下游混用 3x4 / 4x4。
    """
    extri_np = np.asarray(extri_np)
    if extri_np.ndim != 3:
        raise ValueError(
            f"Expected extrinsics with shape [S,3,4] or [S,4,4], got {extri_np.shape}"
        )
    if extri_np.shape[-2:] == (4, 4):
        return extri_np.astype(np.float32, copy=False)
    if extri_np.shape[-2:] != (3, 4):
        raise ValueError(
            f"Expected extrinsics with shape [S,3,4] or [S,4,4], got {extri_np.shape}"
        )

    S = extri_np.shape[0]
    homogeneous = np.zeros((S, 4, 4), dtype=extri_np.dtype)
    homogeneous[:, :3, :] = extri_np
    homogeneous[:, 3, 3] = 1.0
    return homogeneous.astype(np.float32, copy=False)


def _camera_points_to_world(points, extri):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(extri[:3, :3], dtype=np.float64)
    t = np.asarray(extri[:3, 3], dtype=np.float64)
    world = (R.T @ (pts.T - t[:, None])).T
    return world.astype(np.float32, copy=False)


def _mask_points_and_colors(points, colors, mask):
    pts = points.reshape(-1, 3)
    cols = None if colors is None else colors.reshape(-1, 3)
    if mask is None:
        return pts, cols
    valid = mask.reshape(-1) > 0
    pts = pts[valid]
    if cols is not None:
        cols = cols[valid]
    return pts, cols


def _combine_masks(mask_a, mask_b):
    """将两个掩码按位 AND 合并。None 表示全通（不限制）。"""
    if mask_a is None and mask_b is None:
        return None
    if mask_a is None:
        return (mask_b > 0).astype(np.uint8)
    if mask_b is None:
        return (mask_a > 0).astype(np.uint8)
    return ((mask_a > 0) & (mask_b > 0)).astype(np.uint8)


def _resize_long_edge(arr, long_edge_size, interpolation):
    h, w = arr.shape[:2]
    scale = float(long_edge_size) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(arr, (new_w, new_h), interpolation=interpolation)


def _prepare_mask_for_model(
    mask, size, crop, patch_size, target_shape, square_ok=False
):
    if mask is None:
        return None
    long_edge = (
        round(size * max(mask.shape[1] / mask.shape[0], mask.shape[0] / mask.shape[1]))
        if size == 224
        else size
    )
    mask = _resize_long_edge(mask, long_edge, cv2.INTER_NEAREST)

    h, w = mask.shape[:2]
    cx, cy = w // 2, h // 2
    if size == 224:
        half = min(cx, cy)
        target_w = 2 * half
        target_h = 2 * half
        if crop:
            mask = mask[cy - half : cy + half, cx - half : cx + half]
        else:
            mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
    else:
        halfw = ((2 * cx) // patch_size) * (patch_size // 2)
        halfh = ((2 * cy) // patch_size) * (patch_size // 2)
        if not square_ok and w == h:
            halfh = int(3 * halfw / 4)
        target_w = 2 * halfw
        target_h = 2 * halfh
        if crop:
            mask = mask[cy - halfh : cy + halfh, cx - halfw : cx + halfw]
        else:
            mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )

    if mask.shape[:2] != tuple(target_shape):
        mask = cv2.resize(
            mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask


def _save_full_pointcloud(path, point_chunks, color_chunks, max_points=None, seed=0):
    if not point_chunks:
        return
    points = np.concatenate(point_chunks, axis=0)
    colors = None
    if color_chunks and len(color_chunks) == len(point_chunks):
        colors = np.concatenate(color_chunks, axis=0)
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
    np.save(os.path.splitext(path)[0] + ".npy", points.astype(np.float32, copy=False))
    save_pointcloud(path, points, colors=colors, max_points=None, seed=seed)


def _decode_predicted_extri_intri(outputs, keyframe_indices, H, W):
    """统一解码预测位姿，返回 (extri_np, intri_np, rel_pose_enc_or_None)。"""
    rel_pose_enc = None
    if "rel_pose_enc" in outputs:
        rel_pose_enc = outputs["rel_pose_enc"][0]
        abs_pose_enc = compose_abs_from_rel(rel_pose_enc, keyframe_indices[0])
        extri, intri = pose_encoding_to_extri_intri(
            abs_pose_enc[None], image_size_hw=(H, W)
        )
    elif "pose_enc" in outputs:
        extri, intri = pose_encoding_to_extri_intri(
            outputs["pose_enc"][0][None], image_size_hw=(H, W)
        )
    else:
        return None, None, None
    extri_np = _to_homogeneous_w2c(extri[0].detach().cpu().numpy())
    intri_np = intri[0].detach().cpu().numpy()
    return extri_np, intri_np, rel_pose_enc


def _run_tto_pose_forward(
    model,
    images,
    is_keyframe,
    keyframe_indices,
    streaming_mode,
    rel_pose_num_iterations,
):
    """
    可微分位姿 dry-run：跳过稠密头，不更新 KV 缓存，
    专用于 TTO 梯度反传。所有输出均保留计算图（不调用 .detach()）。
    """
    rel_pose_inputs = {
        "is_keyframe": is_keyframe,
        "keyframe_indices": keyframe_indices,
        "num_iterations": rel_pose_num_iterations,
    }
    old_env = os.environ.get("SKIP_DENSE_HEADS")
    os.environ["SKIP_DENSE_HEADS"] = "1"
    try:
        outputs = model(
            images,
            mode=streaming_mode,
            is_keyframe=is_keyframe,
            rel_pose_inputs=rel_pose_inputs,
        )
    finally:
        if old_env is None:
            os.environ.pop("SKIP_DENSE_HEADS", None)
        else:
            os.environ["SKIP_DENSE_HEADS"] = old_env
    return outputs


def _compute_camera_centers_differentiable(
    pose_enc: torch.Tensor,
    keyframe_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    TTO 专用：从 pose_enc 全微分地计算相机中心。

    Args:
        pose_enc: [S, D]，前 3 小次元为平移 t，3:7 为单位四元数 q。
            - keyframe_indices 为 None 时单元局是绕世界坐标的绝对 pose。
            - keyframe_indices 不为 None 时单元局是相对 pose（rel_pose_enc）。
        keyframe_indices: [S]，每帧对应的参考帧内部索引。

    Returns:
        pred_centers: [S, 3]，可微分相机中心（世界坐标）。
    """
    if pose_enc.ndim != 2:
        raise ValueError(
            f"pose_enc 应为 2D 张量 [S, D]，实际得到 {pose_enc.shape}"
        )

    rel_t = pose_enc[:, :3]       # [S, 3]
    rel_q = pose_enc[:, 3:7]      # [S, 4]
    rel_R = quat_to_mat(rel_q)    # [S, 3, 3]

    if keyframe_indices is None:
        # 绝对 pose 直接使用
        abs_R = rel_R
        abs_t = rel_t
    else:
        if keyframe_indices.ndim != 1:
            raise ValueError(
                f"keyframe_indices 应为 1D 张量 [S]，实际得到 {keyframe_indices.shape}"
            )
        S = pose_enc.shape[0]
        R_list = [rel_R[0]]
        t_list = [rel_t[0]]
        # 优化：一次性将索引拷贝至 CPU 列表，避免循环内逐帧调用 .item() 强行同步 GPU。
        kf_indices_cpu = keyframe_indices.detach().cpu().to(torch.int32).tolist()
        for s in range(1, S):
            ref_idx = kf_indices_cpu[s]
            R_s = rel_R[s] @ R_list[ref_idx]
            t_s = rel_t[s] + rel_R[s] @ t_list[ref_idx]
            R_list.append(R_s)
            t_list.append(t_s)
        abs_R = torch.stack(R_list, dim=0)  # [S, 3, 3]
        abs_t = torch.stack(t_list, dim=0)  # [S, 3]

    # 相机中心 = -R^T @ t
    return -torch.einsum("sji,sj->si", abs_R, abs_t)


def _tto_build_window_starts(min_len, window_len, window_step, gps_tensor, min_gps_disp):
    """infer.py 内联 TTO 专用：构建覆盖优先的窗口起始列表。"""
    last_start = max(0, min_len - window_len)
    starts = list(range(0, last_start + 1, window_step))
    if 0 not in starts:
        starts.insert(0, 0)
    if last_start not in starts:
        starts.append(last_start)

    valid = []
    for s in starts:
        disp = torch.linalg.norm(gps_tensor[s + window_len - 1] - gps_tensor[s]).item()
        if torch.isfinite(torch.tensor(disp)) and disp > min_gps_disp:
            valid.append(s)

    for s in (0, last_start):
        if s not in valid:
            valid.insert(0 if s == 0 else len(valid), s)
    return list(dict.fromkeys(valid))


def _tto_select_starts(valid_starts, step, batch_windows, sampling):
    """infer.py 内联 TTO 专用：按采样策略选出本步窗口起始列表。"""
    n = max(1, batch_windows)
    if sampling == "deterministic_coverage":
        base = valid_starts[step] if step < len(valid_starts) else random.choice(valid_starts)
        starts = [base]
        while len(starts) < n:
            starts.append(random.choice(valid_starts))
        return starts
    if sampling == "endpoint_weighted":
        weights = [
            3.0 if s in (valid_starts[0], valid_starts[-1]) else 1.0
            for s in valid_starts
        ]
        return random.choices(valid_starts, weights=weights, k=n)
    return random.choices(valid_starts, k=n)


def _tto_multiscale_loss_inline(pred_centers, gps_tto, pair_strides, min_gps_disp, lambda_dir, lambda_endpoint):
    """infer.py 内联 TTO 专用：多尺度位移 loss。
    优化版：以浮点掩码代替 .any() 控制流，彻底消除 GPU-CPU 同步阻断。"""
    losses = []
    eps = 1.0e-6
    for stride in pair_strides:
        stride = max(1, min(int(stride), pred_centers.shape[1] - 1))
        pred_vec = pred_centers[:, stride:] - pred_centers[:, :-stride]
        gps_vec = gps_tto[:, stride:] - gps_tto[:, :-stride]
        pred_disp = torch.linalg.norm(pred_vec, dim=-1)
        gps_disp = torch.linalg.norm(gps_vec, dim=-1)

        # 以浮点掩码代替布尔索引，避免 valid.any() 触发 CPU/GPU 同步
        valid_mask = (
            torch.isfinite(pred_disp)
            & torch.isfinite(gps_disp)
            & (gps_disp > min_gps_disp)
        ).float()
        valid_sum = valid_mask.sum()
        has_valid = (valid_sum > 0).float()

        huber_disp = torch.nn.functional.huber_loss(
            pred_disp, gps_disp, reduction="none"
        )
        loss_disp = (huber_disp * valid_mask).sum() / valid_sum.clamp(min=1.0)
        losses.append(loss_disp * has_valid)

        if lambda_dir > 0:
            pred_dir = pred_vec / pred_disp.clamp_min(eps).unsqueeze(-1)
            gps_dir = gps_vec / gps_disp.clamp_min(eps).unsqueeze(-1)
            huber_dir = torch.nn.functional.huber_loss(
                pred_dir, gps_dir, reduction="none"
            )
            valid_dir_mask = valid_mask.unsqueeze(-1)
            loss_dir = (
                (huber_dir * valid_dir_mask).sum()
                / (valid_sum.clamp(min=1.0) * 3.0)
            )
            losses.append(lambda_dir * loss_dir * has_valid)

    if lambda_endpoint > 0:
        pred_ep = pred_centers[:, -1] - pred_centers[:, 0]
        gps_ep = gps_tto[:, -1] - gps_tto[:, 0]
        ep_disp = torch.linalg.norm(gps_ep, dim=-1)
        valid_ep_mask = (
            torch.isfinite(ep_disp) & (ep_disp > min_gps_disp)
        ).float()
        valid_ep_sum = valid_ep_mask.sum()
        has_valid_ep = (valid_ep_sum > 0).float()
        huber_ep = torch.nn.functional.huber_loss(
            pred_ep, gps_ep, reduction="none"
        )
        loss_ep = (huber_ep * valid_ep_mask.unsqueeze(-1)).sum() / (valid_ep_sum.clamp(min=1.0) * 3.0)
        losses.append(lambda_endpoint * loss_ep * has_valid_ep)

    if not losses:
        return None
    return torch.stack(losses).sum()


# ---------------------------------------------------------------------------
# Boundary-refresh TTO helpers
# ---------------------------------------------------------------------------

def _tto_build_boundary_indices(S, keyframe_stride, refresh):
    """构建 batch_refresh 新 batch 起点（接缝边界）索引列表。

    step_frames = (refresh-1)*keyframe_stride 是相邻拼接之间的有效帧数。
    boundaries = [step_frames, 2*step_frames, ...] in [0, S)。

    Returns:
        boundaries     : List[int]
        step_frames    : int
        frames_per_batch: int
    """
    step_frames = (refresh - 1) * keyframe_stride
    frames_per_batch = step_frames + 1
    boundaries = list(range(step_frames, S, step_frames))
    return boundaries, step_frames, frames_per_batch


def _run_tto_boundary_refresh_forward(
    model,
    images,              # [B, S, C, H, W]，可在 CPU 或 device 上
    device,
    selector,
    streaming_mode,
    rel_pose_num_iters,
    keyframe_stride,
    refresh,
    boundary,            # int：新 batch 起点（全局帧号）
):
    """对接缝两侧的 prev/next 局部 batch 独立前向推理，返回相机中心。

    两个 batch 分开 forward 并及时释放，不 cat，以控制显存。
    使用 SKIP_DENSE_HEADS=1（通过 _run_tto_pose_forward 保证）。

    Returns:
        dict 含 centers_prev [T_prev,3]、centers_next [T_next,3] 及元信息，
        或 None（帧数不足）。
    """
    S = images.shape[1]
    step_frames = (refresh - 1) * keyframe_stride
    frames_per_batch = step_frames + 1

    prev_start = max(0, boundary - step_frames)
    prev_end = min(prev_start + frames_per_batch, S)
    next_start = boundary
    next_end = min(next_start + frames_per_batch, S)

    T_prev = prev_end - prev_start
    T_next = next_end - next_start

    if T_prev < 2 or T_next < 2:
        return None

    # ---- prev local batch ----
    images_prev = images[:, prev_start:prev_end].to(device)
    B = images_prev.shape[0]
    is_kf_prev, kfi_prev = selector.select_keyframes(T_prev, B, device)
    outputs_prev = _run_tto_pose_forward(
        model, images_prev, is_kf_prev, kfi_prev, streaming_mode, rel_pose_num_iters
    )
    if "rel_pose_enc" in outputs_prev:
        centers_prev = _compute_camera_centers_differentiable(
            outputs_prev["rel_pose_enc"][0], kfi_prev[0]
        )
    elif "pose_enc" in outputs_prev:
        centers_prev = _compute_camera_centers_differentiable(
            outputs_prev["pose_enc"][0], keyframe_indices=None
        )
    else:
        del outputs_prev, images_prev
        return None
    del outputs_prev, images_prev

    # ---- next local batch ----
    images_next = images[:, next_start:next_end].to(device)
    is_kf_next, kfi_next = selector.select_keyframes(T_next, B, device)
    # 与 run_batch_refresh 一致：新 batch 第 0 帧强制为 keyframe
    is_kf_next[:, 0] = True
    kfi_next[:, 0] = 0
    outputs_next = _run_tto_pose_forward(
        model, images_next, is_kf_next, kfi_next, streaming_mode, rel_pose_num_iters
    )
    if "rel_pose_enc" in outputs_next:
        centers_next = _compute_camera_centers_differentiable(
            outputs_next["rel_pose_enc"][0], kfi_next[0]
        )
    elif "pose_enc" in outputs_next:
        centers_next = _compute_camera_centers_differentiable(
            outputs_next["pose_enc"][0], keyframe_indices=None
        )
    else:
        del outputs_next, images_next
        return None
    del outputs_next, images_next

    return {
        "centers_prev": centers_prev,    # [T_prev, 3]
        "centers_next": centers_next,    # [T_next, 3]
        "prev_start": prev_start,
        "next_start": next_start,
        "step_frames": step_frames,
        "T_prev": T_prev,
        "T_next": T_next,
    }


def _tto_boundary_refresh_loss_inline(
    fwd_result,
    gps_tensor,           # [S, 3] on device
    boundary,             # int：新 batch 起点（全局帧号 b）
    min_gps_disp,         # float：boundary 专用 GPS 位移阈值
    duplicate_weight,     # float：同帧一致性损失权重
):
    """计算 batch_refresh 接缝处的诊断 loss。

    b = boundary（新 batch 起点）。真实拼接为 prev 保留 global b，
    next 丢弃 local 0，保留 global b+1，断点是 b -> b+1。

    pair 说明：
      1. exact seam  : global b   -> (b+1)  [真实拼接断点，主 pair，用于日志]
      2. bridge pair : global (b-1) -> (b+1) [跨接缝桥接，辅助约束]
      3. next-start  : global (b+1) -> (b+2) [新 batch 起点局部尺度]
      4. prev-end    : global (b-1) -> b      [旧 batch 末端一跳参照]
      5. duplicate   : prev local b == next local 0 [同帧两上下文一致性]

    每个有效 pair 的 loss = huber_vec + 0.5 * huber_disp。

    Returns:
        (loss_tensor_or_None, info_dict)
        info_dict 含 seam_pred/seam_gps（exact seam）、bridge_pred/bridge_gps、dup。
    """
    F_nn = torch.nn.functional
    centers_prev = fwd_result["centers_prev"]   # [T_prev, 3]
    centers_next = fwd_result["centers_next"]   # [T_next, 3]
    prev_start = fwd_result["prev_start"]
    T_prev = fwd_result["T_prev"]
    T_next = fwd_result["T_next"]
    S_gps = gps_tensor.shape[0]

    losses = []
    info = {}

    # local_dup_prev = index of global frame b in prev batch coords
    local_dup_prev = boundary - prev_start      # global b in prev coords
    local_bridge_prev = local_dup_prev - 1      # global b-1 in prev coords

    # -- 1. Exact seam: prev local b -> next local 1 ; global b -> (b+1) --
    if (
        0 <= local_dup_prev < T_prev
        and T_next > 1
        and boundary + 1 < S_gps
    ):
        gps_exact = gps_tensor[boundary + 1] - gps_tensor[boundary]
        gps_exact_disp = torch.linalg.norm(gps_exact).item()
        if gps_exact_disp > min_gps_disp:
            pred_exact = centers_next[1] - centers_prev[local_dup_prev]
            loss_vec = F_nn.huber_loss(pred_exact, gps_exact)
            loss_disp = F_nn.huber_loss(
                torch.linalg.norm(pred_exact),
                torch.linalg.norm(gps_exact).detach(),
            )
            losses.append(loss_vec + 0.5 * loss_disp)
            info["seam_pred"] = torch.linalg.norm(pred_exact).detach().item()
            info["seam_gps"] = gps_exact_disp

    # -- 2. Bridge pair: prev local (b-1) -> next local 1 ; global (b-1) -> (b+1) --
    if (
        0 <= local_bridge_prev < T_prev
        and T_next > 1
        and boundary - 1 >= 0
        and boundary + 1 < S_gps
    ):
        gps_bridge = gps_tensor[boundary + 1] - gps_tensor[boundary - 1]
        gps_bridge_disp = torch.linalg.norm(gps_bridge).item()
        if gps_bridge_disp > min_gps_disp:
            pred_bridge = centers_next[1] - centers_prev[local_bridge_prev]
            losses.append(
                F_nn.huber_loss(pred_bridge, gps_bridge)
                + 0.5 * F_nn.huber_loss(
                    torch.linalg.norm(pred_bridge),
                    torch.linalg.norm(gps_bridge).detach(),
                )
            )
            info["bridge_pred"] = torch.linalg.norm(pred_bridge).detach().item()
            info["bridge_gps"] = gps_bridge_disp

    # -- 3. Next-start pair: next local 1->2 ; global (b+1)->(b+2) --
    if T_next > 2 and boundary + 2 < S_gps:
        gps_nb = gps_tensor[boundary + 2] - gps_tensor[boundary + 1]
        gps_nb_disp = torch.linalg.norm(gps_nb).item()
        if gps_nb_disp > min_gps_disp:
            pred_nb = centers_next[2] - centers_next[1]
            losses.append(
                F_nn.huber_loss(pred_nb, gps_nb)
                + 0.5 * F_nn.huber_loss(
                    torch.linalg.norm(pred_nb),
                    torch.linalg.norm(gps_nb).detach(),
                )
            )

    # -- 4. Prev-end pair: prev local (b-1)->b ; global (b-1)->b --
    if (
        0 <= local_bridge_prev
        and local_dup_prev < T_prev
        and boundary - 1 >= 0
        and boundary < S_gps
    ):
        gps_pe = gps_tensor[boundary] - gps_tensor[boundary - 1]
        gps_pe_disp = torch.linalg.norm(gps_pe).item()
        if gps_pe_disp > min_gps_disp:
            pred_pe = centers_prev[local_dup_prev] - centers_prev[local_bridge_prev]
            losses.append(
                F_nn.huber_loss(pred_pe, gps_pe)
                + 0.5 * F_nn.huber_loss(
                    torch.linalg.norm(pred_pe),
                    torch.linalg.norm(gps_pe).detach(),
                )
            )

    # -- 5. Duplicate consistency: same global frame b in prev/next --
    if 0 <= local_dup_prev < T_prev:
        loss_dup = F_nn.huber_loss(centers_prev[local_dup_prev], centers_next[0])
        losses.append(duplicate_weight * loss_dup)
        info["dup"] = torch.linalg.norm(
            (centers_prev[local_dup_prev] - centers_next[0]).detach()
        ).item()

    if not losses:
        return None, info

    return torch.stack(losses).sum(), info


# ---------------------------------------------------------------------------
# Overlap-stitching helpers (C1 方案)
# ---------------------------------------------------------------------------

def _center_from_extri(extri_4x4):
    """从 4x4 w2c 矩阵提取相机世界坐标中心 [3]"""
    R = extri_4x4[:3, :3]
    t = extri_4x4[:3, 3]
    return -R.T @ t


def _align_centers_fixed_scale(src, dst, scale):
    """
    Kabsch 对齐：dst ≈ scale * R @ src + t

    src: [N, 3] new overlap centers
    dst: [N, 3] stitched old overlap centers
    Returns: R [3,3], t [3]
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = dst_c.T @ (scale * src_c)
    U, _, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = dst_mean - scale * (R @ src_mean)
    return R, t


def _apply_sim3_to_w2c(extri_local, scale, R_align, t_align):
    """
    对 [N, 4, 4] w2c 矩阵应用 Sim3(scale, R_align, t_align) 变换。

    世界坐标变换：C_new = scale * R_align @ C_old + t_align
    对应旋转变换：Rcw_new = (R_align @ Rwc_old)^T = Rwc_old^T @ R_align^T
    但更直接写法见下方 einsum。
    """
    extri_local = np.asarray(extri_local, dtype=np.float64)
    Rcw = extri_local[:, :3, :3]           # [N, 3, 3]
    tcw = extri_local[:, :3, 3]            # [N, 3]
    C   = -np.einsum("nij,nj->ni", Rcw.transpose(0, 2, 1), tcw)  # [N, 3]
    C2  = scale * (R_align @ C.T).T + t_align                     # [N, 3]
    Rwc  = Rcw.transpose(0, 2, 1)                                  # [N, 3, 3]
    Rwc2 = np.einsum("ij,njk->nik", R_align, Rwc)                 # [N, 3, 3]
    Rcw2 = Rwc2.transpose(0, 2, 1)                                 # [N, 3, 3]
    t2   = -np.einsum("nij,nj->ni", Rcw2, C2)                     # [N, 3]
    out  = np.zeros_like(extri_local)
    out[:, :3, :3] = Rcw2
    out[:, :3, 3]  = t2
    out[:, 3, 3]   = 1.0
    return out.astype(np.float32)


def _run_batch_refresh_overlap_export(
    model,
    images,
    selector,
    streaming_mode,
    keyframe_stride,
    refresh,
    rel_pose_cfg,
    H,
    W,
    stitch_cfg,
    device,
):
    """
    Overlap batch refresh 推理路径（C1 方案）。

    单 batch 仍是 frames_per_batch=(refresh-1)*keyframe_stride+1 帧（不增加显存），
    但相邻 batch 步长缩短为 step_frames = frames_per_batch - overlap_frames，
    使相邻 batch 有 overlap_frames 帧重叠。
    通过深度中值比 + Kabsch 算法估计 batch 间 Sim3 并拼接。

    Returns dict:
        depth        : torch.Tensor [1, S, H, W, 1] on CPU
        depth_conf   : torch.Tensor [1, S, H, W, 1] or None
        extri_np     : np.ndarray [S, 4, 4]
        intri_np     : np.ndarray [S, 3, 3]
        stitch_report: list[dict]
    """
    B, S = images.shape[:2]

    overlap_frames  = int(stitch_cfg.get("overlap_frames", 9))
    scale_min       = float(stitch_cfg.get("scale_min", 0.25))
    scale_max       = float(stitch_cfg.get("scale_max", 1.5))
    min_ov_frames   = int(stitch_cfg.get("min_overlap_frames", 4))
    blend_overlap   = str(stitch_cfg.get("blend_overlap", "keep_old"))

    refresh_intervals = refresh - 1
    frames_per_batch  = refresh_intervals * keyframe_stride + 1
    step_frames       = frames_per_batch - overlap_frames

    if step_frames < 1:
        raise ValueError(
            f"overlap_frames={overlap_frames} must be < "
            f"frames_per_batch={frames_per_batch}"
        )

    # ---------- batch start 列表 ----------
    # 先确定 last_start，再从 0 到 last_start（包含）生成，避免 step 赶过 last_start 后
    # 出现尾部多个无新帧的冠余 batch。
    last_start = max(0, S - frames_per_batch)
    starts = list(range(0, last_start + 1, step_frames))
    if starts[-1] != last_start:
        starts.append(last_start)
    starts = sorted(set(starts))

    # ---------- stitched 缓冲 ----------
    stitched_extri      = [None] * S
    stitched_intri      = [None] * S
    stitched_depth      = np.zeros((S, H, W), dtype=np.float32)
    stitched_depth_conf = [None] * S
    filled              = np.zeros(S, dtype=bool)

    rel_pose_num_iters = rel_pose_cfg.get("num_iterations", 4) if rel_pose_cfg else 4
    stitch_report: list = []

    dev_is_cuda = (
        (isinstance(device, str) and "cuda" in device)
        or (isinstance(device, torch.device) and device.type == "cuda")
    )

    for batch_idx, start in enumerate(starts):
        end = min(start + frames_per_batch, S)
        T   = end - start

        batch_images = images[:, start:end].to(device)
        is_kf, kfi   = selector.select_keyframes(T, B, device)

        if batch_idx > 0:
            is_kf[:, 0] = True
            kfi[:, 0]   = 0

        rel_pose_inputs = None
        if rel_pose_cfg is not None:
            rel_pose_inputs = {
                "is_keyframe":      is_kf,
                "keyframe_indices": kfi,
                "num_iterations":   rel_pose_num_iters,
            }

        with CriticalOperationProfiler(
            f"overlap_batch_refresh_forward_batch_{batch_idx}"
        ):
            batch_outputs = model(
                images=batch_images,
                mode=streaming_mode,
                rel_pose_inputs=rel_pose_inputs,
                is_keyframe=is_kf,
            )
        del batch_images

        # ---------- decode local pose ----------
        extri_local, intri_local, _ = _decode_predicted_extri_intri(
            batch_outputs, kfi, H, W
        )
        if extri_local is None or "depth" not in batch_outputs:
            print(
                f"[overlap-stitch] batch={batch_idx} start={start}: "
                "pose/depth decode failed, skipping",
                flush=True,
            )
            del batch_outputs
            if dev_is_cuda:
                torch.cuda.empty_cache()
            continue

        depth_local = (
            batch_outputs["depth"][0, :, :, :, 0].detach().cpu().numpy()
        )  # [T, H, W]

        depth_conf_local = None
        _raw_dc = batch_outputs.get("depth_conf")
        if _raw_dc is not None:
            try:
                dc_arr = _raw_dc[0].detach().cpu().numpy()
                if dc_arr.ndim == 4:
                    dc_arr = dc_arr[..., 0]
                depth_conf_local = dc_arr  # [T, H, W]
            except Exception:
                depth_conf_local = None

        del batch_outputs
        if dev_is_cuda:
            torch.cuda.empty_cache()

        global_indices = np.arange(start, end)

        # -------- 第 0 批：直接写入 --------
        if batch_idx == 0:
            for li, g in enumerate(global_indices):
                stitched_extri[g]  = extri_local[li]
                stitched_intri[g]  = intri_local[li]
                stitched_depth[g]  = depth_local[li]
                if depth_conf_local is not None:
                    stitched_depth_conf[g] = depth_conf_local[li]
                filled[g] = True
            stitch_report.append({
                "batch_idx": 0,
                "start": int(start),
                "end": int(end),
                "overlap_frames": [],
                "new_only_start": int(start),
                "s_depth": None,
                "scale_used": None,
                "pose_rmse_before": None,
                "pose_rmse_after": None,
                "depth_ratio_median_before": None,
                "depth_ratio_median_after": None,
                "fallback": False,
            })
            print(
                f"[overlap-stitch] batch=0 start={start} end={end}"
                " (first batch, direct write)",
                flush=True,
            )
            continue

        # -------- overlap 对齐 --------
        overlap_global   = [int(g) for g in global_indices if filled[g]]
        new_only_global  = [int(g) for g in global_indices if not filled[g]]
        local_ov_indices = [int(g - start) for g in overlap_global]

        fallback = len(overlap_global) < min_ov_frames
        report: dict = {
            "batch_idx": batch_idx,
            "start": int(start),
            "end": int(end),
            "overlap_frames": overlap_global,
            "new_only_start": int(new_only_global[0]) if new_only_global else int(end),
            "s_depth": None,
            "scale_used": None,
            "pose_rmse_before": None,
            "pose_rmse_after": None,
            "depth_ratio_median_before": None,
            "depth_ratio_median_after": None,
            "fallback": fallback,
        }

        if fallback:
            print(
                f"[overlap-stitch] batch={batch_idx} start={start} "
                f"overlap={len(overlap_global)} < min={min_ov_frames},"
                " fallback (hard append)",
                flush=True,
            )
            for li, g in enumerate(global_indices):
                if not filled[g]:
                    stitched_extri[g]  = extri_local[li]
                    stitched_intri[g]  = intri_local[li]
                    stitched_depth[g]  = depth_local[li]
                    if depth_conf_local is not None:
                        stitched_depth_conf[g] = depth_conf_local[li]
                    filled[g] = True
            stitch_report.append(report)
            continue

        # -------- 深度尺度：robust median --------
        def _median_depth(d):
            pos = d[d > 0]
            return float(np.median(pos)) if pos.size > 0 else float(np.median(d))

        old_meds = np.array([_median_depth(stitched_depth[g]) for g in overlap_global])
        new_meds = np.array([_median_depth(depth_local[li]) for li in local_ov_indices])
        valid    = new_meds > 1e-6
        if valid.sum() < 2:
            s_depth = 1.0
        else:
            ratios  = old_meds[valid] / new_meds[valid]
            s_depth = float(np.median(ratios))
        s_depth = float(np.clip(s_depth, scale_min, scale_max))

        # -------- Kabsch 位姿对齐 --------
        C_old = np.array(
            [_center_from_extri(stitched_extri[g]) for g in overlap_global],
            dtype=np.float64,
        )
        C_new = np.array(
            [_center_from_extri(extri_local[li]) for li in local_ov_indices],
            dtype=np.float64,
        )

        pose_rmse_before = float(np.sqrt(np.mean(
            np.sum((C_old - s_depth * C_new) ** 2, axis=-1)
        )))
        depth_ratio_before = (
            float(np.median(old_meds[valid] / new_meds[valid])) if valid.any() else 1.0
        )

        R_align, t_align   = _align_centers_fixed_scale(C_new, C_old, s_depth)
        extri_aligned      = _apply_sim3_to_w2c(extri_local, s_depth, R_align, t_align)
        depth_local_scaled = depth_local * s_depth

        C_new_after = np.array(
            [_center_from_extri(extri_aligned[li]) for li in local_ov_indices],
            dtype=np.float64,
        )
        pose_rmse_after = float(np.sqrt(np.mean(
            np.sum((C_old - C_new_after) ** 2, axis=-1)
        )))

        # depth_ratio_median_after: median(old / (new * s_depth))，理论上应接近 1.0
        depth_ratio_after = (
            float(np.median(old_meds[valid] / (new_meds[valid] * s_depth)))
            if valid.any() else 1.0
        )
        report.update({
            "s_depth":                   s_depth,
            "scale_used":                s_depth,
            "pose_rmse_before":          pose_rmse_before,
            "pose_rmse_after":           pose_rmse_after,
            "depth_ratio_median_before": depth_ratio_before,
            "depth_ratio_median_after":  depth_ratio_after,
        })

        # -------- 写入帧 --------
        for li, g in enumerate(global_indices):
            if filled[g] and blend_overlap == "keep_old":
                continue
            stitched_extri[g]  = extri_aligned[li]
            stitched_intri[g]  = intri_local[li]
            stitched_depth[g]  = depth_local_scaled[li]
            if depth_conf_local is not None:
                stitched_depth_conf[g] = depth_conf_local[li]
            filled[g] = True

        stitch_report.append(report)
        print(
            f"[overlap-stitch] batch={batch_idx} start={start}"
            f" overlap={len(overlap_global)} s_depth={s_depth:.4f}"
            f" pose_rmse_before={pose_rmse_before:.4f}"
            f" pose_rmse_after={pose_rmse_after:.4f}",
            flush=True,
        )

    # ---------- 完整性检查 ----------
    unfilled = np.where(~filled)[0]
    if len(unfilled) > 0:
        print(
            f"[overlap-stitch] WARNING: {len(unfilled)} frames unfilled"
            f" (first 10: {unfilled[:10].tolist()})",
            flush=True,
        )

    # ---------- 组装输出 ----------
    extri_out    = np.stack(stitched_extri, axis=0)  # [S, 4, 4]
    intri_out    = np.stack(stitched_intri, axis=0)  # [S, 3, 3]
    depth_tensor = (
        torch.from_numpy(stitched_depth).unsqueeze(0).unsqueeze(-1)
    )  # [1, S, H, W, 1]

    depth_conf_tensor = None
    if any(x is not None for x in stitched_depth_conf):
        try:
            dc_arr = np.stack(
                [
                    x if x is not None else np.zeros((H, W), dtype=np.float32)
                    for x in stitched_depth_conf
                ],
                axis=0,
            )  # [S, H, W]
            depth_conf_tensor = (
                torch.from_numpy(dc_arr).unsqueeze(0).unsqueeze(-1)
            )  # [1, S, H, W, 1]
        except Exception:
            depth_conf_tensor = None

    return {
        "depth":          depth_tensor,
        "depth_conf":     depth_conf_tensor,
        "extri_np":       extri_out,
        "intri_np":       intri_out,
        "stitch_report":  stitch_report,
    }


def run_inference_cfg(cfg: dict):
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device_type = torch.device(device).type
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    infer_cfg = cfg.get("inference", {})
    output_cfg = cfg.get("output", {})

    # --- optimizations 配置 ---
    opt_cfg = cfg.get("optimizations", {})
    filter_cfg = opt_cfg.get("filter", {})
    corr_cfg = opt_cfg.get("correction", {})
    pose_post_cfg = opt_cfg.get("pose_post_correction", {})
    pose_post_enabled = bool(pose_post_cfg.get("enabled", False))
    frame_filter_enabled = bool(
        filter_cfg.get("frame_filter_enabled", filter_cfg.get("enabled", False))
    )
    confidence_filter_enabled = bool(
        filter_cfg.get(
            "confidence_filter_enabled",
            filter_cfg.get("enabled", False),
        )
    )
    confidence_threshold = float(filter_cfg.get("confidence_threshold", 0.5))

    print(f"[longstream] device={device}", flush=True)
    model = LongStreamModel(model_cfg).to(device)
    model.eval()
    print("[longstream] model ready", flush=True)

    # --- TTO (Test-Time Optimization) 配置 ---
    tto_enabled = bool(corr_cfg.get("tto_enabled", False))
    tto_steps = int(corr_cfg.get("tto_steps", 20))
    tto_lr = float(corr_cfg.get("tto_lr", 1e-3))
    tto_weight_decay = float(corr_cfg.get("tto_weight_decay", 0.0))
    tto_window_size = int(corr_cfg.get("tto_window_size", 40))
    tto_min_gps_disp = float(corr_cfg.get("tto_min_gps_disp", 1.0))
    tto_max_grad_norm = float(corr_cfg.get("tto_max_grad_norm", 1.0))
    # 新增 TTO 配置项
    tto_sampling = str(corr_cfg.get("tto_sampling", "random"))
    _raw_strides = corr_cfg.get("tto_pair_strides", None)
    if _raw_strides is None:
        _old = corr_cfg.get("tto_pair_stride", None)
        _raw_strides = [_old] if _old is not None else [8]
    if isinstance(_raw_strides, (int, float)):
        _raw_strides = [_raw_strides]
    tto_pair_strides = [int(s) for s in _raw_strides if int(s) > 0] or [8]
    tto_lambda_dir = float(corr_cfg.get("tto_lambda_dir", 0.0))
    tto_lambda_endpoint = float(corr_cfg.get("tto_lambda_endpoint", 0.0))
    tto_early_stop_patience = int(corr_cfg.get("tto_early_stop_patience", 0))
    tto_min_delta = float(corr_cfg.get("tto_min_delta", 1.0e-3))
    tto_batch_windows = int(corr_cfg.get("tto_batch_windows", 1))
    # Boundary-refresh TTO diagnostic loss 配置
    tto_boundary_enabled = bool(corr_cfg.get("tto_boundary_enabled", False))
    tto_boundary_weight = float(corr_cfg.get("tto_boundary_weight", 1.0))
    tto_boundary_min_gps_disp = float(corr_cfg.get("tto_boundary_min_gps_disp", 0.005))
    tto_boundary_batch_windows = int(corr_cfg.get("tto_boundary_batch_windows", 1))
    tto_boundary_log = bool(corr_cfg.get("tto_boundary_log", True))
    tto_boundary_duplicate_weight = float(corr_cfg.get("tto_boundary_duplicate_weight", 0.2))
    initial_scale_token = None
    if tto_enabled:
        if not hasattr(model.longstream, "scale_token"):
            raise RuntimeError(
                "[TTO] model.longstream 没有 scale_token 参数。"
                "请在模型配置中设置 enable_scale_token: true。"
            )
        if not getattr(model.longstream, "enable_scale_token", False):
            raise RuntimeError(
                "[TTO] model.longstream.enable_scale_token 为 False。"
                "请在模型配置中设置 enable_scale_token: true。"
            )
        for param in model.parameters():
            param.requires_grad = False
        model.longstream.scale_token.requires_grad = True
        initial_scale_token = model.longstream.scale_token.detach().clone()
        print("[longstream][TTO] 已启用：每条序列将在推理前优化 scale_token", flush=True)

    # 将帧质量过滤配置合并入 data_cfg传给 LongStreamDataLoader
    data_cfg_with_filter = dict(data_cfg)
    if filter_cfg:
        data_cfg_with_filter["filter"] = filter_cfg
    loader = LongStreamDataLoader(data_cfg_with_filter)

    keyframe_stride = int(infer_cfg.get("keyframe_stride", 8))
    keyframe_mode = infer_cfg.get("keyframe_mode", "fixed")
    refresh = int(
        infer_cfg.get("refresh", int(infer_cfg.get("keyframes_per_batch", 3)) + 1)
    )
    if refresh < 2:
        raise ValueError(
            "refresh must be >= 2 because it counts both keyframe endpoints"
        )
    mode = infer_cfg.get("mode", "streaming_refresh")
    if mode == "streaming":
        mode = "streaming_refresh"
    streaming_mode = infer_cfg.get("streaming_mode", "causal")
    window_size = int(infer_cfg.get("window_size", 5))
    stitch_cfg = dict(infer_cfg.get("stitch", {}))
    # 允许将 inference.overlap_frames 作为顶层配置传入，
    # stitch 子段的同名字段优先级更高。
    stitch_cfg.setdefault("overlap_frames", infer_cfg.get("overlap_frames", 9))

    selector = KeyframeSelector(
        min_interval=keyframe_stride,
        max_interval=keyframe_stride,
        force_first=True,
        mode="random" if keyframe_mode == "random" else "fixed",
    )

    out_root = output_cfg.get("root", "outputs")
    _ensure_dir(out_root)
    save_videos = bool(output_cfg.get("save_videos", True))
    save_points = bool(output_cfg.get("save_points", True))
    save_frame_points = bool(output_cfg.get("save_frame_points", True))
    # 分支开关：缺少时均回退到 save_points 行为（向后兼容旧 YAML）
    save_point_head = bool(output_cfg.get("save_point_head", save_points))
    save_dpt_unproj = bool(output_cfg.get("save_dpt_unproj", save_points))
    save_depth = bool(output_cfg.get("save_depth", True))
    save_images = bool(output_cfg.get("save_images", True))
    # 天空过滤：enable_sky_mask 控制过滤开关，save_sky_mask 控制 PNG 写盘；兼容旧字段 mask_sky
    enable_sky_mask = bool(
        output_cfg.get("enable_sky_mask", output_cfg.get("mask_sky", True))
    )
    save_sky_mask = bool(
        output_cfg.get("save_sky_mask", output_cfg.get("mask_sky", False))
    )
    max_full_pointcloud_points = output_cfg.get("max_full_pointcloud_points", None)
    if max_full_pointcloud_points is not None:
        max_full_pointcloud_points = int(max_full_pointcloud_points)
    max_frame_pointcloud_points = output_cfg.get("max_frame_pointcloud_points", None)
    if max_frame_pointcloud_points is not None:
        max_frame_pointcloud_points = int(max_frame_pointcloud_points)
    skyseg_path = output_cfg.get(
        "skyseg_path",
        os.path.join(os.path.dirname(__file__), "..", "..", "skyseg.onnx"),
    )

    with torch.no_grad():
        for seq in loader:
            images = seq.images
            B, S, C, H, W = images.shape
            print(
                f"[longstream] sequence {seq.name}: inference start ({S} frames)",
                flush=True,
            )

            is_keyframe, keyframe_indices = selector.select_keyframes(
                S, B, images.device
            )

            rel_pose_cfg = infer_cfg.get("rel_pose_head_cfg", {"num_iterations": 4})

            # --- TTO: 使用 GPS 位移监督在最终推理前优化 scale_token ---
            has_gps = getattr(seq, "gps_xyz", None) is not None
            # 无论本序列是否有 GPS，只要 TTO 已启用就先重置 token，
            # 避免有 GPS 序列的优化结果污染后续无 GPS 序列。
            if tto_enabled:
                model.longstream.scale_token.data.copy_(initial_scale_token)

            outputs_tto = None  # 防止 tto_steps=0 或跳过时 del 报 NameError
            if tto_enabled and has_gps:
                tto_seq_optimizer = torch.optim.AdamW(
                    [model.longstream.scale_token],
                    lr=tto_lr,
                    weight_decay=tto_weight_decay,
                )
                gps_tensor = torch.as_tensor(
                    seq.gps_xyz, device=device, dtype=torch.float32
                )
                min_len = min(S, gps_tensor.shape[0])
                gps_tensor = gps_tensor[:min_len]

                # 计算窗口长度，防止单次前向处理全序列触发 OOM
                window_len = min(tto_window_size, min_len)
                tto_skipped = False

                if window_len < 2:
                    print(
                        f"[longstream][TTO] seq={seq.name}"
                        f" window_len={window_len} < 2，跳过 TTO",
                        flush=True,
                    )
                    tto_skipped = True

                if not tto_skipped:
                    window_step = max(1, window_len // 2)
                    valid_start_indices = _tto_build_window_starts(
                        min_len, window_len, window_step, gps_tensor, tto_min_gps_disp
                    )

                    if not valid_start_indices:
                        print(
                            f"[longstream][TTO] seq={seq.name}"
                            f" no valid GPS windows (min_gps_disp={tto_min_gps_disp})，跳过 TTO",
                            flush=True,
                        )
                        tto_skipped = True

                if not tto_skipped:
                    rel_pose_num_iters = rel_pose_cfg.get("num_iterations", 4)
                    _pair_strides = [max(1, min(s, window_len - 1)) for s in tto_pair_strides]
                    print(
                        f"[longstream][TTO] seq={seq.name} steps={tto_steps}"
                        f" lr={tto_lr} window_len={window_len}"
                        f" pair_strides={_pair_strides} sampling={tto_sampling}"
                        f" batch_windows={tto_batch_windows}"
                        f" valid_windows={len(valid_start_indices)}",
                        flush=True,
                    )
                    # 构建 batch_refresh 接缝边界索引
                    _boundary_indices = []
                    if tto_boundary_enabled and mode == "batch_refresh":
                        _boundary_indices, _step_frames_bnd, _ = (
                            _tto_build_boundary_indices(min_len, keyframe_stride, refresh)
                        )
                        if _boundary_indices:
                            print(
                                f"[longstream][TTO-boundary] seq={seq.name}"
                                f" boundaries={_boundary_indices}"
                                f" step_frames={_step_frames_bnd}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[longstream][TTO-boundary] seq={seq.name}"
                                f" no boundaries found (min_len={min_len},"
                                f" step_frames={_step_frames_bnd}), disabled",
                                flush=True,
                            )
                    model.eval()
                    best_loss = float("inf")
                    bad_steps = 0
                    with torch.enable_grad():
                        for step in range(tto_steps):
                            tto_seq_optimizer.zero_grad(set_to_none=True)

                            starts = _tto_select_starts(
                                valid_start_indices, step, tto_batch_windows, tto_sampling
                            )

                            # 多窗口 batch
                            images_tto = torch.cat(
                                [images[:, s : s + window_len] for s in starts], dim=0
                            ).to(device)
                            gps_tto = torch.stack(
                                [gps_tensor[s : s + window_len] for s in starts], dim=0
                            )  # [B_tto, T, 3]
                            B_tto = images_tto.shape[0]

                            is_keyframe_tto, keyframe_indices_tto = (
                                selector.select_keyframes(
                                    window_len, B_tto, images_tto.device
                                )
                            )

                            with CriticalOperationProfiler(f"TTO_forward_seq_{seq.name}_step_{step}"):
                                outputs_tto = _run_tto_pose_forward(
                                    model,
                                    images_tto,
                                    is_keyframe_tto,
                                    keyframe_indices_tto,
                                    streaming_mode,
                                    rel_pose_num_iters,
                                )

                            centers = []
                            for b in range(B_tto):
                                if "rel_pose_enc" in outputs_tto:
                                    centers.append(
                                        _compute_camera_centers_differentiable(
                                            outputs_tto["rel_pose_enc"][b],
                                            keyframe_indices_tto[b],
                                        )
                                    )
                                elif "pose_enc" in outputs_tto:
                                    centers.append(
                                        _compute_camera_centers_differentiable(
                                            outputs_tto["pose_enc"][b],
                                            keyframe_indices=None,
                                        )
                                    )
                                else:
                                    print(
                                        "[longstream][TTO] 无位姿输出，跳过 TTO",
                                        flush=True,
                                    )
                                    tto_skipped = True
                                    break

                            if tto_skipped:
                                del outputs_tto, images_tto
                                outputs_tto = None
                                break

                            pred_centers = torch.stack(centers, dim=0)  # [B_tto, T, 3]

                            # ---- 主窗口 loss：计算后立即 backward，释放 graph ----
                            main_loss = _tto_multiscale_loss_inline(
                                pred_centers,
                                gps_tto,
                                _pair_strides,
                                tto_min_gps_disp,
                                tto_lambda_dir,
                                tto_lambda_endpoint,
                            )
                            main_loss_value = 0.0
                            _had_any_loss = False
                            if main_loss is not None:
                                main_loss_value = float(main_loss.detach().item())
                                _had_any_loss = True
                                with CriticalOperationProfiler(
                                    f"TTO_backward_main_seq_{seq.name}_step_{step}"
                                ):
                                    main_loss.backward()
                            # 主窗口 graph 已释放，立即清理引用
                            del main_loss, pred_centers, centers, gps_tto, outputs_tto, images_tto
                            outputs_tto = None

                            # ---- boundary-refresh seam loss：每 boundary 独立 forward+backward ----
                            boundary_total_value = 0.0
                            _bd_info_list = []
                            if tto_boundary_enabled and _boundary_indices:
                                _sel_bds = [
                                    _boundary_indices[step % len(_boundary_indices)]
                                ]
                                for _ in range(tto_boundary_batch_windows - 1):
                                    _sel_bds.append(random.choice(_boundary_indices))
                                for _b in _sel_bds:
                                    _fwd = _run_tto_boundary_refresh_forward(
                                        model, images, device, selector,
                                        streaming_mode, rel_pose_num_iters,
                                        keyframe_stride, refresh, _b,
                                    )
                                    if _fwd is not None:
                                        _b_loss, _b_info = _tto_boundary_refresh_loss_inline(
                                            _fwd, gps_tensor, _b,
                                            tto_boundary_min_gps_disp,
                                            tto_boundary_duplicate_weight,
                                        )
                                        if _b_loss is not None:
                                            _weighted = tto_boundary_weight * _b_loss
                                            boundary_total_value += float(
                                                _weighted.detach().item()
                                            )
                                            _had_any_loss = True
                                            _weighted.backward()
                                            del _weighted, _b_loss
                                            _b_info["boundary"] = _b
                                            _bd_info_list.append(_b_info)
                                        else:
                                            del _b_loss
                                        del _fwd

                            # 本 step 无任何有效 loss，终止 TTO
                            if not _had_any_loss:
                                if step == 0:
                                    print(
                                        f"[longstream][TTO] seq={seq.name}"
                                        " step=0 无有效 GPS 配对，跳过 TTO",
                                        flush=True,
                                    )
                                tto_skipped = True
                                break

                            with CriticalOperationProfiler(f"TTO_backward_seq_{seq.name}_step_{step}"):
                                torch.nn.utils.clip_grad_norm_(
                                    [model.longstream.scale_token], tto_max_grad_norm
                                )
                                tto_seq_optimizer.step()

                            loss_value = main_loss_value + boundary_total_value

                            if (
                                tto_boundary_log
                                and _bd_info_list
                                and (step == 0 or (step + 1) % 5 == 0)
                            ):
                                for _bi in _bd_info_list:
                                    print(
                                        f"[TTO-boundary] step={step + 1}"
                                        f" boundary={_bi.get('boundary', '?')}"
                                        f" seam_pred={_bi.get('seam_pred', float('nan')):.3f}"
                                        f" seam_gps={_bi.get('seam_gps', float('nan')):.3f}"
                                        f" dup={_bi.get('dup', float('nan')):.3f}",
                                        flush=True,
                                    )

                            if step == 0 or (step + 1) % 5 == 0:
                                print(
                                    f"[longstream][TTO] seq={seq.name}"
                                    f" step={step + 1}/{tto_steps}"
                                    f" wins={starts} loss={loss_value:.6f}",
                                    flush=True,
                                )

                            # 早停
                            if loss_value < best_loss - tto_min_delta:
                                best_loss = loss_value
                                bad_steps = 0
                            else:
                                bad_steps += 1

                            if tto_early_stop_patience > 0 and bad_steps >= tto_early_stop_patience:
                                print(
                                    f"[longstream][TTO] seq={seq.name}"
                                    f" early stop at step={step + 1},"
                                    f" best_loss={best_loss:.6f}",
                                    flush=True,
                                )
                                break

                    if not tto_skipped:
                        print(
                            f"[longstream][TTO] seq={seq.name} 完成,"
                            f" scale_token norm="
                            f"{model.longstream.scale_token.data.norm().item():.4f}",
                            flush=True,
                        )
                        # ---- Post-TTO boundary diagnostic (pose-only, no grad) ----
                        if tto_boundary_enabled and _boundary_indices and tto_boundary_log:
                            print(
                                f"[TTO-boundary] === Post-TTO diagnostic: seq={seq.name} ===\n"
                                f"{'boundary':>10} {'seam_pred':>10} {'seam_gps':>10} {'dup':>10}",
                                flush=True,
                            )
                            for _b_diag in _boundary_indices:
                                _fwd_diag = _run_tto_boundary_refresh_forward(
                                    model, images, device, selector,
                                    streaming_mode, rel_pose_num_iters,
                                    keyframe_stride, refresh, _b_diag,
                                )
                                if _fwd_diag is not None:
                                    _, _bi_diag = _tto_boundary_refresh_loss_inline(
                                        _fwd_diag, gps_tensor, _b_diag,
                                        tto_boundary_min_gps_disp,
                                        tto_boundary_duplicate_weight,
                                    )
                                    print(
                                        f"{_b_diag:>10}"
                                        f" {_bi_diag.get('seam_pred', float('nan')):>10.3f}"
                                        f" {_bi_diag.get('seam_gps', float('nan')):>10.3f}"
                                        f" {_bi_diag.get('dup', float('nan')):>10.3f}",
                                        flush=True,
                                    )

                if outputs_tto is not None:
                    del outputs_tto
                    outputs_tto = None

            overlap_result = None
            if mode == "batch_refresh":
                with CriticalOperationProfiler(f"refresh_batch_refresh_seq_{seq.name}"):
                    outputs = run_batch_refresh(
                        model,
                        images,
                        is_keyframe,
                        keyframe_indices,
                        streaming_mode,
                        keyframe_stride,
                        refresh,
                        rel_pose_cfg,
                    )
            elif mode == "streaming_refresh":
                with CriticalOperationProfiler(f"refresh_streaming_refresh_seq_{seq.name}"):
                    outputs = run_streaming_refresh(
                        model,
                        images,
                        is_keyframe,
                        keyframe_indices,
                        streaming_mode,
                        window_size,
                        refresh,
                        rel_pose_cfg,
                    )
            elif mode == "batch_refresh_overlap":
                with CriticalOperationProfiler(
                    f"refresh_batch_refresh_overlap_seq_{seq.name}"
                ):
                    overlap_result = _run_batch_refresh_overlap_export(
                        model=model,
                        images=images,
                        selector=selector,
                        streaming_mode=streaming_mode,
                        keyframe_stride=keyframe_stride,
                        refresh=refresh,
                        rel_pose_cfg=rel_pose_cfg,
                        H=H,
                        W=W,
                        stitch_cfg=stitch_cfg,
                        device=device,
                    )
                outputs = {
                    "depth":      overlap_result["depth"],
                    "depth_conf": overlap_result.get("depth_conf"),
                }
            else:
                raise ValueError(f"Unsupported inference mode: {mode}")
            print(f"[longstream] sequence {seq.name}: inference done", flush=True)
            if device_type == "cuda":
                torch.cuda.empty_cache()

            seq_dir = os.path.join(out_root, seq.name)
            _ensure_dir(seq_dir)

            frame_ids = list(range(S))
            rgb = _to_uint8_rgb(images[0].permute(0, 2, 3, 1))

            extri_np = None  # 在 if/elif 块外初始化，便于后续校正逻辑判断
            intri_np = None

            # ============================================================
            # 统一解码预测位姿（只做一次）
            # ============================================================
            if overlap_result is not None:
                extri_np     = overlap_result["extri_np"]
                intri_np     = overlap_result["intri_np"]
                rel_pose_enc = None
            else:
                extri_np, intri_np, rel_pose_enc = _decode_predicted_extri_intri(
                    outputs, keyframe_indices, H, W
                )

            # ============================================================
            # 保存 overlap stitch 诊断报告（batch_refresh_overlap 专用）
            # ============================================================
            if overlap_result is not None and stitch_cfg.get("save_report", True):
                _diag_dir = os.path.join(seq_dir, "diagnostics")
                _ensure_dir(_diag_dir)
                _report_path = os.path.join(
                    _diag_dir, "batch_overlap_stitch_report.json"
                )
                with open(_report_path, "w", encoding="utf-8") as _rp_f:
                    json.dump(
                        overlap_result.get("stitch_report", []), _rp_f, indent=2
                    )
                print(
                    f"[overlap-stitch] saved report: {_report_path}", flush=True
                )

            # ============================================================
            # GPS 分段 SE3 后处理（可选，仅在 gps_source="file" 时生效）
            # ============================================================
            extri_raw_np = extri_np.copy() if extri_np is not None else None
            extri_corrected_np = None
            if pose_post_enabled and extri_np is not None:
                gps_source = getattr(seq, "gps_source", "none")
                if getattr(seq, "gps_xyz", None) is None:
                    print("[pose_post] 跳过：序列没有 GPS 数据", flush=True)
                elif gps_source != "file":
                    print(
                        f"[pose_post] 跳过：gps_source={gps_source!r}，"
                        f"pose_post_correction 需要真实 GPS 文件 (source='file')",
                        flush=True,
                    )
                else:
                    gps_filter_cfg = pose_post_cfg.get("gps_filter", {})
                    with CriticalOperationProfiler(f"gps_post_correction_seq_{seq.name}"):
                        extri_corrected_np, post_info = correct_poses_with_gps_segment_se3(
                            extri_np,
                            seq.gps_xyz,
                            segment_size=int(pose_post_cfg.get("segment_size", 160)),
                            overlap=int(pose_post_cfg.get("overlap", 40)),
                            min_points=int(pose_post_cfg.get("min_points", 8)),
                            blend=str(pose_post_cfg.get("blend", "hann")),
                            gps_filter_cfg=gps_filter_cfg,
                        )
                    print(f"[pose_post] 已应用 segment_se3: {post_info}", flush=True)

            # use_for_point_export 控制点云/评估使用哪套位姿
            use_corrected = (
                extri_corrected_np is not None
                and bool(pose_post_cfg.get("use_for_point_export", True))
            )
            extri_for_export = extri_corrected_np if use_corrected else extri_raw_np
            # extri_np 统一指向最终用于下游（点云/评估）的位姿
            extri_np = extri_for_export

            if extri_np is not None:
                pose_dir = os.path.join(seq_dir, "poses")
                _ensure_dir(pose_dir)
                # abs_pose.txt = 下游使用的主位姿（use_for_point_export 控制）
                save_w2c_txt(
                    os.path.join(pose_dir, "abs_pose.txt"), extri_np, frame_ids
                )
                # 若后处理生效，分别保存 raw 和 corrected
                if extri_corrected_np is not None:
                    corrected_name = str(
                        pose_post_cfg.get("corrected_pose_name", "abs_pose_corrected.txt")
                    )
                    save_w2c_txt(
                        os.path.join(pose_dir, corrected_name),
                        extri_corrected_np,
                        frame_ids,
                    )
                    if bool(pose_post_cfg.get("save_raw_pose", True)):
                        save_w2c_txt(
                            os.path.join(pose_dir, "abs_pose_raw.txt"),
                            extri_raw_np,
                            frame_ids,
                        )
                save_intri_txt(os.path.join(pose_dir, "intri.txt"), intri_np, frame_ids)
                if rel_pose_enc is not None:
                    save_rel_pose_txt(
                        os.path.join(pose_dir, "rel_pose.txt"), rel_pose_enc, frame_ids
                    )

            # ============================================================
            # 保存 frame_index_map.json（筛帧映射）
            # ============================================================
            save_fmap = bool(filter_cfg.get("save_frame_index_map", True))
            if save_fmap and seq.original_frame_indices is not None:
                save_frame_index_map(
                    os.path.join(seq_dir, "frame_index_map.json"),
                    seq.original_frame_indices,
                    image_paths=seq.image_paths,
                )

            if save_images:
                print(f"[longstream] sequence {seq.name}: saving rgb", flush=True)
                rgb_dir = os.path.join(seq_dir, "images", "rgb")
                save_image_sequence(rgb_dir, list(rgb))
                if save_videos:
                    save_video(
                        os.path.join(seq_dir, "images", "rgb.mp4"),
                        os.path.join(rgb_dir, "frame_*.png"),
                    )

            sky_masks = None
            if enable_sky_mask:
                sky_target_dir = (
                    os.path.join(seq_dir, "sky_masks") if save_sky_mask else None
                )
                raw_sky_masks = compute_sky_mask(
                    seq.image_paths, skyseg_path, sky_target_dir
                )
                if raw_sky_masks is not None:
                    sky_masks = [
                        _prepare_mask_for_model(
                            mask,
                            size=int(data_cfg.get("size", 518)),
                            crop=bool(data_cfg.get("crop", False)),
                            patch_size=int(data_cfg.get("patch_size", 14)),
                            target_shape=(H, W),
                        )
                        for mask in raw_sky_masks
                    ]

            if save_depth and outputs.get("depth") is not None:
                print(f"[longstream] sequence {seq.name}: saving depth", flush=True)
                depth = outputs["depth"][0, :, :, :, 0].detach().cpu().numpy()
                depth_dir = os.path.join(seq_dir, "depth", "dpt")
                _ensure_dir(depth_dir)
                color_dir = os.path.join(seq_dir, "depth", "dpt_plasma")
                _ensure_dir(color_dir)

                # 提前提取 depth_conf（容错：若不存在则置 None）
                _raw_depth_conf = outputs.get("depth_conf")
                depth_conf_np = None
                if _raw_depth_conf is not None:
                    try:
                        dc_arr = _raw_depth_conf[0].detach().cpu().numpy()  # [S, H, W, ?]
                        if dc_arr.ndim == 4:
                            dc_arr = dc_arr[..., 0]  # [S, H, W]
                        depth_conf_np = dc_arr
                    except Exception:
                        depth_conf_np = None

                color_frames = []
                for i in range(S):
                    d = depth[i]
                    sky_m = sky_masks[i] if (sky_masks is not None and sky_masks[i] is not None) else None
                    conf_m = None
                    if confidence_filter_enabled and depth_conf_np is not None:
                        conf_m = (depth_conf_np[i] > confidence_threshold).astype(np.uint8)
                    combined = _combine_masks(sky_m, conf_m)
                    if combined is not None:
                        d = _apply_sky_mask(d, combined)
                    np.save(os.path.join(depth_dir, f"frame_{i:06d}.npy"), d)
                    colored = colorize_depth(d, cmap="plasma")
                    Image.fromarray(colored).save(
                        os.path.join(color_dir, f"frame_{i:06d}.png")
                    )
                    color_frames.append(colored)
                if save_videos:
                    save_video(
                        os.path.join(seq_dir, "depth", "dpt_plasma.mp4"),
                        os.path.join(color_dir, "frame_*.png"),
                    )

            # ============================================================
            # 点云导出：使用 TTO 后模型输出的 extri_np/intri_np
            # ============================================================
            if save_points:
                print(
                    f"[longstream] sequence {seq.name}: saving point clouds", flush=True
                )
                # --- point_head 分支 ---
                if save_point_head and outputs.get("world_points") is not None and extri_np is not None:
                    pts_dir = os.path.join(seq_dir, "points", "point_head")
                    _ensure_dir(pts_dir)
                    pts = outputs["world_points"][0].detach().cpu().numpy()
                    # 提取 world_points_conf（容错）
                    _raw_wpc = outputs.get("world_points_conf")
                    wpc_np = None
                    if _raw_wpc is not None:
                        try:
                            wpc_arr = _raw_wpc[0].detach().cpu().numpy()  # [S, H, W, ?]
                            if wpc_arr.ndim == 4:
                                wpc_arr = wpc_arr[..., 0]
                            wpc_np = wpc_arr
                        except Exception:
                            wpc_np = None
                    full_pts = []
                    full_cols = []
                    for i in range(S):
                        pts_cam = pts[i]
                        pts_world = _camera_points_to_world(pts_cam, extri_np[i])
                        pts_world = pts_world.reshape(pts[i].shape)
                        sky_m = sky_masks[i] if (sky_masks is not None and sky_masks[i] is not None) else None
                        conf_m = None
                        if confidence_filter_enabled and wpc_np is not None:
                            conf_m = (wpc_np[i] > confidence_threshold).astype(np.uint8)
                        valid_mask = _combine_masks(sky_m, conf_m)
                        pts_i, cols_i = _mask_points_and_colors(
                            pts_world,
                            rgb[i],
                            valid_mask,
                        )
                        if save_frame_points and save_point_head:
                            save_pointcloud(
                                os.path.join(pts_dir, f"frame_{i:06d}.ply"),
                                pts_i,
                                colors=cols_i,
                                max_points=max_frame_pointcloud_points,
                                seed=i,
                            )
                        if len(pts_i):
                            full_pts.append(pts_i)
                            full_cols.append(cols_i)
                    if save_point_head:
                        with CriticalOperationProfiler(f"save_full_pointcloud_point_head_seq_{seq.name}"):
                            _save_full_pointcloud(
                                os.path.join(seq_dir, "points", "point_head_full.ply"),
                                full_pts,
                                full_cols,
                                max_points=max_full_pointcloud_points,
                                seed=0,
                            )

                # --- dpt_unproj 分支 ---
                if save_dpt_unproj and (
                    outputs.get("depth") is not None
                    and extri_np is not None
                    and intri_np is not None
                ):
                    depth_for_pts = outputs["depth"][0, :, :, :, 0]

                    dpt_pts_dir = os.path.join(seq_dir, "points", "dpt_unproj")
                    _ensure_dir(dpt_pts_dir)
                    full_pts = []
                    full_cols = []
                    intri_torch = torch.from_numpy(intri_np).to(depth_for_pts.device)

                    # 提取 depth_conf（容错，dpt_unproj 分支独立提取）
                    _raw_dconf = outputs.get("depth_conf")
                    dconf_np = None
                    if _raw_dconf is not None:
                        try:
                            dconf_arr = _raw_dconf[0].detach().cpu().numpy()  # [S, H, W, ?]
                            if dconf_arr.ndim == 4:
                                dconf_arr = dconf_arr[..., 0]
                            dconf_np = dconf_arr
                        except Exception:
                            dconf_np = None

                    for i in range(S):
                        d = depth_for_pts[i]
                        pts_cam = unproject_depth_to_points(
                            d[None], intri_torch[i : i + 1]
                        )[0]
                        R_np = extri_np[i, :3, :3]
                        t_np = extri_np[i, :3, 3]
                        pts_cam_np = pts_cam.detach().cpu().numpy().reshape(-1, 3)
                        pts_world = (
                            R_np.T @ (pts_cam_np.T - t_np[:, None])
                        ).T.astype(np.float32)
                        sky_m = sky_masks[i] if (sky_masks is not None and sky_masks[i] is not None) else None
                        conf_m = None
                        if confidence_filter_enabled and dconf_np is not None:
                            conf_m = (dconf_np[i] > confidence_threshold).astype(np.uint8)
                        valid_mask = _combine_masks(sky_m, conf_m)
                        pts_i, cols_i = _mask_points_and_colors(
                            pts_world,
                            rgb[i],
                            valid_mask,
                        )
                        if save_frame_points and save_dpt_unproj:
                            save_pointcloud(
                                os.path.join(dpt_pts_dir, f"frame_{i:06d}.ply"),
                                pts_i,
                                colors=cols_i,
                                max_points=max_frame_pointcloud_points,
                                seed=i,
                            )
                        if len(pts_i):
                            full_pts.append(pts_i)
                            full_cols.append(cols_i)
                    if save_dpt_unproj:
                        with CriticalOperationProfiler(f"save_full_pointcloud_dpt_unproj_seq_{seq.name}"):
                            _save_full_pointcloud(
                                os.path.join(seq_dir, "points", "dpt_unproj_full.ply"),
                                full_pts,
                                full_cols,
                                max_points=max_full_pointcloud_points,
                                seed=1,
                            )
            del outputs
            if device_type == "cuda":
                torch.cuda.empty_cache()


def run_inference(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    run_inference_cfg(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_inference(args.config)


if __name__ == "__main__":
    main()
