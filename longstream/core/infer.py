import argparse
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
        for s in range(1, S):
            ref_idx = int(keyframe_indices[s].item())
            R_s = rel_R[s] @ R_list[ref_idx]
            t_s = rel_t[s] + rel_R[s] @ t_list[ref_idx]
            R_list.append(R_s)
            t_list.append(t_s)
        abs_R = torch.stack(R_list, dim=0)  # [S, 3, 3]
        abs_t = torch.stack(t_list, dim=0)  # [S, 3]

    # 相机中心 = -R^T @ t
    return -torch.einsum("sji,sj->si", abs_R, abs_t)


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
    save_depth = bool(output_cfg.get("save_depth", True))
    save_images = bool(output_cfg.get("save_images", True))
    mask_sky = bool(output_cfg.get("mask_sky", True))
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
                pair_stride = int(corr_cfg.get("tto_pair_stride", keyframe_stride))
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
                    # 构建有效窗口池：50% overlap 滑动，头尾 GPS 列向位移 > tto_min_gps_disp
                    window_step = max(1, window_len // 2)
                    valid_start_indices = []
                    for start_idx in range(0, min_len - window_len + 1, window_step):
                        disp = torch.linalg.norm(
                            gps_tensor[start_idx + window_len - 1] - gps_tensor[start_idx]
                        ).item()
                        if disp > tto_min_gps_disp:
                            valid_start_indices.append(start_idx)

                    if not valid_start_indices:
                        print(
                            f"[longstream][TTO] seq={seq.name}"
                            f" no valid GPS windows (min_gps_disp={tto_min_gps_disp})，跳过 TTO",
                            flush=True,
                        )
                        tto_skipped = True

                if not tto_skipped:
                    rel_pose_num_iters = rel_pose_cfg.get("num_iterations", 4)
                    # pair_stride 必须在窗口内有效
                    pair_stride = max(1, min(pair_stride, window_len - 1))
                    print(
                        f"[longstream][TTO] seq={seq.name} steps={tto_steps}"
                        f" lr={tto_lr} window_len={window_len}"
                        f" pair_stride={pair_stride}"
                        f" valid_windows={len(valid_start_indices)}",
                        flush=True,
                    )
                    model.eval()
                    with torch.enable_grad():
                        for step in range(tto_steps):
                            tto_seq_optimizer.zero_grad(set_to_none=True)

                            # 随机采样一个有效窗口（mini-batch SGD 风格）
                            start_idx = random.choice(valid_start_indices)
                            end_idx = start_idx + window_len
                            images_tto = images[:, start_idx:end_idx].to(device)
                            gps_tto = gps_tensor[start_idx:end_idx]

                            # 为局部窗口重新生成 keyframe 索引（内部索引 0 ~ window_len-1）
                            is_keyframe_tto, keyframe_indices_tto = (
                                selector.select_keyframes(
                                    window_len, B, images_tto.device
                                )
                            )

                            outputs_tto = _run_tto_pose_forward(
                                model,
                                images_tto,
                                is_keyframe_tto,
                                keyframe_indices_tto,
                                streaming_mode,
                                rel_pose_num_iters,
                            )

                            if "rel_pose_enc" in outputs_tto:
                                pred_centers = _compute_camera_centers_differentiable(
                                    outputs_tto["rel_pose_enc"][0],
                                    keyframe_indices_tto[0],
                                )
                            elif "pose_enc" in outputs_tto:
                                pred_centers = _compute_camera_centers_differentiable(
                                    outputs_tto["pose_enc"][0],
                                    keyframe_indices=None,
                                )
                            else:
                                print(
                                    "[longstream][TTO] 无位姿输出，跳过 TTO",
                                    flush=True,
                                )
                                tto_skipped = True
                                break

                            pred_disp = torch.linalg.norm(
                                pred_centers[pair_stride:] - pred_centers[:-pair_stride],
                                dim=-1,
                            )
                            target_disp = torch.linalg.norm(
                                gps_tto[pair_stride:] - gps_tto[:-pair_stride],
                                dim=-1,
                            )
                            valid = (
                                torch.isfinite(target_disp)
                                & torch.isfinite(pred_disp)
                                & (target_disp > tto_min_gps_disp)
                            )
                            if not valid.any():
                                if step == 0:
                                    print(
                                        f"[longstream][TTO] seq={seq.name}"
                                        " step=0 无有效 GPS 配对，跳过 TTO",
                                        flush=True,
                                    )
                                    tto_skipped = True
                                break

                            loss = torch.nn.functional.huber_loss(
                                pred_disp[valid], target_disp[valid]
                            )
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                [model.longstream.scale_token], tto_max_grad_norm
                            )
                            tto_seq_optimizer.step()

                            if step == 0 or (step + 1) % 5 == 0:
                                print(
                                    f"[longstream][TTO] seq={seq.name}"
                                    f" step={step + 1}/{tto_steps}"
                                    f" win=[{start_idx},{end_idx})"
                                    f" loss={loss.item():.6f}",
                                    flush=True,
                                )

                            # 显式释放局部引用，避免图引用滤留
                            del outputs_tto, images_tto
                            outputs_tto = None

                    if not tto_skipped:
                        print(
                            f"[longstream][TTO] seq={seq.name} 完成,"
                            f" scale_token norm="
                            f"{model.longstream.scale_token.data.norm().item():.4f}",
                            flush=True,
                        )

                if outputs_tto is not None:
                    del outputs_tto
                    outputs_tto = None

            if mode == "batch_refresh":
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
            extri_np, intri_np, rel_pose_enc = _decode_predicted_extri_intri(
                outputs, keyframe_indices, H, W
            )

            if extri_np is not None:
                pose_dir = os.path.join(seq_dir, "poses")
                _ensure_dir(pose_dir)
                save_w2c_txt(
                    os.path.join(pose_dir, "abs_pose.txt"), extri_np, frame_ids
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
            if mask_sky:
                raw_sky_masks = compute_sky_mask(
                    seq.image_paths, skyseg_path, os.path.join(seq_dir, "sky_masks")
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
                if outputs.get("world_points") is not None and extri_np is not None:
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
                        if save_frame_points:
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
                    _save_full_pointcloud(
                        os.path.join(seq_dir, "points", "point_head_full.ply"),
                        full_pts,
                        full_cols,
                        max_points=max_full_pointcloud_points,
                        seed=0,
                    )

                # --- dpt_unproj 分支 ---
                if (
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
                        pts_cam_np = pts_cam.cpu().numpy().reshape(-1, 3)
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
                        if save_frame_points:
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
