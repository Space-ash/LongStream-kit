"""
longstream/core/tto.py
----------------------
测试时优化（Test-Time Optimization）模块。
从 longstream/core/infer.py 抽离，独立复用于 batch 推理和实时流推理。

公开 API：
  prepare_tto(model, corr_cfg)          → TTOContext
  run_tto_scale_optimization(ctx, ...)  → bool  (是否成功执行，False = 跳过)
  reset_tto(ctx)                        → None  (每条新序列前调用)
内部函数（可被外部直接调用）：
  _run_tto_pose_forward(...)
  _compute_camera_centers_differentiable(...)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from longstream.utils.vendor.models.components.utils.rotation import quat_to_mat
from longstream.utils.camera import compose_abs_from_rel


# ═══════════════════════════════════════════════════════════════════════════
#  内部位姿工具
# ═══════════════════════════════════════════════════════════════════════════

def _run_tto_pose_forward(
    model,
    images: torch.Tensor,
    is_keyframe: torch.Tensor,
    keyframe_indices: torch.Tensor,
    streaming_mode: str,
    rel_pose_num_iterations: int,
) -> Dict[str, Any]:
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
    keyframe_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    TTO 专用：从 pose_enc 全微分地计算相机中心。

    Args:
        pose_enc: [S, D]，前 3 小次元为平移 t，3:7 为单位四元数 q。
            - keyframe_indices 为 None 时表示绝对 pose。
            - keyframe_indices 不为 None 时表示相对 pose（rel_pose_enc）。
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


# ═══════════════════════════════════════════════════════════════════════════
#  TTOContext — 保存每序列 TTO 状态
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TTOContext:
    """每条序列/流的 TTO 运行上下文。"""
    model: Any
    # 超参数
    tto_steps: int
    tto_lr: float
    tto_weight_decay: float
    tto_window_size: int
    tto_min_gps_disp: float
    tto_max_grad_norm: float
    rel_pose_num_iterations: int
    # 新增配置项
    tto_sampling: str
    tto_pair_strides: List[int]
    tto_lambda_dir: float
    tto_lambda_endpoint: float
    tto_early_stop_patience: int
    tto_min_delta: float
    tto_batch_windows: int
    # 初始 scale_token（用于序列间复位）
    initial_scale_token: torch.Tensor
    # 当前序列的优化器（每次 reset_tto 后重新创建）
    optimizer: Optional[torch.optim.Optimizer] = field(default=None, repr=False)


def prepare_tto(model, corr_cfg: dict) -> TTOContext:
    """
    验证模型是否支持 TTO，冻结所有参数，仅开启 scale_token 梯度，
    返回 TTOContext。

    Raises:
        RuntimeError: 若模型没有 scale_token 或 enable_scale_token 为 False。
    """
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

    # 解析 tto_pair_strides，兼容 int/list/tuple；保留旧 tto_pair_stride 回退
    _raw_strides = corr_cfg.get("tto_pair_strides", None)
    if _raw_strides is None:
        _old_stride = corr_cfg.get("tto_pair_stride", None)
        _raw_strides = [_old_stride] if _old_stride is not None else [8]
    if isinstance(_raw_strides, (int, float)):
        _raw_strides = [_raw_strides]
    _pair_strides = [int(s) for s in _raw_strides if int(s) > 0]
    if not _pair_strides:
        _pair_strides = [8]

    ctx = TTOContext(
        model=model,
        tto_steps=int(corr_cfg.get("tto_steps", 20)),
        tto_lr=float(corr_cfg.get("tto_lr", 1e-3)),
        tto_weight_decay=float(corr_cfg.get("tto_weight_decay", 0.0)),
        tto_window_size=int(corr_cfg.get("tto_window_size", 40)),
        tto_min_gps_disp=float(corr_cfg.get("tto_min_gps_disp", 1.0)),
        tto_max_grad_norm=float(corr_cfg.get("tto_max_grad_norm", 1.0)),
        rel_pose_num_iterations=int(corr_cfg.get("rel_pose_num_iterations", 4)),
        tto_sampling=str(corr_cfg.get("tto_sampling", "random")),
        tto_pair_strides=_pair_strides,
        tto_lambda_dir=float(corr_cfg.get("tto_lambda_dir", 0.0)),
        tto_lambda_endpoint=float(corr_cfg.get("tto_lambda_endpoint", 0.0)),
        tto_early_stop_patience=int(corr_cfg.get("tto_early_stop_patience", 0)),
        tto_min_delta=float(corr_cfg.get("tto_min_delta", 1.0e-3)),
        tto_batch_windows=int(corr_cfg.get("tto_batch_windows", 1)),
        initial_scale_token=initial_scale_token,
    )
    return ctx


def reset_tto(ctx: TTOContext) -> None:
    """每条新序列开始前调用：复位 scale_token 至初始值，重建优化器。"""
    ctx.model.longstream.scale_token.data.copy_(ctx.initial_scale_token)
    ctx.optimizer = torch.optim.AdamW(
        [ctx.model.longstream.scale_token],
        lr=ctx.tto_lr,
        weight_decay=ctx.tto_weight_decay,
    )


def _build_tto_window_starts(
    min_len: int,
    window_len: int,
    window_step: int,
    gps_tensor: torch.Tensor,
    min_gps_disp: float,
) -> List[int]:
    """
    构建覆盖优先的窗口起始索引列表，保证首尾端点被纳入。
    """
    last_start = max(0, min_len - window_len)
    starts = list(range(0, last_start + 1, window_step))
    if 0 not in starts:
        starts.insert(0, 0)
    if last_start not in starts:
        starts.append(last_start)

    valid: List[int] = []
    for s in starts:
        disp = torch.linalg.norm(gps_tensor[s + window_len - 1] - gps_tensor[s]).item()
        if torch.isfinite(torch.tensor(disp)) and disp > min_gps_disp:
            valid.append(s)

    # deterministic_coverage 端点兼底：即使位移不足也保留，loss valid mask 会兜底
    for s in (0, last_start):
        if s not in valid:
            valid.insert(0 if s == 0 else len(valid), s)

    # 去重并保序
    return list(dict.fromkeys(valid))


def _select_tto_starts(
    valid_starts: List[int],
    step: int,
    batch_windows: int,
    sampling: str,
) -> List[int]:
    """
    根据采样策略选出本步要使用的窗口起始列表（长度 = batch_windows）。
    """
    n = max(1, batch_windows)

    if sampling == "deterministic_coverage":
        if step < len(valid_starts):
            base = valid_starts[step]
        else:
            base = random.choice(valid_starts)
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

    # default: random
    return random.choices(valid_starts, k=n)


def _tto_multiscale_loss(
    pred_centers: torch.Tensor,
    gps_tto: torch.Tensor,
    pair_strides: List[int],
    min_gps_disp: float,
    lambda_dir: float,
    lambda_endpoint: float,
) -> Optional[torch.Tensor]:
    """
    多尺度位移 loss（含可选方向项和端点项）。

    pred_centers: [B, T, 3]
    gps_tto:      [B, T, 3]
    """
    losses = []
    eps = 1.0e-6

    for stride in pair_strides:
        stride = max(1, min(int(stride), pred_centers.shape[1] - 1))
        pred_vec = pred_centers[:, stride:] - pred_centers[:, :-stride]
        gps_vec = gps_tto[:, stride:] - gps_tto[:, :-stride]

        pred_disp = torch.linalg.norm(pred_vec, dim=-1)
        gps_disp = torch.linalg.norm(gps_vec, dim=-1)
        valid = (
            torch.isfinite(pred_disp)
            & torch.isfinite(gps_disp)
            & (gps_disp > min_gps_disp)
        )
        if valid.any():
            losses.append(
                torch.nn.functional.huber_loss(pred_disp[valid], gps_disp[valid])
            )
            if lambda_dir > 0:
                pred_dir = pred_vec / pred_disp.clamp_min(eps).unsqueeze(-1)
                gps_dir = gps_vec / gps_disp.clamp_min(eps).unsqueeze(-1)
                losses.append(
                    lambda_dir
                    * torch.nn.functional.huber_loss(pred_dir[valid], gps_dir[valid])
                )

    if lambda_endpoint > 0:
        pred_ep = pred_centers[:, -1] - pred_centers[:, 0]
        gps_ep = gps_tto[:, -1] - gps_tto[:, 0]
        ep_disp = torch.linalg.norm(gps_ep, dim=-1)
        valid_ep = torch.isfinite(ep_disp) & (ep_disp > min_gps_disp)
        if valid_ep.any():
            losses.append(
                lambda_endpoint
                * torch.nn.functional.huber_loss(pred_ep[valid_ep], gps_ep[valid_ep])
            )

    if not losses:
        return None
    return torch.stack([x if x.ndim == 0 else x.mean() for x in losses]).sum()


def run_tto_scale_optimization(
    ctx: TTOContext,
    images: torch.Tensor,
    gps_xyz: np.ndarray,
    selector,
    streaming_mode: str,
    device: str,
    seq_name: str = "seq",
    pair_stride: Optional[int] = None,
) -> bool:
    """
    在给定帧序列上运行 TTO，用 GPS 位移监督优化 scale_token。

    TTO 训练段内部不 detach（需要反传）。
    TTO 结束后调用方负责 detach（进 Rerun 前）。

    Parameters
    ----------
    ctx : TTOContext
        由 prepare_tto() 返回的上下文。
    images : torch.Tensor
        CPU Tensor [B, S, C, H, W]。
    gps_xyz : np.ndarray
        [S, 3] float32。
    selector : KeyframeSelector
        关键帧选择器（用于生成 is_keyframe / keyframe_indices）。
    streaming_mode : str
        "causal" | "window"。
    device : str
        推理设备字符串。
    seq_name : str
        序列名称，仅用于日志。
    pair_stride : int | None
        GPS 配对步长，None 时从 ctx.tto_window_size 推断。

    Returns
    -------
    bool
        True  = TTO 正常执行完成
        False = 跳过（帧数不足 / GPS 无效等）
    """
    B, S_total = images.shape[:2]
    gps_tensor = torch.as_tensor(gps_xyz, device=device, dtype=torch.float32)
    min_len = min(S_total, gps_tensor.shape[0])
    gps_tensor = gps_tensor[:min_len]

    window_len = min(ctx.tto_window_size, min_len)
    if window_len < 2:
        print(
            f"[TTO] seq={seq_name} window_len={window_len} < 2，跳过 TTO",
            flush=True,
        )
        return False

    # pair_strides：优先使用 ctx 里的，pair_stride 参数仅作向后兼容覆盖
    pair_strides = ctx.tto_pair_strides
    if pair_stride is not None:
        pair_strides = [max(1, min(int(pair_stride), window_len - 1))]
    pair_strides = [max(1, min(s, window_len - 1)) for s in pair_strides]

    # 构建覆盖优先的窗口起始列表
    window_step = max(1, window_len // 2)
    valid_start_indices = _build_tto_window_starts(
        min_len, window_len, window_step, gps_tensor, ctx.tto_min_gps_disp
    )

    if not valid_start_indices:
        print(
            f"[TTO] seq={seq_name} 没有有效窗口，跳过 TTO",
            flush=True,
        )
        return False

    print(
        f"[TTO] seq={seq_name} steps={ctx.tto_steps} lr={ctx.tto_lr}"
        f" window_len={window_len} pair_strides={pair_strides}"
        f" sampling={ctx.tto_sampling} batch_windows={ctx.tto_batch_windows}"
        f" valid_windows={len(valid_start_indices)}",
        flush=True,
    )

    if ctx.optimizer is None:
        reset_tto(ctx)

    ctx.model.eval()
    outputs_tto = None
    skipped = False

    best_loss = float("inf")
    bad_steps = 0

    with torch.enable_grad():
        for step in range(ctx.tto_steps):
            ctx.optimizer.zero_grad(set_to_none=True)

            starts = _select_tto_starts(
                valid_start_indices, step, ctx.tto_batch_windows, ctx.tto_sampling
            )

            # 多窗口 batch
            images_tto = torch.cat(
                [images[:, s : s + window_len] for s in starts], dim=0
            ).to(device)
            gps_tto = torch.stack(
                [gps_tensor[s : s + window_len] for s in starts], dim=0
            )  # [B_tto, T, 3]
            B_tto = images_tto.shape[0]

            is_keyframe_tto, keyframe_indices_tto = selector.select_keyframes(
                window_len, B_tto, images_tto.device
            )

            outputs_tto = _run_tto_pose_forward(
                ctx.model,
                images_tto,
                is_keyframe_tto,
                keyframe_indices_tto,
                streaming_mode,
                ctx.rel_pose_num_iterations,
            )

            # 对 batch 维度循环求各窗口相机中心
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
                            None,
                        )
                    )
                else:
                    print(f"[TTO] seq={seq_name} 无位姿输出，跳过 TTO", flush=True)
                    skipped = True
                    break

            if skipped:
                del outputs_tto, images_tto
                break

            pred_centers = torch.stack(centers, dim=0)  # [B_tto, T, 3]

            loss = _tto_multiscale_loss(
                pred_centers,
                gps_tto,
                pair_strides,
                ctx.tto_min_gps_disp,
                ctx.tto_lambda_dir,
                ctx.tto_lambda_endpoint,
            )

            if loss is None:
                if step == 0:
                    print(
                        f"[TTO] seq={seq_name} step=0 无有效 GPS 配对，跳过 TTO",
                        flush=True,
                    )
                    skipped = True
                del outputs_tto, images_tto
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [ctx.model.longstream.scale_token], ctx.tto_max_grad_norm
            )
            ctx.optimizer.step()

            loss_value = float(loss.detach().item())

            if step == 0 or (step + 1) % 5 == 0:
                print(
                    f"[TTO] seq={seq_name} step={step + 1}/{ctx.tto_steps}"
                    f" wins={starts} loss={loss_value:.6f}",
                    flush=True,
                )

            # 早停
            if loss_value < best_loss - ctx.tto_min_delta:
                best_loss = loss_value
                bad_steps = 0
            else:
                bad_steps += 1

            if ctx.tto_early_stop_patience > 0 and bad_steps >= ctx.tto_early_stop_patience:
                print(
                    f"[TTO] seq={seq_name} early stop at step={step + 1},"
                    f" best_loss={best_loss:.6f}",
                    flush=True,
                )
                del outputs_tto, images_tto
                outputs_tto = None
                break

            del outputs_tto, images_tto
            outputs_tto = None

    if not skipped:
        print(
            f"[TTO] seq={seq_name} 完成，scale_token norm="
            f"{ctx.model.longstream.scale_token.data.norm().item():.4f}",
            flush=True,
        )

    return not skipped
