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

    ctx = TTOContext(
        model=model,
        tto_steps=int(corr_cfg.get("tto_steps", 20)),
        tto_lr=float(corr_cfg.get("tto_lr", 1e-3)),
        tto_weight_decay=float(corr_cfg.get("tto_weight_decay", 0.0)),
        tto_window_size=int(corr_cfg.get("tto_window_size", 40)),
        tto_min_gps_disp=float(corr_cfg.get("tto_min_gps_disp", 1.0)),
        tto_max_grad_norm=float(corr_cfg.get("tto_max_grad_norm", 1.0)),
        rel_pose_num_iterations=int(corr_cfg.get("rel_pose_num_iterations", 4)),
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

    _pair_stride = pair_stride if pair_stride is not None else max(1, window_len // 4)
    _pair_stride = max(1, min(_pair_stride, window_len - 1))

    # 构建有效窗口池（50% 重叠滑动）
    window_step = max(1, window_len // 2)
    valid_start_indices: List[int] = []
    for start_idx in range(0, min_len - window_len + 1, window_step):
        disp = torch.linalg.norm(
            gps_tensor[start_idx + window_len - 1] - gps_tensor[start_idx]
        ).item()
        if disp > ctx.tto_min_gps_disp:
            valid_start_indices.append(start_idx)

    if not valid_start_indices:
        print(
            f"[TTO] seq={seq_name} 没有 GPS 位移 > {ctx.tto_min_gps_disp} 的有效窗口，跳过 TTO",
            flush=True,
        )
        return False

    print(
        f"[TTO] seq={seq_name} steps={ctx.tto_steps} lr={ctx.tto_lr}"
        f" window_len={window_len} pair_stride={_pair_stride}"
        f" valid_windows={len(valid_start_indices)}",
        flush=True,
    )

    if ctx.optimizer is None:
        reset_tto(ctx)

    ctx.model.eval()
    outputs_tto = None
    skipped = False

    with torch.enable_grad():
        for step in range(ctx.tto_steps):
            ctx.optimizer.zero_grad(set_to_none=True)

            start_idx = random.choice(valid_start_indices)
            end_idx = start_idx + window_len
            images_tto = images[:, start_idx:end_idx].to(device)
            gps_tto = gps_tensor[start_idx:end_idx]

            is_keyframe_tto, keyframe_indices_tto = selector.select_keyframes(
                window_len, B, images_tto.device
            )

            outputs_tto = _run_tto_pose_forward(
                ctx.model,
                images_tto,
                is_keyframe_tto,
                keyframe_indices_tto,
                streaming_mode,
                ctx.rel_pose_num_iterations,
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
                print(f"[TTO] seq={seq_name} 无位姿输出，跳过 TTO", flush=True)
                skipped = True
                del outputs_tto, images_tto
                break

            pred_disp = torch.linalg.norm(
                pred_centers[_pair_stride:] - pred_centers[:-_pair_stride],
                dim=-1,
            )
            target_disp = torch.linalg.norm(
                gps_tto[_pair_stride:] - gps_tto[:-_pair_stride],
                dim=-1,
            )
            valid = (
                torch.isfinite(target_disp)
                & torch.isfinite(pred_disp)
                & (target_disp > ctx.tto_min_gps_disp)
            )
            if not valid.any():
                if step == 0:
                    print(
                        f"[TTO] seq={seq_name} step=0 无有效 GPS 配对，跳过 TTO",
                        flush=True,
                    )
                    skipped = True
                del outputs_tto, images_tto
                break

            loss = torch.nn.functional.huber_loss(
                pred_disp[valid], target_disp[valid]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [ctx.model.longstream.scale_token], ctx.tto_max_grad_norm
            )
            ctx.optimizer.step()

            if step == 0 or (step + 1) % 5 == 0:
                print(
                    f"[TTO] seq={seq_name} step={step + 1}/{ctx.tto_steps}"
                    f" win=[{start_idx},{end_idx}) loss={loss.item():.6f}",
                    flush=True,
                )

            del outputs_tto, images_tto
            outputs_tto = None

    if not skipped:
        print(
            f"[TTO] seq={seq_name} 完成，scale_token norm="
            f"{ctx.model.longstream.scale_token.data.norm().item():.4f}",
            flush=True,
        )

    return not skipped
