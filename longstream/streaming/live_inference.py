"""
longstream/streaming/live_inference.py
---------------------------------------
实时逐帧推理运行器（LiveInferenceRunner）。
在后台线程中运行，通过 stop_event 控制停止。

典型用法：
    runner = LiveInferenceRunner(cfg)
    runner.start()
    ...
    runner.stop()
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import os

from longstream.core.model import LongStreamModel
from longstream.core.tto import TTOContext, prepare_tto, reset_tto, run_tto_scale_optimization
from longstream.data.stream_feeder import FramePacket, StreamFeeder
from longstream.streaming.keyframe_selector import KeyframeSelector
from longstream.streaming.stream_session import StreamSession


# ═══════════════════════════════════════════════════════════════════════════
#  子进程入口（Windows spawn 要求必须为可 pickle 的模块级函数）
# ═══════════════════════════════════════════════════════════════════════════

def _worker_main(
    cfg: dict,
    result_queue: "mp.Queue",
    stop_flag: "mp.Event",
) -> None:
    """
    子进程入口：在独立进程里运行完整推理循环。

    每帧结果格式：(frame_idx: int, outputs_cpu: dict)
    错误格式：    ("__error__", traceback_str: str)
    """
    import traceback as _tb

    try:
        _worker_inference_loop(cfg, result_queue, stop_flag)
    except Exception as exc:
        try:
            result_queue.put_nowait(("__error__", str(exc) + "\n" + _tb.format_exc()))
        except Exception:
            pass
        print(f"[LiveInferenceRunner/worker] 推理异常: {exc}", flush=True)
        _tb.print_exc()
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("[LiveInferenceRunner/worker] 子进程退出", flush=True)


def _worker_inference_loop(
    cfg: dict,
    result_queue: "mp.Queue",
    stop_flag: "mp.Event",
) -> None:
    """完整推理循环（在子进程中运行）。"""
    device_str: str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_cfg: dict = cfg.get("model", {})
    data_cfg: dict = cfg.get("data", {})
    infer_cfg: dict = cfg.get("inference", {})
    opt_cfg: dict = cfg.get("optimizations", {})
    corr_cfg: dict = opt_cfg.get("correction", {})
    _out_cfg_early = {
        **cfg.get("output", {}),
        **cfg.get("optimizations", {}).get("filter", {}),
    }

    # ── CPU 线程控制（防止子进程过度竞争 CPU 核心导致过热）──────────────
    try:
        torch.set_num_threads(int(_out_cfg_early.get("torch_num_threads", 2)))
    except Exception:
        pass
    try:
        import cv2 as _cv2_init
        _cv2_init.setNumThreads(int(_out_cfg_early.get("opencv_num_threads", 1)))
    except Exception:
        pass

    # ── 模型 ──────────────────────────────────────────────────────────────
    print("[LiveInferenceRunner/worker] 初始化模型...", flush=True)
    model = LongStreamModel(model_cfg).to(device_str)
    model.eval()
    print("[LiveInferenceRunner/worker] 模型就绪", flush=True)

    # ── TTO 上下文 ────────────────────────────────────────────────────────
    tto_enabled = bool(corr_cfg.get("tto_enabled", False))
    tto_ctx: Optional[TTOContext] = None
    if tto_enabled:
        tto_ctx = prepare_tto(model, corr_cfg)
        print("[LiveInferenceRunner/worker] TTO 已启用", flush=True)

    # ── 关键帧选择器 ──────────────────────────────────────────────────────
    keyframe_stride = int(infer_cfg.get("keyframe_stride", 8))
    keyframe_mode = infer_cfg.get("keyframe_mode", "fixed")
    refresh = int(
        infer_cfg.get("refresh", int(infer_cfg.get("keyframes_per_batch", 3)) + 1)
    )
    streaming_mode = infer_cfg.get("streaming_mode", "causal")
    window_size = int(infer_cfg.get("window_size", 5))

    selector = KeyframeSelector(
        min_interval=keyframe_stride,
        max_interval=keyframe_stride,
        force_first=True,
        mode="random" if keyframe_mode == "random" else "fixed",
    )

    # ── Feeder（子进程内总是从 cfg 构建，不接受外部 feeder 对象）────────
    feeder = _build_feeder_from_cfg(data_cfg)

    # ── StreamSession ─────────────────────────────────────────────────────
    session = StreamSession(model, mode=streaming_mode, window_size=window_size)

    # ── refresh 参数 ──────────────────────────────────────────────────────
    refresh_intervals = max(1, refresh - 1)
    keyframe_count = 0

    # TTO 相关：reset_tto 仅在流开始时执行一次（不在每个 window 执行）
    if tto_ctx is not None:
        reset_tto(tto_ctx)
    pair_stride_tto = int(corr_cfg.get("tto_pair_stride", keyframe_stride))
    tto_window_size = int(corr_cfg.get("tto_window_size", 40))
    _tto_frame_buffer: list = []
    _tto_gps_buffer: list = []

    # ── 分段追踪（对齐 run_streaming_refresh 逻辑）────────────────────────
    segment_start: int = 0
    seg_R_abs: dict = {}
    seg_t_abs: dict = {}
    seg_R_abs[0] = np.eye(3, dtype=np.float32)
    seg_t_abs[0] = np.zeros(3, dtype=np.float32)

    # ── 输出目录 / 输出配置（循环前构建，避免每帧重复）──────────────────
    _out_cfg_merged = {
        **cfg.get("output", {}),
        **cfg.get("optimizations", {}).get("filter", {}),
    }
    _out_root: str = cfg.get("output", {}).get("root", os.path.join("output", "live_inference"))
    _seq_name: str = _get_stream_seq_name(cfg)
    _out_dir: str = os.path.join(_out_root, _seq_name)
    print(f"[LiveInferenceRunner/worker] 输出目录: {_out_dir}", flush=True)

    # ── 资源监控：在推理进程内启动，记录真实推理进程的 CPU/GPU ──────────
    _live_monitor = None
    try:
        from longstream.utils.resource_monitor import from_cfg as _mon_from_cfg
        _live_monitor = _mon_from_cfg(cfg.get("monitoring", {}))
        if _live_monitor is not None:
            _live_monitor.start(_out_dir)
            print("[LiveInferenceRunner/worker] 资源监控已启动", flush=True)
    except Exception as _mon_err:
        print(f"[LiveInferenceRunner/worker] 资源监控启动失败（忽略）: {_mon_err}", flush=True)
        _live_monitor = None

    # 如果启用 GLB 导出，自动启用逐帧点云
    if _out_cfg_merged.get("export_glb", False) and not _out_cfg_merged.get("save_frame_points", False):
        _out_cfg_merged = dict(_out_cfg_merged)
        _out_cfg_merged["save_frame_points"] = True
        print("[LiveInferenceRunner/worker] 已自动启用 save_frame_points（GLB 需要）", flush=True)

    # ── 位姿累积 ─────────────────────────────────────────────────────────
    _extri_list: list = []
    _pose_frame_ids: list = []

    # ── 全局点云参数 + in-memory reservoir ───────────────────────────────
    _max_frame_pts: int = int(_out_cfg_merged.get("max_frame_pointcloud_points", 8000))
    _max_full_pts: int = int(_out_cfg_merged.get("max_full_pointcloud_points", _max_frame_pts * 100))
    # 分支开关：缺失时回退到 save_points 行为（向后兼容旧 YAML）
    _save_points_base: bool = bool(_out_cfg_merged.get("save_points", False))
    _save_point_head: bool = bool(_out_cfg_merged.get("save_point_head", _save_points_base))
    _save_dpt_unproj: bool = bool(_out_cfg_merged.get("save_dpt_unproj", _save_points_base))
    # reservoir：分块累积（pending 列表 + 已紧凑化 flat 数组，避免每帧 np.concatenate）
    _res_pts: Optional[np.ndarray] = None   # [R, 3] float32，已紧凑化
    _res_cols: Optional[np.ndarray] = None  # [R, 3] uint8
    _res_rng = np.random.default_rng(seed=0)
    _res_pending: list = []       # 待合并小分块列表
    _res_pending_cols: list = []
    _res_pending_n: int = 0
    # dpt_unproj reservoir（与 point_head reservoir 并行）
    _res_dpu_pts: Optional[np.ndarray] = None   # [R, 3] float32
    _res_dpu_cols: Optional[np.ndarray] = None  # [R, 3] uint8
    _res_dpu_rng = np.random.default_rng(seed=1)
    _res_dpu_pending: list = []
    _res_dpu_pending_cols: list = []
    _res_dpu_pending_n: int = 0
    # 崩溃保护 checkpoint：每 N 帧覆盖写一次 points/dpt_unproj_latest.ply
    # 默认 0 = 关闭；适合 Gradio 默认场景，避免周期性大 IO 尖峰
    _checkpoint_interval: int = int(_out_cfg_merged.get("live_global_checkpoint_interval", 0))
    _checkpoint_max_points: int = int(_out_cfg_merged.get("live_global_checkpoint_points", 200000))
    # Rerun 是否开启：仅开启时才跨进程传输大 numpy
    _emit_frame_results: bool = bool(cfg.get("_enable_rerun", False))
    _viewer_max_global: int = int(_out_cfg_merged.get("rerun_global_points", 20000))
    # 全局 reservoir compact 触发阈值（待合并点数达到此值才触发一次合并）
    # 默认 800000：VKITTI 每帧约 7 万点，约 11 帧合并一次，比原来的 0.1*2M=200k 更宽松
    _compact_pending_points: int = int(_out_cfg_merged.get("global_compact_pending_points", 800000))
    # dpt_unproj 警告限频计数器（防止长序列日志刷爆）
    _dpu_warn_count: int = 0
    _dpu_warn_max: int = 3  # 前 3 次打印警告，此后仅累计

    # ── sky segmentation session（真实初始化）────────────────────────────
    _sky_session = None
    _sky_mask_dir: Optional[str] = None
    # enable_sky_mask：控制是否运行 sky segmentation 以过滤点云（不影响保存 mask 文件）
    # save_sky_mask：仅控制 mask PNG 是否落盘；向后兼容旧 YAML mask_sky
    _enable_sky_mask: bool = bool(_out_cfg_merged.get(
        "enable_sky_mask", _out_cfg_merged.get("mask_sky", False)
    ))
    _save_sky_mask: bool = bool(_out_cfg_merged.get(
        "save_sky_mask", _out_cfg_merged.get("mask_sky", False)
    ))
    if _enable_sky_mask:
        try:
            import onnxruntime as _ort
            # 优先读取 cfg["output"]["skyseg_path"]，再回退到项目根默认路径
            # 不自动联网下载：离线/服务器环境下自动下载会卡住
            _sky_cfg_path = _out_cfg_merged.get("skyseg_path") or ""
            if _sky_cfg_path:
                _sky_model_path = os.path.abspath(_sky_cfg_path)
            else:
                _sky_model_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "skyseg.onnx")
                )
            if not os.path.exists(_sky_model_path):
                print(
                    f"[LiveInferenceRunner/worker] 警告：skyseg 模型文件不存在: {_sky_model_path}，"
                    "跳过天空过滤。请在 YAML output.skyseg_path 指定正确路径，"
                    "或将 skyseg.onnx 放到项目根目录。",
                    flush=True,
                )
                raise FileNotFoundError(_sky_model_path)
            _sky_so = _ort.SessionOptions()
            _sky_so.intra_op_num_threads = int(_out_cfg_merged.get("skyseg_num_threads", 1))
            _sky_so.inter_op_num_threads = 1
            _sky_session = _ort.InferenceSession(
                _sky_model_path,
                sess_options=_sky_so,
                providers=["CPUExecutionProvider"],
            )
            # _save_sky_mask=True 时才创建 sky_masks 目录（避免不必要的磁盘目录）
            _sky_mask_dir = os.path.join(_out_dir, "sky_masks") if _save_sky_mask else None
            print("[LiveInferenceRunner/worker] sky segmentation session 初始化成功", flush=True)
        except Exception as _sky_init_err:
            print(
                f"[LiveInferenceRunner/worker] 警告：sky segmentation 初始化失败，本次跳过天空过滤。"
                f"原因: {_sky_init_err}",
                flush=True,
            )
            _sky_session = None
            _sky_mask_dir = None

    print("[LiveInferenceRunner/worker] 开始逐帧推理", flush=True)

    # ── 逐帧 timing 统计（每 N 帧打印一次，默认 10 帧）─────────────────
    _timing_interval: int = int(_out_cfg_merged.get("timing_log_interval", 10))
    _timing_acc = {
        "decode_infer": 0.0, "skyseg": 0.0, "filter": 0.0,
        "save": 0.0, "rerun_payload": 0.0, "queue_put": 0.0, "total": 0.0,
    }
    _timing_n: int = 0

    with torch.no_grad():
        for packet in feeder:
            if stop_flag.is_set():
                break

            g: int = packet.frame_index
            _t_frame_start = time.perf_counter()
            _t0 = _t_frame_start
            local_pos: int = g - segment_start

            is_kf = (g == 0) or (g % keyframe_stride == 0)
            is_keyframe_t = torch.tensor(
                [[is_kf]], dtype=torch.bool, device=device_str
            )

            if g == 0:
                kf_idx_global = 0
            else:
                kf_idx_global = ((g - 1) // keyframe_stride) * keyframe_stride
            kf_idx_local = max(0, kf_idx_global - segment_start)
            keyframe_indices_t = torch.tensor(
                [[kf_idx_local]], dtype=torch.long, device=device_str
            )

            # ── TTO 缓冲 ──────────────────────────────────────────────────
            if tto_ctx is not None and packet.gps_xyz is not None:
                _tto_frame_buffer.append(packet)
                _tto_gps_buffer.append(packet.gps_xyz)
                if len(_tto_frame_buffer) >= tto_window_size and is_kf:
                    _run_tto_window(
                        tto_ctx=tto_ctx,
                        frame_buffer=_tto_frame_buffer[-tto_window_size:],
                        gps_buffer=_tto_gps_buffer[-tto_window_size:],
                        selector=selector,
                        streaming_mode=streaming_mode,
                        device=device_str,
                        frame_idx=g,
                        pair_stride=pair_stride_tto,
                    )
                    slide = tto_window_size // 2
                    _tto_frame_buffer = _tto_frame_buffer[slide:]
                    _tto_gps_buffer = _tto_gps_buffer[slide:]

            # ── 前向推理 ──────────────────────────────────────────────────
            frame_tensor: torch.Tensor = packet.image_tensor.to(
                device_str, non_blocking=True
            )
            outputs = session.forward_stream(
                frame_tensor,
                is_keyframe=is_keyframe_t,
                keyframe_indices=keyframe_indices_t,
                record=False,
            )

            # ── 解码输出到 CPU numpy ───────────────────────────────────────
            outputs_cpu = _decode_outputs_to_cpu(
                outputs, packet, kf_idx_local, local_pos, is_kf,
                seg_R_abs, seg_t_abs,
            )
            _t1 = time.perf_counter()
            _timing_acc["decode_infer"] += _t1 - _t0

            # ── sky mask（decode 完成后、filter 之前追加）────────────────
            if _sky_session is not None:
                outputs_cpu["sky_valid_mask_np"] = _compute_sky_valid_mask(
                    outputs_cpu.get("rgb_np"), packet.image_path,
                    _sky_session, _sky_mask_dir,
                    int(data_cfg.get("size", 518)),
                    bool(data_cfg.get("crop", False)),
                    int(data_cfg.get("patch_size", 14)),
                    outputs_cpu,
                )
            _t2 = time.perf_counter()
            _timing_acc["skyseg"] += _t2 - _t1

            # ── refresh 逻辑 ──────────────────────────────────────────────
            if is_kf and g > 0:
                keyframe_count += 1
                if keyframe_count % refresh_intervals == 0:
                    # 保留当前关键帧全局位姿作为新段 anchor，防止轨迹回原点
                    anchor_R = outputs_cpu.get("R_abs_np")
                    anchor_t = outputs_cpu.get("t_abs_np")
                    # 先释放 GPU 输出 dict，再测量刷新前的干净基线
                    del outputs
                    outputs = None
                    torch.cuda.empty_cache()
                    _log_cuda_memory("before_refresh_clear", g)
                    session.clear_cache_only()
                    torch.cuda.empty_cache()
                    _log_cuda_memory("after_refresh_clear", g)
                    segment_start = g
                    if anchor_R is not None and anchor_t is not None:
                        seg_R_abs = {0: anchor_R.copy()}
                        seg_t_abs = {0: anchor_t.copy()}
                    else:
                        # 兜底，正常情况不应触发
                        seg_R_abs = {0: np.eye(3, dtype=np.float32)}
                        seg_t_abs = {0: np.zeros(3, dtype=np.float32)}
                    anchor_kf_indices = torch.zeros(
                        1, 1, dtype=torch.long, device=device_str
                    )
                    anchor_frame = packet.image_tensor.to(device_str, non_blocking=True)
                    anchor_outputs = session.forward_stream(
                        anchor_frame,
                        is_keyframe=is_keyframe_t,
                        keyframe_indices=anchor_kf_indices,
                        record=False,
                    )
                    del anchor_outputs
                    del anchor_frame
                    torch.cuda.empty_cache()
                    _log_cuda_memory("after_anchor_refresh", g)

            if outputs is not None:
                del outputs
                outputs = None
            del frame_tensor

            # ── 统一过滤（一次计算，供磁盘保存 + Rerun 共用）────────────────
            _t3_start = time.perf_counter()
            _fpts, _fcols = _filter_points_colors(outputs_cpu, _out_cfg_merged)
            if _fpts is not None and len(_fpts) > 0:
                outputs_cpu["filtered_points_np"] = _fpts.astype(np.float32)
                outputs_cpu["filtered_colors_np"] = (
                    _fcols.astype(np.uint8) if _fcols is not None else None
                )
            else:
                outputs_cpu["filtered_points_np"] = None
                outputs_cpu["filtered_colors_np"] = None

            # ── dpt_unproj 统一过滤（使用 depth_conf 而非 world_points_conf）──
            if outputs_cpu.get("dpt_unproj_points_np") is not None:
                _dpu_fake = dict(outputs_cpu)
                _dpu_fake["world_points_np"] = outputs_cpu["dpt_unproj_points_np"]
                # dpt_unproj 分支使用 depth_conf（与 core/infer.py 对齐）
                _dpu_fake["conf_np"] = outputs_cpu.get("depth_conf_np")
                _dpu_fpts, _dpu_fcols = _filter_points_colors(_dpu_fake, _out_cfg_merged)
                if _dpu_fpts is not None and len(_dpu_fpts) > 0:
                    outputs_cpu["filtered_dpu_pts_np"] = _dpu_fpts.astype(np.float32)
                    outputs_cpu["filtered_dpu_cols_np"] = (
                        _dpu_fcols.astype(np.uint8) if _dpu_fcols is not None else None
                    )
                else:
                    outputs_cpu["filtered_dpu_pts_np"] = None
                    outputs_cpu["filtered_dpu_cols_np"] = None
            _t3 = time.perf_counter()
            _timing_acc["filter"] += _t3 - _t3_start

            # ── 累积位姿（用于结束后写 abs_pose.txt）────────────────────────
            R_abs_np = outputs_cpu.get("R_abs_np")
            t_abs_np = outputs_cpu.get("t_abs_np")
            if R_abs_np is not None and t_abs_np is not None:
                extri = np.eye(4, dtype=np.float32)
                extri[:3, :3] = R_abs_np
                extri[:3, 3] = t_abs_np
                _extri_list.append(extri)
                _pose_frame_ids.append(g)

            # ── 每 25 帧打印一次 VRAM 快照（方便定位长序列显存泵露）──
            if g % 25 == 0:
                _log_cuda_memory("live", g)

            # ── 增量保存到磁盘（使用预计算的 filtered_points_np，严禁 GPU Tensor）──
            _t4_start = time.perf_counter()
            _save_frame_outputs(outputs_cpu, _out_cfg_merged, _out_dir, g,
                                max_frame_pts=_max_frame_pts,
                                save_point_head=_save_point_head)

            # ── dpt_unproj 逐帧 PLY 保存 ──────────────────────────────────
            if _save_dpt_unproj and _out_cfg_merged.get("save_frame_points", False):
                _dpu_pts_f = outputs_cpu.get("filtered_dpu_pts_np")
                _dpu_cols_f = outputs_cpu.get("filtered_dpu_cols_np")
                if _dpu_pts_f is not None and len(_dpu_pts_f) > 0:
                    _dpu_save = _dpu_pts_f
                    _dpu_csave = _dpu_cols_f
                    if _max_frame_pts > 0 and len(_dpu_save) > _max_frame_pts:
                        _rng_dpu_f = np.random.default_rng(seed=g)
                        _ki_dpu = _rng_dpu_f.choice(len(_dpu_save), _max_frame_pts, replace=False)
                        _dpu_save = _dpu_save[_ki_dpu]
                        if _dpu_csave is not None:
                            _dpu_csave = _dpu_csave[_ki_dpu]
                    _dpu_dir = _os.path.join(_out_dir, "points", "dpt_unproj")
                    _os.makedirs(_dpu_dir, exist_ok=True)
                    _write_ply(_dpu_save, _os.path.join(_dpu_dir, f"frame_{g:06d}.ply"),
                               colors=_dpu_csave)
            _t4 = time.perf_counter()
            _timing_acc["save"] += _t4 - _t4_start

            # ── 全局 reservoir 更新（不受逐帧下采样预算限制，直接用完整过滤结果）──
            if _save_point_head:
                _new_pts = outputs_cpu.get("filtered_points_np")
                _new_cols = outputs_cpu.get("filtered_colors_np")
                if _new_pts is not None and len(_new_pts) > 0:
                    _new_pts = _new_pts.astype(np.float32)

                    # 若颜色缺失，尝试从 point_colors_np 用同一 sky+conf mask 恢复
                    if _new_cols is None:
                        _raw_colors = outputs_cpu.get("point_colors_np")
                        if _raw_colors is not None:
                            _rcols_flat = _raw_colors.reshape(-1, 3).astype(np.uint8)
                            _rmask = np.ones(len(_rcols_flat), dtype=bool)
                            _r_conf = outputs_cpu.get("conf_np")
                            if _r_conf is not None and _out_cfg_merged.get(
                                "confidence_filter_enabled", True
                            ):
                                _r_thr = float(_out_cfg_merged.get("confidence_threshold", 0.5))
                                _rcf = _r_conf.reshape(-1)
                                if len(_rcf) == len(_rcols_flat):
                                    _rmask &= _rcf >= _r_thr
                            _r_sky = outputs_cpu.get("sky_valid_mask_np")
                            if _r_sky is not None:
                                _rsv = _r_sky.reshape(-1) > 0
                                if len(_rsv) == len(_rcols_flat):
                                    _rmask &= _rsv
                            _rcols_filtered = _rcols_flat[_rmask]
                            if len(_rcols_filtered) == len(_new_pts):
                                _new_cols = _rcols_filtered

                    if _new_cols is not None:
                        _nc = _new_cols.astype(np.uint8, copy=False)
                        _res_pending.append(_new_pts)
                        _res_pending_cols.append(_nc)
                        _res_pending_n += len(_new_pts)
                        if _res_pending_n >= _compact_pending_points:
                            _res_pts, _res_cols, _res_pending, _res_pending_cols, _res_pending_n = \
                                _compact_reservoir(
                                    _res_pts, _res_cols,
                                    _res_pending, _res_pending_cols,
                                    _max_full_pts, _res_rng,
                                )
                    else:
                        print(
                            f"[LiveInferenceRunner/worker] 警告：帧 {g} 颜色不可用，"
                            "跳过该帧点云以保持全局点云 RGB 一致性。",
                            flush=True,
                        )

            # ── dpt_unproj reservoir 更新（分块累积，避免每帧 O(N) 复制）──────
            if _save_dpt_unproj:
                _dpu_new = outputs_cpu.get("filtered_dpu_pts_np")
                _dpu_new_cols = outputs_cpu.get("filtered_dpu_cols_np")
                if _dpu_new is not None and _dpu_new_cols is not None and len(_dpu_new) > 0:
                    _res_dpu_pending.append(_dpu_new.astype(np.float32, copy=False))
                    _res_dpu_pending_cols.append(_dpu_new_cols.astype(np.uint8, copy=False))
                    _res_dpu_pending_n += len(_dpu_new)
                    if _res_dpu_pending_n >= _compact_pending_points:
                        _res_dpu_pts, _res_dpu_cols, _res_dpu_pending, _res_dpu_pending_cols, _res_dpu_pending_n = \
                            _compact_reservoir(
                                _res_dpu_pts, _res_dpu_cols,
                                _res_dpu_pending, _res_dpu_pending_cols,
                                _max_full_pts, _res_dpu_rng,
                            )

            # ── 崩溃保护 checkpoint：每 N 帧写 points/dpt_unproj_latest.ply ──
            if _save_dpt_unproj and _checkpoint_interval > 0 and g % _checkpoint_interval == 0:
                if _res_dpu_pending_n > 0:
                    _res_dpu_pts, _res_dpu_cols, _res_dpu_pending, _res_dpu_pending_cols, _res_dpu_pending_n = \
                        _compact_reservoir(
                            _res_dpu_pts, _res_dpu_cols,
                            _res_dpu_pending, _res_dpu_pending_cols,
                            _max_full_pts, _res_dpu_rng,
                        )
                if _res_dpu_pts is not None and len(_res_dpu_pts) > 0:
                    try:
                        _ckpt_pts, _ckpt_cols = _sample_points_for_viewer(
                            _res_dpu_pts, _res_dpu_cols, _checkpoint_max_points, g + 999
                        )
                        if _ckpt_pts is not None:
                            _ckpt_dir = _os.path.join(_out_dir, "points")
                            _os.makedirs(_ckpt_dir, exist_ok=True)
                            _write_ply(_ckpt_pts, _os.path.join(_ckpt_dir, "dpt_unproj_latest.ply"),
                                       colors=_ckpt_cols)
                    except Exception as _ckpt_exc:
                        print(f"[LiveInferenceRunner/worker] checkpoint 保存失败 (frame {g}): {_ckpt_exc}",
                              flush=True)

            # ── 注入 viewer 全局快照并发送到主进程（仅 Rerun 开启时）──────────
            if _emit_frame_results:
                # ── per-frame entity 累积策略（恢复 889f81 方案，数据源改为 dpt_unproj）──
                # 不再从百万级 reservoir 采样；每帧只传当前帧下采样点到 Rerun，
                # Rerun 侧写 live/global/points/frame_XXXXXX 独立 entity，自然累积。
                _t5_start = time.perf_counter()
                _rerun_global_max: int = int(_out_cfg_merged.get(
                    "rerun_global_frame_points", _viewer_max_global
                ))
                if _save_dpt_unproj:
                    _gfp = outputs_cpu.get("filtered_dpu_pts_np")
                    _gfc = outputs_cpu.get("filtered_dpu_cols_np")
                elif _save_point_head:
                    _gfp = outputs_cpu.get("filtered_points_np")
                    _gfc = outputs_cpu.get("filtered_colors_np")
                else:
                    _gfp, _gfc = None, None
                _gfp, _gfc = _sample_points_for_viewer(_gfp, _gfc, _rerun_global_max, g + 77777)
                _viewer_payload = _build_viewer_payload(
                    outputs_cpu, g,
                    int(_out_cfg_merged.get("rerun_current_points", _max_frame_pts)),
                    _gfp, _gfc,
                )
                _t5 = time.perf_counter()
                _timing_acc["rerun_payload"] += _t5 - _t5_start
                # 丢旧保新：保证 viewer 看到最新帧而非落后多帧的旧帧
                try:
                    result_queue.put_nowait((g, _viewer_payload))
                except Exception:
                    try:
                        result_queue.get_nowait()
                        result_queue.put_nowait((g, _viewer_payload))
                    except Exception:
                        pass
                _t6 = time.perf_counter()
                _timing_acc["queue_put"] += _t6 - _t5

            # ── 帧计时统计（每 N 帧打印一次）────────────────────────────────
            _t_frame_end = time.perf_counter()
            _timing_acc["total"] += _t_frame_end - _t_frame_start
            _timing_n += 1
            if _timing_n > 0 and _timing_n % _timing_interval == 0:
                _n = float(_timing_n)
                print(
                    f"[LiveInferenceRunner/timing] frame={g}  "
                    f"avg/{_timing_interval}帧(ms): "
                    f"decode_infer={1000*_timing_acc['decode_infer']/_n:.1f}  "
                    f"skyseg={1000*_timing_acc['skyseg']/_n:.1f}  "
                    f"filter={1000*_timing_acc['filter']/_n:.1f}  "
                    f"save={1000*_timing_acc['save']/_n:.1f}  "
                    f"rerun_payload={1000*_timing_acc['rerun_payload']/_n:.1f}  "
                    f"queue_put={1000*_timing_acc['queue_put']/_n:.1f}  "
                    f"total={1000*_timing_acc['total']/_n:.1f}  "
                    f"fps={_n/max(_timing_acc['total'],1e-6):.2f}",
                    flush=True,
                )
                # 重置累积（滑动窗口，便于持续观察性能变化）
                for _k in _timing_acc:
                    _timing_acc[_k] = 0.0
                _timing_n = 0

    # ── 循环结束后刷入 pending（确保最后几帧纳入最终保存）──────────────────
    if _save_point_head and _res_pending_n > 0:
        _res_pts, _res_cols, _res_pending, _res_pending_cols, _res_pending_n = \
            _compact_reservoir(
                _res_pts, _res_cols,
                _res_pending, _res_pending_cols,
                _max_full_pts, _res_rng,
            )
    if _save_dpt_unproj and _res_dpu_pending_n > 0:
        _res_dpu_pts, _res_dpu_cols, _res_dpu_pending, _res_dpu_pending_cols, _res_dpu_pending_n = \
            _compact_reservoir(
                _res_dpu_pts, _res_dpu_cols,
                _res_dpu_pending, _res_dpu_pending_cols,
                _max_full_pts, _res_dpu_rng,
            )

    # 全局点云 PLY：从 in-memory reservoir 保存（完整过滤质量，不受逐帧预算影响）
    if _save_point_head and _res_pts is not None and len(_res_pts) > 0:
        try:
            _final_pts = _res_pts.astype(np.float32)
            _final_cols: Optional[np.ndarray] = (
                _res_cols.astype(np.uint8) if _res_cols is not None else None
            )
            if len(_final_pts) > _max_full_pts:
                _rng_final = np.random.default_rng(seed=0)
                _keep_idx = _rng_final.choice(len(_final_pts), _max_full_pts, replace=False)
                _final_pts = _final_pts[_keep_idx]
                if _final_cols is not None:
                    _final_cols = _final_cols[_keep_idx]
            _pts_dir_g = _os.path.join(_out_dir, "points", "point_head")
            _os.makedirs(_pts_dir_g, exist_ok=True)
            full_ply = _os.path.join(_out_dir, "points", "point_head_full.ply")
            _write_ply(_final_pts, full_ply, colors=_final_cols)
            np.save(_os.path.join(_out_dir, "points", "point_head_full.npy"), _final_pts)
            print(f"[LiveInferenceRunner] 全局点云已保存: {full_ply}  ({len(_final_pts)} 点)", flush=True)
        except Exception as _ply_exc:
            print(f"[LiveInferenceRunner] 全局点云保存失败: {_ply_exc}", flush=True)
    elif _save_point_head:
        print("[LiveInferenceRunner] 全局点云：无有效过滤点，跳过。", flush=True)

    # dpt_unproj 全局点云 PLY
    if _save_dpt_unproj and _res_dpu_pts is not None and len(_res_dpu_pts) > 0:
        try:
            _f_dpu = _res_dpu_pts.astype(np.float32)
            _f_dpu_cols = _res_dpu_cols.astype(np.uint8) if _res_dpu_cols is not None else None
            if len(_f_dpu) > _max_full_pts:
                _ki_dpu_f = np.random.default_rng(seed=2).choice(len(_f_dpu), _max_full_pts, replace=False)
                _f_dpu = _f_dpu[_ki_dpu_f]
                if _f_dpu_cols is not None:
                    _f_dpu_cols = _f_dpu_cols[_ki_dpu_f]
            _pts_dir_dpu = _os.path.join(_out_dir, "points", "dpt_unproj")
            _os.makedirs(_pts_dir_dpu, exist_ok=True)
            _full_dpu_ply = _os.path.join(_out_dir, "points", "dpt_unproj_full.ply")
            _write_ply(_f_dpu, _full_dpu_ply, colors=_f_dpu_cols)
            np.save(_os.path.join(_out_dir, "points", "dpt_unproj_full.npy"), _f_dpu)
            print(f"[LiveInferenceRunner] dpt_unproj 全局点云已保存: {_full_dpu_ply}  ({len(_f_dpu)} 点)", flush=True)
        except Exception as _dpu_exc:
            print(f"[LiveInferenceRunner] dpt_unproj 全局点云保存失败: {_dpu_exc}", flush=True)
    elif _save_dpt_unproj:
        print(
            "[LiveInferenceRunner] dpt_unproj 全局点云：无有效点，未生成 dpt_unproj_full.ply。"
            "请检查以下日志确认原因："
            "① pose_enc/rel_pose_enc 不可用（无法计算 intri）"
            "② depth_conf 过滤将所有点过滤掉了（调高 confidence_threshold）"
            "③ dpt_unproj 计算异常（查看上方 警告 日志）",
            flush=True,
        )

    # dpt_unproj 警告总展示（如果有超过限频部分的失败帧）
    if _dpu_warn_count > _dpu_warn_max:
        print(
            f"[LiveInferenceRunner] dpt_unproj: 警告总计 {_dpu_warn_count} 帧失败，"
            f"已抑制输出前 {_dpu_warn_max} 条。"
            "请检查 rel_pose_enc/pose_enc 输出或 depth_conf 配置。",
            flush=True,
        )

    # 位姿文件
    if _extri_list:
        try:
            from longstream.io.save_poses_txt import save_w2c_txt as _save_w2c_txt
            _poses_dir = _os.path.join(_out_dir, "poses")
            _os.makedirs(_poses_dir, exist_ok=True)
            _save_w2c_txt(
                _os.path.join(_poses_dir, "abs_pose.txt"),
                np.stack(_extri_list, axis=0),
                _pose_frame_ids,
            )
            print(f"[LiveInferenceRunner] 位姿已保存: {_os.path.join(_poses_dir, 'abs_pose.txt')}  ({len(_extri_list)} 帧)", flush=True)
        except Exception as _pose_exc:
            print(f"[LiveInferenceRunner] abs_pose.txt 保存失败: {_pose_exc}", flush=True)

    # 视频合成（从已落盘 RGB / dpt_plasma 序列）
    if _out_cfg_merged.get("save_videos", False):
        _fps_out = int(cfg.get("data", {}).get("fps", 0) or 24)
        _assemble_video(_out_dir, fps=_fps_out)
        _assemble_depth_video(_out_dir, fps=_fps_out)

    # 说明暂不导出的非流式对齐项
    print(
        "[LiveInferenceRunner] 注意: 流式模式暂不导出 poses/intri.txt、poses/rel_pose.txt。"
        "内镜标定/相对位姿需要非流式推理链路（BatchInferenceRunner）才可靠。",
        flush=True,
    )

    # GLB 导出（基于已落盘逐帧 PLY）
    if _out_cfg_merged.get("export_glb", False):
        _export_glb_from_plys(_out_dir)

    session.clear()
    torch.cuda.empty_cache()
    # ── 资源监控：推理结束后停止 ─────────────────────────────────────────
    if _live_monitor is not None:
        try:
            _live_monitor.stop()
        except Exception:
            pass
    print("[LiveInferenceRunner/worker] 推理循环正常结束", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  LiveInferenceRunner
# ═══════════════════════════════════════════════════════════════════════════

class LiveInferenceRunner:
    """
    实时推理运行器。内部使用独立子进程（multiprocessing.Process）运行模型。

    为什么用进程而非线程？
      - 模型 forward / TTO 可能耗时数秒，线程无法被强制打断；
      - 进程可通过 terminate() / kill() 立即释放 GPU，确保停止按钮可靠。

    Parameters
    ----------
    cfg : dict
        完整配置字典（与 YAML 结构一致）。
    on_frame : callable, optional
        每帧推理完成后回调，在主进程轮询线程中调用。
        签名：on_frame(frame_idx: int, outputs_cpu: dict) -> None
    feeder :
        保留参数（已废弃）。子进程无法接收不可 pickle 的对象，
        feeder 始终由子进程从 cfg["data"] 自行构建。
    """

    _QUEUE_MAXSIZE = 5   # 结果队列上限；超过则丢弃最新帧，减少共享内存压力

    def __init__(
        self,
        cfg: dict,
        on_frame: Optional[Callable[[int, dict], None]] = None,
        feeder=None,  # 保留签名兼容性，不实际使用
    ) -> None:
        self.cfg = cfg
        self.on_frame = on_frame

        self._process: Optional[mp.Process] = None
        self._result_queue: mp.Queue = mp.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._stop_flag: mp.Event = mp.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    # ── 公开控制接口 ──────────────────────────────────────────────────────

    def start(self) -> None:
        """启动子进程开始推理，同时在主进程开启轮询线程接收结果。"""
        if self._process is not None and self._process.is_alive():
            return
        self._stop_flag.clear()
        # 清空旧结果
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Exception:
                break
        self._error = None

        self._process = mp.Process(
            target=_worker_main,
            args=(self.cfg, self._result_queue, self._stop_flag),
            name="LiveInferenceWorker",
            daemon=True,
        )
        self._process.start()

        self._poll_thread = threading.Thread(
            target=self._poll_results,
            name="LiveInferenceRunner-poll",
            daemon=True,
        )
        self._poll_thread.start()

    def stop(self) -> None:
        """
        停止推理进程。策略：
          1. 发送停止信号，等待 3s（让循环正常退出）
          2. 如仍存活，发送 terminate()，等待 3s
          3. 如仍存活，发送 kill()，等待 2s
        """
        if self._process is None:
            return

        self._stop_flag.set()
        self._process.join(timeout=3)

        if self._process.is_alive():
            print("[LiveInferenceRunner] 子进程未在 3s 内退出，发送 terminate()...", flush=True)
            self._process.terminate()
            self._process.join(timeout=3)

        if self._process.is_alive():
            print("[LiveInferenceRunner] 子进程仍未退出，发送 kill()...", flush=True)
            self._process.kill()
            self._process.join(timeout=2)

        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2)

        print("[LiveInferenceRunner] 子进程已终止", flush=True)

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def last_error(self) -> Optional[Exception]:
        return self._error

    # ── 内部实现 ──────────────────────────────────────────────────────────

    def _poll_results(self) -> None:
        """主进程轮询线程：读取子进程结果队列并调用 on_frame 回调。"""
        while self._process is not None and (
            self._process.is_alive() or not self._result_queue.empty()
        ):
            try:
                item = self._result_queue.get(timeout=0.2)
            except Exception:
                continue

            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                self._error = RuntimeError(item[1])
                print(f"[LiveInferenceRunner] 子进程错误:\n{item[1]}", flush=True)
            elif self.on_frame is not None:
                try:
                    frame_idx, outputs_cpu = item
                    self.on_frame(frame_idx, outputs_cpu)
                except Exception as cb_exc:
                    print(
                        f"[LiveInferenceRunner] on_frame 回调异常: {cb_exc}",
                        flush=True,
                    )


# ═══════════════════════════════════════════════════════════════════════════
#  工具函数（模块私有）
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  增量磁盘保存工具函数
# ═══════════════════════════════════════════════════════════════════════════

import os as _os  # noqa: E402 — 避免顶部 import 顺序约束


def _write_ply(pts: np.ndarray, path: str, colors: np.ndarray = None) -> None:
    """将 [N, 3] float32 数组写出为二进制 PLY 文件（支持可选颜色）。"""
    from longstream.io.save_points import save_pointcloud
    save_pointcloud(path, pts, colors=colors)


def _get_stream_seq_name(cfg: dict) -> str:
    """
    从 cfg 中解析序列名称（与 LongStreamDataLoader 命名规则一致）。
    generalizable 格式取 seq_list[0]，其他格式返回 'live_stream'。
    """
    data = cfg.get("data", {})
    seq_list = data.get("seq_list")
    if data.get("format") == "generalizable" and seq_list:
        seq = str(seq_list[0])
        return seq.replace(_os.path.sep, "_").replace("/", "_")
    # 视频 / image_dir：取文件/目录名
    src = data.get("img_path", data.get("video_path", ""))
    if src:
        base = _os.path.splitext(_os.path.basename(src.rstrip("/\\")))[0]
        if base:
            return base
    return "live_stream"


def _compute_sky_valid_mask(
    rgb_np: Optional[np.ndarray],
    image_path: Optional[str],
    sky_session,
    mask_dir: Optional[str],
    size: int,
    crop: bool,
    patch_size: int,
    outputs_cpu: dict,
) -> Optional[np.ndarray]:
    """
    对单帧图像计算 sky valid mask（True = 非天空 = 保留），
    并调整到与 world_points 相同的分辨率。

    优先使用已解码的 rgb_np（避免重复磁盘读取），
    仅当 mask_dir 不为 None 时才将 mask PNG 保存到磁盘。

    返回 [H_pts * W_pts] bool ndarray 或 None（失败时降级）。
    所有操作均为纯 CPU，无 GPU 张量。
    """
    try:
        import cv2 as _cv2
        import copy as _copy
        from longstream.utils.sky_mask import (
            run_skyseg, _normalize_skyseg_output, sky_mask_filename, SKYSEG_THRESHOLD
        )

        # ── 获取 BGR 图像：优先复用已解码 rgb_np，避免额外磁盘 IO ────────
        if rgb_np is not None and len(rgb_np.shape) == 3:
            img_bgr = rgb_np[:, :, ::-1].copy()
        elif image_path:
            img_bgr = _cv2.imread(image_path)
        else:
            return None
        if img_bgr is None:
            return None

        # ── 运行 skyseg ────────────────────────────────────────────────
        result_map = run_skyseg(sky_session, [320, 320], img_bgr)
        result_map_full = _cv2.resize(
            result_map, (img_bgr.shape[1], img_bgr.shape[0])
        )
        result_map_full = _normalize_skyseg_output(result_map_full)
        # 0=天空（遮挡），255=保留；valid = 非天空
        sky_raw = np.zeros(result_map_full.shape, dtype=np.uint8)
        sky_raw[result_map_full < SKYSEG_THRESHOLD] = 255  # 非天空像素=255

        # 保存 sky mask 到磁盘（仅 _save_sky_mask=True 且 mask_dir 不为 None 时）
        if mask_dir is not None and image_path:
            _os.makedirs(mask_dir, exist_ok=True)
            fname = sky_mask_filename(image_path)
            _cv2.imwrite(_os.path.join(mask_dir, fname), sky_raw)

        # ── 调整到 world_points 分辨率（复用与非流式 core/infer.py 相同的处理逻辑）──
        pts = outputs_cpu.get("world_points_np")
        if pts is None:
            return None
        # 从 depth_np 的 shape 推断点云分辨率
        depth = outputs_cpu.get("depth_np")
        if depth is not None:
            H_pts, W_pts = depth.shape[:2]
        else:
            n = len(pts)
            side = int(n ** 0.5)
            H_pts, W_pts = side, side

        # _prepare_mask_for_model 处理 long-edge resize + crop/resize + 最终 resize 到 target_shape
        # 与非流式 infer.py L642-L648 使用相同参数，避免 crop=True 时 sky mask 错位
        try:
            from longstream.core.infer import _prepare_mask_for_model as _pmfm
            sky_resized = _pmfm(
                sky_raw,
                size=size,
                crop=crop,
                patch_size=patch_size,
                target_shape=(H_pts, W_pts),
            )
        except Exception:
            # 降级：简单 resize
            sky_resized = _cv2.resize(sky_raw, (W_pts, H_pts), interpolation=_cv2.INTER_NEAREST)

        if sky_resized is None:
            return None
        valid_mask = sky_resized.reshape(-1) > 0  # True = 保留（非天空）
        return valid_mask
    except Exception as _sky_err:
        return None


def _downsample_points(
    pts: np.ndarray,
    colors: Optional[np.ndarray],
    max_pts: int,
    rng_seed: int = 0,
) -> tuple:
    """
    随机下采样点云到 max_pts 个点（用于每帧全局 preview chunk）。
    返回 (pts_down, colors_down)，均为 numpy，不持有 GPU 引用。
    """
    n = len(pts)
    if n <= max_pts:
        return pts, colors
    rng = np.random.default_rng(seed=rng_seed)
    keep = rng.choice(n, max_pts, replace=False)
    pts_d = pts[keep]
    cols_d = colors[keep] if colors is not None else None
    return pts_d, cols_d


def _filter_points_colors(
    outputs_cpu: dict,
    out_cfg: dict,
) -> tuple:
    """
    从 outputs_cpu 提取世界点云和颜色，按置信度 + sky mask 过滤后返回。

    支持的过滤：
      1. 置信度（out_cfg["confidence_filter_enabled"] != False 时启用）
      2. sky valid mask（outputs_cpu["sky_valid_mask_np"] 存在时合并）

    所有路径（逐帧 PLY、全局 chunk、Rerun 当前帧/全局帧）统一调用此函数，
    确保过滤逻辑与非流式推理（infer.py）保持一致。

    Returns
    -------
    (pts_filtered, colors_filtered)
        pts_filtered   : [N, 3] float32，或 None（无有效点云）
        colors_filtered: [N, 3] uint8，或 None（无颜色数据）
    """
    pts = outputs_cpu.get("world_points_np")
    if pts is None:
        return None, None

    conf = outputs_cpu.get("conf_np")
    colors = outputs_cpu.get("point_colors_np")
    thr = float(out_cfg.get(
        "confidence_threshold",
        out_cfg.get("max_conf_filter_threshold", 0.5),
    ))

    pts_flat = pts.reshape(-1, 3).astype(np.float32)
    colors_flat = colors.reshape(-1, 3).astype(np.uint8) if colors is not None else None

    # 置信度过滤（默认启用；显式 False 时跳过）
    conf_mask: Optional[np.ndarray] = None
    if out_cfg.get("confidence_filter_enabled", True) and conf is not None:
        conf_flat = conf.reshape(-1)
        if len(conf_flat) == len(pts_flat):
            conf_mask = conf_flat >= thr

    # sky mask 过滤
    sky_mask: Optional[np.ndarray] = None
    sky_raw = outputs_cpu.get("sky_valid_mask_np")
    if sky_raw is not None:
        sky_mask_flat = sky_raw.reshape(-1) > 0
        if len(sky_mask_flat) == len(pts_flat):
            sky_mask = sky_mask_flat

    # 合并两个 mask（AND）
    valid: Optional[np.ndarray] = None
    if conf_mask is not None and sky_mask is not None:
        valid = conf_mask & sky_mask
    elif conf_mask is not None:
        valid = conf_mask
    elif sky_mask is not None:
        valid = sky_mask

    if valid is not None:
        pts_flat = pts_flat[valid]
        if colors_flat is not None:
            colors_flat = colors_flat[valid]

    return pts_flat, colors_flat


# ───────────────────────────────────────────────────────────────────────────
#  全局点云 reservoir 辅助函数
# ───────────────────────────────────────────────────────────────────────────

def _update_point_reservoir(
    res_pts: Optional[np.ndarray],
    res_cols: Optional[np.ndarray],
    new_pts: Optional[np.ndarray],
    new_cols: Optional[np.ndarray],
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    有界 reservoir 更新：合并 new_pts/new_cols 入 res_pts/res_cols，
    始终将总点数控制在 max_points 以内，无两倍膨胀再下采样。
    """
    if new_pts is None or new_cols is None or len(new_pts) == 0:
        return res_pts, res_cols
    new_pts = new_pts.astype(np.float32, copy=False)
    new_cols = new_cols.astype(np.uint8, copy=False)
    if res_pts is None:
        if max_points and len(new_pts) > max_points:
            keep = rng.choice(len(new_pts), max_points, replace=False)
            return new_pts[keep], new_cols[keep]
        return new_pts.copy(), new_cols.copy()
    total = len(res_pts) + len(new_pts)
    if not max_points or total <= max_points:
        return (
            np.concatenate([res_pts, new_pts], axis=0),
            np.concatenate([res_cols, new_cols], axis=0),
        )
    keep = rng.choice(total, max_points, replace=False)
    old_keep = keep[keep < len(res_pts)]
    new_keep = keep[keep >= len(res_pts)] - len(res_pts)
    return (
        np.concatenate([res_pts[old_keep], new_pts[new_keep]], axis=0),
        np.concatenate([res_cols[old_keep], new_cols[new_keep]], axis=0),
    )


def _sample_points_for_viewer(
    pts: Optional[np.ndarray],
    cols: Optional[np.ndarray],
    max_points: int,
    seed: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从 reservoir 下采样供 Rerun viewer 显示，不修改原 reservoir。"""
    if pts is None or cols is None or len(pts) == 0:
        return None, None
    pts = pts.astype(np.float32, copy=False)
    cols = cols.astype(np.uint8, copy=False)
    if max_points is not None and max_points > 0 and len(pts) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(pts), int(max_points), replace=False)
        pts = pts[keep]
        cols = cols[keep]
    return pts, cols


def _compact_reservoir(
    res_pts: Optional[np.ndarray],
    res_cols: Optional[np.ndarray],
    pending_pts: list,
    pending_cols: list,
    max_points: int,
    rng: np.random.Generator,
):
    """
    将 pending 小分块合并入已紧凑化的 res_pts，结果控制在 max_points 以内。
    返回 (new_res_pts, new_res_cols, [], [], 0) — 调用方直接解包五元组。
    """
    if not pending_pts:
        return res_pts, res_cols, [], [], 0
    chunks_pts = ([] if res_pts is None else [res_pts]) + pending_pts
    chunks_cols = ([] if res_cols is None else [res_cols]) + pending_cols
    merged_pts = np.concatenate(chunks_pts, axis=0)
    merged_cols = np.concatenate(chunks_cols, axis=0)
    if max_points and len(merged_pts) > max_points:
        keep = rng.choice(len(merged_pts), max_points, replace=False)
        merged_pts = merged_pts[keep]
        merged_cols = merged_cols[keep]
    return merged_pts, merged_cols, [], [], 0


def _sample_global_snapshot(
    res_pts: Optional[np.ndarray],
    res_cols: Optional[np.ndarray],
    pending_pts: list,
    pending_cols: list,
    max_points: int,
    seed: int,
) -> tuple:
    """
    从全局 reservoir（已紧凑化数组 + pending 分块列表）中随机采样 max_points 点，
    不修改 reservoir 本身，不触发 compact。
    每帧调用，确保 Rerun global 快照与最终保存使用同一套数据。

    返回 (pts: np.ndarray | None, cols: np.ndarray | None)。
    """
    chunks_p = ([] if res_pts is None else [res_pts]) + list(pending_pts or [])
    chunks_c = ([] if res_cols is None else [res_cols]) + list(pending_cols or [])
    chunks = [
        (p, c) for p, c in zip(chunks_p, chunks_c)
        if p is not None and c is not None and len(p) > 0
    ]
    if not chunks:
        return None, None

    sizes = np.array([len(p) for p, _ in chunks], dtype=np.int64)
    total = int(sizes.sum())

    if max_points <= 0 or total <= max_points:
        return (
            np.concatenate([p for p, _ in chunks], axis=0).astype(np.float32, copy=False),
            np.concatenate([c for _, c in chunks], axis=0).astype(np.uint8, copy=False),
        )

    rng = np.random.default_rng(seed)
    global_idx = rng.choice(total, int(max_points), replace=False)
    global_idx.sort()

    out_p: list = []
    out_c: list = []
    offset = 0
    for (p, c), n in zip(chunks, sizes):
        local = global_idx[(global_idx >= offset) & (global_idx < offset + n)] - offset
        if len(local):
            out_p.append(p[local])
            out_c.append(c[local])
        offset += n

    if not out_p:
        return None, None
    return np.concatenate(out_p, axis=0), np.concatenate(out_c, axis=0)


def _build_viewer_payload(
    outputs_cpu: dict,
    frame_idx: int,
    max_frame_points: int,
    global_frame_pts: Optional[np.ndarray],
    global_frame_cols: Optional[np.ndarray],
) -> dict:
    """
    构建瘦 payload 发送给 Rerun viewer 进程。
    仅包含可视化所需字段，避免跨进程 pickle 传输大型滤波数组。

    current 字段：优先 dpt_unproj，颜色缺失时从 point_colors_np 兜底（同一 mask）。
    global 字段：已由调用方按 rerun_global_frame_points 下采样的当前帧点。
    """
    # ── 当前帧点云：优先 dpt_unproj，回退 point_head ──────────────────
    cur_pts = outputs_cpu.get("filtered_dpu_pts_np")
    cur_cols = outputs_cpu.get("filtered_dpu_cols_np")

    # 颜色兜底：dpt_unproj pts 存在但颜色为空时，从 point_colors_np 恢复
    if cur_pts is not None and len(cur_pts) > 0 and cur_cols is None:
        _raw_colors = outputs_cpu.get("point_colors_np")
        _conf = outputs_cpu.get("depth_conf_np") or outputs_cpu.get("conf_np")
        if _raw_colors is not None:
            _rcf = _raw_colors.reshape(-1, 3).astype(np.uint8)
            _rmask = np.ones(len(_rcf), dtype=bool)
            if _conf is not None:
                _cf_flat = _conf.reshape(-1)
                if len(_cf_flat) == len(_rcf):
                    _rmask &= _cf_flat >= 0.5
            _sky = outputs_cpu.get("sky_valid_mask_np")
            if _sky is not None:
                _sv = _sky.reshape(-1) > 0
                if len(_sv) == len(_rcf):
                    _rmask &= _sv
            _rcf = _rcf[_rmask]
            if len(_rcf) == len(cur_pts):
                cur_cols = _rcf

    if cur_pts is None or cur_cols is None:
        cur_pts = outputs_cpu.get("filtered_points_np")
        cur_cols = outputs_cpu.get("filtered_colors_np")

    cur_pts, cur_cols = _sample_points_for_viewer(cur_pts, cur_cols, max_frame_points, frame_idx)
    return {
        "rgb_np":                        outputs_cpu.get("rgb_np"),
        "depth_np":                      outputs_cpu.get("depth_np"),
        "center_np":                     outputs_cpu.get("center_np"),
        "viewer_current_points_np":      cur_pts,
        "viewer_current_colors_np":      cur_cols,
        "viewer_global_frame_points_np": global_frame_pts,
        "viewer_global_frame_colors_np": global_frame_cols,
    }


def _save_frame_outputs(
    outputs_cpu: dict,
    out_cfg: dict,
    out_dir: str,
    frame_idx: int,
    max_frame_pts: int = 8000,
    save_point_head: bool = True,
) -> None:
    """
    增量保存单帧输出到磁盘。

    目录结构与非流式输出对齐：
      out_dir/images/rgb/frame_XXXXXX.png
      out_dir/depth/dpt/frame_XXXXXX.npy
      out_dir/depth/dpt_plasma/frame_XXXXXX.png
      out_dir/points/point_head/frame_XXXXXX.ply（按 max_frame_pts 下采样）

    所有输入均为纯 numpy（已完成 .detach().cpu().numpy()）。
    本函数绝对不导入 torch，也不持有任何 GPU 对象引用。
    """
    import PIL.Image  # 仅在子进程中需要，延迟导入

    _os.makedirs(out_dir, exist_ok=True)

    # ── RGB PNG ───────────────────────────────────────────────────────────
    # 目录：images/rgb/  文件：frame_XXXXXX.png
    if out_cfg.get("save_images", False):
        rgb_np = outputs_cpu.get("rgb_np")
        if rgb_np is not None:
            rgb_dir = _os.path.join(out_dir, "images", "rgb")
            _os.makedirs(rgb_dir, exist_ok=True)
            PIL.Image.fromarray(rgb_np).save(
                _os.path.join(rgb_dir, f"frame_{frame_idx:06d}.png")
            )

    # ── 深度图：dpt (.npy) + dpt_plasma (可视化 PNG) ─────────────────────
    if out_cfg.get("save_depth", False):
        depth_np = outputs_cpu.get("depth_np")
        if depth_np is not None:
            dpt_dir = _os.path.join(out_dir, "depth", "dpt")
            _os.makedirs(dpt_dir, exist_ok=True)
            np.save(_os.path.join(dpt_dir, f"frame_{frame_idx:06d}.npy"), depth_np)
            # dpt_plasma 伪彩色 PNG：save_depth_vis=false 时跳过（默认 True 向后兼容）
            if out_cfg.get("save_depth_vis", True):
                plasma_dir = _os.path.join(out_dir, "depth", "dpt_plasma")
                _os.makedirs(plasma_dir, exist_ok=True)
                # plasma 伪彩色可视化：复用 longstream.utils.depth.colorize_depth
                try:
                    from longstream.utils.depth import colorize_depth as _colorize_depth
                    d_vis = _colorize_depth(depth_np, cmap="plasma")  # uint8 [H,W,3]
                except Exception:
                    # 降级：灰度归一化（应急）
                    d_min, d_max = depth_np.min(), depth_np.max()
                    if d_max > d_min:
                        d_vis = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                    else:
                        d_vis = np.zeros_like(depth_np, dtype=np.uint8)
                    d_vis = np.stack([d_vis, d_vis, d_vis], axis=-1)
                PIL.Image.fromarray(d_vis).save(
                    _os.path.join(plasma_dir, f"frame_{frame_idx:06d}.png")
                )

    # ── 逐帧点云 PLY（使用预计算的 filtered_points_np，与 Rerun 保持完全一致）──
    # 按 max_frame_pts 下采样，与非流式 core/infer.py 相同语义
    # 目录：points/point_head/  文件：frame_XXXXXX.ply
    if save_point_head and out_cfg.get("save_frame_points", False):
        pts_filtered = outputs_cpu.get("filtered_points_np")
        colors_filtered = outputs_cpu.get("filtered_colors_np")
        if pts_filtered is None:
            pts_filtered, colors_filtered = _filter_points_colors(outputs_cpu, out_cfg)
        if pts_filtered is not None and len(pts_filtered) > 0:
            if max_frame_pts > 0 and len(pts_filtered) > max_frame_pts:
                _rng_frame = np.random.default_rng(seed=frame_idx)
                _ki = _rng_frame.choice(len(pts_filtered), max_frame_pts, replace=False)
                pts_filtered = pts_filtered[_ki]
                if colors_filtered is not None:
                    colors_filtered = colors_filtered[_ki]
            pts_dir = _os.path.join(out_dir, "points", "point_head")
            _os.makedirs(pts_dir, exist_ok=True)
            _write_ply(pts_filtered, _os.path.join(pts_dir, f"frame_{frame_idx:06d}.ply"),
                       colors=colors_filtered)
    # ── 轨迹（追加写，路径与非流式 poses/ 对齐）──────────────────────────
    center = outputs_cpu.get("center_np")
    if center is not None:
        poses_dir = _os.path.join(out_dir, "poses")
        _os.makedirs(poses_dir, exist_ok=True)
        traj_path = _os.path.join(poses_dir, "trajectory.txt")
        with open(traj_path, "a", encoding="utf-8") as f:
            f.write(
                f"{frame_idx} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n"
            )


# DEPRECATED: no longer called; kept for reference only
def _save_global_points(
    pts_list: list,
    out_cfg: dict,
    out_dir: str,
    max_pts: int = 800000,
    colors_list: list = None,
) -> None:
    """
    将循环期间累积的世界坐标点 [N, 3] 合并、按置信度均匀下采样，
    写出全局点云 PLY。
    pts_list: list of np.ndarray [N_i, 3] float32
    colors_list: list of (np.ndarray [N_i, 3] uint8 or None), optional
    """
    if not pts_list:
        return
    try:
        all_pts = np.concatenate(pts_list, axis=0)   # [Total, 3]
    except Exception as exc:
        print(f"[LiveInferenceRunner] 全局点云合并失败: {exc}", flush=True)
        return

    # 合并颜色（如果存在且与点数对应）
    all_colors = None
    if colors_list is not None and len(colors_list) == len(pts_list):
        valid = [c for c in colors_list if c is not None]
        if len(valid) == len(pts_list):
            try:
                all_colors = np.concatenate(colors_list, axis=0).astype(np.uint8)
            except Exception:
                all_colors = None

    N = len(all_pts)
    if N > max_pts:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(N, max_pts, replace=False)
        all_pts = all_pts[idx]
        if all_colors is not None:
            all_colors = all_colors[idx]

    pts_dir = _os.path.join(out_dir, "global_points")
    _os.makedirs(pts_dir, exist_ok=True)
    ply_path = _os.path.join(pts_dir, "global_pointcloud.ply")
    _write_ply(all_pts, ply_path, colors=all_colors)
    print(f"[LiveInferenceRunner] 全局点云已保存: {ply_path}  ({len(all_pts)} 点)", flush=True)


def _assemble_video(out_dir: str, fps: int = 24) -> None:
    """
    将 out_dir/images/rgb/*.png 序列合成为 out_dir/images/rgb.mp4。
    优先使用 longstream.io.save_images.save_video（ffmpeg glob 模式）；
    回退到 imageio v2 FFMPEG writer。
    纯 CPU/磁盘操作，不使用任何 GPU 张量。
    """
    import glob

    img_dir = _os.path.join(out_dir, "images", "rgb")
    if not _os.path.isdir(img_dir):
        print("[LiveInferenceRunner] save_videos 跳过：images/rgb/ 目录不存在。", flush=True)
        return

    frames = sorted(glob.glob(_os.path.join(img_dir, "*.png")))
    if not frames:
        print("[LiveInferenceRunner] save_videos 跳过：images/rgb/ 目录为空。", flush=True)
        return

    out_video = _os.path.join(out_dir, "images", "rgb.mp4")
    fps_actual = max(1, int(fps))

    # 优先方案：复用 save_video（调用 ffmpeg subprocess，走 glob 模式）
    try:
        from longstream.io.save_images import save_video
        pattern = _os.path.join(img_dir, "*.png")
        save_video(out_video, pattern, fps=fps_actual)
        print(f"[LiveInferenceRunner] 视频已保存: {out_video}", flush=True)
        return
    except Exception as exc:
        print(f"[LiveInferenceRunner] save_video 失败: {exc}，尝试 imageio...", flush=True)

    # 回退方案：imageio v2 FFMPEG
    try:
        import imageio.v2 as imageio_v2
        import PIL.Image
        writer = imageio_v2.get_writer(out_video, format="FFMPEG", fps=fps_actual,
                                       codec="libx264", pixelformat="yuv420p")
        for fp in frames:
            frame = np.array(PIL.Image.open(fp))
            writer.append_data(frame)
        writer.close()
        print(f"[LiveInferenceRunner] 视频已保存: {out_video}", flush=True)
    except Exception as exc2:
        print(f"[LiveInferenceRunner] 视频合成失败: {exc2}", flush=True)


def _log_cuda_memory(prefix: str, frame_idx: int) -> None:
    """Print a single-line VRAM snapshot (allocated / reserved / peak) for diagnostics."""
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            return
        allocated    = _torch.cuda.memory_allocated()    / 1024 ** 3
        reserved     = _torch.cuda.memory_reserved()     / 1024 ** 3
        max_alloc    = _torch.cuda.max_memory_allocated() / 1024 ** 3
        print(
            f"[VRAM] {prefix} frame={frame_idx} "
            f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
            f"peak={max_alloc:.2f}GiB",
            flush=True,
        )
    except Exception:
        pass


def _assemble_depth_video(out_dir: str, fps: int = 24) -> None:
    """
    将 out_dir/depth/dpt_plasma/*.png 序列合成为 out_dir/depth/dpt_plasma.mp4。
    与非流式 core/infer.py 对齐。
    纯 CPU/磁盘操作，不使用任何 GPU 张量。
    """
    import glob

    plasma_dir = _os.path.join(out_dir, "depth", "dpt_plasma")
    if not _os.path.isdir(plasma_dir):
        print("[LiveInferenceRunner] depth video 跳过：depth/dpt_plasma/ 目录不存在。", flush=True)
        return

    frames = sorted(glob.glob(_os.path.join(plasma_dir, "*.png")))
    if not frames:
        print("[LiveInferenceRunner] depth video 跳过：depth/dpt_plasma/ 目录为空。", flush=True)
        return

    out_video = _os.path.join(out_dir, "depth", "dpt_plasma.mp4")
    fps_actual = max(1, int(fps))

    try:
        from longstream.io.save_images import save_video
        pattern = _os.path.join(plasma_dir, "*.png")
        save_video(out_video, pattern, fps=fps_actual)
        print(f"[LiveInferenceRunner] 深度视频已保存: {out_video}", flush=True)
        return
    except Exception as exc:
        print(f"[LiveInferenceRunner] save_video (depth) 失败: {exc}，尝试 imageio...", flush=True)

    try:
        import imageio.v2 as imageio_v2
        import PIL.Image
        writer = imageio_v2.get_writer(out_video, format="FFMPEG", fps=fps_actual,
                                       codec="libx264", pixelformat="yuv420p")
        for fp in frames:
            frame = np.array(PIL.Image.open(fp))
            if frame.ndim == 2:  # 灰度应急兜底
                frame = np.stack([frame, frame, frame], axis=-1)
            writer.append_data(frame)
        writer.close()
        print(f"[LiveInferenceRunner] 深度视频已保存: {out_video}", flush=True)
    except Exception as exc2:
        print(f"[LiveInferenceRunner] 深度视频合成失败: {exc2}", flush=True)


# DEPRECATED: live/ duplicate writes removed; kept for reference only
def _save_live_outputs(
    outputs_cpu: dict,
    out_cfg: dict,
    out_dir: str,
    frame_idx: int,
    max_frame_pts: int = 8000,
    save_point_head: bool = True,
    save_dpt_unproj: bool = True,
) -> None:
    """将与 Rerun 同源的数据落盘到 live/ 子目录，方便后处理或对比。

    语义变更：
      - live/current/points/   受 save_point_head 控制（每帧覆盖）
      - live/current/dpt_unproj/ 受 save_dpt_unproj 控制（每帧覆盖）
      - 不再写 live/global/points/frame_*.ply 和 live/global/dpt_unproj/frame_*.ply
        （全局点云由 in-memory reservoir 在正常结束时统一写出）
    """
    import PIL.Image as _PILImage

    # live/rgb/frame_N.png
    rgb_np = outputs_cpu.get("rgb_np")
    if rgb_np is not None and out_cfg.get("save_images", False):
        _lrgb = _os.path.join(out_dir, "live", "rgb")
        _os.makedirs(_lrgb, exist_ok=True)
        _PILImage.fromarray(rgb_np).save(_os.path.join(_lrgb, f"frame_{frame_idx:06d}.png"))

    # live/depth/frame_N.npy
    depth_np = outputs_cpu.get("depth_np")
    if depth_np is not None and out_cfg.get("save_depth", False):
        _ldep = _os.path.join(out_dir, "live", "depth")
        _os.makedirs(_ldep, exist_ok=True)
        np.save(_os.path.join(_ldep, f"frame_{frame_idx:06d}.npy"), depth_np)

    # live/current/points/frame_N.ply（point_head，受 save_point_head 控制）
    pts = outputs_cpu.get("filtered_points_np")
    cols = outputs_cpu.get("filtered_colors_np")
    if save_point_head and pts is not None and len(pts) > 0 and cols is not None and out_cfg.get("save_points", False):
        cur_p, cur_c = pts, cols
        if max_frame_pts > 0 and len(cur_p) > max_frame_pts:
            rng_c = np.random.default_rng(seed=frame_idx)
            ki_c = rng_c.choice(len(cur_p), max_frame_pts, replace=False)
            cur_p, cur_c = cur_p[ki_c], cur_c[ki_c]
        _lcur = _os.path.join(out_dir, "live", "current", "points")
        _os.makedirs(_lcur, exist_ok=True)
        _write_ply(cur_p, _os.path.join(_lcur, f"frame_{frame_idx:06d}.ply"), colors=cur_c)

    # live/current/dpt_unproj/frame_N.ply（受 save_dpt_unproj 控制）
    dpu_pts = outputs_cpu.get("filtered_dpu_pts_np")
    dpu_cols = outputs_cpu.get("filtered_dpu_cols_np")
    if save_dpt_unproj and dpu_pts is not None and len(dpu_pts) > 0 and dpu_cols is not None and out_cfg.get("save_points", False):
        dpu_cur, dpu_cur_c = dpu_pts, dpu_cols
        if max_frame_pts > 0 and len(dpu_cur) > max_frame_pts:
            rng_dc = np.random.default_rng(seed=frame_idx + 200000)
            ki_dc = rng_dc.choice(len(dpu_cur), max_frame_pts, replace=False)
            dpu_cur, dpu_cur_c = dpu_cur[ki_dc], dpu_cur_c[ki_dc]
        _ldpu_cur = _os.path.join(out_dir, "live", "current", "dpt_unproj")
        _os.makedirs(_ldpu_cur, exist_ok=True)
        _write_ply(dpu_cur, _os.path.join(_ldpu_cur, f"frame_{frame_idx:06d}.ply"), colors=dpu_cur_c)


def _export_glb_from_plys(out_dir: str) -> None:
    """将已落盘的逐帧 PLY 合并导出为 GLB（使用 trimesh；不可用则跳过）。"""
    import glob

    pts_dir = _os.path.join(out_dir, "points", "point_head")
    if not _os.path.isdir(pts_dir):
        return
    try:
        import trimesh
    except ImportError:
        print("[LiveInferenceRunner] GLB 导出跳过：trimesh 未安装", flush=True)
        return

    ply_files = sorted(glob.glob(_os.path.join(pts_dir, "*.ply")))
    if not ply_files:
        return
    meshes = []
    for fp in ply_files:
        try:
            meshes.append(trimesh.load(fp))
        except Exception:
            pass
    if not meshes:
        return
    combined = trimesh.util.concatenate(meshes)
    glb_path = _os.path.join(out_dir, "scene.glb")
    combined.export(glb_path)
    print(f"[LiveInferenceRunner] GLB 已导出: {glb_path}", flush=True)


def _camera_points_to_world_np(
    pts_cam: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    将相机坐标系点转到世界坐标系（镜像 infer.py _camera_points_to_world）。

    Args:
        pts_cam: [N, 3] 相机空间点
        R:       [3, 3] w2c 旋转矩阵（R_abs）
        t:       [3,]   w2c 平移向量（t_abs）
    Returns:
        pts_world: [N, 3] float32 世界坐标

    公式: pts_world = R^T @ (pts_cam^T - t[:, None])
    """
    pts = pts_cam.reshape(-1, 3).astype(np.float64)
    R64 = R.astype(np.float64)
    t64 = t.astype(np.float64)
    world = (R64.T @ (pts.T - t64[:, None])).T
    return world.astype(np.float32)


def _decode_outputs_to_cpu(
    outputs: dict,
    packet: FramePacket,
    kf_idx_local: int,
    local_pos: int,
    is_kf: bool,
    seg_R_abs: dict,
    seg_t_abs: dict,
) -> dict:
    """
    将 forward_stream 返回的 GPU Tensor dict 全部解码为 CPU numpy，
    同时进行相机位姿组合以提取相机中心，供 RerunViewer 直接使用。

    所有 GPU Tensor 均在此函数内完成 .detach().cpu().numpy() 转换。
    绝对禁止把 GPU Tensor 放入返回 dict。

    seg_R_abs / seg_t_abs 为段内绝对位姿历史（可变 dict，本函数会更新）：
        key = 段内帧位置（local_pos），value = (R_abs [3,3], t_abs [3,])
    """
    result: dict = {}
    result["frame_index"] = packet.frame_index
    result["rgb_np"] = packet.rgb          # uint8 HWC numpy
    result["image_path"] = packet.image_path
    if packet.gps_xyz is not None:
        result["gps_xyz"] = packet.gps_xyz
    if packet.gt_pose is not None:
        result["gt_pose"] = packet.gt_pose

    # ── 深度图 ────────────────────────────────────────────────────────────
    depth_t = outputs.get("depth")
    if isinstance(depth_t, torch.Tensor):
        # [B, S, H, W, 1] → 取最后帧 → [H, W]
        result["depth_np"] = depth_t[0, -1, :, :, 0].detach().cpu().numpy()

    # ── 位姿 → 相机中心 + w2c 矩阵 ─────────────────────────────────────────
    # 必须在点云转换之前解码（需要 R_abs/t_abs 来做 cam→world）
    center_np, R_abs_dec, t_abs_dec = _decode_camera_center(
        outputs, kf_idx_local, local_pos, is_kf, seg_R_abs, seg_t_abs
    )
    if center_np is not None:
        result["center_np"] = center_np
    # 暴露全局 w2c 位姿，供 refresh 段 anchor 使用
    if R_abs_dec is not None:
        result["R_abs_np"] = R_abs_dec
    if t_abs_dec is not None:
        result["t_abs_np"] = t_abs_dec

    # ── 世界点云（必须先将相机坐标系点转到世界坐标系）─────────────
    # 先提取 depth_t（后续 dpt_unproj 也需要，必须在 GPU 上完成反投影）
    depth_t = outputs.get("depth")

    world_t = outputs.get("world_points")
    if isinstance(world_t, torch.Tensor):
        pts_cam = world_t[0, -1].detach().cpu().numpy()  # [H, W, 3] or [N, 3]
        pts_cam_flat = pts_cam.reshape(-1, 3)
        if R_abs_dec is not None and t_abs_dec is not None:
            pts_world = _camera_points_to_world_np(pts_cam_flat, R_abs_dec, t_abs_dec)
        else:
            pts_world = pts_cam_flat
        result["world_points_np"] = pts_world

        # ── 点云颜色：使用 packet.rgb_model（resize/crop 后、归一化前的真实 RGB）──
        # rgb_model 与模型输出点云 H/W 精确对齐，不含 ImageNet 归一化偏差
        try:
            if packet.rgb_model is not None:
                rgb_model_np = packet.rgb_model.astype(np.uint8)  # [H', W', 3]
                if pts_cam.ndim == 3:
                    h_pts, w_pts = pts_cam.shape[:2]
                    if rgb_model_np.shape[:2] != (h_pts, w_pts):
                        from PIL import Image as _PILImage
                        rgb_model_np = np.array(
                            _PILImage.fromarray(rgb_model_np).resize((w_pts, h_pts), _PILImage.BILINEAR),
                            dtype=np.uint8,
                        )
                result["point_colors_np"] = rgb_model_np.reshape(-1, 3)
            else:
                # 旧版兼容 fallback：从 image_tensor 反推（颜色不精确）
                img_t = packet.image_tensor[0, 0]  # [C, H, W], [0,1]
                rgb_fb = np.clip(
                    img_t.permute(1, 2, 0).detach().cpu().numpy() * 255.0, 0, 255
                ).astype(np.uint8)
                if pts_cam.ndim == 3:
                    h_pts, w_pts = pts_cam.shape[:2]
                    if rgb_fb.shape[:2] != (h_pts, w_pts):
                        from PIL import Image as _PILImage
                        rgb_fb = np.array(
                            _PILImage.fromarray(rgb_fb).resize((w_pts, h_pts), _PILImage.BILINEAR),
                            dtype=np.uint8,
                        )
                result["point_colors_np"] = rgb_fb.reshape(-1, 3)
        except Exception:
            pass

    # ── depth_conf（供 dpt_unproj 过滤使用，与 core/infer.py 对齐）────────
    depth_conf_t = outputs.get("depth_conf")
    if isinstance(depth_conf_t, torch.Tensor):
        _dc = depth_conf_t[0, -1].detach().cpu().numpy()
        if _dc.ndim == 3:
            _dc = _dc[..., 0]
        result["depth_conf_np"] = _dc.reshape(-1)

    # ── dpt_unproj：深度反投影点云（与 core/infer.py dpt_unproj 分支对齐）──
    # 在 GPU 上完成反投影后立即转 CPU numpy，保证不把 GPU tensor 放入 result
    if isinstance(depth_t, torch.Tensor) and R_abs_dec is not None and t_abs_dec is not None:
        try:
            from longstream.utils.vendor.models.components.utils.pose_enc import (
                pose_encoding_to_extri_intri as _peti,
            )
            from longstream.utils.camera import compose_abs_from_rel as _cafr

            # 显式 isinstance 检查，避免多元素 Tensor 触发 bool() 歧义异常
            _rel_pe = outputs.get("rel_pose_enc")
            _abs_pe = outputs.get("pose_enc")

            intri_t = None
            H_d = int(depth_t.shape[2])
            W_d = int(depth_t.shape[3])

            if isinstance(_rel_pe, torch.Tensor):
                # compose_abs_from_rel 将 rel 转 abs。
                # 对于 dpt_unproj 只需要 intri（focal length），而 focal 在
                # compose_abs_from_rel 里是直接保留 rel_f，不依赖参考帧位姿。
                # keyframe_indices 全用 0：避免 S_win==1 时 kf_idx_local>0 越界。
                S_win = _rel_pe.shape[1]
                _kf_idx_arr = torch.zeros(S_win, dtype=torch.long, device=_rel_pe.device)
                _abs_pe_composed = _cafr(_rel_pe[0], _kf_idx_arr)  # [S, 9]
                _, intri_t = _peti(_abs_pe_composed[None], image_size_hw=(H_d, W_d))  # [1,S,3,3]
            elif isinstance(_abs_pe, torch.Tensor):
                _, intri_t = _peti(_abs_pe, image_size_hw=(H_d, W_d))

            if intri_t is not None:
                intri_np_frame = intri_t[0, -1].detach().cpu().numpy()  # [3, 3]
                result["intri_np"] = intri_np_frame
                # 反投影：depth [H, W] → 相机空间点 [H, W, 3]
                from longstream.utils.depth import unproject_depth_to_points as _udtp
                d_last = depth_t[0, -1, :, :, 0]  # [H, W]
                intri_gpu = torch.from_numpy(intri_np_frame).unsqueeze(0).to(d_last.device)
                pts_cam_dpu = _udtp(d_last.unsqueeze(0), intri_gpu)[0]  # [H, W, 3]
                pts_cam_dpu_np = pts_cam_dpu.detach().cpu().numpy().reshape(-1, 3)
                # cam → world（同 world_points 的变换公式）
                result["dpt_unproj_points_np"] = _camera_points_to_world_np(
                    pts_cam_dpu_np, R_abs_dec, t_abs_dec
                )
            else:
                _dpu_warn_count += 1
                if _dpu_warn_count <= _dpu_warn_max:
                    print(
                        f"[LiveInferenceRunner/_decode] 警告 ({_dpu_warn_count})：dpt_unproj 跳过——"
                        "pose_enc/rel_pose_enc 均不可用，无法提取 intri。",
                        flush=True,
                    )
        except Exception as _dpu_err:
            _dpu_warn_count += 1
            if _dpu_warn_count <= _dpu_warn_max:
                print(
                    f"[LiveInferenceRunner/_decode] 警告 ({_dpu_warn_count})：dpt_unproj 计算异常: {_dpu_err}",
                    flush=True,
                )

    # ── 置信度 ───────────────────────────────────────────────────────────
    conf_t = outputs.get("world_points_conf")
    if isinstance(conf_t, torch.Tensor):
        conf = conf_t[0, -1].detach().cpu().numpy()
        result["conf_np"] = conf.reshape(-1)

    # ── GPU Tensor 泄漏防线：确保 result 中没有 torch.Tensor ──────────────
    for _k, _v in result.items():
        if isinstance(_v, torch.Tensor):
            raise RuntimeError(
                f"[_decode_outputs_to_cpu] GPU/torch tensor leaked into outputs_cpu: key={_k!r}. "
                "请检查上方解码逻辑，确保所有字段均已 .detach().cpu().numpy()。"
            )

    return result


def _decode_camera_center(
    outputs: dict,
    kf_idx_local: int,
    local_pos: int,
    is_kf: bool,
    seg_R_abs: dict,
    seg_t_abs: dict,
) -> tuple:
    """
    从 rel_pose_enc 或 pose_enc 解码当前帧的相机中心（段内坐标系）。

    返回: (center_np, R_abs, t_abs)
      - center_np: [3,] float32 相机中心（如果解码失败则为 None）
      - R_abs:     [3, 3] float32 w2c 旋转矩阵（用于点云 cam→world 变换）
      - t_abs:     [3,] float32 w2c 平移（用于点云 cam→world 变换）

    正确做法：
      R_abs[s] = R_rel[s] @ R_abs[ref]
      t_abs[s] = t_rel[s] + R_rel[s] @ t_abs[ref]
      center    = -R_abs[s]^T @ t_abs[s]

    若为关键帧，还会将 (R_abs, t_abs) 存入 seg_R_abs/seg_t_abs[local_pos]
    供后续非关键帧组合使用。

    seg_R_abs / seg_t_abs 传入即更新（mutable dict）。
    """
    from longstream.utils.vendor.models.components.utils.rotation import quat_to_mat

    rel_pe_t = outputs.get("rel_pose_enc")
    abs_pe_t = outputs.get("pose_enc")

    try:
        if rel_pe_t is not None and isinstance(rel_pe_t, torch.Tensor):
            # rel_pose_enc: [1, 1, D]（batch=1, seq_len=1）
            pe_cpu = rel_pe_t[0, 0].detach().cpu()  # [D,]
            q_tensor = pe_cpu[3:7].unsqueeze(0)      # [1, 4]
            R_rel = quat_to_mat(q_tensor)[0].numpy().astype(np.float32)  # [3, 3]
            t_rel = pe_cpu[:3].numpy().astype(np.float32)                # [3,]

            # 组合绝对位姿
            R_ref = seg_R_abs.get(kf_idx_local)
            t_ref = seg_t_abs.get(kf_idx_local)
            if R_ref is not None and t_ref is not None:
                R_abs = R_rel @ R_ref
                t_abs = t_rel + R_rel @ t_ref
            else:
                # 参考不在历史中（段首帧或 refresh 刚发生）：直接使用相对位姿
                R_abs = R_rel
                t_abs = t_rel

            # 若为关键帧，存储绝对位姿供后续帧参考
            if is_kf:
                seg_R_abs[local_pos] = R_abs.copy()
                seg_t_abs[local_pos] = t_abs.copy()

            center = -(R_abs.T @ t_abs)
            return center.astype(np.float32), R_abs, t_abs

        elif abs_pe_t is not None and isinstance(abs_pe_t, torch.Tensor):
            # 绝对位姿编码：直接解码
            pe_cpu = abs_pe_t[0, 0].detach().cpu()  # [D,]
            q_tensor = pe_cpu[3:7].unsqueeze(0)
            R_abs = quat_to_mat(q_tensor)[0].numpy().astype(np.float32)
            t_abs = pe_cpu[:3].numpy().astype(np.float32)
            if is_kf:
                seg_R_abs[local_pos] = R_abs.copy()
                seg_t_abs[local_pos] = t_abs.copy()
            center = -(R_abs.T @ t_abs)
            return center.astype(np.float32), R_abs, t_abs

    except Exception as e:
        pass
    return None, None, None


def _run_tto_window(
    tto_ctx: TTOContext,
    frame_buffer: list,
    gps_buffer: list,
    selector: KeyframeSelector,
    streaming_mode: str,
    device: str,
    frame_idx: int,
    pair_stride: int = 1,
) -> None:
    """
    在滑动窗口上运行一次 TTO（不影响 session 的 KV cache）。
    注意：reset_tto 已在流开始时执行，此处不再重置，
    以保留跨窗口累积优化的 scale_token 状态。
    """
    tensors = [p.image_tensor for p in frame_buffer]  # 每个 [1, 1, C, H, W]
    try:
        images_win = torch.cat(tensors, dim=1)  # [1, T, C, H, W]
    except Exception as exc:
        print(f"[LiveInferenceRunner][TTO] 无法拼接帧 tensor: {exc}", flush=True)
        return

    gps_np = np.stack(gps_buffer, axis=0).astype(np.float32)  # [T, 3]
    # 不调用 reset_tto：scale_token 保留跨窗口的优化状态
    run_tto_scale_optimization(
        ctx=tto_ctx,
        images=images_win,
        gps_xyz=gps_np,
        selector=selector,
        streaming_mode=streaming_mode,
        device=device,
        seq_name=f"live_frame{frame_idx}",
        pair_stride=pair_stride,
    )


def _build_feeder_from_cfg(data_cfg: dict) -> StreamFeeder:
    """从 data 配置块构建 StreamFeeder。"""
    fmt = data_cfg.get("format", "image_dir")
    crop = bool(data_cfg.get("crop", False))
    size = int(data_cfg.get("size", 518))
    patch_size = int(data_cfg.get("patch_size", 14))
    max_frames = data_cfg.get("max_frames", None)
    if max_frames is not None:
        max_frames = int(max_frames)
    camera = data_cfg.get("camera", None)
    fps = float(data_cfg.get("fps", 0.0))

    if fmt == "generalizable":
        source_type = "generalizable"
        source = data_cfg.get("img_path", ".")
        # 传递完整 data_cfg 以复用 LongStreamDataLoader 的路径解析逻辑
        return StreamFeeder(
            source=source,
            source_type=source_type,
            fps=fps,
            size=size,
            patch_size=patch_size,
            max_frames=max_frames,
            camera=camera,
            crop=crop,
            data_cfg=data_cfg,
        )
    elif fmt == "video":
        source_type = "video"
        source = data_cfg.get("img_path", data_cfg.get("video_path", "."))
    elif fmt == "npz":
        source_type = "npz"
        source = data_cfg.get("img_path", data_cfg.get("npz_path", "."))
    else:
        source_type = "image_dir"
        source = data_cfg.get("img_path", ".")

    return StreamFeeder(
        source=source,
        source_type=source_type,
        fps=fps,
        size=size,
        patch_size=patch_size,
        max_frames=max_frames,
        camera=camera,
        crop=crop,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  批处理子进程入口
# ═══════════════════════════════════════════════════════════════════════════

def _batch_worker_main(
    cfg: dict,
    result_queue: "mp.Queue",
    stop_flag: "mp.Event",
) -> None:
    """
    批处理模式子进程：调用 run_inference_cfg(cfg) 走完整非流式推理链路。
    输出由 run_inference_cfg 自行按 cfg["output"] 落盘，与 python run.py 完全一致。
    result_queue 仅接收错误/完成信号，无逐帧 on_frame 回调。
    """
    import traceback as _tb

    # ── 资源监控：在推理进程内启动，记录真实推理进程的 CPU/GPU ──────────
    _batch_monitor = None
    try:
        from longstream.utils.resource_monitor import from_cfg as _mon_from_cfg
        _batch_monitor = _mon_from_cfg(cfg.get("monitoring", {}))
        if _batch_monitor is not None:
            _batch_out_dir = os.path.abspath(
                cfg.get("output", {}).get("root", "outputs")
            )
            _batch_monitor.start(_batch_out_dir)
    except Exception:
        _batch_monitor = None

    try:
        from longstream.core.infer import run_inference_cfg
        run_inference_cfg(cfg)
        result_queue.put_nowait(("__done__", "批处理推理完成"))
    except Exception as exc:
        try:
            result_queue.put_nowait(("__error__", str(exc) + "\n" + _tb.format_exc()))
        except Exception:
            pass
        _tb.print_exc()
    finally:
        if _batch_monitor is not None:
            try:
                _batch_monitor.stop()
            except Exception:
                pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


class BatchInferenceRunner:
    """
    批处理推理运行器。

    在独立子进程中调用 run_inference_cfg(cfg)，
    不进行帧率限速，不推送 Rerun Viewer。
    输出由 run_inference_cfg 自行按 cfg["output"] 落盘。
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._process: Optional[mp.Process] = None
        self._result_queue: mp.Queue = mp.Queue(maxsize=5)
        self._stop_flag: mp.Event = mp.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._stop_flag.clear()
        self._error = None
        self._process = mp.Process(
            target=_batch_worker_main,
            args=(self.cfg, self._result_queue, self._stop_flag),
            name="BatchInferenceWorker",
            daemon=True,
        )
        self._process.start()
        self._poll_thread = threading.Thread(
            target=self._poll_results,
            name="BatchInferenceRunner-poll",
            daemon=True,
        )
        self._poll_thread.start()

    def stop(self) -> None:
        if self._process is None:
            return
        self._stop_flag.set()
        self._process.join(timeout=3)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=2)
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2)

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def last_error(self) -> Optional[Exception]:
        return self._error

    def _poll_results(self) -> None:
        while self._process is not None and (
            self._process.is_alive() or not self._result_queue.empty()
        ):
            try:
                item = self._result_queue.get(timeout=0.5)
            except Exception:
                continue
            if isinstance(item, tuple) and len(item) == 2:
                tag, msg = item
                if tag == "__error__":
                    self._error = RuntimeError(msg)
                    print(f"[BatchInferenceRunner] 批处理错误:\n{msg}", flush=True)
                elif tag == "__done__":
                    print(f"[BatchInferenceRunner] {msg}", flush=True)
