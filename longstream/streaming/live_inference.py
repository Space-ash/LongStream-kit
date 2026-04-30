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
from typing import Any, Callable, Dict, Optional

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
    _out_dir = cfg.get("output", {}).get(
        "root", os.path.join("output", "live_inference")
    )
    # 如果启用 GLB 导出但未启用逐帧点云，自动临时开启逐帧点云（GLB 依赖它）
    if _out_cfg_merged.get("export_glb", False) and not _out_cfg_merged.get("save_frame_points", False):
        _out_cfg_merged = dict(_out_cfg_merged)
        _out_cfg_merged["save_frame_points"] = True
        print("[LiveInferenceRunner/worker] 已自动启用 save_frame_points（GLB 导出需要）", flush=True)
    # mask_sky 在 streaming 模式下暂不支持，检测到后警告一次
    if _out_cfg_merged.get("mask_sky", False):
        print("[LiveInferenceRunner/worker] 警告：mask_sky 在流式推理模式下暂未集成，跳过。", flush=True)

    # save_points: 使用有界 Reservoir 采样累计全局点云，防止 CPU RAM 无限增长
    # 策略：将每帧点加入 reservoir；当 reservoir 超过 2×max_full_pts 时随机下采样至 max_full_pts
    _reservoir_pts:    Optional[np.ndarray] = None   # [R, 3] float32
    _reservoir_colors: Optional[np.ndarray] = None   # [R, 3] uint8 或 None
    _reservoir_has_colors: bool = False
    _reservoir_rng = np.random.default_rng(seed=0)
    _max_full_pts: int = int(_out_cfg_merged.get("max_full_pointcloud_points",
                              _out_cfg_merged.get("max_frame_pointcloud_points", 8000) * 100))

    print("[LiveInferenceRunner/worker] 开始逐帧推理", flush=True)

    with torch.no_grad():
        for packet in feeder:
            if stop_flag.is_set():
                break

            g: int = packet.frame_index
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

            # ── refresh 逻辑 ──────────────────────────────────────────────
            if is_kf and g > 0:
                keyframe_count += 1
                if keyframe_count % refresh_intervals == 0:
                    session.clear_cache_only()
                    segment_start = g
                    seg_R_abs = {0: np.eye(3, dtype=np.float32)}
                    seg_t_abs = {0: np.zeros(3, dtype=np.float32)}
                    anchor_kf_indices = torch.zeros(
                        1, 1, dtype=torch.long, device=device_str
                    )
                    anchor_frame = packet.image_tensor.to(device_str, non_blocking=True)
                    session.forward_stream(
                        anchor_frame,
                        is_keyframe=is_keyframe_t,
                        keyframe_indices=anchor_kf_indices,
                        record=False,
                    )
                    del anchor_frame

            del outputs
            del frame_tensor

            # ── 发送结果到主进程（队列满则丢弃，保证推理不被可视化阻塞）──
            try:
                result_queue.put_nowait((g, outputs_cpu))
            except Exception:
                pass

            # ── 增量保存到磁盘（仅使用 outputs_cpu，严禁 GPU Tensor）──────
            _save_frame_outputs(outputs_cpu, _out_cfg_merged, _out_dir, g)

            # ── save_points：Reservoir 采样累积 CPU 世界坐标点 ─────────────
            if _out_cfg_merged.get("save_points", False):
                pts = outputs_cpu.get("world_points_np")
                if pts is not None:
                    new_pts = pts.reshape(-1, 3).astype(np.float32)
                    new_colors = outputs_cpu.get("point_colors_np")
                    if new_colors is not None:
                        new_colors = new_colors.reshape(-1, 3).astype(np.uint8)
                        _reservoir_has_colors = True
                    # 并入 reservoir
                    if _reservoir_pts is None:
                        _reservoir_pts = new_pts
                        _reservoir_colors = new_colors
                    else:
                        _reservoir_pts = np.concatenate([_reservoir_pts, new_pts], axis=0)
                        if _reservoir_has_colors:
                            _c_prev = _reservoir_colors if _reservoir_colors is not None else np.zeros((len(_reservoir_pts) - len(new_pts), 3), dtype=np.uint8)
                            _c_new  = new_colors if new_colors is not None else np.zeros((len(new_pts), 3), dtype=np.uint8)
                            _reservoir_colors = np.concatenate([_c_prev, _c_new], axis=0)
                    # 超过 2x 上限时随机下采样，保持内存有界
                    if len(_reservoir_pts) > 2 * _max_full_pts:
                        idx = _reservoir_rng.choice(len(_reservoir_pts), _max_full_pts, replace=False)
                        _reservoir_pts = _reservoir_pts[idx]
                        if _reservoir_colors is not None:
                            _reservoir_colors = _reservoir_colors[idx]

    # ── 循环结束后处理（纯 CPU / 磁盘，不持有 GPU 引用）─────────────────
    # 全局点云 PLY
    if _out_cfg_merged.get("save_points", False) and _reservoir_pts is not None and len(_reservoir_pts) > 0:
        # 最终下采样（如果还超过上限）
        _final_pts = _reservoir_pts
        _final_colors = _reservoir_colors if _reservoir_has_colors else None
        if len(_final_pts) > _max_full_pts:
            idx = _reservoir_rng.choice(len(_final_pts), _max_full_pts, replace=False)
            _final_pts = _final_pts[idx]
            if _final_colors is not None:
                _final_colors = _final_colors[idx]
        pts_dir = _os.path.join(_out_dir, "global_points")
        _os.makedirs(pts_dir, exist_ok=True)
        ply_path = _os.path.join(pts_dir, "global_pointcloud.ply")
        _write_ply(_final_pts, ply_path, colors=_final_colors)
        print(f"[LiveInferenceRunner] 全局点云已保存: {ply_path}  ({len(_final_pts)} 点)", flush=True)

    # 视频合成（从已落盘 RGB 序列）
    if _out_cfg_merged.get("save_videos", False):
        _fps_out = int(cfg.get("data", {}).get("fps", 0) or 24)
        _assemble_video(_out_dir, fps=_fps_out)

    # GLB 导出（基于已落盘逐帧 PLY）
    if _out_cfg_merged.get("export_glb", False):
        _export_glb_from_plys(_out_dir)

    session.clear()
    torch.cuda.empty_cache()
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

    _QUEUE_MAXSIZE = 30  # 结果队列上限；超过则丢弃最新帧

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


def _save_frame_outputs(
    outputs_cpu: dict,
    out_cfg: dict,
    out_dir: str,
    frame_idx: int,
) -> None:
    """
    增量保存单帧输出到磁盘。

    所有输入均为纯 numpy（已完成 .detach().cpu().numpy()）。
    本函数绝对不导入 torch，也不持有任何 GPU 对象引用。
    """
    import PIL.Image  # 仅在子进程中需要，延迟导入

    _os.makedirs(out_dir, exist_ok=True)

    # ── RGB PNG ──────────────────────────────────────────────────────────
    if out_cfg.get("save_images", False):
        rgb_np = outputs_cpu.get("rgb_np")
        if rgb_np is not None:
            rgb_dir = _os.path.join(out_dir, "images")
            _os.makedirs(rgb_dir, exist_ok=True)
            PIL.Image.fromarray(rgb_np).save(
                _os.path.join(rgb_dir, f"{frame_idx:06d}.png")
            )

    # ── 深度图 .npy + 可视化 PNG ─────────────────────────────────────────
    if out_cfg.get("save_depth", False):
        depth_np = outputs_cpu.get("depth_np")
        if depth_np is not None:
            depth_dir = _os.path.join(out_dir, "depth")
            _os.makedirs(depth_dir, exist_ok=True)
            np.save(_os.path.join(depth_dir, f"{frame_idx:06d}.npy"), depth_np)
            # 归一化可视化
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max > d_min:
                d_vis = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                d_vis = np.zeros_like(depth_np, dtype=np.uint8)
            PIL.Image.fromarray(d_vis).save(
                _os.path.join(depth_dir, f"{frame_idx:06d}_vis.png")
            )

    # ── 逐帧点云 PLY（按置信度过滤）─────────────────────────────────────
    if out_cfg.get("save_frame_points", False):
        pts = outputs_cpu.get("world_points_np")
        if pts is not None:
            conf = outputs_cpu.get("conf_np")
            colors = outputs_cpu.get("point_colors_np")
            thr = float(out_cfg.get(
                "confidence_threshold",
                out_cfg.get("max_conf_filter_threshold", 0.5),
            ))
            if conf is not None:
                mask = conf.reshape(-1) >= thr
                pts_filtered = pts.reshape(-1, 3)[mask]
                colors_filtered = colors.reshape(-1, 3)[mask] if colors is not None else None
            else:
                pts_filtered = pts.reshape(-1, 3)
                colors_filtered = colors.reshape(-1, 3) if colors is not None else None
            pts_dir = _os.path.join(out_dir, "frame_points")
            _os.makedirs(pts_dir, exist_ok=True)
            _write_ply(pts_filtered, _os.path.join(pts_dir, f"{frame_idx:06d}.ply"),
                       colors=colors_filtered)

    # ── 轨迹（追加写）───────────────────────────────────────────────────
    center = outputs_cpu.get("center_np")
    if center is not None:
        traj_path = _os.path.join(out_dir, "trajectory.txt")
        with open(traj_path, "a", encoding="utf-8") as f:
            f.write(
                f"{frame_idx} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n"
            )


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
    将 out_dir/images/*.png 序列合成为 out_dir/output.mp4。
    优先使用 longstream.io.save_images.save_video（ffmpeg glob 模式）；
    回退到 imageio v2 FFMPEG writer。
    纯 CPU/磁盘操作，不使用任何 GPU 张量。
    """
    import glob

    img_dir = _os.path.join(out_dir, "images")
    if not _os.path.isdir(img_dir):
        print("[LiveInferenceRunner] save_videos 跳过：images/ 目录不存在。", flush=True)
        return

    frames = sorted(glob.glob(_os.path.join(img_dir, "*.png")))
    if not frames:
        print("[LiveInferenceRunner] save_videos 跳过：images/ 目录为空。", flush=True)
        return

    out_video = _os.path.join(out_dir, "output.mp4")
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


def _export_glb_from_plys(out_dir: str) -> None:
    """将已落盘的逐帧 PLY 合并导出为 GLB（使用 trimesh；不可用则跳过）。"""
    import glob

    pts_dir = _os.path.join(out_dir, "frame_points")
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

    # ── 世界点云（必须先将相机坐标系点转到世界坐标系）─────────────
    # LongStream-kit infer.py L721 进行同样的转换:
    #   pts_world = _camera_points_to_world(pts_cam, extri)
    # 此处必须保留 .detach().cpu().numpy()，Rerun 侧绝对不能接 GPU Tensor
    world_t = outputs.get("world_points")
    if isinstance(world_t, torch.Tensor):
        pts_cam = world_t[0, -1].detach().cpu().numpy()  # [H, W, 3] or [N, 3]
        H_pts, W_pts = pts_cam.shape[:2] if pts_cam.ndim == 3 else (pts_cam.shape[0], 1)
        pts_cam_flat = pts_cam.reshape(-1, 3)
        if R_abs_dec is not None and t_abs_dec is not None:
            pts_world = _camera_points_to_world_np(pts_cam_flat, R_abs_dec, t_abs_dec)
        else:
            # 位姿解码失败时直接使用相机坐标（降级处理）
            pts_world = pts_cam_flat
        result["world_points_np"] = pts_world

        # ── 点云颜色（与点云像素一一对应）────────────────────────────
        # 将原图 RGB resize 到与 world_points 同分辨率，保证像素对应
        rgb_full = packet.rgb
        if rgb_full is not None and pts_cam.ndim == 3:
            try:
                from PIL import Image as _PILImage
                import numpy as _np2
                _h, _w = pts_cam.shape[:2]
                rgb_resized = _np2.array(
                    _PILImage.fromarray(rgb_full).resize((_w, _h), _PILImage.BILINEAR),
                    dtype=_np2.uint8,
                )
                result["point_colors_np"] = rgb_resized.reshape(-1, 3)
            except Exception:
                pass

    # ── 置信度 ───────────────────────────────────────────────────────────
    conf_t = outputs.get("world_points_conf")
    if isinstance(conf_t, torch.Tensor):
        conf = conf_t[0, -1].detach().cpu().numpy()
        result["conf_np"] = conf.reshape(-1)

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
    批处理模式子进程：复用 _worker_inference_loop，fps=0（无限速）。
    不推送 Rerun Viewer（result_queue 仅接收错误/完成信号，无 on_frame 回调）。
    支持 video/NPZ/image_dir/generalizable，全走 StreamFeeder 路径。
    输出增量落盘逻辑与流式模式完全一致。
    """
    import traceback as _tb
    import copy

    # 强制关闭帧率限速
    batch_cfg = copy.deepcopy(cfg)
    batch_cfg.setdefault("data", {})
    batch_cfg["data"]["fps"] = 0

    try:
        _worker_inference_loop(batch_cfg, result_queue, stop_flag)
        result_queue.put_nowait(("__done__", "批处理推理完成"))
    except Exception as exc:
        try:
            result_queue.put_nowait(("__error__", str(exc) + "\n" + _tb.format_exc()))
        except Exception:
            pass
        _tb.print_exc()
    finally:
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
