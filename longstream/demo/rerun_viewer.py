"""
longstream/demo/rerun_viewer.py
--------------------------------
基于 Rerun SDK 的实时 3D 可视化推送工具。
废弃原 viewer.py 中的 Plotly 静态渲染，改为逐帧推流到 Rerun Viewer 窗口。

使用方式：
    from longstream.demo.rerun_viewer import RerunViewer

    viewer = RerunViewer(confidence_threshold=0.5, max_frame_points=8000)
    viewer.init()   # 启动 Rerun spawn（仅需调用一次）

    # 每帧推理完毕后调用：
    viewer.log_frame(frame_idx=i, outputs_cpu=outputs_cpu)

显存红线规范（此文件内所有地方均遵守）：
  points_np   = outputs["world_points"][0, -1].detach().cpu().numpy()
  depth_np    = outputs["depth"][0, -1, :, :, 0].detach().cpu().numpy()
  pose_np     = tensor.detach().cpu().numpy()
  conf_np     = outputs["world_points_conf"][0, -1].detach().cpu().numpy()

绝对禁止：
  - 把 GPU Tensor 放进 queue / log / 传给 rr.log
  - 在 Rerun viewer 中保存 outputs 引用
  - 用 Plotly 构造 Scatter3d
"""

from __future__ import annotations

import uuid
from typing import List, Optional

import numpy as np

try:
    import rerun as rr
    _RERUN_AVAILABLE = True
except ImportError:
    _RERUN_AVAILABLE = False
    rr = None  # type: ignore[assignment]

# 默认坐标系字符串 → rr.ViewCoordinates 属性名（如 RIGHT_HAND_Z_UP）
_VIEW_COORD_DEFAULT = "RIGHT_HAND_Z_UP"


def _make_spatial3d_view(name: str, origin: str, contents=None):
    """Create a Spatial3DView with default background, optional explicit contents."""
    import rerun.blueprint as rrb
    kwargs = {"name": name, "origin": origin}
    if contents is not None:
        kwargs["contents"] = contents
    return rrb.Spatial3DView(**kwargs)


def _log_rerun_coordinate_system(view_coordinates: str) -> None:
    """Log coordinate system to 'live' entity (static, best-effort)."""
    try:
        coord = getattr(rr.ViewCoordinates, view_coordinates, None)
        if coord is None:
            print(f"[RerunViewer] 警告：未知坐标系 '{view_coordinates}'，跳过设置。", flush=True)
            return
        rr.log("live", coord, static=True)
    except Exception as _ce:
        print(f"[RerunViewer] 坐标系设置失败（{_ce}），跳过。", flush=True)


def _make_blueprint():
    """
    构建 4 面板 Rerun blueprint：
      左上: 当前帧点云  右上: 全局累积点云
      左下: RGB 图像    右下: 深度图

    若 rerun.blueprint 不可用（旧版 SDK）则返回 None（使用默认布局）。
    """
    try:
        import rerun.blueprint as rrb
        blueprint = rrb.Blueprint(
            rrb.Grid(
                _make_spatial3d_view(
                    "Current Frame",
                    "live/current",
                    contents=["/live/current/points", "/live/current/debug_marker"],
                ),
                _make_spatial3d_view("Global Reconstruction", "live/global"),
                rrb.Spatial2DView(name="RGB",   origin="live/rgb"),
                rrb.Spatial2DView(name="Depth", origin="live/depth"),
                grid_columns=2,
            ),
            collapse_panels=True,
        )
        return blueprint
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  RerunViewer
# ═══════════════════════════════════════════════════════════════════════════

class RerunViewer:
    """
    Rerun 推流器。每帧调用 log_frame() 即可实时更新可视化窗口。

    Parameters
    ----------
    app_name : str
        Rerun 应用名称（Viewer 标题栏）。
    confidence_threshold : float
        世界点云置信度过滤阈值（低于此值的点不推送）。
    max_frame_points : int | None
        每帧最多推送点数，避免 CPU/RAM 过载。None 表示不限。
    max_global_frame_points : int | None
        每帧全局点云最多推送点数（下采样到 viewer）。None 表示不限。
    rerun_global_update_interval : int
        每多少帧更新一次全局点云，默认 1。
    image_update_interval : int
        RGB 图像更新频率（帧），默认 1（每帧更新，播放速度由 data.fps 控制）。
    depth_update_interval : int
        深度图更新频率（帧），默认 1（每帧更新，播放速度由 data.fps 控制）。
    view_coordinates : str
        Rerun 坐标系，对应 rr.ViewCoordinates 的属性名，默认 RIGHT_HAND_Z_UP。
    spawn : bool
        是否在 init() 时自动打开 Rerun Viewer 窗口（spawn=True）。
    """

    def __init__(
        self,
        app_name: str = "LongStream Live Reconstruction",
        confidence_threshold: float = 0.5,
        max_frame_points: Optional[int] = 8000,
        max_global_frame_points: Optional[int] = None,
        rerun_global_update_interval: int = 1,
        image_update_interval: int = 1,
        depth_update_interval: int = 1,
        view_coordinates: str = _VIEW_COORD_DEFAULT,
        spawn: bool = True,
    ) -> None:
        if not _RERUN_AVAILABLE:
            raise ImportError(
                "rerun-sdk 未安装。请运行: pip install rerun-sdk"
            )
        self.app_name = app_name
        self.confidence_threshold = float(confidence_threshold)
        self.max_frame_points = max_frame_points
        self.max_global_frame_points = max_global_frame_points
        self.rerun_global_update_interval = max(1, int(rerun_global_update_interval))
        self.image_update_interval = max(1, int(image_update_interval))
        self.depth_update_interval = max(1, int(depth_update_interval))
        self.view_coordinates = str(view_coordinates) or _VIEW_COORD_DEFAULT
        self.spawn = spawn
        self._initialized = False
        # 轨迹缓冲（相机中心序列）
        self._centers: List[np.ndarray] = []

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def init(self) -> None:
        """初始化 Rerun recording，可选 spawn Viewer 窗口。仅调用一次。"""
        blueprint = _make_blueprint()
        recording_id = str(uuid.uuid4())
        # 优先使用 send_blueprint(make_active=True)，确保新 recording 不继承旧视图
        if blueprint is not None and hasattr(rr, "send_blueprint"):
            rr.init(self.app_name, recording_id=recording_id, spawn=self.spawn)
            try:
                rr.send_blueprint(blueprint, make_active=True)
            except Exception:
                pass
        elif blueprint is not None:
            # 旧版 SDK fallback：通过 default_blueprint 传入
            try:
                rr.init(self.app_name, recording_id=recording_id,
                        spawn=self.spawn, default_blueprint=blueprint)
            except TypeError:
                rr.init(self.app_name, recording_id=recording_id, spawn=self.spawn)
        else:
            rr.init(self.app_name, recording_id=recording_id, spawn=self.spawn)
        self._initialized = True
        self._centers = []
        print(f"[RerunViewer] 已初始化: app_name={self.app_name!r}, recording_id={recording_id}", flush=True)
        _log_rerun_coordinate_system(self.view_coordinates)

    def log_frame(self, frame_idx: int, outputs_cpu: dict) -> None:
        """
        推送单帧数据到 Rerun Viewer（4 面板布局）。

        Parameters
        ----------
        frame_idx : int
            帧序号。
        outputs_cpu : dict
            来自 _decode_outputs_to_cpu() 的输出字典。
            所有字段必须已是纯 numpy（无 torch.Tensor）：
              rgb_np          : uint8 HWC numpy
              depth_np        : float [H, W] numpy
              world_points_np : float [N, 3] numpy
              conf_np         : float [N,] numpy
              point_colors_np : uint8 [N, 3] numpy（已与点云像素对齐）
              center_np       : float [3,] numpy（由 _decode_outputs_to_cpu 预先计算）
        """
        if not self._initialized:
            self.init()

        # Rerun API 三段兼容（不同版本 API 不同）
        try:
            if hasattr(rr, "set_time_sequence"):
                rr.set_time_sequence("frame", frame_idx)
            elif hasattr(rr, "set_time"):
                rr.set_time("frame", sequence=frame_idx)
            elif hasattr(rr, "set_time_seconds"):
                rr.set_time_seconds("time", float(frame_idx))
        except Exception:
            pass

        # ── RGB（左下面板）───────────────────────────────────────────────
        rgb_np: Optional[np.ndarray] = outputs_cpu.get("rgb_np")
        if rgb_np is not None and isinstance(rgb_np, np.ndarray):
            if frame_idx % self.image_update_interval == 0:
                rr.log("live/rgb", rr.Image(rgb_np))

        # ── 深度图（右下面板）────────────────────────────────────
        depth_np: Optional[np.ndarray] = outputs_cpu.get("depth_np")
        if depth_np is not None and isinstance(depth_np, np.ndarray):
            if frame_idx % self.depth_update_interval == 0:
                rr.log("live/depth", rr.DepthImage(depth_np))

        # ── 相机位姿轨迹（全局面板）──────────────────────────────────────
        center_np: Optional[np.ndarray] = outputs_cpu.get("center_np")
        if center_np is not None and isinstance(center_np, np.ndarray):
            self._centers.append(center_np)
            rr.log(
                "live/global/camera_center",
                rr.Points3D(
                    center_np[np.newaxis],
                    colors=np.array([[0, 180, 80]], dtype=np.uint8),
                ),
            )
            if len(self._centers) >= 2:
                rr.log(
                    "live/global/trajectory",
                    rr.LineStrips3D(
                        [np.stack(self._centers, axis=0)],
                        colors=np.array([[0, 180, 80]], dtype=np.uint8),
                    ),
                )

        # ── 当前帧 / 全局世界点云 ─────────────────────────────────────────
        # 优先使用 worker 侧已统一过滤的 filtered_points_np / filtered_colors_np
        # 与磁盘落盘保持完全一致（同一 _filter_points_colors 调用结果）
        # ── 当前帧点云（来自 worker slim payload，已预采样）──────────────────
        cur_pts = outputs_cpu.get("viewer_current_points_np")
        cur_cols = outputs_cpu.get("viewer_current_colors_np")
        # fallback for non-Rerun / compatibility paths
        if cur_pts is None or cur_cols is None:
            cur_pts = outputs_cpu.get("filtered_points_np")
            cur_cols = outputs_cpu.get("filtered_colors_np")
        if cur_pts is None or cur_cols is None:
            cur_pts = outputs_cpu.get("filtered_dpu_pts_np")
            cur_cols = outputs_cpu.get("filtered_dpu_cols_np")

        # ── 全局候选：优先用 viewer_global_frame_points_np（per-frame 下采样），
        # 回退 viewer_global_points_np（旧 reservoir 快照），最终回退原始过滤数据 ──
        glb_pts = outputs_cpu.get("viewer_global_frame_points_np")
        glb_cols = outputs_cpu.get("viewer_global_frame_colors_np")
        if glb_pts is None or glb_cols is None:
            glb_pts = outputs_cpu.get("viewer_global_points_np")
            glb_cols = outputs_cpu.get("viewer_global_colors_np")
        if glb_pts is None or glb_cols is None:
            glb_pts = outputs_cpu.get("filtered_dpu_pts_np")
            glb_cols = outputs_cpu.get("filtered_dpu_cols_np")
        if glb_pts is None or glb_cols is None:
            glb_pts = outputs_cpu.get("filtered_points_np")
            glb_cols = outputs_cpu.get("filtered_colors_np")

        # ── 左上：当前帧点云（每帧覆盖同一路径，origin=live/current）─────────
        if frame_idx == 0:
            print(
                f"[RerunViewer] current_points={None if cur_pts is None else len(cur_pts)} "
                f"current_colors={None if cur_cols is None else len(cur_cols)}",
                flush=True,
            )
            # debug marker: 帮助区分“blueprint 不匹配”和“点云坐标看不到”
            rr.log(
                "live/current/debug_marker",
                rr.Points3D(
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    colors=np.array([[255, 0, 0]], dtype=np.uint8),
                    radii=np.array([0.2], dtype=np.float32),
                ),
            )
            print("[RerunViewer/current] debug_marker logged at /live/current/debug_marker", flush=True)
        if cur_pts is not None and cur_cols is not None and len(cur_pts) > 0:
            if self.max_frame_points is not None and len(cur_pts) > self.max_frame_points:
                rng = np.random.default_rng(seed=frame_idx)
                keep = rng.choice(len(cur_pts), size=self.max_frame_points, replace=False)
                cur_pts = cur_pts[keep]
                cur_cols = cur_cols[keep]
            if frame_idx < 3 or frame_idx % 50 == 0:
                finite = np.isfinite(cur_pts).all(axis=1)
                print(
                    "[RerunViewer/current] "
                    f"frame={frame_idx} path=/live/current/points "
                    f"pts_shape={cur_pts.shape} pts_dtype={cur_pts.dtype} "
                    f"finite={int(finite.sum())}/{len(cur_pts)} "
                    f"xyz_min={np.nanmin(cur_pts, axis=0)} "
                    f"xyz_max={np.nanmax(cur_pts, axis=0)} "
                    f"cols_shape={cur_cols.shape} cols_dtype={cur_cols.dtype}",
                    flush=True,
                )
            try:
                rr.log(
                    "live/current/points",
                    rr.Points3D(
                        cur_pts,
                        colors=cur_cols,
                        radii=np.full(len(cur_pts), 0.008, dtype=np.float32),
                    ),
                )
                if frame_idx < 3 or frame_idx % 50 == 0:
                    print(f"[RerunViewer/current] rr.log ok frame={frame_idx}", flush=True)
            except Exception as exc:
                print(f"[RerunViewer/current] rr.log failed frame={frame_idx}: {exc}", flush=True)

        # ── 右上：全局视图，每帧独立 entity 累积（恢复 889f81 策略，数据源换成 dpt_unproj）──
        if glb_pts is not None and glb_cols is not None and len(glb_pts) > 0:
            if frame_idx % self.rerun_global_update_interval == 0:
                send_pts = glb_pts
                send_cols = glb_cols
                if self.max_global_frame_points is not None and len(send_pts) > self.max_global_frame_points:
                    rng_g = np.random.default_rng(seed=frame_idx + 100000)
                    keep_g = rng_g.choice(len(send_pts), size=self.max_global_frame_points, replace=False)
                    send_pts = send_pts[keep_g]
                    send_cols = send_cols[keep_g] if send_cols is not None else None
                rr.log(
                    f"live/global/points/frame_{frame_idx:06d}",
                    rr.Points3D(
                        send_pts,
                        colors=send_cols,
                        radii=np.full(len(send_pts), 0.008, dtype=np.float32),
                    ),
                )

    def reset(self) -> None:
        """清除所有缓冲（新序列开始时调用）。"""
        self._centers = []


# ═══════════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════════

def _extract_point_colors(
    rgb_np: Optional[np.ndarray],
    n_points: int,
    mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    将 HWC uint8 RGB 展开为 [N, 3] 点云颜色。
    若分辨率与点云数量不匹配则返回 None。
    """
    if rgb_np is None:
        return None
    try:
        flat_rgb = rgb_np.reshape(-1, 3)  # [H*W, 3]
        if mask is not None and len(flat_rgb) == len(mask):
            flat_rgb = flat_rgb[mask]
        if len(flat_rgb) == n_points:
            return flat_rgb.astype(np.uint8)
    except Exception:
        pass
    return None
