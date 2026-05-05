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

# 论文截图友好的浅灰背景色


def _make_spatial3d_view(name: str, origin: str):
    """Create Spatial3DView without background kwarg to avoid numpy-ABI BackgroundKindBatch warning.

    Users can configure the theme (light / white) in the Rerun Viewer GUI:
      Settings → Appearance → Theme → Light
    """
    import rerun.blueprint as rrb
    return rrb.Spatial3DView(name=name, origin=origin)


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
                _make_spatial3d_view("Current Frame",        "live/current"),
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
    max_global_points : int
        全局累积点云的上限点数（CPU RAM 有界）。
    spawn : bool
        是否在 init() 时自动打开 Rerun Viewer 窗口（spawn=True）。
    """

    def __init__(
        self,
        app_name: str = "LongStream Live Reconstruction",
        confidence_threshold: float = 0.5,
        max_frame_points: Optional[int] = 8000,
        max_global_frame_points: Optional[int] = 2000,
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
        print(
            "[RerunViewer] 注意：如需展示白色背景（论文截图风格），"
            "请在 Rerun Viewer 界面选择 Settings → Appearance → Theme → Light。",
            flush=True,
        )

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
            rr.log("live/rgb", rr.Image(rgb_np))

        # ── 深度图（右下面板）────────────────────────────────────────────
        depth_np: Optional[np.ndarray] = outputs_cpu.get("depth_np")
        if depth_np is not None and isinstance(depth_np, np.ndarray):
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
        points_np: Optional[np.ndarray] = outputs_cpu.get("filtered_points_np")
        colors_np: Optional[np.ndarray] = outputs_cpu.get("filtered_colors_np")

        if points_np is None:
            # fallback：worker 未注入预过滤结果时（旧版兼容），在 viewer 侧过滤
            raw_pts = outputs_cpu.get("world_points_np")
            conf_np = outputs_cpu.get("conf_np")
            pre_colors = outputs_cpu.get("point_colors_np")
            if raw_pts is not None and isinstance(raw_pts, np.ndarray):
                orig_n = raw_pts.shape[0]
                mask: Optional[np.ndarray] = None
                if conf_np is not None and isinstance(conf_np, np.ndarray):
                    if conf_np.shape[0] == orig_n:
                        mask = conf_np >= self.confidence_threshold
                        raw_pts = raw_pts[mask]
                if pre_colors is not None and isinstance(pre_colors, np.ndarray):
                    if mask is not None and len(pre_colors) == orig_n:
                        colors_np = pre_colors[mask].astype(np.uint8)
                    elif mask is None and len(pre_colors) == len(raw_pts):
                        colors_np = pre_colors.astype(np.uint8)
                points_np = raw_pts

        if points_np is not None and isinstance(points_np, np.ndarray) and len(points_np) > 0:
            # 颜色缺失时跳过全部点云日志（当前帧 + 全局拼接），避免白点污染
            if colors_np is None:
                print(
                    f"[RerunViewer] 帧 {frame_idx}: 点云颜色缺失，"
                    "跳过点云显示（当前帧 + 全局拼接均跳过）。",
                    flush=True,
                )
            else:
                # ── 左上：当前帧点云（每帧覆盖同一路径），使用 point_head ──
                cur_pts = points_np
                cur_cols = colors_np
                if self.max_frame_points is not None and len(cur_pts) > self.max_frame_points:
                    rng = np.random.default_rng(seed=frame_idx)
                    keep = rng.choice(len(cur_pts), size=self.max_frame_points, replace=False)
                    cur_pts = cur_pts[keep]
                    cur_cols = cur_cols[keep]
                rr.log(
                    "live/current/points",
                    rr.Points3D(
                        cur_pts,
                        colors=cur_cols,
                        radii=np.full(len(cur_pts), 0.008, dtype=np.float32),
                    ),
                )

                # ── 右上：全局视图，优先 dpt_unproj，fallback 到 point_head ──
                _glb_raw_pts = outputs_cpu.get("filtered_dpu_pts_np")
                _glb_raw_cols = outputs_cpu.get("filtered_dpu_cols_np")
                if _glb_raw_pts is None or len(_glb_raw_pts) == 0 or _glb_raw_cols is None:
                    _glb_raw_pts = points_np
                    _glb_raw_cols = colors_np
                glb_pts = _glb_raw_pts
                glb_cols = _glb_raw_cols
                if self.max_global_frame_points is not None and len(glb_pts) > self.max_global_frame_points:
                    rng_g = np.random.default_rng(seed=frame_idx + 100000)
                    keep_g = rng_g.choice(len(glb_pts), size=self.max_global_frame_points, replace=False)
                    glb_pts = glb_pts[keep_g]
                    glb_cols = glb_cols[keep_g]
                rr.log(
                    f"live/global/points/frame_{frame_idx:06d}",
                    rr.Points3D(
                        glb_pts,
                        colors=glb_cols,
                        radii=np.full(len(glb_pts), 0.008, dtype=np.float32),
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
