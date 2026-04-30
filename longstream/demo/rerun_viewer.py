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

from typing import List, Optional

import numpy as np

try:
    import rerun as rr
    _RERUN_AVAILABLE = True
except ImportError:
    _RERUN_AVAILABLE = False
    rr = None  # type: ignore[assignment]


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
    spawn : bool
        是否在 init() 时自动打开 Rerun Viewer 窗口（spawn=True）。
    """

    def __init__(
        self,
        app_name: str = "LongStream Live Reconstruction",
        confidence_threshold: float = 0.5,
        max_frame_points: Optional[int] = 8000,
        spawn: bool = True,
    ) -> None:
        if not _RERUN_AVAILABLE:
            raise ImportError(
                "rerun-sdk 未安装。请运行: pip install rerun-sdk"
            )
        self.app_name = app_name
        self.confidence_threshold = float(confidence_threshold)
        self.max_frame_points = max_frame_points
        self.spawn = spawn
        self._initialized = False
        # 轨迹缓冲（相机中心序列）
        self._centers: List[np.ndarray] = []

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def init(self) -> None:
        """初始化 Rerun recording，可选 spawn Viewer 窗口。仅调用一次。"""
        rr.init(self.app_name, spawn=self.spawn)
        self._initialized = True
        self._centers = []
        print(f"[RerunViewer] 已初始化: app_name={self.app_name!r}", flush=True)

    def log_frame(self, frame_idx: int, outputs_cpu: dict) -> None:
        """
        推送单帧数据到 Rerun Viewer。

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
              center_np       : float [3,] numpy（由 _decode_outputs_to_cpu 预先计算）
        """
        if not self._initialized:
            self.init()

        rr.set_time_sequence("frame", frame_idx)

        # ── RGB ───────────────────────────────────────────────────────────
        rgb_np: Optional[np.ndarray] = outputs_cpu.get("rgb_np")
        if rgb_np is not None and isinstance(rgb_np, np.ndarray):
            rr.log("camera/rgb", rr.Image(rgb_np))

        # ── 深度图（已解码为 numpy）──────────────────────────────────────
        depth_np: Optional[np.ndarray] = outputs_cpu.get("depth_np")
        if depth_np is not None and isinstance(depth_np, np.ndarray):
            rr.log("camera/depth", rr.DepthImage(depth_np))

        # ── 相机位姿（已由 _decode_outputs_to_cpu 通过正确位姿组合计算）──
        center_np: Optional[np.ndarray] = outputs_cpu.get("center_np")
        if center_np is not None and isinstance(center_np, np.ndarray):
            self._centers.append(center_np)
            rr.log(
                "world/camera_center",
                rr.Points3D(center_np[np.newaxis], colors=np.array([[0, 200, 100]])),
            )
            if len(self._centers) >= 2:
                rr.log(
                    "world/trajectory",
                    rr.LineStrips3D([np.stack(self._centers, axis=0)]),
                )

        # ── 世界点云（已解码为 numpy）────────────────────────────────────
        points_np: Optional[np.ndarray] = outputs_cpu.get("world_points_np")
        conf_np: Optional[np.ndarray] = outputs_cpu.get("conf_np")

        if points_np is not None and isinstance(points_np, np.ndarray):
            # 置信度过滤
            mask: Optional[np.ndarray] = None
            if conf_np is not None and isinstance(conf_np, np.ndarray):
                mask = conf_np >= self.confidence_threshold
                if mask.shape[0] == points_np.shape[0]:
                    points_np = points_np[mask]

            # 点云颜色
            colors_np: Optional[np.ndarray] = _extract_point_colors(
                rgb_np, points_np.shape[0],
                mask if conf_np is not None else None,
            )

            # 数量限制
            if self.max_frame_points is not None and len(points_np) > self.max_frame_points:
                rng = np.random.default_rng(seed=frame_idx)
                keep = rng.choice(len(points_np), size=self.max_frame_points, replace=False)
                points_np = points_np[keep]
                if colors_np is not None:
                    colors_np = colors_np[keep]

            if len(points_np) > 0:
                rr.log(
                    "world/points",
                    rr.Points3D(
                        points_np,
                        colors=colors_np,
                        radii=np.full(len(points_np), 0.01, dtype=np.float32),
                    ),
                )

    def reset(self) -> None:
        """清除轨迹缓冲（新序列开始时调用）。"""
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
