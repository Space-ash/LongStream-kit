import warnings

import numpy as np

from longstream.demo.backend import load_frame_previews

from .common import load_metadata
from .geometry import camera_geometry, collect_points


# ---------------------------------------------------------------------------
# NOTE: build_interactive_figure() 已废弃。
#   实时可视化请改用 longstream/demo/rerun_viewer.py 中的 RerunViewer。
#   此函数保留仅为向后兼容旧批量推理演示路径。
# ---------------------------------------------------------------------------


def build_interactive_figure(
    session_dir: str,
    branch: str,
    display_mode: str,
    frame_index: int,
    point_size: float,
    opacity: float,
    preview_max_points: int,
    show_cameras: bool,
    camera_scale: float,
    mask_sky: bool,
) -> str:
    """
    [已废弃] 原返回 Plotly Figure；现改为返回轻量状态消息。
    实时 3D 可视化请使用 RerunViewer（longstream/demo/rerun_viewer.py）。
    """
    warnings.warn(
        "build_interactive_figure() 已废弃，不再创建 Plotly 图形。"
        " 请使用 longstream.demo.rerun_viewer.RerunViewer 代替。",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        meta = load_metadata(session_dir)
        num_frames = meta.get("num_frames", 0)
        points, _, _ = collect_points(
            session_dir=session_dir,
            branch=branch,
            display_mode=display_mode,
            frame_index=frame_index,
            mask_sky=mask_sky,
            max_points=preview_max_points,
            seed=frame_index,
        )
        n_pts = len(points) if points is not None else 0
        return (
            f"[兼容模式] session={session_dir} frames={num_frames} points={n_pts}"
            f" branch={branch} — 3D 渲染已迁移至 Rerun Viewer"
        )
    except Exception as exc:
        return f"[兼容模式] 无法加载 session: {exc}"


def build_frame_outputs(session_dir: str, frame_index: int):
    rgb, depth, label = load_frame_previews(session_dir, frame_index)
    return rgb, depth, label
