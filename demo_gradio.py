"""
demo_gradio.py -- LongStream Inference Console (Gradio UI).

Design notes (v3):
  - cfg_state (gr.State): holds base config from the selected template.
    Updated only when user explicitly switches template.
    User slider/checkbox edits do NOT touch cfg_state.
  - advanced_yaml: pure override input (blank by default).
    update_ui_from_yaml() never writes full YAML into it.
  - Template sync uses .input() / .release() on user-editable components,
    so programmatic gr.update() from update_ui_from_yaml() does NOT
    re-trigger the "switch to Custom" callback.
  - simulate_streaming / enable_rerun are included in template sync outputs.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from typing import Optional

import gradio as gr
import yaml

# ─── Config template paths ────────────────────────────────────────────────
_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
_TEMPLATE_PATHS = {
    "optimized": os.path.join(_CONFIGS_DIR, "longstream_infer_optimized.yaml"),
    "baseline":  os.path.join(_CONFIGS_DIR, "longstream_infer_baseline.yaml"),
}

# ─── Save-output choices (label -> YAML key) ─────────────────────────────
SAVE_OUTPUT_CHOICES = [
    ("RGB 图片 (save_images)",            "save_images"),
    ("视频 (save_videos)",                "save_videos"),
    ("深度图 (save_depth)",               "save_depth"),
    ("全局点云 (save_points)",            "save_points"),
    ("逐帧点云 (save_frame_points)",      "save_frame_points"),
    ("天空遮罩 [暂未支持] (mask_sky)",    "mask_sky"),
    ("GLB 导出 (export_glb)",             "export_glb"),
]
_SAVE_LABEL_TO_KEY = {label: key for label, key in SAVE_OUTPUT_CHOICES}
_SAVE_DEFAULT_LABELS = [
    "RGB 图片 (save_images)",
    "视频 (save_videos)",
    "深度图 (save_depth)",
    "全局点云 (save_points)",
    "逐帧点云 (save_frame_points)",
]

# ─── Input type mapping ───────────────────────────────────────────────────
_SRC_TYPE_MAP = {
    "图片目录 (image_dir)": "image_dir",
    "视频文件 (video)":     "video",
    "NPZ 文件 (npz)":       "npz",
    "Generalizable 目录":   "generalizable",
}

# ─── Global runner state ──────────────────────────────────────────────────
_runner_lock   = threading.Lock()
_active_runner = None
_rerun_viewer  = None


# ═════════════════════════════════════════════════════════════════════════
#  Config helpers
# ═════════════════════════════════════════════════════════════════════════

def _load_template(template_name: str) -> dict:
    """Load named template YAML. 'Custom' falls back to {} (no missing-file error)."""
    if template_name == "Custom":
        custom_path = os.path.join(_CONFIGS_DIR, "custom_infer.yaml")
        if os.path.isfile(custom_path):
            with open(custom_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    path = _TEMPLATE_PATHS.get(template_name, _TEMPLATE_PATHS["optimized"])
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_to_save_output_labels(cfg: dict) -> list:
    out = cfg.get("output", {})
    return [label for label, key in SAVE_OUTPUT_CHOICES if bool(out.get(key, False))]


def _merge_basic_into_cfg(
    cfg: dict,
    window_size: int,
    refresh: int,
    keyframe_stride: int,
    confidence_threshold: float,
    max_frame_points: int,
    max_full_points: int,
    simulate_streaming: bool,
    target_fps: int,
    enable_tto: bool,
    enable_filter: bool,
    enable_conf_filter: bool,
    save_outputs: list,
    enable_rerun: bool,
    output_root: str = "",
) -> dict:
    """Apply UI leaf-values onto base cfg. Never modifies model/checkpoint keys."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("inference", {})
    cfg.setdefault("optimizations", {})
    cfg["optimizations"].setdefault("filter", {})
    cfg["optimizations"].setdefault("correction", {})
    cfg.setdefault("output", {})
    cfg.setdefault("data", {})
    cfg.setdefault("runtime", {})

    cfg["inference"]["window_size"]     = int(window_size)
    cfg["inference"]["refresh"]         = int(refresh)
    cfg["inference"]["keyframe_stride"] = int(keyframe_stride)
    cfg["inference"]["mode"] = (
        "streaming_refresh" if simulate_streaming else "batch_refresh"
    )

    # Mirror into model.longstream_cfg (mirrors cli.py L185)
    cfg.setdefault("model", {})
    cfg["model"].setdefault("longstream_cfg", {})
    cfg["model"]["longstream_cfg"]["window_size"] = int(window_size)
    cfg["model"]["longstream_cfg"].setdefault("rel_pose_head_cfg", {})
    cfg["model"]["longstream_cfg"]["rel_pose_head_cfg"]["keyframe_stride"] = int(keyframe_stride)

    cfg["optimizations"]["filter"]["frame_filter_enabled"]      = bool(enable_filter)
    cfg["optimizations"]["filter"]["confidence_filter_enabled"] = bool(enable_conf_filter)
    cfg["optimizations"]["filter"]["confidence_threshold"]      = float(confidence_threshold)
    cfg["optimizations"]["correction"]["tto_enabled"]           = bool(enable_tto)

    selected_keys = {
        _SAVE_LABEL_TO_KEY[label]
        for label in (save_outputs or [])
        if label in _SAVE_LABEL_TO_KEY
    }
    for std_key in ["save_images", "save_videos", "save_depth",
                    "save_points", "save_frame_points", "mask_sky"]:
        cfg["output"][std_key] = std_key in selected_keys
    cfg["output"]["export_glb"]                   = "export_glb" in selected_keys
    cfg["output"]["max_frame_pointcloud_points"]  = int(max_frame_points)
    cfg["output"]["max_full_pointcloud_points"]   = int(max_full_points)
    if output_root and output_root.strip():
        cfg["output"]["root"] = output_root.strip()

    cfg["runtime"]["simulate_streaming"] = bool(simulate_streaming)
    cfg["data"]["fps"] = int(target_fps) if simulate_streaming else 0

    cfg["_enable_rerun"] = bool(enable_rerun)
    return cfg


def _apply_advanced_overrides(cfg: dict, advanced_yaml: str) -> dict:
    """Deep-merge user YAML/JSON overrides. Raises gr.Error on parse failure."""
    stripped = advanced_yaml.strip()
    if not stripped or stripped.startswith("#"):
        # All lines are comments or empty — nothing to apply
        non_comment = [l for l in stripped.splitlines() if l.strip() and not l.strip().startswith("#")]
        if not non_comment:
            return cfg
    try:
        overrides = yaml.safe_load(advanced_yaml) or {}
    except yaml.YAMLError:
        try:
            overrides = json.loads(advanced_yaml)
        except json.JSONDecodeError as exc:
            raise gr.Error(
                f"高级配置解析失败（既不是有效 YAML 也不是有效 JSON）: {exc}\n"
                "请检查格式后重试，或清空高级配置框。"
            ) from exc
    if not isinstance(overrides, dict):
        raise gr.Error("高级配置必须是 YAML/JSON 字典，请检查输入格式。")
    cfg = copy.deepcopy(cfg)
    _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ─── Template sync ────────────────────────────────────────────────────────

_ADVANCED_YAML_PLACEHOLDER = (
    "# 在此输入高级参数覆盖（仅填写需要覆盖的项），例如：\n"
    "# inference:\n"
    "#   streaming_mode: causal\n"
    "# optimizations:\n"
    "#   correction:\n"
    "#     tto_steps: 20\n"
    "#     tto_lr: 0.001\n"
)


def update_ui_from_yaml(template_name: str) -> tuple:
    """
    Load template YAML and refresh bound UI components.
    Returns 14 values:
      window_size, refresh, keyframe_stride, confidence_threshold,
      max_frame_points, enable_tto, enable_filter, enable_conf_filter,
      save_outputs, target_fps, simulate_streaming, enable_rerun,
      cfg_state (new base dict), config_preview (formatted YAML text)
    advanced_yaml is NOT updated here — it remains the user's override box.
    """
    if template_name == "Custom":
        # No-op: keep all current values unchanged
        return tuple(gr.update() for _ in range(14))

    cfg   = _load_template(template_name)
    infer = cfg.get("inference", {})
    opt   = cfg.get("optimizations", {})
    filt  = opt.get("filter", {})
    corr  = opt.get("correction", {})
    data  = cfg.get("data", {})
    rt    = cfg.get("runtime", {})

    fps_val = data.get("fps", 18) or 18
    sim_val = bool(rt.get("simulate_streaming", True))
    rerun_val = bool(cfg.get("_enable_rerun", True))

    out = cfg.get("output", {})
    data_fmt = data.get("format", "image_dir")
    is_gen = (data_fmt == "generalizable")

    preview_text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)

    return (
        gr.update(value=int(infer.get("window_size",     48))),   # 1
        gr.update(value=int(infer.get("refresh",          4))),   # 2
        gr.update(value=int(infer.get("keyframe_stride",  8))),   # 3
        gr.update(value=float(filt.get("confidence_threshold", 0.5))),  # 4
        gr.update(value=int(out.get("max_frame_pointcloud_points", 200000))),  # 5
        gr.update(value=int(out.get("max_full_pointcloud_points", 2000000))),  # 6
        gr.update(value=bool(corr.get("tto_enabled",              False))),   # 7
        gr.update(value=bool(filt.get("frame_filter_enabled",     False))),   # 8
        gr.update(value=bool(filt.get("confidence_filter_enabled", True))),   # 9
        gr.update(value=_cfg_to_save_output_labels(cfg)),   # 10
        gr.update(value=int(fps_val)),                       # 11
        gr.update(value=sim_val),                            # 12 simulate_streaming
        gr.update(value=rerun_val),                          # 13 enable_rerun
        cfg,                                                 # 14 cfg_state (new base)
        gr.update(value=preview_text),                       # 15 config_preview
        gr.update(value=out.get("root", "outputs")),        # 16 output_root
        gr.update(value=data.get("img_path", "")),          # 17 source_path
        gr.update(value=data.get("data_roots_file", "data_roots.txt")),  # 18 data_roots_file
        gr.update(value=", ".join(data.get("seq_list", ["Scene01/clone"]))),  # 19 seq_list
        gr.update(value=str(data.get("camera", "00"))),     # 20 camera
    )


# ═════════════════════════════════════════════════════════════════════════
#  Start / Stop logic
# ═════════════════════════════════════════════════════════════════════════

def _build_cfg_from_ui(
    base_cfg: dict,         # from cfg_state — preserves model/checkpoint keys
    source_path: str,
    source_type: str,
    window_size: int,
    refresh: int,
    keyframe_stride: int,
    confidence_threshold: float,
    max_frame_points: int,
    max_full_points: int,
    simulate_streaming: bool,
    target_fps: int,
    enable_tto: bool,
    enable_filter: bool,
    enable_conf_filter: bool,
    save_outputs: list,
    enable_rerun: bool,
    advanced_yaml: str,
    output_root: str = "",
    data_roots_file: str = "data_roots.txt",
    seq_list: str = "",
    camera: str = "00",
) -> dict:
    """Build complete cfg from UI values using base_cfg (from cfg_state) as the foundation."""
    cfg = _merge_basic_into_cfg(
        base_cfg,
        window_size=window_size, refresh=refresh,
        keyframe_stride=keyframe_stride,
        confidence_threshold=confidence_threshold,
        max_frame_points=max_frame_points,
        max_full_points=max_full_points,
        simulate_streaming=simulate_streaming, target_fps=target_fps,
        enable_tto=enable_tto, enable_filter=enable_filter,
        enable_conf_filter=enable_conf_filter,
        save_outputs=save_outputs, enable_rerun=enable_rerun,
        output_root=output_root,
    )
    cfg = _apply_advanced_overrides(cfg, advanced_yaml)
    cfg.setdefault("data", {})
    fmt = _SRC_TYPE_MAP.get(source_type, "image_dir")
    cfg["data"]["format"] = fmt
    if source_path and source_path.strip():
        cfg["data"]["img_path"] = source_path.strip()

    if fmt == "generalizable":
        if not (source_path and source_path.strip()):
            raise gr.Error("选择 generalizable 时，data.img_path（输入路径）必填。")
        if not seq_list.strip():
            raise gr.Error("选择 generalizable 时，data.seq_list 必填。")
        if not camera.strip():
            raise gr.Error("选择 generalizable 时，data.camera 必填。")
        cfg["data"]["data_roots_file"] = (data_roots_file or "data_roots.txt").strip()
        cfg["data"]["seq_list"] = [s.strip() for s in seq_list.split(",") if s.strip()]
        cfg["data"]["camera"] = camera.strip()
    else:
        cfg["data"].pop("seq_list", None)
        cfg["data"].pop("data_roots_file", None)
        cfg["data"].pop("camera", None)

    # 路径保护：output.root 不得与 data.img_path 重叠
    _out_root = cfg.get("output", {}).get("root", "")
    _img_path = cfg.get("data", {}).get("img_path", "")
    if _out_root and _img_path:
        import os as _os2
        _out_abs = _os2.path.abspath(_os2.path.expanduser(_out_root))
        _img_abs = _os2.path.abspath(_os2.path.expanduser(_img_path))
        if _out_abs == _img_abs or _img_abs.startswith(_out_abs + _os2.sep):
            raise gr.Error(
                f"输出目录 ({_out_root}) 与输入路径 ({_img_path}) 重叠，拒绝执行。"
            )

    return cfg


def _start_runner(
    base_cfg: dict,
    source_path: str,
    source_type: str,
    window_size: int,
    refresh: int,
    keyframe_stride: int,
    confidence_threshold: float,
    max_frame_points: int,
    max_full_points: int,
    simulate_streaming: bool,
    target_fps: int,
    enable_tto: bool,
    enable_filter: bool,
    enable_conf_filter: bool,
    save_outputs: list,
    enable_rerun: bool,
    advanced_yaml: str,
    output_root: str = "",
    data_roots_file: str = "data_roots.txt",
    seq_list: str = "",
    camera: str = "00",
) -> tuple:
    global _active_runner, _rerun_viewer

    if not source_path or not source_path.strip():
        return "警告：请先填写输入路径。", gr.update(), gr.update()

    with _runner_lock:
        if _active_runner is not None and _active_runner.is_running:
            return "警告：推理已在运行中，请先停止。", gr.update(), gr.update()

    cfg = _build_cfg_from_ui(
        base_cfg, source_path, source_type,
        window_size, refresh, keyframe_stride,
        confidence_threshold, max_frame_points, max_full_points,
        simulate_streaming, target_fps,
        enable_tto, enable_filter, enable_conf_filter,
        save_outputs, enable_rerun, advanced_yaml,
        output_root=output_root,
        data_roots_file=data_roots_file,
        seq_list=seq_list,
        camera=camera,
    )

    if simulate_streaming:
        try:
            from longstream.streaming.live_inference import LiveInferenceRunner
        except ImportError as exc:
            return f"导入失败: {exc}", gr.update(), gr.update()

        on_frame_cb = None
        if enable_rerun:
            try:
                from longstream.demo.rerun_viewer import RerunViewer
                viewer = RerunViewer(
                    confidence_threshold=float(confidence_threshold),
                    max_frame_points=int(max_frame_points),
                    spawn=True,
                )
                viewer.init()
                with _runner_lock:
                    _rerun_viewer = viewer

                def on_frame_cb(frame_idx, outputs_cpu):
                    viewer.log_frame(frame_idx, outputs_cpu)
            except ImportError as exc:
                return (
                    f"rerun-sdk 导入失败: {exc}。请运行: pip install rerun-sdk",
                    gr.update(), gr.update(),
                )

        runner = LiveInferenceRunner(cfg=cfg, on_frame=on_frame_cb)
        mode_label = "流式推理"
    else:
        try:
            from longstream.streaming.live_inference import BatchInferenceRunner
        except ImportError as exc:
            return f"导入失败: {exc}", gr.update(), gr.update()

        runner = BatchInferenceRunner(cfg=cfg)
        mode_label = "批处理推理"

    with _runner_lock:
        _active_runner = runner

    runner.start()
    return (
        f"已启动：{mode_label}",
        gr.update(interactive=False, value="运行中…"),
        gr.update(interactive=True),
    )


def _stop_runner() -> tuple:
    global _active_runner, _rerun_viewer

    with _runner_lock:
        runner = _active_runner
        viewer = _rerun_viewer

    if runner is None:
        return (
            "没有正在运行的推理。",
            gr.update(interactive=True, value="启动"),
            gr.update(interactive=False),
        )

    runner.stop()

    with _runner_lock:
        _active_runner = None
        _rerun_viewer  = None

    import torch
    torch.cuda.empty_cache()
    return (
        "已停止。",
        gr.update(interactive=True, value="启动"),
        gr.update(interactive=False),
    )


def _check_status() -> str:
    with _runner_lock:
        runner = _active_runner
    if runner is None:
        return "● 未运行"
    if runner.is_running:
        return "● 推理进行中…"
    err = runner.last_error
    if err:
        return f"● 异常退出: {err}"
    return "● 已完成"


def save_custom_config(
    base_cfg: dict,
    source_path: str,
    source_type: str,
    window_size: int,
    refresh: int,
    keyframe_stride: int,
    confidence_threshold: float,
    max_frame_points: int,
    max_full_points: int,
    simulate_streaming: bool,
    target_fps: int,
    enable_tto: bool,
    enable_filter: bool,
    enable_conf_filter: bool,
    save_outputs: list,
    enable_rerun: bool,
    advanced_yaml: str,
    output_root: str = "",
    data_roots_file: str = "data_roots.txt",
    seq_list: str = "",
    camera: str = "00",
) -> tuple:
    """Save current UI params to custom_infer.yaml. Returns (path, Custom dropdown update, new cfg_state)."""
    cfg = _build_cfg_from_ui(
        base_cfg, source_path, source_type,
        window_size, refresh, keyframe_stride,
        confidence_threshold, max_frame_points, max_full_points,
        simulate_streaming, target_fps,
        enable_tto, enable_filter, enable_conf_filter,
        save_outputs, enable_rerun, advanced_yaml,
        output_root=output_root,
        data_roots_file=data_roots_file,
        seq_list=seq_list,
        camera=camera,
    )
    custom_path = os.path.join(_CONFIGS_DIR, "custom_infer.yaml")
    os.makedirs(_CONFIGS_DIR, exist_ok=True)
    with open(custom_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    _TEMPLATE_PATHS["Custom"] = custom_path
    return custom_path, gr.update(value="Custom"), cfg


# ═════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ═════════════════════════════════════════════════════════════════════════

def main():
    _initial_cfg = _load_template("optimized")

    with gr.Blocks(title="LongStream 推理控制台") as demo:
        gr.Markdown("# LongStream 推理控制台")
        gr.Markdown(
            "配置参数后点击 **启动**。"
            "流式模拟模式逐帧推理并推送 Rerun Viewer；"
            "批处理模式无限速运行 StreamFeeder 完整序列。"
        )

        # ─── State ────────────────────────────────────────────────────
        # Holds the BASE config from the selected template.
        # User slider/checkbox edits do NOT change this state.
        # Only template_name.change() updates it (via update_ui_from_yaml).
        cfg_state = gr.State(value=_initial_cfg)

        # ─── Input source ──────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### 输入来源")
            with gr.Row():
                source_type = gr.Dropdown(
                    label="输入流类型",
                    choices=list(_SRC_TYPE_MAP.keys()),
                    value="图片目录 (image_dir)",
                )
                template_name = gr.Dropdown(
                    label="配置模板",
                    choices=["optimized", "baseline", "Custom"],
                    value="optimized",
                )
            source_path = gr.Textbox(
                label="输入路径 (data.img_path)",
                placeholder="/path/to/images_or_video_or_npz",
            )
            # Generalizable 专用字段（初始隐藏）
            with gr.Row(visible=False) as generalizable_row:
                data_roots_file = gr.Textbox(
                    label="数据根列表 (data.data_roots_file)",
                    value="data_roots.txt",
                    placeholder="data_roots.txt",
                )
                seq_list = gr.Textbox(
                    label="序列列表 (data.seq_list，逗号分隔)",
                    value="Scene01/clone",
                    placeholder="Scene01/clone",
                )
                camera = gr.Textbox(
                    label="相机编号 (data.camera)",
                    value="00",
                    placeholder="00",
                )
            output_root = gr.Textbox(
                label="输出目录 (output.root)",
                value=_initial_cfg.get("output", {}).get("root", "outputs"),
                placeholder="outputs/vkitti2_scene01_clone/optimized",
            )

        # ─── Run mode / Data Settings ──────────────────────────────────
        with gr.Group():
            gr.Markdown("### 运行模式")
            with gr.Row():
                simulate_streaming = gr.Checkbox(
                    label="启用流式模拟 (runtime.simulate_streaming)",
                    value=True,
                )
                target_fps = gr.Slider(
                    label="目标帧率 (data.fps，流式模式生效)",
                    minimum=1, maximum=60, step=1, value=18,
                )

        # ─── Basic config ──────────────────────────────────────────────
        with gr.Accordion("基础配置", open=True):
            with gr.Row():
                window_size = gr.Slider(
                    label="窗口大小 (inference.window_size)",
                    minimum=1, maximum=128, step=1, value=48,
                )
                refresh = gr.Slider(
                    label="刷新间隔 (inference.refresh)",
                    minimum=2, maximum=9, step=1, value=4,
                )
                keyframe_stride = gr.Slider(
                    label="关键帧步长 (inference.keyframe_stride)",
                    minimum=1, maximum=32, step=1, value=8,
                )
            with gr.Row():
                confidence_threshold = gr.Slider(
                    label="置信度阈值 (optimizations.filter.confidence_threshold)",
                    minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                )
                max_frame_points = gr.Slider(
                    label="每帧最大点数 (output.max_frame_pointcloud_points)",
                    minimum=1000, maximum=2000000, step=1000, value=200000,
                )
                max_full_points = gr.Slider(
                    label="全局点云最大点数 (output.max_full_pointcloud_points)",
                    minimum=10000, maximum=5000000, step=10000, value=2000000,
                )
            with gr.Row():
                enable_tto = gr.Checkbox(
                    label="启用 TTO (optimizations.correction.tto_enabled)",
                    value=False,
                )
                enable_filter = gr.Checkbox(
                    label="帧质量过滤 (optimizations.filter.frame_filter_enabled)",
                    value=False,
                )
                enable_conf_filter = gr.Checkbox(
                    label="置信度过滤 (optimizations.filter.confidence_filter_enabled)",
                    value=True,
                )
                enable_rerun = gr.Checkbox(
                    label="Rerun 渲染（仅流式模式）",
                    value=True,
                )

            save_outputs = gr.CheckboxGroup(
                label="保存输出 (output.*)",
                choices=[label for label, _ in SAVE_OUTPUT_CHOICES],
                value=_SAVE_DEFAULT_LABELS,
            )

        # ─── Advanced config ───────────────────────────────────────────
        with gr.Accordion("高级配置（YAML/JSON 额外覆盖）", open=False):
            gr.Markdown(
                "此框只填写需要**额外覆盖**的参数，无需重复填写基础配置。"
                "会在基础配置之上深度合并（deep merge）。"
            )
            advanced_yaml = gr.Code(
                label="YAML 或 JSON 覆盖项",
                language="yaml",
                value=_ADVANCED_YAML_PLACEHOLDER,
                lines=8,
            )
            gr.Markdown("#### 当前模板完整配置（只读预览）")
            config_preview = gr.Code(
                label="当前模板配置（只读）",
                language="yaml",
                value=yaml.safe_dump(_initial_cfg, allow_unicode=True, sort_keys=False),
                lines=15,
                interactive=False,
            )

        # ─── Save custom config ────────────────────────────────────────
        with gr.Accordion("保存自定义配置", open=False):
            with gr.Row():
                save_custom_btn = gr.Button("保存自定义配置")
                custom_config_path = gr.Textbox(
                    label="保存路径",
                    value=os.path.join(_CONFIGS_DIR, "custom_infer.yaml"),
                    interactive=False,
                )

        # ─── Control buttons ───────────────────────────────────────────
        with gr.Row():
            start_btn = gr.Button("启动", variant="primary", interactive=True)
            stop_btn  = gr.Button("停止", variant="stop",    interactive=False)

        status_box = gr.Textbox(
            label="运行状态", value="● 未运行", interactive=False
        )

        # ─── Status polling ────────────────────────────────────────────
        status_timer = gr.Timer(value=2.0)
        status_timer.tick(fn=_check_status, outputs=[status_box])

        # ─── Template sync ─────────────────────────────────────────────
        # update_ui_from_yaml returns 20 values.
        # advanced_yaml is intentionally NOT in this output list.
        _template_sync_outputs = [
            window_size, refresh, keyframe_stride,       # 1-3
            confidence_threshold, max_frame_points,      # 4-5
            max_full_points,                             # 6
            enable_tto, enable_filter, enable_conf_filter,  # 7-9
            save_outputs,                                # 10
            target_fps,                                  # 11
            simulate_streaming, enable_rerun,            # 12-13
            cfg_state,                                   # 14
            config_preview,                              # 15
            output_root,                                 # 16
            source_path,                                 # 17
            data_roots_file,                             # 18
            seq_list,                                    # 19
            camera,                                      # 20
        ]
        template_name.change(
            fn=update_ui_from_yaml,
            inputs=[template_name],
            outputs=_template_sync_outputs,
        )

        # ─── source_type change: show/hide generalizable fields ────────
        def _update_source_fields(source_type_val):
            is_gen = _SRC_TYPE_MAP.get(source_type_val) == "generalizable"
            return gr.update(visible=is_gen)

        source_type.change(
            fn=_update_source_fields,
            inputs=[source_type],
            outputs=[generalizable_row],
        )

        # ─── User edits -> switch dropdown to "Custom" ─────────────────
        # Use .input() / .release() so that programmatic gr.update() from
        # update_ui_from_yaml does NOT re-trigger this callback.
        def _mark_custom(*_):
            return gr.update(value="Custom")

        # Sliders: use .release() (fires only when user releases mouse)
        for _slider in [window_size, refresh, keyframe_stride,
                        confidence_threshold, max_frame_points, max_full_points, target_fps]:
            _slider.release(fn=_mark_custom, inputs=[], outputs=[template_name])

        # Checkboxes / CheckboxGroup: use .input()
        for _comp in [simulate_streaming, enable_tto, enable_filter,
                      enable_conf_filter, enable_rerun, save_outputs]:
            _comp.input(fn=_mark_custom, inputs=[], outputs=[template_name])

        # Textbox / Dropdown / Code: use .input()
        for _comp in [source_type, source_path, advanced_yaml,
                      output_root, data_roots_file, seq_list, camera]:
            _comp.input(fn=_mark_custom, inputs=[], outputs=[template_name])

        # ─── Shared input list (cfg_state replaces template_name) ─────
        _all_inputs = [
            cfg_state, source_path, source_type,
            window_size, refresh, keyframe_stride,
            confidence_threshold, max_frame_points, max_full_points,
            simulate_streaming, target_fps,
            enable_tto, enable_filter, enable_conf_filter,
            save_outputs, enable_rerun, advanced_yaml,
            output_root, data_roots_file, seq_list, camera,
        ]

        # ─── Button events ─────────────────────────────────────────────
        start_btn.click(
            fn=_start_runner,
            inputs=_all_inputs,
            outputs=[status_box, start_btn, stop_btn],
        )
        stop_btn.click(
            fn=_stop_runner,
            outputs=[status_box, start_btn, stop_btn],
        )

        # ─── Save custom config ────────────────────────────────────────
        save_custom_btn.click(
            fn=save_custom_config,
            inputs=_all_inputs,
            outputs=[custom_config_path, template_name, cfg_state],
        )

    demo.launch()


if __name__ == "__main__":
    main()
