"""
demo_gradio.py
--------------
LongStream 实时推理控制台（Gradio UI）。

功能：
  - 选择输入流类型（图片目录 / 视频 / NPZ / generalizable）
  - 选择配置模板（optimized / baseline）
  - 基础配置（中文名 + 原 YAML key 括号标注）
  - 高级配置（原始 YAML/JSON 编辑框）
  - 启动/停止按钮（后台线程 + Rerun Viewer）
"""

from __future__ import annotations

import copy
import json
import os
import threading
from typing import Optional

import gradio as gr
import yaml

# ── 配置模板路径 ──────────────────────────────────────────────────────────
_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
_TEMPLATE_PATHS = {
    "optimized": os.path.join(_CONFIGS_DIR, "longstream_infer_optimized.yaml"),
    "baseline":  os.path.join(_CONFIGS_DIR, "longstream_infer_baseline.yaml"),
}

# ── 全局推理运行器状态 ─────────────────────────────────────────────────────
_runner_lock = threading.Lock()
_active_runner = None   # LiveInferenceRunner 实例
_rerun_viewer  = None   # RerunViewer 实例


# ═══════════════════════════════════════════════════════════════════════════
#  配置工具
# ═══════════════════════════════════════════════════════════════════════════

def _load_template(template_name: str) -> dict:
    path = _TEMPLATE_PATHS.get(template_name, _TEMPLATE_PATHS["optimized"])
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_basic_into_cfg(
    cfg: dict,
    window_size: int,
    refresh: int,
    keyframe_stride: int,
    confidence_threshold: float,
    max_frame_points: int,
    fps: float,
    enable_tto: bool,
    enable_filter: bool,
    enable_conf_filter: bool,
    mask_sky: bool,
    save_output: bool,
    enable_rerun: bool,
) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("inference", {})
    cfg.setdefault("optimizations", {})
    cfg["optimizations"].setdefault("filter", {})
    cfg["optimizations"].setdefault("correction", {})
    cfg.setdefault("output", {})
    cfg.setdefault("data", {})

    cfg["inference"]["window_size"] = int(window_size)
    cfg["inference"]["refresh"] = int(refresh)
    cfg["inference"]["keyframe_stride"] = int(keyframe_stride)

    # 同步到 model.longstream_cfg（对齐 cli.py L185 的行为）
    cfg.setdefault("model", {})
    cfg["model"].setdefault("longstream_cfg", {})
    cfg["model"]["longstream_cfg"]["window_size"] = int(window_size)
    cfg["model"]["longstream_cfg"].setdefault("rel_pose_head_cfg", {})
    cfg["model"]["longstream_cfg"]["rel_pose_head_cfg"]["keyframe_stride"] = int(keyframe_stride)

    cfg["optimizations"]["filter"]["frame_filter_enabled"] = bool(enable_filter)
    cfg["optimizations"]["filter"]["confidence_filter_enabled"] = bool(enable_conf_filter)
    cfg["optimizations"]["filter"]["confidence_threshold"] = float(confidence_threshold)
    cfg["optimizations"]["correction"]["tto_enabled"] = bool(enable_tto)

    cfg["output"]["mask_sky"] = bool(mask_sky)
    cfg["output"]["save_points"] = bool(save_output)
    cfg["output"]["save_videos"] = bool(save_output)

    cfg["data"]["fps"] = float(fps)
    cfg["output"]["max_frame_pointcloud_points"] = int(max_frame_points)

    cfg["_enable_rerun"] = bool(enable_rerun)
    return cfg


def _apply_advanced_overrides(cfg: dict, advanced_yaml: str) -> dict:
    """将高级配置框中的 YAML 字符串深度合并到 cfg。解析失败时抛出 gr.Error。"""
    if not advanced_yaml.strip():
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
        raise gr.Error("高级配置必须是 YAML/JSON 字典（mapping），请检查输入格式。")
    cfg = copy.deepcopy(cfg)
    _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ═══════════════════════════════════════════════════════════════════════════
#  启动 / 停止逻辑
# ═══════════════════════════════════════════════════════════════════════════

def _start_runner(
    source_path: str,
    source_type: str,
    template_name: str,
    window_size: int,
    refresh: int,
    keyframe_stride: int,
    confidence_threshold: float,
    max_frame_points: int,
    fps: float,
    enable_tto: bool,
    enable_filter: bool,
    enable_conf_filter: bool,
    mask_sky: bool,
    save_output: bool,
    enable_rerun: bool,
    advanced_yaml: str,
) -> tuple:
    """
    点击「启动」后触发：构建配置、创建 LiveInferenceRunner，在后台启动。
    返回 (status_msg, start_btn_update, stop_btn_update)
    """
    global _active_runner, _rerun_viewer

    if not source_path or not source_path.strip():
        return "⚠ 请先填写输入路径。", gr.update(), gr.update()

    with _runner_lock:
        if _active_runner is not None and _active_runner.is_running:
            return "⚠ 推理已在运行中，请先停止。", gr.update(), gr.update()

    # 延迟导入（避免无 CUDA 环境 import 失败影响 UI 启动）
    try:
        from longstream.streaming.live_inference import LiveInferenceRunner
    except ImportError as exc:
        return f"❌ 导入失败: {exc}", gr.update(), gr.update()

    # 构建配置
    cfg = _load_template(template_name)
    cfg = _merge_basic_into_cfg(
        cfg,
        window_size=window_size,
        refresh=refresh,
        keyframe_stride=keyframe_stride,
        confidence_threshold=confidence_threshold,
        max_frame_points=max_frame_points,
        fps=fps,
        enable_tto=enable_tto,
        enable_filter=enable_filter,
        enable_conf_filter=enable_conf_filter,
        mask_sky=mask_sky,
        save_output=save_output,
        enable_rerun=enable_rerun,
    )
    cfg = _apply_advanced_overrides(cfg, advanced_yaml)

    # 更新数据源
    cfg.setdefault("data", {})
    _src_type_map = {
        "图片目录 (image_dir)": "image_dir",
        "视频文件 (video)":     "video",
        "NPZ 文件 (npz)":      "npz",
        "Generalizable 目录":  "generalizable",
    }
    cfg["data"]["format"] = _src_type_map.get(source_type, "image_dir")
    cfg["data"]["img_path"] = source_path.strip()

    # Rerun Viewer
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
            # enable_rerun=True 但导入失败：必须返回错误，不允许静默降级
            return (
                f"❌ rerun-sdk 导入失败: {exc}。请运行: pip install rerun-sdk",
                gr.update(),
                gr.update(),
            )

    runner = LiveInferenceRunner(cfg=cfg, on_frame=on_frame_cb, feeder=None)
    with _runner_lock:
        _active_runner = runner

    runner.start()

    return (
        "✅ 推理已启动，Rerun Viewer 将自动弹出（如已开启）。",
        gr.update(interactive=False, value="运行中…"),
        gr.update(interactive=True),
    )


def _stop_runner() -> tuple:
    """点击「停止」后触发。"""
    global _active_runner, _rerun_viewer

    # 先拿到引用，但不立即清空——防止 stop() 执行期间 UI 认为"无运行者"
    # 而允许再次启动，导致两个进程同时占用 GPU。
    with _runner_lock:
        runner = _active_runner
        viewer = _rerun_viewer

    if runner is None:
        return (
            "⚠ 没有正在运行的推理。",
            gr.update(interactive=True, value="启动"),
            gr.update(interactive=False),
        )

    # stop() 内部：信号 → join(3s) → terminate() → join(3s) → kill()
    # 在此期间 _active_runner 保持非 None，新启动请求会被拦截。
    runner.stop()

    # 进程已终止，安全清空全局状态
    with _runner_lock:
        _active_runner = None
        _rerun_viewer = None

    import torch
    torch.cuda.empty_cache()
    return (
        "🛑 推理已停止。",
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
        return f"● 推理异常退出: {err}"
    return "● 推理已完成"


# ═══════════════════════════════════════════════════════════════════════════
#  Gradio UI 构建
# ═══════════════════════════════════════════════════════════════════════════

def main():
    with gr.Blocks(title="LongStream 实时推理控制台") as demo:
        gr.Markdown("# LongStream 实时推理控制台")
        gr.Markdown(
            "配置推理参数后点击 **启动**，Rerun Viewer 将自动弹出并实时显示点云与轨迹。"
        )

        # ── 输入来源 ───────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### 输入来源")
            with gr.Row():
                source_type = gr.Dropdown(
                    label="输入流类型",
                    choices=[
                        "图片目录 (image_dir)",
                        "视频文件 (video)",
                        "NPZ 文件 (npz)",
                        "Generalizable 目录",
                    ],
                    value="图片目录 (image_dir)",
                )
                template_name = gr.Dropdown(
                    label="配置模板",
                    choices=["optimized", "baseline"],
                    value="optimized",
                )
            source_path = gr.Textbox(
                label="输入路径", placeholder="/path/to/images_or_video_or_npz"
            )

        # ── 基础配置 ───────────────────────────────────────────────────────
        with gr.Accordion("基础配置", open=True):
            with gr.Row():
                window_size = gr.Slider(
                    label="窗口大小（inference.window_size）",
                    minimum=1, maximum=128, step=1, value=48,
                )
                refresh = gr.Slider(
                    label="刷新间隔（inference.refresh）",
                    minimum=2, maximum=9, step=1, value=4,
                )
                keyframe_stride = gr.Slider(
                    label="关键帧步长（inference.keyframe_stride）",
                    minimum=1, maximum=32, step=1, value=8,
                )
            with gr.Row():
                confidence_threshold = gr.Slider(
                    label="置信度阈值（optimizations.filter.confidence_threshold）",
                    minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                )
                max_frame_points = gr.Slider(
                    label="每帧最大点数（max_frame_pointcloud_points）",
                    minimum=1000, maximum=100000, step=1000, value=8000,
                )
                fps = gr.Slider(
                    label="目标帧率 FPS（data.fps，0=不限速）",
                    minimum=0.0, maximum=60.0, step=1.0, value=0.0,
                )
            with gr.Row():
                enable_tto = gr.Checkbox(
                    label="启用 TTO（optimizations.correction.tto_enabled）",
                    value=False,
                )
                enable_filter = gr.Checkbox(
                    label="启用帧质量过滤（filter.frame_filter_enabled）",
                    value=False,
                )
                enable_conf_filter = gr.Checkbox(
                    label="启用置信度过滤（filter.confidence_filter_enabled）",
                    value=True,
                )
            with gr.Row():
                mask_sky = gr.Checkbox(
                    label="天空遮罩（output.mask_sky）",
                    value=True,
                )
                save_output = gr.Checkbox(
                    label="保存输出（output.save_points / save_videos）",
                    value=False,
                )
                enable_rerun = gr.Checkbox(
                    label="Rerun 渲染（启动 Rerun Viewer 窗口）",
                    value=True,
                )

        # ── 高级配置 ───────────────────────────────────────────────────────
        with gr.Accordion("高级配置（YAML/JSON 覆盖）", open=False):
            advanced_yaml = gr.Code(
                label="YAML 或 JSON 参数覆盖（将深度合并到基础配置）",
                language="yaml",
                value="# 在此输入高级参数覆盖，例如：\n"
                      "# inference:\n"
                      "#   streaming_mode: causal\n"
                      "# optimizations:\n"
                      "#   correction:\n"
                      "#     tto_steps: 20\n"
                      "#     tto_lr: 0.001\n",
                lines=10,
            )

        # ── 控制按钮 ───────────────────────────────────────────────────────
        with gr.Row():
            start_btn = gr.Button("启动", variant="primary", interactive=True)
            stop_btn  = gr.Button("停止", variant="stop",    interactive=False)

        status_box = gr.Textbox(
            label="运行状态",
            value="● 未运行",
            interactive=False,
        )

        # ── 状态轮询 ──────────────────────────────────────────────────────
        status_timer = gr.Timer(value=2.0)
        status_timer.tick(fn=_check_status, outputs=[status_box])

        # ── 按钮事件 ──────────────────────────────────────────────────────
        start_btn.click(
            fn=_start_runner,
            inputs=[
                source_path,
                source_type,
                template_name,
                window_size,
                refresh,
                keyframe_stride,
                confidence_threshold,
                max_frame_points,
                fps,
                enable_tto,
                enable_filter,
                enable_conf_filter,
                mask_sky,
                save_output,
                enable_rerun,
                advanced_yaml,
            ],
            outputs=[status_box, start_btn, stop_btn],
        )

        stop_btn.click(
            fn=_stop_runner,
            outputs=[status_box, start_btn, stop_btn],
        )

    demo.launch()


if __name__ == "__main__":
    main()
