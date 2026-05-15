"""
drone_scene -> LongStream generalizable 格式预处理脚本。

输入数据（来自 ./drone_scene/）：
  drone_scene.avi       - 视频，3603 帧，AVI 元数据 FPS=24，但实际时长约 132s
  odom_data.csv         - 局部米制里程计（ROS nav_msgs/Odometry）
  global_data.csv       - GPS 经纬高（ROS sensor_msgs/NavSatFix，备用）
  scans.pcd             - 单个聚合点云（不逐帧，不生成深度图）
  drone_scene.bag       - 原始 ROS bag（仅记录在 manifest 中）

输出结构（默认 prepared_inputs/drone_scene/）：
  data_roots.txt               - 写入一行 "drone_scene"
  drone_scene/
    images/
      00/
        000000.png, 000001.png, ...
    gps_xyz.npy                - shape [N, 3]，默认使用 odom position x/y/z
    input_manifest.json
    sync_report.json

注意事项：
  - AVI 元数据 FPS=24，对应时长约 150.1s；但 CSV 轨迹仅 132.1s，存在时间轴不一致。
    默认策略 --sync-mode stretch_to_csv_duration：将全部视频帧均匀拉伸到 CSV 时间段。
  - 不生成 cameras/ 目录（无可靠内参/外参）。
  - 不从 scans.pcd 生成逐帧 depth（它是单个聚合 PCD，不对应任意一帧）。
  - 保存图片使用 cv2.imencode + ndarray.tofile，兼容 Windows 中文路径。

用法示例：
  python scripts/drone_scene_to_generalizable.py
  python scripts/drone_scene_to_generalizable.py --sync-mode video_fps --video-fps 24
  python scripts/drone_scene_to_generalizable.py --max-frames 500 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ------------------------------------------------------------------ #
# 路径设置：将项目根加入 sys.path
# ------------------------------------------------------------------ #
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ------------------------------------------------------------------ #
# 辅助工具
# ------------------------------------------------------------------ #

def log(msg: str) -> None:
    print(msg, flush=True)


def _write_text(path: str, content: str) -> None:
    """Unicode-safe 文本写入。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_json(path: str, obj: object) -> None:
    """写入 JSON 文件（UTF-8，缩进 2）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _imsave_unicode(path: str, img: np.ndarray, ext: str = ".png") -> bool:
    """
    Unicode-safe 图片写入。
    cv2.imwrite 在 Windows 中文路径下会失败，改用 cv2.imencode + ndarray.tofile。
    """
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    np.array(buf).tofile(path)
    return True


# ------------------------------------------------------------------ #
# PCD 头信息解析
# ------------------------------------------------------------------ #

def _read_pcd_header(pcd_path: str) -> Dict[str, str]:
    """
    读取 PCD 文件头（读到 DATA 行为止），返回关键字段字典。
    仅读文件头，不加载点云数据（文件可能很大）。
    """
    header: Dict[str, str] = {}
    try:
        with open(pcd_path, "rb") as f:
            for _ in range(30):  # 头部不超过 30 行
                raw = f.readline()
                line = raw.decode("ascii", errors="replace").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    header[parts[0].upper()] = parts[1]
                elif len(parts) == 1:
                    header[parts[0].upper()] = ""
                if parts[0].upper() == "DATA":
                    break
    except Exception as e:
        header["_error"] = str(e)
    return header


# ------------------------------------------------------------------ #
# CSV 解析
# ------------------------------------------------------------------ #

def _load_odom_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    解析 odom_data.csv。

    返回
    ----
    timestamps : float64 ndarray, shape [M]，单位秒（纳秒戳 / 1e9）
    positions  : float64 ndarray, shape [M, 3]，x/y/z 局部米制坐标
    """
    import csv as _csv

    rows: List[List[str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    col = {h: i for i, h in enumerate(header)}

    # 时间戳列优先 field.header.stamp，其次 %time
    if "field.header.stamp" in col:
        ts_col = col["field.header.stamp"]
    elif "%time" in col:
        ts_col = col["%time"]
    else:
        raise KeyError(
            f"[drone_scene] odom_data.csv 缺少时间戳列 "
            f"(field.header.stamp / %%time)。已有列：{list(col.keys())[:10]}"
        )

    px_col = col["field.pose.pose.position.x"]
    py_col = col["field.pose.pose.position.y"]
    pz_col = col["field.pose.pose.position.z"]

    timestamps = np.array([float(r[ts_col]) for r in rows], dtype=np.float64) / 1e9
    positions = np.array(
        [[float(r[px_col]), float(r[py_col]), float(r[pz_col])] for r in rows],
        dtype=np.float64,
    )
    return timestamps, positions


def _load_global_csv(
    csv_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    解析 global_data.csv（ROS NavSatFix）。

    返回
    ----
    timestamps : float64 ndarray, shape [M]，单位秒
    lla        : float64 ndarray, shape [M, 3]，(latitude, longitude, altitude)
    """
    import csv as _csv

    rows: List[List[str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    col = {h: i for i, h in enumerate(header)}

    if "field.header.stamp" in col:
        ts_col = col["field.header.stamp"]
    elif "%time" in col:
        ts_col = col["%time"]
    else:
        raise KeyError(
            f"[drone_scene] global_data.csv 缺少时间戳列。已有列：{list(col.keys())[:10]}"
        )

    lat_col = col["field.latitude"]
    lon_col = col["field.longitude"]
    alt_col = col["field.altitude"]

    timestamps = np.array([float(r[ts_col]) for r in rows], dtype=np.float64) / 1e9
    lla = np.array(
        [[float(r[lat_col]), float(r[lon_col]), float(r[alt_col])] for r in rows],
        dtype=np.float64,
    )
    return timestamps, lla


def _lla_to_local_enu(lla: np.ndarray) -> np.ndarray:
    """
    将 LLA (lat, lon, alt) 数组转换为以首帧为原点的近似 ENU 局部坐标（米）。

    使用简单的等距离近似（在小范围内精度足够）：
      E = (lon - lon0) * cos(lat0_rad) * 111320
      N = (lat - lat0) * 111320
      U = alt - alt0
    返回 shape [M, 3] 的 (E, N, U) 数组。
    """
    lat0, lon0, alt0 = lla[0, 0], lla[0, 1], lla[0, 2]
    lat0_rad = np.deg2rad(lat0)
    E = (lla[:, 1] - lon0) * np.cos(lat0_rad) * 111320.0
    N = (lla[:, 0] - lat0) * 111320.0
    U = lla[:, 2] - alt0
    return np.stack([E, N, U], axis=-1)


# ------------------------------------------------------------------ #
# 时间同步与插值
# ------------------------------------------------------------------ #

def _build_frame_times(
    selected_source_indices: List[int],
    source_frame_count: int,
    csv_start: float,
    csv_end: float,
    sync_mode: str,
    video_fps: Optional[float],
    time_offset_sec: float,
    wall_start_sec: Optional[float] = None,
    wall_end_sec: Optional[float] = None,
) -> np.ndarray:
    """
    根据原始视频帧索引构建每个输出帧对应的时间戳数组。

    selected_source_indices : 实际写出帧在原始视频中的 0-based 索引列表（长度 = N）
    source_frame_count      : 原始视频实际解码总帧数（用于归一化）

    sync_mode == 'stretch_to_csv_duration'（默认）：
      frame_time[i] = csv_start + src_idx / (source_frame_count - 1) * (csv_end - csv_start)
      时间轴完全由 CSV 起止时间和原始帧位置决定，忽略视频 FPS。

    sync_mode == 'video_fps'：
      frame_time[i] = csv_start + src_idx / fps + time_offset_sec
      fps 来自 --video-fps 参数；若未指定则读取 AVI 元数据 FPS。

    sync_mode == 'wall_clock'：
      frame_time[i] = wall_start_sec + src_idx / (source_frame_count - 1) * (wall_end_sec - wall_start_sec)
      使用真实挂钟 Unix 时间轴（由 --video-start-time/--video-end-time 或 OCR 识别提供）。
    """
    src = np.asarray(selected_source_indices, dtype=np.float64)
    if sync_mode == "stretch_to_csv_duration":
        if source_frame_count <= 1:
            return np.full(len(src), csv_start, dtype=np.float64) + time_offset_sec
        t = csv_start + src / (source_frame_count - 1) * (csv_end - csv_start)
        return t + time_offset_sec
    elif sync_mode == "video_fps":
        if video_fps is None or video_fps <= 0:
            raise ValueError(
                "[drone_scene] --sync-mode video_fps 需要有效的 --video-fps 值（> 0）。"
            )
        t = csv_start + src / video_fps
        return t + time_offset_sec
    elif sync_mode == "wall_clock":
        if wall_start_sec is None or wall_end_sec is None:
            raise ValueError(
                "[drone_scene] wall_clock 模式需要 wall_start_sec/wall_end_sec。\n"
                "  请传 --video-start-time / --video-end-time，"
                "或使用 --ocr-time-overlay。"
            )
        if source_frame_count <= 1:
            return np.full(len(src), wall_start_sec, dtype=np.float64)
        t = wall_start_sec + src / (source_frame_count - 1) * (wall_end_sec - wall_start_sec)
        return t
    else:
        raise ValueError(f"[drone_scene] 未知 --sync-mode: {sync_mode!r}")


def _interp_positions(
    query_times: np.ndarray,
    ref_times: np.ndarray,
    ref_positions: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    将 ref_positions（已有时间戳 ref_times）线性插值到 query_times。

    超出 [ref_times[0], ref_times[-1]] 范围的 query 时间按边界 clamp。

    返回
    ----
    interp_pos : float64 ndarray, shape [len(query_times), 3]
    n_clamped  : int，被 clamp 的帧数
    """
    t_min, t_max = ref_times[0], ref_times[-1]
    clamped = np.clip(query_times, t_min, t_max)
    n_clamped = int(np.sum((query_times < t_min) | (query_times > t_max)))

    # 对每个轴分别插值
    interp_pos = np.stack(
        [
            np.interp(clamped, ref_times, ref_positions[:, dim])
            for dim in range(ref_positions.shape[1])
        ],
        axis=-1,
    )
    return interp_pos.astype(np.float64), n_clamped


# ------------------------------------------------------------------ #
# 挂钟时间 OCR 识别
# ------------------------------------------------------------------ #

def _extract_timestamp_from_ocr_text(raw: str) -> Optional[str]:
    """
    从 OCR 原始文本中提取时间戳，容忍常见 OCR 识别错误。

    支持以下时间格式（日期与时间之间可无空格）：
      HH:MM:SS   标准格式
      HH:MMXSS   中间多读一个字符，如 14:15355 → 14:15:55
      HH:MMSS    MM 后无分隔符
      HHMMSS     全数字
    对 HH(0-23)/MM/SS(0-59) 做合法性校验。返回 "YYYY-MM-DD HH:MM:SS" 或 None。
    """
    text = raw.replace("/", "-")
    date_m = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
    if not date_m:
        return None
    year, month, day = date_m.group(1), date_m.group(2), date_m.group(3)
    try:
        if not (2000 <= int(year) <= 2100 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31):
            return None
    except ValueError:
        return None

    # 日期之后的文本，跳过非数字前缀（空格/T/下划线等）
    after = text[date_m.end():]
    i = 0
    while i < len(after) and not after[i].isdigit():
        i += 1
    tp = after[i : i + 16]  # time_part

    hh = mm = ss = None
    # 1. 严格 HH:MM:SS
    m = re.match(r'(\d{2}):(\d{2}):(\d{2})', tp)
    if m:
        hh, mm, ss = m.group(1), m.group(2), m.group(3)
    # 2. HH:MM + 任意1字符(包含误读字符) + SS — 如 14:15355 → 14:15:55
    if hh is None:
        m = re.match(r'(\d{2}):(\d{2}).(\d{2})', tp)
        if m:
            hh, mm, ss = m.group(1), m.group(2), m.group(3)
    # 3. HH:MMSS — SS 前无分隔符
    if hh is None:
        m = re.match(r'(\d{2}):(\d{2})(\d{2})', tp)
        if m:
            hh, mm, ss = m.group(1), m.group(2), m.group(3)
    # 4. HHMMSS — 全数字
    if hh is None:
        m = re.match(r'(\d{2})(\d{2})(\d{2})', tp)
        if m:
            hh, mm, ss = m.group(1), m.group(2), m.group(3)
    if hh is None:
        return None
    try:
        if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59 and 0 <= int(ss) <= 59):
            return None
    except ValueError:
        return None
    return f"{year}-{month}-{day} {hh}:{mm}:{ss}"


def _ocr_video_frame_timestamp(
    frame: np.ndarray,
    roi_xywh: Tuple[int, int, int, int],
    save_debug_path: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    从视频帧的 ROI 中 OCR 识别 "YYYY-MM-DD HH:MM:SS" 格式的时间戳。

    依赖 pytesseract（可选）；未安装时返回 (None, "<pytesseract not installed>")。
    同时尝试正常和反色二值化，优先返回能匹配时间格式的结果。

    返回 (normalized_timestamp_str_or_None, raw_ocr_text)
    """
    try:
        import pytesseract  # type: ignore
    except ImportError:
        return None, "<pytesseract not installed>"

    x, y, w, h = roi_xywh
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return None, "<empty roi>"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scale = 3
    gray_up = cv2.resize(
        gray,
        (gray.shape[1] * scale, gray.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    _cfg = (
        "--psm 6 --oem 3 "
        "-c tessedit_char_whitelist="
        "0123456789-:/. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    best_ts: Optional[str] = None
    best_raw: str = ""

    for inv in (False, True):
        img = cv2.bitwise_not(gray_up) if inv else gray_up.copy()
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if save_debug_path:
            suffix = "_inv" if inv else "_norm"
            dbg = save_debug_path.replace(".png", f"{suffix}.png")
            _imsave_unicode(dbg, thresh, ".png")
        try:
            raw = pytesseract.image_to_string(thresh, config=_cfg)
        except Exception as exc:
            raw = f"<ocr error: {exc}>"
        ts = _extract_timestamp_from_ocr_text(raw)
        if ts is not None:
            best_ts = ts
            best_raw = raw.strip()
            break
        if not best_raw:
            best_raw = raw.strip()

    return best_ts, best_raw


def _read_first_last_frames(
    video_path: str,
    seek_frame_count: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    读取视频第一帧和最后一帧，不做全量解码。

    seek_frame_count : 用于 seek 末帧的元数据帧数（来自 CAP_PROP_FRAME_COUNT）。
                       seek 失败时自动顺序读到最后一帧。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    ret, first_frame = cap.read()
    first_frame = first_frame if ret else None

    last_frame: Optional[np.ndarray] = None
    if seek_frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame_count - 1)
        ret, last_frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret2, frame = cap.read()
                if not ret2:
                    break
                last_frame = frame
    cap.release()
    return first_frame, last_frame


def _parse_timestamp_to_unix(ts_str: str, timezone_str: str) -> float:
    """
    将 "YYYY-MM-DD HH:MM:SS" 解析为 UTC Unix seconds。

    依次尝试 pytz → dateutil → 硬编码 UTC+8 fallback。
    """
    from datetime import datetime

    ts_str = ts_str.strip()
    fmt = "%Y-%m-%d %H:%M:%S"

    try:
        import pytz  # type: ignore
        tz = pytz.timezone(timezone_str)
        dt = datetime.strptime(ts_str, fmt)
        return tz.localize(dt).timestamp()
    except ImportError:
        pass

    try:
        from dateutil import tz as _dtz  # type: ignore
        tz_obj = _dtz.gettz(timezone_str)
        if tz_obj is None:
            raise ValueError(f"unknown timezone: {timezone_str!r}")
        dt = datetime.strptime(ts_str, fmt).replace(tzinfo=tz_obj)
        return dt.timestamp()
    except ImportError:
        pass

    from datetime import timezone, timedelta
    log(
        f"[drone_scene] WARN: pytz/dateutil 均不可用，"
        f"假设 {timezone_str!r} = UTC+8 进行时间解析。"
    )
    dt = datetime.strptime(ts_str, fmt).replace(
        tzinfo=timezone(timedelta(hours=8))
    )
    return dt.timestamp()


def _resolve_wall_clock_times(
    video_path: str,
    seek_frame_count: int,
    video_start_time: Optional[str],
    video_end_time: Optional[str],
    timezone_str: str,
    ocr_enabled: bool,
    ocr_roi: Tuple[int, int, int, int],
    save_ocr_debug: bool,
    scene_out: str,
) -> Tuple[Optional[float], Optional[float], str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    解析视频挂钟起止时间。

    返回 (start_sec, end_sec, time_source, start_str, end_str, ocr_start_raw, ocr_end_raw)。
    time_source: "manual" | "ocr" | "none"
    start_str/end_str: 人类可读的时间字符串（用于 sync_report）。
    """
    # 手动参数优先
    if video_start_time is not None and video_end_time is not None:
        start_sec = _parse_timestamp_to_unix(video_start_time, timezone_str)
        end_sec   = _parse_timestamp_to_unix(video_end_time,   timezone_str)
        log(
            f"[drone_scene] wall_clock 手动时间: "
            f"[{video_start_time}] -> [{video_end_time}]，"
            f"Unix=[{start_sec:.3f}, {end_sec:.3f}]"
        )
        return start_sec, end_sec, "manual", video_start_time, video_end_time, None, None

    if not ocr_enabled:
        return None, None, "none", None, None, None, None

    log("[drone_scene] wall_clock OCR: 读取视频首末帧...")
    first_frame, last_frame = _read_first_last_frames(video_path, seek_frame_count)

    debug_dir: Optional[str] = None
    if save_ocr_debug:
        debug_dir = os.path.join(scene_out, "debug_ocr")
        os.makedirs(debug_dir, exist_ok=True)

    ocr_start_raw: Optional[str] = None
    ocr_end_raw:   Optional[str] = None
    start_ts: Optional[str] = None
    end_ts:   Optional[str] = None

    if first_frame is not None:
        dbg = os.path.join(debug_dir, "first_frame_roi.png") if debug_dir else None
        start_ts, ocr_start_raw = _ocr_video_frame_timestamp(
            first_frame, ocr_roi, save_debug_path=dbg
        )
        log(f"[drone_scene] OCR 首帧: {start_ts!r}  raw={ocr_start_raw[:80]!r}")
    else:
        log("[drone_scene] WARN: OCR 首帧读取失败")

    if last_frame is not None:
        dbg = os.path.join(debug_dir, "last_frame_roi.png") if debug_dir else None
        end_ts, ocr_end_raw = _ocr_video_frame_timestamp(
            last_frame, ocr_roi, save_debug_path=dbg
        )
        log(f"[drone_scene] OCR 末帧: {end_ts!r}  raw={ocr_end_raw[:80]!r}")
    else:
        log("[drone_scene] WARN: OCR 末帧读取失败")

    if start_ts is None or end_ts is None:
        missing = []
        if start_ts is None:
            missing.append("首帧")
        if end_ts is None:
            missing.append("末帧")
        log(
            f"[drone_scene] WARN: OCR 未能识别时间戳（{'/'.join(missing)}），"
            "请手动传 --video-start-time 和 --video-end-time。"
        )
        return None, None, "none", None, None, ocr_start_raw, ocr_end_raw

    start_sec = _parse_timestamp_to_unix(start_ts, timezone_str)
    end_sec   = _parse_timestamp_to_unix(end_ts,   timezone_str)
    log(
        f"[drone_scene] wall_clock OCR 成功: "
        f"[{start_ts}] -> [{end_ts}]，"
        f"Unix=[{start_sec:.3f}, {end_sec:.3f}]"
    )
    return start_sec, end_sec, "ocr", start_ts, end_ts, ocr_start_raw, ocr_end_raw


# ------------------------------------------------------------------ #
# 视频抽帧
# ------------------------------------------------------------------ #

def _extract_frames(
    video_path: str,
    image_dir: str,
    camera_id: str,
    image_ext: str,
    max_frames: Optional[int],
    overwrite: bool,
) -> Tuple[int, int, int, float, int, List[int], int]:
    """
    从视频中按顺序抽帧并保存。

    若 max_frames 指定，则均匀抽取至多 max_frames 帧；否则抽取全部帧。
    两种情况均采用流式读写，不把全部帧加载到内存。

    返回
    ----
    frames_written          : int   实际成功写入帧数
    width                   : int   视频宽
    height                  : int   视频高
    meta_fps                : float AVI 元数据中的 FPS
    meta_frame_count        : int   AVI 元数据中的帧数
    selected_source_indices : List[int]  写出帧在原始视频中的索引
    actual_decoded_count    : int   实际解码总帧数
    """
    out_dir = os.path.join(image_dir, camera_id)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[drone_scene] 无法打开视频：{video_path}")

    meta_fps = float(cap.get(cv2.CAP_PROP_FPS))
    meta_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log(
        f"[drone_scene] 视频元数据: FPS={meta_fps}, "
        f"FRAME_COUNT={meta_frame_count}, W={width}, H={height}"
    )

    ext = f".{image_ext}"
    do_subsample = (max_frames is not None and max_frames > 0)

    if do_subsample:
        # ---- Pass 1: 统计实际总帧数（不存储帧数据） ----
        log("[drone_scene] Pass 1: 统计实际总帧数...")
        actual_decoded_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            actual_decoded_count += 1
        cap.release()
        log(f"[drone_scene] 实际解码帧数: {actual_decoded_count}")

        if max_frames >= actual_decoded_count:
            pick_indices: List[int] = list(range(actual_decoded_count))
        else:
            pick_indices = (
                np.linspace(0, actual_decoded_count - 1, max_frames)
                .round()
                .astype(int)
                .tolist()
            )
        # src_idx -> out_idx 映射
        pick_map: Dict[int, int] = {
            src: out for out, src in enumerate(pick_indices)
        }

        # ---- Pass 2: 流式写出选定帧 ----
        log("[drone_scene] Pass 2: 写出选定帧...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[drone_scene] Pass 2 无法重新打开视频：{video_path}")

        frames_written = 0
        selected_source_indices: List[int] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in pick_map:
                out_idx = pick_map[frame_idx]
                out_name = f"{out_idx:06d}{ext}"
                out_path = os.path.join(out_dir, out_name)
                if not overwrite and os.path.exists(out_path):
                    frames_written += 1
                    selected_source_indices.append(frame_idx)
                else:
                    ok = _imsave_unicode(out_path, frame, ext)
                    if ok:
                        frames_written += 1
                        selected_source_indices.append(frame_idx)
                    else:
                        log(f"[drone_scene] WARN: 编码失败 frame {frame_idx} -> {out_name}")
            frame_idx += 1
        cap.release()

    else:
        # ---- 全量抽帧：单遍流式写出 ----
        log("[drone_scene] 全量抽帧（单遍流式）...")
        frames_written = 0
        selected_source_indices = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_name = f"{frame_idx:06d}{ext}"
            out_path = os.path.join(out_dir, out_name)
            if not overwrite and os.path.exists(out_path):
                frames_written += 1
                selected_source_indices.append(frame_idx)
            else:
                ok = _imsave_unicode(out_path, frame, ext)
                if ok:
                    frames_written += 1
                    selected_source_indices.append(frame_idx)
                else:
                    log(f"[drone_scene] WARN: 编码失败 frame {frame_idx} -> {out_name}")
            frame_idx += 1
        cap.release()
        actual_decoded_count = frame_idx
        log(f"[drone_scene] 实际解码帧数: {actual_decoded_count}")

    return frames_written, width, height, meta_fps, meta_frame_count, selected_source_indices, actual_decoded_count


# ------------------------------------------------------------------ #
# 主处理函数
# ------------------------------------------------------------------ #

def process_drone_scene(
    src: str,
    out: str,
    scene_name: str,
    camera_id: str,
    image_ext: str,
    overwrite: bool,
    max_frames: Optional[int],
    sync_mode: str,
    video_fps: Optional[float],
    time_offset_sec: float,
    gps_source: str,
    # wall_clock 相关参数
    video_start_time: Optional[str] = None,
    video_end_time: Optional[str] = None,
    timezone: str = "Asia/Shanghai",
    ocr_time_overlay: bool = False,
    ocr_roi: Tuple[int, int, int, int] = (0, 0, 420, 80),
    save_ocr_debug: bool = False,
) -> Dict:
    """
    将 drone_scene 数据转换为 LongStream generalizable 格式。

    参数
    ----
    src              : 输入目录（含 drone_珎ne.avi / odom_data.csv 等）
    out              : 输出根目录（meta_root，对应 data_roots.txt 所在目录）
    scene_name       : 场景名，输出到 <out>/<scene_name>/
    camera_id        : 相机 ID，如 "00"
    image_ext        : 图片格式，"png" 或 "jpg"
    overwrite        : 是否覆盖已有输出
    max_frames       : 最大帧数（None 表示全部）
    sync_mode        : 时间同步策略
    video_fps        : 覆盖视频 FPS（仅 video_fps 模式下使用）
    time_offset_sec  : 视频时间偏移（秒）
    gps_source       : GPS 来源，"odom" / "global" / "none"
    video_start_time : 视频起始时间 "YYYY-MM-DD HH:MM:SS"（wall_clock 模式）
    video_end_time   : 视频结束时间 "YYYY-MM-DD HH:MM:SS"（wall_clock 模式）
    timezone         : 时区字符串，默认 "Asia/Shanghai"
    ocr_time_overlay : 是否启用 OCR 识别时间戳水印
    ocr_roi          : OCR 识别区域 (x, y, w, h)
    save_ocr_debug   : 是否保存 OCR 调试图片到 debug_ocr/

    返回摘要字典。
    """
    src = os.path.abspath(src)
    out = os.path.abspath(out)
    scene_out = os.path.join(out, scene_name)
    image_dir = os.path.join(scene_out, "images")

    # 检查输出目录
    cam_dir = os.path.join(image_dir, camera_id)
    if os.path.isdir(cam_dir):
        existing_imgs = [
            f for f in os.listdir(cam_dir)
            if f.lower().endswith(f".{image_ext}")
        ]
        if existing_imgs:
            if not overwrite:
                raise FileExistsError(
                    f"[drone_scene] 输出目录已存在且包含 {len(existing_imgs)} 张图片："
                    f"{cam_dir}\n"
                    f"  请使用 --overwrite 强制覆盖，或手动删除该目录。"
                )
            else:
                import shutil
                shutil.rmtree(scene_out)
                log(f"[drone_scene] --overwrite: 已删除旧输出目录 {scene_out}")
    os.makedirs(scene_out, exist_ok=True)

    warnings: List[str] = []

    # ---- 1. 读取输入文件路径 ----
    video_path   = os.path.join(src, "drone_scene.avi")
    odom_csv     = os.path.join(src, "odom_data.csv")
    global_csv   = os.path.join(src, "global_data.csv")
    pcd_path     = os.path.join(src, "scans.pcd")
    bag_path     = os.path.join(src, "drone_scene.bag")

    for p, name in [
        (video_path, "drone_scene.avi"),
        (odom_csv,   "odom_data.csv"),
    ]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[drone_scene] 必须文件缺失：{p}")

    # ---- 2. 解析 CSV ----
    log("[drone_scene] 解析 odom_data.csv...")
    odom_times, odom_positions = _load_odom_csv(odom_csv)
    csv_start = float(odom_times[0])
    csv_end   = float(odom_times[-1])
    csv_duration = csv_end - csv_start
    log(
        f"[drone_scene] odom: {len(odom_times)} 行, "
        f"t=[{csv_start:.3f}, {csv_end:.3f}] s, duration={csv_duration:.2f} s"
    )

    global_times: Optional[np.ndarray] = None
    global_lla:   Optional[np.ndarray] = None
    if os.path.isfile(global_csv):
        log("[drone_scene] 解析 global_data.csv...")
        try:
            global_times, global_lla = _load_global_csv(global_csv)
            log(
                f"[drone_scene] global: {len(global_times)} 行, "
                f"lat=[{global_lla[:,0].min():.6f}, {global_lla[:,0].max():.6f}], "
                f"lon=[{global_lla[:,1].min():.6f}, {global_lla[:,1].max():.6f}]"
            )
        except Exception as e:
            warnings.append(f"global_data.csv 解析失败: {e}")
            log(f"[drone_scene] WARN: {warnings[-1]}")
    else:
        warnings.append(f"global_data.csv 不存在，跳过: {global_csv}")

    # ---- 3. 读取 PCD 头信息 ----
    pcd_header: Dict[str, str] = {}
    if os.path.isfile(pcd_path):
        pcd_header = _read_pcd_header(pcd_path)
        log(
            f"[drone_scene] scans.pcd 头: "
            f"POINTS={pcd_header.get('POINTS','?')}, "
            f"FIELDS={pcd_header.get('FIELDS','?')}"
        )
    else:
        warnings.append(f"scans.pcd 不存在，跳过: {pcd_path}")

    # ---- 3b. 挂钟时间解析（wall_clock / auto_wall_clock 模式） ----
    wall_start_sec: Optional[float] = None
    wall_end_sec:   Optional[float] = None
    wall_time_source: str = "none"
    wall_start_str: Optional[str] = None
    wall_end_str:   Optional[str] = None
    ocr_start_raw: Optional[str] = None
    ocr_end_raw:   Optional[str] = None
    resolved_sync_mode = sync_mode  # 可能因 OCR 失败而降级

    if sync_mode in ("wall_clock", "auto_wall_clock"):
        # auto_wall_clock 总启用 OCR；wall_clock 在未提供手动时间时也启用
        ocr_enabled = (
            ocr_time_overlay
            or sync_mode == "auto_wall_clock"
            or (video_start_time is None or video_end_time is None)
        )
        # 提前获取视频元数据，用于 OCR seek（不做全量解码）
        _meta_fps_pre, _meta_fc_pre = _get_video_meta(video_path)
        (
            wall_start_sec, wall_end_sec, wall_time_source,
            wall_start_str, wall_end_str,
            ocr_start_raw, ocr_end_raw,
        ) = _resolve_wall_clock_times(
            video_path=video_path,
            seek_frame_count=_meta_fc_pre,
            video_start_time=video_start_time,
            video_end_time=video_end_time,
            timezone_str=timezone,
            ocr_enabled=ocr_enabled,
            ocr_roi=ocr_roi,
            save_ocr_debug=save_ocr_debug,
            scene_out=scene_out,
        )
        if wall_start_sec is None:
            if sync_mode == "wall_clock":
                raise RuntimeError(
                    "[drone_scene] wall_clock 模式无法确定视频起止时间。\n"
                    "  请传 --video-start-time / --video-end-time，"
                    "或使用 --ocr-time-overlay 启用 OCR 识别。"
                )
            else:  # auto_wall_clock
                warn_msg = (
                    "auto_wall_clock: 挂钟时间获取失败，"
                    "回退到 stretch_to_csv_duration。"
                )
                warnings.append(warn_msg)
                log(f"[drone_scene] WARN: {warn_msg}")
                resolved_sync_mode = "stretch_to_csv_duration"
        else:
            resolved_sync_mode = "wall_clock"

    # ---- 4. 视频抽帧 ----
    log("[drone_scene] 开始抽帧...")
    frames_written, vid_w, vid_h, meta_fps, meta_frame_count, selected_source_indices, actual_decoded_count = _extract_frames(
        video_path=video_path,
        image_dir=image_dir,
        camera_id=camera_id,
        image_ext=image_ext,
        max_frames=max_frames,
        overwrite=overwrite,
    )
    N = frames_written
    log(f"[drone_scene] 抽帧完成，写入 {N} 帧，图像尺寸 {vid_w}x{vid_h}")
    log(f"[drone_scene] 原始帧索引范围: [{selected_source_indices[0] if N else 0}, {selected_source_indices[-1] if N else 0}] / {actual_decoded_count}")
    meta_video_duration = meta_frame_count / max(meta_fps, 1e-6)

    if N == 0:
        raise RuntimeError("[drone_scene] 未能写入任何帧，请检查视频文件。")

    # ---- 5. 构建帧时间戳 ----
    effective_fps = video_fps
    if resolved_sync_mode == "video_fps" and (effective_fps is None or effective_fps <= 0):
        effective_fps = meta_fps
        if effective_fps <= 0:
            effective_fps = 24.0
            warnings.append(
                f"AVI 元数据 FPS={meta_fps}，不可用，回退到 24.0"
            )
        log(f"[drone_scene] video_fps 模式使用 FPS={effective_fps}")

    frame_times = _build_frame_times(
        selected_source_indices=selected_source_indices,
        source_frame_count=actual_decoded_count,
        csv_start=csv_start,
        csv_end=csv_end,
        sync_mode=resolved_sync_mode,
        video_fps=effective_fps,
        time_offset_sec=time_offset_sec,
        wall_start_sec=wall_start_sec,
        wall_end_sec=wall_end_sec,
    )

    # ---- 6. 生成 gps_xyz.npy ----
    gps_xyz: Optional[np.ndarray] = None
    gps_source_used = gps_source
    n_clamped = 0

    if gps_source == "odom":
        log("[drone_scene] 用 odom_data.csv 位置插值生成 gps_xyz.npy...")
        gps_xyz, n_clamped = _interp_positions(frame_times, odom_times, odom_positions)
    elif gps_source == "global":
        if global_times is None or global_lla is None:
            warnings.append(
                "gps_source=global 但 global_data.csv 不可用，回退到 odom。"
            )
            log(f"[drone_scene] WARN: {warnings[-1]}")
            gps_source_used = "odom"
            gps_xyz, n_clamped = _interp_positions(
                frame_times, odom_times, odom_positions
            )
        else:
            log("[drone_scene] 将 global LLA 转换为近似 ENU，插值生成 gps_xyz.npy...")
            global_enu = _lla_to_local_enu(global_lla)
            gps_xyz, n_clamped = _interp_positions(
                frame_times, global_times, global_enu
            )
    elif gps_source == "none":
        log("[drone_scene] gps_source=none，跳过 gps_xyz.npy 生成。")
        gps_source_used = "none"
    else:
        raise ValueError(f"[drone_scene] 未知 --gps-source: {gps_source!r}")

    if gps_xyz is not None:
        assert gps_xyz.shape == (N, 3), (
            f"[drone_scene] gps_xyz shape 错误: {gps_xyz.shape}，期望 ({N}, 3)"
        )
        gps_npy_path = os.path.join(scene_out, "gps_xyz.npy")
        np.save(gps_npy_path, gps_xyz)
        log(f"[drone_scene] gps_xyz.npy 已写入，shape={gps_xyz.shape}")

        if n_clamped > 0:
            warn_msg = (
                f"{n_clamped}/{N} 帧时间超出 CSV 范围，已按边界 clamp。"
            )
            warnings.append(warn_msg)
            log(f"[drone_scene] WARN: {warn_msg}")

    # ---- 7. 写 sync_report.json ----
    if resolved_sync_mode == "video_fps":
        _eff_fps: Optional[float] = effective_fps
    elif (
        resolved_sync_mode == "wall_clock"
        and wall_start_sec is not None
        and wall_end_sec is not None
        and wall_end_sec != wall_start_sec
    ):
        _eff_fps = (
            round((actual_decoded_count - 1) / (wall_end_sec - wall_start_sec), 4)
            if actual_decoded_count > 1
            else None
        )
    else:
        _eff_fps = round((N - 1) / csv_duration, 4) if (N > 1 and csv_duration > 0) else None

    sync_report = {
        "sync_mode": sync_mode,
        "resolved_sync_mode": resolved_sync_mode,
        "n_frames_written": N,
        "csv_start_sec": csv_start,
        "csv_end_sec": csv_end,
        "csv_duration_sec": csv_duration,
        "video_meta_fps": meta_fps,
        "video_meta_frame_count": meta_frame_count,
        "video_meta_duration_sec": round(meta_video_duration, 3),
        "effective_fps_used": _eff_fps,
        "time_offset_sec": time_offset_sec,
        "frame_times_first": float(frame_times[0]),
        "frame_times_last": float(frame_times[-1]),
        "frame_times_span_sec": float(frame_times[-1] - frame_times[0]),
        "first_source_index": selected_source_indices[0],
        "last_source_index": selected_source_indices[-1],
        "source_frame_count": actual_decoded_count,
        "actual_decoded_count": actual_decoded_count,
        "gps_source": gps_source_used,
        "n_frames_clamped": n_clamped,
        "warnings": warnings,
    }
    if sync_mode in ("wall_clock", "auto_wall_clock"):
        _wall_dur = (
            round(wall_end_sec - wall_start_sec, 3)
            if (wall_start_sec is not None and wall_end_sec is not None)
            else None
        )
        # 原始视频有效 FPS（基于挂钟时长）
        _src_eff_fps = (
            round((actual_decoded_count - 1) / (wall_end_sec - wall_start_sec), 4)
            if (_wall_dur is not None and _wall_dur > 0 and actual_decoded_count > 1)
            else None
        )
        # 输出采样帧率（基于实际帧时间跨度）
        _out_sample_fps = (
            round((N - 1) / float(frame_times[-1] - frame_times[0]), 4)
            if (N > 1 and float(frame_times[-1] - frame_times[0]) > 0)
            else None
        )
        sync_report.update({
            "video_time_source": wall_time_source,
            "video_overlay_start_time": wall_start_str,
            "video_overlay_end_time": wall_end_str,
            "video_overlay_duration_sec": _wall_dur,
            "source_effective_fps": _src_eff_fps,
            "output_sample_fps": _out_sample_fps,
            "ocr_start_raw_text": ocr_start_raw,
            "ocr_end_raw_text": ocr_end_raw,
        })
    _write_json(os.path.join(scene_out, "sync_report.json"), sync_report)
    log("[drone_scene] sync_report.json 已写入。")

    # ---- 8. 写 input_manifest.json ----
    manifest = {
        "scene_name": scene_name,
        "camera_id": camera_id,
        "image_ext": image_ext,
        "n_frames": N,
        "image_size": {"width": vid_w, "height": vid_h},
        "sources": {
            "video": {
                "path": os.path.relpath(video_path, out),
                "meta_fps": meta_fps,
                "meta_frame_count": meta_frame_count,
            },
            "odom_csv": {
                "path": os.path.relpath(odom_csv, out),
                "n_rows": len(odom_times),
                "t_start_sec": csv_start,
                "t_end_sec": csv_end,
                "duration_sec": round(csv_duration, 3),
                "position_fields": [
                    "field.pose.pose.position.x",
                    "field.pose.pose.position.y",
                    "field.pose.pose.position.z",
                ],
                "timestamp_field": "field.header.stamp",
            },
            "global_csv": {
                "path": os.path.relpath(global_csv, out) if os.path.isfile(global_csv) else None,
                "available": global_times is not None,
                "n_rows": len(global_times) if global_times is not None else 0,
            },
            "scans_pcd": {
                "path": os.path.relpath(pcd_path, out) if os.path.isfile(pcd_path) else None,
                "available": bool(pcd_header),
                "header": {
                    k: v for k, v in pcd_header.items()
                    if k in ("VERSION", "FIELDS", "SIZE", "TYPE", "COUNT",
                             "WIDTH", "HEIGHT", "POINTS", "DATA")
                },
                "note": (
                    "单个聚合 PCD，非逐帧相机 z-depth，不生成深度图。"
                ),
            },
            "bag": {
                "path": os.path.relpath(bag_path, out) if os.path.isfile(bag_path) else None,
                "available": os.path.isfile(bag_path),
            },
        },
        "gps_xyz": {
            "available": gps_xyz is not None,
            "source": gps_source_used,
            "shape": list(gps_xyz.shape) if gps_xyz is not None else None,
        },
        "cameras": {
            "note": (
                "无可靠相机内参（intrinsics）及 camera-to-base_link 外参（extrinsics），"
                "未生成 cameras/ 目录。"
            ),
        },
        "sync": {
            "mode": sync_mode,
            "n_frames_clamped": n_clamped,
        },
    }
    _write_json(os.path.join(scene_out, "input_manifest.json"), manifest)
    log("[drone_scene] input_manifest.json 已写入。")

    # ---- 9. 写 data_roots.txt ----
    _write_text(os.path.join(out, "data_roots.txt"), f"{scene_name}\n")
    log(f"[drone_scene] data_roots.txt 已写入（{scene_name}）。")

    # ---- 10. 轨迹跨度统计 ----
    if gps_xyz is not None:
        traj_span = float(np.linalg.norm(gps_xyz[-1] - gps_xyz[0]))
        traj_total = float(np.sum(np.linalg.norm(np.diff(gps_xyz, axis=0), axis=1)))
    else:
        traj_span = 0.0
        traj_total = 0.0

    # 动态生成 duration_assumptions.note
    if resolved_sync_mode == "wall_clock":
        _wdur = sync_report.get("video_overlay_duration_sec")
        _sfps = sync_report.get("source_effective_fps")
        _da_note = (
            f"wall_clock 模式：视频角标时间 "
            f"[{wall_start_str or '?'}] → [{wall_end_str or '?'}]，"
            f"时长 {_wdur}s，source_effective_fps≈{_sfps}。"
        )
    elif resolved_sync_mode == "video_fps":
        _da_note = (
            f"video_fps 模式：FPS={effective_fps}，以 CSV 起始时间对齐；"
            "超出 CSV 范围的帧已按边界 clamp。"
        )
    else:  # stretch_to_csv_duration（默认或 auto_wall_clock 回退）
        _da_note = (
            f"stretch_to_csv_duration 模式：AVI 元数据时长约 {round(meta_video_duration, 1)}s，"
            f"CSV 轨迹 {round(csv_duration, 1)}s，视频帧均匀映射到 CSV 时间段。"
        )

    summary = {
        "frames_written": N,
        "image_size": f"{vid_w}x{vid_h}",
        "gps_source": gps_source_used,
        "gps_xyz_shape": list(gps_xyz.shape) if gps_xyz is not None else None,
        "trajectory_span_m": round(traj_span, 3),
        "trajectory_total_m": round(traj_total, 3),
        "duration_assumptions": {
            "video_meta_fps": meta_fps,
            "video_meta_frames": meta_frame_count,
            "video_meta_duration_sec": round(meta_video_duration, 3),
            "csv_duration_sec": round(csv_duration, 3),
            "sync_mode": sync_mode,
            "resolved_sync_mode": resolved_sync_mode,
            "note": _da_note,
        },
        "warnings": warnings,
        "output_root": out,
        "scene_out": scene_out,
    }
    return summary


def _get_video_meta(video_path: str) -> Tuple[float, int]:
    """返回视频 (fps, frame_count)，不解码帧。"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, fc


# ------------------------------------------------------------------ #
# CLI 入口
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert drone_scene data to LongStream generalizable format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-root",
        default="../drone_scene",
        help="输入目录，包含 drone_scene.avi / odom_data.csv 等。默认 ../drone_scene。",
    )
    parser.add_argument(
        "--out",
        default="prepared_inputs/drone_scene",
        help="输出根目录（meta_root）。默认 prepared_inputs/drone_scene。",
    )
    parser.add_argument(
        "--scene-name",
        default="drone_scene",
        help="场景名。输出到 <out>/<scene-name>/。默认 drone_scene。",
    )
    parser.add_argument(
        "--camera-id",
        default="00",
        help="相机 ID（子目录名）。默认 00。",
    )
    parser.add_argument(
        "--image-ext",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="输出图片格式。默认 png。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最多抽取帧数（均匀采样）。不指定则全部抽取。",
    )
    parser.add_argument(
        "--sync-mode",
        choices=["stretch_to_csv_duration", "video_fps", "wall_clock", "auto_wall_clock"],
        default="stretch_to_csv_duration",
        help=(
            "时间同步策略。\n"
            "  stretch_to_csv_duration（默认）：将视频帧均匀拉伸到 CSV 起止时间段，"
            "忽略 AVI FPS。\n"
            "  video_fps：使用 --video-fps（或 AVI 元数据 FPS）计算帧时间，"
            "以 CSV 起始时间对齐。\n"
            "  wall_clock：使用 --video-start-time/--video-end-time 或 --ocr-time-overlay "
            "确定视频真实起止时间。\n"
            "  auto_wall_clock：自动 OCR 识别时间戳水印；失败时回退到 stretch_to_csv_duration。"
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="覆盖视频 FPS（仅 --sync-mode video_fps 时使用）。",
    )
    parser.add_argument(
        "--time-offset-sec",
        type=float,
        default=0.0,
        help="对计算出的帧时间加偏移（秒）。用于手动对齐视频与 CSV 时间轴。默认 0.0。",
    )
    parser.add_argument(
        "--video-start-time",
        default=None,
        help='视频起始时间，格式 "YYYY-MM-DD HH:MM:SS"（wall_clock 模式）。',
    )
    parser.add_argument(
        "--video-end-time",
        default=None,
        help='视频结束时间，格式 "YYYY-MM-DD HH:MM:SS"（wall_clock 模式）。',
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Shanghai",
        help="时区字符串，用于解析 --video-start/end-time。默认 Asia/Shanghai。",
    )
    parser.add_argument(
        "--ocr-time-overlay",
        action="store_true",
        help="启用 OCR 识别视频时间戳水印（需要 pytesseract）。auto_wall_clock 模式自动开启。",
    )
    parser.add_argument(
        "--ocr-roi",
        default="0,0,420,80",
        help='视频时间戳 OCR 识别区域，格式 "x,y,w,h"。默认 "0,0,420,80"（左上角）。',
    )
    parser.add_argument(
        "--save-ocr-debug",
        action="store_true",
        help="将 OCR ROI 调试图片保存到 <scene_out>/debug_ocr/。",
    )
    parser.add_argument(
        "--gps-source",
        choices=["odom", "global", "none"],
        default="odom",
        help=(
            "GPS/位置来源。\n"
            "  odom（默认）：使用 odom_data.csv 的 position.x/y/z。\n"
            "  global：使用 global_data.csv 的 LLA 转换为近似 ENU。\n"
            "  none：不生成 gps_xyz.npy。"
        ),
    )
    args = parser.parse_args()

    # 路径解析（相对于 cwd）
    src = args.input_root
    out = args.out

    # 解析 --ocr-roi
    ocr_roi_raw = args.ocr_roi.split(",")
    if len(ocr_roi_raw) != 4:
        raise ValueError(f"--ocr-roi 格式错误：{args.ocr_roi!r}，期望 'x,y,w,h'")
    ocr_roi: Tuple[int, int, int, int] = tuple(int(v) for v in ocr_roi_raw)  # type: ignore[assignment]

    log("=" * 60)
    log("[drone_scene] 开始预处理")
    log(f"  input_root  : {os.path.abspath(src)}")
    log(f"  out         : {os.path.abspath(out)}")
    log(f"  scene_name  : {args.scene_name}")
    log(f"  camera_id   : {args.camera_id}")
    log(f"  image_ext   : {args.image_ext}")
    log(f"  sync_mode   : {args.sync_mode}")
    log(f"  video_fps   : {args.video_fps}")
    log(f"  time_offset : {args.time_offset_sec}")
    log(f"  gps_source  : {args.gps_source}")
    log(f"  max_frames  : {args.max_frames}")
    log(f"  overwrite   : {args.overwrite}")
    if args.sync_mode in ("wall_clock", "auto_wall_clock"):
        log(f"  video_start : {args.video_start_time}")
        log(f"  video_end   : {args.video_end_time}")
        log(f"  timezone    : {args.timezone}")
        log(f"  ocr_overlay : {args.ocr_time_overlay}")
        log(f"  ocr_roi     : {args.ocr_roi}")
        log(f"  save_ocr_dbg: {args.save_ocr_debug}")
    log("=" * 60)

    summary = process_drone_scene(
        src=src,
        out=out,
        scene_name=args.scene_name,
        camera_id=args.camera_id,
        image_ext=args.image_ext,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
        sync_mode=args.sync_mode,
        video_fps=args.video_fps,
        time_offset_sec=args.time_offset_sec,
        gps_source=args.gps_source,
        video_start_time=args.video_start_time,
        video_end_time=args.video_end_time,
        timezone=args.timezone,
        ocr_time_overlay=args.ocr_time_overlay,
        ocr_roi=ocr_roi,
        save_ocr_debug=args.save_ocr_debug,
    )

    log("")
    log("=" * 60)
    log("[drone_scene] 预处理完成，摘要：")
    log(f"  frames_written     : {summary['frames_written']}")
    log(f"  image_size         : {summary['image_size']}")
    log(f"  gps_source         : {summary['gps_source']}")
    log(f"  gps_xyz shape      : {summary['gps_xyz_shape']}")
    log(f"  trajectory span    : {summary['trajectory_span_m']:.3f} m（首尾直线距离）")
    log(f"  trajectory total   : {summary['trajectory_total_m']:.3f} m（累积路程）")
    log("  duration_assumptions:")
    da = summary["duration_assumptions"]
    log(f"    AVI 元数据 FPS={da['video_meta_fps']}, "
        f"帧数={da['video_meta_frames']}, "
        f"时长={da['video_meta_duration_sec']}s")
    log(f"    CSV 时长={da['csv_duration_sec']}s")
    log(f"    sync_mode={da['sync_mode']}")
    log(f"    {da['note']}")
    if summary["warnings"]:
        log("  warnings:")
        for w in summary["warnings"]:
            log(f"    - {w}")
    log(f"  output_root        : {summary['output_root']}")
    log(f"  scene_out          : {summary['scene_out']}")
    log("=" * 60)


if __name__ == "__main__":
    main()
