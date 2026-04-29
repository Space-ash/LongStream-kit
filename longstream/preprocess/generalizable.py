import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _natural_sort_key(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = []
    token = ""
    is_digit = None
    for ch in stem:
        curr_digit = ch.isdigit()
        if is_digit is None or curr_digit == is_digit:
            token += ch
        else:
            parts.append((0, int(token)) if is_digit else (1, token.lower()))
            token = ch
        is_digit = curr_digit
    if token:
        parts.append((0, int(token)) if is_digit else (1, token.lower()))
    return parts, os.path.basename(path).lower()


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _write_list(path: Path, items: Iterable[str]):
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def _write_manifest(scene_root: Path, payload: Dict):
    with open(scene_root / "input_manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _reset_scene(scene_root: Path, overwrite: bool):
    if not scene_root.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"target scene already exists: {scene_root}. "
            "Enable overwrite to replace it."
        )
    shutil.rmtree(scene_root)


def _collect_image_paths(src: Path, recursive: bool) -> List[str]:
    if src.is_file():
        if src.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            raise ValueError(f"unsupported image file: {src}")
        return [str(src)]

    if not src.is_dir():
        raise FileNotFoundError(src)

    if recursive:
        image_paths = [
            str(path)
            for path in src.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
    else:
        image_paths = [
            str(path)
            for path in src.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
    image_paths.sort(key=_natural_sort_key)
    return image_paths


def _save_frame(frame, dst: Path, image_ext: str):
    if image_ext == "png":
        ok, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif image_ext in {"jpg", "jpeg"}:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        raise ValueError(f"unsupported output extension: {image_ext}")
    if not ok:
        raise RuntimeError(f"failed to write frame: {dst}")
    buf.tofile(str(dst))


def _read_image(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _scene_paths(prepared_root: Path, scene_name: str, camera_id: str):
    scene_root = prepared_root / scene_name
    image_dir = scene_root / "images" / camera_id
    return scene_root, image_dir


def _finalize_prepared_root(prepared_root: Path, scene_name: str):
    _write_list(prepared_root / "data_roots.txt", [scene_name])


def prepare_video_to_generalizable(
    video_path: str,
    prepared_root: str,
    scene_name: str,
    camera_id: str = "00",
    target_fps: Optional[float] = None,
    image_ext: str = "png",
    overwrite: bool = True,
) -> Dict:
    src = Path(video_path).expanduser().resolve()
    prepared_root_path = Path(prepared_root).expanduser().resolve()
    scene_root, image_dir = _scene_paths(prepared_root_path, scene_name, camera_id)

    if not src.is_file():
        raise FileNotFoundError(src)

    _reset_scene(scene_root, overwrite=overwrite)
    _ensure_dir(image_dir)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {src}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 30.0

    keep_all = target_fps is None or float(target_fps) <= 0.0
    sample_period = None if keep_all else 1.0 / float(target_fps)
    next_sample_time = 0.0
    total_frames = 0
    saved_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_time = total_frames / source_fps
        if keep_all or current_time + 1e-9 >= next_sample_time:
            dst = image_dir / f"{saved_frames:06d}.{image_ext}"
            _save_frame(frame, dst, image_ext=image_ext)
            saved_frames += 1
            if sample_period is not None:
                next_sample_time += sample_period
        total_frames += 1

    cap.release()

    if saved_frames == 0:
        raise RuntimeError(f"no frames extracted from video: {src}")

    _finalize_prepared_root(prepared_root_path, scene_name)
    _write_manifest(
        scene_root,
        {
            "mode": "video",
            "source_path": str(src),
            "prepared_root": str(prepared_root_path),
            "scene_name": scene_name,
            "camera_id": camera_id,
            "source_fps": source_fps,
            "target_fps": None if keep_all else float(target_fps),
            "total_frames_read": total_frames,
            "frames_written": saved_frames,
            "image_ext": image_ext,
        },
    )
    return {
        "prepared_root": str(prepared_root_path),
        "scene_name": scene_name,
        "camera_id": camera_id,
        "image_dir": str(image_dir),
        "num_frames": saved_frames,
        "source_fps": source_fps,
    }


def prepare_images_to_generalizable(
    source_path: str,
    prepared_root: str,
    scene_name: str,
    camera_id: str = "00",
    image_ext: str = "png",
    overwrite: bool = True,
    recursive: bool = False,
) -> Dict:
    src = Path(source_path).expanduser().resolve()
    prepared_root_path = Path(prepared_root).expanduser().resolve()
    scene_root, image_dir = _scene_paths(prepared_root_path, scene_name, camera_id)

    image_paths = _collect_image_paths(src, recursive=recursive)
    if not image_paths:
        raise RuntimeError(f"no images found under: {src}")

    _reset_scene(scene_root, overwrite=overwrite)
    _ensure_dir(image_dir)

    for frame_id, image_path in enumerate(image_paths):
        frame = _read_image(image_path)
        if frame is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        dst = image_dir / f"{frame_id:06d}.{image_ext}"
        _save_frame(frame, dst, image_ext=image_ext)

    _finalize_prepared_root(prepared_root_path, scene_name)
    _write_manifest(
        scene_root,
        {
            "mode": "images",
            "source_path": str(src),
            "prepared_root": str(prepared_root_path),
            "scene_name": scene_name,
            "camera_id": camera_id,
            "recursive": bool(recursive),
            "frames_written": len(image_paths),
            "image_ext": image_ext,
        },
    )
    return {
        "prepared_root": str(prepared_root_path),
        "scene_name": scene_name,
        "camera_id": camera_id,
        "image_dir": str(image_dir),
        "num_frames": len(image_paths),
    }

