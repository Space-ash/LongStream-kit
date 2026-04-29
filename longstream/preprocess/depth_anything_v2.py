import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from .generalizable import (
    prepare_images_to_generalizable,
    prepare_video_to_generalizable,
)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _natural_sort_key(path: Path):
    stem = path.stem
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
    return parts, path.name.lower()


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def _write_png(path: Path, image: np.ndarray):
    ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    _ensure_dir(path.parent)
    buf.tofile(str(path))


def _write_json(path: Path, payload: Dict):
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _copy_prepared_scene(
    src_root: Path, dst_root: Path, scene_name: str, overwrite: bool = True
):
    src_scene_root = src_root / scene_name
    dst_scene_root = dst_root / scene_name
    if not src_scene_root.exists():
        raise FileNotFoundError(src_scene_root)
    if dst_scene_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"target scene already exists: {dst_scene_root}. "
                "Enable overwrite to replace it."
            )
        shutil.rmtree(dst_scene_root)
    shutil.copytree(src_scene_root, dst_scene_root)
    _ensure_dir(dst_root)
    with open(dst_root / "data_roots.txt", "w", encoding="utf-8") as f:
        f.write(f"{scene_name}\n")


def _resolve_device(device: Optional[str]) -> str:
    requested = (device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def _import_depth_anything_class(repo_path: Path):
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    module = importlib.import_module("depth_anything_v2.dpt")
    return getattr(module, "DepthAnythingV2")


class DepthAnythingV2Runner:
    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        encoder: str = "vits",
        input_size: int = 518,
        device: str = "auto",
        grayscale: bool = True,
        save_raw_npy: bool = True,
        save_visualization: bool = True,
        save_comparison: bool = True,
    ):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.encoder = str(encoder).lower()
        if self.encoder not in MODEL_CONFIGS:
            raise ValueError(
                f"unsupported Depth Anything V2 encoder: {self.encoder}"
            )
        if not self.repo_path.is_dir():
            raise FileNotFoundError(self.repo_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)

        self.input_size = int(input_size)
        self.device = _resolve_device(device)
        self.grayscale = bool(grayscale)
        self.save_raw_npy = bool(save_raw_npy)
        self.save_visualization = bool(save_visualization)
        self.save_comparison = bool(save_comparison)

        DepthAnythingV2 = _import_depth_anything_class(self.repo_path)
        self.model = DepthAnythingV2(**MODEL_CONFIGS[self.encoder])
        state_dict = torch.load(str(self.checkpoint_path), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

    @classmethod
    def from_preprocess_config(cls, preprocess_cfg: Dict):
        depth_cfg = dict(preprocess_cfg.get("depth_anything_v2", {}))
        depth_cfg.pop("enabled", None)
        return cls(**depth_cfg)

    def infer_image(self, image: np.ndarray) -> np.ndarray:
        depth = self.model.infer_image(image, self.input_size)
        return depth.astype(np.float32, copy=False)

    def _depth_to_vis(self, depth: np.ndarray) -> np.ndarray:
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        if depth_max - depth_min < 1e-6:
            vis = np.zeros(depth.shape, dtype=np.uint8)
        else:
            vis = ((depth - depth_min) / (depth_max - depth_min) * 255.0).astype(
                np.uint8
            )
        if self.grayscale:
            return np.repeat(vis[..., np.newaxis], 3, axis=-1)
        return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)

    def _build_comparison(self, image: np.ndarray, vis: np.ndarray) -> np.ndarray:
        split = np.ones((image.shape[0], 24, 3), dtype=np.uint8) * 255
        return np.hstack([image, split, vis])

    def run_on_prepared_scene(
        self,
        prepared_root: str,
        output_root: str,
        scene_name: str,
        camera_id: str = "00",
        overwrite: bool = True,
    ) -> Dict:
        prepared_root_path = Path(prepared_root).expanduser().resolve()
        output_root_path = Path(output_root).expanduser().resolve()
        src_scene_root = prepared_root_path / scene_name
        src_image_dir = src_scene_root / "images" / camera_id
        if not src_image_dir.is_dir():
            raise FileNotFoundError(src_image_dir)

        if prepared_root_path != output_root_path:
            _copy_prepared_scene(
                prepared_root_path,
                output_root_path,
                scene_name=scene_name,
                overwrite=overwrite,
            )
        else:
            _ensure_dir(output_root_path)
            with open(output_root_path / "data_roots.txt", "w", encoding="utf-8") as f:
                f.write(f"{scene_name}\n")

        dst_scene_root = output_root_path / scene_name
        dst_image_dir = dst_scene_root / "images" / camera_id
        depth_dir = dst_scene_root / "depths" / camera_id
        vis_dir = dst_scene_root / "vis_depths" / camera_id
        compare_dir = dst_scene_root / "comparisons" / camera_id
        debug_dir = dst_scene_root / "depth_anything_v2_debug" / camera_id
        for path in [dst_image_dir, depth_dir, vis_dir, compare_dir, debug_dir]:
            _ensure_dir(path)

        image_paths = sorted(
            [
                path
                for path in src_image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ],
            key=_natural_sort_key,
        )
        if not image_paths:
            raise RuntimeError(f"no images found in: {src_image_dir}")

        records: List[Dict] = []
        for image_path in image_paths:
            image = _read_image(image_path)
            depth = self.infer_image(image)
            vis = self._depth_to_vis(depth)
            comparison = self._build_comparison(image, vis)

            stem = image_path.stem
            if self.save_raw_npy:
                np.save(depth_dir / f"{stem}.npy", depth)
            if self.save_visualization:
                _write_png(vis_dir / f"{stem}.png", vis)
            if self.save_comparison:
                _write_png(compare_dir / f"{stem}.png", comparison)

            records.append(
                {
                    "frame_name": image_path.name,
                    "depth_path": str((depth_dir / f"{stem}.npy").resolve())
                    if self.save_raw_npy
                    else None,
                    "vis_path": str((vis_dir / f"{stem}.png").resolve())
                    if self.save_visualization
                    else None,
                    "comparison_path": str((compare_dir / f"{stem}.png").resolve())
                    if self.save_comparison
                    else None,
                    "min_depth": float(depth.min()),
                    "max_depth": float(depth.max()),
                    "mean_depth": float(depth.mean()),
                }
            )

        summary = {
            "prepared_root": str(prepared_root_path),
            "output_root": str(output_root_path),
            "scene_name": scene_name,
            "camera_id": camera_id,
            "repo_path": str(self.repo_path),
            "checkpoint_path": str(self.checkpoint_path),
            "encoder": self.encoder,
            "input_size": self.input_size,
            "device": self.device,
            "num_frames": len(records),
            "save_raw_npy": self.save_raw_npy,
            "save_visualization": self.save_visualization,
            "save_comparison": self.save_comparison,
            "records": records,
        }
        _write_json(debug_dir / "summary.json", summary)
        return summary


def run_depth_anything_v2_pipeline(preprocess_cfg: Dict) -> Dict:
    input_cfg = dict(preprocess_cfg.get("input", {}))
    io_cfg = dict(preprocess_cfg.get("io", {}))
    video_cfg = dict(preprocess_cfg.get("video", {}))
    depth_cfg = dict(preprocess_cfg.get("depth_anything_v2", {}))

    input_mode = str(input_cfg.get("mode", "video")).lower()
    source_path = input_cfg.get("source_path")
    recursive_images = bool(input_cfg.get("recursive_images", False))
    scene_name = str(io_cfg.get("scene_name", "runtime_scene"))
    camera_id = str(io_cfg.get("camera_id", "00"))
    image_ext = str(io_cfg.get("image_ext", "png")).lower()
    overwrite = bool(io_cfg.get("overwrite", True))
    normalized_root = str(io_cfg.get("normalized_root", "prepared_inputs/preprocess/raw"))
    final_root = str(io_cfg.get("final_root", normalized_root))

    if input_mode in {"video", "images"} and not source_path:
        raise ValueError("input.source_path must be set for video/images preprocessing")

    if input_mode == "video":
        prepare_video_to_generalizable(
            video_path=source_path,
            prepared_root=normalized_root,
            scene_name=scene_name,
            camera_id=camera_id,
            target_fps=video_cfg.get("target_fps", 0.0),
            image_ext=image_ext,
            overwrite=overwrite,
        )
    elif input_mode == "images":
        prepare_images_to_generalizable(
            source_path=source_path,
            prepared_root=normalized_root,
            scene_name=scene_name,
            camera_id=camera_id,
            image_ext=image_ext,
            overwrite=overwrite,
            recursive=recursive_images,
        )
    elif input_mode != "prepared":
        raise ValueError(f"unsupported input.mode: {input_mode}")

    if not bool(depth_cfg.get("enabled", False)):
        normalized_root_path = Path(normalized_root).expanduser().resolve()
        final_root_path = Path(final_root).expanduser().resolve()
        if normalized_root_path != final_root_path:
            _copy_prepared_scene(
                normalized_root_path,
                final_root_path,
                scene_name=scene_name,
                overwrite=overwrite,
            )
        summary = {
            "input_mode": input_mode,
            "depth_anything_v2_enabled": False,
            "scene_name": scene_name,
            "camera_id": camera_id,
            "normalized_root": str(normalized_root_path),
            "final_root": str(final_root_path),
            "source_path": source_path,
        }
        _write_json(final_root_path / scene_name / "preprocess_summary.json", summary)
        return summary

    runner = DepthAnythingV2Runner.from_preprocess_config(preprocess_cfg)
    summary = runner.run_on_prepared_scene(
        prepared_root=normalized_root,
        output_root=final_root,
        scene_name=scene_name,
        camera_id=camera_id,
        overwrite=overwrite,
    )
    summary["depth_anything_v2_enabled"] = True
    summary["final_root"] = summary["output_root"]
    return summary
