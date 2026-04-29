import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longstream.preprocess import default_preprocess_config_path, load_preprocess_config
from longstream.preprocess.config import deep_update
from longstream.preprocess.depth_anything_v2 import run_depth_anything_v2_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_preprocess_config_path())
    parser.add_argument(
        "--input-mode", choices=["video", "images", "prepared"], default=None
    )
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--normalized-root", default=None)
    parser.add_argument("--final-root", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--image-ext", choices=["png", "jpg", "jpeg"], default=None)
    parser.add_argument("--recursive-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--repo-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--disable-depth", action="store_true")
    args = parser.parse_args()

    cfg = load_preprocess_config(args.config)
    overrides = {
        "input": {},
        "io": {},
        "video": {},
        "depth_anything_v2": {},
    }
    if args.input_mode is not None:
        overrides["input"]["mode"] = args.input_mode
    if args.source_path is not None:
        overrides["input"]["source_path"] = args.source_path
    if args.recursive_images:
        overrides["input"]["recursive_images"] = True
    if args.normalized_root is not None:
        overrides["io"]["normalized_root"] = args.normalized_root
    if args.final_root is not None:
        overrides["io"]["final_root"] = args.final_root
    if args.scene_name is not None:
        overrides["io"]["scene_name"] = args.scene_name
    if args.camera_id is not None:
        overrides["io"]["camera_id"] = args.camera_id
    if args.image_ext is not None:
        overrides["io"]["image_ext"] = args.image_ext
    if args.overwrite:
        overrides["io"]["overwrite"] = True
    if args.target_fps is not None:
        overrides["video"]["target_fps"] = args.target_fps

    if args.repo_path is not None:
        overrides["depth_anything_v2"]["repo_path"] = args.repo_path
    if args.checkpoint_path is not None:
        overrides["depth_anything_v2"]["checkpoint_path"] = args.checkpoint_path
    if args.encoder is not None:
        overrides["depth_anything_v2"]["encoder"] = args.encoder
    if args.input_size is not None:
        overrides["depth_anything_v2"]["input_size"] = args.input_size
    if args.device is not None:
        overrides["depth_anything_v2"]["device"] = args.device
    if args.disable_depth:
        overrides["depth_anything_v2"]["enabled"] = False

    cfg = deep_update(cfg, overrides)
    summary = run_depth_anything_v2_pipeline(cfg)

    print("[depth-anything-v2] finished")
    print(f"  scene_name : {summary['scene_name']}")
    if "num_frames" in summary:
        print(f"  num_frames : {summary['num_frames']}")
    print(
        f"  final_root : {summary.get('final_root', summary.get('output_root'))}"
    )
    print(
        f"  enabled    : {summary.get('depth_anything_v2_enabled', False)}"
    )


if __name__ == "__main__":
    main()
