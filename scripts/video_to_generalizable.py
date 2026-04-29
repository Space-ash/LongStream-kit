import argparse

from longstream.preprocess import (
    default_preprocess_config_path,
    load_preprocess_config,
    prepare_video_to_generalizable,
)
from longstream.preprocess.config import deep_update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_preprocess_config_path())
    parser.add_argument("--video", default=None, help="Source video path.")
    parser.add_argument(
        "--out",
        default=None,
        help="Prepared meta_root in LongStream generalizable layout.",
    )
    parser.add_argument(
        "--scene-name", default=None, help="Prepared scene name."
    )
    parser.add_argument("--camera-id", default=None, help="Prepared camera id.")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Target sampling fps. <= 0 keeps all frames.",
    )
    parser.add_argument(
        "--image-ext",
        choices=["png", "jpg", "jpeg"],
        default=None,
        help="Prepared frame format.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the prepared scene if it already exists.",
    )
    args = parser.parse_args()

    cfg = load_preprocess_config(args.config)
    overrides = {"input": {}, "io": {}, "video": {}}
    if args.video is not None:
        overrides["input"]["source_path"] = args.video
    overrides["input"]["mode"] = "video"
    if args.out is not None:
        overrides["io"]["normalized_root"] = args.out
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
    cfg = deep_update(cfg, overrides)

    source_path = cfg.get("input", {}).get("source_path")
    io_cfg = cfg.get("io", {})
    video_cfg = cfg.get("video", {})
    result = prepare_video_to_generalizable(
        video_path=source_path,
        prepared_root=io_cfg.get("normalized_root"),
        scene_name=io_cfg.get("scene_name", "runtime_scene"),
        camera_id=io_cfg.get("camera_id", "00"),
        target_fps=video_cfg.get("target_fps", 0.0),
        image_ext=io_cfg.get("image_ext", "png"),
        overwrite=bool(io_cfg.get("overwrite", True)),
    )
    print(
        "[longstream] prepared video "
        f"scene={result['scene_name']} frames={result['num_frames']} "
        f"image_dir={result['image_dir']}"
    )


if __name__ == "__main__":
    main()
