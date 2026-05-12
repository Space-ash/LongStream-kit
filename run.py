import argparse
import os
import shutil

from longstream.core.cli import (
    add_runtime_arguments,
    load_config_with_overrides,
    parse_runtime_args,
)
from longstream.core.infer import run_inference_cfg
from longstream.eval import evaluate_predictions_cfg
from longstream.utils.resource_monitor import from_cfg as _monitor_from_cfg, CriticalOperationProfiler


def _reset_output_root(cfg: dict):
    output_root = cfg.get("output", {}).get("root", "outputs")
    output_root = os.path.abspath(os.path.expanduser(output_root))
    data_img_path = cfg.get("data", {}).get("img_path")
    if data_img_path:
        data_img_path = os.path.abspath(os.path.expanduser(data_img_path))
        if output_root == data_img_path or data_img_path.startswith(
            output_root + os.sep
        ):
            raise ValueError(
                "refusing to clear output root because it overlaps with "
                f"data.img_path: {output_root}"
            )
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)
    elif os.path.exists(output_root):
        os.remove(output_root)


def main():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--skip-eval", action="store_true")
    args = parse_runtime_args(parser)

    cfg = load_config_with_overrides(args)
    _reset_output_root(cfg)

    output_root = os.path.abspath(
        os.path.expanduser(cfg.get("output", {}).get("root", "outputs"))
    )
    monitor = _monitor_from_cfg(cfg.get("monitoring", {}))
    if monitor is not None:
        monitor.start(output_root)
    CriticalOperationProfiler.initialize(output_root)

    try:
        print("[longstream] run: inference", flush=True)
        with CriticalOperationProfiler("run_inference_cfg"):
            run_inference_cfg(cfg)
        if not args.skip_eval:
            print("[longstream] run: evaluation", flush=True)
            with CriticalOperationProfiler("evaluate_predictions_cfg"):
                evaluate_predictions_cfg(cfg)
            print("[longstream] run: done", flush=True)
    finally:
        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    main()
