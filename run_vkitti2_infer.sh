#!/usr/bin/env bash
set -euo pipefail

# Run LongStream inference + evaluation using YAML config.
#
# Usage:
#   bash run_vkitti2_infer.sh
#   CONFIG_PATH=configs/longstream_infer_optimized.yaml bash run_vkitti2_infer.sh
#   CONFIG_PATH=configs/longstream_infer_optimized.yaml OUTPUT_ROOT=outputs/tmp bash run_vkitti2_infer.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/longstream_infer_optimized.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
MAX_FRAMES="${MAX_FRAMES:-}"
NO_MASK_SKY="${NO_MASK_SKY:-}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[run-vkitti2-infer] error: config not found: $CONFIG_PATH" >&2
  exit 1
fi

yaml_get() {
  local query="$1"
  "$PYTHON_BIN" - "$CONFIG_PATH" "$query" <<'PY'
import sys
import yaml

config_path, query = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

value = cfg
for part in query.split("."):
    if isinstance(value, list):
        try:
            value = value[int(part)]
        except Exception:
            value = None
            break
    elif isinstance(value, dict):
        value = value.get(part)
    else:
        value = None
        break

if value is None:
    print("")
elif isinstance(value, bool):
    print("1" if value else "0")
else:
    print(value)
PY
}

CFG_IMG_PATH="$(yaml_get data.img_path)"
CFG_SEQ="$(yaml_get data.seq_list.0)"
CFG_CAMERA="$(yaml_get data.camera)"
CFG_OUTPUT="$(yaml_get output.root)"
CFG_CHECKPOINT="$(yaml_get model.checkpoint)"
CFG_MASK_SKY="$(yaml_get output.mask_sky)"

echo "[run-vkitti2-infer] root_dir=$ROOT_DIR"
echo "[run-vkitti2-infer] config_path=$CONFIG_PATH"
echo "[run-vkitti2-infer] yaml_img_path=${CFG_IMG_PATH:-<empty>}"
echo "[run-vkitti2-infer] yaml_seq=${CFG_SEQ:-<empty>}"
echo "[run-vkitti2-infer] yaml_camera=${CFG_CAMERA:-<empty>}"
echo "[run-vkitti2-infer] yaml_output_root=${CFG_OUTPUT:-<empty>}"

if [[ -n "$CFG_IMG_PATH" && ! -d "$CFG_IMG_PATH" ]]; then
  echo "[run-vkitti2-infer] warning: yaml data.img_path does not exist yet: $CFG_IMG_PATH" >&2
  echo "[run-vkitti2-infer] tip: run 'bash prepare_vkitti2.sh' first if you have not converted vKITTI2." >&2
fi

if [[ -n "$CFG_CHECKPOINT" && ! -f "$CFG_CHECKPOINT" ]]; then
  echo "[run-vkitti2-infer] warning: yaml checkpoint not found: $CFG_CHECKPOINT" >&2
fi

CMD=(
  "$PYTHON_BIN" run.py
  --config "$CONFIG_PATH"
)

if [[ -n "$OUTPUT_ROOT" ]]; then
  CMD+=(--output-root "$OUTPUT_ROOT")
fi

if [[ -n "$MAX_FRAMES" ]]; then
  CMD+=(--max-frames "$MAX_FRAMES")
fi

if [[ -n "$NO_MASK_SKY" ]]; then
  if [[ "$NO_MASK_SKY" == "1" ]]; then
    CMD+=(--no-mask-sky)
  fi
elif [[ "$CFG_MASK_SKY" == "0" ]]; then
  CMD+=(--no-mask-sky)
fi

printf '[run-vkitti2-infer] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "[run-vkitti2-infer] done"

