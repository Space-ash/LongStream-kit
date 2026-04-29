#!/usr/bin/env bash
set -euo pipefail

# Prepare raw vKITTI2 into LongStream generalizable layout.
#
# Usage:
#   RAW_VKITTI2_ROOT=/path/to/vkitti2_merged bash prepare_vkitti2.sh
#   RAW_VKITTI2_ROOT=/path/to/vkitti2_merged CONFIG_PATH=configs/longstream_infer_optimized.yaml bash prepare_vkitti2.sh
#   RAW_VKITTI2_ROOT=/path/to/vkitti2_merged VKITTI2_SUBSCENE=Scene01/clone bash prepare_vkitti2.sh
#
# Important:
# - RAW_VKITTI2_ROOT must be the dataset root that directly contains Scene01, Scene02, ...
# - Do not pass a scene directory such as .../Scene01
# - Do not pass a subscene directory such as .../Scene01/clone

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/longstream_infer_optimized.yaml}"
RAW_VKITTI2_ROOT="${RAW_VKITTI2_ROOT:-../VKITTI/vkitti2_merged}"
PREPARED_ROOT="${PREPARED_ROOT:-}"
VKITTI2_SUBSCENE="${VKITTI2_SUBSCENE:-}"
NUM_WORKERS="${NUM_WORKERS:-16}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[prepare-vkitti2] error: config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ -z "$RAW_VKITTI2_ROOT" ]]; then
  echo "[prepare-vkitti2] error: please set RAW_VKITTI2_ROOT=/path/to/vkitti2_merged" >&2
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

print("" if value is None else value)
PY
}

PREPARED_ROOT="${PREPARED_ROOT:-$(yaml_get data.img_path)}"
VKITTI2_SUBSCENE="${VKITTI2_SUBSCENE:-$(yaml_get data.seq_list.0)}"

PREPARED_ROOT="${PREPARED_ROOT:-prepared_inputs/vkitti2}"
VKITTI2_SUBSCENE="${VKITTI2_SUBSCENE:-Scene01/clone}"

echo "[prepare-vkitti2] root_dir=$ROOT_DIR"
echo "[prepare-vkitti2] config_path=$CONFIG_PATH"
echo "[prepare-vkitti2] raw_vkitti2_root=$RAW_VKITTI2_ROOT"
echo "[prepare-vkitti2] prepared_root=$PREPARED_ROOT"
echo "[prepare-vkitti2] target_subscene=$VKITTI2_SUBSCENE"

if [[ ! -d "$RAW_VKITTI2_ROOT" ]]; then
  echo "[prepare-vkitti2] error: raw vKITTI2 root not found: $RAW_VKITTI2_ROOT" >&2
  exit 1
fi

SCENE_ROOT="$(dirname "$VKITTI2_SUBSCENE")"
if [[ ! -d "$RAW_VKITTI2_ROOT/$SCENE_ROOT" ]]; then
  echo "[prepare-vkitti2] error: RAW_VKITTI2_ROOT must be the dataset root containing scene directories." >&2
  echo "[prepare-vkitti2] expected directory: $RAW_VKITTI2_ROOT/$SCENE_ROOT" >&2
  echo "[prepare-vkitti2] got RAW_VKITTI2_ROOT=$RAW_VKITTI2_ROOT" >&2
  echo "[prepare-vkitti2] got VKITTI2_SUBSCENE=$VKITTI2_SUBSCENE" >&2
  echo "[prepare-vkitti2] example:" >&2
  echo "  RAW_VKITTI2_ROOT=/path/to/vkitti2_merged" >&2
  exit 1
fi

echo "[prepare-vkitti2] step 1/2: converting raw vKITTI2"
"$PYTHON_BIN" scripts/vkitti2_to_generalizable.py \
  --data_root "$RAW_VKITTI2_ROOT" \
  --easyvolcap_root "$PREPARED_ROOT" \
  --num_workers "$NUM_WORKERS" \
  --scenes "$SCENE_ROOT"

echo "[prepare-vkitti2] step 2/2: refreshing data_roots.txt"
mkdir -p "$PREPARED_ROOT"
find "$PREPARED_ROOT" -mindepth 2 -maxdepth 2 -type d | \
  sed "s#^$PREPARED_ROOT/##" | \
  sort > "$PREPARED_ROOT/data_roots.txt"

if [[ ! -d "$PREPARED_ROOT/$VKITTI2_SUBSCENE" ]]; then
  echo "[prepare-vkitti2] error: prepared subscene not found: $PREPARED_ROOT/$VKITTI2_SUBSCENE" >&2
  exit 1
fi

echo "[prepare-vkitti2] done"
echo "[prepare-vkitti2] prepared dataset root: $PREPARED_ROOT"
