#!/usr/bin/env bash
set -euo pipefail

# Prepare KITTI Odometry sequence(s) into LongStream generalizable layout.
#
# Usage:
#   bash prepare_kitti_odometry.sh
#   KITTI_ROOT=KITTI_Odometry/dataset SEQS="08" bash prepare_kitti_odometry.sh
#   KITTI_ROOT=KITTI_Odometry/dataset SEQS="00 01 08" bash prepare_kitti_odometry.sh
#
# Environment variables (all optional, sensible defaults are provided):
#   KITTI_ROOT    Path to KITTI Odometry dataset root (contains sequences/ and poses/).
#                 Default: KITTI_Odometry/dataset
#   SEQS          Space-separated sequence IDs to process.
#                 Default: 08
#   CAMERA        Camera channel: 02 = left colour, 03 = right colour.
#                 Default: 02
#   OUT_ROOT      Output generalizable meta_root.
#                 Default: prepared_inputs/kitti_odometry
#   CONFIG_PATH   YAML config to run inference after preprocessing (optional).
#                 Default: configs/longstream_infer_kitti08_optimized.yaml
#   PYTHON_BIN    Python executable.
#                 Default: python
#   COPY          Set to "1" to copy image files instead of symlinking.
#                 Default: 0 (symlink)
#
# After preprocessing, run inference with:
#   python run.py --config "$CONFIG_PATH"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
KITTI_ROOT="${KITTI_ROOT:-KITTI_Odometry/dataset}"
SEQS="${SEQS:-08}"
# 规范化 CAMERA：无论用户传 "2" 还是 "02" 都统一成两位零填充
CAMERA_RAW="${CAMERA:-02}"
CAMERA_NUM=$((10#$CAMERA_RAW))
CAMERA=$(printf "%02d" "$CAMERA_NUM")
OUT_ROOT="${OUT_ROOT:-prepared_inputs/kitti_odometry}"
CONFIG_PATH="${CONFIG_PATH:-configs/longstream_infer_kitti08_optimized.yaml}"
COPY_FLAG="${COPY:-0}"

echo "[prepare-kitti-odometry] root_dir=$ROOT_DIR"
echo "[prepare-kitti-odometry] kitti_root=$KITTI_ROOT"
echo "[prepare-kitti-odometry] seqs=$SEQS"
echo "[prepare-kitti-odometry] camera=$CAMERA"
echo "[prepare-kitti-odometry] out_root=$OUT_ROOT"
echo "[prepare-kitti-odometry] config=$CONFIG_PATH"

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
if [[ ! -d "$KITTI_ROOT" ]]; then
  echo "[prepare-kitti-odometry] error: KITTI_ROOT not found: $KITTI_ROOT" >&2
  echo "  Set KITTI_ROOT to the directory that contains sequences/ and poses/." >&2
  exit 1
fi

if [[ ! -d "$KITTI_ROOT/sequences" ]]; then
  echo "[prepare-kitti-odometry] error: $KITTI_ROOT/sequences not found." >&2
  echo "  Expected layout: $KITTI_ROOT/sequences/<seq>/image_2/ + $KITTI_ROOT/poses/<seq>.txt" >&2
  exit 1
fi

if [[ ! -d "$KITTI_ROOT/poses" ]]; then
  echo "[prepare-kitti-odometry] error: $KITTI_ROOT/poses not found." >&2
  exit 1
fi

# ------------------------------------------------------------------
# Step 1: Convert KITTI -> generalizable
# ------------------------------------------------------------------
echo "[prepare-kitti-odometry] step 1/2: converting KITTI Odometry to generalizable format"

COPY_ARG=""
if [[ "$COPY_FLAG" == "1" ]]; then
  COPY_ARG="--copy"
fi

# shellcheck disable=SC2086
"$PYTHON_BIN" scripts/kitti_odometry_to_generalizable.py \
  --kitti_root "$KITTI_ROOT" \
  --out_root   "$OUT_ROOT" \
  --seqs       $SEQS \
  --camera     "$CAMERA" \
  $COPY_ARG

# ------------------------------------------------------------------
# Step 2: Validate output
# ------------------------------------------------------------------
echo "[prepare-kitti-odometry] step 2/2: validating output"

ALL_OK=1
for SEQ in $SEQS; do
  SCENE_DIR="$OUT_ROOT/$SEQ"
  ISSUES=""

  [[ -d "$SCENE_DIR/images/$CAMERA" ]] || ISSUES="$ISSUES images/$CAMERA/"
  [[ -f "$SCENE_DIR/cameras/$CAMERA/extri.yml" ]] || ISSUES="$ISSUES cameras/$CAMERA/extri.yml"
  [[ -f "$SCENE_DIR/cameras/$CAMERA/intri.yml"  ]] || ISSUES="$ISSUES cameras/$CAMERA/intri.yml"
  [[ -f "$SCENE_DIR/gt_poses.npy"               ]] || ISSUES="$ISSUES gt_poses.npy"
  [[ -f "$SCENE_DIR/gt_poses_${CAMERA}.npy"     ]] || ISSUES="$ISSUES gt_poses_${CAMERA}.npy"

  if [[ -n "$ISSUES" ]]; then
    echo "[prepare-kitti-odometry] WARNING: seq $SEQ missing:$ISSUES" >&2
    ALL_OK=0
  else
    echo "[prepare-kitti-odometry] seq $SEQ: OK"
  fi
done

[[ "$ALL_OK" == "1" ]] || { echo "[prepare-kitti-odometry] some sequences have issues (see above)." >&2; exit 1; }

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "[prepare-kitti-odometry] done."
echo "[prepare-kitti-odometry] prepared dataset root : $OUT_ROOT"
echo "[prepare-kitti-odometry] data_roots.txt        : $OUT_ROOT/data_roots.txt"
echo ""
echo "  To run inference:"
echo "    python run.py --config $CONFIG_PATH"
echo ""
echo "  For a quick smoke test (first 100 frames), set in the YAML:"
echo "    data.max_frames: 100"
