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
#                 Default: configs/longstream_infer_kitti_optimized.yaml
#   PYTHON_BIN    Python executable.
#                 Default: python
#   COPY          Set to "1" to copy image files instead of symlinking.
#                 Default: 0 (symlink)
#   MAX_FRAMES    Uniformly sample at most this many frames from each sequence.
#                 When set, appends _<MAX_FRAMES> suffix to OUT_ROOT.
#                 Default: (empty, process all frames)
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
MAX_FRAMES="${MAX_FRAMES:-}"
# 若指定了抽帧数，在输出目录添加 _<max_frames> 后缀
if [[ -n "$MAX_FRAMES" ]]; then
  OUT_ROOT="${OUT_ROOT}_${MAX_FRAMES}"
fi
CONFIG_PATH="${CONFIG_PATH:-configs/longstream_infer_kitti_optimized.yaml}"
COPY_FLAG="${COPY:-0}"

# LiDAR 深度相关
DEPTH_MODE="${DEPTH_MODE:-sparse_lidar}"
MIN_DEPTH="${MIN_DEPTH:-0.1}"
MAX_DEPTH="${MAX_DEPTH:-80.0}"
SAVE_DEPTH_MASK="${SAVE_DEPTH_MASK:-0}"
OVERWRITE_DEPTHS="${OVERWRITE_DEPTHS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ALLOW_MISSING_LIDAR_CALIB="${ALLOW_MISSING_LIDAR_CALIB:-0}"

echo "[prepare-kitti-odometry] root_dir=$ROOT_DIR"
echo "[prepare-kitti-odometry] kitti_root=$KITTI_ROOT"
echo "[prepare-kitti-odometry] seqs=$SEQS"
echo "[prepare-kitti-odometry] camera=$CAMERA"
echo "[prepare-kitti-odometry] out_root=$OUT_ROOT"
echo "[prepare-kitti-odometry] config=$CONFIG_PATH"
echo "[prepare-kitti-odometry] depth_mode=$DEPTH_MODE  min_depth=$MIN_DEPTH  max_depth=$MAX_DEPTH  num_workers=$NUM_WORKERS"
[[ -n "$MAX_FRAMES" ]] && echo "[prepare-kitti-odometry] max_frames=$MAX_FRAMES  (output suffix: _${MAX_FRAMES})"

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
# 深度模式预检查：若 DEPTH_MODE=sparse_lidar，提前检查 velodyne/ 和 LiDAR 外参覆盖
# ------------------------------------------------------------------
if [[ "$DEPTH_MODE" == "sparse_lidar" ]]; then
  for SEQ in $SEQS; do
    SEQ_DIR="$KITTI_ROOT/sequences/$SEQ"
    VELO_DIR="$SEQ_DIR/velodyne"
    if [[ ! -d "$VELO_DIR" ]]; then
      echo "[prepare-kitti-odometry] error: DEPTH_MODE=sparse_lidar 但 velodyne 目录不存在: $VELO_DIR" >&2
      echo "  若该序列无 LiDAR 数据，请显式设置: DEPTH_MODE=none bash prepare_kitti_odometry.sh" >&2
      exit 1
    fi
    # 检查 LiDAR 外参覆盖：calib.txt 有 Tr/Tr_velo_to_cam 或同目录有 calib_velo_to_cam.txt
    CALIB_TXT="$SEQ_DIR/calib.txt"
    VELO_CALIB="$SEQ_DIR/calib_velo_to_cam.txt"
    HAS_TR=$(grep -E "^(Tr|Tr_velo_to_cam):" "$CALIB_TXT" 2>/dev/null | head -1)
    if [[ -z "$HAS_TR" && ! -f "$VELO_CALIB" ]]; then
      echo "[prepare-kitti-odometry] error: seq $SEQ 缺少 LiDAR 外参标定文件。" >&2
      echo "  calib.txt only contains P0-P3." >&2
      echo "  Need calib_velo_to_cam.txt with R/T, or Tr/Tr_velo_to_cam in calib.txt." >&2
      echo "  Expected files are in the same directory as calib.txt: $SEQ_DIR" >&2
      echo "  Or set DEPTH_MODE=none to skip LiDAR depth." >&2
      exit 1
    fi
    # 建议（不强制）存在 calib_cam_to_cam.txt，用于 R_rect_00
    CAM_CALIB="$SEQ_DIR/calib_cam_to_cam.txt"
    if [[ ! -f "$CAM_CALIB" ]]; then
      echo "[prepare-kitti-odometry] WARN: seq $SEQ 缺少 $CAM_CALIB，R_rect_00 将用单位阵 I 代替。" >&2
    fi
  done
fi

# ------------------------------------------------------------------
# Step 1: Convert KITTI -> generalizable
# ------------------------------------------------------------------
echo "[prepare-kitti-odometry] step 1/3: converting KITTI Odometry to generalizable format"

COPY_ARG=""
if [[ "$COPY_FLAG" == "1" ]]; then
  COPY_ARG="--copy"
fi

MAX_FRAMES_ARG=""
if [[ -n "$MAX_FRAMES" ]]; then
  MAX_FRAMES_ARG="--max_frames $MAX_FRAMES"
fi

DEPTH_EXTRA_ARGS=""
if [[ "$SAVE_DEPTH_MASK" == "1" ]]; then
  DEPTH_EXTRA_ARGS="$DEPTH_EXTRA_ARGS --save_depth_mask"
fi
if [[ "$OVERWRITE_DEPTHS" == "1" ]]; then
  DEPTH_EXTRA_ARGS="$DEPTH_EXTRA_ARGS --overwrite_depths"
fi
if [[ "$ALLOW_MISSING_LIDAR_CALIB" == "1" ]]; then
  DEPTH_EXTRA_ARGS="$DEPTH_EXTRA_ARGS --allow_missing_lidar_calib"
fi

# shellcheck disable=SC2086
"$PYTHON_BIN" scripts/kitti_odometry_to_generalizable.py \
  --kitti_root "$KITTI_ROOT" \
  --out_root   "$OUT_ROOT" \
  --seqs       $SEQS \
  --camera     "$CAMERA" \
  --depth_mode "$DEPTH_MODE" \
  --min_depth  "$MIN_DEPTH" \
  --max_depth  "$MAX_DEPTH" \
  --num_workers "$NUM_WORKERS" \
  $COPY_ARG \
  $MAX_FRAMES_ARG \
  $DEPTH_EXTRA_ARGS

# ------------------------------------------------------------------
# Step 2: Validate output
# ------------------------------------------------------------------
echo "[prepare-kitti-odometry] step 2/3: validating output"

ALL_OK=1
for SEQ in $SEQS; do
  SCENE_DIR="$OUT_ROOT/$SEQ"
  ISSUES=""

  [[ -d "$SCENE_DIR/images/$CAMERA" ]] || ISSUES="$ISSUES images/$CAMERA/"
  [[ -f "$SCENE_DIR/cameras/$CAMERA/extri.yml" ]] || ISSUES="$ISSUES cameras/$CAMERA/extri.yml"
  [[ -f "$SCENE_DIR/cameras/$CAMERA/intri.yml"  ]] || ISSUES="$ISSUES cameras/$CAMERA/intri.yml"
  [[ -f "$SCENE_DIR/gt_poses.npy"               ]] || ISSUES="$ISSUES gt_poses.npy"
  [[ -f "$SCENE_DIR/gt_poses_${CAMERA}.npy"     ]] || ISSUES="$ISSUES gt_poses_${CAMERA}.npy"

  # sparse_lidar 深度输出验证
  if [[ "$DEPTH_MODE" == "sparse_lidar" ]]; then
    [[ -d "$SCENE_DIR/depths/$CAMERA" ]]    || ISSUES="$ISSUES depths/$CAMERA/"
    [[ -f "$SCENE_DIR/depth_meta.json" ]]   || ISSUES="$ISSUES depth_meta.json"
  fi

  if [[ -n "$ISSUES" ]]; then
    echo "[prepare-kitti-odometry] WARNING: seq $SEQ missing:$ISSUES" >&2
    ALL_OK=0
  else
    echo "[prepare-kitti-odometry] seq $SEQ: OK"
  fi
done

[[ "$ALL_OK" == "1" ]] || { echo "[prepare-kitti-odometry] some sequences have issues (see above)." >&2; exit 1; }

# ------------------------------------------------------------------
# Step 3: Refresh data_roots.txt by scanning out_root
# ------------------------------------------------------------------
echo "[prepare-kitti-odometry] step 3/3: refreshing data_roots.txt"
find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d \
  | while read -r scene_dir; do
      scene_name="$(basename "$scene_dir")"
      if [[ -d "$scene_dir/images/$CAMERA" && -d "$scene_dir/cameras/$CAMERA" ]]; then
        echo "$scene_name"
      fi
    done \
  | sort > "$OUT_ROOT/data_roots.txt"
echo "[prepare-kitti-odometry] data_roots.txt: $(cat "$OUT_ROOT/data_roots.txt" | tr '\n' ' ')"

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
