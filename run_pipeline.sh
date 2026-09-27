#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash run_pipeline.sh MESH.ply [MODEL.pth]" >&2
  exit 1
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MESH_PATH="$(realpath -m -- "$1")"
MODEL_PATH="$(realpath -m -- "${2:-$REPO_DIR/checkpoints/best_model.pth}")"
[[ -f "$MESH_PATH" ]] || { echo "Missing mesh: $MESH_PATH" >&2; exit 1; }
[[ -f "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
MESH_NAME="$(basename -- "$MESH_PATH")"
MESH_NAME="${MESH_NAME%.*}"
cd "$REPO_DIR/scripts"

python generate_object.py --mesh_path "$MESH_PATH" \
  --output_root ../outputs/objects --points 350 --size 0.2
[[ -f "../outputs/objects/$MESH_NAME/alignment/contact_frames.npy" ]]
python predict_geometry.py --data_root ../outputs/objects \
  --mesh_name "$MESH_NAME" --model_path "$MODEL_PATH"
[[ -f "../outputs/objects/$MESH_NAME/reconstruction/000000_pred_depth.npy" ]]
python export_touchanything.py --data_root ../outputs/objects \
  --mesh_name "$MESH_NAME" --output_dir ../outputs/touchanything
[[ -f "../outputs/touchanything/$MESH_NAME/meta_data.json" ]]
echo "Exported: $REPO_DIR/outputs/touchanything/$MESH_NAME"
