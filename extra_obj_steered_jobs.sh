#!/usr/bin/env bash
# Objectivity-seeking probe: extra steer alphas only (-1, -0.5, 0.5).
# One Python process: loads 70B once, loops alphas × 4 settings.
# Outputs: test_results/objectivity_seeking_steered/alpha_<value>/<setting>.csv

set -euo pipefail

export HF_HOME="${HF_HOME:-/nlp/scr/zope/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=python
OUT_DIR="test_results/objectivity_seeking_steered"
mkdir -p "$OUT_DIR"

MODEL="meta-llama/Llama-3.3-70B-Instruct"
TASK="4dims"
PROBE_DIR="70b_steering_vectors/probe_out_objectivity_seeking_0225_llama70b_FULL_llama70b"

"$PYTHON" parallel_get_multiturn_steered.py --batch \
  --output-dir "$OUT_DIR" \
  --steer-alphas "-1,-0.5,0,0.5,1" \
  --setting "data/valpairs-modified-val-15.csv,valpairs_val_m1_sw5,1,5" \
  --setting "data/valpairs-modified-val-15.csv,valpairs_val_m1_sw11,1,11" \
  --setting "data/valpairs-modified-obj-15.csv,valpairs_obj_m2_sw5,2,5" \
  --setting "data/valpairs-modified-obj-15.csv,valpairs_obj_m2_sw11,2,11" \
  "$MODEL" "$TASK" \
  --probe-dir "$PROBE_DIR" \
  --use-4bit

echo ""
echo "All extra-alpha runs finished. Outputs in ${OUT_DIR}/alpha_*/"
