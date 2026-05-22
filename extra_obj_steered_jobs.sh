#!/usr/bin/env bash
# Objectivity-seeking probe: extra steer alphas only (-1, -0.5, 0.5).
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
EXTRA_ALPHAS=(-1 -0.5 0.5)

alpha_dir_name() {
  printf 'alpha_%s' "$1"
}

run_setting() {
  local input_csv="$1"
  local out_prefix="$2"
  local sim_mode="$3"
  local switch_turn="$4"
  local alpha alpha_subdir out_csv

  echo ""
  echo "=========================================="
  echo "Dataset: $input_csv"
  echo "user_sim_mode=$sim_mode  user_sim_switch_turn=$switch_turn"
  echo "=========================================="

  for alpha in "${EXTRA_ALPHAS[@]}"; do
    alpha_subdir="$(alpha_dir_name "$alpha")"
    mkdir -p "${OUT_DIR}/${alpha_subdir}"
    out_csv="${OUT_DIR}/${alpha_subdir}/${out_prefix}.csv"
    echo "--- steer-alpha=$alpha -> $out_csv ---"
    "$PYTHON" parallel_get_multiturn_steered.py \
      "$input_csv" \
      "$out_csv" \
      "$MODEL" \
      "$TASK" \
      --probe-dir "$PROBE_DIR" \
      --steer-alpha "$alpha" \
      --use-4bit \
      --user-sim-mode "$sim_mode" \
      --user-sim-switch-turn "$switch_turn"
  done
}

run_setting "data/valpairs-modified-obj-15.csv" "valpairs_obj_m2_sw11" 2 11
run_setting "data/valpairs-modified-obj-15.csv" "valpairs_obj_m2_sw5"  2 5
run_setting "data/valpairs-modified-val-15.csv" "valpairs_val_m1_sw5"  1 5
run_setting "data/valpairs-modified-val-15.csv" "valpairs_val_m1_sw11" 1 11

echo ""
echo "All extra-alpha runs finished. Outputs in ${OUT_DIR}/alpha_*/"

