#!/usr/bin/env bash
# User-rightness probe: extra steer alphas only (-1, -0.5, 0.5).
# Skips alpha=0 and alpha=1 (already run via user_rightness_steered_alpha1_jobs.sh).
#
# Example batch submit:
#   nlprun -q sphinx -g 1 -r 200G -c 16 -p standard \
#     'bash -lc "bash /nlp/scr/zope/projects/spring2026_research/user_rightness_extra_alphas_jobs.sh"'

set -euo pipefail

export HF_HOME="${HF_HOME:-/nlp/scr/zope/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="test_results/user_rightness_steered"
mkdir -p "$OUT_DIR"

MODEL="meta-llama/Llama-3.3-70B-Instruct"
TASK="4dims"
PROBE_DIR="70b_steering_vectors/probe_out_user_rightness_0225_llama70b_FULL_llama70b"

EXTRA_ALPHAS=(-1 -0.5 0 0.5 1)

alpha_to_suffix() {
  case "$1" in
    -1)   echo "alpha_m1" ;;
    -0.5) echo "alpha_m0p5" ;;
    0.5)  echo "alpha0p5" ;;
    1)    echo "alpha1" ;;
    0)    echo "base" ;;
    *)    echo "alpha_${1//./p}" ;;
  esac
}

run_setting_extra_alphas() {
  local input_csv="$1"
  local out_prefix="$2"
  local sim_mode="$3"
  local switch_turn="$4"
  local alpha suffix

  echo ""
  echo "=========================================="
  echo "Dataset: $input_csv"
  echo "user_sim_mode=$sim_mode  user_sim_switch_turn=$switch_turn"
  echo "=========================================="

  for alpha in "${EXTRA_ALPHAS[@]}"; do
    suffix="$(alpha_to_suffix "$alpha")"
    echo "--- steer-alpha=$alpha -> ${out_prefix}_${suffix}.csv ---"
    python3 parallel_get_multiturn_steered.py \
      "$input_csv" \
      "${OUT_DIR}/${out_prefix}_${suffix}.csv" \
      "$MODEL" \
      "$TASK" \
      --probe-dir "$PROBE_DIR" \
      --steer-alpha "$alpha" \
      --use-4bit \
      --user-sim-mode "$sim_mode" \
      --user-sim-switch-turn "$switch_turn"
  done
}

run_setting_extra_alphas "data/valpairs-modified-obj-15.csv" "valpairs_obj_m2_sw11" 2 11
run_setting_extra_alphas "data/valpairs-modified-obj-15.csv" "valpairs_obj_m2_sw5"  2 5
run_setting_extra_alphas "data/valpairs-modified-val-15.csv" "valpairs_val_m1_sw5"  1 5
run_setting_extra_alphas "data/valpairs-modified-val-15.csv" "valpairs_val_m1_sw11" 1 11

echo ""
echo "All extra-alpha runs finished. Outputs in ${OUT_DIR}/"

