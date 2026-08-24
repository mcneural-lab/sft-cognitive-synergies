#!/bin/bash
set -euo pipefail

# Define colors for a readable terminal display.
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# -----------------------------
# Configuration
# -----------------------------
# You can still keep defaults here, but this script now supports evaluating an arbitrary
# directory recursively (all *.csv files in that directory + subdirectories).
#
# Usage examples:
#   scripts/run_llm_next_exemplar_eval.sh results/micro/prediction/next-exemplar/gemini-3.0-pro
#   PRED_DIR=results/hills/... MODE=similarity THRESHOLD=0.8 scripts/run_llm_next_exemplar_eval.sh
#
# CLI args:
#   1) PRED_DIR (optional): directory containing CSVs (searched recursively)
#
# Env overrides:
#   MODE (default: similarity)
#   THRESHOLD (default: 0.8)
#   SAVE_REPORTS (default: 1)
#   REPORT_DIR (default: <PRED_DIR>/eval/<MODE>/thr-<THRESHOLD>)
#
PRED_DIR="${1:-${PRED_DIR:-}}"
MODE="${MODE:-similarity}"
THRESHOLD="${THRESHOLD:-0.9}"
SAVE_REPORTS="${SAVE_REPORTS:-1}"

if [[ -z "${PRED_DIR}" ]]; then
  PRED_DIR="results/micro/prediction/next-exemplar/gemini-3.0-pro"
fi

REPORT_DIR="${REPORT_DIR:-${PRED_DIR}/eval/${MODE}/thr-${THRESHOLD}}"

# -----------------------------
# Helpers
# -----------------------------
ts() { date +"%Y-%m-%d %H:%M:%S"; }

die() {
  echo -e "${YELLOW}✖ ${1}${NC}" 1>&2
  exit 1
}

relpath_from() {
  # relpath_from <base_dir> <path>
  # Prints a reasonable relative path for display/report naming.
  #
  # Bash-only implementation (no python dependency). Works for paths under base.
  local base="${1%/}"
  local path="${2}"

  # If path is exactly base, return "."
  if [[ "${path}" == "${base}" ]]; then
    echo "."
    return 0
  fi

  # If path is under base/, strip the prefix
  if [[ "${path}" == "${base}/"* ]]; then
    echo "${path#${base}/}"
    return 0
  fi

  # Fallback: return original path (shouldn't happen for our usage)
  echo "${path}"
}

# -----------------------------
# Header
# -----------------------------
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}STARTING NEXT-EXEMPLAR EVALUATION${NC}"
echo -e "${BLUE}• Predictions dir: ${YELLOW}${PRED_DIR}${NC}"
echo -e "${BLUE}• Mode:            ${YELLOW}${MODE}${NC}"
echo -e "${BLUE}• Threshold:       ${YELLOW}${THRESHOLD}${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

[[ -d "${PRED_DIR}" ]] || die "Predictions directory not found: ${PRED_DIR}"

# Collect CSVs recursively (null-delimited for safety)
mapfile -d '' CSV_FILES < <(find "${PRED_DIR}" -type f -name "*.csv" -print0 | sort -z)

if [[ "${#CSV_FILES[@]}" -eq 0 ]]; then
  die "No CSV files found under: ${PRED_DIR}"
fi

if [[ "${SAVE_REPORTS}" -eq 1 ]]; then
  mkdir -p "${REPORT_DIR}"
fi

total_runs="${#CSV_FILES[@]}"
run_idx=0

for input_csv in "${CSV_FILES[@]}"; do
  run_idx=$(( run_idx + 1 ))

  display_rel="$(relpath_from "${PRED_DIR}" "${input_csv}")"

  echo -e "${GREEN}[${YELLOW}${run_idx}${NC}${GREEN}/${YELLOW}${total_runs}${NC}${GREEN}] ${BLUE}($(ts))${NC} Evaluating ${YELLOW}${display_rel}${NC}"
  echo -e "${BLUE}  • Input: ${YELLOW}${input_csv}${NC}"

  cmd=(uv run src/sftbench/micro/prediction/evaluate_predictions.py \
    "${input_csv}" \
    --mode "${MODE}" \
    --threshold "${THRESHOLD}")

  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "${SAVE_REPORTS}" -eq 1 ]]; then
    # Mirror directory structure under REPORT_DIR, and store a .txt report per CSV
    report_subdir="${REPORT_DIR}/$(dirname "${display_rel}")"
    mkdir -p "${report_subdir}"
    report_path="${REPORT_DIR}/${display_rel}.txt"

    # Preserve the evaluator exit code when piping to tee
    set +e
    "${cmd[@]}" | tee "${report_path}"
    rc="${PIPESTATUS[0]}"
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      die "Evaluation failed for ${input_csv} (exit code ${rc})"
    fi
    echo -e "${BLUE}  • Report: ${YELLOW}${report_path}${NC}"
  else
    "${cmd[@]}"
  fi

  echo -e "${GREEN}Done: ${YELLOW}${display_rel}${NC}"
done

echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Evaluation completed successfully.${NC}"
