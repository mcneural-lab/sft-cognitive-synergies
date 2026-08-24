#!/usr/bin/env bash
# Prepare deterministic intermediate artifacts used by scripts/reproduce_figure.sh.
#
# These targets are local-only and do not call any model API.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=repro_common.sh
source "${SCRIPT_DIR}/repro_common.sh"

cd_repo_root

usage() {
  cat <<'EOF'
Usage:
  scripts/reproduce_analysis.sh <target>

Deterministic local targets:
  tpm-baselines                  Build a fresh TPM leave-one-out baseline into TPMs/regenerated/
                                 (does not reproduce the bundled figure-2c artifact; see README).
  ngram-baselines                Regenerate 2-gram next-exemplar baselines used by figure-3 and SI Fig S4.
  apply-snafu-gemini-3-pro       Re-apply SNAFU subcategory labels to the frozen Gemini-3-Pro animal sequences.
  embedding-switch-gemini-3-pro  Re-apply embedding-switch annotations to the frozen Gemini-3-Pro animal sequences.
  evaluate-next-exemplar-gemini-3-pro  Re-evaluate frozen Gemini-3-Pro next-exemplar predictions.
  evaluate-next-exemplar-llama-70b     Re-evaluate frozen Llama-70B next-exemplar predictions.
EOF
}

run_tpm_baselines() {
  local input="${TPM_INPUT:-$(release_path data/sequences/human/hills/hills.csv)}"
  # Writes beside the bundled baseline, not over it. The sampler cannot reproduce
  # results/hills/generate/TPMs/naive-hills-animals_tpm_human_loo.csv (see README,
  # "Regenerating intermediate artifacts"), and that file is what Figure 2C reads,
  # so overwriting it in place would silently change a published panel.
  local out_dir="${TPM_OUTPUT_DIR:-$(release_path results/hills/generate/TPMs/regenerated)}"
  require_file "$input"
  ensure_dir "$out_dir"

  run_uv python src/sftbench/macro/generate_sequences_from_tpm.py \
    --input "$input" \
    --output "${out_dir}/naive-hills-animals_tpm_human_loo.csv" \
    --model tpm_human_loo \
    --word-col response \
    --min-rank 1 \
    --dataset hills \
    --seq-type seed-1 \
    --seed "${TPM_SEED:-42}"
}

run_ngram_baselines() {
  local main_input="${NGRAM_MAIN_INPUT:-$(release_path data/final/filtered/all_human_wordpreds_unique_filtered.csv)}"
  local main_out_dir="${NGRAM_MAIN_OUTPUT_DIR:-$(release_path results/micro/prediction/next-exemplar/n-gram)}"
  local si_input="${NGRAM_SI_INPUT:-$(release_path data/sequences/human/hills/hills.csv)}"
  local si_out_dir="${NGRAM_SI_OUTPUT_DIR:-$(release_path results/hills/next-exemplar/supp-gemini/zero-shot)}"
  local n="${NGRAM_N:-2}"
  local alpha="${NGRAM_ALPHA:-1.0}"

  require_file "$main_input"
  require_file "$si_input"
  ensure_dir "$main_out_dir"
  ensure_dir "$si_out_dir"

  for category in animals clothes supermarket; do
    run_uv python src/sftbench/micro/prediction/run_ngram.py \
      --sequences "$main_input" \
      --output "${main_out_dir}/${n}-gram-${alpha}-${category}.csv" \
      --n "$n" \
      --alpha "$alpha" \
      --min-rank 2 \
      --category "$category"
  done

  run_uv python src/sftbench/micro/prediction/run_ngram.py \
    --sequences "$si_input" \
    --output "${si_out_dir}/${n}-gram-${alpha}-animals.csv" \
    --n "$n" \
    --alpha "$alpha" \
    --min-rank 2 \
    --category animals
}

run_apply_snafu_gemini_3_pro() {
  local input="${SNAFU_INPUT:-$(release_path results/hills/generate/animals-rank-1-gemini-3.0-pro-prompt-animals-gen.csv)}"
  local scheme="${SNAFU_SCHEME:-$(release_path data/sequences/animals_snafu_scheme.csv)}"
  local output="${SNAFU_OUTPUT:-$(release_path results/hills/generate/animals-rank-1-gemini-3.0-pro-prompt-animals-gen-snafu.csv)}"
  require_file "$input"
  require_file "$scheme"
  ensure_dir "$(dirname "$output")"

  run_uv python src/sftbench/macro/assign_scheme_categories_to_csv.py \
    "$input" \
    "$scheme" \
    --output-csv "$output" \
    --response-col response \
    --switch-col SNAFU_switch \
    --category-col response_category

  run_uv python src/sftbench/macro/assign_scheme_categories_to_csv.py \
    "$output" \
    "$scheme" \
    --output-csv "$output" \
    --response-col human_response \
    --switch-col SNAFU_human_switch \
    --category-col human_response_category
}

run_embedding_switch_gemini_3_pro() {
  local input="${EMBED_INPUT:-$(release_path results/hills/generate/animals-rank-1-gemini-3.0-pro-prompt-animals-gen-snafu.csv)}"
  local output="${EMBED_OUTPUT:-$(release_path results/hills/generate/animals-rank-1-gemini-3.0-pro-prompt-animals-gen-snafu-embedding-switch.csv)}"
  local embedding_path="${EMBEDDING_PATH:-$(release_path data/embeddings/conceptnet)}"
  require_file "$input"
  require_file "$embedding_path"
  ensure_dir "$(dirname "$output")"

  run_uv python src/sftbench/embeddings/embeddings.py \
    "$input" \
    "$output" \
    --text-column response \
    --switch-gt-column SNAFU_switch \
    --embedding-path "$embedding_path" \
    -e "${EMBEDDING_ENGINE:-conceptnet}" \
    -s "${EMBEDDING_SWITCH_STRATEGY:-median}"
}

run_evaluate_dir() {
  local pred_dir="$1"
  local threshold="${2:-0.9}"
  local report_root="${ANALYSIS_OUTPUT_ROOT:-outputs/analysis}"
  local report_name
  report_name="$(basename "$(dirname "$pred_dir")")-$(basename "$pred_dir")"
  local report_dir="${REPORT_DIR:-${report_root}/next-exemplar/${report_name}/${EVAL_MODE:-strict}/thr-${threshold}}"
  require_dir "$pred_dir"
  ensure_dir "$report_dir"
  info "MODE=${EVAL_MODE:-strict} THRESHOLD=${threshold} SAVE_REPORTS=${SAVE_REPORTS:-1} REPORT_DIR=${report_dir} bash scripts/run_llm_next_exemplar_eval.sh ${pred_dir}"
  MODE="${EVAL_MODE:-strict}" THRESHOLD="$threshold" SAVE_REPORTS="${SAVE_REPORTS:-1}" REPORT_DIR="$report_dir" \
    bash scripts/run_llm_next_exemplar_eval.sh "$pred_dir"
}

target="${1:-help}"
case "$target" in
  tpm-baselines) run_tpm_baselines ;;
  ngram-baselines) run_ngram_baselines ;;
  apply-snafu-gemini-3-pro) run_apply_snafu_gemini_3_pro ;;
  embedding-switch-gemini-3-pro) run_embedding_switch_gemini_3_pro ;;
  evaluate-next-exemplar-gemini-3-pro)
    run_evaluate_dir "${GEMINI_NEXT_EXEMPLAR_DIR:-$(release_path results/micro/prediction/next-exemplar/gemini-3.0-pro)}" "${EVAL_THRESHOLD:-0.9}"
    ;;
  evaluate-next-exemplar-llama-70b)
    run_evaluate_dir "${LLAMA_NEXT_EXEMPLAR_DIR:-$(release_path results/micro/prediction/next-exemplar/llama_70b)}" "${EVAL_THRESHOLD:-0.9}"
    ;;
  -h|--help|help) usage ;;
  *)
    usage
    die "Unknown analysis target: $target"
    ;;
esac
