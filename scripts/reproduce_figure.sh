#!/usr/bin/env bash
# Reproduce manuscript figures from frozen public-release data and results.
#
# Plotting only. No model API calls and no generation. Use
# scripts/reproduce_analysis.sh for deterministic intermediate artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=repro_common.sh
source "${SCRIPT_DIR}/repro_common.sh"

cd_repo_root

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/figures}"
PLOTTING_CONFIG="${PLOTTING_CONFIG:-configs/plotting.toml}"
FIGURE_FORMAT_ARGS=()
if [[ -n "${FIGURE_FORMATS:-}" ]]; then
  FIGURE_FORMAT_ARGS=(--formats "$FIGURE_FORMATS")
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/reproduce_figure.sh list
  scripts/reproduce_figure.sh <figure-id>
  scripts/reproduce_figure.sh all

Scripted targets:
  figure-2b   Macro TPM Mantel correlation
  figure-2c   BLEU across humans, TPM-LOO, Gemini suite
  figure-2d   Trajectory geometry (spectral gap, curvature)
  figure-3    Next-exemplar prediction (Gemini-3-Pro vs human)
  figure-4abc         Llama-3.3-70B perplexity controls
  figure-4d           Jaccard generation overlap used in the proof (controlled and size-matched)
  figure-4d-control   Deprecated alias for figure-4d
  figure-5    2x2 composite: partner-midpoint distance, concepts produced, RT change, self-similarity change

  si-switch-prediction    Switch prediction (SI Fig S1, Table S1)
  si-tpm-grid             SI TPM grid (SI Fig S2)
  si-model-scaling        Cross-model scaling (SI Fig S4)
  si-cognitive-prompting  Cognitive prompting (SI Fig S5, Table S2)
  si-llama-70b-prediction Llama-70B prediction (SI Fig S7)

  si-word-count-by-category        Word counts by category (SI Fig S9)
  si-irt-early-late                Response times by task phase (SI Fig S10)
  si-self-vs-other-similarity      Self vs other similarity (SI Fig S11)
  si-word-count-irt                Word counts & RTs by prompt (SI Fig S12)
  si-artificial-delay              RT lognormal model & violin (SI Fig S13)
  si-switch-vs-no-switch           Switch vs no-switch 4-panel (SI Fig S14)
  si-self-other-late-early-change  Similarity change late-early (SI Fig S15)
  si-prompt-condition-change       Late-early change, all prompts (SI Fig S16)
  table-s3-ols                     Response-time regressions (Table S3)

Useful env overrides:
  RELEASE_DATA_ROOT=.
  OUTPUT_ROOT=outputs/figures
  PLOTTING_CONFIG=configs/plotting.toml
  FIGURE_FORMATS=png,pdf,svg
EOF
}

require_plotting_config() {
  require_file "$PLOTTING_CONFIG"
}

run_figure_2b() {
  require_plotting_config
  local input="${FIGURE_2B_INPUT:-$(release_path results/hills/generate/animals-rank-1-gemini-3.0-pro-prompt-animals-gen-snafu.csv)}"
  local out_dir="${FIGURE_2B_OUTPUT_DIR:-${OUTPUT_ROOT}/figure-2}"
  require_file "$input"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/macro/plot_tpm_from_sequences.py \
    --csv-file "$input" \
    --plot-file "${out_dir}/animals_tpm_grid.png" \
    --spearman-plot-file "${out_dir}/spearman_macro.png" \
    --spearman-scale macro \
    --within-between-plot-file "${out_dir}/within_between_macro.png" \
    --within-between-scale macro \
    --mantel "${FIGURE_2B_MANTEL:-10000}" \
    --plotting-config "$PLOTTING_CONFIG" \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_figure_2c() {
  require_plotting_config
  local out_dir="${FIGURE_2C_OUTPUT_DIR:-${OUTPUT_ROOT}/figure-2/bleu}"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/macro/plot_bleu_scores.py \
    --config-path "$PLOTTING_CONFIG" \
    --output-dir "$out_dir" \
    --data-root "$RELEASE_DATA_ROOT" \
    --adjacent-sig-test "${FIGURE_2C_ADJACENT_SIG_TEST:-wilcoxon_less}" \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_figure_2d() {
  require_plotting_config
  local input="${FIGURE_2D_INPUT:-$(release_path results/hills/generate/animals-rank-1-gemini-3.0-pro-prompt-animals-gen-snafu-embedding-switch.csv)}"
  local embedding_path="${FIGURE_2D_EMBEDDINGS:-$(release_path data/embeddings/conceptnet)}"
  local out_dir="${FIGURE_2D_OUTPUT_DIR:-${OUTPUT_ROOT}/figure-2}"
  require_file "$input"
  require_file "$embedding_path"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/macro/geometric_ai_vs_human.py \
    --input "$input" \
    --output "${out_dir}/semantic_structure_comparison.png" \
    --ultra-compact \
    --no-show \
    --embedding-path "$embedding_path" \
    --plotting-config "$PLOTTING_CONFIG" \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_figure_3() {
  require_plotting_config
  local input_dir="${FIGURE_3_INPUT_DIR:-$(release_path results/micro/prediction/next-exemplar/gemini-3.0-pro)}"
  local out_dir="${FIGURE_3_OUTPUT_DIR:-${OUTPUT_ROOT}/figure-3}"
  require_dir "$input_dir"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/micro/prediction/plotting/plot_human_vs_ai_next-exemplar_prediction-results.py \
    --input-dir "$input_dir" \
    --output-dir "$out_dir" \
    --drop-missing-normalized none \
    --model-name "${FIGURE_3_MODEL_NAME:-}" \
    --no-show \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_figure_4abc() {
  require_plotting_config
  local input="${FIGURE_4ABC_INPUT:-$(release_path data/logitlens/cognitive_alignment_results_70b_100.csv.xz)}"
  local out_dir="${FIGURE_4ABC_OUTPUT_DIR:-${OUTPUT_ROOT}/figure-4/llama}"
  require_file "$input"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/micro/llama/plot_prob_seq.py \
    --input-path "$input" \
    --config "$PLOTTING_CONFIG" \
    --output-dir "$out_dir" \
    --split-point "${FIGURE_4ABC_SPLIT_POINT:-18}" \
    --no-show \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_figure_4d() {
  require_plotting_config
  local human_data="${FIGURE_4D_HUMAN_DATA:-$(release_path data/final/filtered/all_human_wordpreds_unique_filtered.csv)}"
  local machine_dir="${FIGURE_4D_MACHINE_DATA_DIR:-$(release_path results/macro/generation/gemini-3.0-pro)}"
  local out_dir="${FIGURE_4D_OUTPUT_DIR:-${OUTPUT_ROOT}/figure-4}"
  require_file "$human_data"
  require_dir "$machine_dir"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/micro/generative/plot_new_jaccard_control.py \
    --human-data "$human_data" \
    --machine-data-dir "$machine_dir" \
    --config "$PLOTTING_CONFIG" \
    --output-dir "$out_dir" \
    --model-name "${FIGURE_4D_MODEL_NAME:-gemini-3.0-pro}" \
    --no-show \
    "${FIGURE_FORMAT_ARGS[@]}"
}



run_figure_5() {
  # Main figure 5: 2x2 composite panel (plot_fig5.py).
  local data_dir="${FIGURE_5_DATA_DIR:-$(release_path data/dyadic/conceptnet)}"
  local out_dir="${FIGURE_5_OUTPUT_DIR:-${OUTPUT_ROOT}/figure-5}"
  require_dir "$data_dir"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/dyadic/plot_fig5.py \
    --data-root "$RELEASE_DATA_ROOT" \
    --output-dir "$out_dir" \
    --no-show \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_dyadic_module() {
  # run_dyadic_module <module.py> <out_dir> [extra args...]
  local module="$1"
  local out_dir="$2"
  shift 2
  run_uv python "src/sftbench/dyadic/${module}" \
    --data-root "$RELEASE_DATA_ROOT" \
    --output-dir "$out_dir" \
    --no-show \
    "$@" \
    "${FIGURE_FORMAT_ARGS[@]}"
}

require_dyadic_data() {
  require_dir "${DYADIC_DATA_DIR:-$(release_path data/dyadic/conceptnet)}"
}

run_si_word_count_by_category() {
  require_dyadic_data
  local out_dir="${SI_WORD_COUNT_BY_CATEGORY_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/word-count-by-category}"
  ensure_dir "$out_dir"
  run_dyadic_module word_count.py "$out_dir" --panels s9
}

run_si_irt_early_late() {
  require_dyadic_data
  local out_dir="${SI_IRT_EARLY_LATE_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/irt-early-late}"
  ensure_dir "$out_dir"
  run_dyadic_module irt_phase.py "$out_dir" --panels s10
}

run_si_self_vs_other_similarity() {
  require_dyadic_data
  local out_dir="${SI_SELF_VS_OTHER_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/self-vs-other-similarity}"
  ensure_dir "$out_dir"
  run_dyadic_module self_other_similarity.py "$out_dir"
}

run_si_word_count_irt() {
  require_dyadic_data
  local out_dir="${SI_WORD_COUNT_IRT_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/word-count-irt}"
  ensure_dir "$out_dir"
  run_dyadic_module word_count.py "$out_dir" --panels s12a,s12b
  run_dyadic_module irt_phase.py "$out_dir" --panels s12c
}

run_si_artificial_delay() {
  require_dyadic_data
  local out_dir="${SI_ARTIFICIAL_DELAY_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/artificial-delay}"
  ensure_dir "$out_dir"
  run_dyadic_module rt_lognorm.py "$out_dir"
  run_dyadic_module irt_violin.py "$out_dir"
}

run_si_switch_vs_no_switch() {
  require_dyadic_data
  local out_dir="${SI_SWITCH_VS_NO_SWITCH_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/switch-vs-no-switch}"
  ensure_dir "$out_dir"
  run_dyadic_module irt_phase.py "$out_dir" --panels s14
}

run_si_self_other_late_early_change() {
  require_dyadic_data
  local out_dir="${SI_SELF_OTHER_CHANGE_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/self-other-late-early-change}"
  ensure_dir "$out_dir"
  run_dyadic_module similarity_change_si.py "$out_dir"
}

run_si_prompt_condition_change() {
  require_dyadic_data
  local out_dir="${SI_PROMPT_CONDITION_CHANGE_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/prompt-condition-change}"
  ensure_dir "$out_dir"
  run_dyadic_module early_late_si.py "$out_dir"
}

run_table_s3_ols() {
  require_dyadic_data
  local out_dir="${TABLE_S3_OLS_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/table-s3}"
  ensure_dir "$out_dir"
  run_dyadic_module ols_word_index.py "$out_dir"
}

run_si_switch_prediction() {
  require_plotting_config
  local input_dir="${SI_SWITCH_INPUT_DIR:-$(release_path results/micro/prediction/switch/gemini-3.0-pro)}"
  local out_dir="${SI_SWITCH_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/switch-prediction}"
  require_dir "$input_dir"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/micro/prediction/plotting/plot_human_vs_ai_next-exemplar_switch-results.py \
    --targets-dir "$input_dir" \
    --config "$PLOTTING_CONFIG" \
    --output-dir "$out_dir" \
    --metric "${SI_SWITCH_METRIC:-accuracy}" \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_si_tpm_grid() {
  require_plotting_config
  local input="${SI_TPM_GRID_INPUT:-$(release_path results/hills/generate/animals-rank-1-gemini-3.0-pro-prompt-animals-gen-snafu-embedding-switch.csv)}"
  local out_dir="${SI_TPM_GRID_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/tpm-grid}"
  require_file "$input"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/macro/plot_tpm_from_sequences.py \
    --csv-file "$input" \
    --plot-file "${out_dir}/animals_tpm_grid.png" \
    --plotting-config "$PLOTTING_CONFIG" \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_si_model_scaling() {
  require_plotting_config
  local input_dir="${SI_MODEL_SCALING_INPUT_DIR:-$(release_path results/hills/next-exemplar/supp-gemini/zero-shot)}"
  local out_dir="${SI_MODEL_SCALING_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/model-scaling}"
  require_dir "$input_dir"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/micro/prediction/plotting/new_plot_prediction_results.py \
    --input-dir "$input_dir" \
    --recursive \
    --output-dir "$out_dir" \
    --category animals \
    --annotate \
    --no-show \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_si_cognitive_prompting() {
  require_plotting_config
  local input_dir="${SI_CP_INPUT_DIR:-$(release_path results/hills/next-exemplar/cognitive_prompting)}"
  local out_dir="${SI_CP_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/cognitive-prompting}"
  require_dir "$input_dir"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/micro/prediction/plotting/new_plot_prediction_results.py \
    --input-dir "$input_dir" \
    --exclude-default-ids \
    --recursive \
    --output-dir "$out_dir" \
    --annotate \
    --category animals \
    --category animals-baseline \
    --category animals-few-shot \
    --category animals-baseline-few-shot \
    --no-show \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_si_llama_70b_prediction() {
  require_plotting_config
  local input_dir="${SI_LLAMA_INPUT_DIR:-$(release_path results/micro/prediction/next-exemplar/llama_70b)}"
  local out_dir="${SI_LLAMA_OUTPUT_DIR:-${OUTPUT_ROOT}/supp/llama-70b-prediction}"
  require_dir "$input_dir"
  ensure_dir "$out_dir"
  run_uv python src/sftbench/micro/prediction/plotting/plot_human_vs_ai_next-exemplar_prediction-results.py \
    --input-dir "$input_dir" \
    --output-dir "$out_dir" \
    --drop-missing-normalized none \
    --model-name "${SI_LLAMA_MODEL_NAME:-Llama 70B 3.3 Instruct}" \
    --no-show \
    "${FIGURE_FORMAT_ARGS[@]}"
}

run_all() {
  run_figure_2b
  run_figure_2c
  run_figure_2d
  run_figure_3
  run_figure_4abc
  run_figure_4d
  run_figure_5
  run_si_switch_prediction
  run_si_tpm_grid
  run_si_model_scaling
  run_si_cognitive_prompting
  run_si_llama_70b_prediction
  run_si_word_count_by_category
  run_si_irt_early_late
  run_si_self_vs_other_similarity
  run_si_word_count_irt
  run_si_artificial_delay
  run_si_switch_vs_no_switch
  run_si_self_other_late_early_change
  run_si_prompt_condition_change
  run_table_s3_ols
}

run_target() {
  local target="$1"
  case "$target" in
    list)
      list_figures_from_config
      ;;
    figure-2b) run_figure_2b ;;
    figure-2c) run_figure_2c ;;
    figure-2d) run_figure_2d ;;
    figure-3) run_figure_3 ;;
    figure-4abc) run_figure_4abc ;;
    figure-4d) run_figure_4d ;;
    figure-4d-control) run_figure_4d ;;   # backwards-compatible alias
    figure-5) run_figure_5 ;;
    si-switch-prediction) run_si_switch_prediction ;;
    si-tpm-grid) run_si_tpm_grid ;;
    si-model-scaling) run_si_model_scaling ;;
    si-cognitive-prompting) run_si_cognitive_prompting ;;
    si-llama-70b-prediction) run_si_llama_70b_prediction ;;
    si-word-count-by-category) run_si_word_count_by_category ;;
    si-irt-early-late) run_si_irt_early_late ;;
    si-self-vs-other-similarity) run_si_self_vs_other_similarity ;;
    si-word-count-irt) run_si_word_count_irt ;;
    si-artificial-delay) run_si_artificial_delay ;;
    si-switch-vs-no-switch) run_si_switch_vs_no_switch ;;
    si-self-other-late-early-change) run_si_self_other_late_early_change ;;
    si-prompt-condition-change) run_si_prompt_condition_change ;;
    table-s3-ols) run_table_s3_ols ;;
    all) run_all ;;
    -h|--help|help) usage ;;
    *)
      usage
      die "Unknown figure target: $target"
      ;;
  esac
}

target="${1:-help}"
run_target "$target"
