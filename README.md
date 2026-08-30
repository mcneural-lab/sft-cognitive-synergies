# sft-cognitive-synergies

> Lacosse, Duarte, Todd, Todd, McNamee. *AI models can predict and
> collaboratively modulate human memory search.* PNAS (in press).

## Scope

- **Figure 2B** — TPM Mantel correlation (Spearman ρ).
- **Figure 2C** — BLEU across humans, TPM-LOO, and the Gemini suite.
- **Figure 2D** — Trajectory geometry (spectral gap, curvature).
- **Figure 3** — Gemini-3-Pro vs human next-exemplar prediction.
- **Figure 4A–C** — Llama-3.3-70B perplexity controls.
- **Figure 4D** — Jaccard generation overlap from seed vs longer human context.
- **Figure 5** — Dyadic results.
- **SI Figure S1 / Table S1** — Switch prediction.
- **SI Figure S2** — Macro and micro TPM heatmaps.
- **SI Figure S4** — Cross-model next-exemplar scaling.
- **SI Figure S5 / Table S2** — Theory-driven cognitive prompting.
- **SI Figure S7** — Llama-3.3-70B next-exemplar prediction.
- **SI Figures S9–S16** — Dyadic word-count, response-time, similarity,
  artificial-delay, switch, and early-versus-late analyses.
- **Table S3** — Response-time regression.

## Setup

Python 3.11+ via [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

All entry points run under `uv run` and resolve dependencies from
`pyproject.toml` and `uv.lock`.

If `data/embeddings/conceptnet` is missing, download the 1.1 GB English
ConceptNet Numberbatch file:

```bash
scripts/download_conceptnet_numberbatch.sh
```

## Quick start

```bash
scripts/reproduce_figure.sh list      # list scripted targets
scripts/reproduce_figure.sh figure-2c 
scripts/reproduce_figure.sh all 
```

Figures land under `outputs/figures/`.

Pick output formats with `FIGURE_FORMATS`:

```bash
FIGURE_FORMATS=pdf,svg scripts/reproduce_figure.sh figure-3
```

Override per-target inputs and outputs with the documented environment
variables in `scripts/reproduce_figure.sh`.

## Layout

```text
configs/      Plotting config and figure inventory
data/         Human data, embeddings, and Llama logit-lens output
results/      LLM and baseline outputs consumed
scripts/      Entry points (reproduce_figure.sh, reproduce_analysis.sh)
src/sftbench/ Python package
```

## Inputs

The release bundles the frozen inputs used by the scripted targets, except for
the large ConceptNet Numberbatch file downloaded on demand as described above.

| Path | Size | Used for |
| --- | --- | --- |
| `data/embeddings/conceptnet` | 1.1 GB | Figure 2D and embedding-switch preprocessing; download with `scripts/download_conceptnet_numberbatch.sh` if missing |
| `data/embeddings/spelling_normalization.csv` | 15 KB | Curated spelling table applied to free-typed responses before ConceptNet lookup |
| `data/logitlens/cognitive_alignment_results_70b_100.csv.xz` | 77 MB | Figure 4A–C Llama token-probability results |
| `data/sequences/human/hills/hills.csv` | 365 KB | TPM-LOO and SI Fig S4 n-gram baselines |
| `data/sequences/animals_snafu_scheme.csv` | 20 KB | Figure 2B / SI Fig S2 SNAFU category labels |
| `data/final/filtered/all_human_wordpreds_unique_filtered.csv` | 854 KB | Human targets, next-exemplar |
| `results/hills/generate/animals-rank-1-gemini-3.0-pro-prompt-animals-gen-snafu*.csv` | 1.1 MB | Figure 2B / 2D / SI Fig S2 |
| `results/hills/generate/animals-rank-1-gemini-*-prompt-animals-gen-snafu-embedding-switch.csv` | 2.8 MB | Figure 2C (BLEU) |
| `results/hills/generate/TPMs/naive-hills-animals_tpm_human_loo.csv` | 299 KB | Figure 2C (TPM-LOO baseline) |
| `results/macro/generation/gemini-3.0-pro/` | 933 KB | Figure 4D |
| `results/micro/prediction/next-exemplar/gemini-3.0-pro/` | 1.5 MB | Figure 3 |
| `results/micro/prediction/next-exemplar/llama_70b/` | 1.7 MB | SI Fig S7 |
| `results/micro/prediction/next-exemplar/n-gram/` | 1.3 MB | Figure 3 2-gram baseline |
| `results/micro/prediction/switch/gemini-3.0-pro/` | 1.6 MB | SI Fig S1 / Table S1 |
| `results/hills/next-exemplar/supp-gemini/zero-shot/` | 8.9 MB | SI Fig S4 model scaling |
| `results/hills/next-exemplar/cognitive_prompting/` | 21 MB | SI Fig S5 / Table S2 |
| `data/dyadic/conceptnet/` | 43 MB | Figure 5, SI Figs S9--S16, and Table S3 |

## Dyadic dataset
Data collected from dyadic experiments is located in `data/dyadic/conceptnet/`.
- `solo_words_processed.csv` for solo data, `hh_words_processed.csv` for human dyad data, `ai_words_processed.csv` for human-AI dyad data: word-level data (each row corresponds to a word).
- `solo_words_with_embeddings_switches.pkl`, `hh_…pkl`, `ai_…pkl`: precomputed word embeddings (ConceptNet) and similarity-defined switches.

## Data Structure

### Word-Level Data CSVs

| Column | Description |
|--------|-------------|
| `text` | The concept generated|
| `source` | Who produced the word (player identifier in human dyads; `user` or `ai` human-AI dyads) |
| `relativeTimestamp` | Time elapsed since the beginning of the task |
| `irt` | Inter-response time (seconds) |
| `original_index` | Position in original sequence (pre filtering of duplicates and invalid items) |
| `playerID` | Unique identifier for player or game session |
| `gameID` | Game instance identifier |
| `category` | Task category (`animals`, `clothes` or `supermarket`) |
| `partner` | `AI` or partner's playerID |
| `dyadType` | `AI` or `HH` (human-human) |
| `roundIndex` | Order of task presentation |
| `switch` | Participant-identified switch events (`1`: switch to a different subcategory; `0`: not a switch)|
| `label` | Participant-described subcategory label |
| `serverStartTime` | UNIX timestamp of start of task |
| `original_text` | Raw word input (may be misspelled)|
| `normalized_text` | Normalized word (lower-case, removed spaces and spec)|
| `snafu_cat` | Subcategory defined by SNAFU norms |
| `switch_snafu` | Switch defined by changing between SNAFU subcategories |
| `word_index` | Position in sequence (0-indexed, post processing) |
| `prompt` | `Divergent`, `adjacent` (corresponding to convergent) or `inferred`|

Note: `text` in the pickle files is processed 

The embedding pipeline rewrites the `text` column present in the pickle files, from which the embeddings are calculated. This consists of removing spaces, applying a manual mapping dictionary for concept normalization (e.g. "hippo" becomes "hippopotamus"), then replacing words absent from the ConceptNet vocabulary to their closest entries (see `sftbench.embeddings.utils`). The pickles therefore hold
two versions of each word:

- `text` — post-transformation; the string actually embedded (`shirt`, `guinea_pig`)
- `text_unmodified` — the value as it appears in the CSV (`T-shirt`, `guinea pig`)


### Loaded Data
Using `sftbench.dyadic.load_data` to load data, the resulting data structure contains the following columns:

| Column | Description |
|--------|-------------|
| `playerID` | Unique identifier for player or game session |
| `gameID` | Game instance identifier |
| `dyadType` | "solo", "ha", or "hh" |
| `source` | Who produced this word (playerID or "ai") |
| `sourceType` | "human" or "ai" |
| `text` | The concept generated (as in the CSV files) |
| `word_index` | Position in sequence (0-indexed, post processing) |
| `irt` | Inter-response time (seconds) |
| `log_irt` | Log-transformed IRT |
| `log_irt_zscore` | Z-scored log IRT (within player) |
| `embedding` | Word embedding vector (normalized) |
| `embedding_similarity` | Cosine similarity with partner |
| `self_similarity` | Cosine similarity to own previous word (all conditions) |
| `other_similarity` | Cosine similarity to partner's previous word (HH/HA only) |
| `partner_midpoint_sim` | Cosine similarity of partner's word to the midpoint of own previous/current words (Figure 5 Panel C; dyadic only, `NaN` for solo) |
| `switch_sim` | Similarity-based switch flag (`1` = cluster switch, `0` = no switch); precomputed in the pickle files via `switch_median` (per-sequence median threshold). Drives the Figure S14 switch/no-switch split. |
| `word_index_split` | Early (`False`) vs late (`True`) median split by word index |


**Individual sequences are identified by grouping on:**

- HA dyads: `playerID` + `category`
- HH dyads: `gameID` + `category`
- Solo: `playerID` + `category`

## Citation

```bibtex
@article{lacosse2026cognitive,
  title   = {AI models can predict and collaboratively modulate human memory
             search},
  author  = {Lacosse, Eric and Duarte, Mariana and Todd, Graham
             and Todd, Peter M. and McNamee, Daniel C.},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2026},
}
```

## Data sources

- **ConceptNet Numberbatch** — [Speer, Chin & Havasi (2017)](https://github.com/commonsense/conceptnet-numberbatch). CC BY-SA 4.0; the 1.1 GB file is downloaded on demand, not redistributed here.
- **SNAFU category norms** — [Zemla, Cao, Mueller & Austerweil (2020)](https://doi.org/10.3758/s13428-019-01343-w).
- **Animal-fluency trajectories** — [Hills, Jones & Todd (2012)](https://doi.org/10.1037/a0027373) and [Zemla, Gooding & Austerweil (2023)](https://doi.org/10.1038/s41598-023-49858-9) (CC BY 4.0), combined.

> This data contains semantic vectors from ConceptNet Numberbatch, by Luminoso
> Technologies, Inc. You may redistribute or modify the data under the terms of
> the CC-By-SA 4.0 license.

## Third-party code

- **forager** — `src/sftbench/embeddings/switch.py` was modified from the
  [forager](https://github.com/thelexiconlab/forager) package: Kumar, A. A.,
  Apsel, M., Zhang, L., Xing, N., & Jones, M. N. (2023). forager: A Python
  package and web interface for modeling mental search. *Behavior Research
  Methods*, 1-17.

## License

Code (`src/`, `scripts/`, `configs/`) is MIT — see `LICENSE`. Bundled
data (`data/`, `results/`) is CC BY-SA 4.0 — see `LICENSE-DATA`
