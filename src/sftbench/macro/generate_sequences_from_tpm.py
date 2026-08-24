import random
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import typer
from loguru import logger

from sftbench.macro.sequence_macro_alignment import (
    _parse_categories,
    create_transition_matrices,
)
from sftbench.reproducibility import seed_everything, track

app = typer.Typer(
    name="sftbench-tpm-gen",
    help="Generate sequences from TPMs calculated from human data (word-level, Leave-One-Out)",
)

OUTPUT_COLUMNS_ORDER = [
    "response",
    "seq_type",
    "id",
    "model",
    "rank",
    "target_length",
    "sub-seq-n",
    "human_response",
    "data-category",
    "dataset",
]


def get_unique_words(series: pd.Series, parse: bool = False) -> list[str]:
    """Get unique words from the series."""
    if parse:
        parsed = series.apply(_parse_categories)
        all_words = set()
        for lst in parsed:
            all_words.update(lst)
        return sorted(all_words)
    return sorted(series.dropna().astype(str).unique().tolist())


def compute_individual_tpms(
    df: pd.DataFrame,
    vocabulary: list[str],
    id_col: str = "id",
    order_col: str = "rank",
    word_col: str = "response",
) -> dict[str, np.ndarray]:
    """
    Compute normalized TPM for each subject individually.
    Returns a dictionary {subject_id: tpm_matrix_numpy_array}.
    """
    vocab_to_idx = {w: i for i, w in enumerate(vocabulary)}
    n_vocab = len(vocabulary)

    individual_tpms = {}

    grouped = df.groupby(id_col)

    for subj_id, group in grouped:
        g = group.sort_values(order_col)
        words = g[word_col].astype(str).tolist()

        if len(words) < 2:
            continue

        counts = np.zeros((n_vocab, n_vocab), dtype=float)
        valid_transitions = 0

        for t in range(len(words) - 1):
            src = words[t]
            dst = words[t + 1]

            if src in vocab_to_idx and dst in vocab_to_idx:
                i_src = vocab_to_idx[src]
                i_dst = vocab_to_idx[dst]
                counts[i_src, i_dst] += 1.0
                valid_transitions += 1

        if valid_transitions == 0:
            continue

        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

        individual_tpms[subj_id] = probs

    return individual_tpms


def get_start_counts_by_id(
    df: pd.DataFrame,
    vocabulary: list[str],
    id_col: str = "id",
    word_col: str = "response",
    rank_col: str = "rank",
    parse: bool = False,
    start_rank: int = 1,
) -> dict[str, np.ndarray]:
    """
    Get the rank-1 word counts vector for each ID.
    Returns {subject_id: counts_vector_numpy_array}.
    """
    vocab_to_idx = {w: i for i, w in enumerate(vocabulary)}
    n_vocab = len(vocabulary)

    start_vectors = {}

    rank1_df = df[df[rank_col] == start_rank]

    grouped = rank1_df.groupby(id_col)

    for subj_id, group in grouped:
        counts = np.zeros(n_vocab)
        items_series = cast(pd.Series, group[word_col])

        if parse:
            parsed_items = items_series.apply(_parse_categories)
            for lst in parsed_items:
                for w in cast(list[str], lst):
                    if w in vocab_to_idx:
                        counts[vocab_to_idx[w]] += 1.0
        else:
            for word in items_series:
                w_str = str(word)
                if w_str in vocab_to_idx:
                    counts[vocab_to_idx[w_str]] += 1.0

        total = counts.sum()
        if total > 0:
            counts = counts / total

        start_vectors[subj_id] = counts

    return start_vectors


def sample_next_word(
    current_word: str, tpm_matrix: np.ndarray, vocabulary: list[str], vocab_to_idx: dict[str, int]
) -> str:
    """
    Sample the next word using a numpy TPM matrix.
    """
    if current_word not in vocab_to_idx:
        return random.choice(vocabulary)

    idx = vocab_to_idx[current_word]
    probs = tpm_matrix[idx]

    if np.isnan(probs).all() or probs.sum() == 0:
        return random.choice(vocabulary)

    probs = np.nan_to_num(probs)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(len(probs)) / len(probs)

    next_idx = np.random.choice(len(vocabulary), p=probs)
    return vocabulary[next_idx]


@app.command()
def main(
    input_file: Path = typer.Option(..., "--input", "-i", exists=True, help="Path to input CSV with human data."),
    output_file: Path = typer.Option("output_tpm.csv", "--output", "-o", help="Path to output CSV."),
    num_sequences: int = typer.Option(
        -1,
        "--num",
        "-n",
        help="Number of sequences to generate. -1 to match unique IDs in input.",
    ),
    fractional_weighting: bool = typer.Option(False, "--fractional", "-f", help="Use fractional weighting."),
    maximum_coherence: bool = typer.Option(False, "--coherence", "-c", help="Use maximum coherence."),
    id_col: str = typer.Option("id", help="Column name for Subject ID."),
    word_col: str = typer.Option("response", help="Column name for Word/Category."),
    rank_col: str = typer.Option("rank", help="Column name for Rank/Order."),
    min_rank: int = typer.Option(1, "--min-rank", help="Minimum rank to include (>=)."),
    dataset: str = typer.Option("hills", "--dataset", help="Dataset marker to include in output CSV."),
    seq_type: str = typer.Option("seed-1", "--seq-type", help="Sequence type marker (aligns with config generator)."),
    model: str = typer.Option("tpm_human_loo", "--model", help="Model name to write to the output CSV."),
    seed: int = typer.Option(42, help="Random seed."),
):
    """
    Generates sequences based on Leave-One-Out (LOO) TPMs.
    For each subject ID, we build a model using all *other* subjects' data.
    """
    logger.info(f"Loading data from {input_file}...")
    df: pd.DataFrame = pd.read_csv(input_file)

    for col in [id_col, word_col, rank_col]:
        if col not in df.columns:
            logger.error(f"Column '{col}' not found in input file. Columns: {df.columns.tolist()}")
            raise typer.Exit(1)

    logger.info(f"Filtering data where {rank_col} >= {min_rank}...")
    df = df.loc[df[rank_col] >= min_rank].copy()
    if df.empty:
        logger.error("No data remaining after rank filtering.")
        raise typer.Exit(1)

    # Keep consistent with config-based generator: ignore the explicit "<START>" token if present.
    df = df.loc[df[word_col] != "<START>"].copy()
    if df.empty:
        logger.error("No data remaining after filtering out '<START>'.")
        raise typer.Exit(1)

    unique_ids = df[id_col].astype(str).unique()

    lengths_by_id: dict[str, int] = df.groupby(id_col)[word_col].size().astype(int).to_dict()

    lengths_by_id = {str(k): v for k, v in lengths_by_id.items()}

    human_response_by_id_rank: dict[tuple[str, int], str] = {}
    for row in df.sort_values(rank_col).itertuples(index=False):
        row_id = str(getattr(row, id_col))
        row_rank = int(getattr(row, rank_col))
        row_resp = str(getattr(row, word_col))
        human_response_by_id_rank[(row_id, row_rank)] = row_resp

    if num_sequences == -1:
        target_ids = unique_ids
    else:
        if num_sequences > len(unique_ids):
            logger.warning(
                f"Requested {num_sequences} sequences but only {len(unique_ids)} unique IDs found. Cycling IDs to meet count."
            )
            target_ids = []
            while len(target_ids) < num_sequences:
                target_ids.extend(unique_ids)
            target_ids = target_ids[:num_sequences]
        else:
            target_ids = unique_ids[:num_sequences]

    count_to_generate = len(target_ids)
    logger.info(f"Found {len(unique_ids)} unique IDs. Will generate {count_to_generate} sequences (LOO mode).")

    advanced_mode = fractional_weighting or maximum_coherence
    mode_str = "Advanced (Fractional/Coherence)" if advanced_mode else "Naive"
    logger.info(f"Mode: {mode_str}")

    logger.info("Extracting vocabulary...")
    vocabulary = get_unique_words(cast(pd.Series, df[word_col]), parse=advanced_mode)
    vocab_to_idx = {w: i for i, w in enumerate(vocabulary)}
    n_vocab = len(vocabulary)
    logger.info(f"Found {len(vocabulary)} unique words/tokens in vocabulary.")

    logger.info("Computing individual TPMs...")

    if advanced_mode:
        matrices_dict = create_transition_matrices(
            df,
            vocabulary,
            id_col=id_col,
            order_col=rank_col,
            category_col=word_col,
            fractional_weighting=fractional_weighting,
            maximum_coherence=maximum_coherence,
            include_self_transitions=True,
        )

        individual_tpms = {}
        for sid, mat_df in matrices_dict.items():
            individual_tpms[sid] = mat_df.values

    else:
        individual_tpms = compute_individual_tpms(df, vocabulary, id_col=id_col, order_col=rank_col, word_col=word_col)

    individual_tpms = {str(k): v for k, v in individual_tpms.items()}

    valid_ids = list(individual_tpms.keys())

    if not valid_ids:
        logger.error("No valid transition matrices could be computed.")
        raise typer.Exit(1)

    global_sum_tpm = np.zeros((n_vocab, n_vocab), dtype=float)
    for tpm in individual_tpms.values():
        tpm = np.nan_to_num(tpm)
        global_sum_tpm += tpm

    total_subjects = len(valid_ids)

    logger.info("Computing start distributions...")
    individual_starts = get_start_counts_by_id(
        df,
        vocabulary,
        id_col=id_col,
        word_col=word_col,
        rank_col=rank_col,
        parse=advanced_mode,
        start_rank=min_rank,
    )
    individual_starts = {str(k): v for k, v in individual_starts.items()}

    if len(target_ids) and not any(str(sid) in individual_tpms for sid in target_ids):
        raise RuntimeError(
            f"Leave-one-out lookup matched 0/{len(target_ids)} target ids in individual_tpms; check `id` key types."
        )

    global_sum_start = np.zeros(n_vocab, dtype=float)
    valid_start_ids = []
    for sid, vec in individual_starts.items():
        global_sum_start += vec
        valid_start_ids.append(sid)

    total_start_subjects = len(valid_start_ids)

    logger.info(f"Generating {count_to_generate} sequences (per-ID target length matched to input length)...")
    np.random.seed(seed)
    random.seed(seed)

    generated_rows = []
    for seq_id in track(list(target_ids), description="Generating"):
        seq_id_str = str(seq_id)
        seq_target_length = lengths_by_id[seq_id_str]
        if seq_target_length < 1:
            continue

        seed_word = human_response_by_id_rank.get((seq_id_str, 1))
        if seed_word is None:
            logger.warning(f"Skipping id={seq_id_str}: no human seed found at rank=1 after filtering.")
            continue

        # --- TPM LOO ---
        if seq_id in individual_tpms:
            loo_sum_tpm = global_sum_tpm - np.nan_to_num(individual_tpms[seq_id])
            loo_count = total_subjects - 1
        else:
            loo_sum_tpm = global_sum_tpm
            loo_count = total_subjects

        if loo_count > 0:
            loo_avg_tpm = loo_sum_tpm / loo_count
        else:
            loo_avg_tpm = np.zeros((n_vocab, n_vocab))

        if seq_id in individual_starts:
            loo_sum_start = global_sum_start - individual_starts[seq_id]
            loo_start_count = total_start_subjects - 1
        else:
            loo_sum_start = global_sum_start
            loo_start_count = total_start_subjects

        if loo_start_count > 0:
            loo_start_probs = loo_sum_start / loo_start_count
            if loo_start_probs.sum() > 0:
                loo_start_probs /= loo_start_probs.sum()
            else:
                loo_start_probs = np.ones(n_vocab) / n_vocab
        else:
            loo_start_probs = np.ones(n_vocab) / n_vocab

        start_word = seed_word

        sequence = [start_word]
        current_word = start_word

        for _ in range(seq_target_length - 1):
            next_word = sample_next_word(current_word, loo_avg_tpm, vocabulary, vocab_to_idx)
            sequence.append(next_word)
            current_word = next_word

        for rank, item in enumerate(sequence, 1):
            human_response = human_response_by_id_rank.get((str(seq_id), int(rank)))
            generated_rows.append(
                {
                    "response": item,
                    "seq_type": seq_type,
                    "id": str(seq_id),
                    "model": model,
                    "rank": rank,
                    "target_length": seq_target_length,
                    "sub-seq-n": 0,
                    "human_response": human_response,
                    "data-category": (df["data-category"].iloc[0] if "data-category" in df.columns else "unknown"),
                    "dataset": dataset,
                }
            )
    out_df = pd.DataFrame(generated_rows)
    for col in OUTPUT_COLUMNS_ORDER:
        if col not in out_df.columns:
            out_df[col] = None
    out_df = out_df.loc[:, OUTPUT_COLUMNS_ORDER]
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_file, index=False)
    logger.info(f"Saved generated sequences to {output_file}")


if __name__ == "__main__":
    seed_everything()
    app()
