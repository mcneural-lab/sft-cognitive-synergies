import dataclasses
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from sftbench.reproducibility import seed_everything

app = typer.Typer(name="SFT Bench N-Gram Prediction")

MAX_SEQ_LENGTH = 100


@dataclass
class PredictionResult:
    id: str
    rank: int
    model: str
    category: str
    actual_response: str
    predicted_response: str
    is_correct: bool
    is_chosen: bool
    choice_index: int
    input_sequence: str
    logprobs: str = ""


def normalize_item(text: Any) -> str:
    if text is None:
        return ""
    return str(text).lower().strip().replace(" ", "")


def format_csv_value(value: Any, force_quote: bool = False) -> str:
    """Formats a value for CSV, forcing quotes if requested or required."""
    if value is None:
        return ""
    val_str = str(value)
    needs_quote = force_quote or "," in val_str or '"' in val_str or "\n" in val_str
    if needs_quote:
        val_str = val_str.replace('"', '""')
        return f'"{val_str}"'
    return val_str


class NGramModel:
    def __init__(self, n: int, alpha: float):
        self.n = n
        self.alpha = alpha
        self.ngrams = defaultdict(Counter)
        self.vocab: set[str] = set()
        self.context_counts = Counter()

    def train(self, sequences: list[list[str]]):
        for seq in sequences:
            normalized_seq = [normalize_item(w) for w in seq]
            for w in normalized_seq:
                self.vocab.add(w)

            for i in range(len(normalized_seq) - 1):
                target = normalized_seq[i + 1]

                history_end = i + 1
                history_start = max(0, history_end - (self.n - 1))
                context = tuple(normalized_seq[history_start:history_end])

                self.ngrams[context][target] += 1
                self.context_counts[context] += 1

    def predict(self, history: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        norm_hist = [normalize_item(w) for w in history]

        hist_len = len(norm_hist)
        start = max(0, hist_len - (self.n - 1))
        context = tuple(norm_hist[start:])

        vocab_size = len(self.vocab)
        denom = self.context_counts[context] + (self.alpha * vocab_size)

        if denom == 0:
            return []

        results = []
        sorted_vocab = sorted(list(self.vocab))

        for w in sorted_vocab:
            count = self.ngrams[context][w]
            prob = (count + self.alpha) / denom
            results.append((w, prob))

        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]


@app.command()
def run(
    sequences: Path = typer.Option(..., "--sequences", "-q", exists=True, help="Path to input CSV (test set)."),
    train_sequences: Path = typer.Option(
        None, "--train-sequences", "-t", help="Path to training CSV. If not provided, uses LOO on sequences."
    ),
    output: Path = typer.Option("output_ngram.csv", "--output", "-o", help="Path to output CSV."),
    n: int = typer.Option(2, "--n", "-n", help="Order of N-gram (e.g. 2 for bigram)."),
    alpha: float = typer.Option(1.0, "--alpha", "-a", help="Laplace smoothing parameter."),
    min_rank: int = typer.Option(2, "--min-rank", "-r", help="Minimum rank to predict (default 2)."),
    category: str = typer.Option(None, "--category", "-C", help="Filter by category (e.g. animals)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
):
    """
    Run N-Gram prediction pipeline with Leave-One-Out cross-validation.

    The released manuscript baseline uses the argmax prediction under the
    smoothed n-gram distribution. It does not mask responses that already appear
    in the input context.
    """
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="DEBUG" if verbose else "INFO", format="{message}")

    logger.info(f"Loading sequences from {sequences}...")
    try:
        df_test = pd.read_csv(sequences)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise typer.Exit(code=1) from e

    df_train = None
    if train_sequences:
        logger.info(f"Loading training sequences from {train_sequences}...")
        try:
            df_train = pd.read_csv(train_sequences)
        except Exception as e:
            logger.error(f"Failed to read Training CSV: {e}")
            raise typer.Exit(code=1) from e

    def preprocess(df):
        if "item_rank" in df.columns:
            df.rename(columns={"item_rank": "rank"}, inplace=True)
        if "response" in df.columns:
            df = df.dropna(subset=["response"])
            df = df[df["response"] != "<START>"]
        if "rank" in df.columns:
            df = df[df["rank"] >= 1]

        cat_col = None
        if "data-category" in df.columns:
            cat_col = "data-category"
        elif "category" in df.columns:
            cat_col = "category"

        if category and cat_col:
            available = df[cat_col].unique()
            if category in available:
                df = df[df[cat_col] == category]

        if "source_id" in df.columns:
            df.drop_duplicates(subset=["source_id", "rank"], inplace=True)

        return df

    df_test = preprocess(df_test)
    if df_train is not None:
        df_train = preprocess(df_train)

    test_ids = df_test["id"].unique()
    logger.info(f"Found {len(test_ids)} unique sequences for testing.")

    test_groups = df_test.groupby("id")

    try:
        df_test_raw = pd.read_csv(sequences)
        df_test_raw_cat = df_test_raw

        raw_cat_col = None
        if "data-category" in df_test_raw_cat.columns:
            raw_cat_col = "data-category"
        elif "category" in df_test_raw_cat.columns:
            raw_cat_col = "category"

        if category and raw_cat_col:
            available = df_test_raw_cat[raw_cat_col].unique()
            if category in available:
                df_test_raw_cat = df_test_raw_cat[df_test_raw_cat[raw_cat_col] == category]

        raw_cat_ids = (
            set(str(x) for x in df_test_raw_cat["id"].dropna().unique()) if "id" in df_test_raw_cat.columns else set()
        )
        processed_ids = set(str(x) for x in test_ids)

        skipped_test_ids = sorted(raw_cat_ids - processed_ids)
        if skipped_test_ids:
            logger.info(
                f"Skipped {len(skipped_test_ids)} sequence(s) within category '{category}' (no usable items after preprocessing): "
                + ", ".join(skipped_test_ids)
            )
        else:
            if category:
                logger.info(f"Skipped 0 sequences within category '{category}'.")
    except Exception:
        if category:
            logger.info(f"Skipped sequence ID report unavailable for category '{category}'.")

    train_groups_map = {}
    if df_train is not None:
        train_ids = df_train["id"].unique()
        logger.info(f"Found {len(train_ids)} unique sequences for training.")
        t_groups = df_train.groupby("id")
        for tid, group in t_groups:
            if "rank" in group.columns:
                group = group.sort_values("rank")
            train_groups_map[tid] = group["response"].tolist()

    all_results = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(),
        transient=True,
    )

    with progress:
        task_id = progress.add_task("Processing sequences...", total=len(test_ids))

        all_sequences_map = {}
        if df_train is None:
            for tid, group in test_groups:
                if "rank" in group.columns:
                    group = group.sort_values("rank")
                all_sequences_map[tid] = group["response"].tolist()

        target_ids = list(all_sequences_map.keys()) if df_train is None else list(test_ids)

        for seq_id in target_ids:
            current_train_seqs = []

            if df_train is not None:
                for tid, seq in train_groups_map.items():
                    if str(tid) != str(seq_id):
                        current_train_seqs.append(seq)
            else:
                for tid, seq in all_sequences_map.items():
                    if str(tid) != str(seq_id):
                        current_train_seqs.append(seq)

            if not current_train_seqs:
                logger.warning(f"No training data for sequence {seq_id} (LOO empty).")

            model = NGramModel(n=n, alpha=alpha)
            model.train(current_train_seqs)

            if df_train is None:
                test_seq = all_sequences_map[seq_id]
            else:
                group = test_groups.get_group(seq_id)
                if "rank" in group.columns:
                    group = group.sort_values("rank")
                test_seq = group["response"].tolist()

            test_seq = test_seq[:MAX_SEQ_LENGTH]

            model_name = f"ngram-{n}"

            for i in range(1, len(test_seq)):
                rank = i + 1
                if rank < min_rank:
                    continue

                input_items = test_seq[:i]
                target = test_seq[i]
                formatted_target = normalize_item(target)

                prediction_probs = model.predict(input_items, top_k=1)

                step_res = []

                for k, (pred_word, prob) in enumerate(prediction_probs):
                    res = PredictionResult(
                        id=str(seq_id),
                        rank=rank,
                        model=model_name,
                        category=str(category) if category else "unknown",
                        input_sequence=",".join(input_items),
                        actual_response=formatted_target,
                        predicted_response=pred_word,
                        is_correct=(pred_word == formatted_target),
                        is_chosen=True,
                        choice_index=k,
                        logprobs=f"{math.log(prob):.4f}" if prob > 0 else "-inf",
                    )
                    step_res.append(res)

                if not step_res:
                    res = PredictionResult(
                        id=str(seq_id),
                        rank=rank,
                        model=model_name,
                        category=str(category) if category else "unknown",
                        input_sequence=",".join(input_items),
                        actual_response=formatted_target,
                        predicted_response="",
                        is_correct=False,
                        is_chosen=True,
                        choice_index=0,
                        logprobs="-inf",
                    )
                    step_res.append(res)

                all_results.extend(step_res)

            progress.advance(task_id)

    logger.info(f"Processing complete. Generated {len(all_results)} predictions.")

    chosen = [r for r in all_results if r.is_chosen]

    if not chosen:
        logger.info("No chosen predictions to calculate accuracy.")
    else:
        subject_predictions = defaultdict(list)
        for r in chosen:
            subject_predictions[r.id].append(r)

        logger.info("--- Subject-wise Accuracy ---")
        for subject_id, predictions_for_subject in sorted(subject_predictions.items()):
            correct_for_subject = sum(1 for r in predictions_for_subject if r.is_correct)
            total_for_subject = len(predictions_for_subject)
            if total_for_subject > 0:
                acc_for_subject = correct_for_subject / total_for_subject
                logger.info(f"Subject {subject_id}: {acc_for_subject:.2%}")
        logger.info("----------------------------")

        correct_overall = sum(1 for r in chosen if r.is_correct)
        acc_overall = correct_overall / len(chosen)
        logger.info(f"Overall Accuracy: {acc_overall:.2%}")

    if output:
        logger.info(f"Saving results to {output}...")
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(r) for r in all_results]
        keys = [f.name for f in dataclasses.fields(PredictionResult)]

        with open(output, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                line = []
                for k in keys:
                    val = row.get(k)
                    force = k == "input_sequence"
                    line.append(format_csv_value(val, force_quote=force))
                f.write(",".join(line) + "\n")
        logger.info("Save complete.")


if __name__ == "__main__":
    seed_everything()
    app()
