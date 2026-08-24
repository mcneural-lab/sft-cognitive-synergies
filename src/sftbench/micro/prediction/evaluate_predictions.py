import argparse
import logging
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from sftbench.reproducibility import seed_everything

DEFAULT_EXCLUDE_IDS: list[str] = ["51", "71", "198", "199", "285"]

HUMAN_PREDICTED_COL: str = "normalizedPredicted"
HUMAN_CORRECT_COL: str = "is_human_correct"


def _levenshtein_distance(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance between two strings (insertion/deletion/substitution).
    This implementation is iterative and uses O(min(len(a), len(b))) memory.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(b) > len(a):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            subst_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, subst_cost))
        previous = current
    return previous[-1]


def _strip_optional_trailing_s(value: str) -> str:
    """
    Return `value` with a single trailing 's' removed, if present.

    This supports accepting singular/plural variants that differ only by a final 's'
    (e.g., "dog" vs "dogs") after normalization.
    """
    return value[:-1] if value.endswith("s") else value


def _normalize_for_compare(value: object) -> str:
    """
    Normalize an item for robust lexical comparison:
    - cast to string (None/NaN -> empty string)
    - lowercase
    - remove all spaces (not just trim)
    """
    if value is None:
        return ""
    try:
        is_missing = pd.isna(value)
        if isinstance(is_missing, bool) and is_missing:
            return ""
    except Exception:
        pass

    return str(value).lower().replace(" ", "")


def compare_items(
    actual: object,
    predicted: object,
    *,
    mode: str = "exact",
    threshold: float = 0.8,
    lev_threshold: int = 1,
    warn_on_nonperfect_similarity: bool = False,
    warn_on_nonperfect_lev: bool = False,
) -> bool:
    """
    Compare two items using flexible, robust lexical logic.

    Normalization (always applied before comparison):
    - lowercase
    - remove all spaces

    Modes:
    - "strict": perfect lexical match after normalization
    - "exact": 1-1 perfect lexical match after normalization (also accepts trailing 's' plural variants)
    - "similarity": SequenceMatcher ratio >= threshold (default threshold=0.8). Optionally logs warnings for every match
      below perfect (ratio < 1.0) via warn_on_nonperfect_similarity=True.
    - "lev": Levenshtein distance <= lev_threshold (default lev_threshold=1).

    Args:
        actual: Ground-truth value.
        predicted: Model output value.
        mode: "strict", "exact", "similarity", or "lev".
        threshold: Similarity threshold (used only for mode="similarity").
        lev_threshold: Maximum Levenshtein edit distance (used only for mode="lev").
        warn_on_nonperfect_similarity: If True and mode="similarity", log a warning
            for each exemplar comparison with ratio < 1.0 (including below-threshold mismatches).
        warn_on_nonperfect_lev: If True and mode="lev", log a warning for each accepted
            exemplar comparison with distance > 0 (i.e., accepted-but-not-perfect).

    Returns:
        True if the comparison is considered correct, otherwise False.
    """
    a = _normalize_for_compare(actual)
    p = _normalize_for_compare(predicted)

    if mode == "strict":
        return a == p

    if mode == "exact":
        if a == p:
            return True
        return _strip_optional_trailing_s(a) == _strip_optional_trailing_s(p)

    if mode == "similarity":
        if not a and not p:
            return True
        if not a or not p:
            return False

        if _strip_optional_trailing_s(a) == _strip_optional_trailing_s(p):
            return True

        ratio = SequenceMatcher(a=a, b=p).ratio()

        is_match = ratio >= threshold

        if warn_on_nonperfect_similarity and is_match and ratio < 1.0:
            logging.warning(
                "Non-perfect similarity exemplar (accepted): ratio=%.3f threshold=%.3f "
                "actual_raw=%r predicted_raw=%r actual_norm=%r predicted_norm=%r",
                ratio,
                threshold,
                actual,
                predicted,
                a,
                p,
            )

        return is_match

    if mode == "lev":
        if not a and not p:
            return True
        if not a or not p:
            return False

        if _strip_optional_trailing_s(a) == _strip_optional_trailing_s(p):
            return True

        dist = _levenshtein_distance(a, p)
        is_match = dist <= lev_threshold

        if warn_on_nonperfect_lev and is_match and dist > 0:
            logging.warning(
                "Non-perfect Levenshtein exemplar (accepted): dist=%d lev_threshold=%d "
                "actual_raw=%r predicted_raw=%r actual_norm=%r predicted_norm=%r",
                dist,
                lev_threshold,
                actual,
                predicted,
                a,
                p,
            )

        return is_match

    raise ValueError(f"Unknown mode: {mode!r}. Expected 'strict', 'exact', 'similarity', or 'lev'.")


def calculate_subject_accuracy(
    data: str | pd.DataFrame,
    *,
    actual_col: str = "actual_response",
    predicted_col: str = "predicted_response",
    id_col: str = "id",
    mode: str = "exact",
    threshold: float = 0.8,
    lev_threshold: int = 1,
    override_is_correct: bool = False,
    is_correct_col: str = "is_correct",
    output_csv: str | None = None,
    exclude_ids: list[str] | None = None,
    human_predicted_col: str = HUMAN_PREDICTED_COL,
    human_correct_col: str = HUMAN_CORRECT_COL,
) -> pd.Series:
    """
    Calculates subject-wise accuracy from predictions data by comparing two columns.

    Args:
        data: Path to the CSV file or a pandas DataFrame containing the predictions.
        actual_col: Column name for the ground-truth response.
        predicted_col: Column name for the predicted response.
        id_col: Column name for the subject identifier.
        mode: "strict" for perfect lexical match, "exact" to additionally accept
            trailing-'s' variants, "similarity" for >= threshold match, or "lev"
            for Levenshtein distance <= lev_threshold.
        threshold: Similarity threshold for mode="similarity" (default 0.8).
        lev_threshold: Maximum Levenshtein edit distance for mode="lev" (default 1).
        override_is_correct: If True, compute and overwrite `is_correct_col` on the DataFrame.
        is_correct_col: Column name to write correctness into (default: is_correct).
        output_csv: If provided, write an updated CSV (including any overwritten is_correct
            and any computed human correctness column) to this path.
        exclude_ids: Optional list of subject IDs to exclude from evaluation. IDs are
            compared as strings (the `id_col` will be string-cast before filtering).
        human_predicted_col: Column to use as the human prediction (default: normalizedPredicted).
        human_correct_col: Column name to write the human correctness into (default: is_human_correct).

    Returns:
        A pandas Series with the accuracy (mean of computed correctness) for each subject.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data

    required = {id_col, actual_col, predicted_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"The input data must contain columns: {missing}")

    if exclude_ids is None:
        exclude_ids = DEFAULT_EXCLUDE_IDS
    if exclude_ids:
        df = df.copy()
        df[id_col] = df[id_col].astype(str)
        df = df[~df[id_col].isin(set(exclude_ids))]

    correctness = df.apply(
        lambda row: compare_items(
            row[actual_col],
            row[predicted_col],
            mode=mode,
            threshold=threshold,
            lev_threshold=lev_threshold,
            warn_on_nonperfect_similarity=(mode == "similarity"),
            warn_on_nonperfect_lev=(mode == "lev"),
        ),
        axis=1,
    )

    if human_predicted_col in df.columns:
        df[human_correct_col] = df.apply(
            lambda row: compare_items(
                row[actual_col],
                row[human_predicted_col],
                mode=mode,
                threshold=threshold,
                lev_threshold=lev_threshold,
                warn_on_nonperfect_similarity=(mode == "similarity"),
                warn_on_nonperfect_lev=(mode == "lev"),
            ),
            axis=1,
        )

    if override_is_correct:
        df[is_correct_col] = correctness

    accuracy = df.assign(_is_correct=correctness).groupby(id_col)["_is_correct"].mean()
    if not isinstance(accuracy, pd.Series):
        raise TypeError(f"Expected groupby mean accuracy to be a pandas Series, got {type(accuracy)!r}")

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return accuracy


def main() -> None:
    """Command-line entry point for subject-wise prediction accuracy."""
    parser = argparse.ArgumentParser(
        description="Calculate subject-wise accuracy from predictions CSV by comparing two columns."
    )
    parser.add_argument("csv_file", type=str, help="Path to the predictions CSV file.")
    parser.add_argument(
        "--actual-col",
        type=str,
        default="actual_response",
        help="Column name for ground-truth/actual response (default: actual_response).",
    )
    parser.add_argument(
        "--predicted-col",
        type=str,
        default="predicted_response",
        help="Column name for predicted response (default: predicted_response).",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="id",
        help="Column name for subject id (default: id).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["strict", "exact", "similarity", "lev"],
        default="exact",
        help="Comparison mode: strict, exact, similarity, or lev (default: exact).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for mode=similarity (default: 0.8).",
    )
    parser.add_argument(
        "--lev-threshold",
        type=int,
        default=1,
        help="Maximum Levenshtein distance for mode=lev (default: 1).",
    )
    parser.add_argument(
        "--override-is-correct",
        action="store_true",
        help="If set, overwrite the is_correct column based on the configured comparison.",
    )
    parser.add_argument(
        "--is-correct-col",
        type=str,
        default="is_correct",
        help="Column name to write correctness into when overriding (default: is_correct).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="If provided, write an updated CSV (including any overwritten is_correct) to this path.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help=(
            "If set, write updates back to the input CSV in-place (same path). "
            "When `normalizedPredicted` is present, this persists the computed `is_human_correct` column."
        ),
    )
    parser.add_argument(
        "--exclude-ids",
        nargs="*",
        default=DEFAULT_EXCLUDE_IDS,
        help=(f"Subject IDs to exclude from evaluation (space-separated). Default: {DEFAULT_EXCLUDE_IDS}"),
    )
    args = parser.parse_args()

    accuracy = calculate_subject_accuracy(
        args.csv_file,
        actual_col=args.actual_col,
        predicted_col=args.predicted_col,
        id_col=args.id_col,
        mode=args.mode,
        threshold=args.threshold,
        lev_threshold=args.lev_threshold,
        override_is_correct=args.override_is_correct,
        is_correct_col=args.is_correct_col,
        output_csv=args.output_csv,
        exclude_ids=[str(x) for x in (args.exclude_ids or [])],
    )

    df_for_summary = pd.read_csv(args.output_csv) if args.output_csv else pd.read_csv(args.csv_file)

    wrote_inplace = False
    if HUMAN_PREDICTED_COL in df_for_summary.columns and HUMAN_CORRECT_COL not in df_for_summary.columns:
        df_for_summary[HUMAN_CORRECT_COL] = df_for_summary.apply(
            lambda row: compare_items(
                row[args.actual_col],
                row[HUMAN_PREDICTED_COL],
                mode=args.mode,
                threshold=args.threshold,
                lev_threshold=args.lev_threshold,
                warn_on_nonperfect_similarity=(args.mode == "similarity"),
                warn_on_nonperfect_lev=(args.mode == "lev"),
            ),
            axis=1,
        )

        if args.inplace and args.output_csv is None:
            df_for_summary.to_csv(args.csv_file, index=False)
            wrote_inplace = True

    machine_overall = float(accuracy.mean()) if len(accuracy) else float("nan")

    human_overall: float | None = None
    human_subject_accuracy: pd.Series | None = None
    if HUMAN_CORRECT_COL in df_for_summary.columns:
        df_tmp = df_for_summary.copy()
        df_tmp[args.id_col] = df_tmp[args.id_col].astype(str)
        excluded = set([str(x) for x in (args.exclude_ids or [])])
        if excluded:
            df_tmp = df_tmp[~df_tmp[args.id_col].isin(excluded)]
        human_subject_accuracy = df_tmp.groupby(args.id_col)[HUMAN_CORRECT_COL].mean()
        human_overall = float(human_subject_accuracy.mean()) if len(human_subject_accuracy) else float("nan")

    print("Subject-wise Accuracy (machine):")
    print(accuracy)

    print("\nOverall Mean Accuracy Summary")
    print("----------------------------")
    print(f"Machine: {machine_overall:.4f} (subjects={len(accuracy)})")
    if human_overall is None:
        print(f"Human:   n/a (missing column: {HUMAN_PREDICTED_COL!r} -> {HUMAN_CORRECT_COL!r} not computed)")
    else:
        print(f"Human:   {human_overall:.4f} (subjects={len(human_subject_accuracy)})")
    if wrote_inplace:
        print(f"\nWrote updated CSV in-place: {args.csv_file}")


if __name__ == "__main__":
    seed_everything()
    main()
