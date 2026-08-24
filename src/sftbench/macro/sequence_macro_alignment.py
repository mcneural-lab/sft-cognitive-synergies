from __future__ import annotations

import ast

import numpy as np
import pandas as pd


def _parse_categories(cell) -> list[str]:
    """Parse a dataframe cell that may already be a list, or a stringified
    list like `"['Canine', 'Pets']"`. Returns a list of strings (possibly
    empty).
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if isinstance(cell, tuple):
        return [str(x) for x in cell]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return [str(x) for x in val]
            if isinstance(val, (tuple, set)):
                return [str(x) for x in val]
        except Exception:
            pass
        return [s]
    return [str(cell)]


def _apply_maximum_coherence(
    df: pd.DataFrame,
    *,
    id_col: str,
    order_col: str,
    seq_col: str | None,
) -> pd.DataFrame:
    """Resolve ambiguous (multi-category) cells to a single category by
    preferring overlaps with neighboring cells. Modifies `_parsed_categories`
    in-place on a copy of `df` and returns the copy.
    """
    work = df.copy()

    group_cols = [id_col]
    if seq_col and seq_col in work.columns:
        group_cols.append(seq_col)

    for _, g in work.groupby(group_cols, sort=False):
        g = g.sort_values(order_col, kind="mergesort")
        indices = g.index.tolist()
        cats_series = g["_parsed_categories"].tolist()
        n_items = len(cats_series)

        for t in range(n_items):
            current_idx = indices[t]
            src_all = cats_series[t]
            candidates = src_all
            overlap_found = False

            if t < n_items - 1:
                common = list(set(src_all).intersection(set(cats_series[t + 1])))
                if common:
                    candidates = common
                    overlap_found = True

            if not overlap_found and t > 0:
                common_prev = list(set(cats_series[t - 1]).intersection(set(src_all)))
                if common_prev:
                    candidates = common_prev

            selected = [candidates[0]] if candidates else []
            work.at[current_idx, "_parsed_categories"] = selected

    return work


def create_transition_matrices(
    df: pd.DataFrame,
    categories: list[str],
    *,
    id_col: str = "id",
    order_col: str = "rank",
    category_col: str = "category",
    seq_col: str | None = "seq_type",
    include_self_transitions: bool = True,
    fractional_weighting: bool = False,
    maximum_coherence: bool = False,
) -> dict[int, pd.DataFrame]:
    """Compute per-subject transition probability matrices over `categories`.

    Returns
    -------
    matrices
        Mapping of subject_id -> row-normalized TPM as a DataFrame indexed and
        columned by `categories`.
    """
    for required in (category_col, id_col, order_col):
        if required not in df.columns:
            raise ValueError(f"Missing required column '{required}'")

    allowed = list(categories)
    allowed_set = set(allowed)
    cat_to_idx = {c: i for i, c in enumerate(allowed)}
    n = len(allowed)

    work = df.copy()
    work["_parsed_categories"] = df[category_col].apply(_parse_categories)

    if maximum_coherence:
        work = _apply_maximum_coherence(
            work,
            id_col=id_col,
            order_col=order_col,
            seq_col=seq_col,
        )

    group_cols = [id_col]
    if seq_col and seq_col in work.columns:
        group_cols.append(seq_col)

    counts_by_id: dict[int, np.ndarray] = {}

    for group_key, g in work.groupby(group_cols, sort=False):
        subj_id = group_key[0] if isinstance(group_key, tuple) else group_key

        g = g.sort_values(order_col, kind="mergesort")
        cats_series = g["_parsed_categories"].tolist()

        if subj_id not in counts_by_id:
            counts_by_id[subj_id] = np.zeros((n, n), dtype=float)
        counts = counts_by_id[subj_id]

        for t in range(len(cats_series) - 1):
            src_all = [c for c in cats_series[t] if c in allowed_set]
            dst_all = [c for c in cats_series[t + 1] if c in allowed_set]
            if not src_all or not dst_all:
                continue

            pair_count = len(src_all) * len(dst_all)
            if pair_count == 0:
                continue
            weight = 1.0 / pair_count if fractional_weighting else 1.0

            for a in src_all:
                ia = cat_to_idx[a]
                for b in dst_all:
                    if (not include_self_transitions) and (a == b):
                        continue
                    counts[ia, cat_to_idx[b]] += weight

    matrices: dict[int, pd.DataFrame] = {}
    for subj_id, counts in counts_by_id.items():
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)
        matrices[subj_id] = pd.DataFrame(probs, index=allowed, columns=allowed)

    return matrices
