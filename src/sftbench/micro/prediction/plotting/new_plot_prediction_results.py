"""
Plot Hills et al. next-item prediction evaluations from a directory of CSVs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

DEFAULT_EXCLUDE_IDS: list[str] = ["51", "71", "198", "199", "285"]

CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "animals": "CP zero-shot",
    "animals-baseline": "baseline",
    "animals-baseline-few-shot": "baseline-few-shot",
    "animals-few-shot": "CP-few-shot",
}


def _category_display_name(category: str) -> str:
    return CATEGORY_DISPLAY_NAMES.get(str(category), str(category))


DISPLAY_NAMES: dict[str, str] = {
    "gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "ngram-2": "2-gram",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemini-3-flash-preview": "Gemini-3-Flash",
    "gemini-3-pro-preview": "Gemini-3-Pro",
    "gpt-5.2-2025-12-11": "GPT-5.2",
    "claude-opus-4-5-20251101": "Claude-Opus-4.5",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": "Llama-3.3-70B-Instruct",
}


def _display_name(model: str) -> str:
    return DISPLAY_NAMES.get(str(model), str(model))


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from scipy.stats import friedmanchisquare, wilcoxon

from sftbench import find_project_root
from sftbench.figure_output import save_figure_formats
from sftbench.micro.prediction.plotting import (
    apply_seaborn_theme_from_config,
    ensure_output_dir,
    maybe_rotate_xticks,
    parse_figsize,
)
from sftbench.reproducibility import seed_everything


def _latex_escape(s: str) -> str:
    """
    Minimal LaTeX escaping for table text.
    """
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


app = typer.Typer(help="CLI to plot prediction evaluation CSVs (boxplots).")


@dataclass(frozen=True)
class LoadedData:
    raw: pd.DataFrame
    per_id: pd.DataFrame
    discovered_files: list[Path]


def _default_root() -> Path:
    root = find_project_root()
    if root is None:
        root = Path.cwd()
    return root


def _infer_run_variant_from_filename(source_file: str) -> str:
    """
    Infer a run variant label from the CSV filename.

    Examples:
    - animals-baseline-gemini-2.5-flash.csv -> "baseline"
    - animals-fewshot-gemini-2.5-pro.csv -> "few-shot"
    - animals-gemini-2.5-pro.csv -> "default"
    """
    name = Path(str(source_file)).name.lower()
    if "fewshot" in name or "few-shot" in name or "few_shot" in name:
        return "few-shot"
    if "baseline" in name:
        return "baseline"
    return "default"


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """
    Normalize common boolean representations to pandas boolean dtype.

    Accepts:
    - True/False
    - 1/0
    - "true"/"false" (any case)
    - "yes"/"no"
    """
    if pd.api.types.is_bool_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    # strings / mixed
    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
                "y": True,
                "n": False,
                "t": True,
                "f": False,
            }
        )
    )
    return mapped.fillna(False).astype(bool)


def _read_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__source_file"] = str(path)
    df["__run_variant"] = _infer_run_variant_from_filename(str(path))
    return df


def _discover_csvs(input_dir: Path, glob: str, recursive: bool) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    pattern = f"**/{glob}" if recursive else glob
    files = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
    return files


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        cols = ", ".join(sorted(df.columns))
        raise ValueError(f"Missing required columns: {missing}. Found columns: [{cols}]")


def load_hills_prediction_csvs(
    input_dir: Path,
    *,
    glob: str = "*.csv",
    recursive: bool = False,
    categories: Sequence[str] | None = None,
    model_regex: str | None = None,
    chosen_only: bool = True,
    exclude_ids: list[str] | None = None,
) -> LoadedData:
    """
    Load and prepare Hills et al. prediction CSVs from a directory.

    Notes:
    - Adds `__source_file` and `__run_variant` derived from the filename.
    - Builds a `model_label` that combines the CSV `model` with the inferred run variant,
      so baseline / few-shot are plotted distinctly even when `model` doesn't encode them.

    Returns both:
    - raw row-level data
    - aggregated per-id per-model accuracy suitable for boxplots
    """
    files = _discover_csvs(input_dir, glob=glob, recursive=recursive)
    if not files:
        raise ValueError(f"No CSV files matched in {input_dir} (glob={glob!r}, recursive={recursive})")

    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            dfs.append(_read_one_csv(f))
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {f} ({e})") from e

    df = cast(pd.DataFrame, pd.concat(dfs, ignore_index=True))

    # Normalize column names (some pipelines might use data-category).
    if "data-category" in df.columns and "category" not in df.columns:
        df = df.rename(columns={"data-category": "category"})

    _validate_columns(df, required=["id", "model", "category", "is_correct"])

    if "__run_variant" not in df.columns:
        df["__run_variant"] = "default"

    if categories is not None:
        cats = [str(c).strip() for c in categories if str(c).strip() != ""]
        if cats:
            df = df[df["category"].astype(str).isin(cats)]

    if model_regex is not None:
        model_s = cast(pd.Series, df["model"])
        mask = model_s.astype(str).str.contains(model_regex, regex=True, na=False)
        df = df[mask]

    if exclude_ids:
        exclude_list_norm = [str(x) for x in exclude_ids]
        id_s = cast(pd.Series, df["id"])
        df = df[~id_s.astype(str).isin(exclude_list_norm)]

    if chosen_only and "is_chosen" in df.columns:
        chosen_s = cast(pd.Series, df["is_chosen"])
        df["is_chosen"] = _coerce_bool_series(chosen_s)
        df = df[df["is_chosen"]]

    correct_s = cast(pd.Series, df["is_correct"])
    df["is_correct"] = _coerce_bool_series(correct_s)

    # Aggregate per participant (id) accuracy per model
    group_cols = ["id", "model", "__run_variant", "category"]
    per_id = cast(
        pd.DataFrame,
        df.groupby(group_cols, sort=False)["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "correct_fraction"}),
    )

    return LoadedData(raw=df, per_id=per_id, discovered_files=files)


def _add_adjacent_one_sided_wilcoxon_annotations(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    order: list[str],
    id_col: str,
    results_sink: list[dict[str, object]] | None = None,
    facet_label: str | None = None,
    alpha: float = 0.05,
) -> None:
    """
    Adjacent-pair one-sided Wilcoxon signed-rank tests with star annotations (BLEU-plot style).

    - Test: Wilcoxon signed-rank (paired), one-sided
    - Alternative: right > left
    - Display: annotate ONLY significant adjacent pairs using statannotations-style "star" labels.

    Also appends a record per attempted comparison into `results_sink` (for CSV export).
    """
    if len(order) < 2:
        return

    wide = (
        data[[id_col, x, y]]
        .dropna()
        .assign(**{x: data[x].astype(str)})
        .pivot_table(index=id_col, columns=x, values=y, aggfunc="mean")
    )

    cols = [c for c in order if c in wide.columns]
    if len(cols) < 2:
        return

    wide = wide[cols].dropna(axis=0, how="any")
    if wide.empty:
        return

    try:
        from statannotations.Annotator import Annotator
    except Exception:
        Annotator = None  # type: ignore[assignment]

    pvalues: list[float] = []
    pairs_display: list[tuple[str, str]] = []

    for i in range(len(cols) - 1):
        left = cols[i]
        right = cols[i + 1]

        left_v = wide[left].to_numpy()
        right_v = wide[right].to_numpy()

        n_paired = int(len(left_v))
        left_mean = float(left_v.mean()) if n_paired else float("nan")
        right_mean = float(right_v.mean()) if n_paired else float("nan")
        mean_diff = float((right_v - left_v).mean()) if n_paired else float("nan")

        record: dict[str, object] = {
            "facet": facet_label or "",
            "x": x,
            "left": str(left),
            "right": str(right),
            "n_paired": n_paired,
            "mean_left": left_mean,
            "mean_right": right_mean,
            "mean_diff_right_minus_left": mean_diff,
            "test": "wilcoxon_signed_rank_one_sided_greater",
            "statistic": None,
            "p_value": None,
            "p_one_sided": None,
            "p_corrected": None,
            "correction": "",
            "note": "",
        }

        if not (right_mean > left_mean):
            record["note"] = "Skipped (right_mean <= left_mean)"
            if results_sink is not None:
                results_sink.append(record)
            continue

        try:
            res = wilcoxon(right_v, left_v, alternative="greater", zero_method="wilcox")
            record["statistic"] = float(res.statistic) if res.statistic is not None else None
            record["p_one_sided"] = float(res.pvalue)
            record["p_value"] = float(res.pvalue)
        except Exception as e:
            record["note"] = f"Error: {e}"
            if results_sink is not None:
                results_sink.append(record)
            continue

        if results_sink is not None:
            results_sink.append(record)

        pv = float(record["p_one_sided"])  # type: ignore[arg-type]
        if pv < alpha:
            pvalues.append(pv)
            pairs_display.append((str(left), str(right)))

    if not pairs_display:
        return

    if Annotator is None:
        return

    pairs_for_plot = pairs_display
    order_for_plot = [str(o) for o in order]
    data_for_plot = data
    x_for_plot = x
    if x == "model":
        data_for_plot = data.assign(model_display=data["model"].astype(str).map(_display_name))
        x_for_plot = "model_display"
        order_for_plot = [_display_name(o) for o in order_for_plot]
        pairs_for_plot = [(_display_name(a), _display_name(b)) for (a, b) in pairs_for_plot]

    annotator = Annotator(
        ax=ax,
        pairs=pairs_for_plot,
        data=data_for_plot,
        x=x_for_plot,
        y=y,
        order=order_for_plot,
    )
    annotator.configure(test=None, text_format="star", loc="inside", verbose=0)
    annotator.set_pvalues_and_annotate(pvalues)


def _apply_bonferroni(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    m = len(pvalues)
    return [min(p * m, 1.0) for p in pvalues]


def _apply_holm_bonferroni(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvalues[int(idx)]
        if adj < prev:
            adj = prev
        prev = adj
        adjusted[int(idx)] = min(adj, 1.0)
    return adjusted


def _correct_pvalues(pvalues: list[float], *, correction: str) -> list[float]:
    method = correction.strip().lower()
    if method in {"holm", "holm-bonferroni", "holm_bonferroni"}:
        return _apply_holm_bonferroni(pvalues)
    if method in {"bonferroni", "bonf"}:
        return _apply_bonferroni(pvalues)
    return pvalues


def _add_within_model_friedman_and_posthoc_annotations(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    id_col: str,
    category_col: str,
    value_col: str,
    category_order: list[str],
    category_order_display: list[str],
    results_sink: list[dict[str, object]] | None = None,
    facet_label: str | None = None,
    alpha: float = 0.05,
    correction: str = "holm",
) -> None:
    if len(category_order) < 3:
        return

    wide = (
        data[[id_col, category_col, value_col]]
        .dropna()
        .assign(**{category_col: data[category_col].astype(str)})
        .pivot_table(index=id_col, columns=category_col, values=value_col, aggfunc="mean")
    )

    cols = [c for c in category_order if c in wide.columns]
    if len(cols) < 3:
        return

    wide = wide[cols].dropna(axis=0, how="any")
    if wide.empty:
        return

    friedman_note = ""
    friedman_stat: float | None = None
    friedman_p: float | None = None
    try:
        res = friedmanchisquare(*[wide[c].to_numpy() for c in cols])
        friedman_stat = float(res.statistic) if res.statistic is not None else None
        friedman_p = float(res.pvalue)
    except Exception as e:
        friedman_note = f"Error: {e}"

    if results_sink is not None:
        results_sink.append(
            {
                "facet": facet_label or "",
                "x": category_col,
                "left": "",
                "right": "",
                "n_paired": int(wide.shape[0]),
                "mean_left": float("nan"),
                "mean_right": float("nan"),
                "mean_diff_right_minus_left": float("nan"),
                "test": "friedmanchisquare",
                "statistic": friedman_stat,
                "p_value": friedman_p,
                "p_one_sided": None,
                "p_corrected": None,
                "correction": "",
                "note": friedman_note,
            }
        )

    if friedman_p is None or friedman_p >= alpha:
        return

    pairs_raw = [
        ("animals", "animals-baseline"),
        ("animals", "animals-baseline-few-shot"),
        ("animals-few-shot", "animals-baseline"),
        ("animals-few-shot", "animals-baseline-few-shot"),
    ]
    pairs_raw = [p for p in pairs_raw if p[0] in cols and p[1] in cols]
    if not pairs_raw:
        return

    pvalues: list[float] = []
    pair_records: list[dict[str, object]] = []

    for left, right in pairs_raw:
        left_v = wide[left].to_numpy()
        right_v = wide[right].to_numpy()
        n_paired = int(len(left_v))
        left_mean = float(left_v.mean()) if n_paired else float("nan")
        right_mean = float(right_v.mean()) if n_paired else float("nan")
        mean_diff = float((right_v - left_v).mean()) if n_paired else float("nan")

        record: dict[str, object] = {
            "facet": facet_label or "",
            "x": category_col,
            "left": str(left),
            "right": str(right),
            "n_paired": n_paired,
            "mean_left": left_mean,
            "mean_right": right_mean,
            "mean_diff_right_minus_left": mean_diff,
            "test": "wilcoxon_signed_rank_two_sided",
            "statistic": None,
            "p_value": None,
            "p_one_sided": None,
            "p_corrected": None,
            "correction": correction,
            "note": "",
        }

        try:
            res = wilcoxon(right_v, left_v, alternative="two-sided", zero_method="wilcox")
            record["statistic"] = float(res.statistic) if res.statistic is not None else None
            record["p_value"] = float(res.pvalue)
            pvalues.append(float(res.pvalue))
        except Exception as e:
            record["note"] = f"Error: {e}"

        pair_records.append(record)

    if not pvalues:
        if results_sink is not None:
            results_sink.extend(pair_records)
        return

    corrected = _correct_pvalues(pvalues, correction=correction)
    display_map = {c: _category_display_name(c) for c in cols}

    pvalues_for_plot: list[float] = []
    pairs_for_plot: list[tuple[str, str]] = []

    p_index = 0
    for record in pair_records:
        if record.get("p_value") is None:
            if results_sink is not None:
                results_sink.append(record)
            continue
        p_adj = float(corrected[p_index])
        record["p_corrected"] = p_adj
        record["correction"] = correction
        if results_sink is not None:
            results_sink.append(record)

        if p_adj < alpha:
            left = str(record["left"])
            right = str(record["right"])
            pairs_for_plot.append((display_map.get(left, left), display_map.get(right, right)))
            pvalues_for_plot.append(p_adj)
        p_index += 1

    if not pairs_for_plot:
        return

    try:
        from statannotations.Annotator import Annotator
    except Exception:
        Annotator = None  # type: ignore[assignment]

    if Annotator is None:
        return

    annotator = Annotator(
        ax=ax,
        pairs=pairs_for_plot,
        data=data,
        x="category_display",
        y=value_col,
        order=category_order_display,
    )
    annotator.configure(test=None, text_format="star", loc="inside", verbose=0)
    annotator.set_pvalues_and_annotate(pvalues_for_plot)


def _add_within_model_friedman_and_posthoc_annotations_dodged(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    id_col: str,
    model_col: str,
    model_display_col: str,
    model_order: list[str],
    model_order_display: list[str],
    category_col: str,
    category_display_col: str,
    category_order: list[str],
    category_order_display: list[str],
    value_col: str,
    results_sink: list[dict[str, object]] | None = None,
    alpha: float = 0.05,
    correction: str = "holm",
) -> None:
    if len(category_order) < 3 or not model_order:
        return

    try:
        from statannotations.Annotator import Annotator
    except Exception:
        Annotator = None  # type: ignore[assignment]

    pairs_for_plot: list[tuple[tuple[str, str], tuple[str, str]]] = []
    pvalues_for_plot: list[float] = []

    for model in model_order:
        sub = data[data[model_col].astype(str) == str(model)]
        if sub.empty:
            continue

        wide = (
            sub[[id_col, category_col, value_col]]
            .dropna()
            .assign(**{category_col: sub[category_col].astype(str)})
            .pivot_table(index=id_col, columns=category_col, values=value_col, aggfunc="mean")
        )

        cols = [c for c in category_order if c in wide.columns]
        if len(cols) < 3:
            continue

        wide = wide[cols].dropna(axis=0, how="any")
        if wide.empty:
            continue

        friedman_note = ""
        friedman_stat: float | None = None
        friedman_p: float | None = None
        try:
            res = friedmanchisquare(*[wide[c].to_numpy() for c in cols])
            friedman_stat = float(res.statistic) if res.statistic is not None else None
            friedman_p = float(res.pvalue)
        except Exception as e:
            friedman_note = f"Error: {e}"

        if results_sink is not None:
            results_sink.append(
                {
                    "facet": f"model={model}",
                    "x": category_col,
                    "left": "",
                    "right": "",
                    "n_paired": int(wide.shape[0]),
                    "mean_left": float("nan"),
                    "mean_right": float("nan"),
                    "mean_diff_right_minus_left": float("nan"),
                    "test": "friedmanchisquare",
                    "statistic": friedman_stat,
                    "p_value": friedman_p,
                    "p_one_sided": None,
                    "p_corrected": None,
                    "correction": "",
                    "note": friedman_note,
                }
            )

        if friedman_p is None or friedman_p >= alpha:
            continue

        pairs_raw = [
            ("animals", "animals-baseline"),
            ("animals", "animals-baseline-few-shot"),
            ("animals-few-shot", "animals-baseline"),
            ("animals-few-shot", "animals-baseline-few-shot"),
        ]
        pairs_raw = [p for p in pairs_raw if p[0] in cols and p[1] in cols]
        if not pairs_raw:
            continue

        pvalues: list[float] = []
        pair_records: list[dict[str, object]] = []

        for left, right in pairs_raw:
            left_v = wide[left].to_numpy()
            right_v = wide[right].to_numpy()
            n_paired = int(len(left_v))
            left_mean = float(left_v.mean()) if n_paired else float("nan")
            right_mean = float(right_v.mean()) if n_paired else float("nan")
            mean_diff = float((right_v - left_v).mean()) if n_paired else float("nan")

            record: dict[str, object] = {
                "facet": f"model={model}",
                "x": category_col,
                "left": str(left),
                "right": str(right),
                "n_paired": n_paired,
                "mean_left": left_mean,
                "mean_right": right_mean,
                "mean_diff_right_minus_left": mean_diff,
                "test": "wilcoxon_signed_rank_two_sided",
                "statistic": None,
                "p_value": None,
                "p_one_sided": None,
                "p_corrected": None,
                "correction": correction,
                "note": "",
            }

            try:
                res = wilcoxon(right_v, left_v, alternative="two-sided", zero_method="wilcox")
                record["statistic"] = float(res.statistic) if res.statistic is not None else None
                record["p_value"] = float(res.pvalue)
                pvalues.append(float(res.pvalue))
            except Exception as e:
                record["note"] = f"Error: {e}"

            pair_records.append(record)

        if not pvalues:
            if results_sink is not None:
                results_sink.extend(pair_records)
            continue

        corrected = _correct_pvalues(pvalues, correction=correction)
        display_map = {c: _category_display_name(c) for c in cols}

        p_index = 0
        for record in pair_records:
            if record.get("p_value") is None:
                if results_sink is not None:
                    results_sink.append(record)
                continue
            p_adj = float(corrected[p_index])
            record["p_corrected"] = p_adj
            record["correction"] = correction
            if results_sink is not None:
                results_sink.append(record)

            if p_adj < alpha:
                left = str(record["left"])
                right = str(record["right"])
                pairs_for_plot.append(
                    (
                        (model_order_display[model_order.index(str(model))], display_map.get(left, left)),
                        (model_order_display[model_order.index(str(model))], display_map.get(right, right)),
                    )
                )
                pvalues_for_plot.append(p_adj)
            p_index += 1

    if not pairs_for_plot or Annotator is None:
        return

    annotator = Annotator(
        ax=ax,
        pairs=pairs_for_plot,
        data=data,
        x=model_display_col,
        y=value_col,
        hue=category_display_col,
        order=model_order_display,
        hue_order=category_order_display,
    )
    annotator.configure(test=None, text_format="star", loc="inside", verbose=0)
    annotator.set_pvalues_and_annotate(pvalues_for_plot)


@app.command()
def plot(
    input_dir: Path = typer.Option(
        _default_root() / "results" / "hills" / "next-exemplar" / "cognitive_prompting",
        exists=False,
        help="Directory containing prediction CSVs.",
    ),
    glob: str = typer.Option("*.csv", help="Glob pattern to match CSV files within input-dir."),
    recursive: bool = typer.Option(False, help="Whether to search input-dir recursively."),
    config: Path = typer.Option(
        _default_root() / "configs" / "plotting.toml",
        help="Path to plotting configuration TOML.",
    ),
    output_dir: Path = typer.Option(Path("plots/hills"), help="Directory to save generated plots."),
    category: list[str] = typer.Option(
        ["animals"],
        help="Category filter (repeatable): --category animals --category animals-baseline-few-shot. Pass empty string to disable filtering.",
    ),
    model_regex: str | None = typer.Option(
        None,
        help="Regex to filter model names. Leave unset to include all models (including baseline and few-shot).",
    ),
    chosen_only: bool = typer.Option(True, help="If `is_chosen` exists, filter to chosen rows only."),
    exclude_default_ids: bool = typer.Option(
        False,
        help="Exclude default problematic ids (DEFAULT_EXCLUDE_IDS).",
    ),
    exclude_ids: list[str] = typer.Option(
        [],
        help="Additional ids to exclude (repeatable: --exclude-ids 51 --exclude-ids 71).",
    ),
    order: str | None = typer.Option(
        None,
        help="Comma-separated explicit model order for x-axis. If omitted, a reasonable default is used.",
    ),
    annotate: bool = typer.Option(True, help="Add within-facet adjacent significance annotations."),
    palette: str = typer.Option("Set2", help="Seaborn palette name (paper-friendly defaults: Set2, tab10)."),
    ylim: float | None = typer.Option(None, help="Upper y-limit for accuracy plot (leave unset for auto)."),
    figsize: str = typer.Option("11,6", help="Figure size as 'W,H' in inches."),
    rotate_xticks: int = typer.Option(30, help="Rotate x tick labels by this angle."),
    category_split: bool = typer.Option(
        True,
        help="When multiple categories are present, plot them side-by-side with hue=category (dodged).",
    ),
    facet_by_category: bool = typer.Option(
        False,
        help="Facet into subplots per category instead of using hue.",
    ),
    facet_by_model: bool = typer.Option(
        False,
        help="Facet into subplots per model with hue=category (best for comparing categories within each model). Overrides --facet-by-category and --category-split.",
    ),
    show: bool = typer.Option(True, help="Show plots interactively."),
    formats: str | None = typer.Option(
        None,
        "--formats",
        help="Comma-separated output formats for figures: png, pdf, svg. Defaults to png,pdf.",
    ),
):
    """
    Load Hills evaluation CSVs and plot per-participant accuracy as model boxplots.

    Output:
    - `hills_model_accuracy_boxplot.png`
    - `hills_model_accuracy_boxplot.pdf`
    - `hills_model_accuracy_summary.csv`
    """
    apply_seaborn_theme_from_config(config)

    ensure_output_dir(output_dir)

    cat_filter: list[str] | None
    if len(category) == 1 and str(category[0]).strip() == "":
        cat_filter = None
    else:
        cat_filter = [str(c).strip() for c in category if str(c).strip() != ""]
        cat_filter = cat_filter if cat_filter else None

    exclude_list: list[str] = []
    if exclude_default_ids:
        exclude_list.extend(DEFAULT_EXCLUDE_IDS)
    if exclude_ids:
        exclude_list.extend(exclude_ids)

    loaded = load_hills_prediction_csvs(
        input_dir,
        glob=glob,
        recursive=recursive,
        categories=cat_filter,
        model_regex=model_regex,
        chosen_only=chosen_only,
        exclude_ids=exclude_list if exclude_list else None,
    )

    df = loaded.raw
    per_id = loaded.per_id

    per_id = per_id.assign(
        model_display=per_id["model"].astype(str).map(_display_name),
        category_display=per_id["category"].astype(str).map(_category_display_name),
    )

    models = per_id["model"].astype(str).unique().tolist()
    categories_present = sorted(per_id["category"].astype(str).unique().tolist())

    per_category_model_order: dict[str, list[str]] = {}
    for cat in categories_present:
        sub = per_id[per_id["category"].astype(str) == cat]
        m = cast(pd.Series, sub.groupby("model", sort=False)["correct_fraction"].mean()).sort_values(ascending=True)
        per_category_model_order[str(cat)] = [str(x) for x in cast(pd.Index, m.index).tolist()]

    macro = (
        per_id.groupby(["category", "model"], sort=False)["correct_fraction"]
        .mean()
        .reset_index()
        .groupby("model", sort=False)["correct_fraction"]
        .mean()
    )
    macro = cast(pd.Series, macro).sort_values(ascending=True)
    perf_order_global = [str(m) for m in cast(pd.Index, macro.index).tolist()]

    if order is not None:
        explicit = [m.strip() for m in order.split(",") if m.strip()]
        in_data = [m for m in explicit if m in models]
        remaining = [m for m in perf_order_global if m not in set(in_data)]
        model_order = in_data + remaining
    else:
        model_order = perf_order_global if perf_order_global else sorted(models)

    fig_w, fig_h = parse_figsize(figsize, default=(7.2, 4.2))

    title = "Cognitive Prompting Evaluation\nNext-exemplar Prediction"

    within_facet_stats: list[dict[str, object]] = []

    saved_plot_paths: tuple[Path, ...]

    if facet_by_model:
        cat_means_global = per_id.groupby("category", sort=False)["correct_fraction"].mean().sort_values(ascending=True)
        cats_global = [str(c) for c in cast(pd.Index, cat_means_global.index).tolist()]
        cats_global_disp = [_category_display_name(c) for c in cats_global]

        model_order_local = model_order

        n_models = max(1, len(model_order_local))
        fig, axes = plt.subplots(1, n_models, figsize=(fig_w * n_models, fig_h), sharey=True)
        if n_models == 1:
            axes = [axes]

        cat_palette = dict(
            zip(cats_global_disp, sns.color_palette(palette, n_colors=len(cats_global_disp)), strict=False)
        )

        for ax, m in zip(axes, model_order_local, strict=False):
            sub = per_id[per_id["model"].astype(str) == str(m)].copy()
            cat_means_local = sub.groupby("category", sort=False)["correct_fraction"].mean().sort_values(ascending=True)
            cats_local = [str(c) for c in cast(pd.Index, cat_means_local.index).tolist()]
            cats_local_disp = [_category_display_name(c) for c in cats_local]

            sns.boxplot(
                data=sub,
                x="category_display",
                y="correct_fraction",
                order=cats_local_disp,
                hue="category_display",
                hue_order=cats_local_disp,
                palette=cat_palette,
                ax=ax,
                dodge=False,
                width=0.6,
                showfliers=False,
                boxprops={"linewidth": 1.0, "edgecolor": "0.2"},
                whiskerprops={"linewidth": 1.0, "color": "0.2"},
                capprops={"linewidth": 1.0, "color": "0.2"},
                medianprops={"linewidth": 1.3, "color": "0.1"},
            )

            ax.set_title(_display_name(str(m)))
            ax.set_xlabel("Category")
            ax.set_ylabel("Mean Accuracy" if ax is axes[0] else "")
            ax.grid(axis="y", color="0.9", linewidth=0.8)
            maybe_rotate_xticks(ax, rotate_xticks)

            if annotate and len(cats_local) >= 3:
                _add_within_model_friedman_and_posthoc_annotations(
                    ax,
                    sub,
                    id_col="id",
                    category_col="category",
                    value_col="correct_fraction",
                    category_order=cats_local,
                    category_order_display=cats_local_disp,
                    results_sink=within_facet_stats,
                    facet_label=f"model={m}",
                    alpha=0.05,
                    correction="holm",
                )
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        handles, labels = axes[0].get_legend_handles_labels()
        handle_by_label = {lab: h for h, lab in zip(handles, labels, strict=False)}
        legend_labels = [c for c in cats_global_disp if c in handle_by_label]
        legend_handles = [handle_by_label[c] for c in legend_labels]
        fig.legend(
            legend_handles,
            legend_labels,
            title="Category",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )
        if ylim is not None:
            axes[0].set_ylim(-0.005, float(ylim))
        else:
            ymax = float(per_id["correct_fraction"].max()) if len(per_id) else 0.0
            axes[0].set_ylim(-0.005, min(1.0, ymax * 1.10 + 0.01))

        fig.suptitle(title, y=1.2)
        fig.tight_layout()

        saved_plot_paths = save_figure_formats(
            fig,
            output_dir / "hills_model_accuracy_boxplot_faceted_by_model.png",
            formats=formats,
            default=("png", "pdf"),
            tight=True,
        )
    elif facet_by_category:
        cat_means = per_id.groupby("category", sort=False)["correct_fraction"].mean().sort_values(ascending=True)
        cats = [str(c) for c in cast(pd.Index, cat_means.index).tolist()]

        n = max(1, len(cats))
        fig, axes = plt.subplots(1, n, figsize=(fig_w * n, fig_h), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, cat in zip(axes, cats, strict=False):
            sub = per_id[per_id["category"].astype(str) == cat].copy()

            cat_order = per_category_model_order.get(str(cat), model_order)

            cat_order_disp = [_display_name(m) for m in cat_order]
            model_palette_disp = dict(
                zip(cat_order_disp, sns.color_palette(palette, n_colors=len(cat_order_disp)), strict=False)
            )

            sns.boxplot(
                data=sub.assign(model_display=sub["model"].astype(str).map(_display_name)),
                x="model_display",
                y="correct_fraction",
                order=cat_order_disp,
                hue="model_display",
                palette=model_palette_disp,
                ax=ax,
                dodge=False,
                width=0.6,
                showfliers=False,
                boxprops={"linewidth": 1.0, "edgecolor": "0.2"},
                whiskerprops={"linewidth": 1.0, "color": "0.2"},
                capprops={"linewidth": 1.0, "color": "0.2"},
                medianprops={"linewidth": 1.3, "color": "0.1"},
            )
            ax.set_title(str(cat))
            ax.set_xlabel("Model")
            ax.set_ylabel("Mean Accuracy" if ax is axes[0] else "")
            ax.grid(axis="y", color="0.9", linewidth=0.8)
            ax.get_legend().remove() if ax.get_legend() is not None else None
            maybe_rotate_xticks(ax, rotate_xticks)

            if annotate and len(cat_order) >= 2:
                _add_adjacent_one_sided_wilcoxon_annotations(
                    ax,
                    sub,
                    x="model",
                    y="correct_fraction",
                    order=cat_order,
                    id_col="id",
                    results_sink=within_facet_stats,
                    facet_label=f"category={cat}",
                )

        if ylim is not None:
            axes[0].set_ylim(-0.005, float(ylim))
        else:
            ymax = float(per_id["correct_fraction"].max()) if len(per_id) else 0.0
            axes[0].set_ylim(-0.005, min(1.0, ymax * 1.10 + 0.01))

        fig.suptitle(title, y=1.2)
        fig.tight_layout()

        saved_plot_paths = save_figure_formats(
            fig,
            output_dir / "hills_model_accuracy_boxplot_faceted_by_category.png",
            formats=formats,
            default=("png", "pdf"),
            tight=True,
        )
    else:
        plt.figure(figsize=(fig_w, fig_h))

        present_categories = sorted(per_id["category"].astype(str).unique().tolist())
        multi_category = len(present_categories) > 1

        if category_split and multi_category:
            cat_means_global = (
                per_id.groupby("category", sort=False)["correct_fraction"].mean().sort_values(ascending=True)
            )
            hue_order = [str(c) for c in cast(pd.Index, cat_means_global.index).tolist()]
            for c in present_categories:
                if str(c) not in set(hue_order):
                    hue_order.append(str(c))

            hue_order_disp = [_category_display_name(c) for c in hue_order]
            cat_palette = dict(
                zip(hue_order_disp, sns.color_palette(palette, n_colors=len(hue_order_disp)), strict=False)
            )

            ax = sns.boxplot(
                data=per_id,
                x="model_display",
                y="correct_fraction",
                order=[_display_name(m) for m in model_order],
                hue="category_display",
                hue_order=hue_order_disp,
                palette=cat_palette,
                dodge=True,
                width=0.62,
                showfliers=False,
                boxprops={"linewidth": 1.0, "edgecolor": "0.2"},
                whiskerprops={"linewidth": 1.0, "color": "0.2"},
                capprops={"linewidth": 1.0, "color": "0.2"},
                medianprops={"linewidth": 1.3, "color": "0.1"},
            )
            ax.set_title(title)
            ax.set_xlabel("Model")
            ax.set_ylabel("Mean Accuracy")
            ax.grid(axis="y", color="0.9", linewidth=0.8)
            maybe_rotate_xticks(ax, rotate_xticks)

            leg = ax.legend(title="Category", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            if leg is not None:
                for t in leg.get_texts():
                    current = t.get_fontsize()
                    try:
                        t.set_fontsize(float(current) * 0.95)
                    except Exception:
                        pass

            if annotate and len(hue_order) >= 3:
                _add_within_model_friedman_and_posthoc_annotations_dodged(
                    ax,
                    per_id,
                    id_col="id",
                    model_col="model",
                    model_display_col="model_display",
                    model_order=model_order,
                    model_order_display=[_display_name(m) for m in model_order],
                    category_col="category",
                    category_display_col="category_display",
                    category_order=hue_order,
                    category_order_display=hue_order_disp,
                    value_col="correct_fraction",
                    results_sink=within_facet_stats,
                    alpha=0.05,
                    correction="holm",
                )
        else:
            model_order_disp = [_display_name(m) for m in model_order]
            model_palette_disp = dict(
                zip(model_order_disp, sns.color_palette(palette, n_colors=len(model_order_disp)), strict=False)
            )
            ax = sns.boxplot(
                data=per_id,
                x="model_display",
                y="correct_fraction",
                order=model_order_disp,
                hue="model_display",
                palette=model_palette_disp,
                dodge=False,
                width=0.52,
                showfliers=False,
                boxprops={"linewidth": 1.0, "edgecolor": "0.2"},
                whiskerprops={"linewidth": 1.0, "color": "0.2"},
                capprops={"linewidth": 1.0, "color": "0.2"},
                medianprops={"linewidth": 1.3, "color": "0.1"},
            )
            ax.set_title(title, pad=6)
            ax.set_xlabel("Model", labelpad=4)
            ax.set_ylabel("Mean Accuracy", labelpad=4)
            ax.grid(axis="y", color="0.9", linewidth=0.8)

            maybe_rotate_xticks(ax, min(int(rotate_xticks), 20))

            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

            if annotate and len(model_order) >= 2:
                _add_adjacent_one_sided_wilcoxon_annotations(
                    ax,
                    per_id,
                    x="model",
                    y="correct_fraction",
                    order=model_order,
                    id_col="id",
                    results_sink=within_facet_stats,
                    facet_label="category=all",
                )

            plt.tight_layout(pad=0.5)
            plt.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.90)

        if ylim is not None:
            ax.set_ylim(-0.005, float(ylim))
        else:
            ymax = float(per_id["correct_fraction"].max()) if len(per_id) else 0.0
            ax.set_ylim(-0.005, min(1.0, ymax * 1.10 + 0.01))

        saved_plot_paths = save_figure_formats(
            None,
            output_dir / "hills_model_accuracy_boxplot.png",
            formats=formats,
            default=("png", "pdf"),
            tight=True,
        )

    summary = (
        per_id.groupby(["model", "category"], sort=False)["correct_fraction"]
        .agg(n="count", mean="mean", std="std")
        .reset_index()
    )
    summary_csv = output_dir / "hills_model_accuracy_summary.csv"
    summary.to_csv(summary_csv, index=False)

    rng = np.random.default_rng(0)
    n_boot = 10_000
    ci_rows: list[dict[str, object]] = []
    for (m, cat), sub in per_id.groupby(["model", "category"], sort=False):
        vals = sub["correct_fraction"].dropna().to_numpy(dtype=float)
        n = int(vals.size)
        if n == 0:
            ci_rows.append(
                {
                    "model": str(m),
                    "model_display": _display_name(str(m)),
                    "category": str(cat),
                    "category_display": _category_display_name(str(cat)),
                    "n": 0,
                    "mean": float("nan"),
                    "ci95_lo": float("nan"),
                    "ci95_hi": float("nan"),
                }
            )
            continue

        mean = float(np.mean(vals))
        boot_means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sample = rng.choice(vals, size=n, replace=True)
            boot_means[i] = float(np.mean(sample))

        lo, hi = np.quantile(boot_means, [0.025, 0.975])
        ci_rows.append(
            {
                "model": str(m),
                "model_display": _display_name(str(m)),
                "category": str(cat),
                "category_display": _category_display_name(str(cat)),
                "n": n,
                "mean": mean,
                "ci95_lo": float(lo),
                "ci95_hi": float(hi),
            }
        )

    ci_df = pd.DataFrame(ci_rows)
    ci_csv = output_dir / "hills_model_accuracy_mean_ci95.csv"
    ci_df.to_csv(ci_csv, index=False)

    # Print LaTeX table (per category, show mean [CI], bold best model(s) per category).
    cats_disp_order = (
        ci_df.groupby("category_display", sort=False)["mean"].mean().sort_values(ascending=True).index.tolist()
    )
    models_disp_order = (
        ci_df.groupby("model_display", sort=False)["mean"].mean().sort_values(ascending=False).index.tolist()
    )

    ci_pivot = ci_df.pivot(index="model_display", columns="category_display", values=["mean", "ci95_lo", "ci95_hi"])
    ci_pivot = ci_pivot.reindex(index=models_disp_order, columns=cats_disp_order, level=1)

    best_by_cat: dict[str, float] = {}
    for c in cats_disp_order:
        try:
            best_by_cat[c] = float(ci_df.loc[ci_df["category_display"] == c, "mean"].max())
        except Exception:
            best_by_cat[c] = float("nan")

    def _fmt_cell(m: str, c: str) -> str:
        try:
            mean = float(ci_pivot.loc[m, ("mean", c)])
            lo = float(ci_pivot.loc[m, ("ci95_lo", c)])
            hi = float(ci_pivot.loc[m, ("ci95_hi", c)])
        except Exception:
            return ""
        cell = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        best = best_by_cat.get(c, float("nan"))
        if np.isfinite(best) and abs(mean - best) <= 1e-12:
            return f"\\textbf{{{cell}}}"
        return cell

    latex_lines: list[str] = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    colspec = "l" + ("c" * len(cats_disp_order))
    latex_lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    latex_lines.append("\\toprule")
    header = "Model"
    for c in cats_disp_order:
        header += " & " + _latex_escape(str(c))
    header += " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    for m in models_disp_order:
        row = _latex_escape(str(m))
        for c in cats_disp_order:
            row += " & " + _fmt_cell(str(m), str(c))
        row += " \\\\"
        latex_lines.append(row)
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(
        "\\caption{Mean per-id accuracy with 95\\% bootstrap confidence intervals. Best model(s) per category are bolded.}"
    )
    latex_lines.append("\\label{tab:hills_accuracy_ci}")
    latex_lines.append("\\end{table}")

    typer.echo("\nLaTeX table (mean [95% CI], best bolded):\n")
    typer.echo("\n".join(latex_lines))
    latex_path = output_dir / "hills_model_accuracy_mean_ci95_table.tex"
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    # Save + print publication-ready within-facet statistics table (adjacent one-sided Wilcoxon tests)
    if within_facet_stats:
        stats_df = pd.DataFrame(within_facet_stats)
        stats_csv = output_dir / "hills_within_facet_adjacent_wilcoxon.csv"
        stats_df.to_csv(stats_csv, index=False)

        typer.echo("\nWithin-facet adjacent comparisons (one-sided Wilcoxon signed-rank; alternative: right > left):")
        # Keep the printed table compact and publication-friendly.
        # Add display labels for model comparisons (when x == "model"), to match plot labels.
        stats_df = stats_df.assign(
            left_display=stats_df.apply(
                lambda r: _display_name(str(r["left"])) if str(r.get("x", "")) == "model" else str(r["left"]),
                axis=1,
            ),
            right_display=stats_df.apply(
                lambda r: _display_name(str(r["right"])) if str(r.get("x", "")) == "model" else str(r["right"]),
                axis=1,
            ),
        )

        cols = [
            "facet",
            "test",
            "left_display",
            "right_display",
            "n_paired",
            "mean_left",
            "mean_right",
            "mean_diff_right_minus_left",
            "statistic",
            "p_value",
            "p_one_sided",
            "p_corrected",
            "correction",
            "note",
        ]
        present_cols = [c for c in cols if c in stats_df.columns]
        typer.echo(
            stats_df[present_cols]
            .sort_values(["facet", "p_one_sided"], na_position="last")
            .to_string(
                index=False,
                formatters={
                    "mean_left": "{:.3f}".format,
                    "mean_right": "{:.3f}".format,
                    "mean_diff_right_minus_left": "{:.3f}".format,
                    "statistic": (lambda v: "" if pd.isna(v) else f"{float(v):.3f}"),
                    "p_value": (lambda v: "" if pd.isna(v) else f"{float(v):.4g}"),
                    "p_one_sided": (lambda v: "" if pd.isna(v) else f"{float(v):.4g}"),
                    "p_corrected": (lambda v: "" if pd.isna(v) else f"{float(v):.4g}"),
                },
            )
        )
        typer.echo(f"\nSaved within-facet stats: {stats_csv}")

        stats_tex_path = output_dir / "hills_within_model_stats_tables.tex"

        def _facet_model_name(facet: object) -> str:
            facet_s = str(facet)
            if facet_s.startswith("model="):
                return facet_s.split("=", 1)[1]
            return facet_s

        friedman_df = stats_df[stats_df["test"] == "friedmanchisquare"].copy()
        posthoc_df = stats_df[stats_df["test"].str.contains("wilcoxon", na=False)].copy()

        if not friedman_df.empty:
            friedman_df = friedman_df.assign(
                model=friedman_df["facet"].map(_facet_model_name),
                model_display=friedman_df["facet"].map(_facet_model_name).map(_display_name),
            )

        if not posthoc_df.empty:
            posthoc_df = posthoc_df.assign(
                model=posthoc_df["facet"].map(_facet_model_name),
                model_display=posthoc_df["facet"].map(_facet_model_name).map(_display_name),
                left_display=posthoc_df["left"].map(_category_display_name),
                right_display=posthoc_df["right"].map(_category_display_name),
            )

        latex_stats_lines: list[str] = []
        latex_stats_lines.append("\\begin{table}[t]")
        latex_stats_lines.append("\\centering")
        latex_stats_lines.append("\\small")

        if not friedman_df.empty:
            latex_stats_lines.append("\\begin{tabular}{lrrrr}")
            latex_stats_lines.append("\\toprule")
            latex_stats_lines.append("Model & $N$ & $\\chi^2$ & $p$ & Note \\\\")
            latex_stats_lines.append("\\midrule")
            for _, r in friedman_df.iterrows():
                model_disp = _latex_escape(str(r.get("model_display", "")))
                n_paired = int(r.get("n_paired", 0))
                chi2 = r.get("statistic", "")
                p_val = r.get("p_value", "")
                note = _latex_escape(str(r.get("note", "")))
                chi2_s = "" if pd.isna(chi2) else f"{float(chi2):.3f}"
                p_s = "" if pd.isna(p_val) else f"{float(p_val):.4g}"
                latex_stats_lines.append(f"{model_disp} & {n_paired} & {chi2_s} & {p_s} & {note} \\\\")
            latex_stats_lines.append("\\bottomrule")
            latex_stats_lines.append("\\end{tabular}")
            latex_stats_lines.append("\\caption{Within-model Friedman tests across strategies (null: no differences).}")
            latex_stats_lines.append("\\label{tab:hills_friedman}")
            latex_stats_lines.append("\\end{table}")
            latex_stats_lines.append("")

        latex_stats_lines.append("\\begin{table}[t]")
        latex_stats_lines.append("\\centering")
        latex_stats_lines.append("\\small")
        if not posthoc_df.empty:
            latex_stats_lines.append("\\begin{tabular}{l l l r r r r r}")
            latex_stats_lines.append("\\toprule")
            latex_stats_lines.append("Model & Left & Right & $N$ & $W$ & $p$ & $p_{corr}$ & Corr. \\\\")
            latex_stats_lines.append("\\midrule")
            for _, r in posthoc_df.iterrows():
                model_disp = _latex_escape(str(r.get("model_display", "")))
                left_disp = _latex_escape(str(r.get("left_display", r.get("left", ""))))
                right_disp = _latex_escape(str(r.get("right_display", r.get("right", ""))))
                n_paired = int(r.get("n_paired", 0))
                stat = r.get("statistic", "")
                p_val = r.get("p_value", "")
                p_corr = r.get("p_corrected", "")
                corr = _latex_escape(str(r.get("correction", "")))
                stat_s = "" if pd.isna(stat) else f"{float(stat):.3f}"
                p_s = "" if pd.isna(p_val) else f"{float(p_val):.4g}"
                pc_s = "" if pd.isna(p_corr) else f"{float(p_corr):.4g}"
                latex_stats_lines.append(
                    f"{model_disp} & {left_disp} & {right_disp} & {n_paired} & {stat_s} & {p_s} & {pc_s} & {corr} \\\\"
                )
            latex_stats_lines.append("\\bottomrule")
            latex_stats_lines.append("\\end{tabular}")
            latex_stats_lines.append(
                "\\caption{Post-hoc paired Wilcoxon signed-rank tests (two-sided) with multiple-comparison correction.}"
            )
            latex_stats_lines.append("\\label{tab:hills_posthoc}")
        else:
            latex_stats_lines.append("\\begin{tabular}{l}")
            latex_stats_lines.append("\\toprule")
            latex_stats_lines.append("No post-hoc tests were run. \\\\")
            latex_stats_lines.append("\\bottomrule")
            latex_stats_lines.append("\\end{tabular}")
            latex_stats_lines.append(
                "\\caption{Post-hoc paired Wilcoxon signed-rank tests were not run (Friedman not significant or insufficient data).}"
            )
            latex_stats_lines.append("\\label{tab:hills_posthoc}")
        latex_stats_lines.append("\\end{table}")

        stats_tex_path.write_text("\\n".join(latex_stats_lines) + "\\n", encoding="utf-8")
        typer.echo(f"Saved within-model stats tables: {stats_tex_path}")

    typer.echo(f"Loaded {len(loaded.discovered_files)} CSV(s) from: {input_dir}")
    typer.echo(f"Row-level records: {len(df):,}")
    typer.echo(f"Per-id aggregated rows: {len(per_id):,}")
    typer.echo("\nSummary (per model):")
    typer.echo(
        summary.sort_values(["category", "mean"], ascending=[True, False]).to_string(
            index=False,
            formatters={"mean": "{:.3f}".format, "std": "{:.3f}".format},
        )
    )
    typer.echo("")
    for saved_plot_path in saved_plot_paths:
        typer.echo(f"Saved: {saved_plot_path}")
    typer.echo(f"Saved: {summary_csv}")
    typer.echo(f"Saved: {ci_csv}")
    typer.echo(f"Saved: {latex_path}")

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    seed_everything()
    app()
