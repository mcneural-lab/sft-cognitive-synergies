from __future__ import annotations

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from scipy import stats
from statannotations.Annotator import Annotator

from sftbench import find_project_root
from sftbench.figure_output import apply_bold_axis_style, parse_figure_formats, save_figure_formats
from sftbench.reproducibility import SEED, seed_everything

app = typer.Typer(help="CLI to plot 3-panel human vs AI accuracy comparisons from next-exemplar CSVs.")


def load_plotting_config(config_path: Path) -> None:
    """Loads seaborn context and style from a TOML file (same convention as original script)."""
    if not config_path.exists():
        sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
        apply_bold_axis_style()
        return

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    seaborn_cfg = config.get("seaborn", {})
    context = seaborn_cfg.get("context", "paper")
    font_scale = seaborn_cfg.get("font_scale", 1.5)
    style = seaborn_cfg.get("style", "whitegrid")
    sns.set_theme(context=context, style=style, font_scale=font_scale)
    apply_bold_axis_style()


def _as_bool_series(s: pd.Series) -> pd.Series:
    """
    Coerce a column to boolean where it might be actual bools, 0/1, or strings.

    Accepts:
      - True/False
      - 1/0
      - "true"/"false" (case-insensitive)
      - "t"/"f"
      - "yes"/"no"
    Missing values remain missing (pd.NA).
    """
    if pd.api.types.is_bool_dtype(s):
        return s

    if pd.api.types.is_numeric_dtype(s):
        out = s.copy()
        out = out.where(out.isna(), out.astype(int).astype(bool))
        return out

    mapping = {
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "1": True,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
        "0": False,
    }
    lowered = s.astype("string").str.strip().str.lower()
    return lowered.map(mapping)


def _project_root_fallback() -> Path:
    root = find_project_root()
    if root is None:
        return Path.cwd()
    return root


@dataclass(frozen=True)
class ColumnMap:
    id_col: str = "id"
    rank_col: str = "rank"
    category_col: str = "category"
    normalized_pred_col: str = "normalizedPredicted"
    ai_correct_col: str = "is_correct"
    human_correct_col: str = "is_human_correct"


DropMode = Literal["none", "rows-with-missing", "ids-with-missing"]


def _validate_required_columns(df: pd.DataFrame, cmap: ColumnMap, source: Path) -> None:
    required = [
        cmap.id_col,
        cmap.rank_col,
        cmap.category_col,
        cmap.ai_correct_col,
        cmap.human_correct_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise typer.BadParameter(
            f"Missing required columns in {source}: {missing}. Found columns: {sorted(df.columns.tolist())}"
        )


def _load_next_exemplar_csvs(csv_paths: Iterable[Path], cmap: ColumnMap) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        tdf = pd.read_csv(p)
        _validate_required_columns(tdf, cmap, p)
        dfs.append(tdf)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _load_ngram_bigram_results(ngram_dir: Path, cmap: ColumnMap) -> pd.DataFrame:
    """
    Load bigram (2-gram) baseline CSVs and return a long-form dataframe with:
      - id
      - data-category
      - rank
      - rank_group
      - correct (0/1 float)
      - type="2-gram"

    Note: We keep `id` so baseline reporting can be computed as participant-level means
    (average within id), matching the human/AI summaries.
    """
    if not ngram_dir.exists():
        return pd.DataFrame()

    csv_paths = sorted(ngram_dir.glob("2-gram-*.csv"))
    if not csv_paths:
        return pd.DataFrame()

    df_raw = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

    required = [cmap.id_col, cmap.rank_col, cmap.category_col, cmap.ai_correct_col]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise typer.BadParameter(
            f"Missing required columns in bigram results under {ngram_dir}: {missing}. "
            f"Found columns: {sorted(df_raw.columns.tolist())}"
        )

    df = df_raw.copy()
    df[cmap.rank_col] = pd.to_numeric(df[cmap.rank_col], errors="coerce")
    df = df.rename(columns={cmap.category_col: "data-category", cmap.rank_col: "rank", cmap.id_col: "id"})

    df["id"] = df["id"].astype(str)
    df["correct"] = _as_bool_series(df[cmap.ai_correct_col]).astype("float")
    df["rank_group"] = ((df["rank"] - 1) // 5) * 5
    df["type"] = "2-gram"

    return df[["id", "data-category", "rank", "rank_group", "correct", "type"]].dropna(subset=["rank"])


def _apply_drop_missing_normalized(
    df: pd.DataFrame, cmap: ColumnMap, drop_mode: DropMode
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Drop missing `normalizedPredicted` at row-level or id-level.

    Returns (filtered_df, stats).
    """
    stats: dict[str, int] = {
        "rows_before": int(len(df)),
        "rows_dropped": 0,
        "ids_dropped": 0,
        "rows_after": 0,
    }

    if drop_mode == "none":
        stats["rows_after"] = int(len(df))
        return df, stats

    if cmap.normalized_pred_col not in df.columns:
        stats["rows_after"] = int(len(df))
        return df, stats

    missing_mask = df[cmap.normalized_pred_col].isna() | (
        df[cmap.normalized_pred_col].astype("string").str.strip() == ""
    )

    if drop_mode == "rows-with-missing":
        out = df.loc[~missing_mask].copy()
        stats["rows_dropped"] = int(missing_mask.sum())
        stats["rows_after"] = int(len(out))
        return out, stats

    ids_with_missing = set(df.loc[missing_mask, cmap.id_col].astype(str).unique().tolist())
    if not ids_with_missing:
        stats["rows_after"] = int(len(df))
        return df, stats

    out = df.loc[~df[cmap.id_col].astype(str).isin(ids_with_missing)].copy()
    stats["ids_dropped"] = int(len(ids_with_missing))
    stats["rows_dropped"] = int(len(df) - len(out))
    stats["rows_after"] = int(len(out))
    return out, stats


def _prepare_long_df(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    """
    Convert wide correctness columns into long format with `type` and `correct`.

    Output columns include:
      - id
      - rank
      - data-category
      - type: "human" or "AI"
      - correct: bool/NA
      - rank_group: integer bucket
    """
    out = df.copy()

    out[cmap.rank_col] = pd.to_numeric(out[cmap.rank_col], errors="coerce")
    out = out.rename(columns={cmap.category_col: "data-category"})

    out[cmap.ai_correct_col] = _as_bool_series(out[cmap.ai_correct_col])
    out[cmap.human_correct_col] = _as_bool_series(out[cmap.human_correct_col])

    df_h = out[[cmap.id_col, cmap.rank_col, "data-category", cmap.human_correct_col]].copy()
    df_h = df_h.rename(columns={cmap.human_correct_col: "correct", cmap.id_col: "id", cmap.rank_col: "rank"})
    df_h["type"] = "human"

    df_ai = out[[cmap.id_col, cmap.rank_col, "data-category", cmap.ai_correct_col]].copy()
    df_ai = df_ai.rename(columns={cmap.ai_correct_col: "correct", cmap.id_col: "id", cmap.rank_col: "rank"})
    df_ai["type"] = "AI"

    long_df = pd.concat([df_h, df_ai], ignore_index=True)

    long_df["rank_group"] = ((long_df["rank"] - 1) // 5) * 5

    return long_df


def _ensure_category_order(df: pd.DataFrame, category_order: list[str]) -> list[str]:
    """Return the subset of category_order that actually exists in df, preserving order."""
    existing = set(df["data-category"].dropna().astype(str).unique().tolist())
    return [c for c in category_order if c in existing]


def _mean_ci95_t(values: pd.Series) -> tuple[float, float, float, int]:
    """
    Compute mean and 95% t-based CI: mean ± t_{n-1,0.975} * (sd/sqrt(n)).

    Returns (mean, lo, hi, n). If n < 2 or variance is undefined, lo/hi are NaN.
    """
    x = pd.to_numeric(values, errors="coerce").dropna()
    n = int(x.shape[0])
    mean = float(x.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan"), float("nan"), n

    sd = float(x.std(ddof=1))
    if not pd.notna(sd) or sd == 0.0:
        return mean, float("nan"), float("nan"), n

    se = sd / (n**0.5)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    lo = mean - tcrit * se
    hi = mean + tcrit * se
    return mean, lo, hi, n


def _print_mean_cis_overall_and_category(
    overall_by_id: pd.DataFrame,
    by_category_by_id: pd.DataFrame,
    cat_order: list[str],
) -> None:
    """
    Print participant-level mean accuracy with 95% t-based CIs:
      - Overall by type (per-id mean correctness across *all rows*; row-weighted)
      - Per-category by type (per-id mean correctness within category; row-weighted)

    Also print paired two-sided Wilcoxon signed-rank test stats comparing AI vs human:
      - Overall (paired within id on overall means)
      - Per-category (paired within id on category means)

    IMPORTANT: This is designed to match `evaluate_predictions.calculate_subject_accuracy`
    semantics (subject accuracy is a row-weighted mean within the subject).
    """
    typer.echo("Participant-level mean accuracy (mean with 95% t CI):")
    typer.echo("Overall:")
    for t in ["human", "AI"]:
        vals = overall_by_id.loc[overall_by_id["type"] == t, "correct_fraction"]
        mean, lo, hi, n = _mean_ci95_t(vals)
        typer.echo(f"  {t}: mean={mean:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}], n={n}")

    wide_overall = overall_by_id.pivot(index="id", columns="type", values="correct_fraction").reset_index(drop=False)
    if ("human" in wide_overall.columns) and ("AI" in wide_overall.columns):
        paired = wide_overall[["human", "AI"]].dropna()
        n_pairs = int(paired.shape[0])
        if n_pairs > 0:
            mean_diff = float((paired["AI"] - paired["human"]).mean())
            typer.echo(f"Mean difference (AI - human), overall: {mean_diff:.4f} (paired n={n_pairs})")

            res = stats.wilcoxon(
                paired["AI"].to_numpy(),
                paired["human"].to_numpy(),
                alternative="two-sided",
            )
            typer.echo("Paired Wilcoxon signed-rank (two-sided), overall (AI vs human):")
            typer.echo(f"  n={n_pairs}, W={float(res.statistic):.6g}, p={float(res.pvalue):.6g}")
        else:
            typer.echo("Mean difference (AI - human), overall: nan (paired n=0)")
            typer.echo("Paired Wilcoxon signed-rank (two-sided), overall (AI vs human): n=0 (no complete pairs)")
    else:
        typer.echo("Mean difference (AI - human), overall: missing columns for pairing")
        typer.echo("Paired Wilcoxon signed-rank (two-sided), overall (AI vs human): missing columns for pairing")

    typer.echo("By category:")
    for cat in cat_order:
        typer.echo(f"  {cat}:")
        for t in ["human", "AI"]:
            vals = by_category_by_id.loc[
                (by_category_by_id["data-category"] == cat) & (by_category_by_id["type"] == t),
                "correct_fraction",
            ]
            mean, lo, hi, n = _mean_ci95_t(vals)
            typer.echo(f"    {t}: mean={mean:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}], n={n}")

        cat_by_id = by_category_by_id.loc[by_category_by_id["data-category"] == cat, ["id", "type", "correct_fraction"]]
        wide_cat = cat_by_id.pivot(index="id", columns="type", values="correct_fraction")
        if ("human" in wide_cat.columns) and ("AI" in wide_cat.columns):
            paired = wide_cat[["human", "AI"]].dropna()
            n_pairs = int(paired.shape[0])
            if n_pairs > 0:
                mean_diff = float((paired["AI"] - paired["human"]).mean())
                typer.echo(f"    mean difference (AI - human): {mean_diff:.4f} (paired n={n_pairs})")

                res = stats.wilcoxon(
                    paired["AI"].to_numpy(),
                    paired["human"].to_numpy(),
                    alternative="two-sided",
                )
                typer.echo(
                    f"    paired Wilcoxon (two-sided) AI vs human: n={n_pairs}, W={float(res.statistic):.6g}, p={float(res.pvalue):.6g}"
                )
            else:
                typer.echo("    mean difference (AI - human): nan (paired n=0)")
                typer.echo("    paired Wilcoxon (two-sided) AI vs human: n=0 (no complete pairs)")
        else:
            typer.echo("    mean difference (AI - human): missing columns for pairing")
            typer.echo("    paired Wilcoxon (two-sided) AI vs human: missing columns for pairing")


def _print_bigram_means(
    bigram_long: pd.DataFrame,
    cat_order: list[str],
    *,
    rank_max: int = 40,
) -> None:
    """
    Print 2-gram baseline mean accuracies as participant-level means (average within `id`),
    to match the human/AI summaries:

      - Overall (restricted to ranks < rank_max, to match the rank panel)
      - By category (restricted to ranks < rank_max)
      - By rank window (rank_group), for ranks < rank_max

    We report:
      - mean: average of per-id means
      - n_ids: number of participants contributing to that estimate
    """
    if bigram_long is None or bigram_long.empty:
        return

    b = bigram_long.copy()
    b["rank"] = pd.to_numeric(b["rank"], errors="coerce")
    b = b.dropna(subset=["rank"])
    b = b.loc[b["rank"] < rank_max].copy()

    typer.echo(f"2-gram baseline mean accuracy (rank < {rank_max}; participant-level):")

    overall_by_id = b.groupby(["id"], dropna=False)["correct"].mean().reset_index()
    overall_vals = pd.to_numeric(overall_by_id["correct"], errors="coerce").dropna()
    overall_mean = float(overall_vals.mean()) if not overall_vals.empty else float("nan")
    typer.echo(f"  overall: mean={overall_mean:.4f}, n_ids={int(overall_vals.shape[0])}")

    typer.echo("  by category:")
    for cat in cat_order:
        cat_by_id = (
            b.loc[b["data-category"].astype(str) == str(cat)]
            .groupby(["id"], dropna=False)["correct"]
            .mean()
            .reset_index()
        )
        vals = pd.to_numeric(cat_by_id["correct"], errors="coerce").dropna()
        mean = float(vals.mean()) if not vals.empty else float("nan")
        typer.echo(f"    {cat}: mean={mean:.4f}, n_ids={int(vals.shape[0])}")

    typer.echo("  by rank window:")
    rank_groups = sorted(pd.to_numeric(b["rank_group"], errors="coerce").dropna().unique().tolist())
    for rg in rank_groups:
        rg_int = int(rg)
        rg_by_id = b.loc[b["rank_group"] == rg_int].groupby(["id"], dropna=False)["correct"].mean().reset_index()
        vals = pd.to_numeric(rg_by_id["correct"], errors="coerce").dropna()
        mean = float(vals.mean()) if not vals.empty else float("nan")
        typer.echo(f"    {rg_int}: mean={mean:.4f}, n_ids={int(vals.shape[0])}")


def plot_three_panel(
    df_long: pd.DataFrame,
    output_path: Path,
    category_order: list[str],
    title: str,
    show: bool,
    *,
    bigram_long: pd.DataFrame | None = None,
    formats: str | None = None,
) -> None:
    """
    Render the 1x3 panel next-exemplar prediction figure.

    The paired Wilcoxon annotations are two-sided: this version of `statannotations`
    does not accept an `alternative` parameter.
    """
    overall_by_id = (
        df_long.groupby(["id", "type"], dropna=False)["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "correct_fraction"})
    )

    by_category_by_id = (
        df_long.groupby(["id", "data-category", "type"], dropna=False)["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "correct_fraction"})
    )

    cat_order = _ensure_category_order(by_category_by_id, category_order) or category_order

    _print_mean_cis_overall_and_category(overall_by_id, by_category_by_id, cat_order)

    _print_bigram_means(bigram_long, cat_order)

    fig = plt.figure(figsize=(17, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.6, 1.2, 1.2], wspace=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)

    for label, ax in zip(["B", "C", "D"], [ax1, ax2, ax3], strict=True):
        ax.text(
            -0.08,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    sns.boxplot(
        data=overall_by_id,
        x="type",
        y="correct_fraction",
        hue="type",
        order=["human", "AI"],
        hue_order=["human", "AI"],
        legend=False,
        ax=ax1,
    )

    if bigram_long is not None and not bigram_long.empty:
        bigram_overall = float(pd.to_numeric(bigram_long["correct"], errors="coerce").mean())
        if pd.notna(bigram_overall):
            ax1.axhline(
                y=bigram_overall,
                color="black",
                linestyle=":",
                linewidth=3.0,
                alpha=1.0,
                zorder=0,
            )

    ax1.set_title("Overall Accuracy")
    ax1.set_xlabel("Type")
    ax1.set_ylabel("Accuracy")

    annotator = Annotator(
        ax1,
        [("human", "AI")],
        data=overall_by_id,
        x="type",
        y="correct_fraction",
    )
    annotator.configure(test="Wilcoxon")
    annotator.apply_and_annotate()

    sns.stripplot(
        data=by_category_by_id,
        x="data-category",
        y="correct_fraction",
        hue="type",
        order=cat_order,
        hue_order=["human", "AI"],
        jitter=0.2,
        dodge=0.2,
        alpha=0.5,
        size=6,
        zorder=1,
        ax=ax2,
    )

    if bigram_long is not None and not bigram_long.empty:
        bigram_cat = (
            bigram_long.groupby("data-category", dropna=False)["correct"]
            .mean()
            .reset_index()
            .rename(columns={"correct": "bigram_mean"})
        )
        for i, cat in enumerate(cat_order):
            row = bigram_cat.loc[bigram_cat["data-category"].astype(str) == str(cat)]
            if row.empty:
                continue
            y = float(row["bigram_mean"].iloc[0])
            if not pd.notna(y):
                continue

            ax2.hlines(
                y=y,
                xmin=i - 0.35,
                xmax=i + 0.35,
                colors="black",
                linestyles=":",
                linewidth=3.0,
                alpha=1.0,
                zorder=2,
            )
    sns.pointplot(
        data=by_category_by_id,
        x="data-category",
        y="correct_fraction",
        hue="type",
        order=cat_order,
        hue_order=["human", "AI"],
        linestyle="none",
        dodge=0.6,
        capsize=0.2,
        errorbar=("ci", 95),
        seed=SEED,
        legend=False,
        ax=ax2,
    )
    ax2.set_title("Accuracy by Category")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("")

    pairs = [[(cat, "AI"), (cat, "human")] for cat in cat_order]
    annotator = Annotator(
        ax2,
        pairs,
        data=by_category_by_id,
        x="data-category",
        y="correct_fraction",
        order=cat_order,
        hue="type",
    )
    annotator.configure(test="Wilcoxon")
    annotator.apply_and_annotate()
    ax2.legend([], [], frameon=False)

    sns.pointplot(
        data=df_long[df_long["rank"] < 40],
        x="rank_group",
        y="correct",
        hue="type",
        hue_order=["human", "AI"],
        capsize=0.2,
        errorbar=("ci", 95),
        seed=SEED,
        ax=ax3,
    )

    if bigram_long is not None and not bigram_long.empty:
        bigram_rank = (
            bigram_long.loc[bigram_long["rank"] < 40]
            .groupby("rank_group", dropna=False)["correct"]
            .mean()
            .reset_index()
            .sort_values("rank_group")
        )
        if not bigram_rank.empty:
            rank_groups = bigram_rank["rank_group"].tolist()
            x_pos = list(range(len(rank_groups)))
            ax3.plot(
                x_pos,
                bigram_rank["correct"].tolist(),
                color="black",
                linestyle=":",
                linewidth=3.0,
                alpha=1.0,
                zorder=3,
            )
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([str(rg) for rg in rank_groups])

    ax3.set_title("Accuracy By Rank")
    ax3.set_xlabel("Rank Window")
    ax3.set_ylabel("")

    handles, labels = ax3.get_legend_handles_labels()
    if bigram_long is not None and not bigram_long.empty:
        if "2-gram" not in labels:
            handles.append(plt.Line2D([0], [0], color="black", linestyle=":", linewidth=3.0, label="2-gram"))
            labels.append("2-gram")
    ax3.legend(handles, labels, title="Type")

    plt.suptitle(title)
    save_figure_formats(fig, output_path, formats=formats, default=("png", "pdf"))
    if show:
        plt.show()
    plt.close(fig)


@app.command()
def plot(
    input_dir: Path = typer.Option(
        ...,
        help="Directory containing next-exemplar CSVs (e.g., results/micro/prediction/next-exemplar/gemini-3.0-pro).",
    ),
    glob_pattern: str = typer.Option(
        "*.csv",
        help="Glob for selecting CSVs within input_dir.",
    ),
    config: Path = typer.Option(
        None,
        help="Plotting configuration TOML. If omitted, uses configs/plotting.toml from project root when available.",
    ),
    output_dir: Path = typer.Option(
        Path("plots"),
        help="Directory to save generated plots.",
    ),
    show: bool = typer.Option(
        False,
        help="Whether to show plots interactively.",
    ),
    drop_missing_normalized: DropMode = typer.Option(
        "none",
        help="How to handle missing `normalizedPredicted`: 'none' | 'rows-with-missing' | 'ids-with-missing'.",
    ),
    category_order: str = typer.Option(
        "animals,clothes,supermarket",
        help="Comma-separated category order for the category panel.",
    ),
    model_name: str = typer.Option(
        "",
        help="Optional model name to include in plot titles (e.g., 'gemini-3.0-pro').",
    ),
    formats: str | None = typer.Option(
        None,
        "--formats",
        help="Comma-separated output formats for figures: png, pdf, svg. Defaults to png,pdf.",
    ),
):
    root = _project_root_fallback()

    if config is None:
        config = root / "configs" / "plotting.toml"

    load_plotting_config(config)

    if not input_dir.exists():
        raise typer.BadParameter(f"input_dir does not exist: {input_dir}")

    csv_paths = sorted(input_dir.glob(glob_pattern))
    if not csv_paths:
        raise typer.BadParameter(f"No CSVs matched {glob_pattern} in {input_dir}")

    cmap = ColumnMap()
    df_raw = _load_next_exemplar_csvs(csv_paths, cmap)
    if df_raw.empty:
        raise typer.BadParameter("Loaded 0 rows from CSVs; nothing to plot.")

    df_filtered, drop_stats = _apply_drop_missing_normalized(df_raw, cmap, drop_missing_normalized)

    df_long = _prepare_long_df(df_filtered, cmap)

    df_long["data-category"] = df_long["data-category"].astype(str)
    df_long["id"] = df_long["id"].astype(str)

    cats = [c.strip() for c in category_order.split(",") if c.strip()]

    ngram_dir = root / "results" / "micro" / "prediction" / "next-exemplar" / "n-gram"
    bigram_long = _load_ngram_bigram_results(ngram_dir=ngram_dir, cmap=cmap)

    output_dir.mkdir(parents=True, exist_ok=True)
    three_panel_path = output_dir / "AI_vs_human_accuracy_comparison_next-exemplar.png"

    title = "AI vs. Human Next Exemplar Prediction Accuracy Comparison"
    if model_name.strip():
        title = f"{title} ({model_name.strip()})"

    plot_three_panel(
        df_long=df_long,
        output_path=three_panel_path,
        category_order=cats,
        title=title,
        show=show,
        bigram_long=bigram_long,
        formats=formats,
    )

    saved_plot_formats = parse_figure_formats(formats, default=("png", "pdf"))
    typer.echo("Saved plots:")
    for figure_format in saved_plot_formats:
        typer.echo(f"  - {three_panel_path.with_suffix(f'.{figure_format}')}")

    typer.echo("Drop-missing-normalized summary:")
    typer.echo(f"  mode: {drop_missing_normalized}")
    typer.echo(f"  rows_before: {drop_stats['rows_before']}")
    typer.echo(f"  ids_dropped: {drop_stats['ids_dropped']}")
    typer.echo(f"  rows_dropped: {drop_stats['rows_dropped']}")
    typer.echo(f"  rows_after: {drop_stats['rows_after']}")


if __name__ == "__main__":
    seed_everything()
    app()
