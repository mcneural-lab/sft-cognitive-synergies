"""Late-vs-early change by dyad type, all prompt conditions (Figure S16).

The Figure-5 analysis kept separate for every HA prompt condition (inferred,
convergent, divergent) rather than collapsing the AI dyads.

Panels produced:
  * Figure S16A -- human response-time change (log, z-scored; late - early).
  * Figure S16B -- self-similarity change (late - early).

"""

from __future__ import annotations

import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console
from scipy.stats import mannwhitneyu, wilcoxon
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests

from sftbench import find_project_root
from sftbench.dyadic.load_data import load_data
from sftbench.dyadic.utils import COLOR_MAP, LABEL_MAP
from sftbench.figure_output import apply_bold_axis_style, save_figure_formats
from sftbench.reproducibility import seed_everything

app = typer.Typer()
console = Console()

DATA_DIR_REL = "data/dyadic/conceptnet"

ORDER = ["solo", "collab_human", "collab_ai_inferred", "collab_ai_convergent", "collab_ai_divergent"]
PAIRS = [
    ("collab_ai_convergent", "collab_ai_inferred"),
    ("collab_ai_convergent", "collab_ai_divergent"),
    ("collab_ai_divergent", "collab_ai_inferred"),
]
XTICKLABELS = [
    LABEL_MAP["solo"],
    LABEL_MAP["collab_human"],
    LABEL_MAP["collab_ai_inferred"],
    LABEL_MAP["collab_ai_convergent"],
    LABEL_MAP["collab_ai_divergent"],
]


def map_dyadtype_to_condition(dyad_type):
    return {
        "solo": "solo",
        "hh": "collab_human",
        "ha_inferred": "collab_ai_inferred",
        "ha_convergent": "collab_ai_convergent",
        "ha_divergent": "collab_ai_divergent",
    }.get(dyad_type, dyad_type)


def assign_condition(row):
    key = f"ha_{row['prompt']}" if row["dyadType"] == "ha" else row["dyadType"]
    return map_dyadtype_to_condition(key)


def summary_statistics(diff_data, value_col, metric_name, order):
    """Per-condition one-sample Wilcoxon summaries against zero."""
    console.print("\n" + "=" * 80)
    console.print(f"{metric_name.upper()} DIFFERENCE (LATE - EARLY) SUMMARY")
    console.print("=" * 80)
    rows = []
    for condition in order:
        vals = diff_data.loc[diff_data["condition"] == condition, value_col].dropna()
        if len(vals) == 0:
            continue
        stat_two, p_two = wilcoxon(vals, alternative="two-sided", zero_method="wilcox", method="auto")
        stat_gt, p_gt = wilcoxon(vals, alternative="greater", zero_method="wilcox", method="auto")
        stat_lt, p_lt = wilcoxon(vals, alternative="less", zero_method="wilcox", method="auto")
        label = LABEL_MAP.get(condition, condition)
        console.print(
            f"\n{label}: N={len(vals)}, Mean={vals.mean():.4f} ± {vals.sem():.4f}, Median={vals.median():.4f}"
        )
        console.print(f"  One-sample Wilcoxon vs 0 two-sided: W={stat_two:.2f}, p={p_two:.4g}")
        rows.append(
            {
                "metric": metric_name,
                "condition": condition,
                "label": label,
                "N": int(len(vals)),
                "mean": float(vals.mean()),
                "sem": float(vals.sem()),
                "median": float(vals.median()),
                "W_two_sided": float(stat_two),
                "p_two_sided": float(p_two),
                "W_greater": float(stat_gt),
                "p_greater": float(p_gt),
                "W_less": float(stat_lt),
                "p_less": float(p_lt),
            }
        )
    return pd.DataFrame(rows)


def pairwise_holm(diff_data, value_col, metric_name):
    """Holm-corrected pairwise comparisons across the HA prompt conditions."""
    console.print("\n--- Pairwise comparisons (Holm-Bonferroni corrected) ---")
    uncorrected = []
    stats = []
    complements = []
    for pair in PAIRS:
        g1 = diff_data[diff_data["condition"] == pair[0]][value_col].dropna()
        g2 = diff_data[diff_data["condition"] == pair[1]][value_col].dropna()
        stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
        stats.append(stat)
        # Complementary U for the reversed group order (U1 + U2 = n1 * n2).
        complements.append(len(g1) * len(g2) - stat)
        uncorrected.append(pval)
    reject, corrected, _, _ = multipletests(uncorrected, alpha=0.05, method="holm")
    rows = []
    for i, pair in enumerate(PAIRS):
        console.print(
            f"  {LABEL_MAP[pair[0]]} vs {LABEL_MAP[pair[1]]}: "
            f"U = {stats[i]:.2f} (complement {complements[i]:.2f}), p = {uncorrected[i]:.4g}, "
            f"corrected p = {corrected[i]:.4g}, significant = {reject[i]}"
        )
        rows.append(
            {
                "metric": metric_name,
                "group1": pair[0],
                "group2": pair[1],
                "U_two_sided": float(stats[i]),
                "U_two_sided_complement": float(complements[i]),
                "p_uncorrected": float(uncorrected[i]),
                "p_holm": float(corrected[i]),
                "significant": bool(reject[i]),
            }
        )
    return pd.DataFrame(rows)


def _barplot(diff_data, value_col, ylabel, ylim=None, line_offset=None):
    """Shared bar-plot rendering for both panels."""
    diff_data = diff_data.copy()
    diff_data["condition"] = pd.Categorical(diff_data["condition"], categories=ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.despine(ax=ax, right=True, top=True)
    sns.barplot(
        data=diff_data,
        x="condition",
        y=value_col,
        hue="condition",
        order=ORDER,
        palette=COLOR_MAP,
        errorbar="se",
        estimator=np.mean,
        ax=ax,
        legend=False,
    )
    ax.axhline(y=0, color="k", linewidth=0.8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Dyad Type")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(XTICKLABELS)))
    ax.set_xticklabels(XTICKLABELS)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    try:
        annotator = Annotator(ax, pairs=PAIRS, data=diff_data, x="condition", y=value_col, order=ORDER, hue=None)
        annotator.configure(
            test="Mann-Whitney", text_format="star", show_test_name=False, comparisons_correction="holm"
        ).apply_and_annotate()
    except Exception as e:
        console.print(f"[yellow]Warning: annotation failed: {e}[/yellow]")
    plt.tight_layout()
    return fig


def create_panel_a(df):
    """Figure S16A: log response-time change (late - early)."""
    console.print("Creating Figure S16A: response-time change...")
    combined_df = df[df["sourceType"] == "human"].copy()
    combined_df["condition"] = combined_df.apply(assign_condition, axis=1)
    combined_mean = (
        combined_df.groupby(["source", "condition", "word_index_split", "category"])[["log_irt_zscore"]]
        .agg({"log_irt_zscore": "mean"})
        .reset_index()
    )
    diff = combined_mean.pivot_table(
        index=["source", "condition"], columns="word_index_split", values="log_irt_zscore", aggfunc="mean"
    ).reset_index()
    diff.columns.name = None
    diff = diff.rename(columns={False: "log_irt_early", True: "log_irt_late"})
    diff["log_irt_diff"] = diff["log_irt_late"] - diff["log_irt_early"]

    fig = _barplot(
        diff,
        "log_irt_diff",
        "Human response time change (log, zscored)\n[Late - Early]",
        ylim=(0.00, 0.30),
    )
    summary = summary_statistics(diff, "log_irt_diff", "IRT", ORDER)
    pairwise = pairwise_holm(diff, "log_irt_diff", "IRT")
    return fig, summary, pairwise


def create_panel_b(df):
    """Figure S16B: self-similarity change (late - early)."""
    console.print("Creating Figure S16B: self-similarity change...")
    analysis_df = df[df["sourceType"] == "human"].copy()
    analysis_df = analysis_df[analysis_df["self_similarity"].notna()].copy()
    analysis_df["condition"] = analysis_df.apply(assign_condition, axis=1)
    analysis_df["portion"] = analysis_df["word_index_split"].apply(lambda x: "late" if x else "early")
    agg = analysis_df.groupby(["source", "condition", "portion", "category"])["self_similarity"].mean().reset_index()
    diff = agg.pivot_table(
        index=["source", "condition"], columns="portion", values="self_similarity", aggfunc="mean"
    ).reset_index()
    diff.columns.name = None
    diff["similarity_diff"] = diff["late"] - diff["early"]

    fig = _barplot(diff, "similarity_diff", "Self-similarity change\n[Late - Early]")
    summary = summary_statistics(diff, "similarity_diff", "Self-Similarity", ORDER)
    pairwise = pairwise_holm(diff, "similarity_diff", "Self-Similarity")
    return fig, summary, pairwise


@app.command()
def plot(
    data_root: pathlib.Path = typer.Option(
        Path("."), "--data-root", help="Root directory for relative input data paths."
    ),
    output_dir: pathlib.Path = typer.Option(
        None,
        "--output-dir",
        help="Directory to save plots and stats. Defaults to project_root/outputs/figures/supp/early-late-allconds.",
    ),
    formats: str | None = typer.Option(
        None, "--formats", help="Comma-separated output formats: png, pdf, svg. Defaults to png."
    ),
    show: bool = typer.Option(False, "--show/--no-show", help="Show the plots instead of just saving."),
):
    """Generate the all-conditions late-vs-early panels (Figure S16)."""
    project_root = find_project_root()
    if project_root is None:
        console.print("[red]Could not find project root.[/red]")
        raise typer.Exit(code=1)

    if not data_root.is_absolute():
        data_root = project_root / data_root
    data_dir = data_root / DATA_DIR_REL
    if not data_dir.exists():
        console.print(f"[red]Dyadic data directory not found: {data_dir}[/red]")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = project_root / "outputs" / "figures" / "supp" / "early-late-allconds"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_context("paper", font_scale=2.0)
    apply_bold_axis_style()

    df = load_data(data_dir=data_dir, cross_condition_only=False, include_solo=True)

    fig_a, summary_a, pairwise_a = create_panel_a(df)
    paths = save_figure_formats(
        fig_a,
        output_dir / "figS16a_irt_change_all_conditions",
        formats=formats,
        default=("png",),
        dpi=300,
        bbox_inches="tight",
    )
    console.print(f"[green]Saved Figure S16A to {', '.join(str(p) for p in paths)}[/green]")

    fig_b, summary_b, pairwise_b = create_panel_b(df)
    paths = save_figure_formats(
        fig_b,
        output_dir / "figS16b_self_similarity_change_all_conditions",
        formats=formats,
        default=("png",),
        dpi=300,
        bbox_inches="tight",
    )
    console.print(f"[green]Saved Figure S16B to {', '.join(str(p) for p in paths)}[/green]")

    pd.concat([summary_a, summary_b], ignore_index=True).to_csv(output_dir / "figS16_summary_stats.csv", index=False)
    pd.concat([pairwise_a, pairwise_b], ignore_index=True).to_csv(output_dir / "figS16_pairwise_holm.csv", index=False)

    if show:
        plt.show()

    console.print("\n[bold]Finished![/bold]")


if __name__ == "__main__":
    seed_everything()
    app()
