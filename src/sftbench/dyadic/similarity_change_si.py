"""Self/other similarity change (late - early) by dyad type (Figure S15).

Two-panel bar plot of the change in cosine similarity (late minus early) for
self-similarity and other-similarity, by dyad type (AI dyads collapsed).
Uses pre-computed ``self_similarity``/``other_similarity`` columns from ``load_data``.
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
from sftbench.dyadic.utils import COLOR_MAP_COLLAPSED, LABEL_MAP
from sftbench.figure_output import apply_bold_axis_style, save_figure_formats
from sftbench.reproducibility import seed_everything

app = typer.Typer()
console = Console()

DATA_DIR_REL = "data/dyadic/conceptnet"
ORDER = ["solo", "collab_human", "collab_ai"]


def map_dyadtype_to_condition(dyad_type):
    return {"solo": "solo", "hh": "collab_human", "ha": "collab_ai"}.get(dyad_type, dyad_type)


def compute_similarity_differences(df):
    """Late-minus-early self/other similarity per source x condition, long format."""
    analysis_df = df[df["sourceType"] == "human"].copy()
    analysis_df["condition"] = analysis_df["dyadType"].apply(map_dyadtype_to_condition)
    analysis_df["portion"] = analysis_df["word_index_split"].apply(lambda x: "late" if x else "early")

    diff_list = []
    for sim_col, sim_type in [("self_similarity", "self"), ("other_similarity", "other")]:
        sim_df = analysis_df[analysis_df[sim_col].notna()].copy()
        if sim_df.empty:
            continue
        agg = (
            sim_df.groupby(["source", "condition", "portion", "category"])[sim_col]
            .mean()
            .reset_index()
            .groupby(["source", "condition", "portion"])[sim_col]
            .mean()
            .reset_index()
        )
        pivot = agg.pivot_table(index=["source", "condition"], columns="portion", values=sim_col).reset_index()
        pivot.columns.name = None
        if "early" in pivot.columns and "late" in pivot.columns:
            pivot["similarity_diff"] = pivot["late"] - pivot["early"]
            pivot["similarity_type"] = sim_type
            diff_list.append(pivot[["source", "condition", "similarity_type", "similarity_diff"]])

    return pd.concat(diff_list, ignore_index=True) if diff_list else pd.DataFrame()


def summary_statistics(diff_data, order):
    """Per-condition one-sample Wilcoxon summaries against zero."""
    console.print("\n" + "=" * 80)
    console.print("SIMILARITY DIFFERENCES (LATE - EARLY) SUMMARY")
    console.print("=" * 80)
    rows = []
    for sim_type in ["self", "other"]:
        sim_data = diff_data[diff_data["similarity_type"] == sim_type]
        for condition in order:
            vals = sim_data.loc[sim_data["condition"] == condition, "similarity_diff"].dropna()
            if len(vals) == 0:
                continue
            stat_two, p_two = wilcoxon(vals, alternative="two-sided", zero_method="wilcox", method="auto")
            stat_gt, p_gt = wilcoxon(vals, alternative="greater", zero_method="wilcox", method="auto")
            stat_lt, p_lt = wilcoxon(vals, alternative="less", zero_method="wilcox", method="auto")
            label = LABEL_MAP.get(condition, condition)
            console.print(f"\n  {sim_type.upper()} / {label}:")
            console.print(
                f"    N = {len(vals)}, Mean = {vals.mean():.4f} ± {vals.sem():.4f}, Median = {vals.median():.4f}"
            )
            console.print(f"    One-sample Wilcoxon vs 0 two-sided: W={stat_two:.2f}, p={p_two:.4g}")
            rows.append(
                {
                    "similarity_type": sim_type,
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


def create_panel(diff_data):
    """Figure S15: self/other similarity change by dyad type."""
    console.print("Creating Figure S15: similarity change (late - early)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    pairwise_rows = []
    for idx, sim_type in enumerate(["self", "other"]):
        ax = axes[idx]
        sim_data = diff_data[diff_data["similarity_type"] == sim_type].copy()
        if sim_data.empty:
            continue

        plot_order = [c for c in ORDER if c != "solo"] if sim_type == "other" else ORDER
        conditions_in_data = [c for c in plot_order if c in sim_data["condition"].values]
        sim_data["condition"] = pd.Categorical(sim_data["condition"], categories=conditions_in_data, ordered=True)
        palette_list = [COLOR_MAP_COLLAPSED.get(c, "#d3d3d3") for c in conditions_in_data]

        sns.barplot(
            data=sim_data,
            x="condition",
            y="similarity_diff",
            hue="condition",
            palette=palette_list,
            order=conditions_in_data,
            errorbar="se",
            estimator=np.mean,
            ax=ax,
            legend=False,
        )
        ax.axhline(y=0, color="k", linewidth=0.8)

        pairs = (
            [("collab_human", c) for c in conditions_in_data if c != "collab_human"]
            if "collab_human" in conditions_in_data
            else []
        )
        if pairs:
            try:
                annotator = Annotator(
                    ax,
                    pairs=pairs,
                    data=sim_data,
                    x="condition",
                    y="similarity_diff",
                    order=conditions_in_data,
                    hue=None,
                )
                annotator.configure(
                    test="Mann-Whitney", text_format="star", show_test_name=False, comparisons_correction="holm"
                ).apply_and_annotate()

                uncorrected = []
                stats = []
                complements = []
                for pair in pairs:
                    g1 = sim_data[sim_data["condition"] == pair[0]]["similarity_diff"].dropna()
                    g2 = sim_data[sim_data["condition"] == pair[1]]["similarity_diff"].dropna()
                    stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
                    stats.append(stat)
                    # Complementary U for the reversed group order (U1 + U2 = n1 * n2).
                    complements.append(len(g1) * len(g2) - stat)
                    uncorrected.append(pval)
                reject, corrected, _, _ = multipletests(uncorrected, alpha=0.05, method="holm")
                console.print(f"\n--- {sim_type.upper()} similarity pairwise (Holm-corrected) ---")
                for i, pair in enumerate(pairs):
                    console.print(
                        f"  {LABEL_MAP.get(pair[0], pair[0])} vs {LABEL_MAP.get(pair[1], pair[1])}: "
                        f"U = {stats[i]:.2f} (complement {complements[i]:.2f}), p = {uncorrected[i]:.4g}, "
                        f"corrected p = {corrected[i]:.4g}, significant = {reject[i]}"
                    )
                    pairwise_rows.append(
                        {
                            "similarity_type": sim_type,
                            "group1": pair[0],
                            "group2": pair[1],
                            "U_two_sided": float(stats[i]),
                            "U_two_sided_complement": float(complements[i]),
                            "p_uncorrected": float(uncorrected[i]),
                            "p_holm": float(corrected[i]),
                            "significant": bool(reject[i]),
                        }
                    )
            except Exception as e:
                console.print(f"[yellow]Warning: annotation failed for {sim_type}: {e}[/yellow]")

        ax.set_xticks(range(len(conditions_in_data)))
        ax.set_xticklabels([LABEL_MAP.get(c, c) for c in conditions_in_data], rotation=30, ha="right")
        ax.set_xlabel("Dyad Type")
        ax.set_ylabel("Change in Cosine Similarity\n[Late - Early]" if idx == 0 else "")
        ax.set_title(f"{'Self' if sim_type == 'self' else 'Other'} Cosine Similarity")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig, pd.DataFrame(pairwise_rows)


@app.command()
def plot(
    data_root: pathlib.Path = typer.Option(
        Path("."), "--data-root", help="Root directory for relative input data paths."
    ),
    output_dir: pathlib.Path = typer.Option(
        None,
        "--output-dir",
        help="Directory to save plots and stats. Defaults to project_root/outputs/figures/supp/similarity-change.",
    ),
    formats: str | None = typer.Option(
        None, "--formats", help="Comma-separated output formats: png, pdf, svg. Defaults to png."
    ),
    show: bool = typer.Option(False, "--show/--no-show", help="Show the plots instead of just saving."),
):
    """Generate the self/other similarity-change panel (Figure S15)."""
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
        output_dir = project_root / "outputs" / "figures" / "supp" / "similarity-change"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_context("paper", font_scale=2.2)
    apply_bold_axis_style()

    df = load_data(data_dir=data_dir, cross_condition_only=False, include_solo=True)
    diff_data = compute_similarity_differences(df)
    if diff_data.empty:
        console.print("[red]No difference data computed.[/red]")
        raise typer.Exit(code=1)

    fig, pairwise = create_panel(diff_data)
    paths = save_figure_formats(
        fig,
        output_dir / "figS15_self_other_similarity_change",
        formats=formats,
        default=("png",),
        dpi=300,
        bbox_inches="tight",
    )
    console.print(f"[green]Saved Figure S15 to {', '.join(str(p) for p in paths)}[/green]")

    summary = summary_statistics(diff_data, ORDER)
    summary.to_csv(output_dir / "figS15_summary_stats.csv", index=False)
    pairwise.to_csv(output_dir / "figS15_pairwise_holm.csv", index=False)
    diff_data.to_csv(output_dir / "figS15_similarity_differences.csv", index=False)

    if show:
        plt.show()

    console.print("\n[bold]Finished![/bold]")


if __name__ == "__main__":
    seed_everything()
    app()
