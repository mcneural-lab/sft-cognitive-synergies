"""Human vs AI-partner response-time violin (Figure S13B).

Violin plot of log response times (seconds) for human-human dyads versus the
AI partner words, split by category (animals, clothes).

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
from scipy import stats
from statannotations.Annotator import Annotator

from sftbench import find_project_root
from sftbench.dyadic.load_data import load_data
from sftbench.dyadic.utils import format_pvalue, holm_corrected_pvalues, significance_stars
from sftbench.figure_output import apply_bold_axis_style, save_figure_formats
from sftbench.reproducibility import seed_everything

app = typer.Typer()
console = Console()

DATA_DIR_REL = "data/dyadic/conceptnet"

VIOLIN_COLOR_MAP = {"AI": "#ff7f0e", "Human": "#165785"}
COLUMNS = ["irt", "log_irt", "category", "playerID", "word_index", "source", "sourceType", "dyadType"]


def calculate_descriptive_stats_irt(data, value_col, group_cols):
    """Descriptive statistics for IRT data with flexible grouping."""
    group_cols = list(group_cols)
    stats_list = []
    for name, group in data.groupby(group_cols):
        values = group[value_col].dropna()
        if len(values) == 0:
            continue
        if len(group_cols) == 1:
            group_desc = f"{name}"
        else:
            group_desc = " - ".join(str(name[i] if isinstance(name, tuple) else name) for i in range(len(group_cols)))
        n = len(values)
        mean_val = values.mean()
        std_val = values.std()
        ci = stats.t.interval(0.95, n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
        stats_list.append(
            {
                "Group": group_desc,
                "N": n,
                "Mean_95CI": f"{mean_val:.3f} ({ci[0]:.3f}, {ci[1]:.3f})",
                "Median_IQR": f"{values.median():.3f} ({values.quantile(0.25):.3f}, {values.quantile(0.75):.3f})",
                "Min_Max": f"{values.min():.3f}, {values.max():.3f}",
            }
        )
    return pd.DataFrame(stats_list)


def extract_statistics_from_annotator(test_results, analysis_name):
    """Pull p-values/statistics out of ``apply_and_annotate()`` results."""
    comparison_data = []
    # statannotations corrects only the drawn stars, not StatResult.pvalue
    corrected_pvals = holm_corrected_pvalues(test_results)
    kept = 0
    for annotation in test_results:
        if not (hasattr(annotation, "data") and hasattr(annotation.data, "pvalue")):
            continue
        group1_raw = getattr(annotation.data, "group1", None)
        group2_raw = getattr(annotation.data, "group2", None)
        if isinstance(group1_raw, tuple) and isinstance(group2_raw, tuple):
            g1_desc = " ".join(str(x) for x in group1_raw)
            g2_desc = " ".join(str(x) for x in group2_raw)
            comparison_desc = f"{g1_desc} vs {g2_desc}"
        else:
            comparison_desc = f"{group1_raw} vs {group2_raw}"

        pval = annotation.data.pvalue
        stat_val = getattr(annotation.data, "stat_value", "N/A")
        test_name = getattr(annotation.data, "test_short_name", "Unknown")
        pval_corrected = corrected_pvals[kept]
        kept += 1
        significance = significance_stars(pval_corrected)

        comparison_data.append(
            {
                "Comparison": f"{analysis_name}: {comparison_desc}",
                "Test": test_name,
                "Statistic": f"{stat_val:.3f}" if stat_val != "N/A" else "N/A",
                "P_value": format_pvalue(pval_corrected),
                "P_value_uncorrected": format_pvalue(pval),
                "Correction": "holm",
                "Significance": significance,
            }
        )
    return pd.DataFrame(comparison_data)


def create_panel(df):
    """Figure S13B: AI vs Human log-RT violin by category."""
    console.print("Creating Figure S13B: AI vs Human RT violin...")
    ai_irts = df[df["sourceType"] == "ai"][COLUMNS].copy()
    hh_irts = df[(df["dyadType"] == "hh") & (df["sourceType"] == "human")][COLUMNS].copy()
    data = pd.concat([hh_irts, ai_irts], ignore_index=True)
    data["sourceType"] = data["sourceType"].replace({"ai": "AI", "human": "Human"})

    pairs = [[("animals", "AI"), ("animals", "Human")], [("clothes", "AI"), ("clothes", "Human")]]
    plot_args = {"data": data, "x": "category", "y": "log_irt", "hue": "sourceType"}

    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.violinplot(**plot_args, palette=VIOLIN_COLOR_MAP, ax=ax)
    annotator = Annotator(ax, pairs, **plot_args)
    _, test_results = annotator.configure(
        test="t-test_ind", comparisons_correction="holm", loc="inside", text_format="star", line_height=0.2
    ).apply_and_annotate()

    descriptive = calculate_descriptive_stats_irt(data, "log_irt", ["category", "sourceType"])
    comparison = extract_statistics_from_annotator(test_results, "AI vs Human IRT Comparison")
    console.print("\n=== AI VS HUMAN IRT DESCRIPTIVE STATISTICS ===")
    console.print(descriptive.to_string(index=False))
    console.print("\n=== AI VS HUMAN IRT STATISTICAL COMPARISONS ===")
    console.print(comparison.to_string(index=False))

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.xlabel("Category")
    plt.ylabel("Log response times\n(seconds)")
    plt.grid(axis="y")
    plt.tight_layout()
    return fig, descriptive, comparison


@app.command()
def plot(
    data_root: pathlib.Path = typer.Option(
        Path("."), "--data-root", help="Root directory for relative input data paths."
    ),
    output_dir: pathlib.Path = typer.Option(
        None,
        "--output-dir",
        help="Directory to save plots and stats. Defaults to project_root/outputs/figures/supp/artificial-delay.",
    ),
    formats: str | None = typer.Option(
        None, "--formats", help="Comma-separated output formats: png, pdf, svg. Defaults to png."
    ),
    show: bool = typer.Option(False, "--show/--no-show", help="Show the plots instead of just saving."),
):
    """Generate the AI vs Human RT violin panel (Figure S13B)."""
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
        output_dir = project_root / "outputs" / "figures" / "supp" / "artificial-delay"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_context("paper", font_scale=1.8)
    apply_bold_axis_style()

    df = load_data(data_dir=data_dir, cross_condition_only=False, include_solo=True)
    fig, descriptive, comparison = create_panel(df)
    paths = save_figure_formats(
        fig,
        output_dir / "figS13b_ai_vs_human_irt_violin",
        formats=formats,
        default=("png",),
        dpi=300,
        bbox_inches="tight",
    )
    console.print(f"[green]Saved Figure S13B to {', '.join(str(p) for p in paths)}[/green]")
    descriptive.to_csv(output_dir / "figS13b_ai_vs_human_irt_descriptive.csv", index=False)
    comparison.to_csv(output_dir / "figS13b_ai_vs_human_irt_comparisons.csv", index=False)

    if show:
        plt.show()

    console.print("\n[bold]Finished![/bold]")


if __name__ == "__main__":
    seed_everything()
    app()
