"""Dyadic self vs other semantic-similarity SI figure (Figure S11).

Panels produced:
  * Figure S11A -- cosine similarity (self vs other) by dyadic condition.
  * Figure S11B -- the same, split by category (animals, clothes).

Uses precomputed ``self_similarity``/``other_similarity`` columns from ``load_data``.
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
from sftbench.dyadic.utils import LABEL_MAP, format_pvalue, holm_corrected_pvalues, significance_stars
from sftbench.figure_output import apply_bold_axis_style, save_figure_formats
from sftbench.reproducibility import seed_everything

app = typer.Typer()
console = Console()

DATA_DIR_REL = "data/dyadic/conceptnet"

SIMILARITY_TYPE_PALETTE = {"self": "#28ae2f", "other": "#b234d5"}
CONDITION_ORDER = ["solo", "collab_human", "collab_ai_inferred", "collab_ai_convergent", "collab_ai_divergent"]
PAIRS = [
    [("collab_human", "self"), ("collab_human", "other")],
    [("collab_ai_inferred", "self"), ("collab_ai_inferred", "other")],
    [("collab_ai_convergent", "self"), ("collab_ai_convergent", "other")],
    [("collab_ai_divergent", "self"), ("collab_ai_divergent", "other")],
    [("solo", "self"), ("collab_human", "self")],
    [("solo", "self"), ("collab_ai_inferred", "self")],
    [("solo", "self"), ("collab_ai_convergent", "self")],
    [("solo", "self"), ("collab_ai_divergent", "self")],
]


def prepare_similarity_data(df):
    """Long-format self/other cosine similarities for human words only."""
    results = []
    human_df = df[df["sourceType"] == "human"].copy()
    for _, row in human_df.iterrows():
        dyad_type = row["dyadType"]
        if dyad_type == "solo":
            condition = "solo"
        elif dyad_type == "hh":
            condition = "collab_human"
        elif dyad_type == "ha":
            condition = f"collab_ai_{row.get('prompt', 'inferred')}"
        else:
            continue

        base = {
            "playerID": row["playerID"],
            "category": row["category"],
            "condition": condition,
            "word_index": row["word_index"],
        }
        if pd.notna(row["self_similarity"]):
            results.append({**base, "similarity_type": "self", "cosine_similarity": row["self_similarity"]})
        if pd.notna(row["other_similarity"]):
            results.append({**base, "similarity_type": "other", "cosine_similarity": row["other_similarity"]})

    return pd.DataFrame(results)


def _format_group(group_raw):
    """Render a ``(condition, similarity_type)`` hue pair as a readable label."""
    if isinstance(group_raw, tuple):
        condition, sim_type = (group_raw + (None, None))[:2]
        label = LABEL_MAP.get(condition, condition)
        if sim_type is not None:
            label = f"{label} - {str(sim_type).capitalize()}"
        return label
    return LABEL_MAP.get(group_raw, group_raw)


def extract_statistics_from_annotator(test_results, analysis_name):
    """Pull p-values/statistics out of ``apply_and_annotate()`` results."""
    comparison_data = []
    # statannotations corrects only the drawn stars, not StatResult.pvalue
    corrected_pvals = holm_corrected_pvalues(test_results)
    kept = 0
    for i, annotation in enumerate(test_results):
        if not (hasattr(annotation, "data") and hasattr(annotation.data, "pvalue")):
            continue
        group1 = getattr(annotation.data, "group1", None)
        group2 = getattr(annotation.data, "group2", None)

        if group1 is not None and group2 is not None:
            comparison_desc = f"{_format_group(group1)} vs {_format_group(group2)}"
        else:
            comparison_desc = f"Unknown Comparison {i + 1}"

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


def plot_self_vs_other(data):
    """Figure S11A: self vs other similarity across conditions."""
    console.print("Creating Figure S11A: self vs other similarities...")
    fig = plt.figure(figsize=(16, 8))
    ax = sns.boxplot(
        data=data,
        x="condition",
        y="cosine_similarity",
        hue="similarity_type",
        palette=SIMILARITY_TYPE_PALETTE,
        order=CONDITION_ORDER,
    )
    annotator = Annotator(
        ax,
        PAIRS,
        data=data,
        x="condition",
        y="cosine_similarity",
        hue="similarity_type",
        order=CONDITION_ORDER,
    )
    _, test_results = annotator.configure(
        test="Mann-Whitney", comparisons_correction="holm", text_format="star", loc="inside"
    ).apply_and_annotate()
    comparison = extract_statistics_from_annotator(test_results, "Self vs Other")

    # Label every slot seaborn drew. It lays out one per entry in `order`, whether or
    # not the data reaches it, so labelling only the conditions present would shift the
    # labels off their bars the moment a condition is missing.
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels([LABEL_MAP.get(c, c) for c in CONDITION_ORDER], rotation=30, ha="right")
    ax.set_xlabel("Dyadic Condition")
    ax.set_ylabel("Cosine Similarity")
    ax.legend(title="Comparison Target", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig, comparison


def plot_self_vs_other_by_category(data):
    """Figure S11B: self vs other similarity split by category."""
    console.print("Creating Figure S11B: self vs other similarities by category...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)
    comparisons = []
    for idx, category in enumerate(["animals", "clothes"]):
        category_data = data[data["category"] == category]
        present_pairs = {(row["condition"], row["similarity_type"]) for _, row in category_data.iterrows()}
        category_pairs = [p for p in PAIRS if p[0] in present_pairs and p[1] in present_pairs]

        ax = sns.boxplot(
            data=category_data,
            x="condition",
            y="cosine_similarity",
            hue="similarity_type",
            palette=SIMILARITY_TYPE_PALETTE,
            order=CONDITION_ORDER,
            ax=axes[idx],
        )
        annotator = Annotator(
            ax,
            category_pairs,
            data=category_data,
            x="condition",
            y="cosine_similarity",
            hue="similarity_type",
            order=CONDITION_ORDER,
        )
        _, test_results = annotator.configure(
            test="Mann-Whitney", comparisons_correction="holm", text_format="star", loc="inside"
        ).apply_and_annotate()
        comparisons.append(extract_statistics_from_annotator(test_results, f"By Category - {category.capitalize()}"))

        ax.set_title(category.capitalize())
        ax.set_xlabel("")
        if idx == 0:
            ax.set_ylabel("Cosine Similarity")
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels([LABEL_MAP.get(c, c) for c in CONDITION_ORDER], rotation=30, ha="right")
        if ax.get_legend():
            ax.get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.02, "Dyadic Condition", ha="center", fontsize=plt.rcParams["font.size"])
    comparison = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    return fig, comparison


def calculate_descriptive_stats(data, value_col="cosine_similarity"):
    """Descriptive statistics per condition x similarity type."""
    stats_list = []
    for (condition, sim_type), group in data.groupby(["condition", "similarity_type"]):
        values = group[value_col].dropna()
        if len(values) == 0:
            continue
        n = len(values)
        mean_val = values.mean()
        std_val = values.std()
        ci = stats.t.interval(0.95, n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
        stats_list.append(
            {
                "Condition": LABEL_MAP.get(condition, condition),
                "Similarity_Type": sim_type.capitalize(),
                "Group": f"{LABEL_MAP.get(condition, condition)} - {sim_type.capitalize()}",
                "N": n,
                "Mean_95CI": f"{mean_val:.3f} ({ci[0]:.3f}, {ci[1]:.3f})",
                "Median_IQR": f"{values.median():.3f} ({values.quantile(0.25):.3f}, {values.quantile(0.75):.3f})",
                "Min_Max": f"{values.min():.3f}, {values.max():.3f}",
            }
        )
    return pd.DataFrame(stats_list)


@app.command()
def plot(
    data_root: pathlib.Path = typer.Option(
        Path("."), "--data-root", help="Root directory for relative input data paths."
    ),
    output_dir: pathlib.Path = typer.Option(
        None,
        "--output-dir",
        help="Directory to save plots and stats. Defaults to project_root/outputs/figures/supp/self-other-similarity.",
    ),
    formats: str | None = typer.Option(
        None, "--formats", help="Comma-separated output formats: png, pdf, svg. Defaults to png."
    ),
    show: bool = typer.Option(False, "--show/--no-show", help="Show the plots instead of just saving."),
):
    """Generate the self vs other similarity panels (Figure S11)."""
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
        output_dir = project_root / "outputs" / "figures" / "supp" / "self-other-similarity"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_context("paper", font_scale=2.2)
    apply_bold_axis_style()

    df = load_data(data_dir=data_dir, cross_condition_only=False, include_solo=True)
    similarity_data = prepare_similarity_data(df)

    console.print("\nMean similarities by condition and type:")
    console.print(similarity_data.groupby(["condition", "similarity_type"])["cosine_similarity"].mean().to_string())

    fig_a, comp_a = plot_self_vs_other(similarity_data)
    paths = save_figure_formats(
        fig_a,
        output_dir / "figS11a_self_vs_other_similarities",
        formats=formats,
        default=("png",),
        dpi=300,
        bbox_inches="tight",
    )
    console.print(f"[green]Saved Figure S11A to {', '.join(str(p) for p in paths)}[/green]")

    fig_b, comp_b = plot_self_vs_other_by_category(similarity_data)
    paths = save_figure_formats(
        fig_b,
        output_dir / "figS11b_self_vs_other_by_category",
        formats=formats,
        default=("png",),
        dpi=300,
        bbox_inches="tight",
    )
    console.print(f"[green]Saved Figure S11B to {', '.join(str(p) for p in paths)}[/green]")

    descriptive = calculate_descriptive_stats(similarity_data)
    console.print("\n=== DESCRIPTIVE STATISTICS ===")
    console.print(descriptive.to_string(index=False))
    descriptive.to_csv(output_dir / "figS11_descriptive_stats.csv", index=False)

    console.print("\n=== STATISTICAL COMPARISONS ===")
    console.print(comp_a.to_string(index=False))
    comp_a.to_csv(output_dir / "figS11a_self_vs_other_comparisons.csv", index=False)
    console.print(comp_b.to_string(index=False))
    comp_b.to_csv(output_dir / "figS11b_self_vs_other_by_category_comparisons.csv", index=False)

    if show:
        plt.show()

    console.print("\n[bold]Finished![/bold]")


if __name__ == "__main__":
    seed_everything()
    app()
