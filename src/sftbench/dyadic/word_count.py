"""Dyadic word-count SI figures.
Panels produced:
  * Figure S9  -- number of words by dyadic condition, split by category.
  * Figure S12A -- number of words by dyadic condition, categories aggregated.
  * Figure S12B -- paired Human-Human vs Human-AI word counts by prompt type.
"""

from __future__ import annotations

import pathlib
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console
from scipy import stats
from scipy.stats import ttest_rel
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests

from sftbench import find_project_root
from sftbench.dyadic.load_data import load_data
from sftbench.dyadic.utils import COLOR_MAP, LABEL_MAP, format_pvalue, holm_corrected_pvalues, significance_stars
from sftbench.figure_output import apply_bold_axis_style, save_figure_formats
from sftbench.reproducibility import seed_everything

app = typer.Typer()
console = Console()

DATA_DIR_REL = "data/dyadic/conceptnet"

PANEL_CHOICES = ("s9", "s12a", "s12b")


def _parse_panels(panels: str | None) -> set[str]:
    """Parse a comma-separated panel selector; ``None`` means all panels."""
    if panels is None:
        return set(PANEL_CHOICES)
    selected = {p.strip().lower() for p in panels.split(",") if p.strip()}
    unknown = selected - set(PANEL_CHOICES)
    if unknown:
        raise typer.BadParameter(
            f"Unknown panel(s): {', '.join(sorted(unknown))}. Choose from {', '.join(PANEL_CHOICES)}."
        )
    return selected


ORDER = ["solo_solo", "hh_instructions", "ha_inferred", "ha_convergent", "ha_divergent"]
PAIRS = [
    ("solo_solo", "hh_instructions"),
    ("solo_solo", "ha_inferred"),
    ("solo_solo", "ha_convergent"),
    ("solo_solo", "ha_divergent"),
    ("hh_instructions", "ha_inferred"),
    ("hh_instructions", "ha_convergent"),
    ("hh_instructions", "ha_divergent"),
    ("ha_inferred", "ha_convergent"),
    ("ha_inferred", "ha_divergent"),
    ("ha_convergent", "ha_divergent"),
]


# helpers


def calculate_descriptive_stats_wordcount(data, value_col="n_words", group_cols=("dyad_prompt",)):
    """Descriptive statistics for word-count data with flexible grouping."""
    group_cols = list(group_cols)
    stats_list: list[dict[str, object]] = []
    for name, group in data.groupby(group_cols):
        values = group[value_col].dropna()
        if len(values) == 0:
            continue

        if len(group_cols) == 1:
            group_desc = LABEL_MAP.get(name, f"{name}")
        else:
            group_parts = []
            for i in range(len(group_cols)):
                val = name[i] if isinstance(name, tuple) else name
                group_parts.append(LABEL_MAP.get(val, str(val)))
            group_desc = " - ".join(group_parts)

        n = len(values)
        mean_val = values.mean()
        std_val = values.std()
        ci = stats.t.interval(0.95, n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
        median_val = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        stats_list.append(
            {
                "Group": group_desc,
                "N": n,
                "Mean_95CI": f"{mean_val:.1f} ({ci[0]:.1f}, {ci[1]:.1f})",
                "Median_IQR": f"{median_val:.1f} ({q1:.1f}, {q3:.1f})",
                "Min_Max": f"{values.min():.0f}, {values.max():.0f}",
            }
        )

    return pd.DataFrame(stats_list)


def extract_statistics_from_annotator_wordcount(test_results, analysis_name):
    """Pull p-values/statistics out of ``apply_and_annotate()`` results."""
    comparison_data = []
    # statannotations corrects only the drawn stars, not StatResult.pvalue
    corrected_pvals = holm_corrected_pvalues(test_results)
    kept = 0
    for i, annotation in enumerate(test_results):
        if not (hasattr(annotation, "data") and hasattr(annotation.data, "pvalue")):
            continue
        group1_raw = getattr(annotation.data, "group1", None)
        group2_raw = getattr(annotation.data, "group2", None)
        group1 = group1_raw[0] if isinstance(group1_raw, tuple) and group1_raw else group1_raw
        group2 = group2_raw[0] if isinstance(group2_raw, tuple) and group2_raw else group2_raw

        if group1 and group2:
            comparison_desc = f"{LABEL_MAP.get(group1, group1)} vs {LABEL_MAP.get(group2, group2)}"
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


# data preparation


def build_combined_df(df):
    """Add ``dyad_prompt`` and keep solo/hh/ha rows."""
    df = df.copy()
    df.loc[df.dyadType == "hh", "prompt"] = "instructions"
    df.loc[df.dyadType == "solo", "prompt"] = "solo"
    combined = df[df["dyadType"].isin(["hh", "ha", "solo"])].copy()
    combined["dyad_prompt"] = combined["dyadType"].astype(str) + "_" + combined["prompt"].astype(str)
    return combined


def nominal_counts_by_category(solo_data):
    """Union word counts over every unordered solo pair, per category."""
    rows = []
    for category, players in solo_data.groupby("category")["source"].unique().items():
        for player1, player2 in combinations(players, 2):
            dyad_id = f"{min(player1, player2)}-{max(player1, player2)}"
            p1_words = solo_data[(solo_data["source"] == player1) & (solo_data["category"] == category)][
                "text"
            ].tolist()
            p2_words = solo_data[(solo_data["source"] == player2) & (solo_data["category"] == category)][
                "text"
            ].tolist()
            rows.append(
                {
                    "playerID": dyad_id,
                    "category": category,
                    "n_words": len(set(p1_words).union(set(p2_words))),
                    "dyad_prompt": "solo_solo",
                }
            )
    return pd.DataFrame(rows)


def build_collapsed_counts(combined_df):
    """Collab counts collapsed across categories + per-category nominal solo counts."""
    collab = combined_df[combined_df["dyadType"].isin(["hh", "ha"])]
    collab_counts = collab.groupby(["playerID", "dyad_prompt"]).size().reset_index(name="n_words")

    solo_data = combined_df[combined_df["dyadType"] == "solo"]
    nominal = nominal_counts_by_category(solo_data)

    parts = [collab_counts[["dyad_prompt", "n_words"]]]
    if not nominal.empty:
        parts.append(nominal[["dyad_prompt", "n_words"]])
    return pd.concat(parts, ignore_index=True)


def build_category_counts(combined_df):
    """Collab and nominal solo counts, category preserved."""
    combined_df = combined_df[combined_df["category"].isin(["animals", "clothes"])]
    collab = combined_df[combined_df["dyadType"].isin(["hh", "ha"])]
    collab_counts = collab.groupby(["playerID", "dyad_prompt", "category"]).size().reset_index(name="n_words")

    solo_data = combined_df[combined_df["dyadType"] == "solo"]
    nominal = nominal_counts_by_category(solo_data)

    parts = [collab_counts[["dyad_prompt", "n_words", "category"]]]
    if not nominal.empty:
        parts.append(nominal[["dyad_prompt", "n_words", "category"]])
    return pd.concat(parts, ignore_index=True)


# panels


def create_collapsed_panel(all_word_counts):
    """Figure S12A: word counts by condition, categories aggregated."""
    console.print("Creating Figure S12A: collapsed word counts...")
    conditions_in_data = all_word_counts["dyad_prompt"].unique()

    fig = plt.figure(figsize=(10, 10))
    sns.despine(ax=plt.gca(), right=True, top=True)
    ax = sns.boxplot(data=all_word_counts, x="dyad_prompt", y="n_words", order=ORDER, palette=COLOR_MAP)

    pairs = [p for p in PAIRS if p[0] in conditions_in_data and p[1] in conditions_in_data]
    descriptive = pd.DataFrame()
    comparison = pd.DataFrame()
    if pairs:
        annotator = Annotator(ax=ax, pairs=pairs, data=all_word_counts, x="dyad_prompt", y="n_words", order=ORDER)
        _, test_results = annotator.configure(
            test="Mann-Whitney", comparisons_correction="holm", text_format="star", loc="inside"
        ).apply_and_annotate()
        descriptive = calculate_descriptive_stats_wordcount(all_word_counts, "n_words", ["dyad_prompt"])
        comparison = extract_statistics_from_annotator_wordcount(test_results, "Collapsed Categories")
        console.print("\n=== COLLAPSED CATEGORIES DESCRIPTIVE STATISTICS ===")
        console.print(descriptive.to_string(index=False))
        console.print("\n=== COLLAPSED CATEGORIES STATISTICAL COMPARISONS ===")
        console.print(comparison.to_string(index=False))

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([LABEL_MAP.get(cond, cond) for cond in ORDER], rotation=15, ha="right")
    ax.set_xlabel("Dyadic Condition")
    ax.set_ylabel("Number of Words Produced")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig, descriptive, comparison


def create_by_category_panel(all_word_counts_cat):
    """Figure S9: word counts by condition, split by category."""
    console.print("Creating Figure S9: word counts by category...")
    categories = ["animals", "clothes"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    comparison_frames = []
    for i, category in enumerate(categories):
        category_data = all_word_counts_cat[all_word_counts_cat["category"] == category].copy()
        if category_data.empty:
            continue
        ax = axes[i]
        sns.despine(ax=ax, right=True, top=True)
        conditions_in_data = category_data["dyad_prompt"].unique()

        sns.boxplot(data=category_data, x="dyad_prompt", y="n_words", order=ORDER, palette=COLOR_MAP, ax=ax)

        pairs = [p for p in PAIRS if p[0] in conditions_in_data and p[1] in conditions_in_data]
        if pairs:
            annotator = Annotator(ax=ax, pairs=pairs, data=category_data, x="dyad_prompt", y="n_words", order=ORDER)
            _, test_results = annotator.configure(
                test="Mann-Whitney", comparisons_correction="holm", text_format="star", loc="outside"
            ).apply_and_annotate()
            comp = extract_statistics_from_annotator_wordcount(test_results, f"By Category - {category.title()}")
            comp.insert(0, "category", category)
            comparison_frames.append(comp)

        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([LABEL_MAP.get(cond, cond) for cond in ORDER], rotation=45, ha="right")
        ax.set_title(category.capitalize())
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Number of Words Produced")
        ax.grid(axis="y", alpha=0.3)

    combined_descriptive = calculate_descriptive_stats_wordcount(
        all_word_counts_cat, "n_words", ["category", "dyad_prompt"]
    )
    console.print("\n=== BY CATEGORY COMBINED DESCRIPTIVE STATISTICS ===")
    console.print(combined_descriptive.to_string(index=False))

    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    fig.text(0.5, 0.02, "Dyadic Condition", ha="center", fontsize=plt.rcParams["font.size"])
    comparison = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    return fig, combined_descriptive, comparison


def create_paired_by_prompt_panel(df):
    """Figure S12B: paired HH vs HA word counts by prompt type."""
    console.print("Creating Figure S12B: paired word counts by prompt...")
    prompts = ["inferred", "convergent", "divergent"]
    source_ids_human = df[(df["dyadType"] == "hh") & (df["sourceType"] == "human")]["source"].unique()

    prompt_participants = {}
    for prompt in prompts:
        source_ids_ha = df[(df["dyadType"] == "ha") & (df["sourceType"] == "human") & (df["prompt"] == prompt)][
            "source"
        ].unique()
        prompt_participants[prompt] = set(source_ids_human).intersection(set(source_ids_ha))
        console.print(f"Participants who did both HH and HA-{prompt}: {len(prompt_participants[prompt])}")

    valid_prompts = [p for p in prompts if len(prompt_participants[p]) > 0]
    n_prompts = len(valid_prompts)
    if n_prompts == 0:
        console.print("[yellow]No prompt types have sufficient participants.[/yellow]")
        return None, pd.DataFrame()

    condition_to_dyad_prompt = {
        "human": "hh_instructions",
        "ai_inferred": "ha_inferred",
        "ai_convergent": "ha_convergent",
        "ai_divergent": "ha_divergent",
    }

    fig, axes = plt.subplots(1, n_prompts, figsize=(5 * n_prompts, 6), sharey=True)
    if n_prompts == 1:
        axes = [axes]

    rows = []
    for plot_idx, prompt in enumerate(valid_prompts):
        common_ids = prompt_participants[prompt]
        hh_data = df[(df["dyadType"] == "hh") & (df["sourceType"] == "human") & (df["source"].isin(common_ids))]
        hh_counts = hh_data.groupby(["source", "category"]).size().reset_index(name="n_words")
        hh_counts["comparison_condition"] = "human"

        ha_data = df[
            (df["dyadType"] == "ha")
            & (df["sourceType"] == "human")
            & (df["prompt"] == prompt)
            & (df["source"].isin(common_ids))
        ]
        ha_counts = ha_data.groupby(["source", "category"]).size().reset_index(name="n_words")
        ha_counts["comparison_condition"] = f"ai_{prompt}"

        comparison_data = pd.concat([hh_counts, ha_counts], ignore_index=True)
        ax = axes[plot_idx]
        condition_order = ["human", f"ai_{prompt}"]
        prompt_palette = {
            cond: COLOR_MAP.get(condition_to_dyad_prompt.get(cond, cond), "#d3d3d3") for cond in condition_order
        }

        sns.boxplot(
            data=comparison_data,
            x="comparison_condition",
            y="n_words",
            order=condition_order,
            palette=prompt_palette,
            ax=ax,
        )

        pairs = [("human", f"ai_{prompt}")]
        try:
            annotator = Annotator(
                ax, pairs, data=comparison_data, x="comparison_condition", y="n_words", order=condition_order
            )
            annotator.configure(
                test="t-test_paired", comparisons_correction="holm", text_format="star"
            ).apply_and_annotate()
        except Exception as e:
            console.print(f"[yellow]Warning: annotation failed for {prompt}: {e}[/yellow]")

        new_labels = []
        for label in [t.get_text() for t in ax.get_xticklabels()]:
            if label == "human":
                new_labels.append("Human-Human")
            elif label.startswith("ai_"):
                new_labels.append(f"Human-AI\n({prompt.title()})")
            else:
                new_labels.append(label)
        ax.set_xticklabels(new_labels, rotation=0, ha="center")
        ax.set_xlabel("Dyadic Condition")
        if plot_idx == 0:
            ax.set_ylabel("Number of Words")
        ax.grid(axis="y", alpha=0.3)

        # Paired t-test stats for the manuscript table.
        hh_means = hh_counts.groupby("source")["n_words"].mean()
        ha_means = ha_counts.groupby("source")["n_words"].mean()
        shared = hh_means.index.intersection(ha_means.index)
        t_stat, p_val = ttest_rel(hh_means.loc[shared].values, ha_means.loc[shared].values)
        rows.append({"prompt": prompt, "n_pairs": len(shared), "t": t_stat, "p_uncorrected": p_val})

    plt.tight_layout()

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        reject, p_holm, _, _ = multipletests(stats_df["p_uncorrected"].values, method="holm")
        stats_df["p_holm"] = p_holm
        stats_df["significant"] = reject
        console.print("\n=== PAIRED WORD-COUNT COMPARISONS (Holm-corrected) ===")
        console.print(stats_df.to_string(index=False))
    return fig, stats_df


@app.command()
def plot(
    data_root: pathlib.Path = typer.Option(
        Path("."), "--data-root", help="Root directory for relative input data paths."
    ),
    output_dir: pathlib.Path = typer.Option(
        None,
        "--output-dir",
        help="Directory to save plots and stats. Defaults to project_root/outputs/figures/supp/word-count.",
    ),
    formats: str | None = typer.Option(
        None, "--formats", help="Comma-separated output formats: png, pdf, svg. Defaults to png."
    ),
    panels: str | None = typer.Option(
        None,
        "--panels",
        help="Comma-separated panels to generate: s9, s12a, s12b. Defaults to all.",
    ),
    show: bool = typer.Option(False, "--show/--no-show", help="Show the plots instead of just saving."),
):
    """Generate the dyadic word-count panels (Figures S9, S12A, S12B)."""
    selected_panels = _parse_panels(panels)
    warnings.filterwarnings("ignore")

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
        output_dir = project_root / "outputs" / "figures" / "supp" / "word-count"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_dir=data_dir, cross_condition_only=False, include_solo=True)
    combined_df = build_combined_df(df)

    # Figure S12A: collapsed across categories.
    if "s12a" in selected_panels:
        sns.set_context("paper", font_scale=2.0)
        apply_bold_axis_style()
        fig_a, desc_a, comp_a = create_collapsed_panel(build_collapsed_counts(combined_df))
        paths = save_figure_formats(
            fig_a,
            output_dir / "figS12a_word_count_collapsed",
            formats=formats,
            default=("png",),
            dpi=300,
            bbox_inches="tight",
        )
        console.print(f"[green]Saved Figure S12A to {', '.join(str(p) for p in paths)}[/green]")
        desc_a.to_csv(output_dir / "figS12a_collapsed_descriptive.csv", index=False)
        comp_a.to_csv(output_dir / "figS12a_collapsed_comparisons.csv", index=False)

    # Figure S9: split by category.
    if "s9" in selected_panels:
        sns.set_context("paper", font_scale=2.2)
        apply_bold_axis_style()
        fig_s9, desc_s9, comp_s9 = create_by_category_panel(build_category_counts(combined_df))
        paths = save_figure_formats(
            fig_s9,
            output_dir / "figS9_word_count_by_category",
            formats=formats,
            default=("png",),
            dpi=300,
            bbox_inches="tight",
        )
        console.print(f"[green]Saved Figure S9 to {', '.join(str(p) for p in paths)}[/green]")
        desc_s9.to_csv(output_dir / "figS9_by_category_descriptive.csv", index=False)
        comp_s9.to_csv(output_dir / "figS9_by_category_comparisons.csv", index=False)

    # Figure S12B: paired by prompt type.
    if "s12b" in selected_panels:
        fig_b, stats_b = create_paired_by_prompt_panel(df)
        if fig_b is not None:
            paths = save_figure_formats(
                fig_b,
                output_dir / "figS12b_word_count_paired_by_prompt",
                formats=formats,
                default=("png",),
                dpi=300,
                bbox_inches="tight",
            )
            console.print(f"[green]Saved Figure S12B to {', '.join(str(p) for p in paths)}[/green]")
            stats_b.to_csv(output_dir / "figS12b_paired_by_prompt_stats.csv", index=False)

    if show:
        plt.show()

    console.print("\n[bold]Finished![/bold]")


if __name__ == "__main__":
    seed_everything()
    app()
