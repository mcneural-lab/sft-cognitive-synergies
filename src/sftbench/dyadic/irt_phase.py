"""Dyadic response-time / similarity SI figures (task phase & switching).

Panels produced (assembled into manuscript figures in the composite source):
  * Figure S10 -- human log response times (z-scored) by task phase
    (early/late), categories collapsed.
  * Figure S12C -- paired Human-Human vs Human-AI response times by prompt type.
  * Figure S14 -- switch vs no-switch cosine similarity and response times,
    4-panel (category x metric).

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

CONDITION_ORDER = ["solo", "collab_human", "collab_ai_inferred", "collab_ai_convergent", "collab_ai_divergent"]
CATEGORIES = ["animals", "clothes"]
PANEL_CHOICES = ("s10", "s12c", "s14")


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


def prepare_df(df):
    """Filter to human, non-supermarket words and add condition columns."""
    df = df[(df["sourceType"] == "human") & (df["category"] != "supermarket")].copy()
    df["condition"] = df["dyadType"].map({"solo": "solo", "hh": "collab_human", "ha": "collab_ai"})
    df.loc[df["dyadType"] == "ha", "condition_detailed"] = "collab_ai_" + df["prompt"]
    df.loc[df["dyadType"] == "hh", "condition_detailed"] = "collab_human"
    df.loc[df["dyadType"] == "solo", "condition_detailed"] = "solo"
    df["condition_detailed"] = pd.Categorical(df["condition_detailed"], categories=CONDITION_ORDER, ordered=True)
    return df


def calculate_switch_descriptive_stats(data, category, metric, value_col):
    """Descriptive statistics per switch label x dyadic condition for one panel."""
    stats_list = []
    for (switch_label, condition), group in data.groupby(["switch_label", "condition_detailed"], observed=True):
        values = group[value_col].dropna()
        if len(values) == 0:
            continue
        n = len(values)
        mean_val = values.mean()
        std_val = values.std()
        ci = stats.t.interval(0.95, n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
        stats_list.append(
            {
                "Category": category.title(),
                "Metric": metric,
                "Exemplar_Type": "Switch" if switch_label == "switch" else "Not Switch",
                "Condition": LABEL_MAP.get(condition, condition),
                "N": n,
                "Mean_95CI": f"{mean_val:.3f} ({ci[0]:.3f}, {ci[1]:.3f})",
                "Median_IQR": f"{values.median():.3f} ({values.quantile(0.25):.3f}, {values.quantile(0.75):.3f})",
                "Min_Max": f"{values.min():.3f}, {values.max():.3f}",
            }
        )
    return pd.DataFrame(stats_list)


def create_switch_panel(df):
    """Figure S14: switch vs no-switch, 4-panel (category x metric).

    Returns the figure plus descriptive-statistics and comparison DataFrames
    (one row group per category x metric panel), mirroring the other SI figures.
    """
    console.print("Creating Figure S14: switch vs no-switch 4-panel...")
    switch_df = df[df["word_index"] != 0].dropna(subset=["switch_sim"]).copy()
    switch_df["switch_label"] = switch_df["switch_sim"].astype(int).map({0: "no_switch", 1: "switch"})

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ("embedding_similarity", "Cosine Similarity\nto Previous Word", "Cosine Similarity"),
        ("log_irt_zscore", "Log Response Time\n(z-scored)", "Log Response Time"),
    ]
    pairs = [
        (("no_switch", "solo"), ("no_switch", "collab_human")),
        (("no_switch", "solo"), ("no_switch", "collab_ai_convergent")),
        (("no_switch", "solo"), ("no_switch", "collab_ai_divergent")),
        (("no_switch", "solo"), ("no_switch", "collab_ai_inferred")),
        (("no_switch", "collab_ai_inferred"), ("no_switch", "collab_human")),
        (("no_switch", "collab_ai_divergent"), ("no_switch", "collab_human")),
        (("no_switch", "collab_ai_convergent"), ("no_switch", "collab_human")),
        (("switch", "solo"), ("switch", "collab_human")),
        (("switch", "solo"), ("switch", "collab_ai_convergent")),
        (("switch", "solo"), ("switch", "collab_ai_divergent")),
        (("switch", "solo"), ("switch", "collab_ai_inferred")),
        (("switch", "collab_ai_inferred"), ("switch", "collab_human")),
        (("switch", "collab_ai_divergent"), ("switch", "collab_human")),
        (("switch", "collab_ai_convergent"), ("switch", "collab_human")),
    ]

    descriptive_frames, comparison_frames = [], []
    for row_idx, (metric, metric_label, metric_short) in enumerate(metrics):
        for col_idx, category in enumerate(CATEGORIES):
            ax = axes[row_idx, col_idx]
            category_data = switch_df[switch_df["category"] == category]
            plot_args = {
                "data": category_data,
                "x": "switch_label",
                "y": metric,
                "hue": "condition_detailed",
                "hue_order": CONDITION_ORDER,
            }
            sns.boxplot(**plot_args, palette=COLOR_MAP, ax=ax)
            annotator = Annotator(ax, pairs, **plot_args)
            _, test_results = annotator.configure(
                test="Mann-Whitney", comparisons_correction="holm", loc="inside", text_format="star"
            ).apply_and_annotate()

            analysis_name = f"{category.title()} - {metric_short}"
            descriptive_frames.append(calculate_switch_descriptive_stats(category_data, category, metric_short, metric))
            comparison_frames.append(extract_statistics_from_annotator(test_results, analysis_name))

            ax.set_title(category.title())
            ax.set_xlabel("Exemplar Type" if row_idx == 1 else "")
            ax.set_ylabel(metric_label if col_idx == 0 else "")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Not Switch", "Switch"])
            ax.grid(axis="y", alpha=0.3)

            if row_idx == 0 and col_idx == 1:
                legend_elements = [
                    plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP[key], edgecolor="black", label=LABEL_MAP[key])
                    for key in CONDITION_ORDER
                    if key in COLOR_MAP and key in category_data["condition_detailed"].values
                ]
                ax.legend(
                    handles=legend_elements,
                    title="Dyadic Condition",
                    bbox_to_anchor=(1.05, 1),
                    loc=2,
                    borderaxespad=0.0,
                )
            elif ax.get_legend():
                ax.get_legend().remove()

    plt.tight_layout()

    descriptive = pd.concat(descriptive_frames, ignore_index=True) if descriptive_frames else pd.DataFrame()
    comparison = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    console.print("\n=== SWITCH VS NO-SWITCH DESCRIPTIVE STATISTICS ===")
    console.print(descriptive.to_string(index=False))
    console.print("\n=== SWITCH VS NO-SWITCH COMPARISONS (Holm-corrected) ===")
    console.print(comparison.to_string(index=False))
    return fig, descriptive, comparison


def _format_group(group_raw):
    """Render a ``(task_portion, condition)`` hue pair as a readable label."""
    if isinstance(group_raw, tuple):
        portion, condition = (group_raw + (None, None))[:2]
        label = LABEL_MAP.get(condition, condition)
        if portion is not None:
            label = f"{label} - {str(portion).capitalize()}"
        return label
    return LABEL_MAP.get(group_raw, group_raw)


def calculate_descriptive_stats(data, value_col="log_irt_zscore"):
    """Descriptive statistics per task phase x dyadic condition."""
    stats_list = []
    for (portion, condition), group in data.groupby(["task_portion", "condition_detailed"], observed=True):
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
                "Task_Phase": str(portion).capitalize(),
                "Group": f"{LABEL_MAP.get(condition, condition)} - {str(portion).capitalize()}",
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


def create_task_portion_panel(df):
    """Figure S10: log response times by task phase, categories collapsed."""
    console.print("Creating Figure S10: task phase (categories collapsed)...")
    df = df.copy()
    df["task_portion"] = df["word_index_split"].map({False: "early", True: "late"})

    fig, ax = plt.subplots(figsize=(12, 6))
    pairs = [
        (("early", "solo"), ("late", "solo")),
        (("early", "collab_human"), ("late", "collab_human")),
        (("early", "collab_ai_convergent"), ("late", "collab_ai_convergent")),
        (("early", "collab_ai_divergent"), ("late", "collab_ai_divergent")),
        (("early", "collab_ai_inferred"), ("late", "collab_ai_inferred")),
    ]

    plot_args = {
        "data": df,
        "x": "task_portion",
        "y": "log_irt_zscore",
        "hue": "condition_detailed",
        "hue_order": CONDITION_ORDER,
    }
    sns.boxplot(**plot_args, palette=COLOR_MAP, ax=ax)
    annotator = Annotator(ax, pairs, **plot_args)
    _, test_results = annotator.configure(
        test="Mann-Whitney", comparisons_correction="holm", loc="inside", text_format="star", line_height=0.2
    ).apply_and_annotate()

    ax.set_xlabel("Task Phase")
    ax.set_ylabel("Human log response times\n(z-scored)")
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP[key], edgecolor="black", label=LABEL_MAP[key])
        for key in CONDITION_ORDER
        if key in COLOR_MAP and key in df["condition_detailed"].values
    ]
    ax.legend(handles=legend_elements, title="Dyadic Condition", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.tight_layout()

    descriptive = calculate_descriptive_stats(df)
    comparison = extract_statistics_from_annotator(test_results, "Task Phase")
    console.print("\n=== TASK PHASE DESCRIPTIVE STATISTICS ===")
    console.print(descriptive.to_string(index=False))
    console.print("\n=== TASK PHASE COMPARISONS (Holm-corrected) ===")
    console.print(comparison.to_string(index=False))
    return fig, descriptive, comparison


def create_paired_by_prompt_panel(df):
    """Figure S12C: paired HH vs HA response times by prompt type."""
    console.print("Creating Figure S12C: paired response times by prompt...")
    prompts = ["inferred", "convergent", "divergent"]
    hh_sources = set(df[df["dyadType"] == "hh"]["source"].unique())

    prompt_data = []
    for prompt in prompts:
        ha_sources = set(df[(df["dyadType"] == "ha") & (df["prompt"] == prompt)]["source"].unique())
        common_sources = hh_sources.intersection(ha_sources)
        if len(common_sources) > 0:
            prompt_data.append({"prompt": prompt, "common_sources": common_sources})
            console.print(f"Participants who did both HH and HA-{prompt}: {len(common_sources)}")

    if not prompt_data:
        console.print("[yellow]No prompt types have sufficient participants.[/yellow]")
        return None, pd.DataFrame()

    all_pvals, all_prompts, all_test_stats, all_comparison_data = [], [], [], []
    for info in prompt_data:
        prompt = info["prompt"]
        common_sources = info["common_sources"]
        hh_data = df[(df["dyadType"] == "hh") & (df["source"].isin(common_sources))].copy()
        hh_data["comparison_condition"] = "human_collab"
        ha_data = df[(df["dyadType"] == "ha") & (df["prompt"] == prompt) & (df["source"].isin(common_sources))].copy()
        ha_data["comparison_condition"] = f"ai_{prompt}"

        comparison_agg = (
            pd.concat([hh_data, ha_data], ignore_index=True)
            .groupby(["source", "comparison_condition"])["log_irt_zscore"]
            .mean()
            .reset_index()
        )
        hh_means = comparison_agg[comparison_agg["comparison_condition"] == "human_collab"]["log_irt_zscore"].values
        ha_means = comparison_agg[comparison_agg["comparison_condition"] == f"ai_{prompt}"]["log_irt_zscore"].values
        t_stat, p_val = ttest_rel(hh_means, ha_means)

        all_pvals.append(p_val)
        all_prompts.append(prompt)
        all_test_stats.append(t_stat)
        all_comparison_data.append(comparison_agg)

    reject, pvals_corrected, _, _ = multipletests(all_pvals, method="holm")

    n_valid = len(prompt_data)
    fig, axes = plt.subplots(1, n_valid, figsize=(5 * n_valid, 6), sharey=True)
    if n_valid == 1:
        axes = [axes]

    for plot_idx, (info, comparison_agg) in enumerate(zip(prompt_data, all_comparison_data, strict=True)):
        prompt = info["prompt"]
        ax = axes[plot_idx]
        sns.despine(ax=ax, right=True, top=True)
        condition_order = ["human_collab", f"ai_{prompt}"]
        palette = {"human_collab": "#1f77b4", f"ai_{prompt}": "#ff7f0e"}

        sns.barplot(
            data=comparison_agg,
            x="comparison_condition",
            y="log_irt_zscore",
            order=condition_order,
            hue="comparison_condition",
            palette=palette,
            legend=False,
            ax=ax,
            errorbar="se",
            estimator=np.mean,
        )
        sns.stripplot(
            x="comparison_condition",
            y="log_irt_zscore",
            data=comparison_agg,
            order=condition_order,
            color="black",
            alpha=0.6,
            size=5,
            ax=ax,
        )

        p_corrected = pvals_corrected[plot_idx]
        p_uncorrected = all_pvals[plot_idx]
        if p_corrected < 0.001:
            annotation_text = "***"
        elif p_corrected < 0.01:
            annotation_text = "**"
        elif p_corrected < 0.05:
            annotation_text = "*"
        elif p_uncorrected < 0.05:
            annotation_text = "* (ns)"
        else:
            annotation_text = "ns"

        y_max = comparison_agg["log_irt_zscore"].max()
        y_range = comparison_agg["log_irt_zscore"].max() - comparison_agg["log_irt_zscore"].min()
        y_pos = y_max + 0.1 * y_range
        ax.plot([0, 1], [y_pos, y_pos], color="black", linewidth=1.5)
        ax.text(
            0.5, y_pos + 0.02 * y_range, annotation_text, ha="center", va="bottom", fontsize=plt.rcParams["font.size"]
        )

        ax.set_xticks(range(len(condition_order)))
        ax.set_xticklabels(["Human-Human", f"Human-AI\n({prompt.title()})"])
        ax.set_xlabel("")
        ax.set_ylabel("Human log response times\n(z-scored)" if plot_idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    fig.text(0.5, 0.04, "Dyadic Condition", ha="center", fontsize=plt.rcParams["font.size"])

    stats_df = pd.DataFrame(
        {
            "prompt": all_prompts,
            "t": all_test_stats,
            "p_uncorrected": all_pvals,
            "p_holm": pvals_corrected,
            "significant": reject,
        }
    )
    console.print("\n=== PAIRED RESPONSE-TIME COMPARISONS (Holm-corrected) ===")
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
        help="Directory to save plots and stats. Defaults to project_root/outputs/figures/supp/irt-phase.",
    ),
    formats: str | None = typer.Option(
        None, "--formats", help="Comma-separated output formats: png, pdf, svg. Defaults to png."
    ),
    panels: str | None = typer.Option(
        None,
        "--panels",
        help="Comma-separated panels to generate: s10, s12c, s14. Defaults to all.",
    ),
    show: bool = typer.Option(False, "--show/--no-show", help="Show the plots instead of just saving."),
):
    """Generate the dyadic task-phase/switch panels (Figures S10, S12C, S14)."""
    selected_panels = _parse_panels(panels)
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
        output_dir = project_root / "outputs" / "figures" / "supp" / "irt-phase"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_context("paper", font_scale=2.0)
    apply_bold_axis_style()

    df = load_data(data_dir=data_dir, cross_condition_only=False, include_solo=True)
    df = prepare_df(df)

    if "s14" in selected_panels:
        fig_s14, desc_s14, comp_s14 = create_switch_panel(df)
        paths = save_figure_formats(
            fig_s14,
            output_dir / "figS14_switch_vs_no_switch_4panel",
            formats=formats,
            default=("png",),
            dpi=300,
            bbox_inches="tight",
        )
        console.print(f"[green]Saved Figure S14 to {', '.join(str(p) for p in paths)}[/green]")
        desc_s14.to_csv(output_dir / "figS14_switch_vs_no_switch_4panel_descriptive.csv", index=False)
        comp_s14.to_csv(output_dir / "figS14_switch_vs_no_switch_4panel_comparisons.csv", index=False)

    if "s10" in selected_panels:
        fig_s10, desc_s10, comp_s10 = create_task_portion_panel(df)
        paths = save_figure_formats(
            fig_s10,
            output_dir / "figS10_task_portion_collapsed",
            formats=formats,
            default=("png",),
            dpi=300,
            bbox_inches="tight",
        )
        console.print(f"[green]Saved Figure S10 to {', '.join(str(p) for p in paths)}[/green]")
        desc_s10.to_csv(output_dir / "figS10_task_portion_collapsed_descriptive.csv", index=False)
        comp_s10.to_csv(output_dir / "figS10_task_portion_collapsed_comparisons.csv", index=False)

    if "s12c" in selected_panels:
        fig_s12c, stats_s12c = create_paired_by_prompt_panel(df)
        if fig_s12c is not None:
            paths = save_figure_formats(
                fig_s12c,
                output_dir / "figS12c_paired_by_prompt",
                formats=formats,
                default=("png",),
                dpi=300,
                bbox_inches="tight",
            )
            console.print(f"[green]Saved Figure S12C to {', '.join(str(p) for p in paths)}[/green]")
            stats_s12c.to_csv(output_dir / "figS12c_paired_by_prompt_stats.csv", index=False)

    if show:
        plt.show()

    console.print("\n[bold]Finished![/bold]")


if __name__ == "__main__":
    seed_everything()
    app()
