"""Figure 5: 2x2 composite combining geometry, production, and timing panels.

Assembles four panels into a single composite figure:

  Panel A: Human response time change [Late - Early] (solo/HH/HA, collapsed AI).
  Panel B: Human self-similarity change [Late - Early] (solo/HH/HA, collapsed AI).
  Panel C: Partner similarity to participant step midpoint
           (HH vs HA, per-player bar + strip).
  Panel D: Number of concepts produced, per-player mean count
           (HH vs HA, bar + strip).

Usage::

    python src/sftbench/dyadic/plot_fig5.py [OPTIONS]
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
from scipy.stats import levene, mannwhitneyu, wilcoxon
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

HH = "hh"
HA = "ha"
DYAD_CONDITIONS = [HH, HA]
CONDITION_LABELS = {HH: "Human-Human", HA: "Human-AI"}
CONDITION_COLORS = {HH: COLOR_MAP_COLLAPSED["collab_human"], HA: COLOR_MAP_COLLAPSED["collab_ai"]}
HUE_ORDER = [CONDITION_LABELS[HH], CONDITION_LABELS[HA]]
PALETTE = {CONDITION_LABELS[c]: CONDITION_COLORS[c] for c in DYAD_CONDITIONS}

BAR_WIDTH_CD = 0.8
BAR_WIDTH_AB = BAR_WIDTH_CD * (2 / 3)

# Large-format styling for all four panels
LARGE_PANEL_LABEL_FONTSIZE = 68
LARGE_PANEL_AXIS_FONTSIZE = 48
LARGE_PANEL_TICK_FONTSIZE = 44
LARGE_PANEL_SPINE_WIDTH = 0.8
LARGE_PANEL_ANNOTATION_FONTSIZE = 40

STRIP_MARKER_SIZE = 4 * 6 * 0.7

STRIP_MARKER_SIZE_CD = STRIP_MARKER_SIZE * 0.8

STRIP_ALPHA = 0.6 * 0.8

# Vertical nudge (in axes-fraction units), tied to the height_ratio=1 row

AB_LABEL_Y_OFFSET = 0.05

# Vertical nudge tied to the height_ratio=1.5 row
CD_LABEL_Y_OFFSET = -0.05

# Horizontal nudge applied only to the left-column panels' label
AC_LABEL_X_OFFSET = 0.05


# ---------------------------------------------------------------------------
# Panels A/B stats (solo/HH/HA, Holm-corrected) -- moved here from the former
# plot_fig5_panels.py so the composite figure is self-contained.
# ---------------------------------------------------------------------------

RT_SIM_ORDER = ["solo", "collab_human", "collab_ai"]
RT_SIM_PAIRS = [
    ("collab_human", "solo"),
    ("collab_human", "collab_ai"),
    ("collab_ai", "solo"),
]
RT_SIM_XTICKLABELS = ["Solo", "Human-\nHuman", "Human-\nAI"]


def map_dyadtype_to_condition(dyad_type):
    """Map dyadType values to standardized condition names."""
    mapping = {"solo": "solo", "hh": "collab_human", "ha": "collab_ai"}
    return mapping.get(dyad_type, dyad_type)


def _wilcoxon_against_zero(values: np.ndarray, alternative: str) -> tuple[float, float]:
    """Return a one-sample Wilcoxon signed-rank statistic and p-value."""
    try:
        result = wilcoxon(
            values,
            alternative=alternative,
            zero_method="wilcox",
            method="auto",
        )
    except ValueError:
        # SciPy raises when every difference is zero under zero_method="wilcox".
        return 0.0, 1.0
    return float(result.statistic), float(result.pvalue)


def summary_statistics(diff_data, value_col, metric_name, order):
    """Print and collect one-sample Wilcoxon tests of changes against zero."""
    console.print("\n" + "=" * 80)
    console.print(f"{metric_name.upper()} DIFFERENCE (LATE - EARLY) SUMMARY")
    console.print("=" * 80)

    rows: list[dict[str, object]] = []
    for condition in order:
        vals = diff_data.loc[diff_data["condition"] == condition, value_col].dropna()
        if len(vals) == 0:
            continue

        values = vals.to_numpy(dtype=float)
        stat_two, p_two = _wilcoxon_against_zero(values, "two-sided")
        stat_gt, p_gt = _wilcoxon_against_zero(values, "greater")
        stat_lt, p_lt = _wilcoxon_against_zero(values, "less")

        label = LABEL_MAP.get(condition, condition)

        console.print(f"\n{label}:")
        console.print(f"  N = {len(vals)}")
        console.print(f"  Mean = {vals.mean():.4f} ± {vals.sem():.4f} (SEM)")
        console.print(f"  Median = {vals.median():.4f}")
        console.print(f"  One-sample Wilcoxon vs 0 (two-sided): W={stat_two:.2f}, p={p_two:.4g}")
        console.print(f"  One-sample Wilcoxon vs 0 (greater):   W={stat_gt:.2f}, p={p_gt:.4g}")
        console.print(f"  One-sample Wilcoxon vs 0 (less):      W={stat_lt:.2f}, p={p_lt:.4g}")

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

    return rows


def pairwise_holm(diff_data, value_col, metric_name):
    """Print and collect Holm-Bonferroni corrected pairwise comparisons."""
    console.print("\n--- Pairwise comparisons (Holm-Bonferroni corrected) ---")
    uncorrected_pvals = []
    stats = []
    complements = []
    for pair in RT_SIM_PAIRS:
        group1 = diff_data[diff_data["condition"] == pair[0]][value_col].dropna()
        group2 = diff_data[diff_data["condition"] == pair[1]][value_col].dropna()
        stat, pval = mannwhitneyu(group1, group2, alternative="two-sided")
        stats.append(stat)
        # Complementary U for the reversed group order (U1 + U2 = n1 * n2).
        complements.append(len(group1) * len(group2) - stat)
        uncorrected_pvals.append(pval)

    reject, corrected_pvals, _, _ = multipletests(uncorrected_pvals, alpha=0.05, method="holm")

    rows: list[dict[str, object]] = []
    for i, pair in enumerate(RT_SIM_PAIRS):
        console.print(
            f"  {LABEL_MAP[pair[0]]} vs {LABEL_MAP[pair[1]]}: "
            f"U = {stats[i]:.2f} (complement {complements[i]:.2f}), "
            f"p = {uncorrected_pvals[i]:.4g}, "
            f"corrected p = {corrected_pvals[i]:.4g}, "
            f"significant = {reject[i]}"
        )
        rows.append(
            {
                "metric": metric_name,
                "group1": pair[0],
                "group2": pair[1],
                "label1": LABEL_MAP[pair[0]],
                "label2": LABEL_MAP[pair[1]],
                "U_two_sided": float(stats[i]),
                "U_two_sided_complement": float(complements[i]),
                "p_uncorrected": float(uncorrected_pvals[i]),
                "p_holm": float(corrected_pvals[i]),
                "significant": bool(reject[i]),
            }
        )

    return rows


def _letter_n_offset_fraction(ax: plt.Axes) -> float:
    """Measure the rendered width of the glyph 'n' at the panel-label fontsize
    and express it as a fraction of the axes height, so it can be used as a
    vertical label offset ("move the label up by the width of the letter n")."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = ax.text(0, 0, "n", fontsize=LARGE_PANEL_LABEL_FONTSIZE, fontweight="bold")
    width_px = probe.get_window_extent(renderer=renderer).width
    probe.remove()
    ax_height_px = ax.get_window_extent(renderer=renderer).height
    return width_px / ax_height_px


def _apply_large_panel_style(
    ax: plt.Axes, letter: str, label_y_offset: float = 0.0, label_x_offset: float = 0.0
) -> None:
    """Apply the bold/oversized styling shared by all four panels, and stamp a
    bare panel-letter label (no descriptive title text) above the axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LARGE_PANEL_SPINE_WIDTH)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=LARGE_PANEL_TICK_FONTSIZE,
        width=LARGE_PANEL_SPINE_WIDTH,
        length=7,
    )
    ax.xaxis.label.set_fontsize(LARGE_PANEL_AXIS_FONTSIZE)
    ax.yaxis.label.set_fontsize(LARGE_PANEL_AXIS_FONTSIZE)
    ax.set_title("")
    n_offset = _letter_n_offset_fraction(ax)
    ax.text(
        -0.18 + label_x_offset,
        1.08 + label_y_offset + n_offset,
        letter,
        transform=ax.transAxes,
        fontsize=LARGE_PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def summary_statistics_hh_ha(player_df: pd.DataFrame, value_col: str, metric_name: str):
    """Print and collect per-condition descriptive stats for a player-level
    HH-vs-HA metric (Panels C/D)."""
    console.print("\n" + "=" * 80)
    console.print(f"{metric_name.upper()} SUMMARY (HH vs HA)")
    console.print("=" * 80)

    rows: list[dict[str, object]] = []
    for cond in DYAD_CONDITIONS:
        label = CONDITION_LABELS[cond]
        vals = player_df.loc[player_df["condition"] == cond, value_col].dropna()
        if len(vals) == 0:
            continue

        console.print(f"\n{label}:")
        console.print(f"  N = {len(vals)}")
        console.print(f"  Mean = {vals.mean():.4f} ± {vals.sem():.4f} (SEM)")
        console.print(f"  Median = {vals.median():.4f}")

        rows.append(
            {
                "metric": metric_name,
                "condition": cond,
                "label": label,
                "N": int(len(vals)),
                "mean": float(vals.mean()),
                "sem": float(vals.sem()),
                "median": float(vals.median()),
            }
        )

    return rows


def pairwise_hh_ha(player_df: pd.DataFrame, value_col: str, metric_name: str):
    """Print and collect the single HH-vs-HA Mann-Whitney comparison for Panels
    A/B (no multiple-comparisons correction needed: only one pair is tested),
    matching the formatting of `pairwise_holm`."""
    console.print("\n--- Pairwise comparison (HH vs HA) ---")
    group1 = player_df[player_df["condition"] == HH][value_col].dropna()
    group2 = player_df[player_df["condition"] == HA][value_col].dropna()
    stat, pval = mannwhitneyu(group1, group2, alternative="two-sided")

    console.print(
        f"  {CONDITION_LABELS[HH]} vs {CONDITION_LABELS[HA]}: U = {stat:.2f}, p = {pval:.4g}, "
        f"significant = {pval < 0.05}"
    )

    return [
        {
            "metric": metric_name,
            "test": "mannwhitney",
            "group1": HH,
            "group2": HA,
            "label1": CONDITION_LABELS[HH],
            "label2": CONDITION_LABELS[HA],
            "U": float(stat),
            "p_uncorrected": float(pval),
            "p_holm": float(pval),
            "significant": bool(pval < 0.05),
        }
    ]


def variance_test_hh_ha(player_df: pd.DataFrame, value_col: str, metric_name: str):
    """Print and collect a Brown-Forsythe (Levene, center='median') test for
    unequal variance between HH and HA, used for Panel A to check whether
    Human-AI partner_midpoint_sim is more variable than Human-Human."""
    console.print("\n--- Variance test (Brown-Forsythe / Levene, center='median'): HH vs HA ---")
    group1 = player_df[player_df["condition"] == HH][value_col].dropna()
    group2 = player_df[player_df["condition"] == HA][value_col].dropna()
    stat, pval = levene(group1, group2, center="median")
    var1, var2 = float(group1.var(ddof=1)), float(group2.var(ddof=1))
    std1, std2 = float(group1.std(ddof=1)), float(group2.std(ddof=1))

    console.print(f"  {CONDITION_LABELS[HH]}: variance = {var1:.4g}, SD = {std1:.4g}, N = {len(group1)}")
    console.print(f"  {CONDITION_LABELS[HA]}: variance = {var2:.4g}, SD = {std2:.4g}, N = {len(group2)}")
    console.print(
        f"  Levene's test (Brown-Forsythe): W = {stat:.4g}, p = {pval:.4g}, "
        f"significant = {pval < 0.05} "
        f"({'HA' if var2 > var1 else 'HH'} has higher variance)"
    )

    return [
        {
            "metric": metric_name,
            "test": "levene_brown_forsythe",
            "group1": HH,
            "group2": HA,
            "label1": CONDITION_LABELS[HH],
            "label2": CONDITION_LABELS[HA],
            "variance_group1": var1,
            "variance_group2": var2,
            "statistic": float(stat),
            "p_uncorrected": float(pval),
            "p_holm": float(pval),
            "significant": bool(pval < 0.05),
        }
    ]


# ---------------------------------------------------------------------------
# Shared per-player bar + strip renderer (Panels C & D)
# ---------------------------------------------------------------------------


def _render_bar_strip(
    ax: plt.Axes,
    player_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    letter: str,
    add_hline: bool = False,
    label_x_offset: float = 0.0,
) -> None:
    """Bar plot (player mean +/- SE) with jittered per-player strip overlay and
    a Mann-Whitney HH-vs-HA annotation, styled to match the reference screenshot."""
    sns.barplot(
        data=player_df,
        x="condition_label",
        y=value_col,
        order=HUE_ORDER,
        hue="condition_label",
        palette=PALETTE,
        legend=False,
        errorbar="se",
        estimator=np.mean,
        ax=ax,
        edgecolor="black",
        linewidth=1.0,
        width=BAR_WIDTH_AB,
    )
    sns.stripplot(
        data=player_df,
        x="condition_label",
        y=value_col,
        order=HUE_ORDER,
        ax=ax,
        color="0.25",
        size=STRIP_MARKER_SIZE,
        alpha=STRIP_ALPHA,
        jitter=0.2,
    )

    if add_hline:
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")

    pairs = [(HUE_ORDER[0], HUE_ORDER[1])]
    try:
        annotator = Annotator(
            ax,
            pairs,
            data=player_df,
            x="condition_label",
            y=value_col,
            order=HUE_ORDER,
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            show_test_name=False,
            comparisons_correction=None,
            fontsize=LARGE_PANEL_ANNOTATION_FONTSIZE,
        )
        annotator.apply_test()
        annotator.annotate()
    except Exception as exc:
        console.print(f"[yellow]Panel {letter} annotation failed: {exc}[/yellow]")

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _apply_large_panel_style(ax, letter, label_y_offset=AB_LABEL_Y_OFFSET, label_x_offset=label_x_offset)


# ---------------------------------------------------------------------------
# Panel C: Partner similarity to participant step midpoint
# ---------------------------------------------------------------------------


def create_panel_c_partner_midpoint_sim(df: pd.DataFrame, ax: plt.Axes):
    """Panel C: per-player mean partner_midpoint_sim, HH vs HA."""
    console.print("Creating Panel C: Partner similarity to participant step midpoint...")

    records = []
    for cond in DYAD_CONDITIONS:
        sub = df[(df["dyadType"] == cond) & (df["sourceType"] == "human")].dropna(subset=["partner_midpoint_sim"])
        for player, grp in sub.groupby("source"):
            val = grp["partner_midpoint_sim"].mean()
            if np.isfinite(val):
                records.append(
                    {
                        "source": player,
                        "condition": cond,
                        "condition_label": CONDITION_LABELS[cond],
                        "partner_midpoint_sim": val,
                    }
                )

    player_df = pd.DataFrame(records)
    _render_bar_strip(
        ax,
        player_df,
        "partner_midpoint_sim",
        ylabel="Partner similarity to human\nsemantic transition midpoint\n(cosine similarity)",
        letter="C",
        add_hline=False,
        label_x_offset=AC_LABEL_X_OFFSET,
    )

    metric_name = "Partner Midpoint Similarity"
    summary_rows = summary_statistics_hh_ha(player_df, "partner_midpoint_sim", metric_name)
    pairwise_rows = pairwise_hh_ha(player_df, "partner_midpoint_sim", metric_name)
    pairwise_rows += variance_test_hh_ha(player_df, "partner_midpoint_sim", metric_name)
    return summary_rows, pairwise_rows


# ---------------------------------------------------------------------------
# Panel D: Number of concepts produced (Individual partner)
# ---------------------------------------------------------------------------


def create_panel_d_words_produced(df: pd.DataFrame, ax: plt.Axes):
    """Panel D: per-player mean count of words/concepts produced, HH vs HA.

    For each human player and category, counts the number of words that player
    individually contributed (i.e. their own row count within that game/category),
    then averages across categories to get one value per player per condition.
    """
    console.print("Creating Panel D: Number of concepts produced (Individual partner)...")

    records = []
    for cond in DYAD_CONDITIONS:
        sub = df[(df["dyadType"] == cond) & (df["sourceType"] == "human")]
        counts = sub.groupby(["source", "category"]).size().reset_index(name="n_words")
        for player, grp in counts.groupby("source"):
            val = grp["n_words"].mean()
            if np.isfinite(val):
                records.append(
                    {
                        "source": player,
                        "condition": cond,
                        "condition_label": CONDITION_LABELS[cond],
                        "n_words": val,
                    }
                )

    player_df = pd.DataFrame(records)
    _render_bar_strip(
        ax,
        player_df,
        "n_words",
        ylabel="Number of concepts produced\n(Individual human partner)",
        letter="D",
        add_hline=False,
    )

    metric_name = "Concepts Produced"
    summary_rows = summary_statistics_hh_ha(player_df, "n_words", metric_name)
    pairwise_rows = pairwise_hh_ha(player_df, "n_words", metric_name)
    return summary_rows, pairwise_rows


# ---------------------------------------------------------------------------
# Panel A: Human response time change [Late - Early]
# ---------------------------------------------------------------------------


def create_panel_a_rt_diff(df: pd.DataFrame, ax: plt.Axes, irt_column: str = "log_irt_zscore"):
    """Panel A: log reaction-time differences (late - early); solo/HH/HA, collapsed AI."""
    console.print("Creating Panel A: Response time change [Late - Early]...")

    combined_df = df[df["sourceType"] == "human"].copy()
    combined_df = combined_df[combined_df[irt_column].notna()].copy()
    combined_df["condition"] = combined_df["dyadType"].apply(map_dyadtype_to_condition)

    combined_mean_df = (
        combined_df.groupby(["source", "condition", "word_index_split", "category"])[[irt_column]]
        .agg({irt_column: "mean"})
        .reset_index()
    )

    log_irt_diff = combined_mean_df.pivot_table(
        index=["source", "condition"], columns="word_index_split", values=irt_column, aggfunc="mean"
    ).reset_index()
    log_irt_diff.columns.name = None
    log_irt_diff = log_irt_diff.rename(columns={False: f"{irt_column}_early", True: f"{irt_column}_late"})
    log_irt_diff[f"{irt_column}_diff"] = log_irt_diff[f"{irt_column}_late"] - log_irt_diff[f"{irt_column}_early"]

    log_irt_diff["condition"] = pd.Categorical(log_irt_diff["condition"], categories=RT_SIM_ORDER, ordered=True)

    value_col = f"{irt_column}_diff"
    _box_colors = [COLOR_MAP_COLLAPSED[c] for c in RT_SIM_ORDER if c in COLOR_MAP_COLLAPSED]
    sns.boxplot(
        data=log_irt_diff,
        x="condition",
        y=value_col,
        order=RT_SIM_ORDER,
        hue="condition",
        palette=_box_colors,
        ax=ax,
        legend=False,
        width=BAR_WIDTH_CD,
        linecolor="black",
        linewidth=1.5,
        fliersize=0,
    )
    sns.stripplot(
        data=log_irt_diff,
        x="condition",
        y=value_col,
        order=RT_SIM_ORDER,
        ax=ax,
        color="0.25",
        size=STRIP_MARKER_SIZE_CD,
        alpha=STRIP_ALPHA,
        jitter=0.2,
    )

    ax.axhline(y=0, color="k", linewidth=1.0, zorder=0)
    ax.set_xlabel("Dyad Type")
    if irt_column == "log_irt_zscore":
        ax.set_ylabel("Human response time change\n[Late - Early] (log, z-scored)")
    else:
        ax.set_ylabel("Human response time change\n[Late - Early] (log, seconds)")
    ax.set_xticks(range(len(RT_SIM_XTICKLABELS)))
    ax.set_xticklabels(RT_SIM_XTICKLABELS)

    try:
        annotator = Annotator(
            ax, pairs=RT_SIM_PAIRS, data=log_irt_diff, x="condition", y=value_col, order=RT_SIM_ORDER, hue=None
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            show_test_name=False,
            comparisons_correction="holm",
            fontsize=LARGE_PANEL_ANNOTATION_FONTSIZE,
        )
        annotator.apply_test()
        annotator.annotate(line_offset_to_group=0.05)
    except Exception as e:
        console.print(f"[yellow]Warning: Statistical annotation failed: {e}[/yellow]")

    ax.yaxis.grid(True, linestyle="-", linewidth=LARGE_PANEL_SPINE_WIDTH * 0.6, color="gray", alpha=0.5)
    ax.set_axisbelow(True)
    # NB: label_y_offset stays tied to this panel's box+strip renderer (and its
    # height_ratio=1.5 row), not to the literal letter "A" -- see CD_LABEL_Y_OFFSET.
    _apply_large_panel_style(ax, "A", label_y_offset=CD_LABEL_Y_OFFSET, label_x_offset=AC_LABEL_X_OFFSET)

    summary_rows = summary_statistics(log_irt_diff, value_col, "IRT", RT_SIM_ORDER)
    pairwise_rows = pairwise_holm(log_irt_diff, value_col, "IRT")
    return summary_rows, pairwise_rows


# ---------------------------------------------------------------------------
# Panel B: Human self-similarity change [Late - Early]
# ---------------------------------------------------------------------------


def create_panel_b_self_sim_diff(df: pd.DataFrame, ax: plt.Axes):
    """Panel B: self-similarity differences (late - early); solo/HH/HA, collapsed AI."""
    console.print("Creating Panel B: Self-similarity change [Late - Early]...")

    analysis_df = df[df["sourceType"] == "human"].copy()
    analysis_df = analysis_df[analysis_df["self_similarity"].notna()].copy()
    analysis_df["condition"] = analysis_df["dyadType"].apply(map_dyadtype_to_condition)
    analysis_df["portion"] = analysis_df["word_index_split"].apply(lambda x: "late" if x else "early")

    agg_df = analysis_df.groupby(["source", "condition", "portion", "category"])["self_similarity"].mean().reset_index()

    similarity_diff = agg_df.pivot_table(
        index=["source", "condition"], columns="portion", values="self_similarity", aggfunc="mean"
    ).reset_index()
    similarity_diff.columns.name = None
    similarity_diff["similarity_diff"] = similarity_diff["late"] - similarity_diff["early"]

    similarity_diff["condition"] = pd.Categorical(similarity_diff["condition"], categories=RT_SIM_ORDER, ordered=True)

    value_col = "similarity_diff"
    _box_colors = [COLOR_MAP_COLLAPSED[c] for c in RT_SIM_ORDER if c in COLOR_MAP_COLLAPSED]
    sns.boxplot(
        data=similarity_diff,
        x="condition",
        y=value_col,
        order=RT_SIM_ORDER,
        hue="condition",
        palette=_box_colors,
        ax=ax,
        legend=False,
        width=BAR_WIDTH_CD,
        linecolor="black",
        linewidth=1.5,
        fliersize=0,
    )
    sns.stripplot(
        data=similarity_diff,
        x="condition",
        y=value_col,
        order=RT_SIM_ORDER,
        ax=ax,
        color="0.25",
        size=STRIP_MARKER_SIZE_CD,
        alpha=STRIP_ALPHA,
        jitter=0.2,
    )

    ax.axhline(y=0, color="k", linewidth=1.0, zorder=0)
    ax.set_xlabel("Dyad Type")
    ax.set_ylabel("Human concept self-similarity change\n[Late - Early] (cosine similarity)")
    ax.set_xticks(range(len(RT_SIM_XTICKLABELS)))
    ax.set_xticklabels(RT_SIM_XTICKLABELS)

    try:
        annotator = Annotator(
            ax, pairs=RT_SIM_PAIRS, data=similarity_diff, x="condition", y=value_col, order=RT_SIM_ORDER, hue=None
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            show_test_name=False,
            comparisons_correction="holm",
            fontsize=LARGE_PANEL_ANNOTATION_FONTSIZE,
        )
        annotator.apply_test()
        annotator.annotate(line_offset_to_group=0.06)
    except Exception as e:
        console.print(f"[yellow]Warning: Statistical annotation failed: {e}[/yellow]")

    ax.yaxis.grid(True, linestyle="-", linewidth=LARGE_PANEL_SPINE_WIDTH * 0.6, color="gray", alpha=0.5)
    ax.set_axisbelow(True)
    _apply_large_panel_style(ax, "B", label_y_offset=CD_LABEL_Y_OFFSET)

    summary_rows = summary_statistics(similarity_diff, value_col, "Self-Similarity", RT_SIM_ORDER)
    pairwise_rows = pairwise_holm(similarity_diff, value_col, "Self-Similarity")
    return summary_rows, pairwise_rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def create_figure(df: pd.DataFrame, irt_column: str = "log_irt_zscore"):
    # Top row (Panels A/B) uses the large/bold box+strip reference styling, so
    # it gets extra height relative to the bottom row (Panels C/D)'s bar+strip
    # panels. constrained_layout (rather than tight_layout) is used so the
    # now-larger label/title text doesn't get clipped at the figure/column edges.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(30, 30),
        gridspec_kw={"height_ratios": [1.5, 1]},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.35, h_pad=0.35, wspace=0.08, hspace=0.06)

    summary_a, pairwise_a = create_panel_a_rt_diff(df, axes[0, 0], irt_column=irt_column)
    summary_b, pairwise_b = create_panel_b_self_sim_diff(df, axes[0, 1])
    summary_c, pairwise_c = create_panel_c_partner_midpoint_sim(df, axes[1, 0])
    summary_d, pairwise_d = create_panel_d_words_produced(df, axes[1, 1])

    summary_rows = summary_a + summary_b + summary_c + summary_d
    pairwise_rows = pairwise_a + pairwise_b + pairwise_c + pairwise_d
    return fig, summary_rows, pairwise_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def plot(
    data_root: pathlib.Path = typer.Option(
        Path("."), "--data-root", help="Root directory for relative input data paths."
    ),
    output_dir: pathlib.Path = typer.Option(
        None,
        "--output-dir",
        help="Directory to save plots and stats. Defaults to project_root/outputs/figures/figure-5.",
    ),
    irt_column: str = typer.Option(
        "log_irt_zscore", "--irt-column", help="IRT column for Panel A ('log_irt_zscore' or 'log_irt')."
    ),
    formats: str | None = typer.Option(
        None, "--formats", help="Comma-separated output formats: png, pdf, svg. Defaults to png."
    ),
    show: bool = typer.Option(False, "--show/--no-show", help="Show the plot instead of just saving."),
):
    """Generate the 2x2 figure-5 composite from the frozen dyadic data."""
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
        output_dir = project_root / "outputs" / "figures" / "figure-5"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.8)
    apply_bold_axis_style(linewidth=0.8)

    df = load_data(
        data_dir=data_dir,
        cross_condition_only=False,
        include_solo=True,
    )

    fig, summary_rows, pairwise_rows = create_figure(df, irt_column=irt_column)

    paths = save_figure_formats(
        fig,
        output_dir / "fig5_2x2",
        formats=formats,
        default=("png",),
        dpi=300,
    )
    console.print(f"[green]Saved figure-5 to {', '.join(str(p) for p in paths)}[/green]")

    summary_csv = output_dir / "fig5_summary_stats.csv"
    pairwise_csv = output_dir / "fig5_pairwise_holm.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_csv, index=False)
    console.print(f"[green]Saved summary statistics to {summary_csv}[/green]")
    console.print(f"[green]Saved pairwise comparisons to {pairwise_csv}[/green]")

    if show:
        plt.show()

    console.print("\n[bold]Finished![/bold]")


if __name__ == "__main__":
    seed_everything()
    app()
