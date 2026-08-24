import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from pandas.errors import DtypeWarning
from scipy import stats
from statannotations.Annotator import Annotator

from sftbench import find_project_root
from sftbench.figure_output import apply_bold_axis_style, save_figure_formats
from sftbench.reproducibility import SEED, seed_everything

try:
    import tomllib
except ImportError:
    import tomli as tomllib

_root = find_project_root()
if _root is None:
    _root = Path(__file__).parent.parent.parent.parent.parent

# Default paths
DEFAULT_DATA_PATH = Path("data/logitlens/cognitive_alignment_results_70b_100.csv.xz")

REQUIRED_COLUMNS = {
    "id",
    "sequence_type",
    "item_rank",
    "permutation",
    "probability",
    "normalized_probability",
    "nll",
    "normalized_nll",
}

app = typer.Typer(help="CLI to plot probability sequence evaluations.")


def load_plotting_config(config_path: Path):
    """Loads seaborn context and style from a TOML file."""
    if not config_path.exists():
        typer.echo(f"Config file not found: {config_path}. Using defaults.")
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


def calculate_perplexity_from_nll(neg_log_likelihoods):
    """
    Calculates perplexity directly from a list of negative log-likelihoods.
    """
    n = len(neg_log_likelihoods)
    if n == 0:
        return np.nan
    total_nll = np.sum(neg_log_likelihoods)
    average_nll = total_nll / n
    perplexity = np.exp(average_nll)
    return perplexity


def validate_input_data(df: pd.DataFrame, input_path: Path) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        joined = ", ".join(missing_columns)
        typer.echo(f"Input file {input_path} is missing required column(s): {joined}")
        raise typer.Exit(code=1)

    required_sequence_types = {
        "original",
        "subject-wise-permutation",
        "random-subject-first-half_second-half",
    }
    observed_sequence_types = set(df["sequence_type"].dropna().unique())
    missing_sequence_types = sorted(required_sequence_types - observed_sequence_types)
    if missing_sequence_types:
        joined = ", ".join(missing_sequence_types)
        typer.echo(f"Input file {input_path} is missing required sequence_type value(s): {joined}")
        raise typer.Exit(code=1)


@app.command()
def plot(
    input_path: Path = typer.Option(
        DEFAULT_DATA_PATH,
        help="Path to the input CSV file.",
        file_okay=True,
        readable=True,
    ),
    config: Path = typer.Option(
        _root / "configs" / "plotting.toml",
        help="Path to the plotting configuration TOML.",
    ),
    output_dir: Path = typer.Option(Path("plots"), help="Directory to save generated plots."),
    show: bool = typer.Option(False, help="Whether to show plots interactively."),
    split_point: int = typer.Option(18, help="Split point for sequence analysis."),
    formats: str | None = typer.Option(
        None,
        "--formats",
        help="Comma-separated output formats for figures: png, pdf, svg. Defaults to png.",
    ),
):
    """Probability sequence evaluation plots."""
    if not input_path.exists():
        typer.echo(f"Input file not found: {input_path}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    load_plotting_config(config)

    typer.echo(f"Loading data from {input_path}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DtypeWarning)
        df = pd.read_csv(input_path)
    validate_input_data(df, input_path)
    typer.echo(f"Loaded DataFrame with shape: {df.shape}")
    typer.echo(f"Columns: {df.columns.tolist()}")

    columns_to_print = ["model", "source", "sequence_type"]
    typer.echo("=" * 60)
    typer.echo("DATASET SUMMARY")
    typer.echo("=" * 60)
    if "id" in df.columns:
        typer.echo(f"Unique IDs: {df['id'].nunique():,}")
    typer.echo("")
    for i, col in enumerate(columns_to_print, 1):
        if col in df.columns:
            unique_values = df[col].unique()
            typer.echo(f"{i}. {col.upper().replace('_', ' ')}")
            typer.echo("   " + "-" * (len(col) + 10))
            for value in sorted(unique_values):
                typer.echo(f"   - {value}")
            typer.echo(f"   Total: {len(unique_values)} unique values")
            if i < len(columns_to_print):
                typer.echo("")
    typer.echo("=" * 60)

    sequence_type_legend_map = {
        "original": "Original",
        "subject-wise-permutation": "Internal Shuffle",
        "random-subject-first-half_second-half": "Mismatched Prefix",
    }

    palette = {
        "Original": "#1f77b4",  # Blue
        "Internal Shuffle": "#d62728",
        "Mismatched Prefix": "#2ca02c",  # Green
    }

    df_filt = df[df["item_rank"] <= 36].copy()
    if "sequence_type" in df_filt.columns:
        df_filt["sequence_type"] = df_filt["sequence_type"].map(sequence_type_legend_map)

    def set_rank_ticks(ax, df_local):
        min_rank = df_local["item_rank"].min()
        max_rank = df_local["item_rank"].max()
        new_tick_positions = [r for r in range(min_rank, max_rank + 1) if r % 2 != 0]

        unique_ranks = sorted(df_local["item_rank"].unique())
        rank_to_idx = {r: i for i, r in enumerate(unique_ranks)}
        tick_indices = [rank_to_idx[r] for r in new_tick_positions if r in rank_to_idx]

        ax.set_xticks(tick_indices)
        ax.set_xticklabels(new_tick_positions)

    # --- Plot 1: Probability Pointplot (Linear) ---
    if "probability" in df_filt.columns:
        plt.figure(figsize=(8, 5))
        ax = sns.pointplot(
            data=df_filt,
            x="item_rank",
            y="probability",
            palette=palette,
            errorbar=("ci", 95),
            seed=SEED,
            capsize=0.2,
            hue="sequence_type",
            hue_order=["Original", "Mismatched Prefix", "Internal Shuffle"],
        )
        ax.set_title("Probability of Concept in Sequence")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Probability")
        ax.axvline(x=split_point - 0.5, color="grey", linestyle="--", linewidth=2, label="Sequence Split")
        _handles, _labels = ax.get_legend_handles_labels()
        _desired = ["Original", "Mismatched Prefix", "Internal Shuffle", "Sequence Split"]
        _hl = dict(zip(_labels, _handles, strict=True))
        ax.legend(
            [_hl[lo] for lo in _desired if lo in _hl],
            [lo for lo in _desired if lo in _hl],
            loc="upper center",
            fontsize="x-small",
            markerscale=0.5,
            handlelength=1.2,
            borderpad=0.3,
            labelspacing=0.2,
        )
        set_rank_ticks(ax, df_filt)

        plt.tight_layout()
        save_figure_formats(
            None,
            output_dir / "probability_next_word_token_pointplot.png",
            formats=formats,
            default=("png",),
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close()

    # --- Plot 2: Probability Pointplot (Log) ---
    if "probability" in df_filt.columns:
        plt.figure(figsize=(8, 5))
        ax = sns.pointplot(
            data=df_filt,
            x="item_rank",
            y="probability",
            palette=palette,
            errorbar=("ci", 95),
            seed=SEED,
            capsize=0.2,
            hue="sequence_type",
            hue_order=["Original", "Mismatched Prefix", "Internal Shuffle"],
        )
        ax.set_title("Probability of Concept in Sequence")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Probability (log scale)")
        ax.set_yscale("log")
        ax.axvline(x=split_point - 0.5, color="grey", linestyle="--", linewidth=2, label="Sequence Split")
        _handles, _labels = ax.get_legend_handles_labels()
        _desired = ["Original", "Mismatched Prefix", "Internal Shuffle", "Sequence Split"]
        _hl = dict(zip(_labels, _handles, strict=True))
        ax.legend(
            [_hl[lo] for lo in _desired if lo in _hl],
            [lo for lo in _desired if lo in _hl],
            loc="upper right",
            fontsize="x-small",
            markerscale=0.5,
            handlelength=1.2,
            borderpad=0.3,
            labelspacing=0.2,
        )
        set_rank_ticks(ax, df_filt)

        plt.tight_layout()
        save_figure_formats(
            None,
            output_dir / "log_probability_next_word_token_pointplot.png",
            formats=formats,
            default=("png",),
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close()

    # --- Plot 3: Normalized NLL Pointplot ---
    if "normalized_nll" in df_filt.columns:
        plt.figure(figsize=(8, 5))

        # Diagnostics
        nan_count = df_filt["normalized_nll"].isna().sum()
        if nan_count > 0:
            typer.echo(f"Warning: {nan_count} NaNs in normalized_nll")

        ax = sns.pointplot(
            data=df_filt,
            x="item_rank",
            y="normalized_nll",
            errorbar="se",
            capsize=0.2,
            hue="sequence_type",
            palette=palette,
            hue_order=["Original", "Mismatched Prefix", "Internal Shuffle"],
        )
        ax.set_title("Probability of Next-Word-Token in Sequence (NLL)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Rank", fontsize=12)
        ax.set_ylabel("Normalized NLL", fontsize=12)
        ax.axvline(x=split_point - 0.5, color="red", linestyle="--", linewidth=2, label="Sequence Split")
        _handles, _labels = ax.get_legend_handles_labels()
        _desired = ["Original", "Mismatched Prefix", "Internal Shuffle", "Sequence Split"]
        _hl = dict(zip(_labels, _handles, strict=True))
        ax.legend(
            [_hl[lo] for lo in _desired if lo in _hl],
            [lo for lo in _desired if lo in _hl],
            loc="upper center",
            fontsize="x-small",
            markerscale=0.5,
            handlelength=1.2,
            borderpad=0.3,
            labelspacing=0.2,
        )
        set_rank_ticks(ax, df_filt)

        plt.ylim(5, 8)
        plt.tight_layout()
        save_figure_formats(
            None,
            output_dir / "normalized_nll_pointplot.png",
            formats=formats,
            default=("png",),
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close()

    typer.echo("Calculating Perplexity...")
    df_full_sequences = df[df["item_rank"] <= 36].copy()

    if "normalized_nll" in df_full_sequences.columns:
        df_perplexity = (
            df_full_sequences.groupby(["id", "sequence_type", "permutation"])["normalized_nll"]
            .apply(calculate_perplexity_from_nll)
            .reset_index(name="perplexity")
        )

        if "sequence_type" in df_perplexity.columns:
            df_perplexity["sequence_type"] = df_perplexity["sequence_type"].map(sequence_type_legend_map)

        df_perplexity_avg = df_perplexity.groupby(["sequence_type", "id"])["perplexity"].mean().reset_index()

        seq_types = df_perplexity_avg["sequence_type"].unique()
        if len(seq_types) > 0:
            common_ids = set(df_perplexity_avg[df_perplexity_avg["sequence_type"] == seq_types[0]]["id"].unique())
            for st in seq_types[1:]:
                st_ids = set(df_perplexity_avg[df_perplexity_avg["sequence_type"] == st]["id"].unique())
                common_ids &= st_ids

            df_perplexity_avg = df_perplexity_avg[df_perplexity_avg["id"].isin(common_ids)]
            typer.echo(f"Filtered to {len(common_ids)} common IDs for Perplexity comparison.")

        typer.echo("Mean perplexity per condition (participant-level):")
        for sequence_type, mean_perplexity in df_perplexity_avg.groupby("sequence_type")["perplexity"].mean().items():
            typer.echo(f"  {sequence_type}: M={float(mean_perplexity):.2f}")

        plt.figure(figsize=(6, 6))
        order = ["Original", "Mismatched Prefix", "Internal Shuffle"]
        order = [o for o in order if o in df_perplexity_avg["sequence_type"].unique()]

        ax = sns.boxplot(
            data=df_perplexity_avg,
            x="sequence_type",
            y="perplexity",
            order=order,
            hue="sequence_type",
            palette=palette,
            showfliers=False,
        )
        ax.set_xlabel("Sequence Type")
        ax.set_ylabel("Perplexity (lower better)")
        title_obj = plt.suptitle("Semantic Sequence Test\n(Llama 3.3 70B-Instruct)", y=0.92)
        title_obj.set_fontsize(title_obj.get_fontsize() * 0.8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [
                label.replace("Mismatched Prefix", "Mismatched\nPrefix").replace(
                    "Internal Shuffle", "Internal\nShuffle"
                )
                for label in order
            ]
        )

        pairs = [("Mismatched Prefix", "Original")]
        valid_pairs = [p for p in pairs if p[0] in order and p[1] in order]

        if valid_pairs:
            typer.echo("=" * 60)
            typer.echo("PAIRED STATS: PERPLEXITY (participant-level)")
            typer.echo("=" * 60)
            for a, b in valid_pairs:
                wide = (
                    df_perplexity_avg[df_perplexity_avg["sequence_type"].isin([a, b])]
                    .pivot(index="id", columns="sequence_type", values="perplexity")
                    .dropna(subset=[a, b])
                )
                n_pairs = int(wide.shape[0])
                if n_pairs == 0:
                    typer.echo(f"{a} vs {b}: n=0 (no complete pairs)")
                    continue
                diff = wide[a] - wide[b]
                mean_a = float(wide[a].mean())
                mean_b = float(wide[b].mean())
                mean_diff = float(diff.mean())
                res = stats.wilcoxon(wide[a].to_numpy(), wide[b].to_numpy(), alternative="two-sided")
                typer.echo(
                    f"{a} vs {b}: n={n_pairs}, mean({a})={mean_a:.6g}, mean({b})={mean_b:.6g}, "
                    f"mean_diff({a}-{b})={mean_diff:.6g}, W={float(res.statistic):.6g}, p={float(res.pvalue):.6g}"
                )

            annotator = Annotator(
                ax, valid_pairs, data=df_perplexity_avg, x="sequence_type", y="perplexity", order=order
            )
            annotator.configure(test="Wilcoxon", text_format="star", loc="inside")
            annotator.apply_and_annotate()

        plt.tight_layout()
        save_figure_formats(
            None,
            output_dir / "boxplot_subject_averaged_perplexity.png",
            formats=formats,
            default=("png",),
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close()

    df_filt_prob = df[(df["item_rank"] <= 36) & (df["item_rank"] >= split_point)].copy()

    if "sequence_type" in df_filt_prob.columns and "id" in df_filt_prob.columns:
        ids_to_keep = df_filt_prob[df_filt_prob["sequence_type"] == "original"]["id"].unique()
        df_filt_prob = df_filt_prob[df_filt_prob["id"].isin(ids_to_keep)]

        df_filt_prob["sequence_type"] = df_filt_prob["sequence_type"].map(sequence_type_legend_map)

    if "normalized_probability" in df_filt_prob.columns:
        df_averaged = df_filt_prob.groupby(["id", "sequence_type"])["normalized_probability"].mean().reset_index()

        plt.figure(figsize=(6, 6))
        order = ["Original", "Internal Shuffle", "Mismatched Prefix"]
        order = [o for o in order if o in df_averaged["sequence_type"].unique()]

        ax = sns.boxplot(
            data=df_averaged,
            x="sequence_type",
            y="normalized_probability",
            order=order,
            hue="sequence_type",
            palette=palette,
            showfliers=False,
        )
        ax.set_xlabel("Sequence Type")
        ax.set_ylabel("Probability")
        plt.suptitle(f"Subject-Averaged Concept Probability\n($Rank \\geq {split_point}$)")

        pairs = [
            ("Internal Shuffle", "Original"),
            ("Mismatched Prefix", "Original"),
        ]
        valid_pairs = [p for p in pairs if p[0] in order and p[1] in order]

        if valid_pairs:
            # Print paired stats (two-sided Wilcoxon signed-rank) and mean differences.
            typer.echo("=" * 60)
            typer.echo("PAIRED STATS: SUBJECT-AVERAGED CONCEPT PROBABILITY (participant-level)")
            typer.echo("=" * 60)
            for a, b in valid_pairs:
                wide = (
                    df_averaged[df_averaged["sequence_type"].isin([a, b])]
                    .pivot(index="id", columns="sequence_type", values="normalized_probability")
                    .dropna(subset=[a, b])
                )
                n_pairs = int(wide.shape[0])
                if n_pairs == 0:
                    typer.echo(f"{a} vs {b}: n=0 (no complete pairs)")
                    continue
                diff = wide[a] - wide[b]
                mean_a = float(wide[a].mean())
                mean_b = float(wide[b].mean())
                mean_diff = float(diff.mean())
                res = stats.wilcoxon(wide[a].to_numpy(), wide[b].to_numpy(), alternative="two-sided")
                typer.echo(
                    f"{a} vs {b}: n={n_pairs}, mean({a})={mean_a:.6g}, mean({b})={mean_b:.6g}, "
                    f"mean_diff({a}-{b})={mean_diff:.6g}, W={float(res.statistic):.6g}, p={float(res.pvalue):.6g}"
                )

            annotator = Annotator(
                ax, valid_pairs, data=df_averaged, x="sequence_type", y="normalized_probability", order=order
            )
            annotator.configure(test="Wilcoxon", text_format="star", loc="inside")
            annotator.apply_and_annotate()

        plt.tight_layout()
        save_figure_formats(
            None,
            output_dir / "boxplot_subject_averaged_word_probability.png",
            formats=formats,
            default=("png",),
            bbox_inches="tight",
        )
        if show:
            plt.show()
        plt.close()

        if "Original" in df_averaged["sequence_type"].unique():
            high_prob_ids = df_averaged[df_averaged["normalized_probability"] > 0.0]["id"].unique()
            if len(high_prob_ids) > 0:
                example_id = high_prob_ids[0]
                df_ex = df_filt_prob[(df_filt_prob["id"] == example_id) & (df_filt_prob["sequence_type"] == "Original")]
                if not df_ex.empty and "nll" in df_ex.columns:
                    plt.figure(figsize=(15, 7))
                    df_ex_sorted = df_ex.sort_values(by="item_rank").copy()

                    ax = sns.lineplot(
                        data=df_ex_sorted, x="item_rank", y="nll", marker="o", linewidth=2, color="steelblue"
                    )
                    ax.set_title(
                        f"Normalized Probability per Token in Sequence for Subject ID: {example_id}",
                        fontsize=16,
                        fontweight="bold",
                    )
                    ax.set_xlabel("Sequence Position (Item Rank)", fontsize=12)
                    ax.set_ylabel("Normalized Probability (NLL)", fontsize=12)

                    ax.set_xticks(df_ex_sorted["item_rank"])
                    if "decoded_token_ids" in df_ex_sorted.columns:
                        ax.set_xticklabels(df_ex_sorted["decoded_token_ids"], rotation=60, ha="right", fontsize=10)

                    ax.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()
                    save_figure_formats(
                        None,
                        output_dir / f"normalized_probability_by_token_ids_subject_{example_id}.png",
                        formats=formats,
                        default=("png",),
                        bbox_inches="tight",
                    )
                    if show:
                        plt.show()
                    plt.close()

    typer.echo(f"Analysis complete. Plots saved to {output_dir}")


if __name__ == "__main__":
    seed_everything()
    app()
