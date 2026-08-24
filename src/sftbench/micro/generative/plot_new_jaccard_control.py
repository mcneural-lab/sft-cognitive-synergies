from pathlib import Path
from types import SimpleNamespace
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from scipy.stats import wilcoxon
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
    _root = Path(__file__).resolve().parents[4]

DEFAULT_HUMAN_DATA_PATH = _root / "data/final/filtered/all_human_wordpreds_unique_filtered.csv"
app = typer.Typer(help="CLI to plot Jaccard index and BLEU score evaluations.")


def load_plotting_config(config_path: Path):
    """Loads seaborn context and style from a TOML file."""
    if not config_path.exists():
        typer.echo(f"Config file not found: {config_path}. Using defaults.")
        sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
        apply_bold_axis_style()
        return

    typer.echo(f"Loading plotting configuration from: {config_path}")
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    seaborn_cfg = config.get("seaborn", {})
    context = seaborn_cfg.get("context", "paper")
    font_scale = seaborn_cfg.get("font_scale", 1.5)
    style = seaborn_cfg.get("style", "whitegrid")

    sns.set_theme(context=context, style=style, font_scale=font_scale)
    apply_bold_axis_style()


def transform_dataframe(df: pd.DataFrame, split_point_index: int = 17, label: str = "new") -> pd.DataFrame:
    """
    Groups responses by ID and category, creates a list of responses,
    and slices the list from split_point_index.
    """
    data_category_responses = cast(
        pd.DataFrame,
        df.groupby(["id", "data-category", "seq_type"])["response"].apply(list).reset_index(),
    )

    data_category_responses["response"] = data_category_responses["response"].apply(
        lambda list_of_strings: list_of_strings[split_point_index:]
    )
    data_category_responses["label"] = label
    return data_category_responses


def _prepare_data(
    human_data: Path,
    machine_data_dir: Path,
    config: Path,
    output_dir: Path,
    split_point: int,
    model_name: str,
) -> SimpleNamespace:
    """
    Shared data-loading / transform / Jaccard-computation pipeline used by both
    `figure-4d-control` and `figure-4d-control-misc`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    load_plotting_config(config)

    if not human_data.exists():
        typer.echo(f"Error: Human data file not found at {human_data}")
        raise typer.Exit(code=1)

    # --- Load Data ---
    typer.echo("Loading human data...")
    df_human = cast(pd.DataFrame, pd.read_csv(human_data))
    df_human["response"] = df_human["response"].astype(str).str.replace(" ", "").str.lower()
    df_human["seq_type"] = "full"

    typer.echo(f"Loading machine data from directory: {machine_data_dir}")
    machine_filepaths = list(machine_data_dir.glob("*.csv"))
    if not machine_filepaths:
        typer.echo("Error: No CSV files found in machine data directory.")
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(machine_filepaths)} files.")
    dfs = []
    for filepath in machine_filepaths:
        df = cast(pd.DataFrame, pd.read_csv(filepath))
        category = filepath.name.split("-")[0]
        df["data-category"] = category
        dfs.append(df)

    df_machine = cast(pd.DataFrame, pd.concat(dfs, ignore_index=True))

    df_machine["response"] = df_machine["response"].astype(str).str.replace(" ", "").str.lower()

    shared_ids = list(set(df_human["id"].unique()) & set(df_machine["id"].unique()))
    typer.echo(f"Number of shared IDs: {len(shared_ids)}")
    df_machine = df_machine[df_machine["id"].isin(shared_ids)]
    df_human = df_human[df_human["id"].isin(shared_ids)]

    # --- Build human first-half filter sets
    typer.echo(f"Building human first-half filter sets (split point: {split_point})...")
    human_first_half_sets = (
        df_human.groupby(["id", "data-category"])["response"]
        .apply(list)
        .reset_index()
        .rename(columns={"response": "_human_first_half"})
    )
    human_first_half_sets["_human_first_half"] = human_first_half_sets["_human_first_half"].apply(
        lambda words: set(words[:split_point])
    )

    typer.echo(f"Transforming data (split point: {split_point})...")
    df_machine_responses = transform_dataframe(df_machine, split_point, "generated")
    df_human_responses = transform_dataframe(df_human, split_point, "human")

    df_machine_responses = df_machine_responses.merge(
        human_first_half_sets,
        on=["id", "data-category"],
        how="left",
    )

    df_machine_responses["response_unfiltered"] = df_machine_responses["response"].apply(list)

    df_machine_responses["n_machine_secondhalf_total"] = df_machine_responses["response"].apply(len)
    df_machine_responses["n_human_firsthalf_words"] = df_machine_responses.apply(
        lambda row: len([w for w in row["response"] if w in (row["_human_first_half"] or set())]),
        axis=1,
    )
    df_machine_responses["n_human_firsthalf_size"] = df_machine_responses["_human_first_half"].apply(
        lambda s: len(s) if isinstance(s, set) else 0
    )
    df_machine_responses["n_human_firsthalf_union_machine"] = df_machine_responses.apply(
        lambda row: len((row["_human_first_half"] or set()) | set(row["response"])),
        axis=1,
    )
    df_machine_responses["response"] = df_machine_responses.apply(
        lambda row: [w for w in row["response"] if w not in (row["_human_first_half"] or set())],
        axis=1,
    )
    df_machine_responses = df_machine_responses.drop(columns=["_human_first_half"])

    _fw_seq_types = {"rank-1", "seed-1"}
    _fh_seq_types = {"rank-17", "sub-seq"}

    _df_fw_echo = df_machine_responses[df_machine_responses["seq_type"].isin(_fw_seq_types)][
        ["id", "data-category", "n_human_firsthalf_words"]
    ].rename(columns={"n_human_firsthalf_words": "_o_fw"})
    df_machine_responses = df_machine_responses.merge(_df_fw_echo, on=["id", "data-category"], how="left")
    df_machine_responses["_o_fw"] = df_machine_responses["_o_fw"].fillna(0).astype(int)

    def _size_match_fh(row):
        if row["seq_type"] in _fh_seq_types and row["_o_fw"] > 0:
            seq = row["response"]
            trim = row["_o_fw"]
            return seq[:-trim] if trim < len(seq) else []
        return row["response"]

    df_machine_responses["n_size_matched_removed"] = df_machine_responses.apply(
        lambda row: row["_o_fw"] if row["seq_type"] in _fh_seq_types else 0, axis=1
    )
    df_machine_responses["response"] = df_machine_responses.apply(_size_match_fh, axis=1)
    df_machine_responses = df_machine_responses.drop(columns=["_o_fw"])

    df_combined_responses = pd.merge(
        df_machine_responses,
        df_human_responses,
        on="id",
        how="inner",
        suffixes=("_machine", "_human"),
    )
    df_combined_responses = df_combined_responses[
        (df_combined_responses["response_machine"].apply(len) > 0)
        & (df_combined_responses["response_human"].apply(len) > 0)
    ]

    typer.echo("Calculating Jaccard indices...")
    jaccard_indices = []
    jaccard_original_indices = []
    intersection_sizes = []
    union_sizes = []
    for machine_resp, human_resp, unfiltered_resp in zip(
        df_combined_responses["response_machine"],
        df_combined_responses["response_human"],
        df_combined_responses["response_unfiltered"],
        strict=True,
    ):
        set_h = set(human_resp)

        # Control JI (filtered)
        set_m = set(machine_resp)
        inter = set_m & set_h
        uni = set_m | set_h
        intersection_sizes.append(len(inter))
        union_sizes.append(len(uni))
        jaccard_indices.append(len(inter) / len(uni) if uni else 1.0)

        # Original JI (unfiltered)
        set_m_orig = set(unfiltered_resp)
        uni_orig = set_m_orig | set_h
        inter_orig = set_m_orig & set_h
        jaccard_original_indices.append(len(inter_orig) / len(uni_orig) if uni_orig else 1.0)

    df_combined_responses["jaccard_index"] = jaccard_indices
    df_combined_responses["jaccard_index_original"] = jaccard_original_indices
    df_combined_responses["memory_avoidance_effect"] = (
        df_combined_responses["jaccard_index"] - df_combined_responses["jaccard_index_original"]
    )
    df_combined_responses["n_intersection"] = intersection_sizes
    df_combined_responses["n_union"] = union_sizes
    # Fraction of machine second-half words that came from the human's first half (pre-filter).
    df_combined_responses["frac_human_firsthalf_words"] = df_combined_responses[
        "n_human_firsthalf_words"
    ] / df_combined_responses["n_machine_secondhalf_total"].replace(0, np.nan)

    df_combined_responses["n_correct_preds_unfiltered"] = df_combined_responses.apply(
        lambda row: len(set(row["response_unfiltered"]) & set(row["response_human"])),
        axis=1,
    )
    df_combined_responses["n_relevant_unfiltered"] = (
        df_combined_responses["n_correct_preds_unfiltered"] + df_combined_responses["n_human_firsthalf_words"]
    )

    df_combined_responses["n_human_secondhalf_words"] = df_combined_responses["response_human"].apply(len)

    df_combined_responses["frac_relevant_unfiltered"] = df_combined_responses[
        "n_relevant_unfiltered"
    ] / df_combined_responses["n_human_secondhalf_words"].replace(0, np.nan)

    # --- Map Sequence Types for Display ---
    # Map rank-1 -> First word, rank-17 -> First 17 words
    seq_type_mapping = {
        "rank-1": "First word",
        "rank-17": f"First {split_point} words",
        # Keep old ones just in case
        "seed-1": "First word",
        "sub-seq": f"First {split_point} words",
    }

    if "seq_type_machine" in df_combined_responses.columns:
        df_combined_responses["seq_type_machine"] = (
            df_combined_responses["seq_type_machine"]
            .map(seq_type_mapping)  # type: ignore
            .fillna(df_combined_responses["seq_type_machine"])
        )

    # --- Sort Data for Plotting and Stats ---
    df_combined_responses = df_combined_responses.sort_values(by=["data-category_machine", "id", "seq_type_machine"])
    seed_label = seq_type_mapping.get("rank-1", "First word")
    sub_label = seq_type_mapping.get("rank-17", f"First {split_point} words")
    n_hues_data = df_combined_responses["seq_type_machine"].nunique()
    hue_order = [seed_label, sub_label] if n_hues_data > 1 else [seed_label]
    palette = {seed_label: "dimgray", sub_label: "#1f77b4"} if n_hues_data > 1 else {seed_label: "dimgray"}
    n_hues = len(hue_order)
    dodge_width = 0.8
    model_title_part = model_name.replace("_", " ").replace("-", " ").title()

    # --- Save sequences CSV ---
    typer.echo("Saving sequences to CSV...")
    df_sequences = df_combined_responses[
        [
            "id",
            "data-category_machine",
            "seq_type_machine",
            "n_human_firsthalf_size",
            "n_machine_secondhalf_total",
            "n_human_firsthalf_words",
            "n_human_firsthalf_union_machine",
            "frac_human_firsthalf_words",
            "n_intersection",
            "n_union",
            "jaccard_index_original",
            "jaccard_index",
            "memory_avoidance_effect",
            "n_correct_preds_unfiltered",
            "n_relevant_unfiltered",
            "n_human_secondhalf_words",
            "frac_relevant_unfiltered",
            "n_size_matched_removed",
            "response_machine",
            "response_human",
        ]
    ].copy()
    df_sequences = df_sequences.rename(
        columns={
            "data-category_machine": "category",
            "seq_type_machine": "condition",
        }
    )
    # Serialise list columns as space-separated strings
    df_sequences["response_machine"] = df_sequences["response_machine"].apply(
        lambda lst: " ".join(lst) if isinstance(lst, list) else lst
    )
    df_sequences["response_human"] = df_sequences["response_human"].apply(
        lambda lst: " ".join(lst) if isinstance(lst, list) else lst
    )
    sequences_csv_path = output_dir / "sequences.csv"
    df_sequences.to_csv(sequences_csv_path, index=False)
    typer.echo(f"Sequences saved to {sequences_csv_path}")

    return SimpleNamespace(
        df_combined_responses=df_combined_responses,
        df_human=df_human,
        df_machine=df_machine,
        df_human_responses=df_human_responses,
        human_first_half_sets=human_first_half_sets,
        seq_type_mapping=seq_type_mapping,
        seed_label=seed_label,
        sub_label=sub_label,
        hue_order=hue_order,
        palette=palette,
        n_hues_data=n_hues_data,
        n_hues=n_hues,
        dodge_width=dodge_width,
        model_title_part=model_title_part,
    )


@app.command("figure-4d-control")
def figure_4d_control(
    human_data: Path = typer.Option(
        DEFAULT_HUMAN_DATA_PATH,
        help="Path to the human data CSV file.",
    ),
    machine_data_dir: Path = typer.Option(
        ...,
        help="Directory containing machine generated CSV files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    config: Path = typer.Option(
        _root / "configs" / "plotting.toml",
        help="Path to the plotting configuration TOML.",
    ),
    output_dir: Path = typer.Option(Path("plots"), help="Directory to save generated plots."),
    show: bool = typer.Option(True, help="Whether to show plots interactively."),
    split_point: int = typer.Option(17, help="Split point for sequence slicing (e.g., 17 for seed length)."),
    model_name: str = typer.Option("gemini-3.0-pro", help="Model name identifier for plot titles."),
    formats: str | None = typer.Option(
        None,
        "--formats",
        help="Comma-separated output formats for figures: png, pdf, svg. Defaults to png.",
    ),
):
    """
    Reproduce ONLY the figure-4d Jaccard index comparison plot
    (jaccard_index_comparison_by_category.png) from a directory of results.

    Control version: for each (id, data-category) pair the words in the HUMAN
    sequence's first half (positions 0 .. split_point-1) are removed from the
    machine-generated second-half word list before Jaccard computation.  This
    applies to both the "First Word" (rank-1) and "First Half" (rank-17) conditions.
    """
    ns = _prepare_data(human_data, machine_data_dir, config, output_dir, split_point, model_name)
    df_combined_responses = ns.df_combined_responses
    seq_type_mapping = ns.seq_type_mapping
    hue_order = ns.hue_order
    palette = ns.palette
    n_hues = ns.n_hues
    n_hues_data = ns.n_hues_data
    dodge_width = ns.dodge_width
    model_name_for_title = model_name

    typer.echo("\nGenerating comparison plot...")
    plt.figure(figsize=(7, 7))
    sns.set_context("paper", font_scale=2.0)
    apply_bold_axis_style()

    ax = sns.stripplot(
        data=df_combined_responses,
        x="data-category_machine",
        y="jaccard_index",
        hue="seq_type_machine",
        hue_order=hue_order,
        palette=palette,
        jitter=0.15,
        dodge=True,
        alpha=0.5,
        size=6,
        zorder=1,
    )

    xticks_locs = ax.get_xticks()
    xticks_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    category_positions = {label: loc for label, loc in zip(xticks_labels, xticks_locs, strict=False)}

    if n_hues > 1:
        offsets = np.linspace(0, dodge_width - dodge_width / n_hues, n_hues)
        offsets -= offsets.mean()
        hue_offsets = {hue: offset for hue, offset in zip(hue_order, offsets, strict=False)}

        pivoted_data = (
            df_combined_responses.pivot_table(
                index=["id", "data-category_machine"], columns="seq_type_machine", values="jaccard_index"
            )
            .dropna()
            .reset_index()
        )

        for _, row in pivoted_data.iterrows():
            category = row["data-category_machine"]
            if category in category_positions:
                x_base = category_positions[category]
                y_values = [row[hue] for hue in hue_order if hue in row]
                x_values = [x_base + hue_offsets[hue] for hue in hue_order if hue in row]

                if len(y_values) == len(hue_order):  # Only draw if we have both points
                    ax.plot(x_values, y_values, color="grey", linestyle="-", linewidth=1, zorder=1)

    # Aesthetics
    model_title_part = model_name_for_title.replace("_", " ").replace("-", " ").title()
    ax.set_title(f"Generated Sequence Map Test\n({model_title_part})")
    ax.set_xlabel("Category")
    ax.set_ylabel("Jaccard Index")

    try:
        dodge_val = 0.6 if n_hues_data > 1 else 0
        sns.pointplot(
            x="data-category_machine",
            y="jaccard_index",
            data=df_combined_responses,
            linestyle="none",
            capsize=0.2,
            ax=ax,
            errorbar=("ci", 95),
            seed=SEED,
            hue="seq_type_machine",
            hue_order=hue_order,
            palette=palette,
            dodge=dodge_val,
            legend=False,
            marker="s",
            markersize=5,
        )
    except Exception as e:
        typer.echo(f"Warning: Could not create pointplot: {e}")

    # Annotations
    unique_cats = sorted(df_combined_responses["data-category_machine"].unique())

    hue1 = seq_type_mapping.get("rank-1", "First word")
    hue2 = seq_type_mapping.get("rank-17", f"First {split_point} words")

    df_for_annot = df_combined_responses.copy()

    if {"id", "data-category_machine", "seq_type_machine", "jaccard_index"}.issubset(df_for_annot.columns):
        df_pairs = (
            df_for_annot.groupby(["data-category_machine", "id"])["seq_type_machine"]
            .nunique()
            .reset_index(name="n_seq_types")
        )
        valid_pairs = df_pairs[df_pairs["n_seq_types"] >= 2][["data-category_machine", "id"]]
        df_for_annot = df_for_annot.merge(valid_pairs, on=["data-category_machine", "id"], how="inner")

    available_hues = df_for_annot["seq_type_machine"].unique()

    if hue1 in available_hues and hue2 in available_hues:
        annot_pairs = [((cat, hue1), (cat, hue2)) for cat in unique_cats]

        hue_plot_params = {
            "data": df_for_annot,
            "x": "data-category_machine",
            "y": "jaccard_index",
            "order": unique_cats,
            "hue": "seq_type_machine",
        }

        try:
            annotator = Annotator(ax, annot_pairs, **hue_plot_params)  # type: ignore
            # Paired Wilcoxon signed-rank test (two-sided by default in statannotations).
            annotator.configure(test="Wilcoxon")
            annotator.apply_and_annotate()
        except Exception as e:
            typer.echo(f"Warning: Could not annotate plot: {e}")

    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")
    _ylo_ji, _yhi_ji = ax.get_ylim()
    ax.set_ylim(_ylo_ji, _yhi_ji * 1.2)
    legend = ax.get_legend()
    if legend:
        legend.set_title("Sequence Context")
        legend.set(loc="upper left")
        legend.get_frame().set_alpha(0.9)
    plt.tight_layout()

    output_path = output_dir / "jaccard_index_comparison_by_category.png"
    saved_paths = save_figure_formats(
        None,
        output_path,
        formats=formats,
        default=("png",),
        bbox_inches="tight",
        dpi=300,
    )
    typer.echo(f"Plot saved to {', '.join(str(path) for path in saved_paths)}")

    # --- Print Stats ---
    typer.echo("\nMean Jaccard Index by Category and Sequence Context:")
    mean_jaccard_by_group = df_combined_responses.groupby(["data-category_machine", "seq_type_machine"])[
        "jaccard_index"
    ].mean()
    typer.echo(mean_jaccard_by_group)

    typer.echo("\nPaired Wilcoxon signed-rank tests (Seed vs Sub) by Category:")

    df_stats = df_for_annot.copy()
    df_pivot = (
        df_stats.pivot_table(
            index=["id", "data-category_machine"], columns="seq_type_machine", values="jaccard_index", aggfunc="mean"
        )
        .dropna()
        .reset_index()
    )

    for cat in unique_cats:
        df_cat = df_pivot[df_pivot["data-category_machine"] == cat]
        if df_cat.empty or (hue1 not in df_cat.columns) or (hue2 not in df_cat.columns):
            typer.echo(f"  {cat}: N/A (missing paired data for both conditions)")
            continue

        x = df_cat[hue1].astype(float).to_numpy()
        y = df_cat[hue2].astype(float).to_numpy()
        n = min(len(x), len(y))
        dof = n - 1

        if n < 2:
            typer.echo(f"  {cat}: N/A (n={n}, dof={dof})")
            continue

        res = wilcoxon(x, y, alternative="two-sided")
        mean_x = float(np.nanmean(x))
        mean_y = float(np.nanmean(y))
        mean_diff = float(np.nanmean(x - y))
        typer.echo(
            f"  {cat}: n={n}, mean({hue1})={mean_x:.4f}, mean({hue2})={mean_y:.4f}, "
            f"mean_diff({hue1}-{hue2})={mean_diff:.4f}, W={float(res.statistic):.6g}, p={float(res.pvalue):.4g}"
        )

    if show:
        plt.show()


if __name__ == "__main__":
    seed_everything()
    app()
