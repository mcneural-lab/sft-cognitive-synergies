import itertools
import json
import warnings
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import tomli
import typer
from nltk.translate.bleu_score import sentence_bleu
from rich.console import Console
from scipy.stats import friedmanchisquare, wilcoxon
from statannotations.Annotator import Annotator

from sftbench import find_project_root
from sftbench.figure_output import apply_bold_axis_style, save_figure_formats
from sftbench.reproducibility import seed_everything, track

app = typer.Typer()
console = Console()

HUMAN_FILEPATH_REL = "data/sequences/human/hills/hills.csv"
MACHINE_FILEPATHS_REL = [
    "results/hills/generate/TPMs/naive-hills-animals_tpm_human_loo.csv",
    "results/hills/generate/animals-rank-1-gemini-2.5-flash-lite-prompt-animals-gen-snafu-embedding-switch.csv",
    "results/hills/generate/animals-rank-1-gemini-2.5-flash-prompt-animals-gen-snafu-embedding-switch.csv",
    "results/hills/generate/animals-rank-1-gemini-2.5-pro-prompt-animals-gen-snafu-embedding-switch.csv",
    "results/hills/generate/animals-rank-1-gemini-3.0-flash-prompt-animals-gen-snafu-embedding-switch.csv",
    "results/hills/generate/animals-rank-1-gemini-3.0-pro-prompt-animals-gen-snafu-embedding-switch.csv",
]
METHODS: list[str] = [
    "human",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "tpm_human_loo",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]

DISPLAY_NAMES: dict[str, str] = {
    "human": "Human",
    "gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "tpm_human_loo": "TPM-Human-LOO",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
    "gemini-3-flash-preview": "Gemini-3-Flash",
    "gemini-3-pro-preview": "Gemini-3-Pro",
}

DEFAULT_EXCLUDE_IDS: list[str] = []


def _bleu(hypothesis: list[str], references: list[list[str]]) -> float:
    """Compute sentence BLEU using the repo's established weight configuration."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        score = sentence_bleu(references, hypothesis, weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    return float(cast(float, score))


def _normalize_responses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["response"] = df["response"].astype(str).str.replace(" ", "").str.lower()
    return df


def _compute_human_refs_by_id(df_human: pd.DataFrame, split_point_index: int) -> dict[int, list[str]]:
    """
    Returns {id: token_list} for human sequences after grouping and split trimming.
    """
    df_human_responses = transform_dataframe(df_human, split_point_index)
    refs_by_id: dict[int, list[str]] = {}
    for _, row in df_human_responses.iterrows():
        block_id = int(row["id"])
        response = row["response"]
        if not isinstance(response, list):
            raise ValueError(f"Expected list response for human id={block_id}, got {type(response)}")
        refs_by_id[block_id] = [str(x) for x in response]
    return refs_by_id


def _compute_machine_hyp_by_id(df_machine: pd.DataFrame, model: str, split_point_index: int) -> dict[int, list[str]]:
    """
    Returns {id: token_list} for a given machine model at seq_type == 'seed-1'.
    """
    df_machine_responses = transform_dataframe(df_machine, split_point_index)
    df_machine_responses = df_machine_responses[
        (df_machine_responses["model"] == model) & (df_machine_responses["seq_type"] == "seed-1")
    ]
    hyp_by_id: dict[int, list[str]] = {}
    for _, row in df_machine_responses.iterrows():
        block_id = int(row["id"])
        response = row["response"]
        if not isinstance(response, list):
            raise ValueError(f"Expected list response for model={model} id={block_id}, got {type(response)}")
        hyp_by_id[block_id] = [str(x) for x in response]
    return hyp_by_id


def transform_dataframe(df: pd.DataFrame, split_point_index: int):
    """Group rows into per-block sequences and drop the shared seed prefix.

    Not interchangeable with the same-named helper in
    ``sftbench.micro.generative.plot_new_jaccard_control``: this one groups by
    ``model`` as well, because Fig 2C concatenates several models into one frame.
    """
    data_category_responses = (
        df.groupby(["id", "data-category", "seq_type", "model"])["response"].apply(list).reset_index()
    )
    data_category_responses["response"] = data_category_responses["response"].apply(
        lambda list_of_strings: list_of_strings[split_point_index:]
    )
    return data_category_responses


@app.command()
def plot(
    config_path: Path = typer.Option(
        Path("configs/plotting.toml"), help="Path to the plotting configuration TOML file."
    ),
    output_dir: Path = typer.Option(None, help="Directory to save the plots. Defaults to project_root/plots."),
    data_root: Path = typer.Option(
        Path("."),
        "--data-root",
        help="Root directory for relative input data/result paths. Use data-final for public-release artifacts.",
    ),
    show: bool = typer.Option(False, help="Show the plot instead of just saving."),
    exclude_humans: bool = typer.Option(
        False,
        "--exclude-humans",
        help="Exclude 'human' and 'tpm_human_loo' from Friedman/Nemenyi statistical testing (still shown in the plot).",
    ),
    exclude_ids: list[str] = typer.Option(
        DEFAULT_EXCLUDE_IDS,
        "--exclude-id",
        help=(
            "Exclude specific block ids from paired BLEU computation/stats. "
            "May be passed multiple times. No ids are excluded by default."
        ),
    ),
    adjacent_sig_test: str = typer.Option(
        "nemenyi",
        "--adjacent-sig-test",
        help=(
            "Which test to use for adjacent-pair plot annotations. "
            "'nemenyi' uses the Nemenyi post-hoc p-values (only if the omnibus Friedman test is significant). "
            "'wilcoxon_less' uses a one-sided paired Wilcoxon signed-rank test for each adjacent pair with "
            "H1: left model < right model (i.e., right has higher BLEU). "
            "When using 'wilcoxon_less', per-pair Wilcoxon statistics and p-values are printed."
        ),
    ),
    formats: str | None = typer.Option(
        None,
        "--formats",
        help="Comma-separated output formats for figures: png, pdf, svg. Defaults to png.",
    ),
):
    """
    Generates BLEU score plots for human vs machine sequences.
    """
    project_root = find_project_root()
    if project_root is None:
        console.print("[red]Could not find project root.[/red]")
        raise typer.Exit(code=1)

    context = "paper"
    style = "whitegrid"
    font_scale = 1.5
    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                config = tomli.load(f)
            seaborn_config = config.get("seaborn", {})
            context = seaborn_config.get("context", context)
            font_scale = seaborn_config.get("font_scale", font_scale)
            style = seaborn_config.get("style", style)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read config file: {e}[/yellow]")

    sns.set_context(context, font_scale=font_scale)
    sns.set_style(style)
    apply_bold_axis_style()

    if output_dir is None:
        output_dir = project_root / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.is_absolute():
        data_root = project_root / data_root

    human_filepath = data_root / HUMAN_FILEPATH_REL
    if not human_filepath.exists():
        console.print(f"[red]Human data file not found: {human_filepath}[/red]")
        raise typer.Exit(code=1)

    df_human = cast(pd.DataFrame, pd.read_csv(human_filepath))
    df_human = _normalize_responses(df_human)
    df_human = df_human[df_human["rank"] > 0]
    df_human["seq_type"] = "human"
    df_human["model"] = "human"
    df_human["data-category"] = "animals"

    filepaths = [data_root / path for path in MACHINE_FILEPATHS_REL]

    dfs = []
    for filepath in filepaths:
        if filepath.exists():
            df = cast(pd.DataFrame, pd.read_csv(filepath))
            df["seq_type"] = df["seq_type"].replace("rank-1", "seed-1")
            dfs.append(df)
        else:
            console.print(f"[yellow]Warning: Machine data file not found: {filepath}[/yellow]")

    if not dfs:
        console.print("[red]No machine data files found.[/red]")
        raise typer.Exit(code=1)

    df_machine = pd.concat(dfs, ignore_index=True)
    df_machine = _normalize_responses(df_machine)

    console.print(f"Models available in machine data: {sorted(df_machine.model.unique())}")

    # ---- Paired per-id BLEU scores (required for Friedman/Nemenyi) ----
    split_point_index = 1

    # Human references keyed by id
    human_refs_by_id = _compute_human_refs_by_id(cast(pd.DataFrame, df_human), split_point_index)
    all_human_ids = set(human_refs_by_id.keys())
    if not all_human_ids:
        console.print("[red]No human reference sequences found after preprocessing.[/red]")
        raise typer.Exit(code=1)

    machine_models = [m for m in METHODS if m != "human"]
    hyp_by_model_by_id: dict[str, dict[int, list[str]]] = {}
    for model in machine_models:
        hyp_by_model_by_id[model] = _compute_machine_hyp_by_id(df_machine, model, split_point_index)

    ids_in_all = set(all_human_ids)
    for model in machine_models:
        ids_in_all &= set(hyp_by_model_by_id[model].keys())

    ids_sorted = sorted(ids_in_all)
    console.print(f"Paired blocks (ids) in intersection across all {len(METHODS)} methods: {len(ids_sorted)}")
    if not ids_sorted:
        console.print("[red]No shared ids across all methods; cannot run paired tests.[/red]")
        raise typer.Exit(code=1)

    alpha = 0.05

    exclude_id_set = {int(x) for x in exclude_ids}
    ids_before_exclusion = len(ids_sorted)
    ids_sorted = [i for i in ids_sorted if i not in exclude_id_set]
    excluded_count = ids_before_exclusion - len(ids_sorted)
    if excluded_count:
        console.print(
            f"[yellow]Excluded {excluded_count} ids from paired BLEU/stats via --exclude-id: {sorted(exclude_id_set)}[/yellow]"
        )

    if not ids_sorted:
        console.print("[red]All paired ids were excluded; cannot run paired tests.[/red]")
        raise typer.Exit(code=1)

    all_human_references = [human_refs_by_id[i] for i in sorted(all_human_ids)]
    rows_long: list[dict[str, object]] = []

    console.print("[bold]Computing paired BLEU scores...[/bold]")
    for block_id in track(ids_sorted, description="Paired BLEU per id"):
        hyp_human = human_refs_by_id[block_id]
        other_refs = [human_refs_by_id[i] for i in sorted(all_human_ids) if i != block_id]
        rows_long.append({"id": block_id, "Model": "human", "BLEU Score": _bleu(hyp_human, other_refs)})

        for model in machine_models:
            hyp = hyp_by_model_by_id[model][block_id]
            rows_long.append({"id": block_id, "Model": model, "BLEU Score": _bleu(hyp, all_human_references)})

    model_scores_df = pd.DataFrame(rows_long)

    paired_scores_csv = output_dir / "bleu_scores_paired_by_id.csv"
    model_scores_df.to_csv(paired_scores_csv, index=False)
    console.print(f"[green]Saved paired per-id BLEU scores to {paired_scores_csv}[/green]")

    wide = cast(pd.DataFrame, model_scores_df.pivot(index="id", columns="Model", values="BLEU Score"))
    wide = wide[METHODS]  # enforce order + ensure all expected columns exist

    stats_methods = list(METHODS)
    if exclude_humans:
        stats_methods = [m for m in stats_methods if m not in {"human", "tpm_human_loo"}]

    wide_stats = wide[stats_methods]

    # ---- Friedman omnibus test (paired) ----
    friedman_args = [np.asarray(wide_stats[m]) for m in stats_methods]
    stat, p = friedmanchisquare(*friedman_args)

    console.print(
        f"[bold]Friedman test[/bold] (k={len(stats_methods)}, n={wide_stats.shape[0]}): statistic={stat:.6g}, p={p:.6g}"
    )

    # ---- Nemenyi post hoc (paired) ----
    nemenyi_stats = sp.posthoc_nemenyi_friedman(cast(pd.DataFrame, wide_stats))
    nemenyi_stats = nemenyi_stats.reindex(index=stats_methods, columns=stats_methods)

    nemenyi = pd.DataFrame(index=METHODS, columns=METHODS, data=np.nan)
    nemenyi.loc[stats_methods, stats_methods] = nemenyi_stats.to_numpy(dtype=float)

    nemenyi_csv = output_dir / "bleu_nemenyi_pvalues.csv"
    nemenyi.to_csv(nemenyi_csv)
    console.print(f"[green]Saved Nemenyi post hoc p-values to {nemenyi_csv}[/green]")

    console.print("\n[bold]Nemenyi post hoc p-values (Friedman):[/bold]")
    console.print(nemenyi_stats.round(6))

    sig_pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(stats_methods):
        for b in stats_methods[i + 1 :]:
            pv = float(nemenyi_stats.loc[a, b])
            if pv < alpha:
                sig_pairs.append((a, b, pv))

    sig_pairs = sorted(sig_pairs, key=lambda t: t[2])
    num_pairs = (len(stats_methods) * (len(stats_methods) - 1)) // 2
    console.print(f"\n[bold]Significant Nemenyi pairs (alpha={alpha}):[/bold] {len(sig_pairs)} / {num_pairs}")
    if sig_pairs:
        for a, b, pv in sig_pairs:
            console.print(f"  {a} vs {b}: p={pv:.6g}")
    else:
        console.print("  (none)")

    stats_json = output_dir / "bleu_paired_stats_summary.json"
    stats_summary = {
        "n_blocks": int(wide_stats.shape[0]),
        "methods_tested": list(stats_methods),
        "excluded_from_stats": ["human", "tpm_human_loo"] if exclude_humans else [],
        "excluded_ids": sorted([int(x) for x in exclude_ids]),
        "friedman_statistic": float(stat),
        "friedman_p": float(p),
        "nemenyi_pvalues_csv": nemenyi_csv.name,
        "paired_scores_csv": paired_scores_csv.name,
    }
    stats_json.write_text(json.dumps(stats_summary, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Saved stats summary to {stats_json}[/green]")

    plt.figure(figsize=(6, 7.5))
    medians = model_scores_df.groupby("Model", observed=False)["BLEU Score"].median().reindex(METHODS)
    model_order = list(medians.sort_values(ascending=True).index)
    display_order = [DISPLAY_NAMES.get(m, m) for m in model_order]

    blue_rgb = "#4C72B0"
    orange_rgb = "#DD8452"

    custom_palette = {}
    for m in model_order:
        if "human" in m.lower():
            custom_palette[m] = blue_rgb
        else:
            custom_palette[m] = orange_rgb

    model_scores_df["Model"] = pd.Categorical(model_scores_df["Model"], categories=model_order, ordered=True)
    model_scores_df["ModelDisplay"] = model_scores_df["Model"].astype(str).map(lambda m: DISPLAY_NAMES.get(m, m))
    model_scores_df["ModelDisplay"] = pd.Categorical(
        model_scores_df["ModelDisplay"], categories=display_order, ordered=True
    )

    model_scores_df = model_scores_df.sort_values("Model")
    ax = sns.boxplot(
        data=model_scores_df,
        x="ModelDisplay",
        y="BLEU Score",
        width=0.3,
        palette=custom_palette,
        hue="Model",
        legend=False,
    )

    human_scores = model_scores_df.loc[model_scores_df["Model"] == "human", "BLEU Score"].to_numpy()
    human_median = float(np.median(human_scores))
    ax.axhline(y=human_median, color="black", linestyle=":", linewidth=1.25, zorder=0.5)
    console.print(f"[bold]Human median BLEU[/bold]: {human_median:.6g}")

    adjacent_pairs: list[tuple[str, str]] = []
    for a, b in itertools.pairwise(model_order):
        if adjacent_sig_test == "nemenyi":
            if a in stats_methods and b in stats_methods:
                adjacent_pairs.append((a, b))
        else:
            adjacent_pairs.append((a, b))

    if not adjacent_pairs:
        console.print("[yellow]Warning: No adjacent model pairs available for annotation.[/yellow]")
    elif adjacent_sig_test == "nemenyi":
        if p >= alpha:
            console.print(f"[bold]Skipping plot significance annotations[/bold]: Friedman p={p:.6g} (alpha={alpha}).")
        else:
            pvalues: list[float] = []
            pairs_display: list[tuple[str, str]] = []

            for a, b in adjacent_pairs:
                pv = float(nemenyi.loc[a, b]) if pd.notna(nemenyi.loc[a, b]) else None
                if pv is None:
                    continue
                if pv >= alpha:
                    continue
                pvalues.append(pv)
                pairs_display.append((DISPLAY_NAMES.get(a, a), DISPLAY_NAMES.get(b, b)))

            if not pairs_display:
                console.print(
                    f"[yellow]Warning: No statistically significant adjacent model pairs to annotate (alpha={alpha}).[/yellow]"
                )
            else:
                annotator = Annotator(
                    ax=ax,
                    pairs=pairs_display,
                    data=model_scores_df,
                    x="ModelDisplay",
                    y="BLEU Score",
                    order=display_order,
                )
                annotator.configure(test=None, text_format="star", loc="inside", verbose=0)
                annotator.set_pvalues_and_annotate(pvalues)
    elif adjacent_sig_test == "wilcoxon_less":
        pvalues = []
        pairs_display = []
        console.print(
            f"\n[bold]Adjacent-pair Wilcoxon signed-rank tests[/bold] (paired, one-sided; H1: left < right; alpha={alpha}):"
        )

        for a, b in adjacent_pairs:
            a_vals = wide[a].to_numpy()
            b_vals = wide[b].to_numpy()
            try:
                res = wilcoxon(a_vals, b_vals, alternative="less", zero_method="wilcox")
                stat_w = float(res.statistic)
                pv = float(res.pvalue)
            except Exception as e:
                console.print(f"  {a} vs {b}: [yellow]Wilcoxon failed[/yellow] ({e})")
                continue

            a_disp = DISPLAY_NAMES.get(a, a)
            b_disp = DISPLAY_NAMES.get(b, b)
            console.print(f"  {a_disp} vs {b_disp}: W={stat_w:.6g}, p={pv:.6g}")

            if pv >= alpha:
                continue

            pvalues.append(pv)
            pairs_display.append((a_disp, b_disp))

        if not pairs_display:
            console.print(
                f"[yellow]Warning: No statistically significant adjacent model pairs to annotate via Wilcoxon (alpha={alpha}).[/yellow]"
            )
        else:
            annotator = Annotator(
                ax=ax,
                pairs=pairs_display,
                data=model_scores_df,
                x="ModelDisplay",
                y="BLEU Score",
                order=display_order,
            )
            annotator.configure(test=None, text_format="star", loc="inside", verbose=0)
            annotator.set_pvalues_and_annotate(pvalues)
    else:
        console.print(
            f"[yellow]Warning: Unknown --adjacent-sig-test={adjacent_sig_test!r}. Expected 'nemenyi' or 'wilcoxon_less'.[/yellow]"
        )

    y_max = float(model_scores_df["BLEU Score"].max())
    plt.ylim(top=min(1.0, y_max + 0.12))

    ax.set_title("BLEU Score Distribution Across Models", fontsize=16, y=1.0, pad=8)

    plt.xlabel("Model")
    plt.ylabel("BLEU Score")
    plt.xticks(rotation=45, ha="right")

    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    output_path = output_dir / "bleu_score_distribution_new.png"
    saved_paths = save_figure_formats(None, output_path, formats=formats, default=("png",), bbox_inches="tight")
    console.print(f"[green]Plot saved to {', '.join(str(path) for path in saved_paths)}[/green]")

    console.print("\n[bold]Note on Fig 2C visualization:[/bold] boxplot shows median/IQR; whiskers = 1.5×IQR.")
    console.print("Paired statistics saved (Friedman + Nemenyi) for manuscript reporting.")

    if show:
        plt.show()

    console.print("\nMean BLEU Scores (95% bootstrap CI):")

    rng = np.random.default_rng(0)
    n_boot = 10_000
    rows_ci: list[dict[str, object]] = []
    for m in METHODS:
        vals = model_scores_df.loc[model_scores_df["Model"] == m, "BLEU Score"].to_numpy()
        mean = float(np.mean(vals))
        boot_means = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)

        lo, hi = np.quantile(boot_means, [0.025, 0.975])
        rows_ci.append({"Model": m, "Mean": round(mean, 4), "CI95": f"[{lo:.4f}, {hi:.4f}]"})

    mean_ci_df = pd.DataFrame(rows_ci)
    mean_ci_df["ModelDisplay"] = mean_ci_df["Model"].map(lambda x: DISPLAY_NAMES.get(x, x))
    console.print(mean_ci_df[["ModelDisplay", "Mean", "CI95"]])


if __name__ == "__main__":
    seed_everything()
    app()
