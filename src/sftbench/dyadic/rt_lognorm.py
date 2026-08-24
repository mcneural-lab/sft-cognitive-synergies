"""Lognormal fit of human-human response times (Figure S13A).

Probability density of the response times (milliseconds) observed in the
human-human data, overlaid with the lognormal model used to generate the
artificial AI partner delays, split by category (animals, clothes).

The lognormal parameters for artificial delays were fit with pilot data
and are reproduced here as constants.
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
from scipy.stats import anderson, kstest

from sftbench import find_project_root
from sftbench.dyadic.load_data import load_data
from sftbench.figure_output import apply_bold_axis_style, save_figure_formats
from sftbench.reproducibility import seed_everything

app = typer.Typer()
console = Console()

DATA_DIR_REL = "data/dyadic/conceptnet"

# Lognormal parameters (s, loc, scale) fit offline to the human-human RTs.
PARAMS_ANIMALS = (0.7065, 467.43, 3579.91)
PARAMS_CLOTHES = (0.6955, 446.74, 3931.26)
CATEGORIES = [
    ("Animals", "animals", PARAMS_ANIMALS),
    ("Clothes", "clothes", PARAMS_CLOTHES),
]


def calculate_fit_measures(data, params):
    """Goodness-of-fit measures for the lognormal distribution."""
    data_clean = data.dropna()
    s_fit, loc_fit, scale_fit = params

    ks_stat, ks_p = kstest(data_clean, lambda x: stats.lognorm.cdf(x, s_fit, loc=loc_fit, scale=scale_fit))
    ad_stat, _, _ = anderson(np.log(data_clean - loc_fit + 1e-10), dist="norm")
    log_likelihood = np.sum(stats.lognorm.logpdf(data_clean, s_fit, loc=loc_fit, scale=scale_fit))

    n = len(data_clean)
    k = 3
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    sorted_data = np.sort(data_clean)
    n_points = len(sorted_data)
    empirical_quantiles = (np.arange(1, n_points + 1) - 0.5) / n_points
    theoretical_quantiles = stats.lognorm.cdf(sorted_data, s_fit, loc=loc_fit, scale=scale_fit)
    r_squared = np.corrcoef(empirical_quantiles, theoretical_quantiles)[0, 1] ** 2

    return {
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "ad_stat": ad_stat,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "r_squared": r_squared,
        "n": n,
    }


def print_fit_summary(fit_measures, category_name):
    """Console summary of fit measures."""
    console.print(f"\n{'=' * 50}")
    console.print(f"FIT MEASURES SUMMARY: {category_name.upper()}")
    console.print(f"{'=' * 50}")
    console.print(f"Sample size (n): {fit_measures['n']}")
    console.print(f"Log-likelihood: {fit_measures['log_likelihood']:.2f}")
    console.print(f"AIC: {fit_measures['aic']:.2f}")
    console.print(f"BIC: {fit_measures['bic']:.2f}")
    console.print(f"Pseudo R²: {fit_measures['r_squared']:.4f}")
    console.print("Goodness-of-fit tests:")
    console.print(f"  Kolmogorov-Smirnov: D = {fit_measures['ks_stat']:.4f}, p-value = {fit_measures['ks_p']:.4f}")
    console.print(f"  Anderson-Darling statistic: {fit_measures['ad_stat']:.3f}")


def create_panel(hh_data):
    """Figure S13A: two-panel histogram + fitted lognormal pdf."""
    console.print("Creating Figure S13A: lognormal RT fit...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fit_rows = []
    for ax, (cat_name, cat_key, params) in zip(axes, CATEGORIES, strict=True):
        cat_data = hh_data[hh_data["category"] == cat_key]["irt"]
        data_clean = cat_data.dropna()
        s_fit, loc_fit, scale_fit = params
        fit_measures = calculate_fit_measures(cat_data, params)

        ax.hist(data_clean, bins=50, density=True, alpha=0.7, color="#2E86AB", edgecolor="white", linewidth=0.5)
        x = np.linspace(data_clean.min(), data_clean.max() * 1.1, 1000)
        ax.plot(x, stats.lognorm.pdf(x, s_fit, loc=loc_fit, scale=scale_fit), color="black", linewidth=1.5)

        ax.set_xlabel("Human-Human Response Time (milliseconds)")
        if ax is axes[0]:
            ax.set_ylabel("Probability density")
        ax.set_title(cat_name, pad=20)
        fit_text = f"s = {s_fit:.4f}\nloc = {loc_fit:.2f}\nscale = {scale_fit:.2f}"
        ax.text(
            0.65,
            0.95,
            fit_text,
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_xlim(0, np.percentile(data_clean, 99))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

        fit_rows.append({"category": cat_name, **fit_measures})

    fig.supxlabel("Human-Human Response Time (milliseconds)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, pd.DataFrame(fit_rows)


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
    """Generate the lognormal RT fit panel (Figure S13A)."""
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

    sns.set_context("paper", font_scale=2.3)
    apply_bold_axis_style()

    df = load_data(data_dir=data_dir, cross_condition_only=False)
    hh_data = df[df["dyadType"] == "hh"]

    fig, fit_df = create_panel(hh_data)
    paths = save_figure_formats(
        fig,
        output_dir / "figS13a_lognormal_fit",
        formats=formats,
        default=("png",),
        dpi=300,
    )
    console.print(f"[green]Saved Figure S13A to {', '.join(str(p) for p in paths)}[/green]")

    for _, row in fit_df.iterrows():
        print_fit_summary(row.to_dict(), row["category"])
    fit_df.to_csv(output_dir / "figS13a_lognormal_fit_measures.csv", index=False)

    if show:
        plt.show()

    console.print("\n[bold]Finished![/bold]")


if __name__ == "__main__":
    seed_everything()
    app()
