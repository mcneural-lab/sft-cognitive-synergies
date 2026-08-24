"""Fixed- and mixed-effects regression on response times (Table S3).

Ordinary least-squares and mixed-effects (random intercept per source)
regressions of (z-scored) log human response times on category, embedding
similarity, task phase, and dyad/prompt condition.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import typer
from rich.console import Console
from statsmodels.regression.mixed_linear_model import MixedLM

from sftbench import find_project_root
from sftbench.dyadic.load_data import load_data
from sftbench.dyadic.utils import save_model_summary
from sftbench.reproducibility import seed_everything

app = typer.Typer()
console = Console()

DATA_DIR_REL = "data/dyadic/conceptnet"

# other_similarity (= embedding_similarity for hh/ha); see also self_similarity.
EMBSIM_VAR = "embedding_similarity"


def build_model_frame(df):
    """Per source x dyad-prompt x phase x category means used by every model."""
    combined_df = df[df["sourceType"] == "human"].copy()
    combined_df.loc[combined_df.dyadType == "hh", "prompt"] = "instructions"
    combined_df["dyad_prompt"] = combined_df["dyadType"].astype(str) + "_" + combined_df["prompt"].astype(str)

    combined_mean_df = (
        combined_df.groupby(["source", "dyad_prompt", "word_index_split", "category"])[
            [EMBSIM_VAR, "log_irt_zscore", "log_irt", "word_index"]
        ]
        .agg(
            {
                EMBSIM_VAR: "mean",
                "log_irt_zscore": "mean",
                "log_irt": "mean",
                "word_index": lambda x: x.iloc[-1] - x.iloc[0],
            }
        )
        .reset_index()
    )
    combined_mean_df = combined_mean_df.rename(columns={"word_index": "word_total"})
    combined_mean_df["word_rate"] = combined_mean_df["word_total"] / combined_mean_df["log_irt"]
    return combined_mean_df


def coefficient_table(model, model_name):
    """Tidy coefficient/p-value table for a fitted statsmodels result."""
    table = pd.DataFrame(
        {
            "model": model_name,
            "term": model.params.index,
            "coef": model.params.values,
            "pvalue": model.pvalues.reindex(model.params.index).values,
        }
    )
    return table


@app.command()
def plot(
    data_root: pathlib.Path = typer.Option(
        Path("."), "--data-root", help="Root directory for relative input data paths."
    ),
    output_dir: pathlib.Path = typer.Option(
        None,
        "--output-dir",
        help="Directory to save model summaries. Defaults to project_root/outputs/figures/supp/table-s3.",
    ),
    formats: str | None = typer.Option(
        None, "--formats", help="Unused (Table S3 has no figure); accepted for harness uniformity."
    ),
    show: bool = typer.Option(False, "--show/--no-show", help="Unused (Table S3 has no figure)."),
):
    """Generate the Table S3 regression summaries."""
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
        output_dir = project_root / "outputs" / "figures" / "supp" / "table-s3"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_dir=data_dir, cross_condition_only=False, include_solo=False)
    combined_mean_df = build_model_frame(df)

    base_formula = (
        f"{{response}} ~ category + {EMBSIM_VAR}*word_index_split"
        '*C(dyad_prompt, Treatment(reference="hh_instructions"))'
    )

    coef_tables = []
    for response, slug, title in [
        ("log_irt", "logIRT_word_index_split_vs_hh_instructions", "OLS: log_irt"),
        ("log_irt_zscore", "logIRTz_word_index_split_vs_hh_instructions", "OLS: log_irt_zscore"),
    ]:
        formula = base_formula.format(response=response)
        model = sm.OLS.from_formula(formula, data=combined_mean_df).fit()
        console.print(f"\n[bold]{title}[/bold]")
        console.print(str(model.summary()))
        save_model_summary(model, str(output_dir / f"{slug}.txt"))
        table = coefficient_table(model, title)
        # OLS fits are solved directly; the flag exists so the CSV column is meaningful
        # for every row rather than empty wherever the model is not a mixed-effects one.
        table["converged"] = True
        coef_tables.append(table)

    for response, slug, title in [
        ("log_irt", "logIRT_word_index_split_mixed_effects", "MixedLM: log_irt"),
        ("log_irt_zscore", "logIRTz_word_index_split_mixed_effects", "MixedLM: log_irt_zscore"),
    ]:
        formula = base_formula.format(response=response)
        model = MixedLM.from_formula(formula, groups=combined_mean_df["source"], data=combined_mean_df).fit()
        console.print(f"\n[bold]{title} (random intercept per source)[/bold]")
        console.print(str(model.summary()))
        # The likelihood optimiser does not always converge on these frames. Table S3
        # reports the OLS fits, so a failed mixed-effects fit is not fatal, but its
        # coefficients go into the same CSV and must not read as if they were sound.
        converged = bool(getattr(model, "converged", True))
        if not converged:
            console.print(f"[yellow]WARNING: {title} did not converge; treat its coefficients as unreliable.[/yellow]")
        save_model_summary(model, str(output_dir / f"{slug}.txt"))
        table = coefficient_table(model, title)
        table["converged"] = converged
        coef_tables.append(table)

    combined = pd.concat(coef_tables, ignore_index=True)
    combined.to_csv(output_dir / "tableS3_coefficients.csv", index=False)
    console.print(f"\n[green]Saved Table S3 summaries to {output_dir}[/green]")
    console.print("\n[bold]Finished![/bold]")


if __name__ == "__main__":
    seed_everything()
    app()
