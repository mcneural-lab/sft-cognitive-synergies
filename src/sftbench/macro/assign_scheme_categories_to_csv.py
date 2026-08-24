# ruff: noqa: B008
from pathlib import Path

import pandas as pd
import typer

from sftbench.reproducibility import seed_everything

app = typer.Typer()


def assign_categories(
    df_cat: pd.DataFrame,
    df: pd.DataFrame,
    response_column_name: str = "response",
    category_column_name: str = "category",
    switch_column_name: str = "switch",
    scheme_key_column: str = "animal",
    scheme_value_column: str = "category",
    id_column_name: str = "id",
) -> pd.DataFrame:
    """
    Maps animal categories to responses and calculates a 'switch' column.

    This function takes a DataFrame of animal-to-category mappings and a main DataFrame
    containing responses. It first maps the categories to the 'response' column,
    creating a new 'category' column. Then, it calculates a 'switch' column that
    indicates whether the category of the current response is different from the
    category of the previous response within the same 'id' group.

    Parameters
    ----------
    df_cat : pd.DataFrame
        A DataFrame containing the mapping scheme (e.g., 'animal' and 'category' columns).
    df : pd.DataFrame
        The main DataFrame containing responses.
    response_column_name : str
        The name of the column in `df` containing the response text.
    category_column_name : str
        The name of the new column to be created in `df` for the mapped categories.
    switch_column_name : str
        The name of the new column to be created in `df` for the switch indicator.
    scheme_key_column : str
        The column name in `df_cat` that matches the values in `response_column_name`.
    scheme_value_column : str
        The column name in `df_cat` that contains the category values.
    id_column_name : str
        The column name in `df` to group by for sequence calculation.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with 'category' and 'switch' columns added.
    """
    animal_categories = df_cat.groupby(scheme_key_column)[scheme_value_column].apply(list).to_dict()

    df[category_column_name] = (
        df[response_column_name].astype(str).str.replace(" ", "").str.lower().map(animal_categories)
    )
    overlap_exists = df.groupby(id_column_name)[category_column_name].transform(
        lambda x: [
            any(item in prev_list for item in current_list)
            if isinstance(prev_list, list) and isinstance(current_list, list)
            else False
            for current_list, prev_list in zip(x, x.shift(1), strict=False)
        ]
    )
    df[switch_column_name] = ~overlap_exists

    return df


@app.command()
def main(
    input_csv: Path = typer.Argument(..., help="Path to the input CSV file containing sequences."),
    scheme_csv: Path = typer.Argument(..., help="Path to the SNAFU scheme CSV file (e.g. animals_snafu_scheme.csv)."),
    output_csv: Path | None = typer.Option(
        None,
        help="Path to save the processed CSV. Defaults to input filename + '_processed.csv'.",
    ),
    response_col: str = typer.Option("response", help="Column name in input CSV containing the response/word."),
    id_col: str = typer.Option("id", help="Column name in input CSV identifying the sequence/participant."),
    scheme_key_col: str = typer.Option("animal", help="Column name in scheme CSV matching the response word."),
    scheme_val_col: str = typer.Option("category", help="Column name in scheme CSV containing the category."),
    category_col: str = typer.Option(
        "category",
        help="Column name to create in the output CSV for categories (from the norm/scheme data).",
    ),
    switch_col: str = typer.Option("switch", help="Column name to use for the computed switch indicator."),
):
    """
    Assigns categories from a scheme CSV to an input CSV and calculates category switches.
    """
    if not input_csv.exists():
        typer.echo(f"Error: Input file '{input_csv}' not found.", err=True)
        raise typer.Exit(code=1)
    if not scheme_csv.exists():
        typer.echo(f"Error: Scheme file '{scheme_csv}' not found.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    df = df[df[response_col] != "<START>"]

    typer.echo(f"Loading scheme CSV: {scheme_csv}")
    df_cat = pd.read_csv(scheme_csv)

    if response_col not in df.columns:
        typer.echo(f"Error: Response column '{response_col}' not found in input CSV.", err=True)
        raise typer.Exit(code=1)
    if id_col not in df.columns:
        typer.echo(f"Error: ID column '{id_col}' not found in input CSV.", err=True)
        raise typer.Exit(code=1)
    if scheme_key_col not in df_cat.columns:
        typer.echo(f"Error: Key column '{scheme_key_col}' not found in scheme CSV.", err=True)
        raise typer.Exit(code=1)
    if scheme_val_col not in df_cat.columns:
        typer.echo(f"Error: Value column '{scheme_val_col}' not found in scheme CSV.", err=True)
        raise typer.Exit(code=1)

    typer.echo("Processing categories and switches...")
    df_processed = assign_categories(
        df_cat=df_cat,
        df=df,
        response_column_name=response_col,
        category_column_name=category_col,
        switch_column_name=switch_col,
        scheme_key_column=scheme_key_col,
        scheme_value_column=scheme_val_col,
        id_column_name=id_col,
    )

    if category_col not in df_processed.columns:
        typer.echo(f"Error: Category column '{category_col}' was not created.", err=True)
        raise typer.Exit(code=1)
    n_mapped = df_processed[category_col].notnull().sum()
    n_total = len(df_processed)
    typer.echo(f"Mapped {n_mapped}/{n_total} items to categories.")

    if output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_processed{input_csv.suffix}")

    typer.echo(f"Saving output to: {output_csv}")
    df_processed.to_csv(output_csv, index=False)
    typer.echo("Done.")


if __name__ == "__main__":
    seed_everything()
    app()
