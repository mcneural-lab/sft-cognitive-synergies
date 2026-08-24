import re
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import typer
from rich.console import Console
from sklearn.metrics import classification_report

from sftbench import find_project_root
from sftbench.embeddings.switch import switch_delta, switch_median, switch_simdrop
from sftbench.embeddings.utils import apply_dict_mapping_to_df, find_closest_words
from sftbench.reproducibility import seed_everything, track

app = typer.Typer()
console = Console()


def _repo_path(*parts: str) -> Path:
    return (find_project_root() or Path.cwd()).joinpath(*parts)


GLOVE_PATH = _repo_path("data", "glove", "glove.6B.300d.txt")
CONCEPTNET_PATH = _repo_path("data", "embeddings", "conceptnet")
FASTTEXT_PATH = _repo_path("data", "embeddings", "cc.en.300.vec")


def calculate_semantic_matrix(embeddings: np.ndarray):
    """
    Description:
        Takes in N word embeddings and returns a semantic similarity matrix (NxN np.array)
    Args:
        (1) embeddings (numpy.ndarray): NxD
    Returns:
        (1) semantic_matrix: semantic similarity matrix (NxN np.array)
    """
    N = len(embeddings)
    semantic_matrix = 1 - scipy.spatial.distance.cdist(embeddings, embeddings, "cosine").reshape(-1)
    semantic_matrix = semantic_matrix.reshape((N, N))
    semantic_matrix[semantic_matrix <= 0] = 0.0001
    return semantic_matrix


def calculate_switches(
    sequence: list[str],
    semantic_matrix: np.ndarray,
    method: str = "simdrop",
    rise_threshold: float = 0.5,
    fall_threshold: float = 0.5,
):
    """
    Description:
        Takes in a sequence of words and a semantic similarity matrix and returns a list of switches
    Args:
        (1) sequence (list[str]): list of words
        (2) semantic_matrix (np.ndarray): semantic similarity matrix (NxN np.array)
        (3) method (str): method to calculate switches
        (4) rise_threshold (float): threshold for considering a switch as a rise
        (5) fall_threshold (float): threshold for considering a switch as a fall
    Returns:
        (1) switches: list of switches
    """
    sim_list = []
    sim_history = []
    for i in range(0, len(sequence)):
        word = sequence[i]
        currentwordindex = sequence.index(word)
        if i > 0:  # get similarity between this word and preceding word
            prevwordindex = sequence.index(sequence[i - 1])
            sim_list.append(semantic_matrix[prevwordindex, currentwordindex])
            sim_history.append(semantic_matrix[prevwordindex, :])
        else:  # first word
            sim_list.append(0.0001)
            sim_history.append(semantic_matrix[currentwordindex, :])

    if method == "simdrop":
        switches = switch_simdrop(sequence, sim_list)
    elif method == "delta":
        switches = switch_delta(sequence, sim_list, rise_threshold, fall_threshold)
    elif method == "median":
        switches = switch_median(sequence, sim_list, z_score=False)
    else:
        raise ValueError(f"Invalid method: {method}")
    return switches


def display_report(report):
    from rich.table import Table

    # Create a rich Table for the classification report
    report_table = Table(title="Classification Report", show_header=True, header_style="bold magenta")
    report_table.add_column("Class", style="dim", width=12)
    report_table.add_column("Precision", justify="right")
    report_table.add_column("Recall", justify="right")
    report_table.add_column("F1-Score", justify="right")
    report_table.add_column("Support", justify="right")

    # Add metrics for each class (e.g., '0' and '1')
    for label, metrics in report.items():
        if isinstance(metrics, dict) and label not in ["macro avg", "weighted avg"]:
            precision = f"{metrics['precision']:.2f}"
            recall = f"{metrics['recall']:.2f}"
            f1_score = f"{metrics['f1-score']:.2f}"
            support = f"{metrics['support']}"
            report_table.add_row(str(label), precision, recall, f1_score, support)

    # Add a separator before overall metrics
    report_table.add_section()

    # Add accuracy if present, usually displayed uniquely
    if "accuracy" in report:
        accuracy = f"{report['accuracy']:.2f}"
        report_table.add_row("[bold green]Accuracy[/bold green]", "", "", "", accuracy, style="bold")

    # Add another separator before averages
    report_table.add_section()

    # Add macro and weighted averages
    for label in ["macro avg", "weighted avg"]:
        if label in report and isinstance(report[label], dict):
            metrics = report[label]
            precision = f"{metrics['precision']:.2f}"
            recall = f"{metrics['recall']:.2f}"
            f1_score = f"{metrics['f1-score']:.2f}"
            support = f"{metrics['support']}"
            report_table.add_row(f"[bold blue]{label}[/bold blue]", precision, recall, f1_score, support)

    # Print the nicely formatted table
    console.print(report_table)


class EmbeddingType(str, Enum):
    """Enum for the available embedding model types."""

    glove = "glove"
    conceptnet = "conceptnet"
    fasttext = "fasttext"


def load_glove_model(glove_file: Path) -> dict[str, np.ndarray]:
    """Loads the GloVe model from a text file into a dictionary."""
    if not glove_file.exists():
        console.print(f"[bold red]Error: GloVe file not found at {glove_file}[/bold red]")
        console.print("Please download it from https://nlp.stanford.edu/projects/glove/")
        raise typer.Exit(code=1)

    console.print(f"Loading GloVe model from [cyan]{glove_file}[/cyan]...")
    model = {}
    with open(glove_file, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]])
            model[word] = vector
    console.print("[bold green]GloVe model loaded successfully.[/bold green]")
    return model


def load_conceptnet_model(filepath: Path) -> dict[str, np.ndarray]:
    console.print(f"Loading ConceptNet model from [cyan]{filepath}[/cyan]...")
    model = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]])
            model[word] = vector
    console.print("[bold green]ConceptNet model loaded successfully.[/bold green]")
    return model


def load_fasttext_model(filepath: Path) -> dict[str, np.ndarray]:
    console.print(f"Loading FastText model from [cyan]{filepath}[/cyan]...")
    model = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]])
            model[word] = vector
    console.print("[bold green]FastText model loaded successfully.[/bold green]")
    return model


def text_to_embedding(text: str, model: dict[str, np.ndarray], embedding_dim: int) -> np.ndarray:
    """
    Calculates the average embedding for a given text.

    1. Cleans and tokenizes the text.
    2. Looks up each token in the model.
    3. Averages the vectors of the words found.
    """
    if not isinstance(text, str):
        return np.zeros(embedding_dim)

    # Clean and tokenize the text: lowercase, alphanumeric words only
    words = re.findall(r"\b\w+\b", text.lower())

    word_vectors = []
    for word in words:
        if word in model:
            word_vectors.append(model[word])

    if not word_vectors:
        # If no words are found in the model, return a zero vector
        console.print(f"[bold red]No words found in the model for text '{text}'. Returning zero vector.[/bold red]")
        return np.zeros(embedding_dim)

    # Return the mean of the word vectors
    return np.mean(word_vectors, axis=0)


@app.command()
def process_csv(
    input_path: Path = typer.Argument(
        ..., help="Path to the input CSV file. Must contain 'id' and the selected text column."
    ),
    output_path: Path = typer.Argument(..., help="Path to save the output CSV file with embeddings."),
    embedding_type: EmbeddingType = typer.Option(
        EmbeddingType.glove,
        "--embedding-type",
        "-e",
        help="The type of embedding model to use.",
        case_sensitive=False,
    ),
    embedding_path: Path | None = typer.Option(
        None,
        "--embedding-path",
        help="Optional explicit embedding file path. Defaults to data/embeddings/conceptnet in the release bundle.",
    ),
    text_column: str = typer.Option(
        "response",
        "--text-column",
        "-c",
        help="Name of the column to compute embeddings/switches over (default: 'response').",
    ),
    switch_gt_column: str = typer.Option(
        "switch",
        "--switch-gt-column",
        help="Name of the ground-truth switch column used for the classification report (default: 'switch').",
    ),
    switch_method: str = typer.Option(
        "simdrop",
        "-s",
        "--switch-method",
        help="Switch detection method: 'simdrop', 'delta', or 'median'.",
    ),
    rise_threshold: float = typer.Option(
        0.5,
        "-r",
        "--rise-threshold",
        help="The threshold for considering a switch as a rise.",
    ),
    fall_threshold: float = typer.Option(
        0.5,
        "-f",
        "--fall-threshold",
        help="The threshold for considering a switch as a fall.",
    ),
):
    """
    Calculates embeddings for text in a CSV file and saves the result.
    """
    # --- 1. Check Input File ---
    if not input_path.exists():
        console.print(f"[bold red]Error: Input file not found at {input_path}[/bold red]")
        raise typer.Exit(code=1)

    # --- 2. Load Model ---
    console.print(f"Using [bold yellow]{embedding_type.value}[/bold yellow] embedding model.")
    if embedding_type == EmbeddingType.glove:
        glove_path = embedding_path or GLOVE_PATH
        model = load_glove_model(glove_path)
        embedding_dim = 300
    elif embedding_type == EmbeddingType.conceptnet:
        conceptnet_path = embedding_path or CONCEPTNET_PATH
        model = load_conceptnet_model(conceptnet_path)
        embedding_dim = 300
    elif embedding_type == EmbeddingType.fasttext:
        fasttext_path = embedding_path or FASTTEXT_PATH
        model = load_fasttext_model(fasttext_path)
        embedding_dim = 300
    else:
        # This is where logic for other embedding types would go.
        console.print(f"[bold red]Error: Embedding type '{embedding_type.value}' is not yet supported.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"Detected embedding dimension: [bold yellow]{embedding_dim}[/bold yellow]")

    # --- 3. Read and Process CSV ---
    console.print(f"\nReading input CSV from [cyan]{input_path}[/cyan]...")
    df_in = pd.read_csv(input_path)
    df = df_in.copy(deep=True)

    if "id" not in df.columns:
        console.print(f"[bold red]Error: 'id' column not found in {input_path}[/bold red]")
        raise typer.Exit(code=1)

    if text_column not in df.columns:
        console.print(
            f"[bold red]Error: text column '{text_column}' not found in {input_path}[/bold red]\n"
            f"Available columns: {', '.join(df.columns)}"
        )
        raise typer.Exit(code=1)

    target_col = text_column
    unmodified_col = f"{target_col}_unmodified"
    df[unmodified_col] = df[text_column].copy()

    nan_responses = df[df[text_column].isna()]
    nan_count = len(nan_responses)
    console.print(f"\n[bold yellow]Found {nan_count} NaN values in '{text_column}':[/bold yellow]")
    if nan_count > 0:
        for idx, row in nan_responses.iterrows():
            console.print(f"  Row {idx}: {text_column}='{row[text_column]}', {unmodified_col}='{row[unmodified_col]}'")
    else:
        console.print("  No NaN values found.")

    # Work on a dedicated column for normalization/mapping (default: overwrite chosen column)
    df[target_col] = df[text_column]

    # remove spaces from target column
    df[target_col] = df[target_col].str.replace(" ", "", regex=False)
    # Drop rows with NaN values in target column
    df = df.dropna(subset=[target_col])
    # Apply dictionary mapping!
    df = apply_dict_mapping_to_df(df, target_col)

    unique_words = df[target_col].unique().tolist()
    embedding_words = list(model.keys())
    # Filter out NaN values from unique_words and embedding_words
    unique_words = [word for word in unique_words if pd.notna(word)]
    embedding_words = [word for word in embedding_words if pd.notna(word)]

    closest_words = find_closest_words(unique_words, embedding_words)
    # Store original responses and map to closest words

    # Map target column values to closest words and track changes
    changed_words = []
    for row_idx, original_word in zip(df.index, df[target_col].tolist(), strict=False):
        if isinstance(original_word, str) and " " not in original_word:
            mapped_word = closest_words.get(original_word, original_word)
            df.loc[row_idx, target_col] = mapped_word

            if original_word != mapped_word:
                print(f"'{row_idx} ----- '{original_word}' → '{mapped_word}'")
                changed_words.append((row_idx, original_word, mapped_word))

    # Print all words that changed
    if changed_words:
        console.print("\n[bold yellow]Words that were mapped to closest matches:[/bold yellow]")
        for _, original, mapped in changed_words:
            console.print(f" '{original}' → '{mapped}'")
    else:
        console.print("\n[bold green]No words needed to be mapped.[/bold green]")

    if target_col not in df.columns:
        console.print(f"[bold red]Error: '{target_col}' column not found in {input_path}[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"Calculating embeddings for each value in column '{target_col}'...")

    # Pre-create columns with stable dtypes to avoid pandas dtype warnings on assignment.
    # Use pandas' nullable BooleanDtype so missing values remain supported.
    if "embedding_switch_prediction" not in df.columns:
        df["embedding_switch_prediction"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    else:
        # If it already exists, coerce to nullable boolean to ensure compatibility.
        df["embedding_switch_prediction"] = df["embedding_switch_prediction"].astype("boolean")

    if "switch_embedding" not in df.columns:
        df["switch_embedding"] = pd.Series(pd.NA, index=df.index, dtype="string")

    # Use pandas apply with a progress bar from rich
    # Iterate through each id
    for seq_id, group in df.groupby("id"):
        typer.echo(f"Processing sequence ID {seq_id}")
        sequence = group[target_col].tolist()
        embeddings = [
            text_to_embedding(text, model, embedding_dim)
            for text in track(group[target_col], description="Processing...")
        ]
        embeddings = np.vstack(embeddings)
        semantic_matrix = calculate_semantic_matrix(embeddings)

        switches = calculate_switches(sequence, semantic_matrix, method=switch_method)
        switches_bool = pd.Series([s > 0 for s in switches], index=group.index, dtype="boolean")

        df.loc[group.index, "embedding_switch_prediction"] = switches_bool
        df.loc[group.index, "switch_embedding"] = switch_method

    if switch_gt_column not in df.columns:
        console.print(
            f"[bold red]Error: '{switch_gt_column}' ground-truth column not found; cannot compute classification report.[/bold red]"
        )
    elif "embedding_switch_prediction" not in df.columns:
        console.print(
            "[bold red]Error: 'embedding_switch_prediction' column not found; cannot compute classification report.[/bold red]"
        )
    else:
        paired = df[[switch_gt_column, "embedding_switch_prediction"]].copy()
        paired = paired.dropna(subset=[switch_gt_column, "embedding_switch_prediction"])

        y_true = paired[switch_gt_column].astype(bool).tolist()[1:]
        y_pred = paired["embedding_switch_prediction"].astype(bool).tolist()[1:]

        if len(y_true) == 0 or len(y_pred) == 0:
            console.print("[bold yellow]Not enough non-NaN rows to compute classification report.[/bold yellow]")
        else:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            display_report(report)

    console.print(f"\nSaving results to [cyan]{output_path}[/cyan]...")
    for col in df_in.columns:
        if col not in df.columns:
            df[col] = df_in[col]
    df.to_csv(output_path, index=False)

    console.print("[bold green]🎉 Success! Processing complete.[/bold green]")


if __name__ == "__main__":
    seed_everything()
    app()
