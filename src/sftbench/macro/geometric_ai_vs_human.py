from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import seaborn as sns
import typer
from statannotations.Annotator import Annotator

from sftbench import find_project_root
from sftbench.embeddings.embeddings import text_to_embedding
from sftbench.embeddings.utils import find_closest_words
from sftbench.figure_output import apply_bold_axis_style, save_figure_formats
from sftbench.micro.prediction.plotting.config import apply_seaborn_theme_from_config
from sftbench.reproducibility import seed_everything

HUMAN_COLOR = "#4C72B0"
AI_COLOR = "#DD8452"


def _repo_path(*parts: str) -> Path:
    return (find_project_root() or Path.cwd()).joinpath(*parts)


CONCEPTNET_PATH = _repo_path("data", "embeddings", "conceptnet")
FASTTEXT_PATH = _repo_path("data", "embeddings", "cc.en.300.vec")

app = typer.Typer(add_completion=False)


class EmbeddingType(str, Enum):
    """Enum for the available embedding model types."""

    conceptnet = "conceptnet"
    fasttext = "fasttext"


def load_conceptnet_model(filepath: Path) -> dict[str, np.ndarray]:
    print(f"Loading ConceptNet model from {filepath}...")
    model: dict[str, np.ndarray] = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]], dtype=float)
            model[word] = vector
    print("ConceptNet model loaded successfully.")
    return model


def load_fasttext_model(filepath: Path) -> dict[str, np.ndarray]:
    print(f"Loading FastText model from {filepath}...")
    model: dict[str, np.ndarray] = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]], dtype=float)
            model[word] = vector
    print("FastText model loaded successfully.")
    return model


def calculate_semantic_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Takes in N word embeddings and returns a semantic similarity matrix (NxN np.array)
    using cosine similarity (1 - cosine distance). Non-positive similarities are floored.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape (N, D), got {embeddings.shape}")

    n = len(embeddings)
    semantic_matrix = 1 - scipy.spatial.distance.cdist(embeddings, embeddings, "cosine").reshape(-1)
    semantic_matrix = semantic_matrix.reshape((n, n))
    semantic_matrix[semantic_matrix <= 0] = 0.0001
    return semantic_matrix


def extract_spectral_properties(matrix: np.ndarray) -> dict[str, float]:
    """
    Extract spectral gap and participation ratio for a single similarity matrix.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    eigenvalues = np.linalg.eigh(matrix)[0]
    eigenvalues = np.sort(eigenvalues)[::-1]

    spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0.0

    eigenvalues_sq = eigenvalues**2
    participation_ratio = (np.sum(eigenvalues)) ** 2 / np.sum(eigenvalues_sq) if np.sum(eigenvalues_sq) > 0 else 0.0

    return {"spectral_gap": float(spectral_gap), "participation_ratio": float(participation_ratio)}


def calculate_paired_curvature(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(a.T @ b)
    dot = max(-1.0, min(1.0, dot))
    return float(np.arccos(dot))


def calculate_layer_average_k_curvature(embeddings: np.ndarray, k: int = 1) -> float:
    summation, counter = 0.0, 0
    while (counter + k) < embeddings.shape[0]:
        summation += calculate_paired_curvature(embeddings[counter, :], embeddings[counter + k, :])
        counter += 1
    if counter == 0:
        return float("nan")
    return float(summation / counter)


def _normalize_token_series(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    return s.str.replace(" ", "", regex=False).str.lower()


def _load_embedding_model(
    embedding_type: EmbeddingType,
    embedding_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    if embedding_type == EmbeddingType.conceptnet:
        model = load_conceptnet_model(embedding_path or CONCEPTNET_PATH)
    else:
        model = load_fasttext_model(embedding_path or FASTTEXT_PATH)
    embedding_dim = 300
    return model, embedding_dim


def _map_to_closest_vocab(df: pd.DataFrame, col: str, model: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Map each unique token to the closest available embedding vocabulary item.
    """
    df = df.copy()
    unique_words = df[col].dropna().unique().tolist()
    embedding_words = list(model)
    closest_words = find_closest_words(unique_words, embedding_words)

    changed: list[tuple[str, str]] = []
    for idx, original_word in df[col].items():
        if not isinstance(original_word, str):
            continue
        mapped_word = closest_words.get(original_word, original_word)
        df.at[idx, col] = mapped_word
        if original_word != mapped_word:
            changed.append((original_word, mapped_word))

    if changed:
        print(f"Mapped {len(changed)} tokens in column '{col}' to closest embedding vocab matches.")
    else:
        print(f"No tokens in column '{col}' required mapping.")
    return df


def _compute_embeddings_by_id(
    df: pd.DataFrame,
    col: str,
    model: dict[str, np.ndarray],
    embedding_dim: int,
    id_col: str = "id",
) -> dict[Any, np.ndarray]:
    """
    Compute embeddings grouped by id, returning dict[id] -> (N_i, D).
    """
    out: dict[Any, np.ndarray] = {}
    for seq_id, group in df.groupby(id_col):
        embeddings: list[np.ndarray] = []
        for text in group[col].tolist():
            emb = text_to_embedding(text, model, embedding_dim)
            embeddings.append(emb)
        out[seq_id] = np.vstack(embeddings)
    return out


def _compute_similarity_matrices_by_id(
    df: pd.DataFrame,
    col: str,
    model: dict[str, np.ndarray],
    embedding_dim: int,
    id_col: str = "id",
) -> dict[Any, np.ndarray]:
    out: dict[Any, np.ndarray] = {}
    for seq_id, group in df.groupby(id_col):
        embeddings = [text_to_embedding(text, model, embedding_dim) for text in group[col].tolist()]
        emb_arr = np.vstack(embeddings)
        out[seq_id] = calculate_semantic_matrix(emb_arr)
    return out


def characterize_and_compare_collections(
    human_matrices: list[np.ndarray],
    machine_matrices: list[np.ndarray],
    df_curvature: pd.DataFrame,
    *,
    save_path: Path | None = None,
    stats_path: Path | None = None,
    formats: str | None = None,
    show: bool = True,
    compact_plot: bool = False,
    ultra_compact_plot: bool = False,
) -> None:
    """
    Build the semantic geometry plot
    """
    human_properties: list[dict[str, float]] = []
    for m in human_matrices:
        try:
            human_properties.append(extract_spectral_properties(m))
        except Exception as e:
            print(f"Error processing human matrix: {e}")
            continue

    machine_properties: list[dict[str, float]] = []
    for m in machine_matrices:
        try:
            machine_properties.append(extract_spectral_properties(m))
        except Exception as e:
            print(f"Error processing machine matrix: {e}")
            continue

    human_gaps = [p["spectral_gap"] for p in human_properties]
    machine_gaps = [p["spectral_gap"] for p in machine_properties]
    human_pr = [p["participation_ratio"] for p in human_properties]
    machine_pr = [p["participation_ratio"] for p in machine_properties]

    gap_data = pd.DataFrame(
        {"Group": ["Human"] * len(human_gaps) + ["AI"] * len(machine_gaps), "Value": human_gaps + machine_gaps}
    )
    pr_data = pd.DataFrame(
        {"Group": ["Human"] * len(human_pr) + ["AI"] * len(machine_pr), "Value": human_pr + machine_pr}
    )

    if ultra_compact_plot:
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(3.4, 3.4))
        ax2 = None
        box_width = 0.26
        box_linewidth = 0.8
        title_fs = 12
        label_fs = 10
        tick_fs = 12
    elif not compact_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
        box_width = 0.8
        box_linewidth = 1.0
        title_fs = 16
        label_fs = 12
        tick_fs = 12
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.2, 3.6))
        box_width = 0.28
        box_linewidth = 0.8
        title_fs = 13
        label_fs = 11
        tick_fs = 10

    def _style_axis(ax: Any) -> None:
        ax.yaxis.grid(True, color="0.88", linewidth=0.8)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=tick_fs, pad=1)
        ax.tick_params(axis="y", labelsize=tick_fs)

    sns.boxplot(
        data=gap_data,
        x="Group",
        y="Value",
        hue="Group",
        ax=ax1,
        palette={"Human": HUMAN_COLOR, "AI": AI_COLOR},
        legend=False,
        width=box_width,
        linewidth=box_linewidth,
        showfliers=not (compact_plot or ultra_compact_plot),
    )
    ax1.set_title("Spectral Gap", fontsize=title_fs)
    ax1.set_xlabel("")
    ax1.set_ylabel("Gap (λ1 - λ2)", fontsize=label_fs)
    _style_axis(ax1)

    if ax2 is not None:
        sns.boxplot(
            data=pr_data,
            x="Group",
            y="Value",
            hue="Group",
            ax=ax2,
            palette={"Human": HUMAN_COLOR, "AI": AI_COLOR},
            legend=False,
            width=box_width,
            linewidth=box_linewidth,
            showfliers=not (compact_plot or ultra_compact_plot),
        )
        ax2.set_title("Participation Ratio", fontsize=title_fs)
        ax2.set_xlabel("")
        ax2.set_ylabel("Effective # Dimensions", fontsize=label_fs)
        _style_axis(ax2)

    df_curvature_plot = df_curvature.copy()
    df_curvature_plot["type"] = df_curvature_plot["type"].replace({"human": "Human", "machine": "AI"})
    sns.boxplot(
        data=df_curvature_plot,
        x="type",
        y="curvature",
        hue="type",
        ax=ax3,
        palette={"Human": HUMAN_COLOR, "AI": AI_COLOR},
        legend=False,
        width=box_width,
        linewidth=box_linewidth,
        showfliers=not (compact_plot or ultra_compact_plot),
    )
    ax3.set_title("Curvature", fontsize=title_fs)
    ax3.set_xlabel("")
    ax3.set_ylabel("Curvature", fontsize=label_fs)
    _style_axis(ax3)

    fig.suptitle(
        "Geometric Properties",
        fontsize=title_fs,
    )

    try:
        pairs_gap = [("Human", "AI")]
        annotator_gap = Annotator(ax1, pairs_gap, data=gap_data, x="Group", y="Value")
        annotator_gap.configure(test="Wilcoxon", text_format="star", loc="inside")
        annotator_gap.apply_and_annotate()

        if ax2 is not None:
            pairs_pr = [("Human", "AI")]
            annotator_pr = Annotator(ax2, pairs_pr, data=pr_data, x="Group", y="Value")
            annotator_pr.configure(test="Mann-Whitney", text_format="star", loc="inside")
            annotator_pr.apply_and_annotate()
    except Exception as e:
        print(f"Statannotation skipped (missing dependency or error): {e}")

    try:
        pairs_curvature = [("Human", "AI")]
        annotator_curvature = Annotator(ax3, pairs_curvature, data=df_curvature_plot, x="type", y="curvature")
        annotator_curvature.configure(test="Wilcoxon", text_format="star", loc="inside")
        annotator_curvature.apply_and_annotate()
    except Exception as e:
        print(f"Curvature statannotation skipped (missing dependency or error): {e}")

    if compact_plot or ultra_compact_plot:
        fig.set_constrained_layout(True)
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

        axes_to_adjust = [ax1, ax3] if ultra_compact_plot else [ax1, ax2, ax3]
        for ax in axes_to_adjust:
            ax.margins(x=0.10)
    else:
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    def _mean_and_bootstrap_ci95(
        values: list[float] | np.ndarray,
        *,
        n_boot: int = 10_000,
        seed: int = 0,
    ) -> tuple[float, float, float]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan"), float("nan"), float("nan")

        rng = np.random.default_rng(seed)
        boot_means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sample = rng.choice(arr, size=arr.size, replace=True)
            boot_means[i] = float(np.mean(sample))

        mean = float(np.mean(arr))
        lo, hi = np.quantile(boot_means, [0.025, 0.975])
        return mean, float(lo), float(hi)

    summary_rows: list[dict[str, Any]] = []

    def _record(metric: str, group: str, values: list[float] | np.ndarray, seed: int) -> tuple[float, float, float]:
        mean, lo, hi = _mean_and_bootstrap_ci95(values, seed=seed)
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        summary_rows.append(
            {
                "metric": metric,
                "group": group,
                "n": int(arr.size),
                "mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "std": float(np.std(arr)) if arr.size else float("nan"),
            }
        )
        return mean, lo, hi

    print("\n--- Summary Statistics (Mean ± 95% bootstrap CI) ---")
    if human_gaps and machine_gaps:
        h_mean, h_lo, h_hi = _record("spectral_gap", "Human", human_gaps, seed=0)
        a_mean, a_lo, a_hi = _record("spectral_gap", "AI", machine_gaps, seed=1)
        print(
            f"Human Spectral Gap:      Mean={h_mean:.3f}  95% CI=[{h_lo:.3f}, {h_hi:.3f}]  Std={np.std(human_gaps):.3f}"
        )
        print(
            f"AI Spectral Gap:         Mean={a_mean:.3f}  95% CI=[{a_lo:.3f}, {a_hi:.3f}]  Std={np.std(machine_gaps):.3f}"
        )
    print("-" * 30)
    if human_pr and machine_pr:
        h_mean, h_lo, h_hi = _record("participation_ratio", "Human", human_pr, seed=2)
        a_mean, a_lo, a_hi = _record("participation_ratio", "AI", machine_pr, seed=3)
        print(
            f"Human Participation Ratio: Mean={h_mean:.3f}  95% CI=[{h_lo:.3f}, {h_hi:.3f}]  Std={np.std(human_pr):.3f}"
        )
        print(
            f"AI Participation Ratio:    Mean={a_mean:.3f}  95% CI=[{a_lo:.3f}, {a_hi:.3f}]  Std={np.std(machine_pr):.3f}"
        )

    print("\n--- Curvature Summary Statistics (Mean ± 95% bootstrap CI) ---")
    human_curv = df_curvature.loc[df_curvature["type"] == "human", "curvature"].dropna().to_numpy(dtype=float)
    machine_curv = df_curvature.loc[df_curvature["type"] == "machine", "curvature"].dropna().to_numpy(dtype=float)
    if human_curv.size > 0:
        h_mean, h_lo, h_hi = _record("curvature", "Human", human_curv, seed=4)
        print(f"Human Curvature:   Mean={h_mean:.3f}  95% CI=[{h_lo:.3f}, {h_hi:.3f}]  Std={np.std(human_curv):.3f}")
    if machine_curv.size > 0:
        a_mean, a_lo, a_hi = _record("curvature", "AI", machine_curv, seed=5)
        print(f"AI Curvature:      Mean={a_mean:.3f}  95% CI=[{a_lo:.3f}, {a_hi:.3f}]  Std={np.std(machine_curv):.3f}")

    print("\n--- Paired Wilcoxon signed-rank tests (two-sided) ---")
    test_rows: list[dict[str, Any]] = []
    paired_tests: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("spectral_gap", np.asarray(human_gaps, dtype=float), np.asarray(machine_gaps, dtype=float)),
        ("participation_ratio", np.asarray(human_pr, dtype=float), np.asarray(machine_pr, dtype=float)),
        ("curvature", human_curv, machine_curv),
    ]
    for metric, human_values, ai_values in paired_tests:
        if human_values.size == 0 or human_values.size != ai_values.size:
            continue
        result = scipy.stats.wilcoxon(human_values, ai_values, alternative="two-sided")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        test_rows.append(
            {
                "metric": metric,
                "test": "wilcoxon_signed_rank_two_sided",
                "n_pairs": int(human_values.size),
                "statistic": statistic,
                "p_value": p_value,
            }
        )
        print(f"{metric}: Human vs AI  W={statistic:.6g}  p={p_value:.6g}  (n={human_values.size} pairs)")

    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(stats_path, index=False)
        tests_path = stats_path.with_name(f"{stats_path.stem}_tests.csv")
        pd.DataFrame(test_rows).to_csv(tests_path, index=False)
        print(f"\nSummary statistics saved to {stats_path}")
        print(f"Paired tests saved to {tests_path}")

    if save_path is not None:
        saved_paths = save_figure_formats(
            fig,
            save_path,
            formats=formats,
            default=(save_path.suffix.lstrip(".") or "png",),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Plot saved to {', '.join(str(path) for path in saved_paths)}")

    if show:
        plt.show()
    else:
        plt.close(fig)


@app.command()
def main(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to the embedding-switch CSV (paired AI+human words).",
    ),
    embedding_type: EmbeddingType = typer.Option(
        EmbeddingType.conceptnet,
        "--embedding-type",
        "-e",
        help="Embedding model type.",
        case_sensitive=False,
    ),
    embedding_path: Path | None = typer.Option(
        None,
        "--embedding-path",
        help="Optional explicit embedding file path. Defaults to data/embeddings/conceptnet in the release bundle.",
    ),
    ai_col: str = typer.Option(
        "response",
        "--ai-col",
        help="Column containing AI responses (words). Default: response",
    ),
    human_col: str = typer.Option(
        "human_response",
        "--human-col",
        help="Column containing human responses (words). Default: human_response",
    ),
    id_col: str = typer.Option(
        "id",
        "--id-col",
        help="Column containing sequence id. Default: id",
    ),
    order_col: str = typer.Option(
        "rank",
        "--order-col",
        help=(
            "Column giving each word's position within its sequence. Curvature is computed between "
            "consecutive words, so rows are sorted by this column within each id. Ignored if absent."
        ),
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to save the resulting figure (PNG).",
    ),
    compact_plot: bool = typer.Option(
        False,
        "--compact-plot",
        help="Render a compact plot (three tightly packed panels with independent y-axes).",
    ),
    ultra_compact: bool = typer.Option(
        False,
        "--ultra-compact",
        help="Render an ultra-compact plot with only Spectral Gap and Curvature panels.",
    ),
    plotting_config: Path | None = typer.Option(
        None,
        "--plotting-config",
        help="Path to plotting configuration TOML. If omitted, uses configs/plotting.toml from project root when available.",
    ),
    no_show: bool = typer.Option(
        False,
        "--no-show",
        help="Do not display the plot window (still saves if --output is provided).",
    ),
    formats: str | None = typer.Option(
        None,
        "--formats",
        help="Comma-separated output formats for figures: png, pdf, svg. Defaults preserve --output suffix.",
    ),
):
    project_root = find_project_root()
    if project_root is None:
        raise typer.BadParameter("Could not determine project root (no .git marker found).")

    config_path = plotting_config if plotting_config is not None else (project_root / "configs" / "plotting.toml")
    apply_seaborn_theme_from_config(config_path)
    apply_bold_axis_style()

    input_path = input
    if not input_path.is_absolute():
        input_path = project_root / input_path

    if not input_path.exists():
        raise typer.BadParameter(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    for required in [id_col, ai_col, human_col]:
        if required not in df.columns:
            raise typer.BadParameter(
                f"Missing required column '{required}'. Available columns: {', '.join(df.columns)}"
            )

    if order_col in df.columns:
        df = df.sort_values([id_col, order_col], kind="mergesort").reset_index(drop=True)
    else:
        print(f"Order column '{order_col}' not present; keeping input row order within each sequence.")

    df[ai_col] = _normalize_token_series(df[ai_col])  # type: ignore[arg-type]
    df[human_col] = _normalize_token_series(df[human_col])  # type: ignore[arg-type]

    # Drop NaNs for each stream separately, to avoid spurious empty tokens.
    df_ai = df[[id_col, ai_col]].dropna(subset=[ai_col]).copy()
    df_human = df[[id_col, human_col]].dropna(subset=[human_col]).copy()

    # Load embedding model once, then vocab-map both streams
    if embedding_path is not None and not embedding_path.is_absolute():
        embedding_path = project_root / embedding_path

    model, embedding_dim = _load_embedding_model(embedding_type, embedding_path)
    print(f"Detected embedding dimension: {embedding_dim}")

    df_ai = _map_to_closest_vocab(df_ai, ai_col, model)
    df_human = _map_to_closest_vocab(df_human, human_col, model)

    semantic_matrices_human = _compute_similarity_matrices_by_id(
        df_human, human_col, model, embedding_dim, id_col=id_col
    )
    semantic_matrices_ai = _compute_similarity_matrices_by_id(df_ai, ai_col, model, embedding_dim, id_col=id_col)

    paired_ids = sorted(set(semantic_matrices_human) & set(semantic_matrices_ai))
    if not paired_ids:
        raise typer.BadParameter("No sequence IDs are shared by the human and AI streams.")
    print(f"Paired human/AI sequence IDs: {len(paired_ids)}")

    human_matrices_collection = [semantic_matrices_human[seq_id] for seq_id in paired_ids]
    llm_matrices_collection = [semantic_matrices_ai[seq_id] for seq_id in paired_ids]

    embeddings_human = _compute_embeddings_by_id(df_human, human_col, model, embedding_dim, id_col=id_col)
    embeddings_ai = _compute_embeddings_by_id(df_ai, ai_col, model, embedding_dim, id_col=id_col)

    curvature_results: list[dict[str, Any]] = []

    for seq_id in paired_ids:
        emb = embeddings_human[seq_id]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_norm = emb / norms
        curvature = calculate_layer_average_k_curvature(emb_norm, k=1)
        curvature_results.append({"id": seq_id, "type": "human", "curvature": curvature})

    for seq_id in paired_ids:
        emb = embeddings_ai[seq_id]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_norm = emb / norms
        curvature = calculate_layer_average_k_curvature(emb_norm, k=1)
        curvature_results.append({"id": seq_id, "type": "machine", "curvature": curvature})

    df_curvature = pd.DataFrame(curvature_results)

    save_path = None
    stats_path = None
    if output is not None:
        save_path = output if output.is_absolute() else (project_root / output)
        stats_path = save_path.with_name(f"{save_path.stem}_summary_stats.csv")

    characterize_and_compare_collections(
        human_matrices_collection,
        llm_matrices_collection,
        df_curvature,
        save_path=save_path,
        stats_path=stats_path,
        formats=formats,
        show=not no_show,
        compact_plot=compact_plot,
        ultra_compact_plot=ultra_compact,
    )


if __name__ == "__main__":
    seed_everything()
    app()
