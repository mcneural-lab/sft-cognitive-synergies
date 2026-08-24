from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from scipy.stats import spearmanr

from sftbench import find_project_root
from sftbench.figure_output import apply_bold_axis_style, save_figure_formats
from sftbench.macro.sequence_macro_alignment import create_transition_matrices
from sftbench.micro.prediction.plotting import apply_seaborn_theme_from_config
from sftbench.reproducibility import SEED, seed_everything

app = typer.Typer(
    help=(
        "Compute micro (vocabulary) and macro (category) Transition Probability Matrices (TPMs) "
        "from a single sequences CSV and plot Human vs LLM matrices.\n\n"
    )
)


def get_unique_categories(series) -> list[str]:
    """
    Collect the unique set of 'items' from a series where each cell may be:
      - a plain string (single item)
      - a list of strings (multiple items)
      - a stringified list like "['A', 'B']"

    """
    categories: list[str] = []
    for category in series:
        if isinstance(category, str) and category.startswith("["):
            try:
                category = ast.literal_eval(category)
            except Exception:
                # Keep as-is; it will be treated as a single string below
                pass

        if isinstance(category, str):
            categories.append(category)
        elif isinstance(category, (list, tuple)):
            categories.extend(str(subcategory) for subcategory in category)
        elif category is not None:
            categories.append(str(category))
    return list(set(categories))


def get_avg_tpm_for_col(
    machine_data: pd.DataFrame,
    human_data: pd.DataFrame,
    col: str,
    *,
    fractional_weighting: bool,
    maximum_coherence: bool,
    include_self_transitions: bool,
    id_col: str,
    order_col: str,
    seq_col: str | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Compute average TPMs (across subjects/ids) for a given column,
    using a shared category/vocabulary set for human+machine.

    """
    machine_items = get_unique_categories(machine_data[col])
    human_items = get_unique_categories(human_data[col])
    vocab_items = sorted(set(machine_items + human_items))

    machine_transition_matrices = create_transition_matrices(
        machine_data,
        vocab_items,
        id_col=id_col,
        order_col=order_col,
        category_col=col,
        seq_col=seq_col,
        include_self_transitions=include_self_transitions,
        fractional_weighting=fractional_weighting,
        maximum_coherence=maximum_coherence,
    )
    if not machine_transition_matrices:
        return None, None
    machine_avg = sum(machine_transition_matrices.values()) / len(machine_transition_matrices)

    human_transition_matrices = create_transition_matrices(
        human_data,
        vocab_items,
        id_col=id_col,
        order_col=order_col,
        category_col=col,
        seq_col=seq_col,
        include_self_transitions=include_self_transitions,
        fractional_weighting=fractional_weighting,
        maximum_coherence=maximum_coherence,
    )
    if not human_transition_matrices:
        return None, None
    human_avg = sum(human_transition_matrices.values()) / len(human_transition_matrices)

    return machine_avg, human_avg


def _flatten_lower_triangle_including_diagonal(matrix: pd.DataFrame) -> np.ndarray:
    """
    Flatten the lower triangle (including diagonal) of a square matrix (as a DataFrame)
    in row-major order.

    This matches the "lower half + diagonal" extraction used for TPM correlation plots.
    """
    arr = matrix.to_numpy()
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape={arr.shape}")
    tri = np.tril_indices(arr.shape[0], k=0)
    return arr[tri]


def _mantel_test_spearman_from_matrices(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Mantel-style permutation test for Spearman correlation between two square matrices.

    Implementation detail (key for non-independence):
    - Compute observed Spearman rho between the lower-triangle (incl diagonal) entries.
    - For each permutation, permute labels of ONE matrix by applying the same permutation
      to both rows and columns (i.e., relabel nodes), then recompute rho.
    - p-value is two-sided with +1 correction: (count + 1) / (permutations + 1).

    Requirements:
    - `a` and `b` must have identical index/column ordering (same labels and order).
    """
    if permutations <= 0:
        raise ValueError("permutations must be a positive integer")
    if list(a.index) != list(b.index) or list(a.columns) != list(b.columns):
        raise ValueError("Matrices must have identical index/column ordering for Mantel test.")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected square matrix for 'a', got shape={a.shape}")
    if b.shape[0] != b.shape[1]:
        raise ValueError(f"Expected square matrix for 'b', got shape={b.shape}")

    tri = np.tril_indices(a.shape[0], k=0)
    a_vals = a.to_numpy()[tri]
    b_mat = b.to_numpy()

    observed_rho, _ = spearmanr(a_vals, b_mat[tri])
    observed_rho = float(observed_rho)

    n = a.shape[0]
    perm_rhos = np.empty(permutations, dtype=float)
    for i in range(permutations):
        p = rng.permutation(n)
        b_perm = b_mat[p][:, p]
        r, _ = spearmanr(a_vals, b_perm[tri])
        perm_rhos[i] = float(r)

    p_two_sided = (np.sum(np.abs(perm_rhos) >= abs(observed_rho)) + 1.0) / (permutations + 1.0)
    return observed_rho, float(p_two_sided)


def plot_group_tpm_within_between_kde(
    human_tpm: pd.DataFrame,
    machine_tpm: pd.DataFrame,
    *,
    title_prefix: str,
    out_file: Path,
    formats: str | None = None,
    fontsize: int = 16,
    x_max: float = 0.6,
) -> None:
    """
    Reproduce the "within-category (diagonal) vs between-category (off-diagonal)" KDE plot
    for the *group-average* TPMs, comparing Humans vs AIs.

    - Within-category values: diagonal entries
    - Between-category values: off-diagonal entries
    """
    if list(human_tpm.index) != list(machine_tpm.index) or list(human_tpm.columns) != list(machine_tpm.columns):
        raise ValueError("TPMs must have identical index/column ordering for within/between KDE plotting.")

    human_intra_values = np.diag(human_tpm.to_numpy())
    machine_intra_values = np.diag(machine_tpm.to_numpy())

    off_diagonal_mask = ~np.eye(human_tpm.shape[0], dtype=bool)
    human_inter_values = human_tpm.values[off_diagonal_mask]
    machine_inter_values = machine_tpm.values[off_diagonal_mask]

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    sns.kdeplot(
        data=human_intra_values,
        label="Human: Within-subcategory (diagonal)",
        fill=True,
        clip=(0, 1),
        ax=axes[0],
    )
    sns.kdeplot(
        data=machine_intra_values,
        label="AI: Within-subcategory (diagonal)",
        fill=True,
        clip=(0, 1),
        ax=axes[0],
    )
    axes[0].set_title(f"{title_prefix}: Within-category", fontsize=fontsize)
    axes[0].set_ylabel("Density", fontsize=fontsize)
    axes[0].legend(fontsize=fontsize - 4)
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlim(right=x_max)
    axes[0].margins(y=0)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].set_xlabel("")

    sns.kdeplot(
        data=human_inter_values,
        label="Human: Between-subcategory (off-diagonal)",
        fill=True,
        clip=(0, 1),
        ax=axes[1],
    )
    sns.kdeplot(
        data=machine_inter_values,
        label="AI: Between-category (off-diagonal)",
        fill=True,
        clip=(0, 1),
        ax=axes[1],
    )
    axes[1].set_title(f"{title_prefix}: Between-subcategory", fontsize=fontsize)
    axes[1].set_xlabel("Transition Probability", fontsize=fontsize)
    axes[1].set_ylabel("Density", fontsize=fontsize)
    axes[1].legend(fontsize=fontsize - 4)
    axes[1].set_ylim(bottom=0)
    axes[1].margins(y=0)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[0].tick_params(axis="both", which="major", labelsize=fontsize)
    axes[1].tick_params(axis="both", which="major", labelsize=fontsize)

    plt.tight_layout()
    save_figure_formats(
        fig,
        out_file,
        formats=formats,
        default=(out_file.suffix.lstrip(".") or "png",),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_tpm_spearman_jointplot(
    human_tpm: pd.DataFrame,
    machine_tpm: pd.DataFrame,
    *,
    title_prefix: str,
    x_label: str,
    y_label: str,
    out_file: Path,
    formats: str | None = None,
    fontsize: int = 16,
    square_axis: bool = False,
    mantel_permutations: int | None = None,
    mantel_seed: int = 0,
) -> tuple[float, float]:
    """
    Create a Seaborn jointplot showing the Spearman correlation between two TPMs,
    comparing only the lower triangle (including diagonal) entries.

    If `mantel_permutations` is provided, the p-value reported is a Mantel-style
    permutation p-value (permuting labels of one matrix by row+col permutation),
    which accounts for non-independence of network edges.

    Returns: (rho, p_value)
    """
    if list(human_tpm.index) != list(machine_tpm.index) or list(human_tpm.columns) != list(machine_tpm.columns):
        raise ValueError("TPMs must have identical index/column ordering for correlation plotting.")

    human_vals = _flatten_lower_triangle_including_diagonal(human_tpm)
    machine_vals = _flatten_lower_triangle_including_diagonal(machine_tpm)

    if mantel_permutations is None:
        rho, p_value = spearmanr(human_vals, machine_vals)
        rho = float(rho)
        p_value = float(p_value)
        p_text = "p < 0.0001" if p_value < 0.0001 else f"p = {p_value:.3f}"
        title = f"{title_prefix} Spearman Correlation: ρ = {rho:.3f}, {p_text}"
    else:
        if mantel_permutations <= 0:
            raise typer.BadParameter("--mantel must be a positive integer (e.g., --mantel 10000).")
        rng = np.random.default_rng(mantel_seed)
        rho, p_value = _mantel_test_spearman_from_matrices(
            human_tpm,
            machine_tpm,
            permutations=mantel_permutations,
            rng=rng,
        )
        min_p = 1.0 / (mantel_permutations + 1.0)
        if p_value <= min_p + 1e-15:
            p_text = f"p ≤ {min_p:.2g} (Mantel)"
        else:
            p_text = f"p = {p_value:.3g} (Mantel)"
        title = f"{title_prefix}\nρ = {rho:.3f}, {p_text}"

    data = pd.DataFrame({"Human": human_vals, "AI": machine_vals})

    g = sns.jointplot(
        data=data,
        x="Human",
        y="AI",
        kind="reg",
        height=7,
        joint_kws={
            "seed": SEED,
            "scatter_kws": {"alpha": 0.6, "color": "lightgray", "edgecolors": "dimgray", "s": 80},
        },
        line_kws={"color": "red"},
        marginal_kws={"color": "dimgray"},
    )

    g.ax_joint.grid(False)
    g.ax_marg_x.grid(False)
    g.ax_marg_y.grid(False)
    g.ax_joint.tick_params(axis="both", which="major", labelsize=fontsize)
    g.ax_marg_x.tick_params(axis="x", labelsize=fontsize)
    g.ax_marg_y.tick_params(axis="y", labelsize=fontsize)

    if square_axis:
        xmin, xmax = g.ax_joint.get_xlim()
        ymin, ymax = g.ax_joint.get_ylim()
        lo = float(min(xmin, ymin))
        hi = float(max(xmax, ymax))
        g.ax_joint.set_xlim(lo, hi)
        g.ax_joint.set_ylim(lo, hi)

    g.fig.suptitle(title, fontsize=fontsize, y=1.02)
    g.set_axis_labels(x_label, y_label, fontsize=fontsize)

    save_figure_formats(
        g.fig,
        out_file,
        formats=formats,
        default=(out_file.suffix.lstrip(".") or "png",),
        bbox_inches="tight",
    )
    plt.close(g.fig)

    return float(rho), float(p_value)


@app.command()
def main(
    csv_file: Path = typer.Option(
        None,
        help=(
            "Path to the sequences CSV. Defaults to "
            "results/hills/generate/animals-rank-1-gemini-2.5-flash-lite-prompt-animals-gen-snafu.csv"
        ),
    ),
    plot_file: Path = typer.Option(
        Path("results/plots/tpm_grid.png"),
        help="Path to save the 2x2 grid plot (Macro/Micro x Human/LLM).",
    ),
    spearman_plot_file: Path = typer.Option(
        None,
        help=(
            "Optional: path to save a Spearman jointplot comparing Human vs LLM TPM entries "
            "(lower triangle incl diagonal). If not provided, no Spearman plot is produced."
        ),
    ),
    spearman_scale: str = typer.Option(
        "macro",
        help="Which TPM scale to use for the Spearman plot: 'macro' or 'micro'.",
    ),
    mantel: int | None = typer.Option(
        None,
        help=(
            "If set (e.g., --mantel 10000), compute a Mantel-style permutation test for the "
            "Spearman correlation between flattened TPM entries, by permuting matrix labels "
            "to account for non-independence of edges. Value is number of permutations."
        ),
    ),
    mantel_seed: int = typer.Option(
        0,
        help="Random seed for Mantel permutations (only used when --mantel is set).",
    ),
    spearman_square_axis: bool = typer.Option(
        False,
        help="If set, makes the Spearman jointplot scatter panel use a square x/y range.",
    ),
    within_between_plot_file: Path = typer.Option(
        None,
        help=(
            "Optional: path to save a within- vs between-category KDE plot "
            "from the group-average TPMs (Human vs Machine)."
        ),
    ),
    within_between_scale: str = typer.Option(
        "macro",
        help="Which TPM scale to use for the within/between KDE plot: 'macro' or 'micro'.",
    ),
    within_between_xmax: float = typer.Option(
        0.6,
        help="Right x-axis limit for the within/between KDE plot.",
    ),
    fractional_weighting: bool = typer.Option(
        False,
        help="Use fractional weighting for transitions (recommended when multiple categories per item).",
    ),
    maximum_coherence: bool = typer.Option(
        False,
        help="Use maximum coherence for transitions (resolves multi-category ambiguity by overlap with neighbors).",
    ),
    include_self_transitions: bool = typer.Option(True, help="Whether to include self-transitions."),
    vmin_macro: float = typer.Option(0.0, help="Heatmap vmin for macro TPM panels."),
    vmax_macro: float = typer.Option(0.4, help="Heatmap vmax for macro TPM panels."),
    vmin_micro: float = typer.Option(0.0, help="Heatmap vmin for micro TPM panels."),
    vmax_micro: float = typer.Option(0.025, help="Heatmap vmax for micro TPM panels."),
    plotting_config: Path | None = typer.Option(
        None,
        "--plotting-config",
        help="Path to plotting configuration TOML. If omitted, uses configs/plotting.toml from project root when available.",
    ),
    formats: str | None = typer.Option(
        None,
        "--formats",
        help="Comma-separated output formats for figures: png, pdf, svg. Defaults preserve each output path suffix.",
    ),
):
    """
    Load a single CSV containing both human and LLM sequences (one row per item),
    compute Human vs LLM TPMs at micro (word) and macro (category) scales, and plot them.

    Expected columns in the CSV:
      - 'human_response' (token/word)
      - 'response' (token/word)
      - 'human_response_category' (list-like categories; stringified python list is allowed)
      - 'response_category' (list-like categories; stringified python list is allowed)
      - 'id' (subject/run id)
      - 'rank' (position in sequence; 1..N)
      - 'seq_type' (optional; used to group sequences within id if present)
    """
    project_dir = find_project_root()
    if project_dir is None:
        raise typer.BadParameter("Could not determine project root (no .git marker found).")

    config_path = plotting_config if plotting_config is not None else (project_dir / "configs" / "plotting.toml")
    apply_seaborn_theme_from_config(config_path)
    apply_bold_axis_style()

    if csv_file is None:
        csv_file = (
            project_dir / "results/hills/generate/animals-rank-1-gemini-2.5-flash-lite-prompt-animals-gen-snafu.csv"
        )
    if not csv_file.exists():
        raise typer.BadParameter(f"CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)

    required = [
        "id",
        "rank",
        "human_response",
        "response",
        "human_response_category",
        "response_category",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise typer.BadParameter(f"Missing required columns: {missing}")

    id_col = "id"
    order_col = "rank"
    seq_col = "seq_type" if "seq_type" in df.columns else None

    human_data = df[
        [id_col, order_col] + ([seq_col] if seq_col else []) + ["human_response", "human_response_category"]
    ].copy()
    human_data = human_data.rename(
        columns={
            "human_response": "response",
            "human_response_category": "category",
        }
    )

    machine_data = df[[id_col, order_col] + ([seq_col] if seq_col else []) + ["response", "response_category"]].copy()
    machine_data = machine_data.rename(columns={"response_category": "category"})

    m_micro, h_micro = get_avg_tpm_for_col(
        machine_data,
        human_data,
        "response",
        fractional_weighting=fractional_weighting,
        maximum_coherence=maximum_coherence,
        include_self_transitions=include_self_transitions,
        id_col=id_col,
        order_col=order_col,
        seq_col=seq_col,
    )
    m_macro, h_macro = get_avg_tpm_for_col(
        machine_data,
        human_data,
        "category",
        fractional_weighting=fractional_weighting,
        maximum_coherence=maximum_coherence,
        include_self_transitions=include_self_transitions,
        id_col=id_col,
        order_col=order_col,
        seq_col=seq_col,
    )

    if m_micro is None or h_micro is None or m_macro is None or h_macro is None:
        raise RuntimeError("Could not compute all TPMs (one or more are None).")

    plot_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(24, 20))

    panel_labels = [["A", "B"], ["C", "D"]]
    for r in range(2):
        for c in range(2):
            axes[r, c].text(
                -0.08,
                1.02,
                panel_labels[r][c],
                transform=axes[r, c].transAxes,
                ha="left",
                va="bottom",
                fontsize=20,
                fontweight="bold",
                color="black",
                clip_on=False,
            )

    # Row 0: Macro (Category)
    sns.heatmap(h_macro, ax=axes[0, 0], cmap="viridis", cbar=True, vmin=vmin_macro, vmax=vmax_macro)
    axes[0, 0].set_title("Human Macro TPM (Subcategory)")
    axes[0, 0].grid(False)

    sns.heatmap(m_macro, ax=axes[0, 1], cmap="viridis", cbar=True, vmin=vmin_macro, vmax=vmax_macro)
    axes[0, 1].set_title("LLM Macro TPM (Subcategory)")
    axes[0, 1].grid(False)

    # Row 1: Micro (Vocabulary)
    sns.heatmap(h_micro, ax=axes[1, 0], cmap="viridis", cbar=True, vmin=vmin_micro, vmax=vmax_micro)
    axes[1, 0].set_title("Human Micro TPM (Vocabulary)")
    axes[1, 0].grid(False)

    sns.heatmap(m_micro, ax=axes[1, 1], cmap="viridis", cbar=True, vmin=vmin_micro, vmax=vmax_micro)
    axes[1, 1].set_title("LLM Micro TPM (Vocabulary)")
    axes[1, 1].grid(False)

    plt.suptitle("TPM - Human vs. LLM - Macro vs. Micro", fontsize=20, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure_formats(
        fig,
        plot_file,
        formats=formats,
        default=(plot_file.suffix.lstrip(".") or "png",),
    )
    plt.close()

    typer.echo(f"Saved plot to: {plot_file}")

    if spearman_plot_file is not None:
        scale = spearman_scale.strip().lower()
        if scale not in {"macro", "micro"}:
            raise typer.BadParameter("--spearman-scale must be either 'macro' or 'micro'.")

        if scale == "macro":
            _rho, _p = plot_tpm_spearman_jointplot(
                human_tpm=h_macro,
                machine_tpm=m_macro,
                title_prefix="Macro TPM (Subcategory)",
                x_label="Human Subcategory Probability Transitions",
                y_label="AI Subcategory Probability Transitions",
                out_file=spearman_plot_file,
                formats=formats,
                square_axis=spearman_square_axis,
                mantel_permutations=mantel,
                mantel_seed=mantel_seed,
            )
        else:
            _rho, _p = plot_tpm_spearman_jointplot(
                human_tpm=h_micro,
                machine_tpm=m_micro,
                title_prefix="Micro TPM (Vocabulary)",
                x_label="Human Vocabulary Probability Transitions",
                y_label="AI Vocabulary Probability Transitions",
                out_file=spearman_plot_file,
                formats=formats,
                square_axis=spearman_square_axis,
                mantel_permutations=mantel,
                mantel_seed=mantel_seed,
            )

        if mantel is None:
            typer.echo(f"Saved Spearman jointplot to: {spearman_plot_file} (ρ={_rho:.3f}, p={_p:.3g})")
        else:
            typer.echo(
                f"Saved Spearman jointplot to: {spearman_plot_file} (ρ={_rho:.3f}, Mantel p={_p:.3g}, perms={mantel})"
            )

    if within_between_plot_file is not None:
        scale = within_between_scale.strip().lower()
        if scale not in {"macro", "micro"}:
            raise typer.BadParameter("--within-between-scale must be either 'macro' or 'micro'.")

        if scale == "macro":
            plot_group_tpm_within_between_kde(
                human_tpm=h_macro,
                machine_tpm=m_macro,
                title_prefix="Macro TPM (Subcategory) Human vs. AI",
                out_file=within_between_plot_file,
                formats=formats,
                fontsize=16,
                x_max=within_between_xmax,
            )
        else:
            plot_group_tpm_within_between_kde(
                human_tpm=h_micro,
                machine_tpm=m_micro,
                title_prefix="Micro TPM (Vocabulary) Human vs. AI",
                out_file=within_between_plot_file,
                formats=formats,
                fontsize=16,
                x_max=within_between_xmax,
            )

        typer.echo(f"Saved within/between KDE plot to: {within_between_plot_file}")


if __name__ == "__main__":
    seed_everything()
    app()
