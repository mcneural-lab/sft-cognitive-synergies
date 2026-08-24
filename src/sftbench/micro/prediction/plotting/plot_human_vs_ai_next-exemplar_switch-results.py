from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
from statannotations.Annotator import Annotator


def _mean_ci95_t(values: pd.Series) -> tuple[float, float, float, int]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    n = int(x.shape[0])
    mean = float(x.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan"), float("nan"), n

    sd = float(x.std(ddof=1))
    if not pd.notna(sd) or sd == 0.0:
        return mean, float("nan"), float("nan"), n

    se = sd / (n**0.5)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    lo = mean - tcrit * se
    hi = mean + tcrit * se
    return mean, lo, hi, n


def _paired_wilcoxon_from_wide(wide: pd.DataFrame, col_a: str, col_b: str) -> tuple[float, float, int]:
    """
    Paired Wilcoxon signed-rank test between two columns in a wide per-id table.

    Returns (w_stat, p_value, n_pairs). If <2 pairs, returns NaNs for stats.

    Notes:
    - Uses scipy.stats.wilcoxon with default two-sided alternative.
    - Drops NaNs pairwise; if all paired differences are zero, SciPy can raise a ValueError.
      In that case, we return (nan, nan, n_pairs) rather than crashing.
    """
    if wide is None or wide.empty:
        return float("nan"), float("nan"), 0

    if col_a not in wide.columns or col_b not in wide.columns:
        return float("nan"), float("nan"), 0

    pairs = wide[[col_a, col_b]].dropna()
    n_pairs = int(pairs.shape[0])
    if n_pairs < 2:
        return float("nan"), float("nan"), n_pairs

    x = pairs[col_a].astype(float)
    y = pairs[col_b].astype(float)

    try:
        res = stats.wilcoxon(x, y)
    except ValueError:
        return float("nan"), float("nan"), n_pairs

    return float(res.statistic), float(res.pvalue), n_pairs


from sftbench import find_project_root
from sftbench.figure_output import save_figure_formats
from sftbench.micro.prediction.plotting import apply_seaborn_theme_from_config
from sftbench.reproducibility import SEED, seed_everything

app = typer.Typer(
    help="CLI to plot switch prediction evaluation for merged micro prediction CSVs.",
    add_completion=False,
)


_root = find_project_root() or Path(".")
DEFAULT_TARGETS_DIR = _root / "results" / "micro" / "prediction" / "switch" / "gemini-3.0-pro"
DEFAULT_SNAFU_PATH = _root / "data" / "sequences" / "animals_snafu_scheme.csv"
DEFAULT_PLOTTING_CFG = _root / "configs" / "plotting.toml"


ID_COL = "id"
RANK_COL = "rank"

HUMAN_PRED_COL = "human_switch_prediction"
MACHINE_PRED_COL = "predicted_switch_response"

GT_COL = "embedding_switch_prediction"


@dataclass(frozen=True)
class DatasetSpec:
    """Holds data used for evaluation and plotting."""

    df_long: pd.DataFrame
    targets_dir: Path
    files: list[Path]


def load_plotting_config(config_path: Path) -> None:
    """Load seaborn context/style from a TOML file, falling back to robust defaults."""
    apply_seaborn_theme_from_config(config_path)


def _iter_csvs_non_recursive(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Targets dir not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Targets path is not a directory: {dir_path}")
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])


def _coerce_switch_series(s: pd.Series, *, context: str) -> pd.Series:
    """
    Coerce a switch-like series to pandas nullable boolean (BooleanDtype).

    Accepts:
      - booleans
      - 0/1 integers
      - strings like: "true"/"false", "True"/"False", "0"/"1", "yes"/"no"
      - empty strings => NA
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")

    if pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        invalid = ~(s_num.isna() | s_num.isin([0, 1]))
        if invalid.any():
            bad = s[invalid].head(10).tolist()
            raise ValueError(f"{context}: invalid numeric switch values (expected 0/1). Examples: {bad}")
        return (s_num == 1).astype("boolean")

    s_obj = s.astype("string")
    s_str = s_obj.str.strip().str.lower()

    s_str = s_str.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})

    true_set = {"true", "t", "1", "yes", "y", "True", "TRUE"}
    false_set = {"false", "f", "0", "no", "n", "False", "FALSE"}

    mapped = pd.Series(pd.NA, index=s_str.index, dtype="boolean")
    mapped[s_str.isin(true_set)] = True
    mapped[s_str.isin(false_set)] = False

    bad_mask = s_str.notna() & mapped.isna()
    if bad_mask.any():
        bad = s.loc[bad_mask].head(10).tolist()
        raise ValueError(f"{context}: could not parse some switch values. Examples: {bad}")

    return mapped


def _as_int_rank(series: pd.Series, *, context: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any():
        bad = series[s.isna()].head(10).tolist()
        raise ValueError(f"{context}: could not parse some `{RANK_COL}` values as numbers. Examples: {bad}")
    return s.astype("Int64")


def _infer_data_category_from_path(csv_path: Path) -> str:
    """
    Infer a readable data-category from filename when the column isn't present.

    Examples:
      animals-switch-gemini-3.0-pro.csv -> animals
      clothes-switch-... -> clothes
      supermarket-switch-... -> supermarket
    """
    stem = csv_path.stem
    for prefix in ("animals", "clothes", "supermarket"):
        if stem.startswith(prefix):
            return prefix
    return stem


def _validate_required_columns(df: pd.DataFrame, *, csv_path: Path) -> None:
    required = {ID_COL, RANK_COL, HUMAN_PRED_COL, MACHINE_PRED_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {sorted(missing)}. Found columns: {list(df.columns)}")


def _prepare_long_df_from_merged_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load one merged micro prediction CSV and produce a "long" dataframe with:
      id, rank, data-category, type, y_true, y_pred

    Ground truth is defined as embedding_switch_prediction (GT_COL), as requested.
    """
    df = pd.read_csv(csv_path)
    _validate_required_columns(df, csv_path=csv_path)

    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str)
    df[RANK_COL] = _as_int_rank(df[RANK_COL], context=str(csv_path))

    if "data-category" in df.columns:
        df["data-category"] = df["data-category"].astype(str)
    elif "category" in df.columns:
        df["data-category"] = df["category"].astype(str).str.replace("-switch", "", regex=False)
    else:
        df["data-category"] = _infer_data_category_from_path(csv_path)

    df[HUMAN_PRED_COL] = _coerce_switch_series(df[HUMAN_PRED_COL], context=f"{csv_path}:{HUMAN_PRED_COL}")
    df[MACHINE_PRED_COL] = _coerce_switch_series(df[MACHINE_PRED_COL], context=f"{csv_path}:{MACHINE_PRED_COL}")

    stable_seed = int(pd.util.hash_pandas_object(pd.Index([csv_path.name]), index=False).iloc[0] % (2**32))
    rng = np.random.default_rng(stable_seed)

    y_true = df[GT_COL]

    base = df.loc[y_true.notna(), [ID_COL, RANK_COL, "data-category", HUMAN_PRED_COL, MACHINE_PRED_COL]].copy()
    base = base.rename(columns={HUMAN_PRED_COL: "y_pred_human", MACHINE_PRED_COL: "y_pred_machine"})
    base["y_true"] = _coerce_switch_series(y_true.loc[y_true.notna()], context=f"{csv_path}:{GT_COL}")

    p_true = float(base["y_true"].astype(bool).mean()) if len(base) else 0.0
    base["random_switch_baseline"] = rng.random(len(base)) < p_true
    long_human = base[[ID_COL, RANK_COL, "data-category", "y_true", "y_pred_human"]].rename(
        columns={"y_pred_human": "y_pred"}
    )
    long_human["type"] = "human"

    long_machine = base[[ID_COL, RANK_COL, "data-category", "y_true", "y_pred_machine"]].rename(
        columns={"y_pred_machine": "y_pred"}
    )
    long_machine["type"] = "machine"

    long_random = base[[ID_COL, RANK_COL, "data-category", "y_true", "random_switch_baseline"]].rename(
        columns={"random_switch_baseline": "y_pred"}
    )
    long_random["type"] = "random_switch_baseline"

    long_df = pd.concat([long_human, long_machine, long_random], ignore_index=True)

    long_df = long_df.loc[long_df["y_pred"].notna()].copy()

    return long_df


def load_merged_micro_predictions(targets_dir: Path) -> DatasetSpec:
    csvs = _iter_csvs_non_recursive(targets_dir)
    if not csvs:
        raise ValueError(f"No CSV files found in {targets_dir}")

    long_parts: list[pd.DataFrame] = []
    for p in csvs:
        long_parts.append(_prepare_long_df_from_merged_csv(p))
    df_long = pd.concat(long_parts, ignore_index=True)

    if df_long.empty:
        raise ValueError("After loading and filtering, there were no evaluable rows (empty dataframe).")

    df_long["rank_group"] = ((df_long[RANK_COL].astype(int) - 1) // 5) * 5

    return DatasetSpec(df_long=df_long, targets_dir=targets_dir, files=csvs)


def _classification_report_df(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """
    Returns a tidy dataframe version of sklearn's classification_report.

    Uses boolean labels; ensures stable ordering.
    """
    y_true_b = y_true.astype(bool)
    y_pred_b = y_pred.astype(bool)

    rep = classification_report(
        y_true_b,
        y_pred_b,
        labels=[False, True],
        target_names=["False", "True"],
        output_dict=True,
        zero_division=0,
    )
    df_rep = pd.DataFrame(rep).T
    return df_rep


def calculate_per_id_metrics(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Per-id metrics for each participant/sequence id and each evaluator type.

    Expected columns:
      id, type, data-category, y_true, y_pred
    """
    if df_long.empty:
        return pd.DataFrame()

    required = {ID_COL, "type", "data-category", "y_true", "y_pred"}
    missing = required - set(df_long.columns)
    if missing:
        raise ValueError(f"calculate_per_id_metrics: missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for (an_id, typ), sub in df_long.groupby([ID_COL, "type"], sort=False):
        y_true = sub["y_true"].astype(bool)
        y_pred = sub["y_pred"].astype(bool)

        rep = classification_report(
            y_true,
            y_pred,
            labels=[False, True],
            target_names=["False", "True"],
            output_dict=True,
            zero_division=0,
        )

        cat_vals = sub["data-category"].astype(str).unique().tolist()
        data_cat = cat_vals[0] if len(cat_vals) == 1 else "mixed"

        rows.append(
            {
                "id": an_id,
                "type": typ,
                "data-category": data_cat,
                "False_precision": rep.get("False", {}).get("precision", np.nan),
                "False_recall": rep.get("False", {}).get("recall", np.nan),
                "False_f1-score": rep.get("False", {}).get("f1-score", np.nan),
                "True_precision": rep.get("True", {}).get("precision", np.nan),
                "True_recall": rep.get("True", {}).get("recall", np.nan),
                "True_f1-score": rep.get("True", {}).get("f1-score", np.nan),
                "macro_precision": rep.get("macro avg", {}).get("precision", np.nan),
                "macro_recall": rep.get("macro avg", {}).get("recall", np.nan),
                "macro_f1-score": rep.get("macro avg", {}).get("f1-score", np.nan),
                "accuracy": rep.get("accuracy", np.nan),
                "support": len(sub),
            }
        )

    return pd.DataFrame(rows)


def create_latex_classification_table(df_long: pd.DataFrame) -> str:
    """
    Generate a LaTeX table with classification report rows for:
      - Human vs GT
      - Machine vs GT

    Ground truth is embedding_switch_prediction as specified (i.e., same column as machine prediction in source),
    so machine-vs-GT is expected to be perfect unless rows were filtered or coerced differently.
    """
    out_rows: list[pd.DataFrame] = []
    for typ in ["human", "machine"]:
        sub = df_long[df_long["type"] == typ]
        rep = _classification_report_df(sub["y_true"], sub["y_pred"])

        keep_index = ["False", "True", "macro avg", "accuracy"]
        rep = rep.loc[[i for i in keep_index if i in rep.index]].copy()

        rep = rep.reset_index().rename(columns={"index": "Class"})
        rep["Type"] = "Human" if typ == "human" else "Machine"

        for col in ["precision", "recall", "f1-score", "support"]:
            if col not in rep.columns:
                rep[col] = np.nan

        out_rows.append(rep[["Type", "Class", "precision", "recall", "f1-score", "support"]])

    df_combined = pd.concat(out_rows, ignore_index=True)

    def _latex_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value).replace("_", r"\_")

    lines = [
        r"\begin{table}",
        r"\caption{Classification Report: Human vs Machine switch predictions against embedding-based ground truth}",
        r"\label{tab:classification_report_switch}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Type & Class & precision & recall & f1-score & support \\",
        r"\midrule",
    ]
    for row in df_combined.itertuples(index=False):
        lines.append(" & ".join(_latex_value(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _savefig(output_dir: Path, filename: str, *, formats: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    save_figure_formats(None, path, formats=formats, default=("png", "pdf"), dpi=300, bbox_inches="tight")


def _plot_confusion_matrices(df_long: pd.DataFrame, output_dir: Path, show: bool, *, formats: str | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, typ, title in zip(
        axes,
        ["human", "machine"],
        ["Confusion Matrix - Human vs GT", "Confusion Matrix - Machine vs GT"],
        strict=False,
    ):
        sub = df_long[df_long["type"] == typ]
        y_true = sub["y_true"].astype(bool)
        y_pred = sub["y_pred"].astype(bool)

        cm = confusion_matrix(y_true, y_pred, labels=[False, True])
        cm_df = pd.DataFrame(cm, index=["GT False", "GT True"], columns=["Pred False", "Pred True"])

        sns.heatmap(cm_df, annot=True, fmt="d", cmap="viridis", cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    _savefig(output_dir, "confusion_matrices_human_machine_vs_gt.png", formats=formats)
    if show:
        plt.show()
    plt.close(fig)


def _plot_classification_heatmaps(df_long: pd.DataFrame, output_dir: Path, show: bool, *, formats: str | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, typ, title in zip(
        axes,
        ["human", "machine"],
        ["Classification Report - Human vs GT", "Classification Report - Machine vs GT"],
        strict=False,
    ):
        sub = df_long[df_long["type"] == typ]
        rep = _classification_report_df(sub["y_true"], sub["y_pred"])

        rep_show = rep.loc[rep.index.isin(["False", "True", "macro avg", "accuracy"])].copy()
        sns.heatmap(rep_show, annot=True, cmap="viridis", fmt=".3f", ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    _savefig(output_dir, "classification_reports_human_machine_vs_gt.png", formats=formats)
    if show:
        plt.show()
    plt.close(fig)


def _plot_three_panel(
    df_long: pd.DataFrame,
    df_metrics_plot: pd.DataFrame,
    metric: str,
    output_dir: Path,
    show: bool,
    formats: str | None,
) -> None:
    """
    Single 1x3 panel figure, matching the layout style of the prediction results plot:
      - Overall F1 (per-id)
      - F1 by category (per-id)
      - Switch prediction accuracy by rank window (GT=True)
    """
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.6, 1.2, 1.2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2])

    for label, ax in zip(["B", "C", "D"], [ax1, ax2, ax3], strict=True):
        ax.text(
            -0.08,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    df_metrics_main = df_metrics_plot[df_metrics_plot["type"].isin(["human", "machine"])].copy()
    sns.boxplot(data=df_metrics_main, x="type", y=metric, hue="type", legend=False, ax=ax1)

    if "random_switch_baseline" in df_metrics_plot["type"].unique():
        random_overall = (
            df_metrics_plot.loc[df_metrics_plot["type"] == "random_switch_baseline", metric].astype(float).mean()
        )
        if pd.notna(random_overall):
            ax1.axhline(
                y=float(random_overall),
                color="black",
                linestyle=":",
                linewidth=3.0,
                alpha=1.0,
                zorder=0,
            )
            handles, labels = ax1.get_legend_handles_labels()
            if "random" not in labels:
                handles.append(plt.Line2D([0], [0], color="black", linestyle=":", linewidth=3.0, label="random"))
                labels.append("random")
            ax1.legend(handles, labels, title="Type", loc="lower right")

    ax1.set_title("Overall Accuracy")
    ax1.set_xlabel("Type")
    ax1.set_ylabel("Switch Accuracy")

    annotator = Annotator(
        ax1,
        [("human", "machine")],
        data=df_metrics_main,
        x="type",
        y=metric,
        order=["human", "machine"],
    )
    annotator.configure(test="Wilcoxon", text_format="star")
    annotator.apply_and_annotate()

    category_order = [
        c for c in ["animals", "clothes", "supermarket"] if c in df_metrics_plot["data-category"].unique()
    ]
    if not category_order:
        category_order = sorted(df_metrics_plot["data-category"].unique().tolist())

    sns.stripplot(
        data=df_metrics_main,
        x="data-category",
        y=metric,
        hue="type",
        order=category_order,
        jitter=0.2,
        dodge=0.2,
        alpha=0.5,
        size=6,
        zorder=1,
        ax=ax2,
    )

    if "random_switch_baseline" in df_metrics_plot["type"].unique():
        random_cat = (
            df_metrics_plot.loc[df_metrics_plot["type"] == "random_switch_baseline"]
            .groupby("data-category", dropna=False)[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: "random_mean"})
        )
        for i, cat in enumerate(category_order):
            row = random_cat.loc[random_cat["data-category"].astype(str) == str(cat)]
            if row.empty:
                continue
            y = float(row["random_mean"].iloc[0])
            if not pd.notna(y):
                continue
            ax2.hlines(
                y=y,
                xmin=i - 0.35,
                xmax=i + 0.35,
                colors="black",
                linestyles=":",
                linewidth=3.0,
                alpha=1.0,
                zorder=2,
            )
    sns.pointplot(
        data=df_metrics_main,
        x="data-category",
        y=metric,
        hue="type",
        order=category_order,
        linestyle="none",
        dodge=0.6,
        capsize=0.2,
        errorbar=("ci", 95),
        seed=SEED,
        legend=False,
        ax=ax2,
    )
    ax2.set_title("Accuracy by Category")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("")

    pairs = [[(cat, "machine"), (cat, "human")] for cat in category_order]
    annotator = Annotator(
        ax2,
        pairs,
        data=df_metrics_main,
        x="data-category",
        y=metric,
        order=category_order,
        hue="type",
        hue_order=["human", "machine"],
    )
    annotator.configure(test="Wilcoxon", text_format="star")
    annotator.apply_and_annotate()
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, title="Type", loc="upper right", bbox_to_anchor=(1.0, 0.82))

    df_rank = df_long.copy()

    if not df_rank.empty:

        def _metric_for_group(g: pd.DataFrame) -> float:
            """
            Compute the requested per-id metric within a rank window.

            This mirrors the per-id metric computation (classification_report-derived columns)
            used elsewhere in this script, but scoped to (id, type, rank_group).

            Special-case: accuracy is directly computed as mean correctness.
            """
            if metric == "accuracy":
                return float((g["y_pred"].astype(bool) == g["y_true"].astype(bool)).mean())

            rep = classification_report(
                g["y_true"].astype(bool),
                g["y_pred"].astype(bool),
                labels=[False, True],
                target_names=["False", "True"],
                zero_division=0,
                output_dict=True,
            )

            if metric.startswith("True_"):
                key = metric.replace("True_", "")
                return float(rep["True"][key])
            if metric.startswith("False_"):
                key = metric.replace("False_", "")
                return float(rep["False"][key])
            if metric.startswith("macro_"):
                key = metric.replace("macro_", "")
                return float(rep["macro avg"][key])

            raise ValueError(f"Unsupported metric for rank panel: {metric}")

        df_metric_rank_by_id = (
            df_rank.groupby(["id", "type", "rank_group"], dropna=False)
            .apply(_metric_for_group, include_groups=False)
            .reset_index(name="metric_value")
        )

        max_rank = 40
        df_metric_rank_by_id = df_metric_rank_by_id[df_metric_rank_by_id["rank_group"].astype(int) < max_rank].copy()

        df_metric_rank_main = df_metric_rank_by_id[df_metric_rank_by_id["type"].isin(["human", "machine"])].copy()
        sns.pointplot(
            data=df_metric_rank_main,
            x="rank_group",
            y="metric_value",
            hue="type",
            capsize=0.2,
            errorbar=("ci", 95),
            seed=SEED,
            ax=ax3,
        )

        random_rank = (
            df_metric_rank_by_id.loc[df_metric_rank_by_id["type"] == "random_switch_baseline"]
            .groupby("rank_group", dropna=False)["metric_value"]
            .mean()
            .reset_index()
            .sort_values("rank_group")
        )
        if not random_rank.empty:
            rank_groups = random_rank["rank_group"].tolist()
            x_pos = list(range(len(rank_groups)))
            ax3.plot(
                x_pos,
                random_rank["metric_value"].tolist(),
                color="black",
                linestyle=":",
                linewidth=3.0,
                alpha=1.0,
                zorder=3,
            )
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([str(rg) for rg in rank_groups])

        ax3.set_title("Accuracy by Rank")
        ax3.set_xlabel("Rank Window")
        ax3.set_ylabel("")

        handles, labels = ax3.get_legend_handles_labels()
        if "random_switch_baseline" in df_metric_rank_by_id["type"].unique():
            if "random" not in labels:
                handles.append(plt.Line2D([0], [0], color="black", linestyle=":", linewidth=3.0, label="random"))
                labels.append("random")
        ax3.legend(handles, labels, title="Type")
    else:
        ax3.set_title("Accuracy by Rank")
        ax3.set_xlabel("Rank Window")
        ax3.set_ylabel("")
        ax3.text(
            0.5,
            0.5,
            "No GT=True rows available after filtering",
            transform=ax3.transAxes,
            ha="center",
            va="center",
        )

    plt.suptitle("AI vs. Human Switch Prediction Comparison")
    plt.tight_layout()

    _savefig(output_dir, "switch_three_panel_by_rank.png", formats=formats)
    if show:
        plt.show()
    plt.close(fig)


def _write_text_summary(df_long: pd.DataFrame, df_metrics: pd.DataFrame, output_dir: Path, *, metric: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=== Switch Evaluation Summary ===")
    lines.append(f"Ground truth column: {GT_COL}")
    lines.append(f"Compared predictors: {HUMAN_PRED_COL} (human), {MACHINE_PRED_COL} (machine)")
    lines.append(f"Total evaluable rows (after filtering NAs): {len(df_long)}")
    lines.append(f"Plotted metric: {metric}")
    lines.append("")

    gt = df_long["y_true"].astype(bool)
    gt_counts = gt.value_counts(dropna=False)
    gt_false = int(gt_counts.get(False, 0))
    gt_true = int(gt_counts.get(True, 0))
    gt_total = int(len(gt))
    gt_true_frac = (gt_true / gt_total) if gt_total else 0.0
    gt_true_pct = gt_true_frac * 100.0
    gt_false_pct = (gt_false / gt_total * 100.0) if gt_total else 0.0
    lines.append("--- Ground-truth support (y_true) ---")
    lines.append(f"False: {gt_false} ({gt_false_pct:.2f}%)")
    lines.append(f"True:  {gt_true} ({gt_true_pct:.2f}%)")
    lines.append(f"Fraction Ground Truth switches (y_true=True): {gt_true_frac:.6f}")
    lines.append("")

    lines.append("--- Ground-truth support by category (y_true) ---")
    if "data-category" not in df_long.columns:
        lines.append("WARNING: missing 'data-category' column; cannot report per-category ground-truth support.")
        lines.append("")
    else:
        cats = df_long["data-category"].dropna().astype(str).unique().tolist()
        for cat in sorted(cats):
            cat_mask = df_long["data-category"].astype(str) == str(cat)
            gt_cat = df_long.loc[cat_mask, "y_true"].astype(bool)
            gt_cat_counts = gt_cat.value_counts(dropna=False)
            cat_false = int(gt_cat_counts.get(False, 0))
            cat_true = int(gt_cat_counts.get(True, 0))
            cat_total = int(len(gt_cat))
            cat_true_frac = (cat_true / cat_total) if cat_total else 0.0
            cat_true_pct = cat_true_frac * 100.0
            cat_false_pct = (cat_false / cat_total * 100.0) if cat_total else 0.0
            lines.append(f"{cat}:")
            lines.append(f"  False: {cat_false} ({cat_false_pct:.2f}%)")
            lines.append(f"  True:  {cat_true} ({cat_true_pct:.2f}%)")
            lines.append(f"  Fraction Ground Truth switches (y_true=True): {cat_true_frac:.6f}")
        lines.append("")

    for typ in ["human", "machine"]:
        sub = df_long[df_long["type"] == typ]
        rep = classification_report(
            sub["y_true"].astype(bool),
            sub["y_pred"].astype(bool),
            labels=[False, True],
            target_names=["False", "True"],
            zero_division=0,
        )
        lines.append(f"--- Classification report ({typ}) ---")
        lines.append(rep)

    lines.append("")
    lines.append(f"--- Participant-level {metric} (M ± 95% t CI) + paired Wilcoxon signed-rank (machine vs human) ---")

    if df_metrics.empty:
        lines.append("No per-id metrics available; skipping participant-level CI and paired Wilcoxon reporting.")
        (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
        return

    overall_units = df_metrics.copy()
    overall_wide = overall_units.pivot_table(index=["id", "data-category"], columns="type", values=metric, dropna=False)

    lines.append("Overall (all categories pooled; per-id-category paired units):")
    for typ in ["human", "machine"]:
        vals = overall_units.loc[overall_units["type"] == typ, metric]
        mean, lo, hi, n = _mean_ci95_t(vals)
        lines.append(f"  {typ}: M={mean:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}], n={n}")

    w_stat, p_value, n_pairs = _paired_wilcoxon_from_wide(overall_wide, "machine", "human")
    lines.append(
        f"  paired Wilcoxon signed-rank (machine vs human): W={w_stat:.4f}, p={p_value:.6g}, n_pairs={n_pairs}"
    )

    category_order = [c for c in ["animals", "clothes", "supermarket"] if c in df_metrics["data-category"].unique()]
    if not category_order:
        category_order = sorted(df_metrics["data-category"].unique().tolist())

    lines.append("")
    lines.append("By category (paired within id):")
    for cat in category_order:
        lines.append(f"  {cat}:")
        sub_cat = df_metrics[df_metrics["data-category"] == cat]
        for typ in ["human", "machine"]:
            vals = sub_cat.loc[sub_cat["type"] == typ, metric]
            mean, lo, hi, n = _mean_ci95_t(vals)
            lines.append(f"    {typ}: M={mean:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}], n={n}")

        wide = sub_cat.pivot(index="id", columns="type", values=metric)
        w_stat, p_value, n_pairs = _paired_wilcoxon_from_wide(wide, "machine", "human")
        lines.append(
            f"    paired Wilcoxon signed-rank (machine vs human): W={w_stat:.4f}, p={p_value:.6g}, n_pairs={n_pairs}"
        )

    lines.append("")
    lines.append("--- Per-id summary (mean of per-id metrics) ---")
    for typ in ["human", "machine"]:
        sub_m = df_metrics[df_metrics["type"] == typ]
        if sub_m.empty:
            continue
        lines.append(f"{typ}: mean {metric} = {sub_m[metric].mean():.4f} (n_ids={sub_m['id'].nunique()})")

    if not df_metrics.empty:
        try:
            metric_by_cat = df_metrics.groupby(["data-category", "type"])[metric].mean().unstack()
            if "human" in metric_by_cat.columns and "machine" in metric_by_cat.columns:
                metric_by_cat["delta(machine-human)"] = metric_by_cat["machine"] - metric_by_cat["human"]
            lines.append("")
            lines.append("Per-id mean metric by category:")
            lines.append(metric_by_cat.round(4).to_string())
        except Exception as e:
            lines.append("")
            lines.append(f"WARNING: could not compute by-category summary: {e}")

    summary_text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    typer.echo("")
    typer.echo(summary_text)


@app.command()
def plot(
    targets_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_TARGETS_DIR,
        help="Directory containing merged micro prediction CSVs (non-recursive).",
    ),
    config: Path = typer.Option(  # noqa: B008
        DEFAULT_PLOTTING_CFG,
        help="Path to the plotting configuration TOML (seaborn settings).",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("plots"),
        help="Directory to save generated plots and tables.",
    ),
    show: bool = typer.Option(  # noqa: B008
        False,
        help="Whether to show plots interactively.",
    ),
    min_rank: int = typer.Option(  # noqa: B008
        2,
        help="Minimum rank to include (default excludes rank==1).",
    ),
    metric: str = typer.Option(  # noqa: B008
        "True_f1-score",
        help=(
            "Per-id metric column to plot (from the per-id classification report). "
            "Examples: 'True_f1-score' (default), 'True_precision', 'True_recall', "
            "'False_precision', 'False_recall', 'accuracy', 'macro_f1-score', 'macro_precision', 'macro_recall'."
        ),
    ),
    overall_test: str = typer.Option(  # noqa: B008
        "mann-whitney",
        help="Statistical test for overall Human vs Machine per-id comparison in the F1 plot. "
        "Options: 'mann-whitney' (default) or 'paired-t'.",
    ),
    formats: str | None = typer.Option(
        None,
        "--formats",
        help="Comma-separated output formats for figures: png, pdf, svg. Defaults to png,pdf.",
    ),
):
    """
    Plot evaluation of switch predictions for merged micro prediction CSVs.

    This script expects CSVs that include:
      - `human_switch_prediction` (human predictions)
      - `embedding_switch_prediction` (machine predictions AND ground truth per specification)
    """
    load_plotting_config(config)

    spec = load_merged_micro_predictions(targets_dir)
    df_long = spec.df_long.copy()

    df_long = df_long[df_long[RANK_COL].astype(int) >= min_rank].copy()
    if df_long.empty:
        raise typer.Exit(f"No rows remain after filtering to rank >= {min_rank}.")

    df_metrics = calculate_per_id_metrics(df_long)
    df_metrics_plot = df_metrics.copy()

    if metric not in df_metrics_plot.columns:
        valid = sorted([c for c in df_metrics_plot.columns if c not in {"id", "type", "data-category", "support"}])
        raise typer.BadParameter(f"Unknown metric column: {metric}. Available metric columns include: {valid}")

    _plot_three_panel(
        df_long,
        df_metrics_plot,
        metric=metric,
        output_dir=output_dir,
        show=show,
        formats=formats,
    )
    _plot_classification_heatmaps(df_long, output_dir=output_dir, show=show, formats=formats)
    _plot_confusion_matrices(df_long, output_dir=output_dir, show=show, formats=formats)

    latex_table = create_latex_classification_table(df_long)
    (output_dir / "classification_report.tex").write_text(latex_table, encoding="utf-8")

    df_metrics.to_csv(output_dir / "per_id_metrics.csv", index=False)

    _write_text_summary(df_long, df_metrics, output_dir, metric=metric)

    typer.echo(f"Loaded {len(spec.files)} CSVs from: {targets_dir}")
    typer.echo(f"Total evaluable rows: {len(df_long)}")
    typer.echo(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    seed_everything()
    app()
