"""
Plotting utilities for micro-level prediction experiments.

This package is intended to keep plotting code consistent across CLIs and
analysis scripts (e.g., seaborn theme configuration loaded from
`configs/plotting.toml`, figure styling, etc.).

General guidance:
- Put reusable, dependency-light helpers here so other plotting modules can do:
  `from sftbench.micro.prediction.plotting import apply_seaborn_theme_from_config, savefig, ...`

Primary entrypoints:
- `sftbench.micro.prediction.plotting`: shared plotting helpers (this module)
- `sftbench.micro.prediction.plotting.new_plot_prediction_results`: SI model and prompting plots
- `sftbench.micro.prediction.plotting.plot_human_vs_ai_next-exemplar_prediction-results`: Figure 3-style plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .config import (  # noqa: E402
    SeabornThemeConfig,
    apply_seaborn_theme_from_config,
    load_plotting_toml,
    seaborn_theme_from_toml,
)

__all__ = [
    # seaborn/theme config
    "SeabornThemeConfig",
    "apply_seaborn_theme_from_config",
    "load_plotting_toml",
    "seaborn_theme_from_toml",
    # general plotting helpers
    "ensure_output_dir",
    "parse_figsize",
    "maybe_rotate_xticks",
    "savefig",
]


def ensure_output_dir(output_dir: Path) -> None:
    """Create `output_dir` (and parents) if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_figsize(figsize: str, *, default: tuple[float, float] = (9.0, 6.0)) -> tuple[float, float]:
    """
    Parse a figure size string formatted as 'W,H' (inches).

    Returns `default` if parsing fails.
    """
    try:
        w_str, h_str = [p.strip() for p in figsize.split(",")]
        return float(w_str), float(h_str)
    except Exception:
        return default


def maybe_rotate_xticks(ax: Any, rotate: int) -> None:
    """Rotate x tick labels by `rotate` degrees (no-op if rotate is falsy)."""
    if not rotate:
        return
    for label in ax.get_xticklabels():
        label.set_rotation(rotate)
        label.set_horizontalalignment("right")


def savefig(
    fig: Any | None,
    output_path: Path,
    *,
    dpi: int | None = None,
    tight: bool = True,
) -> None:
    """
    Save a figure to disk, ensuring the parent directory exists.

    If `fig` is None, saves the current matplotlib figure.
    """
    output_path = Path(output_path)
    ensure_output_dir(output_path.parent)
    if tight:
        try:
            plt.tight_layout()
        except Exception:
            pass
    target = fig if fig is not None else plt
    save_kwargs: dict[str, Any] = {}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    target.savefig(output_path, **save_kwargs)
