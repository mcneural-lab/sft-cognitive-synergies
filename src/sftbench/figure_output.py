from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

VALID_FIGURE_FORMATS = frozenset({"png", "pdf", "svg"})

# matplotlib stamps the current time into PDF and SVG output. Suppressing it is
# what lets two runs of the same target produce byte-identical vector figures.
# PNG carries no timestamp, so it needs nothing.
_REPRODUCIBLE_METADATA = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}


def apply_bold_axis_style(linewidth: float = 1.5) -> None:
    """Thicken axis spines and tick marks so they read as substantial next to data elements.

    Default matplotlib `axes.linewidth` is 0.8pt, which looks thin against boxplot edges,
    markers, and annotations. Call once before plotting; safe to call multiple times.
    Must be called *after* any seaborn `set_context`/`set_style` to override their values.
    """
    mpl.rcParams["axes.linewidth"] = linewidth
    mpl.rcParams["xtick.major.width"] = linewidth
    mpl.rcParams["ytick.major.width"] = linewidth
    minor_width = max(linewidth * 0.6, 0.8)
    mpl.rcParams["xtick.minor.width"] = minor_width
    mpl.rcParams["ytick.minor.width"] = minor_width


def parse_figure_formats(formats: str | Iterable[str] | None, *, default: Sequence[str]) -> tuple[str, ...]:
    """Parse and validate requested matplotlib figure output formats."""
    raw_formats: Iterable[str]
    if formats is None:
        raw_formats = default
    elif isinstance(formats, str):
        raw_formats = formats.split(",") if formats.strip() else default
    else:
        raw_formats = formats

    parsed: list[str] = []
    invalid: list[str] = []
    for raw_format in raw_formats:
        figure_format = str(raw_format).strip().lower().lstrip(".")
        if not figure_format:
            continue
        if figure_format not in VALID_FIGURE_FORMATS:
            invalid.append(figure_format)
            continue
        if figure_format not in parsed:
            parsed.append(figure_format)

    if invalid:
        valid = ", ".join(sorted(VALID_FIGURE_FORMATS))
        invalid_text = ", ".join(invalid)
        raise ValueError(f"Unsupported figure format(s): {invalid_text}. Valid formats: {valid}.")

    if not parsed:
        raise ValueError("At least one figure format is required.")

    return tuple(parsed)


def save_figure_formats(
    fig: Figure | None,
    output_path: Path,
    *,
    formats: str | Iterable[str] | None,
    default: Sequence[str],
    tight: bool = False,
    dpi: int | None = 300,
    bbox_inches: str | None = None,
) -> tuple[Path, ...]:
    """Save `fig` to `output_path` using one or more validated output formats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_formats = parse_figure_formats(formats, default=default)
    if tight:
        # Some composite figures use axes tight_layout cannot handle; matplotlib
        # warns and leaves the layout alone, which is the behaviour we want.
        plt.tight_layout()

    saved_paths: list[Path] = []
    for figure_format in parsed_formats:
        path = output_path.with_suffix(f".{figure_format}")
        kwargs = {"dpi": dpi, "bbox_inches": bbox_inches, "metadata": _REPRODUCIBLE_METADATA.get(figure_format)}
        if kwargs["metadata"] is None:
            del kwargs["metadata"]
        if fig is None:
            plt.savefig(path, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        else:
            fig.savefig(path, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        saved_paths.append(path)

    return tuple(saved_paths)
