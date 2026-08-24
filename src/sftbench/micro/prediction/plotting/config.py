"""
Reusable plotting configuration utilities.

This module centralizes the plotting configuration shared by the release
plotting CLIs.

Currently supported config keys:

- [seaborn]
  - context: str (default "paper")
  - style: str (default "whitegrid")
  - font_scale: float (default 1.5)

Example `plotting.toml`:

    [seaborn]
    context = "paper"
    font_scale = 1.5
    style = "whitegrid"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import seaborn as sns

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class SeabornThemeConfig:
    context: str = "paper"
    style: str = "whitegrid"
    font_scale: float = 1.5


def _coerce_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_plotting_toml(config_path: Path) -> dict[str, Any]:
    """
    Load a TOML plotting config from disk.

    Returns an empty dict if the file doesn't exist.
    """
    if not config_path.exists():
        return {}
    with config_path.open("rb") as f:
        return tomllib.load(f)


def seaborn_theme_from_toml(config: dict[str, Any]) -> SeabornThemeConfig:
    """
    Convert a parsed TOML dict into a `SeabornThemeConfig`, applying defaults.
    """
    seaborn_cfg = config.get("seaborn", {}) if isinstance(config, dict) else {}
    context = seaborn_cfg.get("context", "paper")
    style = seaborn_cfg.get("style", "whitegrid")
    font_scale = _coerce_float(seaborn_cfg.get("font_scale", 1.5), default=1.5)
    return SeabornThemeConfig(context=str(context), style=str(style), font_scale=font_scale)


def apply_seaborn_theme_from_config(config_path: Path, *, echo: bool = False) -> SeabornThemeConfig:
    """
    Load plotting config from `config_path` and apply seaborn theme.

    If the config file doesn't exist, defaults are applied.

    Returns the applied `SeabornThemeConfig`.
    """
    if not config_path.exists():
        theme = SeabornThemeConfig()
        sns.set_theme(context=theme.context, style=theme.style, font_scale=theme.font_scale)
        if echo:
            print(f"Config file not found: {config_path}. Using defaults.")
        return theme

    cfg = load_plotting_toml(config_path)
    theme = seaborn_theme_from_toml(cfg)
    sns.set_theme(context=theme.context, style=theme.style, font_scale=theme.font_scale)
    return theme
