from __future__ import annotations

import random
import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

import matplotlib as mpl
import numpy as np
import rich.progress

T = TypeVar("T")

SEED = 0


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mpl.rcParams["svg.hashsalt"] = str(seed)


def track(sequence: Iterable[T], description: str) -> Iterator[T]:
    return rich.progress.track(sequence, description=description, disable=not sys.stdout.isatty())
