"""Shared utilities: logging, timers, memory tracking, deterministic helpers.

All measurement helpers are dependency-light so they work even when psutil is
unavailable (a graceful fallback reads /proc/self/statm on Linux).
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np

_LOGGER_NAME = "amer"


def get_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


LOG = get_logger()


def set_seed(seed: int) -> None:
    """Seed every RNG we touch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Memory / timing measurement
# --------------------------------------------------------------------------- #
def _rss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", "r") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def bytes_to_human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


class PeakMemory:
    """Context manager tracking peak RSS; exposes ``.peak`` and ``.delta``."""

    def __init__(self) -> None:
        self.start = 0
        self.peak = 0
        self.end = 0

    def __enter__(self) -> "PeakMemory":
        self.start = _rss_bytes()
        self.peak = self.start
        return self

    def _sample(self) -> None:
        cur = _rss_bytes()
        if cur > self.peak:
            self.peak = cur

    def __exit__(self, *exc: Any) -> None:
        self.end = _rss_bytes()
        self._sample()

    @property
    def delta(self) -> int:
        return max(0, self.peak - self.start)

    def report(self) -> str:
        return (
            f"peak={bytes_to_human(self.peak)} "
            f"(+{bytes_to_human(self.delta)} over {bytes_to_human(self.start)})"
        )


@contextmanager
def timed(label: str, store: Dict[str, float] | None = None, logger: logging.Logger | None = None):
    """Time a block, print it, and optionally record seconds into ``store``."""
    log = logger or LOG
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    log.info("[timer] %s: %.2fs", label, dt)
    if store is not None:
        store[label] = dt


def human_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m{s:.0f}s"


# --------------------------------------------------------------------------- #
# JSON helpers (numpy / Path safe)
# --------------------------------------------------------------------------- #
class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
