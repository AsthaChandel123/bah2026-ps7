"""Lightweight, dependency-light utilities shared across the pipeline.

Logging, timing, seeding, and a small set of NaN-safe numerical helpers. Only
``numpy``/``scipy`` are required; nothing here imports the optional science or
ML stacks.
"""

from __future__ import annotations

import logging
import sys
import time as _time
from types import TracebackType

import numpy as np

__all__ = [
    "get_logger",
    "Timer",
    "set_seed",
    "nanmad",
    "robust_std",
    "phase_fold",
    "running_median",
]

# 1 / Phi^{-1}(3/4): scales the median absolute deviation to a Gaussian sigma.
_MAD_TO_STD = 1.4826

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "exopipe", level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stderr exactly once.

    Idempotent: repeated calls with the same name do not stack handlers, so
    importing this from many modules is safe.
    """
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    # Only attach our handler once; respect any handler the host app added.
    if not any(getattr(h, "_exopipe_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        handler._exopipe_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
class Timer:
    """Context manager (and reusable object) that measures wall-clock time.

    Examples
    --------
    >>> with Timer("search") as t:        # doctest: +SKIP
    ...     run_search()
    >>> t.elapsed                          # doctest: +SKIP
    1.234
    """

    def __init__(self, label: str = "", logger: logging.Logger | None = None) -> None:
        self.label = label
        self.logger = logger
        self.start: float | None = None
        self.end: float | None = None
        self.elapsed: float = float("nan")

    def __enter__(self) -> Timer:
        self.start = _time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.end = _time.perf_counter()
        self.elapsed = self.end - (self.start if self.start is not None else self.end)
        message = f"{self.label or 'block'} took {self.elapsed:.3f}s"
        if self.logger is not None:
            self.logger.info(message)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Timer(label={self.label!r}, elapsed={self.elapsed:.3f})"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int | None) -> np.random.Generator:
    """Seed Python's ``random`` and NumPy's legacy global RNG.

    Returns a fresh :class:`numpy.random.Generator` for code that prefers the
    modern API (the recommended way to get reproducible draws). Also attempts to
    seed ``torch`` if it happens to be importable, but never requires it.
    """
    import random as _random

    if seed is not None:
        seed = int(seed)
        _random.seed(seed)
        np.random.seed(seed)
        try:  # optional: only if the DL extra is installed
            import torch  # type: ignore

            torch.manual_seed(seed)
            if torch.cuda.is_available():  # pragma: no cover - hardware dependent
                torch.cuda.manual_seed_all(seed)
        except Exception:  # pragma: no cover - torch absent is fine
            pass
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Robust statistics (NaN-safe)
# --------------------------------------------------------------------------- #
def nanmad(x: np.ndarray, axis: int | None = None, scale: float = 1.0) -> np.ndarray | float:
    """NaN-safe median absolute deviation about the median.

    ``mad = median(|x - median(x)|)`` ignoring NaNs. Multiply the result by
    ``scale`` (use ``1.4826`` to approximate a Gaussian standard deviation, or
    call :func:`robust_std`).
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis)
    return mad * scale


def robust_std(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Robust standard-deviation estimate via ``1.4826 * MAD`` (NaN-safe)."""
    return nanmad(x, axis=axis, scale=_MAD_TO_STD)


# --------------------------------------------------------------------------- #
# Phase folding
# --------------------------------------------------------------------------- #
def phase_fold(time: np.ndarray, period: float, t0: float = 0.0) -> np.ndarray:
    """Fold ``time`` onto ``period`` returning phase in ``[-0.5, 0.5)``.

    The transit/event centre maps to phase ``0``. Unlike
    :meth:`LightCurve.fold` this does *not* sort, so the returned array is
    element-aligned with the input ``time`` (and therefore with ``flux``).
    """
    period = float(period)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be a positive, finite number")
    time = np.asarray(time, dtype=np.float64)
    return (((time - t0) / period + 0.5) % 1.0) - 0.5


# --------------------------------------------------------------------------- #
# Running median
# --------------------------------------------------------------------------- #
def running_median(x: np.ndarray, w: int) -> np.ndarray:
    """Sliding-window median with a centred, edge-reflected window.

    Parameters
    ----------
    x:
        Input 1-D array (NaNs allowed; they are ignored within each window).
    w:
        Window length in samples. Even values are bumped up to the next odd
        number so the window stays centred. ``w <= 1`` returns a copy.

    Notes
    -----
    Returned dtype is ``float64`` and the output has the same length as ``x``.
    This is a readable, vectorised reference implementation; the high-throughput
    path uses ``bottleneck.move_median`` (see ``perf`` extra) when available.
    """
    x = np.asarray(x, dtype=np.float64)
    w = int(w)
    if w <= 1 or x.size == 0:
        return x.copy()
    if w % 2 == 0:
        w += 1
    half = w // 2
    # Reflect the edges so the window is full-width everywhere.
    padded = np.pad(x, half, mode="reflect")
    # Strided sliding window view -> shape (len(x), w); median along the window.
    windows = np.lib.stride_tricks.sliding_window_view(padded, w)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(windows, axis=1)
