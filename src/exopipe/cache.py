"""O(1) / performance caching layer for ``exopipe``.

This module implements the "amortise repeated work to constant time" patterns
from ``ARCHITECTURE.md`` Section 9 and ``research/05_performance_architecture.md``
Section D. Every optional dependency (``joblib``) is imported **lazily** with a
pure-Python fallback, so importing this module never fails and the demo runs on
core deps alone.

The pieces and their O(1) rationale
-----------------------------------
``get_memory`` / ``cached``
    Disk-backed memoisation of *stage outputs* via :class:`joblib.Memory`
    (D.1). Re-calling an expensive idempotent stage (detrend, search) with the
    same arguments becomes a constant-time disk load instead of a recompute, so
    a code change to one stage re-runs the sector in minutes. Falls back to a
    transparent no-op decorator when joblib is absent.
``lc_hash``
    Content hash of a light curve. Stable hash of the (time, flux) bytes plus
    the identifying meta — the cache key that makes memoisation *content
    addressed* (D.1, §E5 reproducibility).
``Manifest``
    A ``dict``-backed ``tic_id -> record`` index (D.3). ``contains`` / ``add`` /
    ``get`` are O(1) regardless of whether there are 25k or 2M light curves —
    this both resolves a TIC to its file path and lets the driver skip
    already-completed work on resume (idempotent restart) in constant time.
``BloomFilter``
    A compact bit-array + k hashes answering "have we seen X?" in O(1) with zero
    false negatives (D.6). Used to gate the expensive vetting/fitting on
    constant-time membership against large known-EB / known-TOI sets, and to
    skip finished TICs on resume. Pure-Python ``bytearray`` implementation, no
    third-party dependency.
``period_grid``
    An ``lru_cache``-memoised trial-period grid (D.2/D.7). The grid depends only
    on ``(period_min, period_max, baseline, oversample)`` — compute it once and
    reuse for all light curves instead of rebuilding it per light curve.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .utils import get_logger

__all__ = [
    "get_memory",
    "cached",
    "lc_hash",
    "Manifest",
    "BloomFilter",
    "period_grid",
    "clear_memory_cache",
]

_LOG = get_logger("exopipe.cache")


# --------------------------------------------------------------------------- #
# Disk-backed memoisation -- joblib.Memory (lazy) with a no-op fallback
# --------------------------------------------------------------------------- #
_MEMORY_SINGLETON: Any | None = None
_MEMORY_LOCATION: str | None = None


def get_memory(cache_dir: str | os.PathLike[str] | None = None, verbose: int = 0) -> Any:
    """Return a process-wide :class:`joblib.Memory` for stage-output caching.

    Parameters
    ----------
    cache_dir:
        Root directory for the on-disk cache. Defaults to ``.cache/joblib`` (and
        honours :class:`~exopipe.config.PerfConfig.cache_dir` when callers pass
        ``perf.cache_dir``). The directory is created if missing.
    verbose:
        joblib verbosity (0 == silent).

    Returns
    -------
    Any
        A configured ``joblib.Memory`` instance, or a lightweight ``_NullMemory``
        shim exposing the same ``.cache`` API when joblib is not installed. The
        instance is memoised so repeated calls with the same location are O(1).

    Notes
    -----
    The O(1) win: wrapping an idempotent stage with ``memory.cache`` turns a
    recompute into a constant-time disk load keyed by a hash of the arguments
    (``research/05`` D.1). Re-running the pipeline after a downstream code change
    serves every unchanged upstream stage from cache.
    """
    global _MEMORY_SINGLETON, _MEMORY_LOCATION

    location = str(Path(cache_dir if cache_dir is not None else ".cache") / "joblib")
    if _MEMORY_SINGLETON is not None and _MEMORY_LOCATION == location:
        return _MEMORY_SINGLETON

    try:
        from joblib import Memory  # type: ignore

        Path(location).mkdir(parents=True, exist_ok=True)
        memory = Memory(location=location, verbose=verbose)
    except Exception as exc:  # pragma: no cover - joblib absent / unwritable
        _LOG.debug("joblib.Memory unavailable (%s); using no-op cache.", exc)
        memory = _NullMemory()

    _MEMORY_SINGLETON = memory
    _MEMORY_LOCATION = location
    return memory


class _NullMemory:
    """Drop-in stand-in for ``joblib.Memory`` when joblib is unavailable.

    Provides ``.cache`` (a transparent pass-through decorator) and a no-op
    ``.clear`` so calling code does not need to special-case the fallback.
    """

    location = None

    def cache(self, func: Callable | None = None, **_kw: Any) -> Callable:
        """Return ``func`` unchanged (no caching)."""
        if func is None:
            return lambda f: f
        return func

    def clear(self, warn: bool = True) -> None:  # pragma: no cover - trivial
        """No-op clear."""
        return None


def cached(func: Callable | None = None, *, cache_dir: str | None = None) -> Callable:
    """Decorator: disk-memoise ``func`` via the shared :class:`joblib.Memory`.

    Usage::

        @cached
        def detrend_cached(time_bytes, cfg_hash): ...

    Degrades to a no-op (returns the undecorated function) when joblib is not
    installed, so the call site behaves identically either way — only the speed
    of a *repeat* call changes. Constant-time cache hit vs. full recompute.
    """

    def _wrap(f: Callable) -> Callable:
        memory = get_memory(cache_dir)
        try:
            return memory.cache(f)
        except Exception:  # pragma: no cover - defensive
            return f

    if func is not None:
        return _wrap(func)
    return _wrap


def clear_memory_cache(cache_dir: str | None = None) -> None:
    """Clear the on-disk joblib cache (best effort; safe if absent)."""
    memory = get_memory(cache_dir)
    try:
        memory.clear(warn=False)
    except Exception:  # pragma: no cover - defensive
        pass


# --------------------------------------------------------------------------- #
# Content hashing of light curves -- the cache key
# --------------------------------------------------------------------------- #
def lc_hash(lc: Any, length: int = 16) -> str:
    """Return a stable content hash for a :class:`~exopipe.types.LightCurve`.

    The hash is computed from the raw bytes of ``time`` and ``flux`` plus the
    identifying metadata (``tic_id``, ``sector``). The same light curve always
    hashes to the same string (reproducibility, ``research/05`` §E5), making it a
    safe **content-addressed** cache key for stage memoisation and manifests.

    Parameters
    ----------
    lc:
        A ``LightCurve`` (duck-typed: needs ``time``/``flux`` arrays and an
        optional ``meta`` dict). Plain arrays are also accepted.
    length:
        Number of leading hex characters to return (default 16 == 64 bits, ample
        for collision avoidance across a sector).

    Returns
    -------
    str
        A short, stable hex digest.
    """
    hasher = hashlib.blake2b(digest_size=16)

    time = getattr(lc, "time", None)
    flux = getattr(lc, "flux", None)
    meta = getattr(lc, "meta", None)

    if time is None and flux is None:
        # treat ``lc`` itself as an array-like
        arr = np.ascontiguousarray(np.asarray(lc))
        hasher.update(arr.tobytes())
    else:
        if time is not None:
            hasher.update(np.ascontiguousarray(np.asarray(time, dtype=np.float64)).tobytes())
        if flux is not None:
            hasher.update(np.ascontiguousarray(np.asarray(flux, dtype=np.float32)).tobytes())

    if isinstance(meta, dict):
        ident = (meta.get("tic_id"), meta.get("sector"), meta.get("mission"))
        hasher.update(repr(ident).encode("utf-8"))

    return hasher.hexdigest()[:length]


# --------------------------------------------------------------------------- #
# Manifest -- O(1) TIC -> record index (and skip-completed gate)
# --------------------------------------------------------------------------- #
@dataclass
class Manifest:
    """A constant-time ``tic_id -> record`` index over processed light curves.

    Backed by a plain Python ``dict`` so :meth:`contains`, :meth:`add`, and
    :meth:`get` are **O(1)** no matter how many entries it holds — the
    ingest/results manifest pattern from ``research/05`` D.3. Two jobs:

    1. **Resolution.** Map a TIC to its on-disk path / byte offset so every
       stage resolves a light curve in constant time instead of scanning a
       directory or a giant table.
    2. **Idempotent resume.** A driver checks ``contains(tic_id)`` to skip work
       that is already done — turning "skip already-completed" into an O(1)
       lookup on restart (``research/05`` §E4).

    The manifest serialises to a small JSON file (``save`` / ``load``), so it is
    cheap to persist between runs.

    Attributes
    ----------
    records:
        ``str(tic_id) -> dict`` mapping. Keys are stringified so the structure is
        JSON-round-trippable; :meth:`get` / :meth:`contains` accept int or str.
    """

    records: dict[str, dict] = field(default_factory=dict)

    # -- core O(1) operations ----------------------------------------------- #
    @staticmethod
    def _key(tic_id: Any) -> str:
        return str(tic_id)

    def contains(self, tic_id: Any) -> bool:
        """``True`` if ``tic_id`` is recorded. O(1) dict membership."""
        return self._key(tic_id) in self.records

    def __contains__(self, tic_id: Any) -> bool:
        return self.contains(tic_id)

    def add(self, tic_id: Any, **record: Any) -> None:
        """Insert/overwrite the record for ``tic_id``. O(1)."""
        self.records[self._key(tic_id)] = dict(record)

    def get(self, tic_id: Any, default: Any = None) -> Any:
        """Return the record for ``tic_id`` (or ``default``). O(1)."""
        return self.records.get(self._key(tic_id), default)

    def update_many(self, items: Iterable[tuple[Any, dict]]) -> None:
        """Bulk-insert ``(tic_id, record)`` pairs."""
        for tic_id, record in items:
            self.add(tic_id, **dict(record))

    def __len__(self) -> int:
        return len(self.records)

    def tic_ids(self) -> list[str]:
        """Return all recorded TIC keys."""
        return list(self.records.keys())

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the manifest to ``path`` as JSON (parent dirs created)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.records, handle, default=_json_default)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Manifest":
        """Load a manifest from JSON ``path``; returns empty if absent/corrupt."""
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            with open(path, encoding="utf-8") as handle:
                records = json.load(handle)
            if not isinstance(records, dict):
                return cls()
            return cls(records={str(k): dict(v) for k, v in records.items()})
        except Exception as exc:  # pragma: no cover - corrupt file
            _LOG.warning("Manifest load failed for %s (%s); starting empty.", path, exc)
            return cls()


def _json_default(obj: Any) -> Any:
    """JSON encoder hook for numpy scalars/arrays inside manifest records."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return str(obj)


# --------------------------------------------------------------------------- #
# Bloom filter -- O(1) membership for large known-EB / known-TOI sets
# --------------------------------------------------------------------------- #
class BloomFilter:
    """A space-efficient probabilistic set with O(1) membership tests.

    A Bloom filter answers "have we seen X?" in constant time using a bit array
    and ``k`` hash functions (``research/05`` D.6). It guarantees **no false
    negatives** (a "no" is definitive); a "yes" may occasionally be a false
    positive at a tunable rate, in which case the caller does the exact check
    against the backing manifest/catalog.

    Where it is used in ``exopipe``:

    * **Known-EB / known-TOI gate.** Load a large catalogue of known eclipsing
      binaries or TESS Objects of Interest into the filter at startup; gate the
      expensive vetting/fitting on a constant-time ``contains`` test instead of a
      linear catalogue scan.
    * **Skip-completed on resume.** A filter of already-processed TIC IDs lets
      the driver skip finished work instantly on restart, backed by the exact
      :class:`Manifest` for the rare false positive.

    Pure ``bytearray`` implementation — no third-party dependency — so it works
    with core deps only.

    Parameters
    ----------
    capacity:
        Expected number of items. The bit-array size is chosen to hold this many
        with the target false-positive rate.
    error_rate:
        Desired false-positive probability at ``capacity`` (default 1%).
    """

    def __init__(self, capacity: int = 10_000, error_rate: float = 0.01) -> None:
        capacity = max(int(capacity), 1)
        error_rate = float(min(max(error_rate, 1e-9), 0.5))
        self.capacity = capacity
        self.error_rate = error_rate

        # Optimal bit count m = -n ln p / (ln 2)^2 ; hash count k = (m/n) ln 2.
        m = math.ceil(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.n_bits = int(max(m, 8))
        self.n_hashes = int(max(round((self.n_bits / capacity) * math.log(2)), 1))
        self._bytes = bytearray((self.n_bits + 7) // 8)
        self._count = 0

    # -- hashing ------------------------------------------------------------ #
    def _positions(self, item: Any) -> list[int]:
        """Derive ``k`` bit positions from two base hashes (Kirsch–Mitzenmacher).

        Two independent 64-bit hashes are combined as ``h1 + i*h2`` to synthesise
        ``k`` hashes cheaply without computing ``k`` full digests.
        """
        data = str(item).encode("utf-8")
        h1 = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")
        h2 = int.from_bytes(hashlib.blake2b(data, digest_size=8, salt=b"exopipe!").digest(), "little")
        h2 |= 1  # ensure odd so the sequence visits distinct slots
        return [(h1 + i * h2) % self.n_bits for i in range(self.n_hashes)]

    @staticmethod
    def _set_bit(buf: bytearray, pos: int) -> None:
        buf[pos >> 3] |= 1 << (pos & 7)

    @staticmethod
    def _get_bit(buf: bytearray, pos: int) -> bool:
        return bool(buf[pos >> 3] & (1 << (pos & 7)))

    # -- public API --------------------------------------------------------- #
    def add(self, item: Any) -> None:
        """Insert ``item`` into the filter. O(k) ~ O(1)."""
        for pos in self._positions(item):
            self._set_bit(self._bytes, pos)
        self._count += 1

    def update(self, items: Iterable[Any]) -> None:
        """Insert many items."""
        for item in items:
            self.add(item)

    def contains(self, item: Any) -> bool:
        """Test membership. Constant time; ``False`` is always correct."""
        return all(self._get_bit(self._bytes, pos) for pos in self._positions(item))

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def __len__(self) -> int:
        """Number of items added (not the bit count)."""
        return self._count

    @classmethod
    def from_iterable(
        cls, items: Iterable[Any], error_rate: float = 0.01
    ) -> "BloomFilter":
        """Build a right-sized filter pre-loaded with ``items``."""
        items = list(items)
        bloom = cls(capacity=max(len(items), 1), error_rate=error_rate)
        bloom.update(items)
        return bloom


# --------------------------------------------------------------------------- #
# Precomputed period grid -- lru_cache (build once, reuse for every LC)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=256)
def period_grid(
    period_min: float,
    period_max: float,
    baseline: float,
    oversample: int = 5,
) -> np.ndarray:
    """Return a trial-period grid, memoised by its defining parameters.

    The grid depends only on ``(period_min, period_max, baseline, oversample)``,
    so it is computed **once** and reused for all light curves in a sector
    instead of being rebuilt per light curve (``research/05`` D.2/D.7). The
    spacing follows the standard frequency-space prescription where the period
    step scales as ``P^2 / baseline`` (constant fractional frequency resolution),
    which is the physically correct sampling for a transit search.

    Parameters
    ----------
    period_min, period_max:
        Bounds of the search in days. ``period_max`` is clamped to the baseline.
    baseline:
        Total time span of the data in days (sets the frequency resolution).
    oversample:
        Frequency-grid oversampling factor (higher == finer grid, linear cost).

    Returns
    -------
    numpy.ndarray
        Monotonically increasing ``float64`` array of trial periods.

    Notes
    -----
    Because :func:`functools.lru_cache` caches by argument identity, repeated
    calls with the same parameters return the *same* cached array in O(1) with no
    recomputation. Arguments must be hashable scalars (they are).
    """
    period_min = float(period_min)
    period_max = float(period_max)
    baseline = float(baseline)
    oversample = max(int(oversample), 1)

    if not np.isfinite(baseline) or baseline <= 0:
        baseline = max(period_max, period_min, 1.0)
    # A period longer than the baseline yields < 1 cycle -> not searchable.
    period_max = min(period_max, baseline)
    if period_max <= period_min:
        period_max = period_min * 1.5 + 1e-6

    # Frequency-space grid with df = 1 / (oversample * baseline).
    f_min = 1.0 / period_max
    f_max = 1.0 / period_min
    df = 1.0 / (oversample * baseline)
    n = int(max(math.ceil((f_max - f_min) / df), 2))
    freqs = f_min + df * np.arange(n, dtype=np.float64)
    freqs = freqs[freqs > 0]
    periods = 1.0 / freqs
    periods.sort()
    return periods
