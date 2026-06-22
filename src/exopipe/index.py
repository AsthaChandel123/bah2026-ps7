"""Approximate-nearest-neighbour index over light-curve shape embeddings.

Implements the "O(1)-ish shape lookup" pattern from ``ARCHITECTURE.md`` Section 9
(row 7) and ``research/05_performance_architecture.md`` Section D.5. Given a
fixed-length **embedding** of each light curve (e.g. a phase-folded binned view,
a BLS/TLS feature vector, or a learned embedding), the index answers
*"what known signals look like this?"* in sub-linear — effectively constant —
time instead of comparing against every reference.

Three concrete uses in ``exopipe`` (research/05 D.5):

1. **Known-signal matching / triage** — match a candidate's shape against a
   library of confirmed planets / EBs / blends to seed the classifier or flag
   obvious EBs.
2. **Deduplication** — the same astrophysical signal can appear on neighbouring
   TICs via blending; ANN finds near-duplicate shapes so a blend is not reported
   N times.
3. **Few-shot label propagation** — nearest labelled neighbours vote on an
   unlabelled candidate.

Backend selection is graceful and fully lazy:

``hnswlib`` (HNSW graph, best recall/speed) → ``faiss`` (with optional GPU) →
``sklearn.neighbors.NearestNeighbors`` (exact KD/ball tree) → pure-NumPy brute
force. Whichever is importable wins; the public API
(:meth:`ShapeIndex.add` / :meth:`ShapeIndex.build` / :meth:`ShapeIndex.query`) is
identical across all four, so callers never branch on the backend. With only
core deps installed the NumPy brute-force path is used (still correct, just
O(N·d) per query rather than ~O(log N)).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .utils import get_logger

__all__ = ["ShapeIndex"]

_LOG = get_logger("exopipe.index")


class ShapeIndex:
    """Nearest-neighbour index over fixed-length light-curve embeddings.

    Parameters
    ----------
    dim:
        Embedding dimensionality. All vectors added/queried must have this
        length (they are coerced to ``float32`` and length-checked).
    metric:
        ``"l2"`` (Euclidean, default) or ``"cosine"``. Cosine is implemented by
        L2-normalising vectors and using Euclidean/inner-product geometry, so it
        works across every backend.
    backend:
        Force a backend (``"hnswlib"``, ``"faiss"``, ``"sklearn"``, ``"numpy"``)
        or ``"auto"`` (default) to pick the best available.
    max_elements:
        Hint for the maximum number of vectors (used to size the HNSW index).

    Notes
    -----
    Typical lifecycle: :meth:`add` references once, :meth:`build`, then
    :meth:`query` forever. Building the index is the one-off cost; each query is
    then ~constant time (research/05 D.5).
    """

    _BACKENDS = ("hnswlib", "faiss", "sklearn", "numpy")

    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        backend: str = "auto",
        max_elements: int = 100_000,
    ) -> None:
        if int(dim) <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = int(dim)
        self.metric = str(metric).lower()
        if self.metric not in ("l2", "cosine"):
            raise ValueError("metric must be 'l2' or 'cosine'")
        self.max_elements = int(max_elements)

        self._ids: list[Any] = []
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None  # built (N, dim) float32 matrix
        self._built = False

        self._backend_name = self._resolve_backend(backend)
        self._impl: Any = None  # backend-specific handle (hnsw index / faiss / sklearn)
        _LOG.debug("ShapeIndex backend = %s (dim=%d, metric=%s)", self._backend_name, self.dim, self.metric)

    # -- backend resolution -------------------------------------------------- #
    @property
    def backend(self) -> str:
        """Name of the active backend (after auto-resolution)."""
        return self._backend_name

    def _resolve_backend(self, requested: str) -> str:
        requested = (requested or "auto").lower()
        if requested != "auto":
            if requested not in self._BACKENDS:
                raise ValueError(f"backend must be one of {self._BACKENDS} or 'auto'")
            return requested
        for name in ("hnswlib", "faiss", "sklearn"):
            if self._backend_available(name):
                return name
        return "numpy"

    @staticmethod
    def _backend_available(name: str) -> bool:
        import importlib.util

        modmap = {"hnswlib": "hnswlib", "faiss": "faiss", "sklearn": "sklearn"}
        mod = modmap.get(name)
        if mod is None:
            return False
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:  # pragma: no cover - defensive
            return False

    # -- vector prep -------------------------------------------------------- #
    def _prep(self, vec: Sequence[float] | np.ndarray) -> np.ndarray:
        """Coerce a single vector to canonical ``float32`` shape ``(dim,)``."""
        arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float32)).ravel()
        if arr.size != self.dim:
            raise ValueError(f"embedding has length {arr.size}, expected {self.dim}")
        if self.metric == "cosine":
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
        return arr

    # -- public: add -------------------------------------------------------- #
    def add(self, item_id: Any, vec: Sequence[float] | np.ndarray) -> None:
        """Add one reference embedding ``vec`` under label ``item_id``.

        Adding invalidates a previously-built index; call :meth:`build` again
        (or just call :meth:`query`, which auto-builds) before querying.
        """
        self._ids.append(item_id)
        self._vectors.append(self._prep(vec))
        self._built = False

    def add_many(
        self,
        ids: Sequence[Any],
        vecs: np.ndarray | Sequence[Sequence[float]],
    ) -> None:
        """Add a batch of embeddings (``ids`` aligned with rows of ``vecs``)."""
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2:
            raise ValueError("vecs must be a 2-D array (n_items, dim)")
        if len(ids) != vecs.shape[0]:
            raise ValueError("ids and vecs must have the same length")
        for item_id, row in zip(ids, vecs):
            self.add(item_id, row)

    def __len__(self) -> int:
        return len(self._ids)

    # -- public: build ------------------------------------------------------ #
    def build(self) -> "ShapeIndex":
        """Construct the backend index from all added vectors.

        Idempotent and cheap to re-call. Returns ``self`` for chaining. With no
        vectors added this is a no-op that leaves :meth:`query` returning empty
        results.
        """
        if not self._vectors:
            self._matrix = np.empty((0, self.dim), dtype=np.float32)
            self._built = True
            self._impl = None
            return self

        self._matrix = np.ascontiguousarray(np.vstack(self._vectors).astype(np.float32))

        try:
            if self._backend_name == "hnswlib":
                self._build_hnswlib()
            elif self._backend_name == "faiss":
                self._build_faiss()
            elif self._backend_name == "sklearn":
                self._build_sklearn()
            else:
                self._impl = None  # numpy brute force uses self._matrix directly
        except Exception as exc:  # pragma: no cover - backend hiccup -> degrade
            _LOG.warning(
                "ShapeIndex backend %r failed to build (%s); falling back to numpy brute force.",
                self._backend_name,
                exc,
            )
            self._backend_name = "numpy"
            self._impl = None

        self._built = True
        return self

    def _build_hnswlib(self) -> None:
        import hnswlib  # type: ignore

        space = "cosine" if self.metric == "cosine" else "l2"
        n = self._matrix.shape[0]
        index = hnswlib.Index(space=space, dim=self.dim)
        index.init_index(
            max_elements=max(self.max_elements, n),
            ef_construction=200,
            M=16,
        )
        index.add_items(self._matrix, np.arange(n))
        index.set_ef(max(64, 2))
        self._impl = index

    def _build_faiss(self) -> None:
        import faiss  # type: ignore

        if self.metric == "cosine":
            index = faiss.IndexFlatIP(self.dim)  # vectors already L2-normalised
        else:
            index = faiss.IndexFlatL2(self.dim)
        index.add(self._matrix)
        self._impl = index

    def _build_sklearn(self) -> None:
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        # Cosine vectors are L2-normalised, so Euclidean nn ordering matches.
        nn = NearestNeighbors(metric="euclidean")
        nn.fit(self._matrix)
        self._impl = nn

    # -- public: query ------------------------------------------------------ #
    def query(
        self,
        vec: Sequence[float] | np.ndarray,
        k: int = 5,
    ) -> list[tuple[Any, float]]:
        """Return the ``k`` nearest references to ``vec``.

        Parameters
        ----------
        vec:
            Query embedding (length ``dim``).
        k:
            Number of neighbours to return.

        Returns
        -------
        list[tuple[Any, float]]
            ``(item_id, distance)`` pairs sorted by ascending distance (nearest
            first). Distance is squared-L2 (or ``1 - cosine_similarity`` for the
            cosine metric, depending on backend) — comparable *within* a single
            index, which is all the callers (dedup / matching / voting) need.
        """
        if not self._built:
            self.build()
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []

        q = self._prep(vec)
        k = int(max(min(k, len(self._ids)), 1))

        if self._backend_name == "hnswlib" and self._impl is not None:
            labels, dists = self._impl.knn_query(q.reshape(1, -1), k=k)
            idxs = np.asarray(labels).ravel()
            dvals = np.asarray(dists, dtype=float).ravel()
        elif self._backend_name == "faiss" and self._impl is not None:
            dvals, labels = self._impl.search(q.reshape(1, -1), k)
            idxs = np.asarray(labels).ravel()
            dvals = np.asarray(dvals, dtype=float).ravel()
            if self.metric == "cosine":  # IP similarity -> distance for sorting
                dvals = 1.0 - dvals
        elif self._backend_name == "sklearn" and self._impl is not None:
            dvals, labels = self._impl.kneighbors(q.reshape(1, -1), n_neighbors=k)
            idxs = np.asarray(labels).ravel()
            dvals = np.asarray(dvals, dtype=float).ravel() ** 2  # match L2^2 scale
        else:
            idxs, dvals = self._query_numpy(q, k)

        out: list[tuple[Any, float]] = []
        for idx, dist in zip(idxs, dvals):
            if idx < 0 or idx >= len(self._ids):  # faiss pads with -1 when k>N
                continue
            out.append((self._ids[int(idx)], float(dist)))
        # Backends mostly return sorted, but normalise to be safe.
        out.sort(key=lambda pair: pair[1])
        return out

    def _query_numpy(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Pure-NumPy brute-force k-NN (squared-L2). O(N·d) per query.

        Uses ``argpartition`` so selecting the top-k is linear in N rather than a
        full sort — the cheapest correct fallback when no ANN backend exists.
        """
        diff = self._matrix - q[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)  # squared L2, vectorised
        n = d2.shape[0]
        if k >= n:
            order = np.argsort(d2)
        else:
            part = np.argpartition(d2, k)[:k]
            order = part[np.argsort(d2[part])]
        return order, d2[order]
