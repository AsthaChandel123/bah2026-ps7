"""Optional AstroNet-style 1-D CNN view classifier (lazy ``torch``).

:class:`CNNClassifier` implements a lightweight version of the
Shallue & Vanderburg (2018) / Yu et al. (2019) architecture extended to four
classes: two disjoint 1-D convolutional columns -- one over the **global** view
(2001 bins, full phase-folded orbit) and one over the **local** view (201 bins,
the transit zoom) -- whose flattened outputs are concatenated and passed through
dense layers to a 4-way softmax.

The module is **always importable**: ``torch`` is imported lazily inside the
methods. When it is missing, :attr:`CNNClassifier.available` is ``False`` and
the ensemble simply skips this stream. The views consumed here are produced by
:func:`exopipe.features.build_views` (keys ``'global'``, ``'local'``,
``'secondary'``, ``'odd'``, ``'even'``); only ``'global'`` and ``'local'`` are
required by this implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

from ..types import Classification
from .rules import CLASSES

__all__ = ["CNNClassifier", "torch_available"]

logger = logging.getLogger(__name__)

# Default view sizes (match :func:`exopipe.features.build_views` defaults).
GLOBAL_BINS = 2001
LOCAL_BINS = 201


def torch_available() -> bool:
    """Return ``True`` iff :mod:`torch` can be imported."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _build_module(global_bins: int, local_bins: int, n_classes: int) -> Any:
    """Construct the two-column CNN as a ``torch.nn.Module`` (lazy import)."""
    import torch
    from torch import nn

    class _AstroNet(nn.Module):
        """Two disjoint conv columns (global + local) -> concat -> dense softmax."""

        def __init__(self) -> None:
            super().__init__()

            def conv_column(n_blocks: int, init_filters: int, pool: int) -> nn.Sequential:
                layers: list[nn.Module] = []
                in_ch = 1
                f = init_filters
                for _ in range(n_blocks):
                    layers.append(nn.Conv1d(in_ch, f, kernel_size=5, padding=2))
                    layers.append(nn.ReLU())
                    layers.append(nn.Conv1d(f, f, kernel_size=5, padding=2))
                    layers.append(nn.ReLU())
                    layers.append(nn.MaxPool1d(kernel_size=pool, stride=2))
                    in_ch = f
                    f *= 2
                return nn.Sequential(*layers)

            # Lighter than the paper (fewer blocks/filters) to stay CPU-friendly.
            self.global_col = conv_column(n_blocks=4, init_filters=16, pool=5)
            self.local_col = conv_column(n_blocks=2, init_filters=16, pool=7)

            # Infer the flattened size with a dry run.
            with torch.no_grad():
                g = torch.zeros(1, 1, global_bins)
                l = torch.zeros(1, 1, local_bins)
                gflat = self.global_col(g).flatten(1).shape[1]
                lflat = self.local_col(l).flatten(1).shape[1]

            self.head = nn.Sequential(
                nn.Linear(gflat + lflat, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_classes),
            )

        def forward(self, g: Any, l: Any) -> Any:  # noqa: D401 - torch forward
            gf = self.global_col(g).flatten(1)
            lf = self.local_col(l).flatten(1)
            return self.head(torch.cat([gf, lf], dim=1))

    return _AstroNet()


class CNNClassifier:
    """AstroNet-style multi-branch CNN over phase-folded views (optional).

    Parameters
    ----------
    global_bins, local_bins:
        Expected lengths of the global and local views (defaults match
        :func:`exopipe.features.build_views`). Views of other lengths are
        resampled.
    epochs, batch_size, lr:
        Training hyper-parameters (kept small for a lightweight CPU fit).
    device:
        ``'cpu'`` / ``'cuda'`` / ``None`` (auto). Only used when ``torch`` is
        present.

    Attributes
    ----------
    available:
        ``True`` iff :mod:`torch` is importable. When ``False`` the object is
        inert: :meth:`fit` warns and :meth:`predict` returns a uniform
        distribution so the ensemble can detect and skip it.
    """

    def __init__(
        self,
        global_bins: int = GLOBAL_BINS,
        local_bins: int = LOCAL_BINS,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.global_bins = int(global_bins)
        self.local_bins = int(local_bins)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.available = torch_available()
        self._device_spec = device
        self._model: Any = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # View preparation
    # ------------------------------------------------------------------ #
    def _prep_view(self, view: Any, target_len: int) -> np.ndarray:
        """Clean + (if needed) resample a single 1-D view to ``target_len``.

        NaNs are filled with the view median (then 0). The view is then linearly
        interpolated onto ``target_len`` points if its length differs.
        """
        arr = np.asarray(view, dtype=np.float64).ravel()
        if arr.size == 0:
            return np.zeros(target_len, dtype=np.float32)
        finite = np.isfinite(arr)
        if not finite.all():
            fill = np.nanmedian(arr[finite]) if finite.any() else 0.0
            arr = np.where(finite, arr, fill)
        if arr.size != target_len:
            xp = np.linspace(0.0, 1.0, arr.size)
            xq = np.linspace(0.0, 1.0, target_len)
            arr = np.interp(xq, xp, arr)
        return arr.astype(np.float32)

    def _stack(self, views_list: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Stack a list of view dicts into ``(global, local)`` 3-D tensors."""
        g = np.stack(
            [self._prep_view(v.get("global"), self.global_bins) for v in views_list]
        )
        l = np.stack(
            [self._prep_view(v.get("local"), self.local_bins) for v in views_list]
        )
        return g[:, None, :], l[:, None, :]  # add channel dim

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        views_list: Sequence[Mapping[str, Any]],
        y: Sequence[Any],
    ) -> "CNNClassifier":
        """Train the CNN on a list of view dicts and their labels.

        ``y`` accepts canonical class strings or integer ids. No-op (with a
        warning) when ``torch`` is unavailable.
        """
        if not self.available:
            import warnings

            warnings.warn(
                "torch unavailable: CNNClassifier.fit() is a no-op; the ensemble "
                "will skip the CNN stream.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self

        import torch
        from torch import nn

        device = torch.device(
            self._device_spec
            if self._device_spec is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        g, l = self._stack(views_list)
        yint = np.asarray([_label_to_int(label) for label in y], dtype=np.int64)

        gt = torch.tensor(g, dtype=torch.float32, device=device)
        lt = torch.tensor(l, dtype=torch.float32, device=device)
        yt = torch.tensor(yint, dtype=torch.long, device=device)

        self._model = _build_module(self.global_bins, self.local_bins, len(CLASSES)).to(device)

        # Class-balanced loss to counter imbalance.
        counts = np.bincount(yint, minlength=len(CLASSES)).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / (len(CLASSES) * counts), 0.0)
        weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_t)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        n = gt.shape[0]
        self._model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=device)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                logits = self._model(gt[idx], lt[idx])
                loss = criterion(logits, yt[idx])
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * idx.numel()
            logger.debug("CNN epoch %d/%d loss=%.4f", epoch + 1, self.epochs, total_loss / n)
        self._model.eval()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict_proba(self, views: Mapping[str, Any]) -> np.ndarray:
        """Return a length-4 probability vector over :data:`CLASSES`.

        Uniform when ``torch`` is missing or the model is unfitted, so callers
        can detect "no signal" and skip the stream.
        """
        if not self.available or not self._fitted or self._model is None:
            return np.full(len(CLASSES), 1.0 / len(CLASSES), dtype=np.float64)

        import torch

        g, l = self._stack([views])
        device = next(self._model.parameters()).device
        gt = torch.tensor(g, dtype=torch.float32, device=device)
        lt = torch.tensor(l, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = self._model(gt, lt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return np.asarray(probs, dtype=np.float64)

    def predict(self, views: Mapping[str, Any]) -> Classification:
        """Classify a single view dict into a :class:`Classification`.

        ``method='cnn'``. When unavailable/unfitted returns a uniform
        distribution flagged in the rationale.
        """
        proba = self.predict_proba(views)
        prob_map = {cls: float(p) for cls, p in zip(CLASSES, proba)}
        total = sum(prob_map.values())
        if total > 0:
            prob_map = {k: v / total for k, v in prob_map.items()}
        label = max(prob_map, key=prob_map.get)
        if not self.available:
            rationale = ["CNN unavailable (torch not installed)"]
        elif not self._fitted:
            rationale = ["CNN not fitted; returning uniform probabilities"]
        else:
            rationale = [f"CNN prediction: {label} p={prob_map[label]:.2f}"]
        return Classification(
            label=label,
            confidence=float(prob_map[label]),
            probabilities=prob_map,
            method="cnn",
            rationale=rationale,
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save the trained weights + config (requires ``torch``)."""
        if not self.available or self._model is None:
            raise RuntimeError("nothing to save: torch unavailable or model unfitted")
        import torch

        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "global_bins": self.global_bins,
                "local_bins": self.local_bins,
            },
            path,
        )

    def load(self, path: str) -> "CNNClassifier":
        """Load weights previously written by :meth:`save`."""
        if not self.available:
            raise RuntimeError("torch unavailable: cannot load a CNN model")
        import torch

        payload = torch.load(path, map_location="cpu")
        self.global_bins = int(payload.get("global_bins", self.global_bins))
        self.local_bins = int(payload.get("local_bins", self.local_bins))
        self._model = _build_module(self.global_bins, self.local_bins, len(CLASSES))
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
        self._fitted = True
        return self


def _label_to_int(label: Any) -> int:
    """Map a class label (string or int) to its canonical integer index."""
    if isinstance(label, str):
        if label in CLASSES:
            return CLASSES.index(label)
        raise ValueError(f"unknown class label {label!r}; expected one of {CLASSES}")
    idx = int(label)
    if 0 <= idx < len(CLASSES):
        return idx
    raise ValueError(f"class index {idx} out of range for {CLASSES}")
