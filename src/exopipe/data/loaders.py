"""Multi-mission data loaders for ``exopipe`` (network/optional, offline-safe).

Implements the acquisition surface from ``ARCHITECTURE.md`` Section 4 and
``research/01_data_multimission.md``: pull TESS light curves from MAST via
``lightkurve``, fetch ground-truth disposition labels from the NASA Exoplanet
Archive / ExoFOP, cross-match neighbours against Gaia DR3 for blend features, and
read local FITS/CSV light curves into the canonical
:class:`~exopipe.types.LightCurve`.

Design rules (per the build contract):

* **Every** heavy/network dependency (``lightkurve``, ``astroquery``, ``astropy``)
  is imported **lazily inside the function that needs it**. Importing this module
  pulls in nothing but ``numpy`` + the foundation, so it is always importable and
  the offline demo is unaffected.
* When a library or the network is unavailable, loaders raise a clear
  :class:`DataUnavailable` (light-curve loaders) or return an **empty
  DataFrame + warning** (label/cross-match helpers), so callers can fall back to
  the synthetic generator rather than crash (``ARCHITECTURE`` §14).

All returned light curves use the foundation's
:func:`exopipe.data.from_arrays`, so dtype/normalisation conventions stay
consistent with the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..types import LightCurve
from ..utils import get_logger
from .lightcurve import from_arrays

__all__ = [
    "DataUnavailable",
    "load_tess",
    "fetch_labels",
    "gaia_crossmatch",
    "load_from_csv",
    "load_from_fits",
]

_LOG = get_logger("exopipe.data.loaders")


class DataUnavailable(RuntimeError):
    """Raised when remote/optional data cannot be obtained.

    Signals to callers (e.g. :func:`exopipe.driver.run_on_tics`) that they should
    fall back to a synthetic stand-in or skip the target, rather than treating
    the failure as a hard error. The message records *why* (no network, missing
    ``lightkurve``, empty search result, ...).
    """


# --------------------------------------------------------------------------- #
# TESS light curves via lightkurve (MAST)
# --------------------------------------------------------------------------- #
def load_tess(
    tic_id: int | str,
    sector: int | None = None,
    author: str = "SPOC",
    exptime: int | None = 120,
    flux_column: str = "pdcsap_flux",
    quality_bitmask: str = "default",
) -> LightCurve | list[LightCurve]:
    """Download a TESS light curve for ``tic_id`` and convert to a LightCurve.

    Uses ``lightkurve`` to search MAST for the requested product (2-minute SPOC
    PDCSAP by default — the PS7 primary science set, ``research/01`` §1.1), reads
    the PDCSAP flux with the chosen quality bitmask, normalises it, and populates
    ``meta`` with the identifiers and stellar/contamination context the
    downstream physics needs (``tic_id``, ``sector``, ``crowdsap`` from the FITS
    header, ``teff``/``logg``/``radius`` when present, ``ra``/``dec``/``tmag``).

    Parameters
    ----------
    tic_id:
        TESS Input Catalog identifier (int or string; ``"TIC "`` prefix optional).
    sector:
        Restrict to one sector. ``None`` returns **all** matching sectors as a
        ``list[LightCurve]`` (each a separate sector; stitch them with
        :func:`exopipe.data.stitch` if desired).
    author:
        Pipeline/HLSP author string (``"SPOC"``, ``"QLP"``, ``"TESS-SPOC"``,
        ``"Kepler"``, ``"K2"``, ...).
    exptime:
        Exposure time in seconds (120 == 2-min). ``None`` lets lightkurve pick.
    flux_column:
        Which flux column to use (``"pdcsap_flux"`` default; ``"sap_flux"`` for
        uncorrected).
    quality_bitmask:
        lightkurve quality masking (``"none"``/``"default"``/``"hard"``/
        ``"hardest"``).

    Returns
    -------
    LightCurve | list[LightCurve]
        One light curve when ``sector`` is given (or only one is found),
        otherwise a list over sectors.

    Raises
    ------
    DataUnavailable
        If ``lightkurve`` is not installed, the network is unreachable, or the
        search returns no products — so the caller can fall back to synthetic.
    """
    target = _format_tic(tic_id)

    try:
        import lightkurve as lk  # type: ignore
    except Exception as exc:
        raise DataUnavailable(
            f"lightkurve is not installed ({exc}); cannot fetch {target}. "
            "Install the 'science' extra or fall back to synthetic data."
        ) from exc

    try:
        search = lk.search_lightcurve(
            target,
            mission="TESS",
            author=author,
            exptime=exptime,
            sector=sector,
        )
    except Exception as exc:  # network / MAST errors
        raise DataUnavailable(
            f"MAST search failed for {target} ({exc}). Likely no network access."
        ) from exc

    if search is None or len(search) == 0:
        raise DataUnavailable(
            f"No TESS products found for {target} "
            f"(author={author!r}, exptime={exptime}, sector={sector})."
        )

    try:
        if sector is not None:
            collection = [search.download(quality_bitmask=quality_bitmask)]
        else:
            downloaded = search.download_all(quality_bitmask=quality_bitmask)
            collection = list(downloaded) if downloaded is not None else []
    except Exception as exc:  # download / parse errors
        raise DataUnavailable(f"Failed to download {target} ({exc}).") from exc

    light_curves = [
        _lk_to_lightcurve(obj, tic_id=tic_id, flux_column=flux_column)
        for obj in collection
        if obj is not None
    ]
    if not light_curves:
        raise DataUnavailable(f"Downloaded products for {target} were empty/unreadable.")

    if sector is not None or len(light_curves) == 1:
        return light_curves[0]
    return light_curves


def _format_tic(tic_id: int | str) -> str:
    """Normalise a TIC identifier to the ``"TIC <id>"`` string lightkurve wants."""
    text = str(tic_id).strip()
    if text.upper().startswith("TIC"):
        return "TIC " + text[3:].strip()
    return f"TIC {text}"


def _lk_to_lightcurve(
    lk_obj: Any,
    tic_id: int | str,
    flux_column: str = "pdcsap_flux",
) -> LightCurve:
    """Convert a ``lightkurve.LightCurve`` to an :class:`exopipe` LightCurve.

    Extracts time/flux/flux_err arrays, falling back from ``flux_column`` to the
    object's default flux when the named column is absent, and harvests rich
    metadata from the lightkurve ``.meta`` (FITS header) — ``CROWDSAP``, stellar
    parameters, coordinates, magnitude.
    """
    obj_meta = dict(getattr(lk_obj, "meta", {}) or {})

    # -- time -------------------------------------------------------------- #
    time_attr = getattr(lk_obj, "time", None)
    time = np.asarray(getattr(time_attr, "value", time_attr), dtype=np.float64)

    # -- flux (named column with graceful fallback) ------------------------ #
    flux = None
    try:
        if flux_column in getattr(lk_obj, "columns", []):
            flux = np.asarray(lk_obj[flux_column].value, dtype=np.float64)
    except Exception:  # pragma: no cover - column access quirks
        flux = None
    if flux is None:
        flux_attr = getattr(lk_obj, "flux", None)
        flux = np.asarray(getattr(flux_attr, "value", flux_attr), dtype=np.float64)

    # -- flux error -------------------------------------------------------- #
    flux_err = None
    err_col = flux_column + "_err"
    try:
        if err_col in getattr(lk_obj, "columns", []):
            flux_err = np.asarray(lk_obj[err_col].value, dtype=np.float64)
        else:
            err_attr = getattr(lk_obj, "flux_err", None)
            if err_attr is not None:
                flux_err = np.asarray(getattr(err_attr, "value", err_attr), dtype=np.float64)
    except Exception:  # pragma: no cover
        flux_err = None

    meta = _harvest_meta(obj_meta, tic_id=tic_id)
    return from_arrays(time, flux, flux_err, meta=meta)


def _harvest_meta(obj_meta: dict, tic_id: int | str) -> dict:
    """Pull well-known fields out of a lightkurve/FITS header dict into our meta.

    Keys are matched case-insensitively against common SPOC/Kepler header names
    so the same harvester works for TESS and Kepler products.
    """
    lower = {str(k).lower(): v for k, v in obj_meta.items()}

    def pick(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in lower and lower[name] is not None:
                return lower[name]
        return default

    meta: dict[str, Any] = {
        "tic_id": _coerce_int(pick("ticid", "tic_id", "keplerid", "kepid", default=tic_id), tic_id),
        "sector": _coerce_int(pick("sector", "campaign", "quarter"), None),
        "mission": pick("mission", "telescop", default="TESS"),
        "cadence_s": _safe_float(pick("exptime", "telapse", "framtim")),
        "crowdsap": _safe_float(pick("crowdsap", "crowdsap")),
        "flfrcsap": _safe_float(pick("flfrcsap")),
        "ra": _safe_float(pick("ra", "ra_obj")),
        "dec": _safe_float(pick("dec", "dec_obj")),
        "tmag": _safe_float(pick("tessmag", "tmag", "kepmag")),
        "teff": _safe_float(pick("teff")),
        "logg": _safe_float(pick("logg")),
        "radius": _safe_float(pick("radius", "rad", "srad")),
        "mass": _safe_float(pick("mass", "smass")),
    }
    # Drop keys that came back NaN/None so they do not shadow real defaults.
    return {k: v for k, v in meta.items() if v is not None and not _is_nan(v)}


# --------------------------------------------------------------------------- #
# Ground-truth labels: NASA Exoplanet Archive TOI / ExoFOP dispositions
# --------------------------------------------------------------------------- #
def fetch_labels(mission: str = "TESS") -> Any:
    """Fetch ground-truth disposition labels as a pandas DataFrame.

    For ``mission="TESS"`` this queries the NASA Exoplanet Archive ``toi`` table
    (``tfopwg_disp``: PC/CP/KP/FP/APC/FA) via ``astroquery``; for ``"Kepler"`` it
    queries the ``cumulative`` (KOI) table (``koi_disposition`` + the four
    ``koi_fpflag_*`` reasons). These are the ML label backbone described in
    ``research/01`` §4.

    Parameters
    ----------
    mission:
        ``"TESS"`` (default) or ``"Kepler"``.

    Returns
    -------
    pandas.DataFrame
        Columns include the catalog id, disposition, period, duration and depth.
        On any failure (no ``astroquery``, no network, query error) an **empty**
        DataFrame is returned and a warning is logged — never an exception — so a
        training run can proceed offline against synthetic labels.
    """
    import pandas as pd

    mission = str(mission).upper()
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import (  # type: ignore
            NasaExoplanetArchive,
        )
    except Exception as exc:
        _LOG.warning("astroquery unavailable (%s); returning empty label table.", exc)
        return pd.DataFrame()

    try:
        if mission == "TESS":
            table = NasaExoplanetArchive.query_criteria(
                table="toi",
                select="toi,tid,tfopwg_disp,pl_orbper,pl_trandurh,pl_trandep,ra,dec",
            )
        elif mission in ("KEPLER", "KOI"):
            table = NasaExoplanetArchive.query_criteria(
                table="cumulative",
                select=(
                    "kepid,kepoi_name,koi_disposition,koi_period,koi_depth,"
                    "koi_duration,koi_prad,koi_fpflag_nt,koi_fpflag_ss,"
                    "koi_fpflag_co,koi_fpflag_ec"
                ),
            )
        else:
            _LOG.warning("Unknown mission %r for fetch_labels; returning empty.", mission)
            return pd.DataFrame()
    except Exception as exc:  # network / TAP errors
        _LOG.warning("Label query failed for %s (%s); returning empty table.", mission, exc)
        return pd.DataFrame()

    try:
        return table.to_pandas()
    except Exception:  # pragma: no cover - already a DataFrame or convertible
        return pd.DataFrame(table)


# --------------------------------------------------------------------------- #
# Gaia DR3 cone search for blend / contamination features
# --------------------------------------------------------------------------- #
def gaia_crossmatch(ra: float, dec: float, radius_arcsec: float = 21.0) -> Any:
    """Cone-search Gaia DR3 around ``(ra, dec)`` and summarise blend features.

    A TESS pixel is ~21'', so a deep eclipse on a faint neighbour can be diluted
    into the target aperture and masquerade as a transit (``research/01`` §1.4,
    §6.1). This returns the nearby Gaia sources plus the headline blend
    diagnostics the vetting/feature stages consume:

    * ``n_neighbors`` — count of additional sources within the cone,
    * ``brightest_neighbor_delta_mag`` — Δ(G mag) of the brightest neighbour
      relative to the target (smaller ⇒ stronger potential dilution),
    * ``target_ruwe`` — Renormalised Unit Weight Error of the nearest (target)
      source; ``> ~1.4`` flags an unresolved binary.

    Parameters
    ----------
    ra, dec:
        Target coordinates in degrees (ICRS).
    radius_arcsec:
        Cone radius in arcseconds (default 21'' ≈ one TESS pixel; pass ~42'' for
        two pixels).

    Returns
    -------
    pandas.DataFrame
        One row per Gaia source (``source_id``, ``ra``, ``dec``,
        ``phot_g_mean_mag``, ``bp_rp``, ``parallax``, ``ruwe``), with the summary
        blend diagnostics attached on ``df.attrs``. Empty DataFrame + warning on
        any failure (no astroquery/astropy/network).
    """
    import pandas as pd

    try:
        import astropy.units as u  # type: ignore
        from astropy.coordinates import SkyCoord  # type: ignore
        from astroquery.gaia import Gaia  # type: ignore
    except Exception as exc:
        _LOG.warning("Gaia cross-match unavailable (%s); returning empty table.", exc)
        return pd.DataFrame()

    try:
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        coord = SkyCoord(ra=float(ra), dec=float(dec), unit="deg", frame="icrs")
        job = Gaia.cone_search_async(coord, radius=u.Quantity(float(radius_arcsec), u.arcsec))
        table = job.get_results()
        df = table.to_pandas()
    except Exception as exc:  # network / ADQL errors
        _LOG.warning("Gaia cone search failed at (%.4f, %.4f): %s", ra, dec, exc)
        return pd.DataFrame()

    df.attrs.update(_summarize_gaia(df))
    return df


def _summarize_gaia(df: Any) -> dict:
    """Compute blend-diagnostic scalars from a Gaia cone-search DataFrame."""
    summary: dict[str, Any] = {
        "n_neighbors": 0,
        "brightest_neighbor_delta_mag": np.nan,
        "target_ruwe": np.nan,
    }
    if df is None or len(df) == 0 or "phot_g_mean_mag" not in df:
        return summary

    mags = np.asarray(df["phot_g_mean_mag"], dtype=float)
    finite = np.isfinite(mags)
    if not finite.any():
        return summary

    target_idx = int(np.nanargmin(mags))  # brightest == assumed target
    summary["n_neighbors"] = int(finite.sum() - 1)
    if "ruwe" in df:
        summary["target_ruwe"] = _safe_float(np.asarray(df["ruwe"], dtype=float)[target_idx])

    neigh = mags.copy()
    neigh[target_idx] = np.inf  # exclude the target itself
    if np.isfinite(neigh).any():
        brightest_neighbor = float(np.nanmin(neigh))
        summary["brightest_neighbor_delta_mag"] = brightest_neighbor - float(mags[target_idx])
    return summary


# --------------------------------------------------------------------------- #
# Local file loaders
# --------------------------------------------------------------------------- #
def load_from_csv(
    path: str | Path,
    time_col: str | None = None,
    flux_col: str | None = None,
    flux_err_col: str | None = None,
) -> LightCurve:
    """Load a light curve from a CSV file into a :class:`LightCurve`.

    Column names are auto-detected from common aliases (``time``/``bjd``/``btjd``
    for time; ``flux``/``pdcsap_flux``/``sap_flux`` for flux) unless given
    explicitly. Any remaining scalar columns are ignored; ``tic_id``/``sector``
    are lifted into ``meta`` when present.

    Parameters
    ----------
    path:
        Path to a CSV with at least a time and a flux column.
    time_col, flux_col, flux_err_col:
        Explicit column names; auto-detected when ``None``.

    Returns
    -------
    LightCurve
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    frame = pd.read_csv(path)
    cols = {c.lower(): c for c in frame.columns}

    tcol = time_col or _first_present(cols, ("time", "bjd", "btjd", "bkjd", "jd", "t"))
    fcol = flux_col or _first_present(
        cols, ("flux", "pdcsap_flux", "sap_flux", "norm_flux", "f")
    )
    if tcol is None or fcol is None:
        raise ValueError(
            f"Could not identify time/flux columns in {path}; "
            f"available columns: {list(frame.columns)}"
        )
    ecol = flux_err_col or _first_present(
        cols, ("flux_err", "pdcsap_flux_err", "sap_flux_err", "err", "ferr")
    )

    time = np.asarray(frame[tcol], dtype=np.float64)
    flux = np.asarray(frame[fcol], dtype=np.float64)
    flux_err = np.asarray(frame[ecol], dtype=np.float64) if ecol else None

    meta: dict[str, Any] = {"source_path": str(path), "mission": "local-csv"}
    for key in ("tic_id", "sector", "tmag", "teff", "radius", "crowdsap"):
        if key in cols:
            value = frame[cols[key]].iloc[0] if len(frame) else None
            if value is not None and not _is_nan(value):
                meta[key] = value.item() if isinstance(value, np.generic) else value
    return from_arrays(time, flux, flux_err, meta=meta)


def load_from_fits(
    path: str | Path,
    flux_column: str = "pdcsap_flux",
    quality_bitmask: str = "default",
) -> LightCurve:
    """Load a local SPOC/QLP/Kepler FITS light curve into a :class:`LightCurve`.

    Prefers ``lightkurve.read`` (handles all mission flavours, quality masking,
    and header metadata) and falls back to a direct ``astropy.io.fits`` reader of
    the ``LIGHTCURVE`` HDU when lightkurve is absent. Picks the PDCSAP flux column
    by default and harvests ``CROWDSAP`` + stellar params from the primary header.

    Parameters
    ----------
    path:
        Path to a ``*_lc.fits`` file.
    flux_column:
        Flux column to use (``"pdcsap_flux"`` default).
    quality_bitmask:
        lightkurve quality masking when lightkurve is used.

    Returns
    -------
    LightCurve

    Raises
    ------
    DataUnavailable
        If neither ``lightkurve`` nor ``astropy`` can read the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FITS not found: {path}")

    # -- preferred: lightkurve (rich metadata, quality masking) ------------- #
    try:
        import lightkurve as lk  # type: ignore

        obj = lk.read(str(path), quality_bitmask=quality_bitmask)
        return _lk_to_lightcurve(obj, tic_id=None, flux_column=flux_column)
    except Exception as exc:
        _LOG.debug("lightkurve.read failed for %s (%s); trying astropy.io.fits.", path, exc)

    # -- fallback: raw astropy FITS reader ---------------------------------- #
    try:
        from astropy.io import fits  # type: ignore
    except Exception as exc:
        raise DataUnavailable(
            f"Cannot read {path}: neither lightkurve nor astropy is available ({exc})."
        ) from exc

    try:
        with fits.open(path) as hdul:
            header = dict(hdul[0].header)
            data = hdul[1].data
            names = {n.lower(): n for n in data.columns.names}
            tname = names.get("time", "TIME")
            fname = names.get(flux_column, names.get("pdcsap_flux", names.get("sap_flux")))
            ename = names.get(flux_column + "_err", names.get("pdcsap_flux_err"))
            if fname is None:
                raise DataUnavailable(f"No usable flux column in {path}.")
            time = np.asarray(data[tname], dtype=np.float64)
            flux = np.asarray(data[fname], dtype=np.float64)
            flux_err = np.asarray(data[ename], dtype=np.float64) if ename else None
    except DataUnavailable:
        raise
    except Exception as exc:
        raise DataUnavailable(f"Failed to parse FITS {path} ({exc}).") from exc

    meta = _harvest_meta(header, tic_id=header.get("TICID"))
    meta.setdefault("source_path", str(path))
    return from_arrays(time, flux, flux_err, meta=meta)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _first_present(cols: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in cols:
            return cols[name]
    return None


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _coerce_int(value: Any, default: Any = None) -> Any:
    try:
        if value is None or _is_nan(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_nan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
