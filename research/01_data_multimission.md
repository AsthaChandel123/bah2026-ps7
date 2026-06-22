# 01 — Multi-Mission Data Sources & Acquisition (PS7)

**Scope:** Concrete, implementation-ready reference for acquiring exoplanet-transit light curves, training labels, and cross-verification data from **many** missions, surveys, and catalogs worldwide. The PS7 brief mandates not relying on a single dataset: we use TESS as the primary science target, Kepler/K2 as a large labeled augmentation set, **Gaia DR3** for blend/contamination resolution, ground-based surveys + RV archives for confirmation, and the NASA Exoplanet Archive / ExoFOP / EB catalogs for ML ground-truth labels.

> All code assumes Python ≥3.9 with `lightkurve>=2.6`, `astroquery>=0.4.7`, `astropy>=5.3`, `numpy`, `pandas`. Install:
> ```bash
> python -m pip install --upgrade lightkurve astroquery astropy numpy pandas \
>     transitleastsquares wotan dace-query pycheops
> ```

---

## 0. TL;DR — Recommended integration stack

| Role | Source | Why |
|---|---|---|
| **Primary science target** | TESS 2-min SPOC light curves (one sector, ~20k stars) | Required by PS7; high cadence; PDCSAP detrended |
| **Backfill / FFI targets** | TESS-SPOC & QLP HLSPs (FFI-derived) | Covers stars without 2-min data; ~millions of light curves |
| **Large labeled training set** | Kepler + K2 light curves + KOI/cumulative table dispositions | ~150k Kepler targets, mature CONFIRMED/FALSE POSITIVE labels for transfer learning |
| **Transit/EB/FP labels (TESS)** | NASA Exoplanet Archive `toi` table + ExoFOP-TESS `download_toi.php` | TFOPWG disposition: PC/CP/KP/FP/APC/FA |
| **Eclipsing-binary labels** | Villanova/MAST TESS-EBs + Kepler EB catalog | Direct EB class labels w/ morphology |
| **Confirmed-planet labels** | NASA Exoplanet Archive `ps` / `pscomppars` | Ground-truth CONFIRMED planets + true depth/period |
| **Blend / contamination resolution** | **Gaia DR3** `gaiadr3.gaia_source` cone search + TIC `contratio` | Nearby stars in aperture → blend / NEB diagnosis |
| **Stellar parameters** | TIC v8.2 (Tmag, Teff, R*, contamination) + Gaia DR3 (parallax, RUWE, BP/RP) | Convert depth→radius; flag unresolved binaries (RUWE) |
| **Independent photometric cross-check** | ZTF, ASAS-SN, WASP/SuperWASP, NGTS, KELT | Confirm period/EB on independent instrument |
| **Confirmation / ground truth** | RV archives via DACE (`dace-query`) + ExoFOP | Mass → planet vs. EB; spectroscopic FP flags |

**Recommended datasets to actually integrate (priority order):**
1. TESS 2-min SPOC LCs (one sector) — primary.
2. NASA Exoplanet Archive `toi`, `ps`/`pscomppars`, `cumulative` (KOI) — labels.
3. ExoFOP-TESS TOI CSV — authoritative TFOPWG dispositions.
4. TESS-EBs + Kepler EB catalog — EB-class labels.
5. Kepler/K2 LCs — domain-augmentation for the classifier.
6. Gaia DR3 cone search + TIC v8.2 — blend/contamination features.
7. (Optional confirmation) ZTF/ASAS-SN archival photometry; DACE RVs.

---

## 1. Space missions / satellites

### 1.1 TESS (Transiting Exoplanet Survey Satellite) — PRIMARY

- **Operator/archive:** NASA / MAST (`https://archive.stsci.edu/tess/`). Bulk page: `https://archive.stsci.edu/tess/bulk_downloads.html`. TIC/CTL page (PS7 link): `https://archive.stsci.edu/tess/tic_ctl.html`.
- **Lifetime:** Launched 2018; primary mission 2018–2020, extended missions ongoing (sectors now 1000+ in the second extended mission numbering). Each **sector ≈ 27.4 days**, ~2 sectors/observing cycle per pointing.
- **Cadences:** **2-minute** (postage-stamp targets, the PS7 "high-cadence" set, ~20k/sector), **20-second** (fast, subset), and **Full-Frame Images (FFI)** at 30 min (primary), 10 min (EM1), **200 s** (EM2).
- **Data products:**
  - **2-min SPOC** Light Curve Files (LCF) and Target Pixel Files (TPF). Columns include `SAP_FLUX`, `PDCSAP_FLUX`, `QUALITY`, `TIME`, etc. (see §7).
  - **Data Validation (DV)** reports/time series (`*dvt.fits`, `*dvr.pdf`) — SPOC's own transit search output; useful as a sanity/label cross-check (sectors ≥36 in bulk scripts).
  - **HLSPs derived from FFIs** (cover stars without 2-min data):
    - **TESS-SPOC** — `https://archive.stsci.edu/hlsp/tess-spoc` (SPOC pipeline on FFIs).
    - **QLP (Quick-Look Pipeline, MIT)** — `https://tess.mit.edu/qlp/`; delivered to MAST, improved precision from S56+ and again from S94+; `lightkurve` author string `"QLP"`.
    - **GSFC-ELEANOR-LITE**, **T16**, **CDIPS**, **PATHOS** — additional FFI HLSPs on MAST.
  - **TESScut** — on-demand FFI cutouts (make your own light curve for any RA/Dec): `https://mast.stsci.edu/tesscut/`.
- **Catalogs:** **TIC v8.2** (TESS Input Catalog) and **CTL / xCTL v08.01** (Candidate Target List — curated likely-dwarf exoplanet hosts). Per `tic_ctl.html`: TIC v8.2 ships as **90 declination-band gzipped CSV files** (−90°…+90°); xCTL v08.01 is a 497 MB CSV (or 9.5 GB cross-matched). Key columns: `Tmag`, `Teff`, `rad`, `mass`, `ra`/`dec`, and crucially **`contratio`** (contamination ratio — flux fraction in aperture from neighbors). Also queryable via MAST (`Catalogs.query_object(... catalog="TIC")`) and VizieR (`IV/39`).

**Python — search & download (lightkurve):**
```python
import lightkurve as lk

# Find all available products for a target, then pick 2-min SPOC
sr = lk.search_lightcurve("TIC 307210830", mission="TESS",
                          author="SPOC", exptime=120)
print(sr)                      # table of sectors/products
lc = sr.download()             # single LightCurve
# or all sectors:
lcc = sr.download_all()        # LightCurveCollection
lc  = lcc.stitch()             # concatenate sectors (normalizes each first)

# FFI-derived alternatives when no 2-min data exists:
sr_qlp  = lk.search_lightcurve("TIC 307210830", author="QLP")
sr_tsp  = lk.search_lightcurve("TIC 307210830", author="TESS-SPOC")
# Make your own LC from an FFI cutout:
sr_cut  = lk.search_tesscut("TIC 307210830")     # then .download(cutout_size=11)
```

**Python — bulk-download an entire sector's 2-min LCs (PS7 requirement):**
MAST provides per-sector **curl shell scripts** at
`https://archive.stsci.edu/missions/tess/download_scripts/sector/`.
Naming convention (verified): **`tesscurl_sector_<NN>_lc.sh`** (light curves), `tesscurl_sector_<NN>_tp.sh` (target pixel), `tesscurl_sector_<NN>_fast-tp.sh` (20-s), and DV scripts for sectors ≥36. Each script is a long list of `curl` lines, one per FITS file, of the form:
```bash
curl -C - -L -o tess2019..._s0014_..._lc.fits \
  "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/tess2019..._lc.fits"
```
Download and run a whole sector (≈20–30k files):
```bash
SECTOR=14
curl -L -O "https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_${SECTOR}_lc.sh"
bash tesscurl_sector_${SECTOR}_lc.sh        # pulls all *_lc.fits for the sector
```
> Tip for a manageable PS7 demo: subsample the script (e.g. `head -2000`) or filter the FITS list by TIC to ~a few thousand light curves. You can also bulk-query with `astroquery.mast.Observations` (programmatic alternative to the shell script):
```python
from astroquery.mast import Observations
obs = Observations.query_criteria(obs_collection="TESS",
                                  dataproduct_type="timeseries",
                                  sequence_number=14,        # sector
                                  target_name="*",           # all targets
                                  provenance_name="SPOC")
prod = Observations.get_product_list(obs)
lc_only = Observations.filter_products(prod, productSubGroupDescription="LC")
Observations.download_products(lc_only, download_dir="tess_s14")
```

### 1.2 Kepler — large labeled augmentation set

- **Archive:** MAST. **Lifetime:** 2009–2013 (one ~115 deg² field, ~150k–200k targets). **Cadence:** **long = 29.4 min** (1765.5 s), **short = 58.9 s**. Quarters Q0–Q17.
- **Products:** LCF (`SAP_FLUX`, `PDCSAP_FLUX`, `SAP_QUALITY`), TPF. Author string `"Kepler"`.
- **Why for PS7:** Mature, homogeneous labels (**KOI cumulative table**: CONFIRMED / CANDIDATE / FALSE POSITIVE) → ideal for **transfer learning / domain adaptation** into a TESS classifier (see §6). Kepler EB catalog gives EB labels.
```python
sr = lk.search_lightcurve("Kepler-10", author="Kepler", cadence="long")
lc = sr.download_all().stitch()
```

### 1.3 K2 — ecliptic survey, extra labels

- **Archive:** MAST. **Lifetime:** 2014–2018, **20 Campaigns** along the ecliptic (~80 days each, 10k–20k LC targets/campaign). Same cadences as Kepler.
- **Products:** K2 LCF/TPF (author `"K2"`); HLSPs **EVEREST**, **K2SFF**, **K2SC** correct the spacecraft-roll systematics. Labels via NASA Archive **`k2pandc`** and `k2names` tables.
```python
sr = lk.search_lightcurve("K2-18", author="K2")          # or author="EVEREST"
```

### 1.4 Gaia (ESA) DR3 — blend/contamination + stellar params (CRITICAL)

- **Archive:** ESA Gaia (`https://gea.esac.esa.int/`), mirrored in `astroquery.gaia`. **DR3** (2022) covers ~1.8 billion sources.
- **Products/tables:**
  - `gaiadr3.gaia_source` — astrometry (`ra,dec,parallax,pmra,pmdec`), photometry (`phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag`), **`ruwe`** (Renormalized Unit Weight Error — RUWE > ~1.4 flags unresolved binaries/blends), `bp_rp` color.
  - `gaiadr3.epoch_photometry` (DataLink) — time-series G/BP/RP for ~11.8M variables (independent variability cross-check).
  - Variability / classification tables (`vari_*`).
- **Why for PS7:** The single best tool for the **blend** class. A TESS pixel is ~21″; many "transits" are actually a deep eclipse on a faint **neighbor** diluted into the target aperture. Gaia resolves those neighbors. RUWE flags the target itself as an unresolved binary. Gaia parallax+photometry give R* and luminosity class to separate dwarf (planet-plausible) from giant.
```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

coord = SkyCoord(ra=280.0, dec=-60.0, unit="deg", frame="icrs")
job = Gaia.cone_search_async(coord, radius=u.Quantity(42, u.arcsec))  # ~2 TESS pixels
neighbors = job.get_results()        # source_id, ra, dec, phot_g_mean_mag, parallax, ruwe...
# ADQL equivalent with explicit columns + magnitude filter:
adql = """
SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, ruwe
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 280.0, -60.0, 0.0117)) = 1
  AND phot_g_mean_mag < 17
ORDER BY phot_g_mean_mag
"""
neighbors = Gaia.launch_job_async(adql).get_results()
```

### 1.5 CHEOPS (ESA) — targeted follow-up photometry

- **Archive:** ESA mission archive + **DACE** (`https://dace.unige.ch/`). **Lifetime:** launched 2019, operating. Single-target, ultra-high-precision visits (not a survey).
- **Access:** `dace-query` (`from dace_query.cheops import Cheops`) or **`pycheops`** (`https://github.com/pmaxted/pycheops`).
- **Why:** Confirm/refine a specific TESS candidate's depth/shape at higher precision (good for the report's "confidence" narrative, optional for the pipeline).
```python
from dace_query.cheops import Cheops
visits = Cheops.query_database(filters={"obj_id_catname": {"equal": ["HD88111"]}})
```

### 1.6 Others (context / optional)

- **CoRoT** (CNES/ESA, 2006–2012): early transit survey; archive at IAS/CDS & VizieR. Historical labels.
- **JWST**: not a transit-survey source; relevant only for atmospheric follow-up of confirmed planets (out of PS7 scope).
- **Hubble (HST)**: targeted transit spectroscopy of known planets (out of scope for detection).
- **PLATO (ESA)**: future (launch ~2026–2027); no data yet — note as future cross-verification source.

---

## 2. Ground-based surveys (cross-validation / independent confirmation / training)

These give **independent instruments** to confirm a period or unmask an EB, and broaden training diversity. Most are queryable by cone search on their archive or via VizieR; ZTF/ASAS-SN have first-class Python APIs.

| Survey | What | Access (Python) |
|---|---|---|
| **ZTF** (Zwicky Transient Facility) | g/r/i time-domain photometry, all-sky North | IRSA: `astroquery.ipac.irsa.Irsa.query_region(... catalog="ztf_objects")` and ZTF lightcurve API; `ztfquery` package |
| **ASAS-SN** | All-sky V/g photometry, variable-star DB | Sky-patrol API: `pyasassn` client (`from pyasassn.client import SkyPatrolClient`); web `asas-sn.osu.edu` |
| **WASP / SuperWASP** | Bright-star hot-Jupiter survey archive | SuperWASP photometry archive (`wasp.cerit-sc.cz` / NASA Archive ingest); VizieR catalogs |
| **HATNet / HATSouth** | Wide-field transit survey (HAT-P planets) | Project archive + VizieR; labels via discovered planets in NASA Archive |
| **KELT** | Bright-star survey (KELT-N/S) | VizieR catalogs; labels via NASA Archive (`disc_facility like '%KELT%'`) |
| **NGTS** | Next-Gen Transit Survey, ESO Paranal | ESO archive (`astroquery.eso`), DACE photometry; VizieR |
| **OGLE** | Microlensing + transit photometry (Galactic bulge) | OGLE archive (`ogle.astrouw.edu.pl`); VizieR |
| **Evryscope** | All-sky gigapixel survey | Project data releases; VizieR |

```python
# ZTF light curve for a position (IRSA):
from astroquery.ipac.irsa import Irsa
import astropy.units as u
from astropy.coordinates import SkyCoord
ztf = Irsa.query_region(SkyCoord(280, -60, unit="deg"),
                        catalog="ztf_objects_dr22", radius=3*u.arcsec)

# ASAS-SN Sky Patrol:
# pip install skypatrol  ->  from pyasassn.client import SkyPatrolClient
```
> For PS7, ground-based data is best used as **opportunistic confirmation** (does an independent survey see the same period?) rather than a core pipeline input. Use VizieR (§4) as the universal entry point when a survey lacks a Python client.

---

## 3. Radial-velocity & spectroscopy archives (confirmation / ground-truth labels)

RV mass measurements are the gold-standard discriminator: a transit-shaped signal with a **stellar-mass** companion is an EB; a planetary-mass one is a planet. Spectroscopy also yields SB1/SB2 FP flags on ExoFOP.

| Instrument | Host | Access |
|---|---|---|
| **HARPS / HARPS-N** | ESO 3.6 m / TNG | **DACE** `dace-query` (`Spectroscopy.get_timeseries`); ESO archive (`astroquery.eso`) |
| **ESPRESSO** | ESO VLT | DACE; ESO archive |
| **HIRES** | Keck | DACE (public reduced RVs); KOA (Keck Observatory Archive) |
| **NEID** | WIYN | NExScI NEID archive; ExoFOP links |
| **CORALIE / PFS** | Euler / Magellan | DACE |

```python
from dace_query import Spectroscopy
rv = Spectroscopy.get_timeseries(target="HD40307",
                                 sorted_by_instrument=False,
                                 output_format="numpy")
# -> dict of arrays (rjd, rv, rv_err, instrument, ...). Filter by rv_err, SNR, DRS QC.
```
> DACE auth: place credentials in `~/.dacerc` for proprietary data; **public** RVs need no login. For PS7, RV is a label/sanity source, not a pipeline input.

---

## 4. Catalogs & cross-match services (the ML label backbone)

### 4.1 NASA Exoplanet Archive (NExScI) — primary label source

- **TAP/astroquery:** `from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive`. TAP base URL: `https://exoplanetarchive.ipac.caltech.edu/TAP`.
- **Key tables (verified):**
  - `ps` — Planetary Systems (all published parameter sets, `soltype` includes `Published Confirmed`).
  - `pscomppars` — Composite (one row per planet, best params). Columns: `pl_orbper`, `pl_trandep`, `pl_trandur`, `pl_rade`, `st_rad`, `disc_facility`, `discoverymethod`, etc.
  - `toi` — **TESS Objects of Interest** with `tfopwg_disp` (PC/CP/KP/FP/APC/FA), `toi`, `tid` (TIC ID), `pl_orbper`, `pl_trandurh`, `pl_trandep`, `ra`, `dec`.
  - `cumulative` (a.k.a. KOI cumulative) — Kepler `koi_disposition` (CONFIRMED/CANDIDATE/FALSE POSITIVE), `koi_period`, `koi_depth`, `koi_duration`, `koi_prad`, plus the four **`koi_fpflag_*`** false-positive flags (NT, SS, CO, EC).
  - `k2pandc`, `keplernames`, `k2names`.
```python
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# Confirmed planets with true depth/period (positive-class labels + regression truth)
conf = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="pl_name,hostname,pl_orbper,pl_trandep,pl_trandur,pl_rade,st_rad,disc_facility",
    where="default_flag=1")

# All TESS TOIs with disposition (multi-class labels for TESS targets)
toi = NasaExoplanetArchive.query_criteria(
    table="toi",
    select="toi,tid,tfopwg_disp,pl_orbper,pl_trandurh,pl_trandep,ra,dec")

# Kepler KOIs (large labeled set incl. false positives + FP-flag breakdown)
koi = NasaExoplanetArchive.query_criteria(
    table="cumulative",
    select="kepid,kepoi_name,koi_disposition,koi_period,koi_depth,"
           "koi_duration,koi_prad,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec")
```

### 4.2 ExoFOP-TESS — authoritative TFOPWG dispositions

- **Bulk CSV (verified, no auth):** `https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv` — all TOIs. Columns include **`TIC ID`**, **`TOI`**, **`TESS Disposition`**, **`TFOPWG Disposition`** (PC/CP/KP/FP/APC/FA/EB/O), `Period (days)`, `Duration (hours)`, `Depth (ppm)`, `RA`, `Dec`, plus stellar params and follow-up flags.
- Per-target detail endpoints exist (`download_target.php?id=<TIC>`). A community helper: `lundmb/ExoFOP-Tools` (`TOI_lookup.py`).
```python
import pandas as pd
toi_ef = pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
labels = toi_ef[["TIC ID","TOI","TFOPWG Disposition","Period (days)",
                 "Duration (hours)","Depth (ppm)"]]
```
**Disposition decoding (label map for PS7 classes):**
`CP`=Confirmed Planet, `KP`=Known Planet → **transit**; `EB`=Eclipsing Binary → **eclipsing-binary**; `FP`=False Positive (often blend/NEB/instrument) → **blend/other**; `PC`=Planet Candidate (weak label), `APC`=Ambiguous PC, `FA`=False Alarm (instrumental) → **other/noise**.

### 4.3 Eclipsing-binary catalogs — EB-class labels

- **TESS-EBs HLSP** (Prša et al. 2022): `https://archive.stsci.edu/hlsp/tess-ebs`. v1.0 = **~4580 EBs**, Sectors 1–26. Direct CSV: `hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv`. Columns: `tess_id` (TIC), `signal_id`, period, ephemeris (BJD), morphology, eclipse depths. Also on **CasJobs** (SQL + Gaia/PanSTARRS cross-match) and Villanova portal `tessebs.villanova.edu`. DOI 10.17909/t9-9gm4-fx30.
- **Kepler EB catalog** (Villanova, Prša/Kirk et al.): `http://keplerEBs.villanova.edu` — ~2900 Kepler EBs with morphology parameter; also VizieR.
```python
import pandas as pd
tess_ebs = pd.read_csv(
  "https://archive.stsci.edu/hlsps/tess-ebs/"
  "hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv")
# -> TIC IDs labeled as eclipsing binaries (negative class for "planet")
```

### 4.4 MAST / VizieR / SIMBAD / TIC — cross-match plumbing

```python
# --- MAST: TIC stellar params + contamination for a TESS target ---
from astroquery.mast import Catalogs
tic = Catalogs.query_object("TIC 307210830", catalog="TIC")
# columns incl. Tmag, Teff, rad, mass, contratio (contamination ratio), ra, dec

# --- VizieR: any survey/EB/TIC catalog by ID ---
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1
cat = Vizier.get_catalogs("J/ApJS/258/16")     # TESS-EBs (Prsa+ 2022) on VizieR
tic_vz = Vizier.get_catalogs("IV/39")          # TIC v8.2 on VizieR
# cone search with a column filter:
res = Vizier(column_filters={"Tmag": "<12"}).query_region(
        SkyCoord(280,-60,unit="deg"), radius=1*u.arcmin, catalog="IV/39")

# --- SIMBAD: object type / identifiers (is it a known EB*/RR Lyr*?) ---
from astroquery.simbad import Simbad
s = Simbad.query_object("TIC 307210830")       # OTYPE helps flag variables
```

---

## 5. Python access tooling — quick reference

| Tool | Import | Use |
|---|---|---|
| `lightkurve` | `import lightkurve as lk` | search/download TESS/Kepler/K2 LCs & TPFs, TESScut, BLS/LS periodograms, fold/flatten/bin |
| `astroquery.mast` | `from astroquery.mast import Observations, Catalogs, Tesscut` | bulk MAST queries, TIC catalog, FFI cutouts |
| `astroquery ... nasa_exoplanet_archive` | `NasaExoplanetArchive` | `ps`,`pscomppars`,`toi`,`cumulative`,`k2pandc` labels |
| `astroquery.gaia` | `from astroquery.gaia import Gaia` | DR3 cone search, ADQL, RUWE, epoch photometry |
| `astroquery.vizier` | `from astroquery.vizier import Vizier` | any published catalog (EB, TIC, survey) |
| `astroquery.simbad` | `from astroquery.simbad import Simbad` | object types/identifiers |
| `astroquery.ipac.irsa` | `from astroquery.ipac.irsa import Irsa` | ZTF & IRSA catalogs |
| `astroquery.eso` | `from astroquery.eso import Eso` | NGTS/HARPS/ESPRESSO raw via ESO |
| `dace-query` | `from dace_query import Spectroscopy` ; `from dace_query.cheops import Cheops` | RVs (HARPS/ESPRESSO/HIRES), CHEOPS |
| `pycheops` | `import pycheops` | CHEOPS light-curve analysis |
| `transitleastsquares` | `from transitleastsquares import transitleastsquares` | TLS detection + SDE significance |
| `wotan` | `from wotan import flatten` | robust detrending (biweight) |
| `astropy.timeseries` | `from astropy.timeseries import BoxLeastSquares` | BLS periodogram + FAP |

**End-to-end micro-example (target → detection → significance):**
```python
import numpy as np, lightkurve as lk
from transitleastsquares import transitleastsquares

lc = (lk.search_lightcurve("TIC 307210830", author="SPOC", exptime=120)
        .download_all().stitch()
        .remove_nans().remove_outliers(sigma=5)
        .normalize())
flat = lc.flatten(window_length=401)                 # detrend stellar variability
t, f = np.ascontiguousarray(flat.time.value), np.ascontiguousarray(flat.flux.value)
model = transitleastsquares(t, f)
res = model.power()                                   # period, T0, duration, depth
print(res.period, res.T0, res.duration, res.depth, "SDE=", res.SDE)
# SDE>~9 => FAP ~1e-4 (Hippke & Heller 2019); also res.FAP available.

# Or BLS via astropy (matches PS7's depth/period/duration deliverable):
from astropy.timeseries import BoxLeastSquares
bls = BoxLeastSquares(flat.time.value, flat.flux.value)
pg  = bls.autopower(0.05)                             # 0.05 d trial durations
best = np.argmax(pg.power)
period, t0, dur = pg.period[best], pg.transit_time[best], pg.duration[best]
depth = bls.compute_stats(period, dur, t0)["depth"]
```

---

## 6. Gap-filling & cross-verification strategy

1. **Blend / contamination detection (the hard PS7 class):**
   - Gaia DR3 cone search at ~2 TESS pixels (≈42″). If a neighbor within the aperture is bright enough that *its* full eclipse, **diluted** by the flux ratio, could reproduce the observed depth → flag **blend / nearby-eclipsing-binary (NEB)**. Test: `depth_target ≈ depth_neighbor × f_neighbor/(f_target+Σf)`.
   - Cross-check with TIC **`contratio`** (precomputed contamination) — high `contratio` ⇒ untrustworthy depth.
   - **Difference-imaging / centroid** logic: if the in-transit flux centroid shifts toward a neighbor (from TPF pixels), the source is the neighbor. lightkurve TPFs expose per-pixel light curves for this.
   - **RUWE > ~1.4** on the target ⇒ unresolved companion ⇒ EB-prone.
   - **Odd/even depth difference** and **secondary eclipse** at phase 0.5 ⇒ EB rather than planet (compute from phase-folded LC).

2. **Augment the TESS classifier with Kepler/K2 (transfer / domain adaptation):**
   - Kepler `cumulative` (KOI) gives tens of thousands of vetted CONFIRMED/FALSE-POSITIVE labels with the four `koi_fpflag_*` reasons — far more than TESS alone. Train base features on Kepler, then **fine-tune** on TESS TOIs.
   - Harmonize cadence: bin/resample Kepler 29.4-min and TESS 2-min light curves to a common phase grid before feeding a CNN, or extract cadence-invariant phase-folded "global/local views" (à la Shallue & Vanderburg 2018 / `astronet`).
   - Domain gap: normalize per-light-curve (median-divide), use the same detrending (`wotan` biweight, window = 3× transit duration) on all missions.

3. **Multiple sectors of the same star:** stitch all TESS sectors (`download_all().stitch()`) to lower noise and confirm the **period repeats** across sectors (a transient single dip across one sector is likely systematics/FA). Consistent depth across sectors argues against chromatic blends.

4. **Reconcile conflicting dispositions across archives:** build a per-TIC label by precedence — **spectroscopically confirmed (RV/ExoFOP `CP`/`KP`) > EB catalog membership > TFOPWG `FP`/`EB` > KOI/TOI `PC`**. Keep a `label_source` and `label_confidence` column; drop or down-weight rows where NExScI `toi`, ExoFOP, and the EB catalog disagree. Treat `PC`/`APC` as weak/semi-supervised labels.

5. **Independent-instrument confirmation:** for a high-value candidate, query ZTF/ASAS-SN/SuperWASP at the same coordinates; an EB usually shows the same period in ground data, while a shallow planet transit is typically below ground sensitivity (its *absence* in deep ground data is itself weak evidence for "planet, not EB").

---

## 7. Data-format details (FITS light curves)

**Structure (SPOC LCF, 2-min):** multi-extension FITS.
- HDU0 = primary header (TICID, sector, camera/CCD, RA/Dec, Tmag, `CROWDSAP`, `FLFRCSAP`).
- HDU1 = `LIGHTCURVE` binary table. Key columns:
  - `TIME` — BJD − 2457000 (BTJD), days.
  - `SAP_FLUX`, `SAP_FLUX_ERR` — simple aperture photometry (e⁻/s).
  - `PDCSAP_FLUX`, `PDCSAP_FLUX_ERR` — systematics-corrected (CBV detrended); **default for transit search**.
  - `SAP_BKG`, `MOM_CENTR1/2`, `POS_CORR1/2`.
  - **`QUALITY`** — integer bitmask; bit meanings per Twicken et al. 2020, Table 32. `QUALITY == 0` ⇒ clean cadence.
- HDU2 = `APERTURE` mask image.

**Quality flags / bad-cadence removal:** `lightkurve` applies `quality_bitmask` on read — options `'none'`, `'default'`, `'hard'`, `'hardest'` (increasingly aggressive masking of momentum dumps, desats, manual excludes). For custom masking, bitwise-AND `QUALITY` with the bits you want to drop.

**Gaps / NaNs:** cadences with no valid flux are `NaN` (and data gaps exist at each ~13.7-day momentum dump / sector download). Handle with `lc.remove_nans()`; **do not** interpolate across large gaps for BLS — BLS/TLS handle uneven sampling natively. Mask thruster-firing cadences before detrending.

**Normalization conventions:** divide by the median (or per-sector median before stitching) so out-of-transit flux ≈ 1.0; transit depth then reads directly as `1 − f_min` (×10⁶ = ppm). `lc.normalize()` does this. Always **detrend stellar variability/systematics** (`flatten` / `wotan`) *before* measuring depth, but recover the true (untrended) depth for the final parameter estimate by fitting the transit on lightly-detrended data, since aggressive flattening can dilute deep transits.

**Cross-mission notes:** Kepler/K2 use the same column scheme but `SAP_QUALITY` and time = BJD − 2454833 (BKJD). QLP HLSPs use `KSPSAP_FLUX`/`SAP_FLUX` with their own detrending and a `QUALITY`/`ORBITID` column; lightkurve reads them transparently.

---

## 8. Source-by-source master table

| # | Source | Type | Access method | Python call | Good for | Labels? |
|---|---|---|---|---|---|---|
| 1 | TESS 2-min SPOC LC | Space photometry | MAST / lightkurve / bulk curl | `lk.search_lightcurve(..., author="SPOC", exptime=120)` ; `tesscurl_sector_NN_lc.sh` | **Primary science target** | via TOI/ExoFOP |
| 2 | TESS-SPOC / QLP HLSP | FFI photometry | MAST / lightkurve | `lk.search_lightcurve(..., author="QLP"/"TESS-SPOC")` | Backfill non-2-min stars | via TOI |
| 3 | TESScut | FFI cutout svc | MAST | `lk.search_tesscut(...)` / `Tesscut.get_cutouts` | Custom LC any RA/Dec | — |
| 4 | TIC v8.2 / CTL | Stellar catalog | MAST/VizieR/bulk | `Catalogs.query_object(...,catalog="TIC")` ; VizieR `IV/39` | Tmag, Teff, R*, **contratio** | stellar props |
| 5 | Kepler LC | Space photometry | MAST/lightkurve | `lk.search_lightcurve(..., author="Kepler")` | **Training augmentation** | via KOI |
| 6 | K2 LC | Space photometry | MAST/lightkurve | `lk.search_lightcurve(..., author="K2"/"EVEREST")` | Extra labels/diversity | via k2pandc |
| 7 | Gaia DR3 | Astrometry+phot | ESA / astroquery.gaia | `Gaia.cone_search_async(coord,r)` | **Blend/contamination**, R*, RUWE | variability class |
| 8 | CHEOPS | Space photometry | DACE / pycheops | `Cheops.query_database(...)` | Hi-precision confirmation | — |
| 9 | CoRoT | Space photometry | IAS/VizieR | `Vizier.get_catalogs(...)` | Historical labels | some |
| 10 | NASA Exoplanet Archive `ps`/`pscomppars` | Catalog | TAP/astroquery | `NasaExoplanetArchive.query_criteria(table="pscomppars",...)` | **CONFIRMED labels + true params** | **yes** |
| 11 | NASA Archive `toi` | Catalog | TAP/astroquery | `query_criteria(table="toi",...)` | **TESS multi-class labels** | **yes (tfopwg_disp)** |
| 12 | NASA Archive `cumulative` (KOI) | Catalog | TAP/astroquery | `query_criteria(table="cumulative",...)` | **Kepler labels + FP flags** | **yes** |
| 13 | ExoFOP-TESS TOI CSV | Disposition DB | HTTP CSV | `pd.read_csv("...download_toi.php?output=csv")` | **Authoritative TFOPWG disp.** | **yes (CP/KP/EB/FP/PC)** |
| 14 | TESS-EBs HLSP | EB catalog | MAST CSV / CasJobs / VizieR `J/ApJS/258/16` | `pd.read_csv("...hlsp_tess-ebs...csv")` | **EB-class labels** | **yes (EB)** |
| 15 | Kepler EB catalog | EB catalog | Villanova / VizieR | `Vizier.get_catalogs(...)` | **EB labels + morphology** | **yes (EB)** |
| 16 | ZTF | Ground photometry | IRSA / ztfquery | `Irsa.query_region(...,catalog="ztf_objects_dr22")` | Independent period check | variable flags |
| 17 | ASAS-SN | Ground photometry | Sky Patrol API | `SkyPatrolClient()` | Independent confirmation | variable DB |
| 18 | WASP/SuperWASP, HATNet, KELT, NGTS, OGLE, Evryscope | Ground surveys | VizieR / ESO / project | `Vizier.query_region(...)` ; `Eso(...)` | Cross-validation, diversity | via NASA Archive `disc_facility` |
| 19 | HARPS/ESPRESSO/HIRES/NEID (RV) | Spectroscopy | DACE / ESO / KOA | `Spectroscopy.get_timeseries(target=...)` | **Mass → planet vs EB** | spectroscopic FP flags |
| 20 | SIMBAD | Object DB | astroquery.simbad | `Simbad.query_object(...)` | Known-variable cross-check | OTYPE |

---

## 9. Key references / URLs (all verified live)

- TESS bulk downloads: <https://archive.stsci.edu/tess/bulk_downloads.html> ; FFI-TP-LC-DV: <https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html>
- TIC/CTL (PS7 link): <https://archive.stsci.edu/tess/tic_ctl.html>
- TESS-SPOC HLSP: <https://archive.stsci.edu/hlsp/tess-spoc> ; QLP: <https://tess.mit.edu/qlp/>
- TESS-EBs HLSP: <https://archive.stsci.edu/hlsp/tess-ebs> ; Villanova Kepler EBs: <http://keplerEBs.villanova.edu>
- lightkurve (v2.6): <https://github.com/lightkurve/lightkurve> ; tutorial: <https://lightkurve.github.io/lightkurve/tutorials/3-science-examples/exoplanets-identifying-transiting-planet-signals.html>
- astroquery NExScI: <https://astroquery.readthedocs.io/en/stable/ipac/nexsci/nasa_exoplanet_archive.html> ; Exoplanet Archive TAP: <https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html>
- astroquery Gaia: <https://astroquery.readthedocs.io/en/stable/gaia/gaia.html> ; Gaia DR3 epoch_photometry: <https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_photometry/ssec_dm_epoch_photometry.html>
- astroquery Vizier: <https://astroquery.readthedocs.io/en/stable/vizier/vizier.html>
- ExoFOP-TESS: <https://exofop.ipac.caltech.edu/tess/> (CSV: `download_toi.php?sort=toi&output=csv`); NEA+ExoFOP paper: <https://arxiv.org/html/2506.03299v1>
- DACE / dace-query: <https://dace-query.readthedocs.io/en/latest/usage_examples.html> ; CHEOPS via DACE: <https://dace.unige.ch/tutorials/?tutorialId=14> ; pycheops: <https://github.com/pmaxted/pycheops>
- Kepler/K2 data products: <https://keplerscience.arc.nasa.gov/data-products.html>
- TLS: <https://github.com/hippke/tls> & <https://transitleastsquares.readthedocs.io/> ; wotan: <https://github.com/hippke/wotan>
- TESS LC format / quality flags: <https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview> ; tour: <https://spacetelescope.github.io/mast_notebooks/notebooks/TESS/beginner_how_to_use_lc/beginner_how_to_use_lc.html>
