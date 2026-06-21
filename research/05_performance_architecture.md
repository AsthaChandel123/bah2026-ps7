# 05 — High-Performance Computing, "O(1)" Data Access & Scalable Architecture

**BAH 2026 — Problem Statement 7: AI-enabled detection of exoplanets from noisy TESS light curves.**

Scope of this document: how to make a pipeline that ingests **~20,000–30,000 noisy TESS light curves per sector**, detrends, searches for transits, vets, classifies, and fits parameters — **fast, reproducible, and incrementally re-runnable.** The brief from the user is explicit: *"fastest platform"* and *"O(1) techniques."* In practice "O(1)" here means **amortise all repeated work to constant time via precomputation, caching, hashing, columnar random-access storage, and approximate-nearest-neighbour lookup**, and make the per-light-curve hot loop run at machine speed (compiled/vectorised/GPU) rather than in the Python interpreter.

> **Headline target.** A full sector (~25k LCs) must complete in *hours, not days*, on commodity hardware, and a *re-run after a code change to one stage* must be *minutes* (because everything upstream is cached). For comparison, the TESS team's own **Quick-Look Pipeline (QLP)** moved its transit search to GPU and now searches **an entire sector in ~1 day** ([QLP DRN-003, arXiv:2302.01293](https://arxiv.org/abs/2302.01293)); the **CETRA** GPU algorithm is *"up to a few orders of magnitude faster for high-cadence light curves"* than Transit Least Squares while finding *"at least 20 per cent more low-SNR transits"* ([arXiv:2503.20875](https://arxiv.org/abs/2503.20875)). We can hit the hours target on CPU with good engineering, and minutes with a GPU.

---

## 0. TL;DR — the recommended stack

| Layer | Recommendation | Why |
|---|---|---|
| **Storage / data lake** | **Parquet** (metadata, features, results) + **Zarr or HDF5** (raw/detrended flux arrays), partitioned by `sector/` then `tic_group/` | Columnar + predicate pushdown + chunked random access → O(1)-ish slicing of any TIC |
| **Hot numeric loops** | **NumPy vectorisation first**, then **Numba `@njit(parallel=True, fastmath=True, cache=True)`**, **`bottleneck`** for moving-window detrend, **SciPy/`pyfftw` FFT** | Removes Python-interpreter overhead; near-C speed; FFT turns O(N²) into O(N log N) |
| **Transit search** | CPU: **`astropy.timeseries.BoxLeastSquares`** (Cython) + **`transitleastsquares`** (Numba). GPU (optional, big win): **CETRA / cuvarbase** or a custom **JAX `vmap`+`jit`** batched BLS | TLS already Numba-compiled; GPU batches thousands of LCs at once |
| **Per-LC parallelism** | **`joblib.Parallel(n_jobs=-1)`** on one node; **Dask** (`bag`/`delayed`/`futures`) to scale to a cluster; **Ray** if you want actors/stateful workers | Pipeline is *embarrassingly parallel per light curve* |
| **ML inference** | **XGBoost / scikit-learn RandomForest** on CPU; **RAPIDS cuML / XGBoost `device=cuda`** + **Forest Inference Library (FIL)** on GPU; **PyTorch** for any CNN vetter | cuML RF is **20–45× faster** than sklearn on a V100 ([NVIDIA](https://developer.nvidia.com/blog/accelerating-random-forests-up-to-45x-using-cuml/)) |
| **Caching / "O(1)"** | **`functools.lru_cache`** (in-proc memo), **`joblib.Memory`** (disk memo of stage outputs), **`dict`/key-value index** of TIC→file offset, **`faiss`/`hnswlib`** ANN for shape lookup, **Bloom filter** for known-EB membership, **`numpy.memmap`** for huge arrays | Repeated work and lookups become constant-time |
| **Orchestration** | **Snakemake** (file-based DAG, native checkpoints, dry-run, HPC/cloud) — or **Prefect** if you want a Python-native task API with observability | Rule-level caching + resumable runs + provenance |
| **Config** | **Hydra** (hierarchical/compositional YAML + CLI overrides) validated by **pydantic** | Reproducible, override-able experiments |
| **CLI / packaging** | **Typer** CLI, **`pyproject.toml`** (PEP 621), **pytest**, pinned env | Production-grade ergonomics |
| **Reproducibility** | Fixed seeds, content-hashed inputs, versioned data dirs, logged config per run | Same input ⇒ same output, every time |

---

## A) CPU performance

The single biggest win for Python numeric code is **deleting Python-level loops**: every iteration of a `for` loop over samples pays interpreter, type-check, and boxing overhead. Push the loop into compiled code.

### A.1 NumPy vectorisation (do this first, always)
- **What:** express per-sample math as whole-array operations so the loop runs in C inside NumPy.
- **Where in pipeline:** sigma-clipping, normalisation (`flux / np.nanmedian(flux)`), phase-folding (`((t - t0) / P) % 1`), SNR/depth computation, vectorised χ² over a trial grid.
- **Benefit:** typically **10–100×** over a naive Python loop, zero extra dependencies.

### A.2 `bottleneck` — fast moving-window / NaN-aware reductions
- **What:** drop-in C implementations of `move_mean`, `move_median`, `move_std`, `nanmean`, `nanmedian`, `nansum`. TESS light curves are full of NaNs (masked cadences), and these are far faster than pandas `.rolling()` or `np.nan*`.
- **API:** `import bottleneck as bn; trend = bn.move_median(flux, window, min_count=1)`.
- **Where:** **detrending** (running-median/biweight baseline), rolling-stats outlier rejection.
- **Benefit:** often **5–10×** over pandas rolling and handles NaNs natively. ([bottleneck docs](https://bottleneck.readthedocs.io/))

### A.3 Numba JIT — `@njit`, `parallel=True`, `prange`, `fastmath`, `cache`
- **What:** compiles a Python function to machine code via LLVM. `nopython` (`njit`) mode is the fast path; `parallel=True` + `prange` auto-threads across cores; `fastmath` relaxes IEEE for speed; `cache=True` writes the compiled artifact to disk so you pay the ~seconds compile cost **once**, not every process start.
- **Where:** any custom inner loop the vectorised form can't express cleanly — e.g. a **box-search / matched-filter** over a trial-period × trial-duration grid, a custom biweight detrend, BLS folding/binning. (Note: **`transitleastsquares` is itself Numba-compiled**, which is why it's fast — [TLS FAQ](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html).)
- **Benefit:** **50–200×** over pure Python; with `parallel=True`, near-linear scaling across cores for embarrassingly-parallel inner loops. ([Numba parallel docs](https://numba.pydata.org/numba-doc/dev/user/parallel.html), [Performance tips](https://numba.pydata.org/numba-doc/dev/user/performance-tips.html))

```python
# pipeline/search/box_kernel.py
import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True, fastmath=True)
def boxsearch_snr(phase, flux, ivar, durations):
    """For each trial transit duration, find the phase window minimising chi2,
    return best (depth, snr, t0_phase, duration). Embarrassingly parallel over durations."""
    n = phase.size
    best_snr = np.full(durations.size, -1.0)
    best_depth = np.zeros(durations.size)
    best_phase = np.zeros(durations.size)
    order = np.argsort(phase)            # sort once outside is even better; shown here for clarity
    p = phase[order]; f = flux[order]; w = ivar[order]
    for di in prange(durations.size):    # <-- threads here
        dur = durations[di]
        half = dur * 0.5
        snr_best = -1.0
        # slide the box across phase
        start = 0
        for c in range(n):
            lo = p[c] - half; hi = p[c] + half
            # advance window start
            while p[start] < lo:
                start += 1
            sw = 0.0; swf = 0.0
            for k in range(start, n):
                if p[k] > hi:
                    break
                sw += w[k]; swf += w[k] * f[k]
            if sw <= 0:
                continue
            depth = (np.sum(w) * np.sum(w * f) / np.sum(w) - swf) / sw  # schematic
            snr = depth * np.sqrt(sw)
            if snr > snr_best:
                snr_best = snr; d_best = depth; ph_best = p[c]
        best_snr[di] = snr_best; best_depth[di] = d_best; best_phase[di] = ph_best
    return best_snr, best_depth, best_phase
```
> First call JIT-compiles (~1–3 s); thereafter it loads from `__pycache__`/numba cache instantly. Keep one warm-up call at process start so worker pools don't each eat the compile cost.

### A.4 FFT-accelerated algorithms
- **What:** replace O(N²) brute force with O(N log N). A **matched filter / cross-correlation** of a transit template against the light curve, or **autocorrelation** for period hunting, is an FFT. A **Lomb–Scargle periodogram** (`astropy.timeseries.LombScargle`, `method="fast"`) uses an FFT-based extirpolation (Press & Rybicki) and is excellent for a cheap first-pass period prior before the expensive BLS/TLS.
- **API:** `scipy.fft` / **`pyfftw`** (wraps FFTW; supports *cached wisdom/plans* — see §D), `numpy.fft`.
- **Where:** fast pre-screen for periodicity; template matched-filter; deconvolution of instrument response.
- **Benefit:** for N≈20k samples, **N/log N ≈ 1500×** fewer operations than the naive O(N²) correlation.

### A.5 Cython (escape hatch)
- **What:** compile annotated Python to a C extension. Use when Numba can't express the construct (complex structs, calling existing C libs) or you need a stable shippable wheel. **Astropy's `BoxLeastSquares` is already a Cython/C implementation** — you rarely need to write your own. ([Wōtan paper, IOP](https://iopscience.iop.org/article/10.3847/1538-3881/ab3984) compares the Cython Astropy BLS against Numba TLS.)
- **Benefit:** C-speed; more boilerplate than Numba. Prefer Numba unless you have a concrete reason.

### A.6 Memory layout & dtype
- **Contiguity:** keep arrays **C-contiguous** (`np.ascontiguousarray`) so the kernel walks memory linearly (cache-friendly). For a stack of LCs, store as `(n_lc, n_time)` row-major so each LC is contiguous.
- **dtype:** TESS flux precision does **not** need float64. Use **`float32`** for flux/time arrays:
  - **½ the memory** (a 25k×20k `float32` matrix ≈ **2 GB** vs 4 GB), **½ the memory-bandwidth**, and on AVX/GPU roughly **2× throughput** because twice as many lanes fit per SIMD register.
  - Keep **float64 only** for accumulators where catastrophic cancellation matters (e.g. summed χ², BJD time stamps where 1e-6-day precision over a 2457000+ BJD offset is required — store time as `bjd - 2457000.0` in float64 or as float32 offset).
- **Avoid copies:** prefer in-place ops (`out=`), views over slices, and `np.where`/boolean masks over Python comprehensions.

---

## B) GPU / accelerator

**When does GPU pay off for 25k LCs?** When the per-LC work is large and *uniform* enough to batch. Transit search is the obvious candidate: it's a brute-force scan over a big period×duration grid, identical structure for every LC. Detrend+search of a full sector on CPU is hours; on GPU it's minutes — this is exactly why the TESS QLP switched to GPU (**~1 day/sector** GPU search, [arXiv:2302.01293](https://arxiv.org/abs/2302.01293)). For the *classifier*, GPU pays off if you have many trees / large feature matrices.

### B.1 JAX — `jit` + `vmap` (+ `pmap`) for batched search & model eval
- **What:** NumPy-API array library that JIT-compiles (via XLA) to CPU/GPU/TPU and **auto-vectorises** with `vmap` (map a function over a batch axis with zero Python loop) and **auto-parallelises** across devices with `pmap`. Perfect for "run the same BLS/transit-model over 5,000 light curves at once."
- **API:** write the per-LC search as a pure function of `(time, flux, period)`, then `vmap` it over the period grid, then `vmap` again over the LC batch → one fused GPU kernel. Add `jax.jit` to compile.
- **Where:** **batched BLS/box-search**, **batched transit-model evaluation** (limb-darkened Mandel–Agol via `jaxoplanet`), batched χ² over period grids, and **gradient-based parameter fitting** (JAX gives you autodiff for free — couple with `numpyro`/`BlackJAX` for HMC posteriors on depth/period/duration). ([JAX](https://github.com/jax-ml/jax))
- **Benefit:** orders of magnitude on high-cadence data (mirrors CETRA's GPU result); plus free autodiff for the fitting stage.

```python
import jax, jax.numpy as jnp
from jax import vmap, jit

def single_period_snr(time, flux, period, t0_grid, dur):
    phase = ((time - t0_grid[:, None]) / period) % 1.0      # (n_t0, n_time)
    in_transit = jnp.abs(phase - 0.5) < (dur / period) / 2
    depth = jnp.sum(jnp.where(in_transit, flux, 0.0), 1) / jnp.sum(in_transit, 1)
    snr = depth * jnp.sqrt(jnp.sum(in_transit, 1))
    return jnp.max(snr)                                      # best t0 for this period

# vmap over periods, then over a batch of light curves -> one fused kernel
search_lc   = jit(vmap(single_period_snr, in_axes=(None, None, 0, None, None)))   # over periods
search_batch = jit(vmap(search_lc, in_axes=(None, 0, None, None, None)))          # over LCs
```

### B.2 CuPy — drop-in NumPy on CUDA
- **What:** `cupy` mirrors the NumPy API but runs on the GPU. Easiest "lift" if you already have vectorised NumPy: `import cupy as cp` and move arrays with `cp.asarray`.
- **Where:** detrending and folding of LC *batches* held as a `(n_lc, n_time)` GPU matrix; custom `RawKernel` for a bespoke box-search.
- **Benefit:** large speedups for memory-bound batched array math; minimal code change. Watch host↔device transfer — keep the whole batch resident on the GPU and only pull back the scalar results.

### B.3 RAPIDS cuML / cuDF, XGBoost-GPU, FIL — the classifier
- **What:** GPU DataFrames (`cuDF`) and GPU ML (`cuML`: RandomForest, KNN, SVM, UMAP/HDBSCAN). **cuML RandomForest is 20–45× faster** than scikit-learn on a single V100 ([NVIDIA blog](https://developer.nvidia.com/blog/accelerating-random-forests-up-to-45x-using-cuml/)); broader sklearn-acceleration is **5–175×** ([DataCamp](https://www.datacamp.com/blog/nvidia-cuml-GPU-scikit-learn)). **XGBoost** with `device="cuda"` (formerly `tree_method="gpu_hist"`) trains on GPU; **Forest Inference Library (FIL)** gives blazing batched inference for trees trained anywhere ([NVIDIA FIL](https://developer.nvidia.com/blog/supercharge-tree-based-model-inference-with-forest-inference-library-in-nvidia-cuml/)). `cuml.accel` even gives **zero-code-change** acceleration of existing sklearn scripts.
- **Where:** training the transit-vs-EB-vs-blend classifier on the curated labelled set; **batched inference** over all 25k candidates' feature vectors at once.
- **Benefit:** training minutes→seconds; classifying the whole sector in one GPU batch.

### B.4 PyTorch — DL vetters
- **What:** if you add a CNN that classifies *phase-folded + full-orbit views* (à la **Astronet/ExoMiner/Nigraha**, [arXiv:2101.09227](https://arxiv.org/abs/2101.09227)), PyTorch gives batched GPU inference with `torch.no_grad()`, `torch.compile`, AMP (fp16/bf16), and `DataLoader(num_workers=…)`.
- **Where:** the **vetting** stage (triage candidates before expensive fitting).
- **Benefit:** thousands of candidate views per second on one GPU.

**GPU economics rule of thumb for this project:** GPU is a *force multiplier on the transit search and the classifier*, which together dominate runtime. If a GPU is available, route those two stages to it; keep ingest/detrend on CPU workers (I/O-bound). If no GPU, the CPU plan (§A + §C) still hits the hours target.

---

## C) Parallel & distributed — the pipeline is embarrassingly parallel per light curve

Each light curve is processed **independently**; there is no cross-LC dependency until the final aggregation/report. This is the textbook *embarrassingly parallel* pattern ([Dask example](https://examples.dask.org/applications/embarrassingly-parallel.html)). Note that **Astropy BLS is single-threaded and TLS's internal threading scales sub-linearly**, so the right pattern is *one single-thread search instance per LC, fanned out across cores* — this gives **~linear scaling with core count** ([TLS FAQ](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html); the FAQ confirms running many single-thread instances over different LCs scales ~linearly).

### C.1 Single node: `joblib.Parallel` (recommended default)
- **What:** lightweight parallel for-loops with a `loky` process backend (sidesteps the GIL), batching, and progress. Pairs naturally with `joblib.Memory` caching (§D).
- **Benefit:** trivial to add; near-linear speedup on N cores for CPU-bound per-LC work.

```python
# pipeline/driver.py
from joblib import Parallel, delayed
from .stages import process_one_lc          # ingest->detrend->search->vet->classify->fit

def run_sector(tic_ids, cfg, n_jobs=-1):
    return Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto", verbose=10)(
        delayed(process_one_lc)(tic, cfg) for tic in tic_ids
    )
```

### C.2 `concurrent.futures` / `multiprocessing`
- **What:** stdlib `ProcessPoolExecutor.map(...)` for the same fan-out without extra deps; `ThreadPoolExecutor` only for the **I/O-bound download/ingest** stage (threads are fine when waiting on network/disk, and Numba/NumPy release the GIL during compute).
- **Where:** download stage → threads; compute stages → processes.

### C.3 Dask — scale from laptop to cluster
- **What:** `dask.delayed`/`dask.bag` express the same per-LC graph; the distributed scheduler runs it across many machines, with a dashboard, spill-to-disk, and **scaling to thousand-node clusters** ([Dask & Ray, SFU](https://ggbaker.ca/732/content/dask-ray.html)). `dask.dataframe` reads the **partitioned Parquet** lake lazily with predicate pushdown.
- **Where:** when one node isn't enough, or to process *all sectors*; also great for the final group-by/aggregation into the results table.
- **Benefit:** same code, bigger box; resilient and observable.

```python
import dask.bag as db
bag = db.from_sequence(tic_ids, npartitions=256)
results = bag.map(lambda t: process_one_lc(t, cfg)).compute()   # runs on the cluster
```

### C.4 Ray — tasks & stateful actors
- **What:** `@ray.remote` tasks for fan-out; **actors** hold state (e.g. a worker that loads the trained classifier *once* and reuses it across thousands of candidates, avoiding repeated model-load cost). Decentralised scheduler, Plasma shared-memory object store for zero-copy hand-off ([Domino: Spark/Dask/Ray](https://domino.ai/blog/spark-dask-ray-choosing-the-right-framework)).
- **Where:** if you want long-lived GPU-resident model actors, or a service-style deployment.

**Batching strategy.** Don't dispatch 25k tasks of one LC each (scheduler overhead dominates) — **batch ~50–200 LCs per task** so each task does meaningful work and amortises model-load/JIT-warmup. `joblib`'s `batch_size="auto"` and Dask `npartitions` control this.

---

## D) "O(1)" data-access & lookup techniques (the core of the brief)

The phrase "O(1) techniques" is best delivered as: **amortise repeated work to constant time** (memoise/precompute) and **make every lookup constant-time** (hash/index/columnar). Below, each pattern says *what it does*, *the API*, and *exactly where it goes in this pipeline.*

### D.1 Disk-backed memoisation of stage outputs — `joblib.Memory`
- **What:** transparently caches a function's return value on disk keyed by a hash of its arguments. Re-calling with the same args is a constant-time disk load instead of recomputation. Ideal for **expensive, idempotent stages with complex array I/O** (detrend, search). ([joblib Memory docs](https://joblib.readthedocs.io/en/latest/memory.html))
- **Where:** wrap `detrend(tic, raw_flux, cfg)` and `search(tic, detrended, cfg)`. Change code in `classify` and re-run → ingest/detrend/search are served from cache; **only the changed stage recomputes.** This is the single biggest "re-run in minutes" lever.

```python
from joblib import Memory
mem = Memory(location=".cache/joblib", verbose=0)

@mem.cache
def detrend(tic_id, flux, window, cfg_hash):     # cfg_hash makes cache config-aware
    return biweight_detrend(flux, window)
```

### D.2 In-process memoisation — `functools.lru_cache`
- **What:** O(1) cache of recent calls inside one process, no disk. Best for *small* args/returns called repeatedly ([joblib docs note Memory-vs-memoize trade-off](https://joblib.readthedocs.io/en/latest/memory.html)).
- **Where:** **precomputed period grids** (`get_period_grid(P_min, P_max, n)`), limb-darkening coefficient lookups by `(Teff, logg)`, star-parameter fetches by TIC, FFT plan retrieval — anything pure and hot.

```python
from functools import lru_cache
@lru_cache(maxsize=None)
def period_grid(p_min, p_max, n, oversample):
    return np.geomspace(p_min, p_max, int(n * oversample)).astype(np.float32)
```

### D.3 Hash-based O(1) retrieval of light curves — dict / key-value index + manifest
- **What:** build a **manifest** mapping `TIC ID → (file path, row-group / chunk offset, byte range)` once at ingest. A Python `dict` (or RocksDB/LMDB/SQLite for out-of-core) then resolves any TIC to its bytes in **O(1)**, instead of scanning the directory or a giant table.
- **Where:** the **ingest index**. Persist as a small Parquet/`parquet`+`dict` you load at startup. Every later stage does `path, off = manifest[tic]` — constant time regardless of 25k or 2M LCs.

### D.4 Columnar on-disk formats with O(1)-ish random access + predicate pushdown
- **Parquet** (tables: metadata, extracted features, candidate results): columnar, compressed, with **row-group min/max statistics enabling predicate pushdown** so a filtered read *skips* non-matching row-groups and unselected columns entirely ([Arrow datasets](https://arrow.apache.org/docs/python/dataset.html)). **Partition by `sector=…/tic_group=…`** so a query for one sector touches only that directory (partition pruning — *"avoid loading files at all if they contain no matching rows"*).
- **Zarr** (raw & detrended flux *arrays*): chunked, compressed N-D arrays where **each chunk is an independently addressable object**, giving O(1)-ish random access to any `(lc, time-window)` slice and excellent cloud/parallel behaviour ([Earthmover: What is Zarr](https://www.earthmover.io/blog/what-is-zarr/)). Store the sector as a `(n_lc, n_time)` chunked array.
- **HDF5** (alternative to Zarr; single-file): hierarchical, chunked, seek-anywhere; *"what supercomputers use in HPC"* once tuned ([format comparison](https://medium.com/towards-data-engineering/emergence-of-modern-file-formats-in-data-pipelines-and-storage-b8bf22c24a95)). Use `h5py` with chunk shape ≈ one LC.
- **Feather/Arrow IPC:** fastest *in-memory*-to-disk round-trip for interim hand-off between stages (zero-copy via Arrow).
- **Where / rule:** **arrays → Zarr/HDF5; tables → Parquet; interim → Feather.** Partition both by `sector` then a TIC bucket (e.g. `tic % 256`) so any slice is a small, local read.

```python
# Read just the rows you need — predicate pushdown + partition pruning, no full scan
import pyarrow.dataset as ds
dset = ds.dataset("lake/results", format="parquet", partitioning="hive")
hot = dset.to_table(filter=(ds.field("sector") == 40) & (ds.field("snr") > 7.0),
                    columns=["tic", "period", "depth", "duration", "snr"])
```

### D.5 Approximate nearest neighbour (LSH/ANN) for O(1)-ish shape lookup — `faiss` / `hnswlib` / `annoy`
- **What:** index fixed-length **light-curve embeddings** (e.g. phase-folded binned vector, BLS/TLS feature vector, or a learned embedding) so you can answer *"what known signals look like this?"* in sub-linear (effectively constant) time instead of comparing against all references. **HNSW graphs** give the best recall/speed and **FAISS** adds GPU and IVF/PQ compression ([ANN-Benchmarks](https://ann-benchmarks.com/), [Zilliz FAISS vs HNSWlib](https://zilliz.com/blog/faiss-vs-hnswlib-choosing-the-right-tool-for-vector-search)).
- **Where (three concrete uses):**
  1. **Known-signal matching / triage:** match a candidate's shape against a library of confirmed planets / EBs / blends to seed the classifier or flag obvious EBs.
  2. **Deduplication:** the same astrophysical signal can appear on neighbouring TICs via blending — ANN finds near-duplicate light-curve shapes so you don't report a blend N times.
  3. **Few-shot label propagation:** nearest labelled neighbours vote on an unlabelled candidate.
- **Benefit:** brute-force is O(N·d) per query (25k × dim); HNSW query is ~O(log N) ≈ constant in practice. Build the index once (FAISS IVF-PQ builds in ~60–230 s for 1M vectors — [benchmark](https://arxiv.org/pdf/2412.01555)), query forever.

```python
import hnswlib, numpy as np
dim = 256                                   # embedding length
idx = hnswlib.Index(space="l2", dim=dim)
idx.init_index(max_elements=200_000, ef_construction=200, M=16)
idx.add_items(reference_embeddings, reference_ids)   # build ONCE
idx.set_ef(64)
labels, dists = idx.knn_query(candidate_embeddings, k=10)   # O(log N)-ish per query
```

### D.6 Bloom filter — O(1) membership tests
- **What:** a compact bit-array + k hash functions answering *"have we seen X?"* in **O(1)** with zero false-negatives (a "no" is definitive; a "yes" may rarely be a false positive — then do the exact check) ([bloom-filter explainer](https://people.duke.edu/~ccc14/sta-663-2017/17B_Big_Data_Structures.html)).
- **Where:**
  - *"Is this TIC a known eclipsing binary / known TOI / already-processed?"* — load the known-EB and TOI catalogues into a Bloom filter at startup; gate the expensive vetting/fitting on a constant-time membership test.
  - **Idempotent resumes:** a Bloom filter of already-completed TICs lets the driver skip finished work instantly on restart (backed by the exact manifest for the rare false-positive).
- **API:** `pybloom-live`, `rbloom`.

### D.7 Precomputed period grids & cached FFT plans
- **What:** the trial-**period grid** depends only on `(P_min, P_max, oversampling, baseline)` — compute once, `lru_cache` it, reuse for all 25k LCs (don't rebuild per LC). Same for **duration grids**. For FFT-based steps, **`pyfftw` caches plans/"wisdom"** so repeated transforms of the same length skip planning.
- **Where:** search stage setup. TLS's own speed comes partly from this style of precomputation — it caches to avoid **~96% of redundant calculations** ([TLS FAQ](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html)).

### D.8 Cache detrended & phase-folded views (intermediate artifacts)
- **What:** detrending and phase-folding are deterministic given config; persist them (Zarr chunk per LC) so vetting/plotting/refitting never recompute them.
- **Where:** outputs of detrend and (best-period) fold → cached artifacts keyed by `(tic, cfg_hash)`. Visualisation and the human-vetting UI then read **O(1)**.

### D.9 Feature store / embeddings cache for the classifier
- **What:** persist the per-candidate feature vector (BLS/TLS stats, odd-even depth, secondary-eclipse metric, shape embedding) in a Parquet "feature store" keyed by TIC. Training and re-training read features directly — no recompute from raw flux. Cache CNN embeddings the same way for the ANN index.
- **Where:** between vetting and classification; this also makes ablation/retraining experiments fast.

### D.10 Memory-mapping huge arrays — `numpy.memmap`
- **What:** map an on-disk array into the address space so you can **slice/index it like RAM without loading it all**; the OS pages in only the touched chunks → O(1) random access to any LC's row, with memory bounded by working set ([NumPy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)).
- **Where:** the consolidated `(n_lc, n_time)` `float32` flux matrix for a sector (≈2 GB) — workers memmap it read-only and each grabs its rows. Combine with Parquet/Zarr for the canonical store and memmap for the hot compute matrix. (Caveat: random access to a *huge* memmap on cold cache can be slow — keep chunks/rows contiguous and prefer sequential-ish access per worker.)

#### Summary table — "O(1)" data-access patterns

| Pattern | Library / API | Complexity benefit | Where in THIS pipeline |
|---|---|---|---|
| Disk memoisation of stage outputs | `joblib.Memory.cache` | Recompute → O(1) cache load | detrend, search; *re-run only changed stage* |
| In-proc memoisation | `functools.lru_cache` | O(1) repeat calls | period/duration grids, limb-darkening, star params, FFT plans |
| Hash index TIC→bytes | `dict`, LMDB/SQLite, Parquet manifest | Directory scan → O(1) lookup | ingest manifest used by every stage |
| Columnar + predicate pushdown | **Parquet** + `pyarrow.dataset` | Full scan → read only matching row-groups/cols | metadata, features, results tables |
| Chunked N-D array random access | **Zarr** / **HDF5** (`h5py`) | Load-all → O(1)-ish chunk read | raw & detrended flux arrays |
| Zero-copy interim hand-off | **Feather/Arrow IPC** | Fast round-trip between stages | stage-to-stage intermediates |
| Approx. nearest neighbour (LSH) | **faiss** / **hnswlib** / **annoy** | O(N·d) → ~O(log N) per query | known-signal match, dedup, label propagation |
| Bloom filter membership | `pybloom-live` / `rbloom` | Set lookup → O(1), tiny memory | known-EB/TOI gate; skip-completed on resume |
| Precomputed grids / FFT plans | `lru_cache`, `pyfftw` wisdom | Rebuild-per-LC → build once | search setup |
| Cached detrended/folded views | Zarr chunk per `(tic,cfg)` | Recompute → O(1) read | vetting, plotting, refitting |
| Feature / embedding store | Parquet keyed by TIC | Recompute features → O(1) read | classifier train + inference |
| Memory-mapped array | `numpy.memmap` | Load-all-into-RAM → O(1) row access | sector flux matrix for workers |

---

## E) Pipeline architecture & MLOps

### E.1 Modular stage design (single responsibility, cached boundaries)
```
ingest → detrend → search → vet → classify → fit → report
```
Each stage: **pure-ish function**, typed I/O, reads from and writes to the data lake, and is **independently cacheable** (joblib.Memory / Snakemake rule). Crossing a stage boundary writes a durable artifact (Parquet/Zarr), so any stage can be re-run in isolation and downstream stages pick up the cached input.

- **ingest** — resolve TICs, download (`lightkurve`/MAST or local FITS), build the **manifest** (D.3), write raw flux to Zarr, write per-LC metadata to Parquet.
- **detrend** — `bottleneck`/Numba biweight or `wōtan`; cache outputs.
- **search** — Astropy BLS + TLS (CPU) or JAX/CETRA (GPU); precomputed grids (D.7); emit candidate periods/depths/durations/SNR.
- **vet** — odd/even depth, secondary-eclipse, V-shape vs U-shape, centroid checks; **Bloom-filter** known-EB gate (D.6); optional CNN.
- **classify** — XGBoost/RF (CPU) or cuML/FIL (GPU) over the **feature store** (D.9): transit / eclipse / blend / other + probability.
- **fit** — limb-darkened transit fit (`batman`/`exoplanet`/`jaxoplanet`) for confirmed transits → period, depth, duration + uncertainties (MCMC/HMC or least-squares + Fisher).
- **report** — visualisation (folded LC + model + classification + confidence) and the results table; per the PS deliverable.

### E.2 Configuration — Hydra + pydantic
- **Hydra** composes hierarchical YAML and supports CLI overrides (`python -m pipeline run search.oversample=5`), making each run a reproducible config tree ([Towards Data Science](https://towardsdatascience.com/configuration-management-for-model-training-experiments-using-pydantic-and-hydra-d14a6ae84c13/)). **pydantic** validates/coerces the resolved config into typed objects (constrained types, defaults), failing fast on bad params. Hydra writes the *exact* config used into each run's output dir → provenance.

### E.3 Orchestration / DAG — Snakemake (recommended) or Prefect
- **Snakemake**: Python-based, **file-driven rules** that declare inputs/outputs; gives **native checkpoints** (251 checkpoint references in its docs per the comparison study), **dry-run**, **DAG visualisation**, automatic re-execution of only out-of-date rules, and **HPC/cloud** execution ([Snakemake vs Nextflow study](https://dl.acm.org/doi/fullHtml/10.1145/3676288.3676290)). Its file-output model maps perfectly onto our cached-artifact boundaries: touch one rule's code, `snakemake` recomputes only the affected sub-DAG. *"If your team lives in Python and runs on on-prem clusters, Snakemake gets you there with less ceremony"* ([Cytogence](https://www.cytogence.com/blog/reproducible-pipelines-nextflow-vs-snakemake/)).
- **Prefect**: Python-native task/flow API with retries, caching, scheduling, and a nice observability UI — choose if you want a programmatic flow rather than a rules file. (Nextflow is the cloud/HPC-at-massive-scale option but adds a Groovy DSL learning curve — overkill here.)

### E.4 Incremental / streaming & checkpointing
- **Incremental:** the manifest + per-stage cache means new TICs (or a new sector) only process the *delta*. The **Bloom filter of completed TICs** (D.6) makes "skip already-done" an O(1) test on resume.
- **Checkpointing:** write results in **batches** (e.g. flush every 200 LCs to a partitioned Parquet) so a crash loses at most one batch; the driver resumes from the manifest.
- **Streaming:** process LCs as they download (producer/consumer queue: download threads → compute processes), so search starts before all 25k are on disk.

### E.5 CLI, packaging, logging, testing, reproducibility
- **CLI:** **Typer** (type-hint-driven) subcommands: `pipeline ingest`, `pipeline run --sector 40`, `pipeline report`. (Click is the lower-level alternative.)
- **Packaging:** **`pyproject.toml`** (PEP 621), `src/` layout, pinned `requirements.lock`/`uv`/`conda` env for reproducible installs.
- **Logging:** structured logging (`logging`/`structlog` or `loguru`) with per-stage timers; emit a per-run JSON of timings + config hash.
- **Testing:** `pytest` unit tests on detrend/search/feature kernels with synthetic injected transits (known answers); a small end-to-end smoke test on a handful of real LCs.
- **Reproducibility:** fix all seeds (`numpy`, `random`, framework), **content-hash inputs** into cache keys, version data dirs (`data/processed/v1/…`), and record library versions per run. Same input ⇒ identical output.

### E.6 Recommended project structure
```
bah2026-ps7/
├── pyproject.toml                 # PEP 621 metadata, deps, entry points
├── README.md
├── requirements.lock              # pinned env (or environment.yml / uv.lock)
├── Snakefile                      # DAG: ingest->detrend->search->vet->classify->fit->report
├── configs/                       # Hydra config tree
│   ├── config.yaml                # defaults / composition root
│   ├── data/sector40.yaml
│   ├── detrend/biweight.yaml
│   ├── search/{bls.yaml,tls.yaml,jax_gpu.yaml}
│   ├── classify/{xgb.yaml,cuml_rf.yaml}
│   └── compute/{local.yaml,dask_cluster.yaml,gpu.yaml}
├── src/
│   └── exopipe/
│       ├── __init__.py
│       ├── cli.py                 # Typer entry point
│       ├── config.py              # pydantic schemas validating Hydra config
│       ├── io/
│       │   ├── manifest.py        # TIC -> (path, offset) hash index  (D.3)
│       │   ├── store.py           # Parquet/Zarr/HDF5 read-write helpers (D.4)
│       │   └── cache.py           # joblib.Memory + lru_cache wrappers   (D.1/D.2)
│       ├── stages/
│       │   ├── ingest.py
│       │   ├── detrend.py         # bottleneck/Numba/wotan
│       │   ├── search.py          # BLS + TLS (CPU) / JAX (GPU)
│       │   ├── vet.py             # odd-even, secondary, bloom gate (D.6)
│       │   ├── classify.py        # XGB/RF or cuML/FIL over feature store
│       │   ├── fit.py             # batman/jaxoplanet transit fit + uncertainties
│       │   └── report.py          # plots + results table
│       ├── kernels/
│       │   ├── box_kernel.py      # @njit(parallel=True) box search   (A.3)
│       │   └── jax_search.py      # jit+vmap batched search           (B.1)
│       ├── features/
│       │   ├── extract.py         # feature engineering
│       │   └── store.py           # Parquet feature store             (D.9)
│       ├── index/
│       │   └── ann.py             # faiss/hnswlib similarity index    (D.5)
│       └── driver.py              # joblib/Dask batch fan-out         (C.1/C.3)
├── data/                          # gitignored
│   ├── raw/                       # downloaded FITS / raw flux
│   ├── lake/                      # Parquet (hive-partitioned by sector/tic_group)
│   │   ├── metadata/  features/  results/
│   ├── arrays/                    # Zarr/HDF5 flux matrices
│   └── processed/v1/              # versioned outputs
├── models/                        # trained classifier + ANN index artifacts
├── .cache/joblib/                 # disk memo
├── notebooks/                     # exploration only
├── reports/                       # the 3-page PS report + figures
└── tests/                         # pytest (synthetic-injection unit + e2e smoke)
```

---

## F) Benchmarks / numbers & memory budget

### F.1 Realistic per-LC timing
- **TLS** searches an entire **unbinned Kepler K2 light curve (90 d, ~4,000 points)** in *"a few seconds on a typical laptop"*; internally it performs **~3×10⁸ model evaluations across ~8,500 trial periods** at **~230 ns/evaluation** on a 2.4 GHz i5, hitting **72% of theoretical FLOP throughput** and avoiding **~96% of redundant calculations** via caching ([TLS FAQ](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html)). TESS high-cadence (2-min) LCs of a full sector (~27 d, **~19,000 points**) are larger → budget **~2–10 s/LC for TLS**, **~0.3–1 s/LC for Astropy BLS** (coarser grid), **~0.1–0.5 s/LC for detrend** with `bottleneck`/Numba.
- **Detrending** with GP/biweight on 10⁴-point data: *"acceptable speed (10 seconds)"* for the heaviest estimators in `wōtan`; the **biweight slider is far faster** (sub-second) ([Wōtan, IOP](https://iopscience.iop.org/article/10.3847/1538-3881/ab3984)) — use biweight as default, reserve GP for flagged LCs.
- **Oversampling** scales runtime **linearly** (factor 2–5 typical; test empirically — [TLS FAQ](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html)). Keep a coarse grid for the *screen*, fine grid only for *candidates*.

### F.2 Hitting full-sector throughput
- **CPU plan:** assume **~3 s/LC** average (detrend + BLS screen + TLS on survivors). 25k LCs × 3 s = **75,000 CPU-seconds ≈ 20.8 core-hours**. On a **32-core** node with ~linear `joblib` scaling → **≈40 min/sector**; on **64 cores → ≈20 min**. (TLS confirms ~linear scaling running one single-thread instance per LC across cores — [TLS FAQ](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html).) **Two-stage screening** (cheap BLS/Lomb–Scargle first, expensive TLS only on the top few % SNR) cuts this several-fold.
- **GPU plan:** batch the search with JAX/CETRA/cuvarbase. CETRA is *"up to a few orders of magnitude faster"* on high-cadence LCs ([arXiv:2503.20875](https://arxiv.org/abs/2503.20875)); QLP's GPU search does **a whole sector in ~1 day** *including* its heavier processing ([arXiv:2302.01293](https://arxiv.org/abs/2302.01293)). A lean batched search on a modern GPU brings the **search** stage to **minutes**.
- **Classifier:** training on the curated set is **seconds** on GPU (cuML RF **20–45×** sklearn — [NVIDIA](https://developer.nvidia.com/blog/accelerating-random-forests-up-to-45x-using-cuml/)); inference over all 25k candidates is one batched call.
- **Re-run after a code change:** with `joblib.Memory`/Snakemake, unchanged upstream stages are cache hits → a classifier tweak re-runs the sector in **minutes**, not from scratch.

### F.3 Memory budget
- One sector as **`float32` `(25,000 × 19,000)`** flux matrix ≈ **1.9 GB** (vs 3.8 GB float64). Add time + quality arrays → ~**3–4 GB** resident if fully loaded; with **Zarr chunking + `numpy.memmap`**, working-set memory is bounded by the active batch (e.g. 200 LCs ≈ tens of MB).
- Keep **float32 for flux**, **float64 only for time (BJD-offset) and χ² accumulators**.
- ANN index for ~10⁵ reference embeddings @256-d float32 ≈ **100 MB** (HNSW) — trivially fits in RAM; build once, persist to `models/`.
- Per-worker footprint: detrended LC + grids + one search ≈ tens of MB → a 32-worker pool fits comfortably in **<16 GB**.

---

## Key sources
- TESS Quick-Look Pipeline GPU transit search — full sector in ~1 day: [arXiv:2302.01293](https://arxiv.org/abs/2302.01293)
- CETRA GPU transit detection (orders-of-magnitude faster, +20% low-SNR recovery): [arXiv:2503.20875](https://arxiv.org/abs/2503.20875)
- Transit Least Squares (Numba-compiled) runtime/caching FAQ: [transitleastsquares.readthedocs.io/FAQ](https://transitleastsquares.readthedocs.io/en/latest/FAQ.html)
- Wōtan detrending (biweight vs GP speed; BLS-vs-TLS comparison): [IOPscience 10.3847/1538-3881/ab3984](https://iopscience.iop.org/article/10.3847/1538-3881/ab3984)
- Nigraha ML TESS pipeline (architecture reference): [arXiv:2101.09227](https://arxiv.org/abs/2101.09227)
- Numba parallel & performance: [parallel](https://numba.pydata.org/numba-doc/dev/user/parallel.html), [performance tips](https://numba.pydata.org/numba-doc/dev/user/performance-tips.html)
- JAX (jit/vmap/pmap): [github.com/jax-ml/jax](https://github.com/jax-ml/jax)
- RAPIDS cuML RandomForest 20–45×: [NVIDIA blog](https://developer.nvidia.com/blog/accelerating-random-forests-up-to-45x-using-cuml/); FIL inference: [NVIDIA FIL](https://developer.nvidia.com/blog/supercharge-tree-based-model-inference-with-forest-inference-library-in-nvidia-cuml/); cuML-accel 5–175×: [DataCamp](https://www.datacamp.com/blog/nvidia-cuml-GPU-scikit-learn)
- joblib `Memory` disk cache: [joblib.readthedocs.io/memory](https://joblib.readthedocs.io/en/latest/memory.html)
- Dask embarrassingly parallel: [examples.dask.org](https://examples.dask.org/applications/embarrassingly-parallel.html); Dask vs Ray: [SFU](https://ggbaker.ca/732/content/dask-ray.html); Spark/Dask/Ray: [Domino](https://domino.ai/blog/spark-dask-ray-choosing-the-right-framework)
- Parquet predicate pushdown / partitioning: [Apache Arrow datasets](https://arrow.apache.org/docs/python/dataset.html)
- Zarr chunked random access: [Earthmover](https://www.earthmover.io/blog/what-is-zarr/); format comparison: [Towards Data Engineering](https://medium.com/towards-data-engineering/emergence-of-modern-file-formats-in-data-pipelines-and-storage-b8bf22c24a95)
- ANN benchmarks (FAISS/HNSW/Annoy): [ann-benchmarks.com](https://ann-benchmarks.com/); [Zilliz FAISS vs HNSWlib](https://zilliz.com/blog/faiss-vs-hnswlib-choosing-the-right-tool-for-vector-search); [FAISS/Annoy benchmark](https://arxiv.org/pdf/2412.01555)
- Bloom filters & memmap (big-data structures): [Duke STA-663](https://people.duke.edu/~ccc14/sta-663-2017/17B_Big_Data_Structures.html); [NumPy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- Snakemake/Prefect/Nextflow comparison: [ACM study](https://dl.acm.org/doi/fullHtml/10.1145/3676288.3676290); [Cytogence](https://www.cytogence.com/blog/reproducible-pipelines-nextflow-vs-snakemake/)
- Hydra + pydantic config: [Towards Data Science](https://towardsdatascience.com/configuration-management-for-model-training-experiments-using-pydantic-and-hydra-d14a6ae84c13/)
