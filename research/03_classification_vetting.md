# PS7 Research — Classification (ML/DL) + Astrophysical Vetting

**Scope:** Methods to classify TESS transit signals into four classes:
**`transit` (planet) / `EB` (eclipsing binary) / `blend` (contamination from nearby/background source) / `other` (starspots, pulsations, systematics, scattered light).**
Each classification must carry a **calibrated confidence**.

This document covers (A) deep-learning view-based classifiers, (B) classical tabular ML, (C) physics-based vetting diagnostics (each a *computable test*), (D) imbalance/calibration/cross-validation, and (E) a recommended ensemble. It contains runnable code snippets and a feature list.

> **Key mapping to PS7 classes.** The published TESS triage taxonomy maps almost 1:1 onto our four labels:
> - **PC** (planet candidate) → `transit`
> - **EB** (eclipsing binary) → `EB`
> - **NEB/BEB/contamination scenarios** (nearby/background eclipsing source) → `blend`
> - **V** (stellar variability), **IS/J** (instrumental/junk), scattered light → `other`
>
> Yu et al. (2019) explicitly used the labels **PC, EB, V (variability), IS (instrumental)** for TESS TCEs, plus **O (other)/J (junk)** at the vetting stage. This is the natural class set for our problem. (Source: [arXiv:1904.02726](https://arxiv.org/pdf/1904.02726), [IOPscience](https://iopscience.iop.org/article/10.3847/1538-3881/ab21d6))

---

## 0. Executive summary of recommended architecture

A **three-stream ensemble** producing a single calibrated 4-class probability vector:

1. **DL view-based classifier (ExoMiner-style multi-branch 1D CNN).** Inputs = phase-folded *views*: global flux (2001 bins), local flux (201 bins), secondary-eclipse local view, odd & even local views, centroid (in/out-of-transit) view, plus scalar stellar/DV features. This stream is the strongest single discriminator and natively encodes the physics that separates planet/EB/blend.
2. **Tabular ML classifier (XGBoost/LightGBM + Random Forest).** Inputs = engineered transit + stellar + vetting-metric features (period, depth, duration/period, odd–even depth diff, secondary depth, V-shape metric, transit SNR, centroid offset, CROWDSAP, ρ\*-consistency, etc.). Fast, interpretable, robust on small data.
3. **Deterministic vetting-flag layer.** A bank of physics-based diagnostics (LEO-vetter / TRICERATOPS-style) that produce hard flags and a statistical false-positive probability (FPP/NFPP). These both *feed features* to streams 1–2 and act as a final **veto/override** (e.g., a confirmed centroid offset forces `blend`; a deep secondary forces `EB`).

A **meta-learner (stacking)** combines the per-stream probabilities; outputs are **probability-calibrated** (isotonic/Platt or temperature scaling) and reported as the confidence. Cross-validation is **grouped by host star and by sector** to prevent leakage.

---

# A) Deep-learning classifiers (published architectures)

### A1. AstroNet — Shallue & Vanderburg (2018)
**What it is:** The foundational 1D CNN for transit vetting. Takes a phase-folded light curve as **two views** through **two disjoint convolutional columns**, concatenates them, then shared fully-connected layers → sigmoid (planet vs not-planet). Kepler accuracy ≈ 96%, AUC ≈ 0.988.
**Repo:** Google `exoplanet-ml` → `astronet/` ([github.com/google-research/exoplanet-ml](https://github.com/google-research/exoplanet-ml)). Paper: [2018AJ....155...94S](https://iopscience.iop.org/article/10.3847/1538-3881/aa9e09).

**Exact architecture (from `astro_cnn_model/configurations.py`, `local_global` config):**
- **Global view:** 2001 bins (full phase-folded orbit). Conv column: `cnn_num_blocks=5`, `cnn_block_size=2` (2 conv layers per block), `cnn_initial_num_filters=16`, `cnn_block_filter_factor=2` (filters double each block: 16→32→64→128→256), `cnn_kernel_size=5`, max-pool `pool_size=5, pool_strides=2`.
- **Local view:** 201 bins (zoom on the transit, ±~2–4 transit durations). Conv column: `cnn_num_blocks=2`, `cnn_block_size=2`, `cnn_initial_num_filters=16`, `cnn_block_filter_factor=2`, `cnn_kernel_size=5`, max-pool `pool_size=7, pool_strides=2`.
- **Combine:** flatten both columns → concatenate → **4 fully-connected layers × 512 units** (`local_global`) → 1 logit → sigmoid.
- **Preprocessing:** flatten/detrend out-of-transit (spline), phase-fold on the candidate period, **median-bin** into the fixed bin counts, then **normalize** so the median = 0 and the transit depth = −1 (so depth is scale-free). Local view zooms to the transit window and rebins.

**How it separates classes:** the **global view** exposes secondary eclipses and out-of-transit variability (EB/variable signatures); the **local view** captures transit *shape* (U-shaped planet vs V-shaped EB). By design AstroNet is binary; for PS7 we replace the final sigmoid with a **softmax over 4 classes**.

```python
# Pseudo-architecture of AstroNet (Keras), extended to 4 classes
import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_column(x, n_blocks, init_filters=16, k=5, pool=5, pstride=2):
    f = init_filters
    for b in range(n_blocks):
        for _ in range(2):                       # cnn_block_size = 2
            x = layers.Conv1D(f, k, padding="same", activation="relu")(x)
        x = layers.MaxPool1D(pool_size=pool, strides=pstride)(x)
        f *= 2                                    # cnn_block_filter_factor = 2
    return layers.Flatten()(x)

g_in = layers.Input((2001, 1), name="global_view")
l_in = layers.Input((201, 1),  name="local_view")
g = conv_column(g_in, n_blocks=5, pool=5)
l = conv_column(l_in, n_blocks=2, pool=7)
h = layers.Concatenate()([g, l])
for _ in range(4):                                # 4 FC layers x 512
    h = layers.Dense(512, activation="relu")(h)
out = layers.Dense(4, activation="softmax")(h)    # transit/EB/blend/other
model = Model([g_in, l_in], out)
```

### A2. Astronet-Triage / Astronet-Vetting — Yu et al. (2019) (TESS)
**What it is:** AstroNet ported to TESS, run as a **two-stage cascade**. Paper: [arXiv:1904.02726](https://arxiv.org/pdf/1904.02726) / [IOPscience](https://iopscience.iop.org/article/10.3847/1538-3881/ab21d6). Repos: [yuliang419/Astronet-Triage](https://github.com/yuliang419/Astronet-Triage), Astronet-Vetting.
- **Triage:** multi-class on TESS TCEs into **PC, EB, V (stellar variability), IS (instrumental artifact)** — keeps "eclipse-like" events (PC+EB), discards V+IS. Trained on **16,516 labelled TCEs** (493 PC, 2,155 EB, 13,868 noise/systematics) from QLP light curves, Sectors 1–5.
- **Vetting:** finer separation of PC from astrophysical false positives, adding a **secondary-eclipse view** and **centroid view** beyond global+local. Vetting dispositions: PC / EB / IS / V / O / J.
**Why it matters for PS7:** this is the *direct precedent* for our 4-class problem and the source of the label taxonomy above. The added **secondary** and **centroid** views are exactly what discriminate EB and blend.

### A3. ExoMiner — Valizadegan et al. (2022, NASA) — RECOMMENDED BACKBONE
**What it is:** A **multi-branch deep neural network** that mirrors the human vetting workflow: each branch ingests one **Data Validation (DV) diagnostic** and learns features specific to it, then branches are fused. Highly accurate *and explainable*. Paper: [IOPscience ApJ 926, 120](https://iopscience.iop.org/article/10.3847/1538-4357/ac4399); repo: [github.com/nasa/ExoMiner](https://github.com/nasa/ExoMiner).
**Performance:** at **99% precision it recovers 93.6% of exoplanets** (vs 76.3% for the best prior classifier); validated **301 new Kepler planets**.

**Branches / inputs (the multi-branch design):**
- *Time-series branches* (each: conv blocks → maxpool → FC), fused with branch-specific scalars:
  - Full-orbit **flux** view (global; ~301–2001 bins)
  - Transit **flux** view (local; ~31–201 bins)
  - Transit-view **secondary eclipse** flux (occultation) → EB discriminator
  - **Odd & even** transit views processed in parallel branches with **element-wise subtraction** → EB-at-half-period discriminator
  - Full-orbit & transit-view **centroid motion** → blend/nearby-EB discriminator
- *Scalar branch* (stellar + DV diagnostics): stellar mass/radius/Teff/logg; **optical ghost diagnostic**; **bootstrap false-alarm probability**; **rolling-band** contamination histogram; secondary-event metrics (**geometric albedo, planet effective temperature**); **centroid offset & uncertainty**; transit depth.
- Architecture tuned with **BOHB** (Bayesian Optimization + HyperBand). Output: sigmoid score (extend to **softmax 4-class** for PS7).

> ExoMiner is the recommended DL backbone for PS7 because its branch structure *is* the vetting physics: separate branches for secondary, odd/even, and centroid map directly to EB vs blend vs planet. See also **ExoMiner++ / TESS transfer learning** ([arXiv:2502.09790](https://arxiv.org/html/2502.09790v1)).

### A4. Exonet / "domain knowledge" CNN — Ansdell et al. (2018)
**What it is:** AstroNet + **centroid time-series view** + **scalar stellar parameters** (Teff, logg, stellar radius, metallicity, magnitude). Adding domain knowledge measurably improved precision/recall over flux-only views. Paper: [arXiv:1810.13434](https://arxiv.org/pdf/1810.13434). This is the template for *how to inject stellar context* into a view-based net.

### A5. Other published nets to borrow from
- **RAMjeT / RAmjet** and **one-armed CNN** ([arXiv:2105.06292](https://arxiv.org/pdf/2105.06292)) — direct-from-light-curve detection.
- **Astronet-K2** ([arXiv:1903.10025 / IOPscience ab0e12](https://iopscience.iop.org/article/10.3847/1538-3881/ab0e12)).
- **Transformer / attention** for light curves: **ATAT** (Astronomical Transformer for time series **And** Tabular; trains on light curve + metadata + engineered features jointly) ([A&A aa49475-24](https://www.aanda.org/articles/aa/full_html/2024/09/aa49475-24/aa49475-24.html)); Time-series Transformer for photometric classification ([arXiv:2105.06178](https://arxiv.org/pdf/2105.06178)); TESS-FFI transformer ([arXiv:2502.07542](https://arxiv.org/html/2502.07542v2)); pedagogical review ([arXiv:2310.12069](https://arxiv.org/pdf/2310.12069)).
- **ExoNet (2026) multimodal** — phase-folded LC + stellar params + **multi-head attention fusion**, with *calibration* ([arXiv:2604.15560](https://arxiv.org/html/2604.15560)).
- **LSTM/GRU / Temporal Convolutional Nets (TCN)** — sequence models over raw or detrended light curves; good when phase-folding is unreliable (uncertain period).
- **TabNet** — attention-based deep net for the *tabular* feature stream (alternative to XGBoost when you want an all-DL stack).
- **Autoencoders (LSTM-AE / conv-AE)** — unsupervised; "exoplanet detection = periodic-anomaly detection." Reconstruction error flags `other`/novel signals and rare planets ([CHAOS→Clarity arXiv:2403.10220](https://arxiv.org/pdf/2403.10220); [unsupervised exoplanet ID](https://snaveenmathew.medium.com/unsupervised-learning-in-astronomy-for-exoplanet-candidate-identification-997f3f958dae)).

---

# B) Classical ML classifiers (tabular features)

All consume the **engineered feature vector** in §F. Recommended primary: **XGBoost / LightGBM**; ensemble with **Random Forest**. Precedent: **Armstrong et al. 2018/2021** use **Random Forest + Self-Organizing Map (SOM)** on transit shapes; **McCauliff et al. 2015** use RF on Kepler ([MNRAS 483, 5534](https://academic.oup.com/mnras/article/483/4/5534/5199219); SOM ranking [arXiv:1611.01968](https://arxiv.org/abs/1611.01968); A&A CNN comparison [aa35345-19](https://www.aanda.org/articles/aa/full_html/2020/01/aa35345-19/aa35345-19.html); RAVEN ranking/validation [arXiv:2509.17645](https://arxiv.org/pdf/2509.17645)).

| Classifier | Library / call | Role for PS7 | Notes |
|---|---|---|---|
| **XGBoost** | `xgboost.XGBClassifier(objective="multi:softprob")` | Primary tabular 4-class | Handles missing values, monotonic constraints; `scale_pos_weight`/sample weights for imbalance |
| **LightGBM** | `lightgbm.LGBMClassifier(class_weight="balanced")` | Primary tabular (fast) | Native categorical, leaf-wise; great with many engineered features |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier(class_weight="balanced_subsample")` | Robust baseline / ensemble member | Armstrong-style; bagging reduces variance on small data |
| **Gradient Boosting** | `sklearn.ensemble.HistGradientBoostingClassifier` | Alt boosting baseline | Fast histogram GBM in sklearn |
| **SVM (RBF)** | `sklearn.svm.SVC(probability=True, class_weight="balanced")` | Member on standardized features | Needs scaling; `probability=True` (Platt) for confidence |
| **kNN** | `sklearn.neighbors.KNeighborsClassifier` | Weak member / local density | Useful inside SOM-style neighborhoods |
| **SOM** | `minisom.MiniSom` | Unsupervised transit-shape map | Armstrong et al.: project local view onto SOM; node statistics → feature/ranking |
| **Naive Bayes** | `sklearn.naive_bayes.GaussianNB` | Cheap baseline | Calibration reference |
| **Logistic Regression** | `sklearn.linear_model.LogisticRegression(class_weight="balanced")` | Interpretable baseline + **meta-learner** for stacking | Coefs = feature directions |
| **Isolation Forest** | `sklearn.ensemble.IsolationForest` | Outlier/novelty → `other` | Flags signals unlike any training class |
| **One-Class SVM** | `sklearn.svm.OneClassSVM` | Novelty detection → `other` | Train on "clean planets," score outliers |

```python
# Tabular XGBoost 4-class classifier with grouped CV + calibration
import numpy as np, xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight

# X: (n, n_features) engineered features (see Section F); y in {0:transit,1:EB,2:blend,3:other}
# groups: host-star (TIC) id array, so the same star never spans train/test
sample_w = compute_sample_weight(class_weight="balanced", y=y)

base = xgb.XGBClassifier(
    objective="multi:softprob", num_class=4,
    n_estimators=600, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
    tree_method="hist", eval_metric="mlogloss",
)

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# Calibrate probabilities (isotonic) using the grouped CV folds
clf = CalibratedClassifierCV(base, method="isotonic", cv=cv)
clf.fit(X, y, sample_weight=sample_w)        # base gets sample_weight via routing
proba = clf.predict_proba(X_new)             # calibrated 4-class confidence
```

---

# C) Astrophysical vetting diagnostics (each = a computable test)

These are the physics-based tests astronomers use to split planet/EB/blend/other. Most are implemented in **LEO-vetter** ([github.com/mkunimoto/LEO-vetter](https://github.com/mkunimoto/LEO-vetter), [paper arXiv:2509.10619](https://arxiv.org/html/2509.10619v1)) — a pure-Python, Robovetter-inspired tool that computes metrics and applies pass/fail thresholds (91% completeness, 97% reliability against systematics). Statistical validation via **TRICERATOPS** ([github.com/stevengiacalone/triceratops](https://github.com/stevengiacalone/triceratops), [paper arXiv:2002.00691](https://arxiv.org/pdf/2002.00691)). Reference truth: **TESS SPOC Data Validation (DV) reports**.

### Vetting test → class-discrimination table

| # | Test | What it computes | Flags class | Library / how |
|---|---|---|---|---|
| C1 | **Odd–even depth/timing** | depth(odd) vs depth(even) transits; >3σ diff ⇒ true period is 2×; also timing OE | **EB** (at half period) | LEO-vetter `OE_box/OE_trap/OE_transit`; or fit both subsets |
| C2 | **Secondary eclipse search** | search phase ~0.5 (and all phases) for occultation; significant deep secondary | **EB** (shallow secondary can still be planet ⇒ check albedo) | LEO-vetter model-shift `MS4/5/6`; check **geometric albedo<1** |
| C3 | **Transit shape: V vs U/box** | trapezoid/transit-model fit; ingress=egress vs total; **V-metric = Rp/R\* + b** | **EB** (grazing → V-shaped, V<1.5 fails) | LEO-vetter `V`; fit `batman`/trapezoid, compare χ² |
| C4 | **Centroid offset / difference imaging** | photocenter shift in vs out of transit; PRF-fit offset Δθ; >~15″ ⇒ off-target | **blend** (NEB/BEB) | LEO-vetter pixel-level `Δθ, prfFitQuality`; `tpfplotter`, DV difference image |
| C5 | **Aperture contamination (CROWDSAP/FLFRCSAP)** | fraction of aperture flux from target; crowded field dilution | **blend** | `lc.meta['CROWDSAP']`, `lc.meta['FLFRCSAP']`; FluxCT, `tpfplotter` |
| C6 | **Depth-vs-aperture / per-pixel depth** | recompute depth in apertures of different size / per pixel; depth grows off-target | **blend** | build masks in `lightkurve`, fit depth per mask/pixel |
| C7 | **Implied radius sanity (Rp/R\*)** | from depth & stellar R\*; Rp ≳ 2 RJup (~22 R⊕) ⇒ stellar | **EB** | LEO-vetter candidate-size test `Rp>22 R⊕` |
| C8 | **Stellar-density consistency** | ρ\*,transit = (3π/GP²)(a/R\*)³ vs catalog/seismic ρ\*; mismatch ⇒ blend or wrong period/ecc | **blend / EB / wrong-P** | Seager & Mallén-Ornelas 2003; LEO-vetter `a/R*<1.5` fail |
| C9 | **Transit duration / shape physicality** | duration ratio q, q_circ, a/R\*; SWEET sine-fit; asymmetry ASYM; depth mean/median DMM | **other** (variability/systematics) | LEO-vetter `SWEET, ASYM, DMM, q` |
| C10 | **Uniqueness (model-shift)** | is the primary event unique vs strongest secondary/tertiary/positive bumps | **other** (systematics) / **EB** (if 2ndary unique) | LEO-vetter `MS1/2/3` |
| C11 | **Chases / single-event domination / data-gap** | SES-based local-noise checks; one event dominating SNR; transits near gaps | **other** (scattered light, single outliers) | LEO-vetter `Chases, SNR_max/SNR, data-gap` |
| C12 | **Ghost / halo diagnostic** | core-vs-halo aperture correlation; optical ghost statistic | **blend** | DV optical-ghost diagnostic; ExoMiner ghost branch |
| C13 | **Ephemeris matching** | match (P, t0) against known EB catalogs & scattered-light/momentum-dump cadences | **EB / other** | cross-match TESS EB catalog, sector systematics |
| C14 | **TTV / transit-timing** | O−C of mid-transit times; large/coherent TTV | context (multi-planet vs EB) | fit per-transit t0; LEO-vetter timing OE |
| C15 | **Statistical validation (FPP/NFPP)** | Bayesian prob over astrophysical scenarios (planet vs EB/blend) | **all four** (FPP→EB/blend; NFPP→blend) | **TRICERATOPS** (below) |

> **Other validation/vetting tools to cite:** **vespa** (Morton; FPP via stellar pops), **DAVE** (Discovery And Vetting of Exoplanets — centroid + odd/even + flux), **TESS DV reports** (SPOC), and the **Kepler Robovetter** lineage that LEO-vetter follows.

### C-detail: TRICERATOPS statistical validation
Bayesian tool computing **FPP** (prob signal is *not* a planet on the target) and **NFPP** (prob the signal comes from a *resolved nearby star* — i.e., the **blend** probability). Validation thresholds: **FPP < 0.015 and NFPP < 10⁻³**.

**Scenarios modeled (codes):**
`TP` target planet · `EB` target eclipsing binary · `EBx2P` target EB at 2× period · primary-star: `PTP`/`PEB` · bound-companion (secondary) diluted: `STP`/`SEB` · unbound-tertiary diluted: `DTP`/`DEB` · background star: `BTP`/`BEB` · nearby resolved star: `NTP`/`NEB`(/`NEBx2P`).
- **FPP** = P(all non-TP scenarios) ≈ 1 − P(TP) (after normalization over scenarios).
- **NFPP** = sum of probabilities of the **nearby/background** scenarios (`NEB, NTP, BEB, BTP, NEBx2P`) → this is precisely the **blend** signal. (Note: a web summary mistakenly defined NFPP as 1−FPP; the correct definition is the *nearby-star* contribution per Giacalone & Dressing 2020, [arXiv:2002.00691](https://arxiv.org/pdf/2002.00691).)

```python
# TRICERATOPS: FPP/NFPP for a TOI  (pip install triceratops)
import numpy as np
from triceratops.triceratops import target

tgt = target(ID=261136679, sectors=[3])          # TIC ID + sectors
# ap_pixels = aperture mask pixels used to extract the light curve:
tgt.calc_depths(tdepth=0.0010, all_ap_pixels=ap_pixels)   # observed transit depth
tgt.calc_probs(time=phase, flux_0=flux, flux_err_0=err,   # phase-folded transit
               P_orb=3.21, contrast_curve_file=None)      # add AO contrast if available
print("FPP =", tgt.FPP, " NFPP =", tgt.NFPP)
print(tgt.probs)        # per-scenario probabilities (TP, EB, NEB, BEB, ... )
# Decision: FPP<0.015 and NFPP<1e-3 -> validated planet;
#           high NEB/BEB share     -> label 'blend';  high EB/EBx2P -> 'EB'
```

### C-detail: LEO-vetter flux + pixel metrics
```python
# LEO-vetter computes a metric dict + pass/fail flags from a light curve + ephemeris.
# (see github.com/mkunimoto/LEO-vetter for the FluxVetter / PixelVetter API)
# Conceptual usage:
from leo_vetter.metrics import compute_flux_metrics, compute_pixel_metrics
fm = compute_flux_metrics(time, flux, flux_err, period=P, t0=t0, duration=dur)
#   -> {'SNR':..., 'OE_transit':..., 'MS4':..., 'V':..., 'Rp':..., 'SWEET':..., 'ASYM':..., 'DMM':...}
pm = compute_pixel_metrics(tpf, period=P, t0=t0, duration=dur)
#   -> {'offset_arcsec':..., 'prfFitQuality':...}    # centroid/difference-image
```

### C-detail: contamination keywords + per-aperture depth (lightkurve)
```python
import lightkurve as lk
lc = lk.search_lightcurve("TIC 307210830", mission="TESS", author="SPOC").download()
crowdsap  = lc.meta.get("CROWDSAP")    # target_flux / total_flux in aperture  (->blend if <~0.9)
flfrcsap  = lc.meta.get("FLFRCSAP")    # fraction of target flux captured
# Dilution-correct depth: subtract (1-CROWDSAP)*median, then divide by FLFRCSAP, then re-measure depth.
# Per-aperture depth test: re-extract with bigger/smaller masks; depth growing off-target -> blend (C6).
```

---

# D) Imbalance, label noise, calibration, cross-validation

**Class imbalance** (planets are rare; EB/other dominate):
- **Resampling:** `imblearn.over_sampling.SMOTE`, `BorderlineSMOTE`, `ADASYN`; or `imblearn.pipeline.Pipeline` to keep it inside CV (never oversample before the split).
- **Class weights:** `class_weight="balanced"` (sklearn/LightGBM), `scale_pos_weight` / per-row `sample_weight` (XGBoost), `compute_sample_weight`.
- **Augmentation (DL views):** time-jitter the phase, add Gaussian/realistic red noise, depth scaling, random masking, mirror odd/even. For light curves, **masked-reconstruction / denoising transformer** pretraining helps.
- **Focal loss** for the DL head to down-weight easy majority examples:
```python
import tensorflow as tf
def categorical_focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        w  = tf.pow(1 - y_pred, gamma)
        if alpha is not None: w *= alpha          # per-class weights, e.g. [1,1,2,1]
        return tf.reduce_sum(w * ce, axis=-1)
    return loss
model.compile(optimizer="adam", loss=categorical_focal_loss(gamma=2.0))
```
- **Outlier head for `other`/novel:** Isolation Forest / One-Class SVM / autoencoder reconstruction error as an extra gate.

**Label noise:** training labels (community dispositions) are imperfect. Use **soft labels / label smoothing**, **co-teaching or confident-learning (`cleanlab`)** to find mislabeled TCEs, and weight by **label provenance** (confirmed > validated > community).

**Probability calibration → trustworthy confidence:**
- `sklearn.calibration.CalibratedClassifierCV(base, method="isotonic"|"sigmoid", cv=grouped_cv)`. **Isotonic** if enough data; **sigmoid/Platt** (fits an intercept) for small/imbalanced sets. ([scikit-learn calibration docs](https://scikit-learn.org/stable/modules/calibration.html), [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html))
- For the **neural net**: **temperature scaling** on a held-out set (single scalar T on logits) — the standard modern calibration; ExoNet 2026 emphasizes calibrated multimodal outputs ([arXiv:2604.15560](https://arxiv.org/html/2604.15560)).
- **Diagnostics:** reliability diagram + **Expected Calibration Error (ECE)**; report **Brier score** and per-class precision–recall, not just accuracy.
- **Report confidence** as the calibrated max-class probability *plus* a flag if the FPP/NFPP veto disagrees (so users see physics-vs-ML conflicts).

**Cross-validation (avoid leakage — critical):**
- **Group by host star (TIC) AND by sector**: the same star/sector must never appear in both train and test (transits of one star are correlated). Use `sklearn.model_selection.StratifiedGroupKFold` (group = TIC), and additionally hold out **entire sectors** to test instrument generalization.
- Keep **SMOTE/scaling/calibration inside the CV pipeline**.

---

# E) Recommended ensemble (final 4-class decision + calibrated confidence)

**Design (stacking + physics veto):**

1. **Stream 1 — DL (ExoMiner-style multi-branch CNN):** branches = global flux, local flux, secondary view, odd & even views (with subtraction), centroid view, scalar stellar/DV features → softmax `p_dl ∈ R⁴`. Trained with focal loss + augmentation; calibrated by temperature scaling.
2. **Stream 2 — Tabular (XGBoost + LightGBM + RandomForest, soft-voted):** engineered features (§F) → `p_tab ∈ R⁴`. Calibrated with isotonic CV.
3. **Stream 3 — Vetting metrics + FPP/NFPP:** LEO-vetter flags + TRICERATOPS `FPP, NFPP, per-scenario probs`. Provide both as (a) **features** to Streams 1–2 and (b) a **deterministic override**.
4. **Meta-learner (stacking):** out-of-fold `[p_dl, p_tab, vetting_features]` → **multinomial Logistic Regression** (or a small XGBoost) → `p_meta ∈ R⁴`. Train with grouped CV so meta-features are out-of-fold (no leakage).
5. **Calibration of the meta-output:** isotonic/temperature on a held-out grouped split → final confidence.
6. **Physics override (applied last, with thresholds tuned on validation):**
   - Confirmed **centroid offset** (Δθ large, good PRF) **or** high `NFPP`/`NEB+BEB` share ⇒ force **`blend`**.
   - **Significant deep secondary** (and albedo>1) **or** **odd–even** depth diff **or** implied **Rp>~2 RJup** ⇒ force **`EB`**.
   - Fails **uniqueness/SWEET/asymmetry/data-gap/single-event** ⇒ push toward **`other`**.
   - Else keep the calibrated ML class; **`transit`** only if `FPP<0.015 & NFPP<1e-3` for high-confidence validation.

```python
# Stacking skeleton: out-of-fold meta-features -> calibrated 4-class
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def oof_proba(fit_predict, X, y, groups, n_classes=4, splits=5):
    oof = np.zeros((len(y), n_classes))
    cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=0)
    for tr, te in cv.split(X, y, groups):
        oof[te] = fit_predict(X[tr], y[tr], X[te])
    return oof

p_dl  = oof_proba(fit_dl,  X_views, y, groups)     # Stream 1 (CNN)
p_tab = oof_proba(fit_tab, X_feats, y, groups)     # Stream 2 (XGB/LGBM/RF)
V     = vetting_feature_matrix                      # Stream 3 (FPP, NFPP, flags)

meta_X = np.hstack([p_dl, p_tab, V])
meta   = CalibratedClassifierCV(LogisticRegression(max_iter=1000,
                 class_weight="balanced", multi_class="multinomial"),
                 method="isotonic", cv=StratifiedGroupKFold(5))
meta.fit(meta_X, y)                                 # groups passed via cv.split in practice
p_final = meta.predict_proba(meta_X_new)            # calibrated confidence
label   = apply_physics_override(p_final, vetting_flags_new)   # blend/EB/other vetoes
```

**Why this wins for PS7:** the DL stream captures shape/secondary/centroid morphology; the tabular stream is data-efficient and interpretable; the vetting/FPP layer injects hard astrophysical priors and gives a *defensible* blend/EB call. Stacking + calibration yields a single **trustworthy confidence**, and the override guarantees we never validate a planet that fails a decisive physics test.

---

# F) Engineered feature list (for tabular ML + scalar DL branch)

**Transit / orbit:** orbital period `P`; epoch `t0`; transit **depth** `δ`; **duration** `T14`; ingress/egress `T12`; **duration/period** `T14/P`; **q = T14/P** and `q_circ`; **a/R\*** (from transit); impact parameter `b`; number of observed transits `N_tr`; **transit SNR / MES**; per-transit SNR scatter (`CHI`); reduced χ² of transit-model fit; χ² of trapezoid vs box fit.

**Shape / EB discriminators:** **V-shape metric** `V = Rp/R* + b`; trapezoid ingress/total ratio; **odd–even depth difference** (σ); odd–even timing diff; **secondary-eclipse depth** & significance (`MS4/5/6`); **geometric albedo**; secondary phase; symmetry `ASYM`; depth mean/median `DMM`.

**Blend / contamination:** **centroid offset** (in−out of transit, arcsec) + uncertainty; PRF-fit quality; **CROWDSAP**; **FLFRCSAP**; ghost-diagnostic statistic; per-aperture depth ratio; number/brightness of Gaia neighbors in aperture; **NFPP / per-scenario (NEB,BEB) probability**.

**Implied physical sanity:** **Rp/R\*** and **implied Rp (R⊕/RJup)**; **ρ\*,transit** vs catalog/seismic **ρ\*** ratio; implied stellar density flag.

**Stellar context (TIC/Gaia):** **Teff, logg, [Fe/H], stellar radius R\*, mass M\***, Tmag/Gmag, distance/parallax, contamination ratio.

**Systematics / variability (→ other):** **SWEET** sine-fit significance (½, 1, 2×P); single-event domination `SNR_max/SNR`; chases metric; fraction of transits near data gaps; out-of-transit RMS / red-noise (pink-noise) level; rolling-band contamination.

**Statistical-validation summary:** **FPP**, **NFPP** (TRICERATOPS); LEO-vetter overall PASS/FAIL flag count.

---

## Key sources
- AstroNet — Shallue & Vanderburg 2018: [IOPscience](https://iopscience.iop.org/article/10.3847/1538-3881/aa9e09) · code [google-research/exoplanet-ml](https://github.com/google-research/exoplanet-ml)
- Astronet-Triage/Vetting — Yu et al. 2019: [arXiv:1904.02726](https://arxiv.org/pdf/1904.02726) · [IOPscience](https://iopscience.iop.org/article/10.3847/1538-3881/ab21d6) · [yuliang419/Astronet-Triage](https://github.com/yuliang419/Astronet-Triage)
- ExoMiner — Valizadegan et al. 2022: [ApJ 926,120](https://iopscience.iop.org/article/10.3847/1538-4357/ac4399) · [nasa/ExoMiner](https://github.com/nasa/ExoMiner) · ExoMiner++ [arXiv:2502.09790](https://arxiv.org/html/2502.09790v1) · multiplicity boost [arXiv:2305.02470](https://arxiv.org/html/2305.02470)
- Exonet (domain knowledge) — Ansdell et al. 2018: [arXiv:1810.13434](https://arxiv.org/pdf/1810.13434)
- Astronet-K2: [IOPscience ab0e12](https://iopscience.iop.org/article/10.3847/1538-3881/ab0e12)
- Transformers/multimodal: ATAT [A&A](https://www.aanda.org/articles/aa/full_html/2024/09/aa49475-24/aa49475-24.html) · ExoNet 2026 [arXiv:2604.15560](https://arxiv.org/html/2604.15560) · TS-Transformer [arXiv:2105.06178](https://arxiv.org/pdf/2105.06178) · review [arXiv:2310.12069](https://arxiv.org/pdf/2310.12069)
- Classical ML / SOM / RF — Armstrong et al.: [MNRAS 483,5534](https://academic.oup.com/mnras/article/483/4/5534/5199219) · SOM [arXiv:1611.01968](https://arxiv.org/abs/1611.01968) · A&A CNN [aa35345-19](https://www.aanda.org/articles/aa/full_html/2020/01/aa35345-19/aa35345-19.html) · RAVEN [arXiv:2509.17645](https://arxiv.org/pdf/2509.17645)
- LEO-vetter — Kunimoto et al. 2025: [arXiv:2509.10619](https://arxiv.org/html/2509.10619v1) · [mkunimoto/LEO-vetter](https://github.com/mkunimoto/LEO-vetter)
- TRICERATOPS — Giacalone & Dressing 2020/2021: [arXiv:2002.00691](https://arxiv.org/pdf/2002.00691) · [stevengiacalone/triceratops](https://github.com/stevengiacalone/triceratops)
- Stellar density: Seager & Mallén-Ornelas 2003; "Using stellar densities to evaluate transiting candidates" [ResearchGate](https://www.researchgate.net/publication/231017828)
- Contamination keywords / lightkurve: [TESS crowding (HEASARC)](https://heasarc.gsfc.nasa.gov/docs/tess/UnderstandingCrowdingv2.html) · [lightkurve #1152](https://github.com/lightkurve/lightkurve/issues/1152)
- Calibration: [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html) · [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- Anomaly/unsupervised: [arXiv:2403.10220](https://arxiv.org/pdf/2403.10220)
