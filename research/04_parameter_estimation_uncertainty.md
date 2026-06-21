# PS7 — Transit Parameter Estimation, Bayesian Inference & Uncertainty Quantification

**Scope:** Given a detected transit signal in a noisy TESS light curve, estimate the physical
parameters — **orbital period `P`, transit depth `δ`, transit duration `T14`** (plus impact
parameter `b`, scaled semi-major axis `a/R★`, radius ratio `Rp/R★ = k`, inclination `i`,
ephemeris `T0`) — **with rigorous, well-calibrated uncertainties**, and report **SNR /
significance**. This document covers transit light-curve modeling, the inference/optimization
engines, limb-darkening priors, uncertainty quantification, and the exact formulas to derive
period/depth/duration/SNR with errors.

This is one of several PS7 research domains; it contributes **≥ 8 distinct methods/tools** toward
the combined catalogue.

---

## 0. Recommended stack (TL;DR)

A **two-stage** (and optionally three-stage) pipeline gives the best speed/rigor trade-off:

1. **Detection & coarse params (seconds):** `TLS` (Transit Least Squares) or `astropy` BLS →
   period `P`, epoch `T0`, depth `δ`, duration `T14`, plus the **SDE** significance.
2. **Fast point estimate + analytic errors (sub-second to seconds):** `batman` forward model +
   `lmfit`/`scipy.optimize` Levenberg–Marquardt least-squares → MAP parameters + covariance-matrix
   1σ errors. This **seeds** the sampler.
3. **Full posterior with calibrated uncertainties (minutes):**
   - `emcee` (affine-invariant ensemble MCMC) with a **batman** likelihood for the headline
     posterior + corner plot, **or**
   - `dynesty`/`ultranest` (nested sampling) when you also need **Bayesian evidence `ln Z`** for
     **transit-vs-flat** and **transit-vs-eclipsing-binary** model comparison.
   - Put a **GP (celerite2) red-noise term jointly with the transit** to get correlated-noise-aware
     error bars, or apply the **Carter & Winn (2009)** time-averaging `β`-factor / wavelet
     likelihood.
4. **Limb-darkening priors:** `ldtk` (LDTk) from the TESS-band PHOENIX intensities using the
   star's `Teff, logg, [Fe/H]`, sampled in the **Kipping (2013) `q1, q2`** reparameterization.
5. **Report:** 16/50/84 percentile credible intervals on every parameter, propagate stellar-radius
   uncertainty into the physical `Rp`, and quote transit **SNR**, **TLS SDE**, and **ΔlnZ** (or
   **ΔBIC**) for significance.

Turn-key alternative that wraps stages 2–4: **`juliet`** (nested sampling via `dynesty`, gives
evidence) or **`exoplanet`/PyMC** (gradient-based NUTS HMC, fastest in high dimensions).

---

## 1. Master comparison table

| Tool / method | Forward model | Inference engine | Gives evidence `lnZ`? | Speed | When to use |
|---|---|---|---|---|---|
| **astropy BLS** | box (top-hat) | grid χ² over (P, t0, dur) | no | very fast (s) | initial period/depth/duration screen |
| **TLS** (`transitleastsquares`) | limb-darkened transit template | grid + χ² (SDE) | no (SDE instead) | fast (~10 s / TESS sector) | best initial detection; realistic ingress/egress; SDE significance |
| **Trapezoid / box fit** | trapezoid (4 knots) | LM least-squares | no | very fast (s) | quick depth+duration; V-shape vetting |
| **batman** | Mandel & Agol (2002) analytic | (model only — pair w/ engine) | n/a | ~30 µs/model | fast accurate forward model for any LD law |
| **PyTransit** (`QuadraticModel`/`RoadRunner`) | Mandel&Agol / numerical | (model only) | n/a | Fortran-class (numba) | many light curves / heterogeneous data |
| **occultquad** (Agol IDL/py port) | Mandel&Agol quadratic | (model only) | n/a | very fast | reference/legacy quadratic LD |
| **ellc** | triaxial ellipsoid (EB-capable) | (model only — pair w/ engine) | n/a | fast | eclipsing-binary hypothesis; grazing/secondary |
| **scipy / lmfit (LM)** | any (e.g. batman) | Levenberg–Marquardt least-squares | no (covariance σ only) | sub-s–s | fast MAP + analytic 1σ; seeds MCMC |
| **emcee** | any (batman) | affine-invariant ensemble MCMC | no (use thermo. integ.) | minutes | robust posteriors, corner plots, no gradients |
| **PyMC + exoplanet** | starry / `LimbDarkLightCurve` | NUTS / HMC (gradients) | no (use `arviz`/SMC) | min (scales to high-dim) | many planets/params; GP joint fits; fastest mixing |
| **dynesty** | any (batman) | (dynamic) nested sampling | **yes** (`logz ± logzerr`) | min–hours | posterior **and** evidence; multimodal |
| **ultranest** | any (batman) | reactive nested sampling (MLFriends) | **yes** (`logz ± logzerr`) | min–hours | robust evidence; hard/multimodal posteriors |
| **juliet** | batman (+ GP) | nested sampling (`dynesty`/`MultiNest`) | **yes** | min–hours | turn-key transit+RV; model comparison built-in |
| **allesfitter** | ellc (+ GP) | `emcee` **or** `dynesty` | **yes** (via `dynesty`) | min–hours | turn-key; EBs, spots, flares, TTVs, phase curves |
| **EXOFASTv2** | analytic + MIST/SED | DEMCMC (diff. evol. MCMC) | no (uses GR/Tz) | hours | joint SED+stellar+transit+RV; publication-grade |
| **ldtk (LDTk)** | PHOENIX intensities | — (prior generator) | n/a | s–min | LD priors / likelihood from `Teff, logg, [Fe/H]` |

`lnZ` = marginal likelihood (Bayesian evidence). "Gives evidence: no" tools can still do model
comparison via **BIC/AIC** (Sec. 6.3) or thermodynamic integration.

---

## 2. (A) Analytic transit models

### 2.1 Mandel & Agol (2002) formalism

The observed flux during transit is `F(t) = 1 − λ(p, z(t))`, where the **occultation function**
`λ` is the fraction of the (limb-darkened) stellar disk blocked by an opaque planet of radius ratio
`p = Rp/R★ = k` at projected center-to-center sky separation `z(t)` (in units of `R★`). Mandel &
Agol (2002, ApJ 580, L171) give **closed-form** expressions for `λ` for uniform, **quadratic**, and
**nonlinear** limb darkening in terms of complete elliptic integrals. The geometry:

```
z(t) = (a/R★) * sqrt( sin²(2π(t−t0)/P) + (cos i · cos(2π(t−t0)/P))² )   # circular orbit
```
- Out of transit `z > 1+p` ⇒ `F = 1`. Full transit `z < 1−p`. Ingress/egress `1−p ≤ z ≤ 1+p`.
- Reference: Mandel & Agol (2002), arXiv `astro-ph/0210099`; original routines at
  <https://faculty.washington.edu/agol/transit.html>.

### 2.2 Limb-darkening laws

Stellar specific intensity `I(μ)/I(1)` as a function of `μ = cos θ` (`θ` = angle from disk center):

- **Linear:** `1 − u(1−μ)`.
- **Quadratic (default):** `1 − u1(1−μ) − u2(1−μ)²`.  ← recommended for TESS.
- **Square-root:** `1 − c1(1−μ) − c2(1−√μ)`.
- **Logarithmic:** `1 − c1(1−μ) − c2 μ ln μ`.
- **Power-2:** `1 − c(1 − μ^α)`  (good accuracy/parameter trade-off; Maxted/Hestroffer).
- **Nonlinear (Claret 4-param):** `1 − Σ_{n=1..4} c_n (1 − μ^{n/2})`.

**Kipping (2013) `q1, q2` reparameterization** (quadratic law) — *use this for sampling.*
Kipping (2013, MNRAS 435, 2152; arXiv `1308.0009`) showed the physically valid `(u1, u2)` region
(intensity everywhere positive **and** monotonically decreasing outward) is a triangle, and that the
substitution below maps it to the **unit square**, so you sample `q1, q2 ~ Uniform(0, 1)` with no
rejection and no boundary pathologies:

```
Forward (u → q):
    q1 = (u1 + u2)²
    q2 = u1 / (2 (u1 + u2))

Inverse (q → u), used inside the model:
    u1 = 2 * sqrt(q1) * q2
    u2 = sqrt(q1) * (1 - 2 * q2)
```

This is exactly what `exoplanet`'s `QuadLimbDark` distribution implements internally. Always fit in
`(q1, q2)` (or impose the Kipping triangle) — never fit `(u1, u2)` with independent uniform priors.

### 2.3 Libraries — exact API/usage

**`batman` (BATMAN — BAsic Transit Model cAlculatioN; Kreidberg 2015, PASP 127, 1161;
arXiv `1507.08285`).** Uses the Mandel & Agol quadratic solution with the fast EXOFAST Bulirsch
elliptic-integral evaluation, and numerical integration for arbitrary radially-symmetric LD laws.
~1M quadratic models in 30 s on one core (~30 µs/model). Docs:
<https://lkreidberg.github.io/batman/docs/html/index.html>.

```python
import batman
import numpy as np

params = batman.TransitParams()
params.t0        = 0.0          # time of inferior conjunction (T0), days
params.per       = 3.52         # orbital period P, days
params.rp        = 0.1          # planet radius in stellar radii = Rp/R★ = k  (depth ≈ rp²)
params.a         = 8.8          # semi-major axis in stellar radii = a/R★
params.inc       = 87.0         # inclination, degrees
params.ecc       = 0.0          # eccentricity
params.w         = 90.0         # longitude of periastron, degrees
params.limb_dark = "quadratic"  # "uniform","linear","quadratic","square-root",
                                #  "logarithmic","exponential","power2","nonlinear","custom"
params.u         = [0.4, 0.3]   # LD coefficients (length matches the law)

t = np.linspace(-0.1, 0.1, 1000)        # times (days) at which to evaluate
m = batman.TransitModel(params, t)      # precompute geometry (do ONCE)
flux = m.light_curve(params)            # normalized flux; recompute fast when params change

# In a likelihood, only update params and call m.light_curve(params) — do NOT re-init the model.
params.rp = 0.105
flux2 = m.light_curve(params)

# Supersampling for TESS 30-min (or 10-min) cadence (finite-exposure smearing):
m_ss = batman.TransitModel(params, t, supersample_factor=7, exp_time=30./60./24.)
```

**`PyTransit` (Parviainen 2015, MNRAS 450, 3233; numba-accelerated; Fortran-class speed).**
Unified API across models; `QuadraticModel` is the Mandel&Agol quadratic, `RoadRunner` is a fast
numerical model for any LD law. Docs: <https://pytransit.readthedocs.io/>.

```python
from pytransit import QuadraticModel
import numpy as np

tm = QuadraticModel()
tm.set_data(times)                              # mid-exposure times (and optional lcids/pbids)
# evaluate(k=Rp/R*, ldc=[u1,u2], t0, p=period, a=a/R*, i=inclination_radians, e=0, w=0)
flux = tm.evaluate(k=0.1, ldc=[0.4, 0.3], t0=0.0, p=3.52, a=8.8, i=np.radians(87.))

# RoadRunner for arbitrary LD law (e.g. 'power-2'):
from pytransit import RoadRunnerModel
tm2 = RoadRunnerModel('power-2'); tm2.set_data(times)
flux2 = tm2.evaluate(k=0.1, ldc=[0.5, 0.4], t0=0.0, p=3.52, a=8.8, i=np.radians(87.))
```
First call is slow (numba JIT compile); subsequent calls are very fast. Supports vectorized
evaluation over many parameter sets.

**`exoplanet` / `starry` (Foreman-Mackey et al. 2021, JOSS 6, 3285).** Provides
`xo.LimbDarkLightCurve` (limb-darkened light curve as a differentiable Theano/`pytensor` op) on top
of `xo.orbits.KeplerianOrbit`, so the model plugs into **PyMC** for gradient-based **NUTS/HMC**
(Sec. 3.3). Docs: <https://docs.exoplanet.codes/>.

**`ellc` (Maxted 2016, A&A 591, A111; arXiv `1603.08484`).** Represents stars as triaxial
ellipsoids (Gauss-Legendre integration); models **detached eclipsing binaries** and transiting
planets, including spots, Doppler boosting, light-travel time, eccentric orbits, and **secondary
eclipses** — essential for the **EB false-positive hypothesis**. Repo:
<https://github.com/pmaxted/ellc>.

```python
import ellc
flux = ellc.lc(t_obs, radius_1, radius_2, sbratio, incl,
               t_zero=0.0, period=3.52, q=..., f_c=0.0, f_s=0.0,
               ld_1='quad', ldc_1=[0.4,0.3])
# radius_1 = R1/a, radius_2 = R2/a (fractional radii); sbratio = surface-brightness ratio
# (sbratio≈0 → planet-like dark companion; sbratio>0 → EB with measurable secondary).
```

**`occultquad`.** The original Mandel & Agol quadratic routine (`mandelagol`/`occultquad`),
available as IDL and Python ports (e.g. in `PyAstronomy.modelSuite` and various exoplanet utility
packages). Use as a lightweight, dependency-free reference implementation of the quadratic solution;
for production prefer `batman`/`PyTransit`.

### 2.4 Trapezoid / box model (fast initial duration & depth)

For a quick depth+duration estimate and **V-shape vetting**, fit a **trapezoid** with 4 knots: total
duration `T14`, flat (full) duration `T23`, depth `δ`, and center `T0`. A pure **box** (top-hat) is
even faster but ignores ingress/egress.

```python
import numpy as np
def trapezoid(t, t0, depth, T14, T23):
    """Symmetric trapezoid: T14 = total (1st-4th contact), T23 = flat (2nd-3rd contact)."""
    x = np.abs(t - t0)
    f = np.ones_like(t)
    half14, half23 = T14/2., T23/2.
    f[x <= half23] = 1.0 - depth                                    # flat bottom
    ing = (x > half23) & (x < half14)                              # linear ingress/egress
    f[ing] = 1.0 - depth * (half14 - x[ing]) / (half14 - half23)
    return f
```
- **Grazing/V-shaped** transits have `T23 → 0` (no flat bottom). A best-fit `T23 ≈ 0` is a red flag
  for a grazing planet **or** an eclipsing binary, and signals a strong `b`–`k` degeneracy
  (Sec. 7.4). `astropy.timeseries.BoxLeastSquares` returns `depth`, `duration`, and
  `depth_snr` directly for the box case.

---

## 3. (B) Inference / optimization engines (≥ 4)

### 3.1 Levenberg–Marquardt / least-squares (`scipy.optimize`, `lmfit`) — fast point estimate

LM minimizes `χ²(θ) = Σ_i ((f_i − model_i(θ)) / σ_i)²` and returns the parameter covariance from the
Jacobian (`Cov ≈ (JᵀJ)⁻¹ σ̂²`). 1σ errors are `sqrt(diag(Cov))`. Fast and ideal to **seed** MCMC.

```python
import numpy as np, batman
from scipy.optimize import least_squares

def residuals(theta, t, flux, ferr, m, params):
    params.t0, params.per, params.rp, params.a, params.inc = theta
    return (flux - m.light_curve(params)) / ferr

theta0 = [t0_bls, per_bls, np.sqrt(depth_bls), a_guess, 89.0]
sol = least_squares(residuals, theta0, args=(t, flux, ferr, m, params), method='trf')
# Covariance from the Jacobian at the solution:
J = sol.jac
cov = np.linalg.inv(J.T @ J) * (2*sol.cost/(len(flux)-len(theta0)))   # scaled by reduced chi2
perr = np.sqrt(np.diag(cov))                                          # 1-sigma point errors
```

**`lmfit`** (built on `scipy.optimize`, default `leastsq`/MINPACK LM) is more ergonomic: named
`Parameters` with bounds/expressions, `report_fit()`, automatic covariance `stderr`, and robust
`conf_interval()` (profile-likelihood intervals, slower but better when errors are non-Gaussian).
Docs: <https://lmfit.github.io/lmfit-py/>.

```python
import lmfit
def model_fn(t, t0, per, rp, a, inc):
    params.t0, params.per, params.rp, params.a, params.inc = t0, per, rp, a, inc
    return m.light_curve(params)

model = lmfit.Model(model_fn)
p = model.make_params(t0=t0_bls, per=per_bls, rp=np.sqrt(depth_bls), a=8.8, inc=89.0)
p['rp'].min, p['inc'].max = 0.0, 90.0
res = model.fit(flux, p, t=t, weights=1.0/ferr)
print(lmfit.fit_report(res))                 # best-fit values + 1-sigma stderr + correlations
ci = res.conf_interval()                       # profile-likelihood confidence intervals (robust)
```
**Caveats:** covariance errors assume a locally-Gaussian, white-noise likelihood; they **under**estimate
uncertainty for V-shaped/grazing transits, correlated noise, or strong `b`–`a/R★` degeneracy. Use
them only as a seed and a sanity check; trust MCMC/nested credible intervals for the deliverable.

### 3.2 MCMC — `emcee` (affine-invariant ensemble sampler)

`emcee` (Foreman-Mackey et al. 2013, PASP 125, 306; arXiv `1202.3665`) is a pure-Python
implementation of the Goodman & Weare affine-invariant ensemble sampler. **Affine invariance** means
performance is unchanged under linear reparameterization, so it handles the correlated
`b`–`a/R★`–`k` posteriors well without manual tuning. Docs: <https://emcee.readthedocs.io/>.

**Setup recommendations:**
- **Walkers:** `nwalkers ≥ 2·ndim`, in practice `4–8·ndim` (e.g. 32–64 for a 6–8 param transit).
- **Initialization:** a tight Gaussian ball around the LM/MAP seed (`p0 = sol.x + 1e-4*randn`).
- **Likelihood:** Gaussian, `lnL = −0.5 Σ [((f−model)/σ)² + ln(2π σ²)]`; optionally fit a jitter
  term `σ²→σ²+s²` to absorb underestimated errors.
- **Priors:** uniform/physical on `P, T0, k(=rp)`; `q1,q2 ~ U(0,1)`; `b ~ U(0,1+k)` (or
  `cos i` uniform); enforce `a/R★ > 0`.

```python
import emcee, numpy as np

def log_prob(theta, t, flux, ferr, m, params):
    t0, per, rp, a, b, q1, q2 = theta
    if not (0 < rp < 0.5 and 0 < a < 50 and 0 <= b < 1+rp and per > 0
            and 0 < q1 < 1 and 0 < q2 < 1):
        return -np.inf                                   # log-prior (uniform within bounds)
    inc = np.degrees(np.arccos(b / a))                   # b = (a/R*) cos i  (circular)
    params.t0, params.per, params.rp, params.a, params.inc = t0, per, rp, a, inc
    params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Kipping q->u
    model = m.light_curve(params)
    return -0.5*np.sum(((flux-model)/ferr)**2 + np.log(2*np.pi*ferr**2))

ndim, nwalkers = 7, 50
p0 = best_fit + 1e-4*np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(t, flux, ferr, m, params))
sampler.run_mcmc(p0, 20000, progress=True)
```

**Convergence & burn-in (the rigorous way):** use the **integrated autocorrelation time τ**.
emcee's own guidance: *chains longer than ~50 τ are usually sufficient*; discard a few τ as burn-in
and thin by ~τ/2.

```python
tau = sampler.get_autocorr_time(tol=0)               # per-parameter τ (tol=0 = no error if short)
burnin = int(5 * np.max(tau)); thin = int(0.5*np.min(tau)) if np.min(tau)>=2 else 1
chain = sampler.get_chain(discard=burnin, thin=thin, flat=True)   # independent posterior samples
print("N_eff ≈", chain.shape[0])                     # want >> a few thousand
```
For **multi-chain Gelman–Rubin `R̂`** (between- vs within-chain variance; converged when
`R̂ ≲ 1.01`, with `1.1` a loose upper bound), run several independent `emcee` runs and feed them to
`arviz.rhat`. Practical convergence checklist: (i) `N_steps > 50 τ`; (ii) acceptance fraction
~0.2–0.5; (iii) `R̂ < 1.01`; (iv) trace plots stationary; (v) corner plots smooth/unimodal.

### 3.3 Hamiltonian / NUTS via `PyMC` + `exoplanet`

Gradient-based **No-U-Turn Sampler (NUTS)** mixes far faster than ensemble MCMC in high dimensions
(many planets, GP hyperparameters, per-sector baselines). `exoplanet` supplies the differentiable
light curve; PyMC does the sampling. The canonical model:

```python
import pymc as pm           # (PyMC 5; older tutorials use pymc3 + pymc3_ext as pmx)
import pymc_ext as pmx
import exoplanet as xo
import numpy as np

with pm.Model() as model:
    mean = pm.Normal("mean", mu=0.0, sigma=1.0)
    t0   = pm.Normal("t0", mu=t0_guess, sigma=0.1)
    logP = pm.Normal("logP", mu=np.log(per_guess), sigma=0.1)
    period = pm.Deterministic("period", pm.math.exp(logP))

    u = xo.distributions.QuadLimbDark("u")               # Kipping (2013) q1,q2 internally
    r = pm.Uniform("r", lower=0.01, upper=0.3)           # Rp/R*  (depth ~ r^2)
    b = xo.distributions.ImpactParameter("b", ror=r)     # b ~ U(0, 1+r), respects geometry

    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)   # add r_star/m_star for a/R*
    light_curve = xo.LimbDarkLightCurve(u[0], u[1]).get_light_curve(orbit=orbit, r=r, t=t)
    mu = pm.math.sum(light_curve, axis=-1) + mean

    pm.Normal("obs", mu=mu, sigma=yerr, observed=flux)   # add a jitter term in quadrature if needed
    map_soln = pmx.optimize(start=model.test_point)      # gradient MAP seed
    trace = pmx.sample(tune=2000, draws=2000, start=map_soln,
                       chains=2, cores=2, target_accept=0.9)   # NUTS
```
Diagnose with `arviz.summary(trace)` → per-parameter `mean, sd, hdi_3%, hdi_97%, r_hat, ess_bulk`.
`target_accept=0.9–0.95` reduces divergences. Reparameterize in `log` for scale parameters. NUTS
does **not** give evidence directly — use SMC (`pm.sample_smc`) or thermodynamic integration if you
need `lnZ` from PyMC.

### 3.4 Nested sampling — `dynesty` and `ultranest` (give Bayesian evidence)

Nested sampling integrates the likelihood over the prior to return both the **posterior** and the
**evidence `Z = ∫ L(θ) π(θ) dθ`** (with a principled error). This is the natural tool for
**model comparison** (transit vs. flat; planet vs. EB). You provide a `loglike(θ)` and a
**prior transform** `ptform(u)` mapping the unit cube `u ∈ [0,1]^d` to physical parameters.

**`dynesty`** (Speagle 2020, MNRAS 493, 3132; arXiv `1904.02180`).
Docs: <https://dynesty.readthedocs.io/>.

```python
import numpy as np, dynesty
from dynesty import utils as dyfunc

def ptform(u):                         # unit cube -> physical priors
    x = np.empty_like(u)
    x[0] = u[0]*0.2 + (per0-0.1)       # period  ~ U(per0-0.1, per0+0.1)
    x[1] = u[1]*0.2 + (t0_0-0.1)       # t0
    x[2] = u[2]*0.29 + 0.01            # rp = Rp/R*  ~ U(0.01, 0.30)
    x[3] = u[3]*49 + 1                 # a/R*  ~ U(1, 50)
    x[4] = u[4]*(1+x[2])               # b ~ U(0, 1+rp)
    x[5] = u[5]; x[6] = u[6]           # q1,q2 ~ U(0,1)
    return x

def loglike(theta):
    per, t0, rp, a, b, q1, q2 = theta
    inc = np.degrees(np.arccos(np.clip(b/a, 0, 1)))
    params.t0,params.per,params.rp,params.a,params.inc = t0,per,rp,a,inc
    params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]
    model = m.light_curve(params)
    return -0.5*np.sum(((flux-model)/ferr)**2 + np.log(2*np.pi*ferr**2))

dsampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim=7,
                                        bound='multi', sample='rwalk')
dsampler.run_nested()                  # also: NestedSampler(..., nlive=1000).run_nested(dlogz=0.01)
res = dsampler.results
logZ, logZerr = res.logz[-1], res.logzerr[-1]      # Bayesian evidence (+/- error)
samples_equal = res.samples_equal()                # equal-weight posterior (Sec. 5)
# (equivalently: dyfunc.resample_equal(res.samples, np.exp(res.logwt - res.logz[-1])))
```

**`ultranest`** (Buchner 2021, JOSS 6, 3001; reactive nested sampling with the parameter-free
MLFriends region sampler — very robust on hard/multimodal posteriors).
Docs: <https://johannesbuchner.github.io/UltraNest/>.

```python
import ultranest
sampler = ultranest.ReactiveNestedSampler(
    ['per','t0','rp','a','b','q1','q2'], loglike, ptform)
result = sampler.run(min_num_live_points=400, dlogz=0.5)
print("logZ = %.2f +- %.2f" % (result['logz'], result['logzerr']))
post = result['samples']                # equal-weight posterior samples
```
**Cost:** nested sampling is typically slower than a well-tuned `emcee`/NUTS for pure parameter
estimation, but it is the cleanest route to **`lnZ`**. Use `nlive ≈ 400–1000`; increase for tight
evidence error.

### 3.5 Turn-key transit fitters (compare)

- **`TLS`** — *initial detection*, not a posterior fitter. Returns `period, T0, depth, duration,
  SDE, SNR, odd/even depth, transit times`. Realistic limb-darkened template ⇒ higher SDE than BLS
  (Hippke & Heller 2019; e.g. SDE 66.7 vs BLS 16.9 in their example). Feed its outputs as seeds.
- **`exoplanet`/PyMC** — fastest, most scalable (gradients), great for GPs and many planets; no
  direct evidence.
- **`juliet`** (Espinoza et al. 2019, MNRAS 490, 2262; arXiv `1812.08549`) — wraps `batman` (+GP via
  `celerite`/`george`) with **nested sampling** (`dynesty`/`MultiNest`) ⇒ **posteriors + evidence**,
  so model comparison is built in. Ideal turn-key Bayesian fitter for the PS7 deliverable.
- **`allesfitter`** (Günther & Daylan 2021, ApJS 254, 13; arXiv `2003.14371`) — wraps **`ellc`**
  (+GP), runs **either `emcee` or `dynesty`**, supports EBs, spots, flares, TTVs, phase curves, and
  provides both **parameter estimation and Bayesian model selection**.
- **`EXOFASTv2`** (Eastman et al. 2019, arXiv `1907.09480`) — IDL, differential-evolution MCMC,
  jointly fits SED + stellar models (MIST) + transit + RV for self-consistent stellar+planet params
  with full covariances; publication-grade but heavier.

---

## 4. (C) Limb-darkening priors

Limb darkening is degenerate with `b`, `a/R★`, and depth, especially for the shallow,
single-band TESS transits in PS7. Do **not** fix LD blindly; instead set a **physically-motivated
prior** from the host star's `Teff, logg, [Fe/H]` and sample in Kipping `(q1, q2)`.

### 4.1 `ldtk` (Limb Darkening Toolkit; Parviainen & Aigrain 2015, MNRAS 453, 3821)

LDTk computes custom LD profiles + model-specific coefficients (with uncertainties **propagated from
the stellar-parameter uncertainties**) for any passband from the Husser et al. (2013) PHOENIX
specific-intensity library — perfect for the TESS bandpass. Repo:
<https://github.com/hpparvi/ldtk>.

```python
from ldtk import LDPSetCreator, BoxcarFilter
# TESS band ~600-1000 nm; or use the dedicated TESS filter via ldtk.filters
filters = [BoxcarFilter('TESS', 600, 1000)]
sc = LDPSetCreator(filters,
                   teff=[5777, 80],     # (value, 1-sigma) effective temperature
                   logg=[4.44, 0.10],   # surface gravity
                   z=[0.00, 0.05])      # metallicity [Fe/H]
ps = sc.create_profiles(nsamples=2000)
qc, qe = ps.coeffs_qd(do_mc=True)        # quadratic LD coeffs [u1,u2] + 1-sigma errors (qe)
# qc, qe = ps.coeffs_nl()                # nonlinear 4-coeff version
```
Two ways to use the result:
1. **Gaussian priors** on `(u1, u2)` (or convert to `(q1, q2)`) centered at `qc` with width `qe`.
2. **LD likelihood term** added to the posterior — LDTk supplies `ps.lnlike_qd(coeffs)`:
   `log_posterior = log_prior + log_like_transit + ps.lnlike_qd(theta_ld)`. This is the most rigorous
   route (constrains LD by the actual PHOENIX profile, not just a Gaussian on coefficients).

### 4.2 Claret tables (alternative / cross-check)

Claret (2017, A&A 600, A30) and Claret & Bloemen (2011) tabulate quadratic and nonlinear LD
coefficients for the **TESS** passband on grids of `(Teff, logg, [Fe/H], v_t)` from ATLAS/PHOENIX.
Interpolate to the host-star values to set the **prior mean**; widen the prior to cover the
table/model spread (typically `σ_u ≈ 0.05–0.1`). Use as a prior centroid or independent check on
LDTk. For TESS-specific empirical values see also Patel & Espinoza (2022, AJ 163, 228).

**Rule of thumb:** for high-SNR transits, fit `(q1, q2)` with a **weak** LDTk/Claret prior (let the
data speak); for shallow/low-SNR transits, use a **tight** prior so LD does not soak up depth/`b`
information and inflate the radius uncertainty.

---

## 5. (D) Uncertainty quantification & significance

### 5.1 Posterior credible intervals & corner plots

Report the **median and 16th/84th percentiles** (the central 68.3% credible interval) of the
marginal posterior of every parameter — robust to skew/non-Gaussianity, unlike a covariance σ.

```python
import numpy as np, corner
# chain: (Nsamples, ndim) equal-weight posterior from emcee / dynesty / ultranest
q16, q50, q84 = np.percentile(chain, [16, 50, 84], axis=0)
val   = q50
sigma_minus = q50 - q16
sigma_plus  = q84 - q50          # report as  val (+sigma_plus / -sigma_minus)
labels = ['P','T0','Rp/R*','a/R*','b','q1','q2']
fig = corner.corner(chain, labels=labels, quantiles=[0.16,0.5,0.84], show_titles=True)
```
The **corner plot** visualizes all 1-D marginals and 2-D joint posteriors — the place to *see* the
`b`–`a/R★`–`k` degeneracy and confirm unimodality.

### 5.2 Propagating stellar-parameter uncertainty into physical `Rp`

The light curve constrains the **ratio** `k = Rp/R★` (and `a/R★`), not absolute sizes. To get the
physical planet radius and its (often dominant) uncertainty, **sample-multiply** the posterior of `k`
by an independent posterior/Gaussian for `R★` (from the TIC / Gaia / SED / asteroseismology):

```python
import numpy as np
k_samples  = chain[:, labels.index('Rp/R*')]                     # Rp/R* posterior
Rstar_samp = np.random.normal(Rstar, Rstar_err, size=len(k_samples))  # R* in R_sun
Rp_samples = k_samples * Rstar_samp * 109.076                    # R_sun -> R_earth (=695700/6371)
Rp_med, Rp_lo, Rp_hi = np.percentile(Rp_samples, [50,16,84])
# sigma_Rp/Rp ≈ sqrt( (sigma_k/k)^2 + (sigma_R*/R*)^2 )  in the Gaussian limit
```
This Monte-Carlo propagation automatically captures correlations and is the standard way to deliver
`Rp ± σ` in Earth/Jupiter radii. The same approach turns the `a/R★` posterior + `M★` into the
physical semi-major axis and, with `P`, the **stellar density** `ρ★` (Sec. 7.5).

### 5.3 SNR / significance metrics

- **Per-point / single-transit SNR (white-noise, box approximation):**
  `SNR_1 = (δ / σ) · √(N_in)`, where `δ` = depth, `σ` = per-point photometric scatter, `N_in` =
  number of in-transit points. For a single transit, `SNR ∝ δ·√(T14)`.
- **Phase-folded multi-transit SNR:**
  `SNR = (δ / σ) · √(N_tr · N_pts)` where `N_tr` = number of transits and `N_pts` = points per
  transit; equivalently `SNR = δ·√(N_total,in)/σ`.
- **CDPP-based (TESS/Kepler standard, red-noise aware):**
  `SNR = δ / CDPP(T14)`, where `CDPP(T14)` is the Combined Differential Photometric Precision on the
  transit-duration timescale. Often expressed as a **MES** (Multiple Event Statistic) by the
  pipeline. Use this, not the white-noise formula, on real TESS data.
- **TLS SDE (Signal Detection Efficiency):** significance of the χ² minimum vs. the surrounding
  periodogram, `SDE = (SR_peak − ⟨SR⟩)/std(SR)`. Hippke & Heller (2019) find **SDE ≥ 9 ⇒ FPR < 1e-4**
  for TLS. Report this from the detection stage.
- **Detection threshold context:** TESS SPOC uses a transit-search MES threshold of ~7.1σ
  (inherited from Kepler) to control false alarms. Quote both the per-transit SNR and the
  multi-transit MES/SDE.

### 5.4 Model comparison: BIC / AIC and Bayesian evidence

**Information criteria** (cheap; need only the maximum likelihood `L̂`, `k` free params, `n` data):
```
AIC  = 2k − 2 ln L̂                     (AICc adds 2k(k+1)/(n−k−1) for small n)
BIC  = k ln n − 2 ln L̂
χ²_red = χ²_min / (n − k)               (target ≈ 1; >1 ⇒ underestimated errors / poor model)
```
Lower is better. `ΔBIC = BIC_simple − BIC_complex` ≈ `2 ln(Bayes factor)`. Rule of thumb:
`ΔBIC` 0–2 weak, 2–6 positive, 6–10 strong, >10 very strong evidence for the lower-BIC model. Use
`ΔBIC`/`ΔAIC` for transit-vs-flat and #-of-planets questions when a full evidence is too expensive.

**Bayesian evidence (gold standard; from `dynesty`/`ultranest`/`juliet`/`allesfitter`):** compare
hypotheses by the difference in log-evidence `Δ ln Z = ln Z_A − ln Z_B` (the log Bayes factor).
**Jeffreys scale:** `ΔlnZ < 1` inconclusive, `1–2.5` significant, `2.5–5` strong, `>5` decisive.
Two PS7-critical comparisons:
- **Transit vs. flat line** (`δ = 0`): `Δ ln Z = ln Z_transit − ln Z_flat` quantifies the detection
  significance in a fully Bayesian way (complements SDE/MES).
- **Transit vs. eclipsing binary:** fit a planet model (`batman`) and an EB model (`ellc`, with
  `radius_2`, `sbratio`, possible secondary eclipse, V-shape, odd/even depth differences) and compare
  `ln Z`. A V-shaped, deep, or odd/even-asymmetric event with comparable or higher EB evidence is a
  likely false positive. (Note evidence can be inconclusive when dilution is degenerate with depth.)

### 5.5 Red-noise-aware uncertainties (essential for honest error bars)

TESS light curves contain **correlated (red) noise** (stellar granulation, rotation, scattered light,
systematics). Ignoring it makes white-noise error bars **too small** by a factor that can be ≳ 2.
Three standard remedies:

1. **GP + transit jointly (recommended):** model the correlated noise with a Gaussian process whose
   hyperparameters are sampled simultaneously with the transit parameters, so the transit
   uncertainties automatically inflate to account for the noise. Use **`celerite2`** (Foreman-Mackey
   et al. 2017, AJ 154, 220; scalable O(N)) with a **`SHOTerm`** (stochastically-driven damped
   simple-harmonic-oscillator) kernel for granulation/rotation, optionally a **`RotationTerm`**.

   ```python
   import celerite2
   from celerite2 import terms
   # rho = undamped period, sigma = amplitude, tau/Q = damping; jointly with transit params:
   kernel = terms.SHOTerm(sigma=sig_gp, rho=rho_gp, Q=1/np.sqrt(2)) \
          + terms.JitterTerm(sigma=jit)
   gp = celerite2.GaussianProcess(kernel, mean=0.0)
   gp.compute(t, yerr=ferr)
   resid = flux - transit_model(theta)           # transit removed
   loglike = gp.log_likelihood(resid)            # marginalizes over the red noise
   ```
   `exoplanet`/PyMC integrates `celerite2` directly for NUTS joint fits; `juliet`/`allesfitter`
   expose GP kernels as options.

2. **Time-averaging "β" factor (Pont, Zucker & Queloz 2006; Winn 2008):** bin the residuals into
   bins of `N` points; for pure white noise the binned RMS scales as `σ_N = σ_1/√N`. Red noise makes
   the observed `σ_N` larger; define `β = σ_N,obs / (σ_1/√N)` (averaged over relevant bin sizes near
   the transit-duration timescale). **Multiply the white-noise parameter error bars by `β`** (often
   `β ≈ 1.0–2.5`). Cheap, post-hoc, model-independent.

3. **Wavelet likelihood (Carter & Winn 2009, ApJ 704, 51):** parameterizes time-correlated noise as
   `1/f^γ` with just two numbers — the white amplitude `σ_w` and red amplitude `σ_r` — and evaluates
   the likelihood in an orthonormal **wavelet basis** where the covariance is nearly diagonal
   (`O(N)`). Fit `σ_w, σ_r` jointly with the transit so credible intervals are correct under
   correlated noise. This is the classic rigorous alternative to a GP when the noise is `1/f`-like.

### 5.6 Reporting calibrated confidence for the PS7 deliverable

For each detected signal, report:
- **Point + interval per parameter:** `P, T0, δ, T14, T23, b, a/R★, Rp/R★, i` as
  `median (+σ⁺ / −σ⁻)` from 16/50/84 percentiles of the **red-noise-aware** posterior.
- **Physical `Rp ± σ`** (Earth & Jupiter radii) via stellar-radius MC propagation (Sec. 5.2).
- **Significance:** per-transit **SNR**, phase-folded **SNR/MES**, **TLS SDE**, reduced χ².
- **Model comparison:** `Δ ln Z` (transit-vs-flat and transit-vs-EB) and/or `ΔBIC`, with the
  Jeffreys/BIC verdict.
- **Calibration check:** validate the pipeline on **injection–recovery** tests — inject known
  synthetic transits into real TESS light curves and confirm that the recovered parameters fall
  within the quoted credible intervals at the stated frequency (e.g. ~68% of truths inside the 68%
  interval ⇒ calibrated). This is the most convincing evidence that the reported confidences are
  trustworthy.

---

## 6. (E) Deriving period / depth / duration robustly + exact formulas

### 6.1 Pipeline: BLS/TLS initial → model refine

1. **Detect** with `TLS` (preferred) or `astropy` BLS → coarse `P, T0, δ, T14`, plus **SDE/SNR**.
2. **Phase-fold** on `P, T0`: `phase = ((t − T0 + 0.5P) mod P) − 0.5P`; **bin** for visualization.
3. **Refine** with `batman` + LM (Sec. 3.1) to get MAP + analytic σ.
4. **Sample** the full posterior (`emcee`/`dynesty`/NUTS) for calibrated credible intervals; refine
   `P, T0` further by fitting all epochs simultaneously (long baseline tightens `P` dramatically).

```python
import numpy as np
def fold(t, P, T0):
    phase = (t - T0 + 0.5*P) % P - 0.5*P
    return phase
def bin_phase(phase, flux, nbins=200):
    order = np.argsort(phase); ph, fl = phase[order], flux[order]
    edges = np.linspace(ph.min(), ph.max(), nbins+1)
    idx = np.digitize(ph, edges)
    bx = np.array([ph[idx==i].mean() for i in range(1, nbins+1)])
    by = np.array([fl[idx==i].mean() for i in range(1, nbins+1)])
    be = np.array([fl[idx==i].std()/max(1,np.sqrt((idx==i).sum())) for i in range(1, nbins+1)])
    return bx, by, be
```

### 6.2 Period (with uncertainty)

- BLS/TLS gives the periodogram-peak `P`; its width sets a first error. The **rigorous** `P, σ_P`
  come from the joint fit of **all** transits — `σ_P ≈ σ_{T0} / N_epochs` shrinks with the time
  baseline (linear ephemeris `T_n = T0 + n·P`; fit `T0, P` by weighted least-squares to the measured
  per-transit centers, or sample directly).

### 6.3 Depth → Rp/R★

Transit depth and radius ratio (Winn 2010, Eq. 22):
```
δ ≈ k² = (Rp/R★)²          ⇒        k = Rp/R★ = √δ
```
- This is the **flat-bottom** (non-grazing) depth; limb darkening makes the *observed* minimum
  slightly deeper than `k²` at mid-transit, so fit the full limb-darkened model rather than reading
  the minimum. Propagate to physical `Rp` via Sec. 5.2.
- `σ_k ≈ 0.5 · σ_δ / √δ` in the Gaussian limit; from the posterior, take percentiles of `k`.

### 6.4 Transit duration (exact + approximate)

**Winn (2010), "Transits and Occultations" (arXiv `1001.2010`).**

Total (1st–4th contact) and full/flat (2nd–3rd contact) durations (Eqs. 14–15):
```
T14 = (P/π) · arcsin[ (R★/a) · √((1+k)² − b²) / sin i ]
T23 = (P/π) · arcsin[ (R★/a) · √((1−k)² − b²) / sin i ]
```
Small-planet / large-`a/R★` approximation (drop `sin i ≈ 1`):
```
T14 ≈ (P/π) · arcsin[ (R★/a) · √((1+k)² − b²) ]
    ≈ (P / π) · (R★/a) · √((1+k)² − b²)          (further, for a/R★ ≫ 1)
```
Impact parameter (Winn 2010, Eqs. 7–8), with the eccentricity correction:
```
b = (a/R★) · cos i · [ (1 − e²) / (1 + e·sin ω) ]        (transit)
b_occ = (a/R★) · cos i · [ (1 − e²) / (1 − e·sin ω) ]      (secondary eclipse)
```
Eccentricity also rescales the duration by the velocity factor (Eq. 16):
```
T14(e) = T14(e=0) · √(1 − e²) / (1 ± e·sin ω)
```
Ingress/egress duration (small-planet limit):
```
τ ≈ T14 − T23 ≈ (P/π) · (R★/a) · [ 2k / √(1 − b²) ]   (grazing-free)
```
**Uncertainty on `T14`:** the posterior already encodes it — compute `T14` from each posterior
sample of `(P, a/R★, b, k, i)` using the equation above and take 16/50/84 percentiles. This
correctly propagates the `b`–`a/R★` correlation into `σ_{T14}`.

```python
import numpy as np
def duration_T14(P, aRs, b, k, inc_deg):
    inc = np.radians(inc_deg)
    arg = (1.0/aRs) * np.sqrt(np.maximum((1+k)**2 - b**2, 0.0)) / np.sin(inc)
    return (P/np.pi) * np.arcsin(np.clip(arg, -1, 1))
# Propagate over the posterior chain -> percentiles give T14 with asymmetric error bars
T14_samps = duration_T14(chain[:,0], chain[:,3], chain[:,4], chain[:,2],
                         np.degrees(np.arccos(chain[:,4]/chain[:,3])))
T14_med, T14_lo, T14_hi = np.percentile(T14_samps, [50,16,84])
```

### 6.5 SNR (exact, with uncertainty)

```
Single transit (white, box):   SNR = (δ / σ) · √N_in           with σ_SNR from σ_δ, σ
Phase-folded:                  SNR = (δ / σ) · √(N_tr · N_pts)
Red-noise aware (TESS):        SNR = δ / CDPP(T14)   (== MES from the pipeline)
TLS detection significance:    SDE  (threshold ≥ 9 ⇒ FPR < 1e-4)
```
Report `SNR` with its uncertainty by propagating the posterior `σ_δ` (and the measured `σ`/CDPP).

### 6.6 Handling grazing / V-shaped degeneracy (`b`–`k`)

- A **V-shaped** transit (`T23 → 0`, i.e. `b ≳ 1 − k`) makes `b`, `k`, and `a/R★` strongly
  degenerate: many `(b, k)` pairs fit equally well, so `Rp/R★` becomes **highly uncertain and
  upper-limit-like**, and the event may instead be a **grazing eclipsing binary**.
- **Mitigations:** (i) impose a **stellar-density prior** on `a/R★` (Sec. 7.5) to break the
  degeneracy via Kepler's third law; (ii) put a physically-motivated prior on LD; (iii) report the
  **full 2-D `b`–`k` posterior** (corner plot) and quote `Rp/R★` as a credible interval (often
  one-sided); (iv) run the **EB hypothesis** (`ellc`) and compare evidence; (v) check **odd/even**
  transit depths and any **secondary eclipse** (EB signatures). Never report a single grazing radius
  without these caveats.

---

## 7. Key auxiliary relations

### 7.1 Geometry / inclination
```
b = (a/R★) cos i      ⇒      i = arccos( b / (a/R★) )      (circular orbit)
```

### 7.2 Sky-projected separation (circular)
```
z(t) = (a/R★) √[ sin²(2π(t−T0)/P) + (cos i · cos(2π(t−T0)/P))² ]
```

### 7.3 Transit probability (context for priors)
```
p_tr = (R★ + Rp)/a = (1 + k)/(a/R★)        (circular; ×(1+e sin ω)/(1−e²) for eccentric)
```

### 7.4 Depth–shape sanity
- Flat-bottom, U-shaped, depth ≈ `k²` ≲ a few %, with consistent odd/even depths and **no**
  secondary ⇒ planet-like. Deep (≳ 10%), V-shaped, odd/even mismatch, or visible secondary ⇒ EB
  suspect.

### 7.5 Stellar density from the transit (breaks degeneracies)
Seager & Mallén-Ornelas (2003, ApJ 585, 1038) + Kepler's third law: for a circular orbit the light
curve alone constrains the **mean stellar density**:
```
ρ★ ≈ (3π / (G P²)) · (a/R★)³
```
- Therefore `a/R★ = [ G P² ρ★ / (3π) ]^{1/3}`. Using an **independent `ρ★`** (Gaia/SED/
  asteroseismology) as a **prior on `a/R★`** sharply reduces the `b`–`a/R★`–`k` correlation and
  tightens `Rp/R★`, `b`, and `i`. (For eccentric orbits the inferred density is biased by the
  "photoeccentric" factor `(1 + e sin ω)³/(1 − e²)^{3/2}` — Dawson & Johnson 2012 — which can be
  turned around to constrain `e`.)

---

## 8. Recommended two-stage fit — runnable reference code

End-to-end **fast LM seed → emcee posterior** with `batman`, including supersampling for TESS
cadence, Kipping LD, red-noise-aware reporting hooks, and derived `T14`/SNR with uncertainties.
(Swap the `emcee` block for the `dynesty` block in Sec. 3.4 when you also need `ln Z`.)

```python
import numpy as np
import batman
from scipy.optimize import least_squares
import emcee
import corner

# ---------- 0. Inputs from detection (TLS/BLS) ----------
# t, flux, ferr : 1-D arrays (days, normalized flux, per-point errors)
# Seeds from TLS/BLS:
P0, T0_0, depth0, dur0 = tls_period, tls_T0, tls_depth, tls_duration
k0   = np.sqrt(depth0)                 # Rp/R*  from depth
aRs0 = 8.0                             # rough a/R* (refine; or from rho_star prior)
b0   = 0.3
exp_time = 2.0/60.0/24.0               # 2-min TESS cadence in days (use 30/60/24 for FFI 30-min)

# ---------- 1. batman forward model (Kipping LD) ----------
pars = batman.TransitParams()
pars.t0, pars.per, pars.rp, pars.a = T0_0, P0, k0, aRs0
pars.inc = np.degrees(np.arccos(b0/aRs0)); pars.ecc = 0.0; pars.w = 90.0
pars.limb_dark = "quadratic"; pars.u = [0.4, 0.3]
mod = batman.TransitModel(pars, t, supersample_factor=7, exp_time=exp_time)

def set_theta(theta):
    P, T0, k, aRs, b, q1, q2 = theta
    pars.per, pars.t0, pars.rp, pars.a = P, T0, k, aRs
    pars.inc = np.degrees(np.arccos(np.clip(b/aRs, 0, 1)))
    pars.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]      # Kipping q -> u
    return mod.light_curve(pars)

# ---------- 2. STAGE 1: Levenberg-Marquardt point estimate + covariance ----------
def resid(theta):
    return (flux - set_theta(theta)) / ferr

theta0 = np.array([P0, T0_0, k0, aRs0, b0, 0.3, 0.3])
lb = [P0-0.05, T0_0-0.05, 0.001,  1.0, 0.0, 0.0, 0.0]
ub = [P0+0.05, T0_0+0.05, 0.5,   50.0, 1.0, 1.0, 1.0]
sol = least_squares(resid, theta0, bounds=(lb, ub), method='trf')
J = sol.jac
dof = max(1, len(flux) - len(theta0))
cov = np.linalg.inv(J.T @ J) * (2*sol.cost/dof)
perr = np.sqrt(np.diag(cov))
print("LM best-fit:", sol.x)
print("LM 1-sigma :", perr)

# ---------- 3. STAGE 2: emcee posterior ----------
def log_prob(theta):
    P, T0, k, aRs, b, q1, q2 = theta
    if not (P0-0.05 < P < P0+0.05 and T0_0-0.05 < T0 < T0_0+0.05
            and 0.001 < k < 0.5 and 1.0 < aRs < 50.0
            and 0.0 <= b < 1.0+k and 0.0 < q1 < 1.0 and 0.0 < q2 < 1.0):
        return -np.inf
    model = set_theta(theta)
    return -0.5*np.sum(((flux-model)/ferr)**2 + np.log(2*np.pi*ferr**2))

ndim = 7
nwalkers = 50
p0 = sol.x + 1e-4*np.random.randn(nwalkers, ndim)
p0 = np.clip(p0, np.array(lb)+1e-6, np.array(ub)-1e-6)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(p0, 20000, progress=True)

# Convergence + burn-in via autocorrelation time
tau = sampler.get_autocorr_time(tol=0)
burn = int(5*np.nanmax(tau)); thin = max(1, int(0.5*np.nanmin(tau)))
chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
labels = ['P','T0','Rp/R*','a/R*','b','q1','q2']

# ---------- 4. Credible intervals (16/50/84) ----------
q16, q50, q84 = np.percentile(chain, [16, 50, 84], axis=0)
for i, lab in enumerate(labels):
    print(f"{lab:7s} = {q50[i]:.6f}  +{q84[i]-q50[i]:.6f} / -{q50[i]-q16[i]:.6f}")

# ---------- 5. Derived: depth, duration, SNR with uncertainties ----------
P_s, k_s, aRs_s, b_s = chain[:,0], chain[:,2], chain[:,3], chain[:,4]
depth_s = k_s**2
inc_s = np.degrees(np.arccos(np.clip(b_s/aRs_s, 0, 1)))
arg = (1.0/aRs_s)*np.sqrt(np.maximum((1+k_s)**2 - b_s**2, 0.0))/np.sin(np.radians(inc_s))
T14_s = (P_s/np.pi)*np.arcsin(np.clip(arg, -1, 1))                 # days
sigma_white = np.median(ferr)
N_in = np.sum(np.abs(fold(t, q50[0], q50[1])) < 0.5*np.median(T14_s)) if 'fold' in dir() else len(t)
SNR_s = depth_s/sigma_white*np.sqrt(max(N_in,1))

for name, arr, unit in [('depth(δ)', depth_s*1e6, 'ppm'),
                        ('T14', T14_s*24.0, 'hr'),
                        ('SNR', SNR_s, '')]:
    m_, lo, hi = np.percentile(arr, [50,16,84])
    print(f"{name:9s} = {m_:.3f}  +{hi-m_:.3f} / -{m_-lo:.3f} {unit}")

# ---------- 6. Physical Rp (propagate stellar radius) ----------
Rstar, Rstar_err = 0.95, 0.04           # R_sun (from TIC/Gaia/SED)
Rp_earth = k_s * np.random.normal(Rstar, Rstar_err, size=len(k_s)) * 109.076
print("Rp [Re] = %.3f +%.3f/-%.3f" % tuple(np.percentile(Rp_earth,[50,84,16])
                                           - np.r_[0, np.percentile(Rp_earth,50), np.percentile(Rp_earth,50)]))

# ---------- 7. Corner plot ----------
fig = corner.corner(chain, labels=labels, quantiles=[0.16,0.5,0.84], show_titles=True)
fig.savefig("transit_corner.png", dpi=150)
```

---

## 9. Practical recommendations for PS7

- **Always two-stage** (LM seed → sampler). The LM seed slashes burn-in and avoids multimodal traps;
  the sampler delivers the calibrated intervals that PS7 requires.
- **Default model:** `batman` quadratic LD + Kipping `(q1, q2)` + LDTk prior in the TESS band.
- **Default sampler:** `emcee` for the headline posterior; add **`dynesty`/`ultranest`** when you need
  `ln Z` for transit-vs-flat / transit-vs-EB; use **PyMC/`exoplanet`** (NUTS) when dimensionality
  grows (multi-planet, joint GP). For a turn-key Bayesian fit with evidence, use **`juliet`**.
- **Never trust white-noise error bars on real TESS data** — use a **celerite2 GP joint fit** (best),
  or apply the **β-factor**/**Carter & Winn wavelet** correction.
- **Break the `b`–`a/R★`–`k` degeneracy** with a **stellar-density prior** (`ρ★ → a/R★`).
- **Report:** 16/50/84 intervals on `P, T0, δ, T14, T23, b, a/R★, Rp/R★, i`; physical `Rp ± σ`;
  **SNR / MES / SDE**; **Δ ln Z** (or **ΔBIC**) for significance and EB rejection; and validate the
  whole chain with **injection–recovery** to prove the confidences are calibrated.

---

## 10. References (real URLs)

**Models / formalism**
- Mandel & Agol (2002), ApJ 580, L171 — analytic transit light curves:
  <https://arxiv.org/abs/astro-ph/0210099> ; routines: <https://faculty.washington.edu/agol/transit.html>
- Winn (2010), "Transits and Occultations" — duration/depth/impact equations:
  <https://arxiv.org/abs/1001.2010>
- Seager & Mallén-Ornelas (2003), ApJ 585, 1038 — unique solution & `ρ★` from a light curve:
  <https://iopscience.iop.org/article/10.1086/346105>
- Kipping (2013), MNRAS 435, 2152 — `q1, q2` LD reparameterization:
  <https://arxiv.org/abs/1308.0009>
- Claret (2017), A&A 600, A30 — TESS limb-darkening tables:
  <https://ui.adsabs.harvard.edu/abs/2017A%26A...600A..30C/abstract>

**Libraries**
- `batman` (Kreidberg 2015): <https://arxiv.org/abs/1507.08285> ; docs:
  <https://lkreidberg.github.io/batman/docs/html/index.html>
- `PyTransit` (Parviainen 2015): <https://pytransit.readthedocs.io/> ;
  <https://github.com/hpparvi/PyTransit>
- `exoplanet`/`starry` (Foreman-Mackey et al. 2021): <https://docs.exoplanet.codes/>
- `ellc` (Maxted 2016): <https://arxiv.org/abs/1603.08484> ; <https://github.com/pmaxted/ellc>
- `ldtk` / LDTk (Parviainen & Aigrain 2015): <https://arxiv.org/abs/1508.02634> ;
  <https://github.com/hpparvi/ldtk>
- `celerite`/`celerite2` (Foreman-Mackey et al. 2017): <https://celerite2.readthedocs.io/>

**Inference engines**
- `lmfit`: <https://lmfit.github.io/lmfit-py/>
- `emcee` (Foreman-Mackey et al. 2013): <https://arxiv.org/abs/1202.3665> ;
  autocorr/convergence: <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/>
- `dynesty` (Speagle 2020): <https://arxiv.org/abs/1904.02180> ;
  <https://dynesty.readthedocs.io/>
- `ultranest` (Buchner 2021): <https://johannesbuchner.github.io/UltraNest/>

**Turn-key fitters / detection**
- `TLS` (Hippke & Heller 2019): <https://arxiv.org/abs/1901.02015> ;
  <https://transitleastsquares.readthedocs.io/>
- `juliet` (Espinoza et al. 2019): <https://arxiv.org/abs/1812.08549> ;
  <https://juliet.readthedocs.io/>
- `allesfitter` (Günther & Daylan 2021): <https://arxiv.org/abs/2003.14371>
- `EXOFASTv2` (Eastman et al. 2019): <https://arxiv.org/abs/1907.09480>

**Uncertainty / red noise / significance**
- Carter & Winn (2009), ApJ 704, 51 — wavelet likelihood for correlated noise:
  <https://iopscience.iop.org/article/10.1088/0004-637X/704/1/51>
- Pont, Zucker & Queloz (2006) — red noise / time-averaging β: 
  <https://ui.adsabs.harvard.edu/abs/2006MNRAS.373..231P/abstract>
- Foreman-Mackey et al. (2017), AJ 154, 220 — celerite scalable GP:
  <https://iopscience.iop.org/article/10.3847/1538-3881/aa9332>
- SNR of a transit (review): <https://academic.oup.com/mnras/article/523/1/1182/7179431>
