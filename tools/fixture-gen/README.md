# Fixture generators

Test fixtures whose expected values come from an **independent** reference
implementation, so that they say what a model *should* score rather than what
cPMML *currently* scores.

## Why this exists

A fixture is only evidence if something other than cPMML produced its expected
column. Several fixtures in this repository were committed alongside the
evaluator they exercise, with expectations that were computed from the same
reading of the spec — those catch regressions, but they cannot catch a
misread. The generators here close that gap for the model types JPMML does not
cover.

Two rules keep the property from eroding:

1. **The oracle reads the PMML document.** It walks the same XML cPMML walks,
   rather than sharing a Python-side copy of the model parameters. A generator
   bug therefore cannot quietly cancel out an evaluator bug.
2. **Every generator supports `--check`**, which regenerates in memory and
   diffs against the committed files instead of overwriting them. That makes
   "these fixtures still agree with the oracle" a runnable assertion.

## Usage

```sh
python -m venv .venv
.venv/bin/pip install -r tools/fixture-gen/requirements.txt

.venv/bin/python tools/fixture-gen/gen_gp.py            # regenerate
.venv/bin/python tools/fixture-gen/gen_gp.py --check    # verify, non-zero exit on drift
.venv/bin/python tools/fixture-gen/gen_gp.py --list     # names only
.venv/bin/python tools/fixture-gen/gen_gp.py fix_gp_ard # a single fixture
```

Fixtures land in `test/data/model/<name>.zip` and `test/data/dataset/<name>.csv`,
where `test/CMakeLists.txt` discovers them automatically — no test registration
step.

Python is **not** a build or test dependency. `cmake && make && ctest` consumes
the committed CSVs and needs nothing but a compiler; these scripts are for
developers and CI.

## Oracles

| Generator | Oracle | Covers |
|---|---|---|
| `gen_gp.py` | scikit-learn `GaussianProcessRegressor` | `GaussianProcessModel`, all four kernels |
| `gen_timeseries.py` | statsmodels `SARIMAX`, plus spec transcription | `TimeSeriesModel`: ARIMA, `StateSpaceModel`, GARCH conditional mean |
| `gen_ets.py` | statsmodels Holt-Winters | `TimeSeriesModel`: `ExponentialSmoothing` |

JPMML remains the oracle for everything it supports, via
`test/generate_fixtures_jpmml.py` and `tools/cross-validate.sh`. It supports
neither `GaussianProcessModel` nor `TimeSeriesModel`, which is what these
generators are for.

### Coverage limits, stated rather than hidden

`gen_gp.py` covers `RadialBasisKernel` and `ARDSquaredExponentialKernel` in any
number of dimensions. `AbsoluteExponentialKernel` is covered in **one**
dimension only: PMML sums `|x_i - z_i| / lambda_i` coordinate-wise, whereas
scikit-learn's `Matern(nu=0.5)` uses the L2 norm of the scaled difference, and
the two coincide only when there is a single input. scikit-learn has no
anisotropic L1 exponential kernel, and writing one by hand would just restate
cPMML's own formula, so the multi-input case is left uncovered.

`GeneralizedExponentialKernel` is covered at `degree=1` and `degree=2`, where
the spec formula degenerates to the two kernels above. Other degrees have no
scikit-learn equivalent.

`gen_timeseries.py` uses **two** oracles, split by whether a third-party
implementation actually constrains the answer:

- **statsmodels `SARIMAX`** for pure AR models with non-seasonal and seasonal
  differencing. Real recursion, genuinely independent.
- **Spec transcription** (`oracle_spec.py`) for everything SARIMAX cannot
  express: MA terms driven by the `<Residuals>` the document supplies rather
  than residuals re-derived by filtering, and `StateSpaceModel`, whose forecast
  the spec defines as the closed form `Y = G·Fʰ·X` with "the errors ignored" —
  there is no Kalman filter to mirror and no Q/H that PMML ever states.

A transcription is weaker evidence than an independent implementation, so it is
constrained two ways: every pure-AR fixture is computed by *both* oracles and
the generator fails if they disagree, and `fix_arima_spec_example` reproduces
the worked example the specification computes arithmetically in full. That
example is the strongest fixture in the repository — its expected value is the
DMG's own number.

Note that `fix_arima_spec_example` is not on its own discriminating: the
series runs in the thousands, so an error of ±1.6 in the MA term sits inside
the harness's 1e-3 relative tolerance. `fix_arima_ma2` uses small magnitudes
for that reason. Prefer small-magnitude series when the point is to pin a
formula rather than to reproduce a document.

`gen_ets.py` is the clearest case for a third-party oracle in the whole
repository: the spec declines to write the ETS forecast equations at all,
deferring to Gardner (2005), so there is nothing to transcribe and statsmodels'
Holt-Winters implementation of that same taxonomy is genuine evidence. PMML's
*final* smoothed states are injected by zeroing every smoothing weight, which
collapses the state update into the forecast recursion — legitimate because the
spec says the weights are "not needed for scoring". `fix_ets_weights` asserts
exactly that. `trend="polynomial_exponential"` (Brown's method) is uncovered:
statsmodels does not implement it either, and cPMML now rejects it at load.

That same silence is why the ETS *variance* has no oracle at all — see the NaN
column below. Note the asymmetry: the forecast recursion is fully determined by
the smoothed states the document supplies, so a third-party implementation of
Gardner's taxonomy answers it; the variance is a function of the smoothing
weights, whose recursion the document never writes.

GARCH is covered for both the conditional mean and the conditional variance.
The mean fixture exists mainly to pin the MA sign, which the GARCH section of
the spec deliberately writes the opposite way round from the ARIMA section
(`+θ·ε` rather than `(1 − θB)`). The two forecast paths must not be unified.

### Asserting predictive variance

A forecast fixture with an optional `variance` column turns on predictive
variance checking: the harness passes `_variance` and compares
`series_with_variance()` alongside the point forecasts. Same convention as the
GP fixtures' `variance` column, so it reads the same across model types.

Use it wherever the spec defines a variance — `oracle_spec.psi_variance` for
`StateSpaceModel` with a `PsiVector`, `oracle_spec.arima_psi_variance` for
ARIMA, `oracle_spec.garch_variance` for GARCH.

`ExponentialSmoothing` is the exception, and its fixtures assert **NaN**. A
`variance` cell of `nan` is a real expectation: the harness passes it when the
scorer also returns NaN and fails otherwise, in both directions. An ETS document
does not determine a forecast variance — the spec defines no formula, and writes
no smoothing update equations either, so `alpha`/`gamma`/`delta` are not tied to
a parameterisation Gardner's `c_j` weights could be rebuilt from. That is
measurable, not theoretical: statsmodels' `holtwinters` and `ETSModel` agree only
under β = α·γ for the trend, and their seasonal recursions differ by a factor of
(1−α). Nothing in a PMML document chooses between them, so nothing here should
either.

ARIMA variance is the one case with a genuine third-party oracle:
`oracle_statsmodels.arima_forecast_variance` reads `get_forecast().var_pred_mean`,
and because SARIMAX carries differencing inside its state space that is the
variance of the *original* series. `cross_checked_variance` runs it against the
transcription on every pure-AR fixture and fails the build if they disagree —
the two reach the same answer by different routes, SARIMAX through the state
space and `oracle_spec` by convolving the polynomials, so agreement is evidence
rather than two copies of one reading.

Three traps worth knowing, each of which cost a debugging round here:

- GARCH's `<Residuals>` under `ResidualSquareCoefficients` holds ε, not ε².
  The element name suggests otherwise, but `Residuals` means the same thing it
  does in ARIMA's `MA` — the formula does the squaring.
- `gq` is the order of *residual squares* (the α terms) and `gp` the order of
  *variance* (the β terms). Transposing them is invisible unless gp ≠ gq.
- The differencing operators belong in the ψ recursion. `(1−B)^d (1−B^m)^D`
  multiplies into Φ before the weights are derived; run it on the stationary φ
  alone and the answer describes the differenced series. Every ARIMA fixture
  here has d ≥ 1, so the two are never accidentally equal.

### The exactLeastSquares path

`fix_arima_kalman_state` is a **pure AR on purpose**. A pure AR's terminal state
is an exact function of the observed data whichever basis the filter works in,
so statsmodels pins the whole path — forecast and variance — without the fixture
having to commit to a state-space convention.

It carries no `MaximumLikelihoodStat`. cPMML's supplied-state branch reads
`FinalStateVector` in a basis the spec does not define it in (M-S13), so a
fixture for it would be a fixture for the bug. ARMA models on this path are
wrong for a related reason — the MA polynomial never reaches the forecast — and
`ARIMA_Kalman.zip` still holds self-derived expectations that record that
behaviour rather than contradict it. Do not treat it as evidence.

## Invariants the generators assert

Beyond agreeing with the oracle, `gen_gp.py` checks two things itself, and
fails loudly rather than emitting a fixture it cannot stand behind:

- **Kernel mapping.** `oracle_sklearn.check_mapping` compares the scikit-learn
  kernel matrix against the PMML formula transcribed literally from the spec,
  to 1e-12. This audits the one hand-written step between the PMML document
  and the oracle.
- **Degenerate-kernel identities.** `fix_gp_genexp_deg2` must score
  identically to `fix_gp_ard`, and `fix_gp_genexp_deg1` to `fix_gp_absexp`,
  because the spec formulas make each pair the same function.

## Adding a generator

Write `gen_<area>.py` next to these, import `Fixture`/`main` from `_harness`,
and return a list of fixtures from `build()`. A fixture with `pmml=None` keeps
the committed model and refreshes only the expected columns — the right mode
for a hand-authored PMML whose expectations were never checked against a
reference (`IrisGP` is the example).

Record provenance in the generated `<Header>` via `_harness.provenance`. Both
evaluators ignore Header content, so it costs nothing at scoring time and makes
an oracle-derived fixture distinguishable at a glance from a hand-computed one.
