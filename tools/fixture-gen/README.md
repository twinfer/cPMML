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
