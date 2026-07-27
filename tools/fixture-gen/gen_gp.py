#!/usr/bin/env python3
"""
GaussianProcessModel fixtures, with scikit-learn as the oracle.

    python tools/fixture-gen/gen_gp.py            # regenerate every GP fixture
    python tools/fixture-gen/gen_gp.py --check    # diff against what is committed
    python tools/fixture-gen/gen_gp.py fix_gp_ard # just one

Each fixture asserts two numbers per row: the predicted mean (the target
column) and the predictive variance, which cPMML exposes as the output field
"variance".  The variance exercises the Cholesky solve and k(x*,x*), neither
of which the mean alone pins down.

Coverage of the four PMML kernels, and what each fixture is for:

  fix_gp_rbf          RadialBasisKernel with gamma != 1 and lambda != 1.
                      The lambda attribute is easy to miss: it is a scalar on
                      the element rather than a Lambda child like the other
                      three kernels.
  fix_gp_defaults     RadialBasisKernel with no attributes at all, so every
                      spec default (gamma=1, noiseVariance=1, lambda=1) has to
                      be right.
  fix_gp_ard          ARDSquaredExponentialKernel, two inputs, distinct
                      lambdas -- fixes lambda as a divisor and the 1/2 factor.
  fix_gp_absexp       AbsoluteExponentialKernel, one input.
  fix_gp_genexp_deg2  GeneralizedExponentialKernel degree=2 with the same
                      gamma/lambdas/noise as fix_gp_ard.  The spec formulas
                      make these two models mathematically identical, so the
                      expected columns must match row for row.
  fix_gp_genexp_deg1  Likewise degree=1 against fix_gp_absexp.
  IrisGP              Pre-existing hand-authored model whose expected column
                      was never checked against a reference; the model is left
                      untouched and only the expectations are regenerated.

See oracle_sklearn.py for the kernel formulas and the scikit-learn mapping,
including why AbsoluteExponentialKernel is covered in one dimension only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle_sklearn
from _harness import Fixture, format_number, main, provenance, read_model

SCRIPT = "tools/fixture-gen/gen_gp.py"

# Training sets are literal rather than sampled: no RNG version to pin, and a
# reader can check the kernel matrix by hand if it ever comes to that.
TRAIN_1D = [(1.0, 1.1), (2.0, 2.0), (3.0, 2.9), (4.0, 4.1), (5.0, 5.2)]
TEST_1D = [1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0]

TRAIN_2D = [
    (0.0, 0.0, 0.10),
    (1.0, 0.5, 1.25),
    (2.0, 1.0, 1.80),
    (3.0, 1.5, 1.40),
    (4.0, 2.0, 0.30),
    (5.0, 2.5, -0.60),
]
# The last point sits outside the training hull: variance should be large there.
TEST_2D = [(0.5, 0.25), (1.5, 0.75), (2.5, 1.25), (3.5, 1.75), (4.5, 2.25), (7.0, 3.5)]


# ---------------------------------------------------------------------------
# PMML construction
# ---------------------------------------------------------------------------


def build_pmml(name: str, kernel: str, features: list[str], train: list[tuple], note: str) -> str:
    target = "y"
    fields = features + [target]

    data_fields = "\n".join(
        f'    <DataField name="{f}" optype="continuous" dataType="double"/>' for f in fields
    )
    mining_fields = "\n".join(
        f'      <MiningField name="{f}" usageType="{"target" if f == target else "active"}"/>' for f in fields
    )
    instance_fields = "\n".join(f'        <InstanceField field="{f}" column="{f}"/>' for f in fields)
    rows = "\n".join(
        "        <row>" + "".join(f"<{f}>{format_number(v)}</{f}>" for f, v in zip(fields, row)) + "</row>"
        for row in train
    )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">\n'
        + provenance(SCRIPT, oracle_sklearn.VERSION, note)
        + f'  <DataDictionary numberOfFields="{len(fields)}">\n'
        + data_fields
        + "\n  </DataDictionary>\n"
        + f'  <GaussianProcessModel modelName="{name}" functionName="regression">\n'
        + "    <MiningSchema>\n"
        + mining_fields
        + "\n    </MiningSchema>\n"
        + f"    {kernel}\n"
        + f'    <TrainingInstances recordCount="{len(train)}" fieldCount="{len(fields)}">\n'
        + "      <InstanceFields>\n"
        + instance_fields
        + "\n      </InstanceFields>\n"
        + "      <InlineTable>\n"
        + rows
        + "\n      </InlineTable>\n"
        + "    </TrainingInstances>\n"
        + "  </GaussianProcessModel>\n"
        + "</PMML>\n"
    )


# ---------------------------------------------------------------------------
# Fixture assembly
# ---------------------------------------------------------------------------


def score(name: str, pmml: str, features: list[str], test_points: list) -> Fixture:
    """Attach the oracle's mean and variance to each test point."""
    test_rows = [
        {f: v for f, v in zip(features, point if isinstance(point, tuple) else (point,))}
        for point in test_points
    ]

    _, mean, variance = oracle_sklearn.predict(pmml, test_rows)

    rows = []
    for row, m, v in zip(test_rows, mean, variance):
        rows.append(
            {
                **{f: format_number(row[f]) for f in features},
                "y": format_number(m),
                "variance": format_number(v),
            }
        )

    return Fixture(name=name, columns=features + ["y", "variance"], rows=rows, pmml=pmml)


def generated(name: str, kernel: str, features: list[str], train: list, test: list, note: str) -> Fixture:
    pmml = build_pmml(name, kernel, features, train, note)
    return score(name, pmml, features, test)


def existing(name: str, features: list[str], test: list) -> Fixture:
    """Refresh only the expectations of a committed, hand-authored model."""
    return Fixture(**{**score(name, read_model(name), features, test).__dict__, "pmml": None})


def build() -> list[Fixture]:
    ard_kernel = (
        '<ARDSquaredExponentialKernel gamma="1.7" noiseVariance="0.05">'
        '<Lambda><Array n="2" type="real">0.8 2.5</Array></Lambda>'
        "</ARDSquaredExponentialKernel>"
    )
    genexp_deg2_kernel = (
        '<GeneralizedExponentialKernel gamma="1.7" noiseVariance="0.05" degree="2">'
        '<Lambda><Array n="2" type="real">0.8 2.5</Array></Lambda>'
        "</GeneralizedExponentialKernel>"
    )
    absexp_kernel = (
        '<AbsoluteExponentialKernel gamma="1.3" noiseVariance="0.08">'
        '<Lambda><Array n="1" type="real">1.4</Array></Lambda>'
        "</AbsoluteExponentialKernel>"
    )
    genexp_deg1_kernel = (
        '<GeneralizedExponentialKernel gamma="1.3" noiseVariance="0.08" degree="1">'
        '<Lambda><Array n="1" type="real">1.4</Array></Lambda>'
        "</GeneralizedExponentialKernel>"
    )

    fixtures = [
        generated(
            "fix_gp_rbf",
            '<RadialBasisKernel gamma="1.4" noiseVariance="0.05" lambda="1.6"/>',
            ["x1", "x2"],
            TRAIN_2D,
            TEST_2D,
            "RadialBasisKernel, explicit gamma/lambda/noiseVariance",
        ),
        generated(
            "fix_gp_defaults",
            "<RadialBasisKernel/>",
            ["x1", "x2"],
            TRAIN_2D,
            TEST_2D,
            "RadialBasisKernel with all attributes defaulted (gamma=1, noiseVariance=1, lambda=1)",
        ),
        generated(
            "fix_gp_ard",
            ard_kernel,
            ["x1", "x2"],
            TRAIN_2D,
            TEST_2D,
            "ARDSquaredExponentialKernel, per-dimension lambda",
        ),
        generated(
            "fix_gp_genexp_deg2",
            genexp_deg2_kernel,
            ["x1", "x2"],
            TRAIN_2D,
            TEST_2D,
            "GeneralizedExponentialKernel degree=2, identical by construction to fix_gp_ard",
        ),
        generated(
            "fix_gp_absexp",
            absexp_kernel,
            ["x1"],
            [(x, y) for x, y in TRAIN_1D],
            TEST_1D,
            "AbsoluteExponentialKernel, single input",
        ),
        generated(
            "fix_gp_genexp_deg1",
            genexp_deg1_kernel,
            ["x1"],
            [(x, y) for x, y in TRAIN_1D],
            TEST_1D,
            "GeneralizedExponentialKernel degree=1, identical by construction to fix_gp_absexp",
        ),
        existing("IrisGP", ["x"], TEST_1D),
    ]

    by_name = {f.name: f for f in fixtures}
    # The spec formulas make these pairs the same function, so the oracle must
    # say so too. If this ever trips, the kernel mapping in oracle_sklearn.py
    # is wrong and every expectation it produced is suspect.
    for degenerate, equivalent in (
        ("fix_gp_genexp_deg2", "fix_gp_ard"),
        ("fix_gp_genexp_deg1", "fix_gp_absexp"),
    ):
        if by_name[degenerate].rows != by_name[equivalent].rows:
            raise AssertionError(
                f"{degenerate} and {equivalent} describe the same kernel but the oracle "
                f"scored them differently; check the mapping in oracle_sklearn.py"
            )

    return fixtures


if __name__ == "__main__":
    sys.exit(main(build, __doc__.strip().splitlines()[0]))
