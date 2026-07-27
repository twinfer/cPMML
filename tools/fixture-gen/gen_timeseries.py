#!/usr/bin/env python3
"""
TimeSeriesModel fixtures for ARIMA and StateSpaceModel.

    python tools/fixture-gen/gen_timeseries.py            # regenerate
    python tools/fixture-gen/gen_timeseries.py --check     # diff against committed
    python tools/fixture-gen/gen_timeseries.py fix_arima_ma2

Forecast fixtures hold a single `forecast` column; the row count is the
horizon (see run_forecast in test/pmml_tester.cc).

Which oracle answers which fixture, and why
-------------------------------------------
JPMML-Evaluator does not support TimeSeriesModel at all, so there is no
drop-in reference. Two oracles split the work by whether a third-party
implementation actually constrains the answer:

  statsmodels (oracle_statsmodels.py)   pure AR models with non-seasonal and
    seasonal differencing. Real recursion, real machinery, genuinely
    independent -- and the home of the open findings about differencing.

  spec transcription (oracle_spec.py)   everything statsmodels cannot express:
    MA terms driven by the residuals the PMML supplies rather than residuals
    re-derived by filtering, the constantTerm shift, and StateSpaceModel,
    whose forecast the spec defines as a closed form with the errors ignored.

The two are cross-checked against each other on every pure-AR fixture, where
both apply, and the transcription oracle is additionally anchored on the
worked example the specification computes arithmetically in full
(fix_arima_spec_example). That example is the strongest fixture here: its
expected value is the DMG's own number, not one this repository produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import oracle_spec
import oracle_statsmodels
from _harness import Fixture, format_number, main, provenance

SCRIPT = "tools/fixture-gen/gen_timeseries.py"

# The series the specification uses in its ARIMA scoring example.
SPEC_SERIES = [11357.92, 10605.95, 16998.57, 6563.75, 6607.69, 9839.0]
# The value it derives arithmetically for index 7, quoted from the spec:
#   orders = 1.05*9839 + 0.05*6607.69 + 0.2*6563.75 - 0.3*16998.57 - (-0.4)*2
SPEC_FIRST_FORECAST = 1.05 * 9839 + 0.05 * 6607.69 + 0.2 * 6563.75 - 0.3 * 16998.57 - (-0.4) * 2

SERIES = [100.0, 102.0, 105.0, 107.0, 110.0, 114.0, 117.0, 121.0, 124.0, 128.0]


# ---------------------------------------------------------------------------
# PMML construction
# ---------------------------------------------------------------------------


def _array(values, indent: str) -> str:
    body = " ".join(format_number(v) for v in values)
    return f'{indent}<Array type="real" n="{len(values)}">{body}</Array>'


def _document(name: str, note: str, body: str, series: list[float] | None) -> str:
    time_series = ""
    if series is not None:
        values = "\n".join(
            f'      <TimeValue index="{i}" value="{format_number(v)}"/>' for i, v in enumerate(series, start=1)
        )
        time_series = f'    <TimeSeries usage="original">\n{values}\n    </TimeSeries>\n'

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">\n'
        + provenance(SCRIPT, note.split(";")[0], note)
        + '  <DataDictionary numberOfFields="1">\n'
        + '    <DataField name="ts" optype="continuous" dataType="double"/>\n'
        + "  </DataDictionary>\n"
        + f'  <TimeSeriesModel modelName="{name}" functionName="timeSeries" bestFit="{"ARIMA" if series is not None else "StateSpaceModel"}">\n'
        + "    <MiningSchema>\n"
        + '      <MiningField name="ts" usageType="target"/>\n'
        + "    </MiningSchema>\n"
        + time_series
        + body
        + "  </TimeSeriesModel>\n"
        + "</PMML>\n"
    )


def arima_document(name, note, series, *, ar=None, ma=None, residuals=None, d=0, seasonal_ar=None,
                   seasonal_d=0, period=1, constant=0.0) -> str:
    p, q = len(ar or []), len(ma or [])
    parts = [f'    <ARIMA constantTerm="{format_number(constant)}" predictionMethod="conditionalLeastSquares">']
    parts.append(f'      <NonseasonalComponent p="{p}" d="{d}" q="{q}">')
    if ar:
        parts.append("        <AR>")
        parts.append(_array(ar, "          "))
        parts.append("        </AR>")
    if ma:
        parts.append("        <MA>")
        parts.append("          <MACoefficients>")
        parts.append(_array(ma, "            "))
        parts.append("          </MACoefficients>")
        parts.append("          <Residuals>")
        parts.append(_array(residuals or [], "            "))
        parts.append("          </Residuals>")
        parts.append("        </MA>")
    parts.append("      </NonseasonalComponent>")
    if seasonal_ar or seasonal_d:
        parts.append(
            f'      <SeasonalComponent P="{len(seasonal_ar or [])}" D="{seasonal_d}" Q="0" period="{period}">'
        )
        if seasonal_ar:
            parts.append("        <AR>")
            parts.append(_array(seasonal_ar, "          "))
            parts.append("        </AR>")
        parts.append("      </SeasonalComponent>")
    parts.append("    </ARIMA>")

    return _document(name, note, "\n".join(parts) + "\n", series)


def _matrix(element: str, rows: list[list[float]]) -> str:
    arrays = "\n".join(_array(row, "          ") for row in rows)
    return (
        f"      <{element}>\n"
        f'        <Matrix nbRows="{len(rows)}" nbCols="{len(rows[0])}">\n'
        f"{arrays}\n"
        "        </Matrix>\n"
        f"      </{element}>"
    )


def ssm_document(name, note, state, transition, measurement, *, intercept=0.0, variance=None, psi=None) -> str:
    attrs = f' intercept="{format_number(intercept)}"'
    if variance is not None:
        attrs += f' variance="{format_number(variance)}"'

    psi_element = ""
    if psi is not None:
        psi_element = f"      <PsiVector>\n{_array(psi, '        ')}\n      </PsiVector>\n"

    body = (
        f"    <StateSpaceModel{attrs}>\n"
        + psi_element
        + "      <StateVector>\n"
        f"{_array(state, '        ')}\n"
        "      </StateVector>\n"
        f"{_matrix('TransitionMatrix', transition)}\n"
        f"{_matrix('MeasurementMatrix', measurement)}\n"
        "    </StateSpaceModel>\n"
    )
    return _document(name, note, body, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def forecast_fixture(name: str, pmml: str, values) -> Fixture:
    return Fixture(
        name=name,
        columns=["forecast"],
        rows=[{"forecast": format_number(v)} for v in values],
        pmml=pmml,
    )


def build() -> list[Fixture]:
    fixtures: list[Fixture] = []
    horizon = 5

    # --- 1. The specification's own worked example --------------------------
    spec_kw = dict(ar=[0.05, 0.1, 0.3], d=1, ma=[-0.4], residuals=[2.0])
    spec_values = oracle_spec.arima_cls_forecast(SPEC_SERIES, horizon, **spec_kw)
    if not np.isclose(spec_values[0], SPEC_FIRST_FORECAST, rtol=1e-12):
        raise AssertionError(
            f"transcription oracle gives {spec_values[0]!r} for the spec's worked example, "
            f"but the spec computes {SPEC_FIRST_FORECAST!r}"
        )
    fixtures.append(
        forecast_fixture(
            "fix_arima_spec_example",
            arima_document(
                "fix_arima_spec_example",
                "PMML 4.4 spec transcription; ARIMA(3,1,1) worked example from TimeSeriesModel.html, "
                "h=1 equals the value the spec computes arithmetically",
                SPEC_SERIES,
                **spec_kw,
            ),
            spec_values,
        )
    )

    # --- 2. Pure AR: statsmodels, cross-checked against the transcription ----
    ar_cases = [
        ("fix_arima_ar2_d2", dict(ar=[0.5, -0.2], d=2), "non-seasonal differencing of order 2"),
        (
            "fix_arima_sarima",
            dict(ar=[0.3], d=1, seasonal_ar=[0.5], seasonal_d=1, period=4),
            "seasonal and non-seasonal differencing combined",
        ),
        # constantTerm is cross-checked rather than transcribed: the spec says
        # the constant is "added to the predicted value", which reads as a
        # post-hoc shift, but mu sits inside the equation and the shifted value
        # feeds the next step. The two readings agree at h=1 and diverge after,
        # and statsmodels' trend='c' settles it.
        ("fix_arima_constant", dict(ar=[0.6], constant=12.5), "constantTerm inside the recursion"),
    ]
    for name, kw, description in ar_cases:
        by_statsmodels = oracle_statsmodels.arima_forecast(SERIES, horizon, **kw)
        by_spec = oracle_spec.arima_cls_forecast(SERIES, horizon, **kw)
        if not np.allclose(by_statsmodels, by_spec, rtol=1e-9):
            raise AssertionError(
                f"{name}: statsmodels and the spec transcription disagree\n"
                f"  statsmodels {by_statsmodels}\n  transcription {by_spec}"
            )
        fixtures.append(
            forecast_fixture(
                name,
                arima_document(
                    name,
                    f"{oracle_statsmodels.VERSION} SARIMAX, cross-checked against the spec "
                    f"transcription; {description}",
                    SERIES,
                    **kw,
                ),
                by_statsmodels,
            )
        )

    # --- 3. MA with supplied residuals: transcription only -------------------
    # statsmodels re-derives residuals by filtering; PMML applies the ones the
    # document supplies, most recent first.
    ma_kw = dict(ar=[0.4], d=1, ma=[-0.3, 0.15], residuals=[1.5, -2.0])
    fixtures.append(
        forecast_fixture(
            "fix_arima_ma2",
            arima_document(
                "fix_arima_ma2",
                "PMML 4.4 spec transcription; q=2 with supplied Residuals, most recent first, "
                "zero after time t",
                SERIES,
                **ma_kw,
            ),
            oracle_spec.arima_cls_forecast(SERIES, horizon, **ma_kw),
        )
    )

    # --- 4. StateSpaceModel, in the schema-conformant encoding ---------------
    # The three pre-existing SSM_* fixtures are regenerated here rather than
    # left alone: they encoded the matrices in a shape PMML does not define
    # (rows/cols attributes and flat text on TransitionMatrix), so they had to
    # be rewritten, and their expected values were self-derived anyway.
    ssm_cases = [
        (
            "fix_ssm_local_trend",
            dict(state=[100.0, 3.0], transition=[[1.0, 1.0], [0.0, 1.0]], measurement=[[1.0, 0.0]]),
            "local linear trend",
        ),
        (
            "fix_ssm_coupled",
            dict(
                state=[50.0, -1.5, 2.0],
                transition=[[0.9, 0.2, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.8]],
                measurement=[[1.0, 1.0, 0.0]],
                intercept=7.25,
            ),
            "non-trivial F with the intercept attribute",
        ),
        (
            "SSM_LinearTrend",
            dict(state=[100.0, 3.0], transition=[[1.0, 1.0], [0.0, 1.0]], measurement=[[1.0, 0.0]]),
            "local linear trend, re-encoded to the schema-conformant Matrix form",
        ),
        (
            "SSM_Intercept",
            dict(state=[5.0], transition=[[1.0]], measurement=[[1.0]], intercept=10.0),
            "constant state plus intercept, re-encoded",
        ),
        (
            "SSM_PsiVector",
            dict(
                state=[5.0],
                transition=[[0.8]],
                measurement=[[1.0]],
                variance=2.0,
                psi=[0.8, 0.64, 0.512, 0.4096],
            ),
            "PsiVector present, re-encoded; note the spec's array is psi_1 onward, "
            "the implicit psi_0 = 1 is not written out",
        ),
    ]
    for name, kw, description in ssm_cases:
        values = oracle_spec.ssm_forecast(
            kw["state"], kw["transition"], kw["measurement"], horizon, kw.get("intercept", 0.0)
        )
        fixtures.append(
            forecast_fixture(
                name,
                ssm_document(
                    name,
                    f"PMML 4.4 spec transcription; Y=G*F^h*X with errors ignored, {description}",
                    **kw,
                ),
                values,
            )
        )

    return fixtures


if __name__ == "__main__":
    sys.exit(main(build, __doc__.strip().splitlines()[0]))
