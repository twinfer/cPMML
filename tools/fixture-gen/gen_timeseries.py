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
    seasonal differencing, and the whole exactLeastSquares path including
    q > 0. Real recursion, real machinery, genuinely independent -- and the
    home of the findings about differencing and about the state space.

  spec transcription (oracle_spec.py)   what statsmodels cannot express:
    conditionalLeastSquares MA terms, driven by the residuals the PMML
    supplies rather than by residuals re-derived by filtering, and
    StateSpaceModel, whose forecast the spec defines as a closed form with
    the errors ignored.

The two are cross-checked against each other on every pure-AR fixture, where
both apply, and the transcription oracle is additionally anchored on the
worked example the specification computes arithmetically in full
(fix_arima_spec_example). That example is the strongest fixture here: its
expected value is the DMG's own number, not one this repository produced.
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import oracle_spec
import oracle_statsmodels
from _harness import Fixture, format_number, main, provenance, read_model

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


def _kalman_state(final_state, h_vector) -> str:
    """<MaximumLikelihoodStat> as the schema defines it: a child of ARIMA, with
    FinalOmega and FinalStateVector required and HVector optional.

    FinalOmega is emitted because the schema requires it, not because scoring
    needs it -- the spec's Omega(h) recursion runs in the differenced basis, so
    it describes w rather than y, and the psi-weight variance answers the
    question the forecast actually poses. A conformant document has to carry it
    either way, which is the point of including it here.
    """
    mm = len(final_state)
    omega = [[0.0] * mm for _ in range(mm)]
    h_element = ""
    if h_vector is not None:
        h_element = f"        <HVector>\n{_array(h_vector, '          ')}\n        </HVector>\n"
    return (
        '      <MaximumLikelihoodStat method="kalman">\n'
        "        <KalmanState>\n"
        f"{_matrix('FinalOmega', omega, '          ')}\n"
        f"          <FinalStateVector>\n{_array(final_state, '            ')}\n          </FinalStateVector>\n"
        f"{h_element}"
        "        </KalmanState>\n"
        "      </MaximumLikelihoodStat>\n"
    )


def arima_document(name, note, series, *, ar=None, ma=None, residuals=None, d=0, seasonal_ar=None,
                   seasonal_d=0, period=1, constant=0.0, rmse=None,
                   prediction_method="conditionalLeastSquares",
                   final_state=None, h_vector=None) -> str:
    p, q = len(ar or []), len(ma or [])
    rmse_attr = "" if rmse is None else f' RMSE="{format_number(rmse)}"'
    parts = [f'    <ARIMA{rmse_attr} constantTerm="{format_number(constant)}" '
             f'predictionMethod="{prediction_method}">']
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
    body = "\n".join(parts) + "\n"
    if final_state is not None:
        body += _kalman_state(final_state, h_vector)
    body += "    </ARIMA>\n"

    return _document(name, note, body, series)


def garch_document(name, *, ar, ma, residuals, constant, omega, alpha, garch_residuals, beta,
                   past_variances) -> str:
    """A GARCH model: ARMAPart drives the conditional mean, GARCHPart the
    conditional variance. gq is the order of residual squares (the alphas) and
    gp the order of variance (the betas), per the spec's own wording.

    There are two distinct <Residuals> arrays: `residuals` under ARMAPart/MA
    feeds the mean, `garch_residuals` under ResidualSquareCoefficients feeds
    the variance and is squared by the formula."""
    ma_block = ""
    if ma:
        ma_block = (
            f"        <MA>\n"
            f"          <MACoefficients>\n{_array(ma, '            ')}\n          </MACoefficients>\n"
            f"          <Residuals>\n{_array(residuals, '            ')}\n          </Residuals>\n"
            f"        </MA>\n"
        )

    body = (
        f"    <GARCH>\n"
        f'      <ARMAPart p="{len(ar)}" q="{len(ma)}" constant="{format_number(constant)}">\n'
        f"        <AR>\n{_array(ar, '          ')}\n        </AR>\n"
        f"{ma_block}"
        f"      </ARMAPart>\n"
        f'      <GARCHPart gp="{len(beta)}" gq="{len(alpha)}" constant="{format_number(omega)}">\n'
        f"        <ResidualSquareCoefficients>\n"
        f"          <MACoefficients>\n{_array(alpha, '            ')}\n          </MACoefficients>\n"
        f"          <Residuals>\n{_array(garch_residuals, '            ')}\n          </Residuals>\n"
        f"        </ResidualSquareCoefficients>\n"
        f"        <VarianceCoefficients>\n"
        f"          <MACoefficients>\n{_array(beta, '            ')}\n          </MACoefficients>\n"
        f"          <PastVariances>\n{_array(past_variances, '            ')}\n          </PastVariances>\n"
        f"        </VarianceCoefficients>\n"
        f"      </GARCHPart>\n"
        f"    </GARCH>\n"
    )
    note = ("PMML 4.4 spec transcription; GARCH conditional mean and variance. The mean's MA term is "
            "+theta*eps here, the opposite sign to the ARIMA section")
    return _document(name, note, body, SERIES).replace('bestFit="ARIMA"', 'bestFit="GARCH"')


def _matrix(element: str, rows: list[list[float]], indent: str = "      ") -> str:
    arrays = "\n".join(_array(row, indent + "    ") for row in rows)
    return (
        f"{indent}<{element}>\n"
        f'{indent}  <Matrix nbRows="{len(rows)}" nbCols="{len(rows[0])}">\n'
        f"{arrays}\n"
        f"{indent}  </Matrix>\n"
        f"{indent}</{element}>"
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


def forecast_fixture(name: str, pmml: str, values, variance=None) -> Fixture:
    """A forecast fixture. An optional variance column turns on predictive
    variance checking in the harness (see run_forecast in pmml_tester.cc)."""
    columns = ["forecast"] if variance is None else ["forecast", "variance"]
    rows = []
    for i, v in enumerate(values):
        row = {"forecast": format_number(v)}
        if variance is not None:
            row["variance"] = format_number(variance[i])
        rows.append(row)

    return Fixture(name=name, columns=columns, rows=rows, pmml=pmml)


# RMSE for the pure-AR fixtures. sigma^2 scales the variance linearly, so any
# value works; a non-unit one keeps a dropped or doubled sigma^2 visible.
AR_RMSE = 1.2


def cross_checked_variance(name: str, horizon: int, rmse: float, kw: dict, rtol: float = 1e-9) -> np.ndarray:
    """Forecast variance from statsmodels, cross-checked against the transcription.

    Both oracles have to fold the differencing operators into the AR polynomial
    to get this right, and they arrive there by different routes: SARIMAX
    carries (1-B)^d inside its state space, while oracle_spec convolves the
    polynomials explicitly. Agreement is therefore real evidence and not two
    copies of one reading.

    constantTerm shifts the mean, not the spread, so it is dropped here; so are
    the `<Residuals>`, which the variance does not read.

    rtol is loosened for q > 0, where the two answer genuinely different
    questions: statsmodels keeps the finite-sample uncertainty of the filtered
    state, while the psi formula is the asymptotic variance. On the ten-point
    SERIES that is a 0.6% gap at h=1, decaying with h -- larger than the
    harness's own 0.1% tolerance, so the committed column has to be one or the
    other and it is the spec's. The loose band is still worth running: it would
    catch a sign or a factor, just not a convention.
    """
    var_kw = {k: v for k, v in kw.items() if k not in ("constant", "residuals")}
    by_statsmodels = oracle_statsmodels.arima_forecast_variance(SERIES, horizon, rmse**2, **var_kw)
    by_spec = oracle_spec.arima_psi_variance(horizon, rmse**2, **var_kw)
    if not np.allclose(by_statsmodels, by_spec, rtol=rtol):
        raise AssertionError(
            f"{name}: statsmodels and the spec transcription disagree on forecast variance\n"
            f"  statsmodels {by_statsmodels}\n  transcription {by_spec}"
        )
    return by_spec if var_kw.get("ma") else by_statsmodels


def regenerate_arima_kalman(horizon: int) -> Fixture:
    """ARIMA_Kalman: keep the committed hand-authored model, regenerate the
    expectations from it.

    The oracle reads the document rather than a copy of its numbers, which is
    the whole point of the pmml=None mode -- the committed expectations were
    produced by the evaluator under test and cannot be trusted as a starting
    point for anything.
    """
    ns = {"p": "http://www.dmg.org/PMML-4_4"}
    root = ElementTree.fromstring(read_model("ARIMA_Kalman"))

    def numbers(path: str) -> list[float]:
        node = root.find(path, ns)
        return [] if node is None else [float(v) for v in node.text.split()]

    series = [float(tv.get("value")) for tv in root.findall(".//p:TimeSeries/p:TimeValue", ns)]
    ar = numbers(".//p:NonseasonalComponent/p:AR/p:Array")
    ma = numbers(".//p:NonseasonalComponent/p:MA/p:MACoefficients/p:Array")
    nsc = root.find(".//p:NonseasonalComponent", ns)
    d = int(nsc.get("d", "0"))
    arima = root.find(".//p:ARIMA", ns)
    constant = float(arima.get("constantTerm", "0"))
    # No RMSE attribute means sigma^2 = 1, which is what cPMML defaults to.
    sigma2 = float(arima.get("RMSE", "1")) ** 2

    kw = dict(ar=ar, ma=ma, d=d, constant=constant)
    return Fixture(
        **{
            **forecast_fixture(
                "ARIMA_Kalman",
                "",
                oracle_statsmodels.arima_exact_forecast(series, horizon, **kw),
                oracle_spec.arima_psi_variance(horizon, sigma2, ar=ar, ma=ma, d=d),
            ).__dict__,
            "pmml": None,
        }
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
        variance = cross_checked_variance(name, horizon, AR_RMSE, kw)
        fixtures.append(
            forecast_fixture(
                name,
                arima_document(
                    name,
                    f"{oracle_statsmodels.VERSION} SARIMAX, cross-checked against the spec "
                    f"transcription; {description}",
                    SERIES,
                    rmse=AR_RMSE,
                    **kw,
                ),
                by_statsmodels,
                variance,
            )
        )

    # --- 2b. exactLeastSquares ----------------------------------------------
    # The spec's state space, S(t+1) = F S(t) + H e(t), Y(t) = G S(t) + e(t),
    # with m = max(p,q) and G = (1,0,...,0). statsmodels is a full oracle on
    # this path including for q > 0, because exactLeastSquares recovers the
    # innovations by filtering rather than reading <Residuals> -- see the scope
    # note in oracle_statsmodels.py.
    #
    # A pure AR is not enough on its own: it is exactly the case a wrong state
    # space can still get right, since its terminal state is an exact function
    # of the observed data whichever basis the filter works in. fix_arima_kalman
    # _arma is the one that pins the form.
    exact_cases = [
        (
            "fix_arima_kalman_state",
            dict(ar=[0.5, -0.2], d=1),
            "pure AR; exactLeastSquares must agree with conditionalLeastSquares on the same coefficients",
        ),
        (
            "fix_arima_kalman_arma",
            dict(ar=[0.5, -0.2], ma=[-0.4, 0.3], d=1),
            "ARMA with q>0: the moving-average polynomial reaches the forecast only through H",
        ),
        (
            "fix_arima_kalman_constant",
            dict(ar=[0.5, -0.2], d=1, constant=0.8),
            "constantTerm on the exactLeastSquares path, where the zero-mean state space "
            "takes it as mu = c/(1-sum phi) rather than as an intercept",
        ),
    ]
    for name, kw, description in exact_cases:
        # q>0 loosens the variance cross-check; see cross_checked_variance.
        variance = cross_checked_variance(name, horizon, AR_RMSE, kw, rtol=1e-2 if kw.get("ma") else 1e-9)
        fixtures.append(
            forecast_fixture(
                name,
                arima_document(
                    name,
                    f"{oracle_statsmodels.VERSION} SARIMAX; {description}",
                    SERIES,
                    rmse=AR_RMSE,
                    prediction_method="exactLeastSquares",
                    **kw,
                ),
                oracle_statsmodels.arima_exact_forecast(SERIES, horizon, **kw),
                variance,
            )
        )

    # --- 2c. the supplied-state shortcut -------------------------------------
    # ARIMA/MaximumLikelihoodStat/KalmanState lets a producer ship the state its
    # own filter reached, so the scorer does not re-derive it. Testing that it is
    # actually honoured needs a document where re-deriving gives a *different*
    # answer, so the TimeSeries here carries only the tail of the data the state
    # was estimated on -- a realistic shape for a producer that ships a fitted
    # model rather than its whole training set.
    #
    # The state itself is oracle-derived rather than transcribed. Since
    # G = (1,0,...,0), S = (Y(n+1|n), ..., Y(n+m|n)) in the zero-mean basis, so
    # the m numbers a producer would write are just statsmodels' first m
    # forecasts less mu. Nothing here commits to an implementation's internal
    # representation, which is what made the earlier fixture for this branch
    # impossible to write honestly.
    supplied_kw = dict(ar=[0.6, -0.2], ma=[-0.3, 0.15], constant=60.0)
    supplied_state = oracle_statsmodels.arima_exact_state(SERIES, **supplied_kw)
    supplied_values = oracle_statsmodels.arima_exact_forecast(SERIES, horizon, **supplied_kw)
    shipped = SERIES[-4:]
    if np.allclose(supplied_values, oracle_statsmodels.arima_exact_forecast(shipped, horizon, **supplied_kw),
                   rtol=1e-3):
        raise AssertionError(
            "fix_arima_kalman_supplied: filtering the shipped TimeSeries reproduces the supplied state's "
            "forecast to within the harness tolerance, so the fixture cannot tell the two apart"
        )
    fixtures.append(
        forecast_fixture(
            "fix_arima_kalman_supplied",
            arima_document(
                "fix_arima_kalman_supplied",
                f"{oracle_statsmodels.VERSION} SARIMAX; FinalStateVector and HVector supplied, and the "
                "TimeSeries deliberately shorter than the data they were estimated on",
                shipped,
                rmse=AR_RMSE,
                prediction_method="exactLeastSquares",
                final_state=supplied_state,
                h_vector=oracle_spec.arima_h_vector(supplied_kw["ar"], supplied_kw["ma"]),
                **supplied_kw,
            ),
            supplied_values,
            cross_checked_variance("fix_arima_kalman_supplied", horizon, AR_RMSE, supplied_kw, rtol=1e-2),
        )
    )

    # --- 2d. the pre-existing hand-authored Kalman fixture -------------------
    # ARIMA_Kalman shipped with expectations that recorded cPMML's own output
    # (h=1 = 1.1, which is 0.5 * 2.2 -- the AR term alone, with the MA
    # polynomial dropped). The model is left untouched and only the expected
    # columns are regenerated, from the document itself.
    fixtures.append(regenerate_arima_kalman(horizon))

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
                "zero after time t; RMSE drives the psi-weight forecast variance",
                SERIES,
                rmse=1.5,
                **ma_kw,
            ),
            oracle_spec.arima_cls_forecast(SERIES, horizon, **ma_kw),
            # The psi recursion subtracts theta for the same reason the forecast
            # does, so this column is a second, independent check on that sign.
            # d=1 matters here: the differencing operator folds into the AR
            # polynomial, so this is not the variance of the differenced series.
            oracle_spec.arima_psi_variance(
                horizon, 1.5**2, ar=ma_kw["ar"], ma=ma_kw["ma"], d=ma_kw["d"]
            ),
        )
    )

    # --- 4. GARCH conditional mean ------------------------------------------
    # The only GARCH fixture in the repository was an AR(1) in disguise: q=0,
    # so its expected values were identical to ARIMA_AR1's and the ARMAPart's
    # MA handling was never exercised. This uses q=2 to pin the sign, which the
    # spec deliberately writes the opposite way round from ARIMA.
    garch_kw = dict(ar=[0.5, -0.2], ma=[0.3, 0.15], residuals=[1.5, -2.0], constant=1.25)
    # The oracle takes `residuals`; the document builder calls the same array
    # `garch_residuals`, to keep it apart from ARMAPart's own <Residuals>.
    def variance_of(spec):
        return oracle_spec.garch_variance(horizon, **{
            ("residuals" if k == "garch_residuals" else k): v for k, v in spec.items()
        })

    garch_var = dict(omega=0.2, alpha=[0.3], garch_residuals=[1.0], beta=[0.5], past_variances=[2.0])
    fixtures.append(
        forecast_fixture(
            "fix_garch_mean_ma2",
            garch_document("fix_garch_mean_ma2", **garch_kw, **garch_var),
            oracle_spec.garch_mean_forecast(SERIES, horizon, **garch_kw),
            variance_of(garch_var),
        )
    )

    # gq (residual squares) and gp (variance) differ here on purpose. With
    # gp == gq -- every other GARCH fixture -- transposing the two orders is
    # invisible, because both loops happen to run the same number of times.
    asym_var = dict(
        omega=0.15,
        alpha=[0.25, 0.1, 0.05],  # gq = 3
        garch_residuals=[1.2, 0.8, 1.6],
        beta=[0.4],  # gp = 1
        past_variances=[2.5],
    )
    fixtures.append(
        forecast_fixture(
            "fix_garch_variance_asym",
            garch_document("fix_garch_variance_asym", ar=[0.5], ma=[], residuals=[], constant=0.0, **asym_var),
            oracle_spec.garch_mean_forecast(SERIES, horizon, ar=[0.5]),
            variance_of(asym_var),
        )
    )

    # --- 5. StateSpaceModel, in the schema-conformant encoding ---------------
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
        # PsiVector gives an exact forecast variance; without one there is
        # nothing spec-defined to assert, so the column is omitted.
        variance = None
        if kw.get("psi") is not None:
            variance = oracle_spec.psi_variance(kw["variance"], kw["psi"], horizon)

        fixtures.append(
            forecast_fixture(
                name,
                ssm_document(
                    name,
                    f"PMML 4.4 spec transcription; Y=G*F^h*X with errors ignored, {description}",
                    **kw,
                ),
                values,
                variance,
            )
        )

    return fixtures


if __name__ == "__main__":
    sys.exit(main(build, __doc__.strip().splitlines()[0]))
