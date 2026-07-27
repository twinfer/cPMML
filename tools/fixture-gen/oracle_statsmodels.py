#!/usr/bin/env python3
"""
statsmodels oracle for PMML ARIMA forecasts.

Used where a third-party implementation genuinely constrains the answer: the
AR recursion combined with non-seasonal and seasonal differencing, which is
substantial machinery and is where the open time-series findings live.

Scope, and why it is narrow -- but not on the exactLeastSquares path
--------------------------------------------------------------------
For **conditionalLeastSquares** this oracle covers q = Q = 0 only.

That method takes the past residuals as *given* -- an explicit `<Residuals>`
array in the document -- and applies the MA coefficients to those supplied
numbers. statsmodels derives residuals by running the Kalman filter over the
data instead, so for q > 0 it answers a different question and would agree with
a conformant PMML scorer only by coincidence. Those cases are covered by
oracle_spec.arima_cls_forecast, anchored on the worked example the
specification itself works through arithmetically.

**exactLeastSquares lifts the restriction**, and that is worth stating plainly
because it is easy to carry the q=0 rule over by habit. On that path PMML does
not consult `<Residuals>` at all: the spec's state space recovers the
innovations by filtering, which is precisely what statsmodels does. The two
implement the same estimator, so statsmodels is a genuine oracle for ARMA there
-- see arima_exact_forecast. The MA sign is the only convention that has to be
converted.

Two conventions differ and are handled here rather than assumed away:

  * MA sign. PMML writes (1 - theta B); statsmodels writes (1 + theta B). Not
    exercised while q = 0, but noted so the restriction is not silently lifted.
  * constantTerm. PMML says the constant "should be added to the predicted
    value", a shift applied to the forecast. statsmodels' trend='c' places an
    intercept inside the differenced mean equation, which is a different model.
    This oracle therefore fits with no trend and applies the shift afterwards,
    matching the specification's wording.
"""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

VERSION = f"statsmodels {statsmodels.__version__}"


class Unsupported(Exception):
    """The oracle cannot express this model; use the transcription oracle."""


def _filtered(
    series: list[float],
    ar: list[float] | None,
    d: int,
    seasonal_ar: list[float] | None,
    seasonal_d: int,
    period: int,
    constant: float,
    sigma2: float,
    ma: list[float] | None = None,
    stationary_init: bool = False,
):
    ar = list(ar or [])
    ma = list(ma or [])
    seasonal_ar = list(seasonal_ar or [])

    endog = np.asarray(series, dtype=float)
    order = (len(ar), d, len(ma))
    seasonal_order = (len(seasonal_ar), seasonal_d, 0, period if seasonal_ar or seasonal_d else 0)

    model = SARIMAX(
        endog,
        order=order,
        seasonal_order=seasonal_order,
        trend="c" if constant else None,
        # enforce_* are what SARIMAX keys its *initialisation* off in .filter():
        # left False it starts approximately diffuse, set True it starts from the
        # stationary distribution of the state. That choice is not cosmetic for
        # q > 0 -- on ARIMA_Kalman it is 0.608 against 0.594 -- and the spec's
        # exact least squares is the stationary one: Omega_1 is the unconditional
        # covariance solving Omega = F Omega F' + H H'. Pure AR models are
        # insensitive to it, which is why it went unnoticed until an ARMA fixture
        # existed. See stationary_init on the callers.
        enforce_stationarity=stationary_init,
        enforce_invertibility=stationary_init,
    )

    # PMML's AR array is phi_1..phi_p, the same order SARIMAX names them in;
    # the intercept leads when present and sigma2 is last. PMML writes the MA
    # polynomial as (1 - theta B) and statsmodels as (1 + theta B), so the sign
    # flips on the way across.
    expected = list(model.param_names)
    params = ([constant] if constant else []) + ar + [-t for t in ma] + seasonal_ar + [sigma2]
    if len(params) != len(expected):
        raise Unsupported(f"parameter mismatch: SARIMAX wants {expected}, built {len(params)} values")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.filter(params)


def arima_forecast(
    series: list[float],
    horizon: int,
    ar: list[float] | None = None,
    d: int = 0,
    seasonal_ar: list[float] | None = None,
    seasonal_d: int = 0,
    period: int = 1,
    constant: float = 0.0,
) -> np.ndarray:
    """Forecast a pure AR/ARI/SARI model with the coefficients PMML supplies.

    sigma2 does not affect the point forecast, so it is left at 1.
    """
    result = _filtered(series, ar, d, seasonal_ar, seasonal_d, period, constant, 1.0)
    return np.asarray(result.forecast(horizon), dtype=float)


def arima_forecast_variance(
    series: list[float],
    horizon: int,
    sigma2: float,
    ar: list[float] | None = None,
    d: int = 0,
    seasonal_ar: list[float] | None = None,
    seasonal_d: int = 0,
    period: int = 1,
    ma: list[float] | None = None,
) -> np.ndarray:
    """h-step forecast variance, from statsmodels rather than from a formula.

    This is the independent check on the psi-weight recursion, and specifically
    on whether the differencing operators are folded into the AR polynomial:
    SARIMAX carries differencing inside the state space, so var_pred_mean is the
    variance of the *original* series, not of the differenced one. A scorer that
    forgets the (1-B)^d factor disagrees from h=2 onward.

    For a pure AR the terminal state is an exact function of the observed data,
    so var_pred_mean is the model variance exactly, and the agreement with the
    psi formula is to machine precision. With q > 0 it is not: statsmodels keeps
    the finite-sample uncertainty in the filtered state, while the psi formula is
    the asymptotic variance. They differ by around 1e-5 relative on the fixtures
    here, so a q > 0 cross-check has to be run at a loosened tolerance and read
    as corroboration rather than as identity.
    """
    result = _filtered(series, ar, d, seasonal_ar, seasonal_d, period, 0.0, sigma2, ma=ma)
    return np.asarray(result.get_forecast(horizon).var_pred_mean, dtype=float)


# ---------------------------------------------------------------------------
# exactLeastSquares
# ---------------------------------------------------------------------------


def arima_exact_forecast(
    series: list[float],
    horizon: int,
    ar: list[float] | None = None,
    ma: list[float] | None = None,
    d: int = 0,
    constant: float = 0.0,
) -> np.ndarray:
    """Forecast by PMML's predictionMethod="exactLeastSquares".

    Unrestricted in q, unlike arima_forecast: this method recovers the
    innovations by filtering rather than reading them out of `<Residuals>`, so
    statsmodels is computing the same estimator rather than a neighbouring one.

    The spec's state space is

        S(t+1) = F S(t) + H e(t),   Y(t) = G S(t) + e(t)

    with m = max(p, q), G = (1,0,...,0), F carrying 1's above the diagonal and
    phi_m..phi_1 along the bottom row, and H the first m psi weights. Because
    G = (1,0,...,0) the state *is* the forecast vector, which is what
    arima_exact_state below exploits.
    """
    result = _filtered(series, ar, d, None, 0, 1, constant, 1.0, ma=ma, stationary_init=True)
    return np.asarray(result.forecast(horizon), dtype=float)


def arima_exact_state(
    series: list[float],
    ar: list[float] | None = None,
    ma: list[float] | None = None,
    constant: float = 0.0,
) -> np.ndarray:
    """The spec's FinalStateVector, taken from the oracle rather than derived.

    S = (Y(n+1|n), ..., Y(n+m|n)) in the zero-mean basis, because the spec's
    G = (1,0,...,0) makes the state and the forecast vector the same object.
    So the m values a producer would write into `<FinalStateVector>` are just
    statsmodels' first m forecasts, less the mean the constantTerm implies.

    That is what makes a fixture for the supplied-state branch possible at all:
    the state can be written down without committing to any implementation's
    internal basis, and without cPMML having produced it.

    Restricted to d = 0. With differencing the state lives in the differenced
    series and would have to be lifted across statsmodels' own handling of the
    integration, which is exactly the sort of unstated convention this file
    exists to avoid.
    """
    mm = max(len(ar or []), len(ma or []), 1)
    mean = _arma_mean(ar, constant)
    return arima_exact_forecast(series, mm, ar=ar, ma=ma, d=0, constant=constant) - mean


def _arma_mean(ar: list[float] | None, constant: float) -> float:
    """mu = c / (1 - sum phi): PMML's constantTerm is an intercept in the mean
    equation, but the spec's state space is zero-mean, so it enters as the mean."""
    if not constant:
        return 0.0
    return constant / (1.0 - sum(ar or []))
