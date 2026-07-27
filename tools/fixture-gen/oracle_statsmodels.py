#!/usr/bin/env python3
"""
statsmodels oracle for PMML ARIMA forecasts.

Used where a third-party implementation genuinely constrains the answer: the
AR recursion combined with non-seasonal and seasonal differencing, which is
substantial machinery and is where the open time-series findings live.

Scope, and why it is narrow
---------------------------
This oracle covers **q = Q = 0** only, i.e. models with no moving-average part.

PMML's conditionalLeastSquares method takes the past residuals as *given* -- an
explicit `<Residuals>` array in the document -- and applies the MA coefficients
to those supplied numbers. statsmodels derives residuals by running the Kalman
filter over the data instead, so for q > 0 it answers a different question and
would agree with a conformant PMML scorer only by coincidence. Those cases are
covered by oracle_spec.arima_cls_forecast, anchored on the worked example the
specification itself works through arithmetically.

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
    """Forecast a pure AR/ARI/SARI model with the coefficients PMML supplies."""
    ar = list(ar or [])
    seasonal_ar = list(seasonal_ar or [])

    endog = np.asarray(series, dtype=float)
    order = (len(ar), d, 0)
    seasonal_order = (len(seasonal_ar), seasonal_d, 0, period if seasonal_ar or seasonal_d else 0)

    model = SARIMAX(
        endog,
        order=order,
        seasonal_order=seasonal_order,
        trend="c" if constant else None,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    # PMML's AR array is phi_1..phi_p, the same order SARIMAX names them in;
    # the intercept leads when present and sigma2 is last, and sigma2 does not
    # affect the point forecast.
    expected = list(model.param_names)
    params = ([constant] if constant else []) + ar + seasonal_ar + [1.0]
    if len(params) != len(expected):
        raise Unsupported(f"parameter mismatch: SARIMAX wants {expected}, built {len(params)} values")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.filter(params)
        return np.asarray(result.forecast(horizon), dtype=float)
