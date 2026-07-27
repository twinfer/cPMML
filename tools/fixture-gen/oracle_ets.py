#!/usr/bin/env python3
"""
statsmodels oracle for PMML ExponentialSmoothing forecasts.

Unlike StateSpaceModel, this one genuinely warrants a third-party library. The
specification does not write the forecast equations down at all -- it defers:

    "The predicted time points are calculated from Gardner, Jr., E.S.,
     'Exponential smoothing: The state of the art - Part II', 3 Jun. 2005
     for the given Trend combined with the Seasonality type."

So there is no spec text to transcribe. statsmodels' Holt-Winters
implementation covers exactly that Gardner/Hyndman taxonomy, which makes it
independent evidence rather than a restatement of cPMML's own arithmetic.

How the states are injected
---------------------------
PMML supplies the *final* smoothed states (Level@smoothedValue,
Trend_ExpoSmooth@smoothedValue, the Seasonality_ExpoSmooth array), whereas
statsmodels fits from data and exposes *initial* states.

The bridge: set every smoothing weight to zero. The state update then carries
no data term and collapses to exactly the forecast recursion, so running the
smoother over h dummy observations makes `fittedvalues` the h-step forecast
path from the supplied states. Zeroing the weights is legitimate rather than a
trick -- the spec says so outright:

    "The optional attribute alpha is the optimal smoothing parameter for the
     level. It can be used to continue the fitting process if more data become
     known, but it is not needed for scoring."

Note this uses statsmodels.tsa.holtwinters and not the newer
tsa.exponential_smoothing.ets: the latter produces NaN for the trend state
when the smoothing weights are zero.

Seasonal phase
--------------
PMML's `phase` is "the season index of the last known data point", defaulting
to `period`. Step h therefore lands on array index (phase + h - 1) mod period.
statsmodels consumes `initial_seasonal` with element 0 used at h=1 (verified
empirically, not assumed), so the array is rotated left by phase mod period
before being handed over.
"""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels
from statsmodels.tsa.holtwinters import ExponentialSmoothing

VERSION = f"statsmodels {statsmodels.__version__}"

_TREND = {
    "additive": ("add", False),
    "damped_additive": ("add", True),
    "multiplicative": ("mul", False),
    "damped_multiplicative": ("mul", True),
}
_SEASONAL = {"additive": "add", "multiplicative": "mul"}


class Unsupported(Exception):
    """The oracle cannot express this model."""


def ets_forecast(
    horizon: int,
    level: float,
    trend: str = "none",
    trend_value: float = 0.0,
    phi: float = 1.0,
    seasonal: str = "none",
    seasonal_values: list[float] | None = None,
    period: int = 1,
    phase: int | None = None,
) -> np.ndarray:
    """Forecast h steps from PMML's final smoothed states."""
    if trend not in _TREND and trend != "none":
        raise Unsupported(f"trend={trend!r} has no statsmodels equivalent")
    if seasonal not in _SEASONAL and seasonal != "none":
        raise Unsupported(f"seasonal={seasonal!r} has no statsmodels equivalent")

    kwargs: dict = {"initial_level": float(level)}
    fit_kwargs: dict = {"smoothing_level": 0.0, "optimized": False}

    if trend != "none":
        kind, damped = _TREND[trend]
        kwargs.update(trend=kind, damped_trend=damped, initial_trend=float(trend_value))
        fit_kwargs["smoothing_trend"] = 0.0
        if damped:
            fit_kwargs["damping_trend"] = float(phi)

    if seasonal != "none":
        if not seasonal_values or period != len(seasonal_values):
            raise Unsupported(f"seasonal array of length {len(seasonal_values or [])} does not match period {period}")
        rotation = (period if phase is None else phase) % period
        rotated = list(seasonal_values[rotation:]) + list(seasonal_values[:rotation])
        kwargs.update(
            seasonal=_SEASONAL[seasonal],
            seasonal_periods=period,
            initial_seasonal=np.asarray(rotated, dtype=float),
        )
        fit_kwargs["smoothing_seasonal"] = 0.0

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The dummy observations never reach the states (every smoothing weight
        # is zero) but must be strictly positive for multiplicative components.
        model = ExponentialSmoothing(np.ones(horizon + 1), initialization_method="known", **kwargs)
        fitted = np.asarray(model.fit(**fit_kwargs).fittedvalues, dtype=float)

    return fitted[:horizon]
