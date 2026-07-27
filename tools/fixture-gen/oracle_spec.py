#!/usr/bin/env python3
"""
Transcription oracle: PMML forecast formulas written straight from the spec.

Not every model type has a third-party implementation worth borrowing. Where
PMML hands over a *fitted* model and defines the forecast as a closed form over
those parameters, there is nothing for a library to independently discover --
routing the arithmetic through statsmodels would mean inventing the parameters
statsmodels needs (disturbance covariances, initialisation) that PMML never
states, and then testing the invention.

For those cases this module is the oracle: the spec's own equations, in numpy,
written from the specification text rather than from cPMML's source. That is
weaker evidence than an independent implementation and this module says so --
but it is stronger than expectations derived from the evaluator under test, and
it is honest about which of the two it is.

Use oracle_statsmodels.py instead wherever a real recursion has to be reproduced
(ARIMA differencing, MA residual handling), because there a third-party
implementation genuinely constrains the answer.

--------------------------------------------------------------------------
StateSpaceModel  (http://dmg.org/pmml/v4-4/TimeSeriesModel.html)

The spec gives three parameters -- state vector X, transition matrix F,
measurement matrix G -- and states:

    "For the purposes of prediction, the errors are ignored."
    Y_t = G (F X_t),  Y_{t+1} = G (F (F X_t))

so the h-step forecast is deterministic:

    Y_{t+h} = G F^h X

There is no disturbance term, no Q, no H and no Kalman filter in the
prediction path. The `intercept` attribute is added to the result.

PsiVector supplies the forecast variance instead, as an infinite-order MA
representation:

    sigma^2_{t+h} = sigma^2_t (1 + psi_1^2 + ... + psi_{h-1}^2)
"""

from __future__ import annotations

import numpy as np

VERSION = "PMML 4.4 spec transcription"


def ssm_forecast(
    state: np.ndarray,
    transition: np.ndarray,
    measurement: np.ndarray,
    horizon: int,
    intercept: float = 0.0,
) -> np.ndarray:
    """Y_{t+h} = G F^h X + intercept, for h = 1..horizon.

    Returns one value per step for a univariate model (G has a single row);
    for k>1 observed series the result is (horizon, k).
    """
    x = np.asarray(state, dtype=float).reshape(-1)
    f = np.asarray(transition, dtype=float)
    g = np.asarray(measurement, dtype=float)

    if f.shape[0] != f.shape[1]:
        raise ValueError(f"transition matrix must be square, got {f.shape}")
    if f.shape[0] != x.size:
        raise ValueError(f"transition matrix {f.shape} does not match state of length {x.size}")
    if g.shape[1] != x.size:
        raise ValueError(f"measurement matrix {g.shape} does not match state of length {x.size}")

    out = []
    for _ in range(horizon):
        x = f @ x  # advance first, then measure: Y_t = G(F X_t)
        out.append(g @ x + intercept)

    result = np.asarray(out, dtype=float)
    return result[:, 0] if result.shape[1] == 1 else result


def _difference_polynomial(order: int, period: int = 1) -> np.ndarray:
    """(1 - B^period)^order as coefficients of B^0, B^1, ..."""
    poly = np.array([1.0])
    factor = np.zeros(period + 1)
    factor[0], factor[period] = 1.0, -1.0
    for _ in range(order):
        poly = np.convolve(poly, factor)
    return poly


def _lag_polynomial(coefficients: list[float], period: int = 1) -> np.ndarray:
    """1 - c_1 B^period - c_2 B^(2 period) - ..."""
    poly = np.zeros(len(coefficients) * period + 1)
    poly[0] = 1.0
    for i, c in enumerate(coefficients, start=1):
        poly[i * period] = -c
    return poly


def arima_cls_forecast(
    series: list[float],
    horizon: int,
    ar: list[float] | None = None,
    ma: list[float] | None = None,
    residuals: list[float] | None = None,
    d: int = 0,
    seasonal_ar: list[float] | None = None,
    seasonal_ma: list[float] | None = None,
    seasonal_d: int = 0,
    period: int = 1,
    constant: float = 0.0,
) -> np.ndarray:
    """ARIMA forecast by the conditionalLeastSquares method.

    Transcribed from the spec's own derivation. For an ARIMA(3,1,1) it writes

        (1 - f1 B - f2 B^2 - f3 B^3)(1-B) y_t = (1 - t1 B) a_t

    and reads the prediction off the expanded left-hand side, so the general
    form multiplies the AR, seasonal AR and both differencing polynomials
    together and solves for y_t:

        PHI(B) y_t = THETA(B) a_t
        y_t = -sum_{i>=1} PHI_i y_{t-i} + a_t + sum_{j>=1} THETA_j a_{t-j}

    where PHI and THETA are the *expanded* polynomials, so THETA_1 = -theta_1.
    The spec's example writes the last term as "- theta_1 a_t", which is the
    same thing.

    Note the MINUS on the MA terms: PMML writes the MA polynomial as
    (1 - theta B), the Box-Jenkins convention, where statsmodels uses
    (1 + theta B). A theta of -0.4 in PMML is +0.4 in statsmodels.

    Array ordering follows the spec: "The coefficient array gives the values
    of the coefficients starting at time t, t-1 and so on" -- coefficients and
    residuals are MOST RECENT FIRST, while TimeSeries/TimeValue runs oldest to
    newest by ascending index.

    Per the spec, residuals after time t are zero and series values after t are
    the predictions already computed.

    `constant` is PMML's constantTerm, "the constant value mu in ARIMA part".
    The spec adds that it "should be added to the predicted value", which reads
    as a post-hoc shift, but mu belongs *inside* the equation -- the prediction
    it is added to then feeds the next step. statsmodels' trend='c' agrees, and
    the two readings diverge from h=2 onward, so this is not academic.
    """
    phi = np.convolve(
        np.convolve(_lag_polynomial(ar or []), _difference_polynomial(d)),
        np.convolve(_lag_polynomial(seasonal_ar or [], period), _difference_polynomial(seasonal_d, period)),
    )
    theta = np.convolve(_lag_polynomial(ma or []), _lag_polynomial(seasonal_ma or [], period))

    history = list(reversed([float(v) for v in series]))  # most recent first
    errors = list(residuals or [])  # already most recent first

    out = []
    for _ in range(horizon):
        value = constant
        value -= sum(phi[i] * history[i - 1] for i in range(1, len(phi)) if i - 1 < len(history))
        value += sum(theta[j] * errors[j - 1] for j in range(1, len(theta)) if j - 1 < len(errors))

        out.append(value)
        history.insert(0, value)  # the prediction becomes history for the next step
        errors.insert(0, 0.0)  # residuals after time t are zero

    return np.asarray(out, dtype=float)


def garch_mean_forecast(
    series: list[float],
    horizon: int,
    ar: list[float] | None = None,
    ma: list[float] | None = None,
    residuals: list[float] | None = None,
    constant: float = 0.0,
) -> np.ndarray:
    """Conditional mean of a PMML GARCH model (the ARMAPart).

    The GARCH section writes the mean equation out in full, and with the
    OPPOSITE MA sign to the ARIMA section:

        Y_t = mu + sum_{i=1..p} phi_i Y_{t-i} + sum_{j=1..q} theta_j eps_{t-j} + eps_t

    ARIMA writes its MA polynomial as (1 - theta B), so its residual terms are
    subtracted; GARCH writes "+ sum theta_j eps_{t-j}", so they are added. That
    divergence is in the specification itself, not a transcription slip, and it
    is why the two forecast paths in cPMML must not be unified.

    Only the conditional mean is computed. PMML's variance recursion
    (h_t = alpha_0 + sum alpha_i eps^2_{t-i} + sum beta_j h_{t-j}) has nowhere
    to go: cPMML does not compute it and forecast mode compares point
    forecasts only.
    """
    phi = list(ar or [])
    theta = list(ma or [])
    history = list(reversed([float(v) for v in series]))  # most recent first
    errors = list(residuals or [])

    out = []
    for _ in range(horizon):
        value = constant
        value += sum(phi[i] * history[i] for i in range(len(phi)) if i < len(history))
        value += sum(theta[j] * errors[j] for j in range(len(theta)) if j < len(errors))

        out.append(value)
        history.insert(0, value)
        errors.insert(0, 0.0)

    return np.asarray(out, dtype=float)


def garch_variance(
    horizon: int,
    omega: float,
    alpha: list[float] | None = None,
    residuals: list[float] | None = None,
    beta: list[float] | None = None,
    past_variances: list[float] | None = None,
) -> np.ndarray:
    """Conditional variance of a PMML GARCH model.

    The spec writes this one out in full, unlike the ETS forecasts:

        h_t = a_0 + sum_{i=1..gq} alpha_i eps^2_{t-i} + sum_{j=1..gp} beta_j h_{t-j}

    Note which order goes with which coefficient: "gp and gq are order of
    variance and order of residual squares, respectively", so the alpha terms
    run to gq and the beta terms to gp. The two are easy to transpose and a
    model with gp == gq will never notice.

    Beyond the known history, E[eps^2] is the forecast variance itself, so both
    sums fall back to values already computed.

    `residuals` is the <Residuals> array verbatim: it holds the residual errors
    eps, which the formula squares. The element sits inside
    ResidualSquareCoefficients, which makes it easy to read as already-squared,
    but Residuals carries the same meaning here as in ARIMA's MA -- "the past
    differences between the actual time series values and the model
    predictions".
    """
    alpha = list(alpha or [])
    beta = list(beta or [])
    eps_sq = [e * e for e in (residuals or [])]  # most recent first
    past = list(past_variances or [])  # most recent first

    out: list[float] = []
    for h in range(1, horizon + 1):
        value = omega
        for i, a in enumerate(alpha, start=1):
            known = eps_sq[i - h] if i >= h and (i - h) < len(eps_sq) else None
            value += a * (known if known is not None else (out[h - i - 1] if i < h else 0.0))
        for j, b in enumerate(beta, start=1):
            known = past[j - h] if j >= h and (j - h) < len(past) else None
            value += b * (known if known is not None else (out[h - j - 1] if j < h else omega))
        out.append(value)

    return np.asarray(out, dtype=float)


def arima_psi_variance(
    horizon: int,
    sigma2: float,
    ar: list[float] | None = None,
    ma: list[float] | None = None,
) -> np.ndarray:
    """ARIMA h-step forecast variance from derived MA(inf) weights.

        var(h) = sigma^2 * sum_{j=0}^{h-1} psi_j^2,  psi_0 = 1
        psi_j  = sum_i phi_i psi_{j-i} - theta_j

    theta is subtracted for the same reason the forecast subtracts its MA
    terms: PMML's MA polynomial is (1 - theta B).
    """
    phi = list(ar or [])
    theta = list(ma or [])

    psi = [1.0]
    for j in range(1, horizon):
        value = sum(phi[i] * psi[j - 1 - i] for i in range(len(phi)) if i < j)
        if j <= len(theta):
            value -= theta[j - 1]
        psi.append(value)

    weights = np.asarray(psi, dtype=float)
    return np.asarray([sigma2 * np.sum(weights[:h] ** 2) for h in range(1, horizon + 1)], dtype=float)


def psi_variance(sigma2: float, psi: list[float], horizon: int) -> np.ndarray:
    """sigma^2_{t+h} = sigma^2 (1 + psi_1^2 + ... + psi_{h-1}^2), h = 1..horizon.

    psi is the PsiVector array as written in the PMML, i.e. psi_1 onward; the
    leading 1 in the formula is the implicit psi_0.
    """
    weights = np.concatenate(([1.0], np.asarray(psi, dtype=float)))
    return np.asarray([sigma2 * np.sum(weights[:h] ** 2) for h in range(1, horizon + 1)], dtype=float)
