
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TIMESERIESMODEL_H
#define CPMML_TIMESERIESMODEL_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>
#include <vector>

#include "cPMML.h"
#include "core/xmlnode.h"
#include "utils/utils.h"

/**
 * @class TimeSeriesModel
 *
 * Standalone (non-InternalModel) implementation of
 * <a href="http://dmg.org/pmml/v4-4/TimeSeriesModel.html">PMML TimeSeriesModel</a>.
 *
 * Supported algorithms (bestFit + predictionMethod attributes):
 *   ExponentialSmoothing  — ETS models (all 15 trend × seasonal combinations)
 *   StateSpaceModel       — explicit F/G/x matrices; ŷ(T+h) = G · Fʰ · x + intercept
 *   ARIMA conditionalLS   — scalar AR/MA recursion (nonseasonal + seasonal expansion)
 *   ARIMA exactLS         — Kalman recursion on the spec's state space
 *                           S(t+1) = F·S(t) + H·e(t), Y(t) = G·S(t) + e(t);
 *                           Ω₁ solved via Kronecker–Lyapunov equation
 *
 * Extra PMML features supported:
 *   - transformation="logarithmic|squareroot" on ETS/SSM/ARIMA
 *   - StateSpaceModel/@intercept
 *   - ARIMA/MaximumLikelihoodStat/KalmanState — FinalStateVector and HVector
 *     are honoured as a scoring shortcut; FinalOmega is not needed (see
 *     parse_arima). method="thetaRecursion" falls back to filtering.
 *   - SeasonalComponent (P, D, Q, period) — polynomial convolution with nonseasonal
 *   - OutlierEffect (additive, levelShift, transientChange)
 *
 * Call forecast(h) for point forecasts, forecast_with_variance(h) for
 * {point, variance} pairs.
 */
class TimeSeriesModel {
 public:
  enum class Algorithm { ETS, SSM, ARIMA_CLS, ARIMA_KALMAN, GARCH };
  enum class Transform { NONE, LOG, SQRT };

  Algorithm algorithm = Algorithm::ETS;

  // -------------------------------------------------------------------------
  // ETS state
  // -------------------------------------------------------------------------
  struct ETSState {
    std::string trend = "none";     // none|additive|damped_additive|multiplicative|damped_multiplicative
    std::string seasonal = "none";  // none|additive|multiplicative
    double alpha = 0.0, gamma = 0.0, delta = 0.0, phi = 1.0;
    double l = 0.0, b = 0.0;
    int period = 1;
    // Season index of the last known data point, 1-based. The spec defaults it
    // to period, which is why omitting it lines the array up with h=1.
    int phase = 0;
    std::vector<double> s;  // seasonal components, one per season index
    Transform transform = Transform::NONE;
    // Parsed because the spec defines it, but nothing consumes it: the forecast
    // does not need it and there is no ETS variance to scale — see variance_ets.
    double rmse = 1.0;
  } ets;

  // -------------------------------------------------------------------------
  // SSM state
  // -------------------------------------------------------------------------
  struct SSMState {
    Eigen::VectorXd x;       // state  (n×1)
    Eigen::MatrixXd F;       // transition (n×n)
    Eigen::MatrixXd G;       // measurement (1×n)
    double intercept = 0.0;  // additive offset in measurement equation
    Transform transform = Transform::NONE;
    double sigma2 = 1.0;      // innovation variance (from PsiVector/@variance or RMSE²)
    std::vector<double> psi;  // MA(∞) weights ψ₀,ψ₁,... from PsiVector/Array
  } ssm;

  // -------------------------------------------------------------------------
  // ARIMA conditional LS state
  // -------------------------------------------------------------------------
  struct ARIMAState {
    int p = 0, d = 0, q = 0, D = 0, m = 1;
    double constant = 0.0;
    bool include_constant = false;
    std::vector<double> phi, theta;
    std::vector<double> y_diff;  // last max(p_expanded,1) w-series values (most-recent first)
    std::vector<double> eps;     // last q_expanded residuals (most-recent first)
    std::vector<double> z_tail;  // last d values of seasonally-differenced series (ascending)
    std::vector<double> y_tail;  // last max(m*D,1) raw values (ascending) for seasonal undo
    Transform transform = Transform::NONE;
    double sigma2 = 1.0;
  } arima;

  // -------------------------------------------------------------------------
  // ARIMA Kalman (exact LS) state
  // -------------------------------------------------------------------------
  struct KalmanState {
    int d = 0, D = 0, m = 1;
    std::vector<double> z_tail;  // last d values of seasonally-differenced series
    std::vector<double> y_tail;  // last max(m*D,1) raw values for seasonal undo
    // The spec's state space, verbatim:  S(t+1) = F·S(t) + H·e(t),  Y(t) = G·S(t) + e(t).
    // Note that PMML's H is the ψ-weight *noise* vector, not the observation
    // row — the observation row is G. Naming them the other way round is what
    // hid M-S13 for as long as it did.
    //
    // H is not kept: it drives the filter that *derives* the state, and once
    // the state exists the forecast is G·F^(h−1)·S with no H in it.
    Eigen::MatrixXd F;     // transition (mm×mm)
    Eigen::RowVectorXd G;  // observation (1×mm), = (1,0,...,0)
    Eigen::VectorXd x_T;   // Ŝ(n+1): the one-step-ahead predictor state given all data
    double mean = 0.0;      // μ of the differenced series; the state space is zero-mean
    // Expanded ARMA coefficients, kept for the forecast variance. That variance
    // is a property of the model rather than of the filter, so both prediction
    // methods compute it the same way — see arima_variance().
    std::vector<double> phi, theta;
    Transform transform = Transform::NONE;
    double sigma2 = 1.0;
  } kalman;

  // -------------------------------------------------------------------------
  // GARCH state (bestFit="GARCH")
  // -------------------------------------------------------------------------
  struct GARCHState {
    // ARMAPart — conditional mean (ARMA recursion)
    int p_mean = 0, q_mean = 0;
    double constant_mean = 0.0;
    std::vector<double> phi, theta;
    std::vector<double> y_diff;    // past mean values (most-recent first)
    std::vector<double> eps_mean;  // past MA residuals (most-recent first)
    // GARCHPart — conditional variance
    int gp = 0, gq = 0;
    double omega = 0.0;               // intercept ω in variance equation
    // Per the spec, gq is the "order of residual squares" and gp the "order of
    // variance": h_t = a_0 + sum_{i=1..gq} alpha_i eps^2_{t-i}
    //                      + sum_{j=1..gp} beta_j h_{t-j}
    std::vector<double> alpha;        // ARCH coefficients α₁,..,α_{gq}
    std::vector<double> beta_garch;   // GARCH coefficients β₁,..,β_{gp}
    std::vector<double> eps_sq;       // past ε² (most-recent first)
    std::vector<double> sigma2_hist;  // past σ² (most-recent first)
  } garch;

  // -------------------------------------------------------------------------
  // DynamicRegressor state (appended to any algorithm's forecast)
  // -------------------------------------------------------------------------
  struct DynamicRegressorState {
    std::string field;
    int delay = 0;
    Transform transform = Transform::NONE;
    bool user_supplied = false;       // true if futureValuesMethod="userSupplied"
    std::vector<double> numerator;    // N(B) coefficients β₀,β₁,...
    std::vector<double> denominator;  // D(B) coefficients δ₁,δ₂,... (without leading 1)
    std::vector<double> x_history;    // past x(T),x(T-1),... (most-recent first, transformed)
    std::vector<double> v_history;    // past filter output v(T),v(T-1),... (most-recent first)
  };
  std::vector<DynamicRegressorState> dynamic_regressors;

  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  TimeSeriesModel() = default;

  explicit TimeSeriesModel(const XmlNode& ts_node) {
    std::string bf = to_lower(ts_node.get_attribute("bestFit"));
    if (bf == "exponentialsmoothing") {
      algorithm = Algorithm::ETS;
      parse_ets(ts_node.get_child("ExponentialSmoothing"));
    } else if (bf == "statespacemodel") {
      algorithm = Algorithm::SSM;
      parse_ssm(ts_node.get_child("StateSpaceModel"));
    } else if (bf == "arima") {
      parse_arima(ts_node.get_child("ARIMA"), ts_node);  // sets algorithm internally
    } else if (bf == "garch") {
      algorithm = Algorithm::GARCH;
      parse_garch(ts_node.get_child("GARCH"), ts_node);
    } else {
      throw cpmml::ParsingException("TimeSeriesModel: unsupported bestFit=\"" + ts_node.get_attribute("bestFit") +
                                    "\"; supported: ExponentialSmoothing, StateSpaceModel, ARIMA, GARCH");
    }
    parse_dynamic_regressors(ts_node);
  }

  // -------------------------------------------------------------------------
  // Forecast API
  // -------------------------------------------------------------------------

  std::vector<double> forecast(int horizon,
                               const std::unordered_map<std::string, std::vector<double>>& regressors = {}) const {
    if (horizon <= 0) return {};
    std::vector<double> base;
    switch (algorithm) {
      case Algorithm::ETS:
        base = forecast_ets(horizon);
        break;
      case Algorithm::SSM:
        base = forecast_ssm(horizon);
        break;
      case Algorithm::ARIMA_CLS:
        base = forecast_arima_cls(horizon);
        break;
      case Algorithm::ARIMA_KALMAN:
        base = forecast_arima_kalman(horizon);
        break;
      case Algorithm::GARCH:
        base = forecast_garch_mean(horizon);
        break;
    }
    apply_dynamic_regressors(base, horizon, regressors);
    return base;
  }

  std::vector<std::pair<double, double>> forecast_with_variance(
      int horizon, const std::unordered_map<std::string, std::vector<double>>& regressors = {}) const {
    if (horizon <= 0) return {};
    std::vector<double> pts = forecast(horizon, regressors);
    std::vector<double> vars;
    switch (algorithm) {
      case Algorithm::ETS:
        vars = variance_ets(horizon);
        break;
      case Algorithm::SSM:
        vars = variance_ssm(horizon);
        break;
      case Algorithm::ARIMA_CLS:
        vars = arima_variance(arima.phi, arima.theta, arima.d, arima.D, arima.m, arima.sigma2, horizon);
        break;
      case Algorithm::ARIMA_KALMAN:
        vars = arima_variance(kalman.phi, kalman.theta, kalman.d, kalman.D, kalman.m, kalman.sigma2, horizon);
        break;
      case Algorithm::GARCH:
        vars = variance_garch(horizon);
        break;
    }
    std::vector<std::pair<double, double>> out;
    out.reserve(horizon);
    for (int h = 0; h < horizon; ++h) out.push_back({pts[h], h < static_cast<int>(vars.size()) ? vars[h] : 0.0});
    return out;
  }

 private:
  // =========================================================================
  // Transform helpers
  // =========================================================================

  static Transform parse_transform(const std::string& attr) {
    if (attr == "logarithmic") return Transform::LOG;
    if (attr == "squareroot") return Transform::SQRT;
    return Transform::NONE;
  }

  static double apply_transform(double v, Transform t) {
    if (t == Transform::LOG) return std::log(v);
    if (t == Transform::SQRT) return std::sqrt(v);
    return v;
  }

  static double invert_transform(double v, Transform t) {
    if (t == Transform::LOG) return std::exp(v);
    if (t == Transform::SQRT) return v * v;
    return v;
  }

  static std::vector<double> invert_transform_vec(std::vector<double> v, Transform t) {
    if (t != Transform::NONE)
      for (auto& x : v) x = invert_transform(x, t);
    return v;
  }

  // =========================================================================
  // ETS forecasting
  // =========================================================================

  std::vector<double> forecast_ets(int horizon) const {
    const auto& e = ets;
    int m = e.period > 0 ? e.period : 1;
    bool has_seas = e.seasonal != "none" && !e.s.empty();
    bool mult_trend = (e.trend == "multiplicative" || e.trend == "damped_multiplicative");

    std::vector<double> out;
    out.reserve(horizon);

    for (int h = 1; h <= horizon; ++h) {
      double trend_add = 0.0, trend_mult = 1.0;
      if (e.trend == "additive") {
        trend_add = static_cast<double>(h) * e.b;
      } else if (e.trend == "damped_additive") {
        double ps = 0.0, pp = e.phi;
        for (int i = 0; i < h; ++i) {
          ps += pp;
          pp *= e.phi;
        }
        trend_add = ps * e.b;
      } else if (e.trend == "multiplicative") {
        trend_mult = std::pow(e.b, static_cast<double>(h));
      } else if (e.trend == "damped_multiplicative") {
        double ps = 0.0, pp = e.phi;
        for (int i = 0; i < h; ++i) {
          ps += pp;
          pp *= e.phi;
        }
        trend_mult = std::pow(e.b, ps);
      }

      double base = mult_trend ? (e.l * trend_mult) : (e.l + trend_add);

      // phase is the season index of the last known point, so step h lands on
      // index (phase + h - 1) mod period. The default phase == period makes
      // that (h - 1) mod period, which is what this used to assume outright.
      const int index = ((e.phase + h - 1) % m + m) % m;
      double seas = has_seas ? e.s[static_cast<size_t>(index)] : (e.seasonal == "multiplicative" ? 1.0 : 0.0);

      double y_hat;
      if (!has_seas || e.seasonal == "none")
        y_hat = base;
      else if (e.seasonal == "additive")
        y_hat = base + seas;
      else
        y_hat = base * seas;

      out.push_back(invert_transform(y_hat, e.transform));
    }
    return out;
  }

  /**
   * NaN: an ExponentialSmoothing document does not determine a forecast variance.
   *
   * Two independent reasons, both from TimeSeriesModel.html:
   *
   *  1. No formula. The spec defines no prediction interval, standard error or
   *     variance for ExponentialSmoothing anywhere. It defers the point
   *     forecasts to Gardner (2005) and does not reproduce his formulas, and it
   *     never takes up the variance at all.
   *
   *  2. No parameterisation to reconstruct one from. Gardner's variance is a
   *     function of the smoothing weights, but the spec writes no update
   *     equations, so `alpha`/`gamma`/`delta` are not tied to a particular form
   *     of the recursion — consistently with the spec calling them optional and
   *     "not needed for scoring". The common forms differ: the seasonal weight
   *     enters the error-form recursion as δ under one and δ(1−α) under
   *     another, and the trend weight as γ or αγ, which change the answer from
   *     h > m and from h = 2 respectively. Nothing in the document chooses.
   *
   * This used to return σ²·h. That is not a bound in either direction, so it
   * could not even be defended as conservative: with an additive trend the true
   * variance grows like σ²(αγ)²h³/3 and σ²·h understates it badly, while for a
   * level-only model with α < 1 it overstates. NaN says what is actually known.
   *
   * Every other algorithm here does report a variance, because for those the
   * spec either writes the recursion out (GARCH), supplies the weights
   * (PsiVector) or determines them from stated coefficients (ARIMA).
   */
  std::vector<double> variance_ets(int horizon) const {
    return std::vector<double>(horizon, std::numeric_limits<double>::quiet_NaN());
  }

  // =========================================================================
  // SSM forecasting: ŷ(T+h) = (G · Fʰ · x)(0) + intercept
  // =========================================================================

  std::vector<double> forecast_ssm(int horizon) const {
    std::vector<double> out;
    out.reserve(horizon);
    Eigen::VectorXd state = ssm.x;
    for (int h = 1; h <= horizon; ++h) {
      state = ssm.F * state;
      double y_hat = (ssm.G * state)(0, 0) + ssm.intercept;
      out.push_back(invert_transform(y_hat, ssm.transform));
    }
    return out;
  }

  std::vector<double> variance_ssm(int horizon) const {
    // Spec: sigma^2(t+h) = sigma^2 (1 + psi_1^2 + ... + psi_{h-1}^2).
    //
    // The leading 1 is psi_0 and is implicit: the PsiVector array holds psi_1
    // onward, as the spec's own example shows (three weights for a model whose
    // psi_0 is never written). Weights beyond the stored horizon are treated as
    // 0. With no PsiVector, fall back to a constant sigma^2 per step.
    if (!ssm.psi.empty()) {
      std::vector<double> vars;
      vars.reserve(horizon);
      double cumsum = 1.0;  // psi_0 = 1
      for (int h = 1; h <= horizon; ++h) {
        vars.push_back(ssm.sigma2 * cumsum);
        if (h - 1 < static_cast<int>(ssm.psi.size())) cumsum += ssm.psi[h - 1] * ssm.psi[h - 1];
      }
      return vars;
    }
    return std::vector<double>(horizon, ssm.sigma2);
  }

  // =========================================================================
  // ARIMA conditional LS forecasting
  // =========================================================================

  std::vector<double> forecast_arima_cls(int horizon) const {
    const auto& a = arima;
    std::vector<double> y(a.y_diff), eps(a.eps);
    std::vector<double> w_out;
    w_out.reserve(horizon);

    int p_eff = static_cast<int>(a.phi.size());
    int q_eff = static_cast<int>(a.theta.size());

    for (int h = 1; h <= horizon; ++h) {
      double yh = a.include_constant ? a.constant : 0.0;
      for (int i = 0; i < p_eff && i < static_cast<int>(y.size()); ++i) yh += a.phi[i] * y[i];
      // PMML writes the MA polynomial as (1 - theta_1 B - theta_2 B^2 - ...),
      // so the residual terms are subtracted. The spec's own worked example
      // spells this out: "- theta_1 a_t" with theta_1 = -0.4 contributes +0.8.
      for (int j = 0; j < q_eff && j < static_cast<int>(eps.size()); ++j) yh -= a.theta[j] * eps[j];
      w_out.push_back(yh);
      y.insert(y.begin(), yh);
      eps.insert(eps.begin(), 0.0);
    }

    auto result = undifference_sarima(w_out, a.d, a.D, a.m, a.z_tail, a.y_tail);
    return invert_transform_vec(std::move(result), a.transform);
  }

  /**
   * Φ(B) = φ(B)·(1−B)^d·(1−B^m)^D, in PMML's sign convention.
   *
   * Input and output are both the coefficients c_i of (1 − c₁B − c₂B² − …),
   * which is how AR/Array is written and how expand_arma_poly returns the
   * seasonal × nonseasonal product.
   */
  static std::vector<double> integrate_ar_poly(const std::vector<double>& phi, int d, int D, int m) {
    std::vector<double> poly{1.0};
    for (double c : phi) poly.push_back(-c);

    auto multiply = [](const std::vector<double>& a, const std::vector<double>& b) {
      std::vector<double> out(a.size() + b.size() - 1, 0.0);
      for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) out[i + j] += a[i] * b[j];
      return out;
    };

    for (int i = 0; i < d; ++i) poly = multiply(poly, {1.0, -1.0});
    if (D > 0 && m > 0) {
      std::vector<double> seasonal(m + 1, 0.0);
      seasonal[0] = 1.0;
      seasonal[m] = -1.0;
      for (int i = 0; i < D; ++i) poly = multiply(poly, seasonal);
    }

    std::vector<double> expanded;
    expanded.reserve(poly.size() - 1);
    for (size_t i = 1; i < poly.size(); ++i) expanded.push_back(-poly[i]);
    return expanded;
  }

  /**
   * ARIMA h-step forecast variance, shared by both prediction methods.
   *
   *   var(h) = σ² · Σ_{j=0}^{h-1} ψ_j²,   ψ₀ = 1
   *   ψ_j    = Σ_i Φ_i ψ_{j-i} − θ_j
   *
   * `predictionMethod` selects how the point forecast is *computed* — filter
   * recursion versus expanded-polynomial recursion — not a different stochastic
   * process, so this is the variance for exactLeastSquares too. It depends only
   * on (φ, θ, σ²), every one of which the document states; the terminal filter
   * covariance is an artefact of the estimation path and does not enter.
   *
   * Φ is the AR operator of the *undifferenced* series, so the differencing
   * operators fold in. Without them the result would describe the differenced
   * series w rather than y.
   *
   * θ is subtracted for the same reason the forecast subtracts its MA terms:
   * PMML writes the MA polynomial as (1 − θ₁B − θ₂B² − …).
   */
  static std::vector<double> arima_variance(const std::vector<double>& phi, const std::vector<double>& theta, int d,
                                            int D, int m, double sigma2, int horizon) {
    if (horizon <= 0) return {};
    const std::vector<double> full = integrate_ar_poly(phi, d, D, m);
    const int p_eff = static_cast<int>(full.size());
    const int q_eff = static_cast<int>(theta.size());

    std::vector<double> psi(horizon, 0.0);
    psi[0] = 1.0;
    for (int j = 1; j < horizon; ++j) {
      double v = 0.0;
      for (int i = 0; i < p_eff && i < j; ++i) v += full[i] * psi[j - 1 - i];
      if (j <= q_eff) v -= theta[j - 1];
      psi[j] = v;
    }

    std::vector<double> vars;
    vars.reserve(horizon);
    double cumsum = 0.0;
    for (int h = 0; h < horizon; ++h) {
      cumsum += psi[h] * psi[h];
      vars.push_back(sigma2 * cumsum);
    }
    return vars;
  }

  // =========================================================================
  // ARIMA exact-least-squares forecasting: ŵ(n+h) = G · F^(h−1) · Ŝ(n+1) + μ
  //
  // Because G = (1,0,...,0), the state *is* the forecast vector: Ŝ(n+1) holds
  // ŵ(n+1) … ŵ(n+mm) and F shifts it along. So the first value comes straight
  // off the stored state — advancing before the first read would forecast from
  // Ŝ(n) and silently throw away the last observation.
  // =========================================================================

  std::vector<double> forecast_arima_kalman(int horizon) const {
    std::vector<double> w_out;
    w_out.reserve(horizon);

    Eigen::VectorXd state = kalman.x_T;
    for (int h = 1; h <= horizon; ++h) {
      w_out.push_back((kalman.G * state)(0) + kalman.mean);
      state = kalman.F * state;
    }

    auto result = undifference_sarima(w_out, kalman.d, kalman.D, kalman.m, kalman.z_tail, kalman.y_tail);
    return invert_transform_vec(std::move(result), kalman.transform);
  }

  /**
   * The spec's state space for predictionMethod="exactLeastSquares".
   *
   *   S(t+1) = F·S(t) + H·e(t)
   *   Y(t)   = G·S(t) + e(t)
   *
   * with mm = max(p, q); F an mm×mm matrix "with 1's above the main diagonal,
   * the bottom row containing φm, φm-1, ..., φ1, and zeros elsewhere", where
   * "φi = 0 if i > p"; G = (1,0,...,0); and H the first mm ψ weights of
   * (1 − θ₁B − …)/(1 − φ₁B − …) = 1 + ψ₁B + ψ₂B² + ….
   *
   * The three pieces are consistent with each other in a way worth stating,
   * because it is what makes the rest of this file simple: with G = (1,0,…,0)
   * the state is the vector of forecasts, S(t) = (Ŷ(t|t−1), …, Ŷ(t+mm−1|t−1)).
   * F shifts it along and extends it by one AR step; H·e(t) is the revision
   * Ŷ(t+j|t) = Ŷ(t+j|t−1) + ψ_j·e(t) that arrives with the new observation.
   * That is also why the spec can offer HVector as a scoring-time shortcut —
   * the ψ weights are the only model-derived quantity scoring needs.
   */
  /**
   * μ of the differenced series implied by ARIMA/@constantTerm.
   *
   * The conditional-LS recursion carries c as an intercept, w(t) = c + Σφ·w + …,
   * whose stationary mean is c / (1 − Σφ). The exact-LS state space is
   * zero-mean, so the same document has to enter it as that mean instead.
   */
  static double arma_mean(const std::vector<double>& phi, double constant) {
    if (constant == 0.0) return 0.0;
    double denom = 1.0;
    for (double c : phi) denom -= c;
    if (std::abs(denom) < 1e-12)
      throw cpmml::ParsingException("TimeSeriesModel: ARIMA constantTerm with a unit-root AR polynomial has no mean");
    return constant / denom;
  }

  static void build_state_space(const std::vector<double>& phi, const std::vector<double>& theta, Eigen::MatrixXd& F,
                                Eigen::RowVectorXd& G, Eigen::VectorXd& Hvec) {
    const int p = static_cast<int>(phi.size());
    const int q = static_cast<int>(theta.size());
    const int mm = std::max({p, q, 1});

    F = Eigen::MatrixXd::Zero(mm, mm);
    for (int i = 0; i + 1 < mm; ++i) F(i, i + 1) = 1.0;
    // Bottom row is φ_mm, φ_{mm−1}, …, φ_1 — reversed, and zero-padded past p.
    for (int k = 0; k < mm; ++k) {
      const int order = mm - k;  // 1-based coefficient index
      F(mm - 1, k) = (order <= p) ? phi[order - 1] : 0.0;
    }

    G = Eigen::RowVectorXd::Zero(mm);
    G(0) = 1.0;

    Hvec = Eigen::VectorXd::Zero(mm);
    std::vector<double> psi(mm + 1, 0.0);
    psi[0] = 1.0;
    for (int j = 1; j <= mm; ++j) {
      double v = 0.0;
      for (int i = 1; i <= p && i <= j; ++i) v += phi[i - 1] * psi[j - i];
      if (j <= q) v -= theta[j - 1];  // PMML writes (1 − θ₁B − …), so θ subtracts
      psi[j] = v;
      Hvec(j - 1) = v;
    }
  }

  // =========================================================================
  // Exact-least-squares filter — called once at load time
  //
  // Differences the training series, builds the spec's F/G/H, solves Ω₁ via
  // Kronecker–Lyapunov and runs the Kalman recursion, leaving Ŝ(n+1) in x_T.
  // =========================================================================

  void run_kalman_filter(const std::vector<double>& phi, const std::vector<double>& theta, int d, int D, int m,
                         double constant, const std::vector<double>& y_raw_chron) {
    kalman.d = d;
    kalman.D = D;
    kalman.m = m;
    kalman.phi = phi;
    kalman.theta = theta;
    kalman.mean = arma_mean(phi, constant);

    // --- Store y_tail for seasonal undifferencing ---
    int y_tail_need = std::max(m * D, 1);
    {
      int from = std::max(0, static_cast<int>(y_raw_chron.size()) - y_tail_need);
      kalman.y_tail.assign(y_raw_chron.begin() + from, y_raw_chron.end());
    }

    // --- Apply seasonal differencing D times ---
    std::vector<double> z_series = y_raw_chron;
    for (int di = 0; di < D; ++di) {
      if (static_cast<int>(z_series.size()) <= m)
        throw cpmml::ParsingException("TimeSeriesModel: not enough training points for seasonal differencing");
      std::vector<double> tmp;
      tmp.reserve(z_series.size() - m);
      for (size_t i = m; i < z_series.size(); ++i) tmp.push_back(z_series[i] - z_series[i - m]);
      z_series = std::move(tmp);
    }

    // --- Store z_tail for regular undifferencing ---
    if (d > 0 && !z_series.empty()) {
      int from = std::max(0, static_cast<int>(z_series.size()) - d);
      kalman.z_tail.assign(z_series.begin() + from, z_series.end());
    }

    // --- Apply regular differencing d times ---
    std::vector<double> y_series = z_series;
    for (int di = 0; di < d; ++di) {
      std::vector<double> tmp;
      tmp.reserve(y_series.size() > 0 ? y_series.size() - 1 : 0);
      for (size_t i = 1; i < y_series.size(); ++i) tmp.push_back(y_series[i] - y_series[i - 1]);
      y_series = std::move(tmp);
    }

    if (y_series.empty())
      throw cpmml::ParsingException("TimeSeriesModel: ARIMA exactLeastSquares requires TimeSeries training data");

    Eigen::VectorXd Hvec;
    build_state_space(phi, theta, kalman.F, kalman.G, Hvec);
    const int mm = static_cast<int>(kalman.F.rows());

    // The state space is zero-mean, so the constant enters as the mean of the
    // differenced series, μ = c / (1 − Σφ). The conditional-LS path carries the
    // same c as an intercept inside its recursion; the two agree.
    for (double& w : y_series) w -= kalman.mean;

    // --- Ω₁: unconditional covariance of the state ---
    //
    // S(t+1) = F·S(t) + H·e(t) with e(t) independent of S(t), so Var(S) solves
    // the discrete Lyapunov equation Ω = F·Ω·Fᵀ + H·Hᵀ. Vectorized as
    // (I_{mm²} − F⊗F)·vec(Ω) = vec(HHᵀ); colPivHouseholderQr handles
    // near-unit-root cases robustly.
    //
    // Initialising from the *unconditional* covariance rather than from the
    // steady state is what makes this exact least squares rather than
    // conditional least squares. It is not cosmetic: on an ARMA(2,2) with d=1
    // the steady-state start is off by 3e-2 against statsmodels where this is
    // off by 4e-7.
    const Eigen::MatrixXd Q = Hvec * Hvec.transpose();
    Eigen::MatrixXd IFF =
        Eigen::MatrixXd::Identity(mm * mm, mm * mm) - Eigen::kroneckerProduct(kalman.F, kalman.F).eval();
    Eigen::Map<const Eigen::VectorXd> vecQ(Q.data(), mm * mm);
    Eigen::VectorXd vecOmega = IFF.colPivHouseholderQr().solve(vecQ);

    Eigen::MatrixXd Omega = Eigen::Map<Eigen::MatrixXd>(vecOmega.data(), mm, mm);
    // Symmetrize. The .eval() is required, not defensive: `A = A + A.transpose()`
    // is Eigen's canonical aliasing hazard — the assignment writes into A
    // coefficient by coefficient, so reading A(j,i) after A(i,j) has already
    // been overwritten yields a matrix that is neither the sum nor symmetric.
    Omega = ((Omega + Omega.transpose()) * 0.5).eval();

    // --- Kalman recursion through the differenced training data ---
    //
    // The state and observation noises are *the same* e(t) here — H·e(t) drives
    // the state and e(t) enters the measurement — so this is the correlated-noise
    // form, not the textbook independent-noise one. With σ² factored out
    // (it cancels in the gain):
    //
    //   Δ(t) = G·Ω·Gᵀ + 1                     innovation variance
    //   Θ(t) = F·Ω·Gᵀ + H                     the covariance term H is the cross-term
    //   Ŝ(t+1) = F·Ŝ(t) + Θ·Δ⁻¹·(Y(t) − G·Ŝ(t))
    //   Ω(t+1) = F·Ω·Fᵀ + H·Hᵀ − Θ·Δ⁻¹·Θᵀ
    //
    // Ω decays to 0 and Θ·Δ⁻¹ to H, which is the steady state the HVector
    // shortcut assumes. Dropping the cross-term (taking Θ = F·Ω·Gᵀ) would leave
    // the MA structure out of the gain entirely.
    kalman.x_T = Eigen::VectorXd::Zero(mm);
    for (double w_obs : y_series) {
      const double delta = (kalman.G * Omega * kalman.G.transpose())(0, 0) + 1.0;
      const Eigen::VectorXd theta_t = Omega * kalman.G.transpose();
      const Eigen::VectorXd Theta = kalman.F * theta_t + Hvec;

      const double innovation = w_obs - (kalman.G * kalman.x_T)(0);
      kalman.x_T = kalman.F * kalman.x_T + Theta * (innovation / delta);

      Omega = kalman.F * Omega * kalman.F.transpose() + Q - (Theta * Theta.transpose()) / delta;
      Omega = ((Omega + Omega.transpose()) * 0.5).eval();  // aliasing — see Ω₁ above
    }
  }

  // =========================================================================
  // Undifferencing: reverse d regular + D seasonal differences
  //
  //   w_fcast  — forecasts of doubly-differenced series
  //   d        — order of regular differencing
  //   D        — order of seasonal differencing
  //   m        — seasonal period
  //   z_tail   — last d values of seasonally-differenced series (ascending)
  //   y_tail   — last max(m*D,1) raw values (ascending)
  // =========================================================================

  static std::vector<double> undifference_sarima(const std::vector<double>& w_fcast, int d, int D, int m,
                                                 const std::vector<double>& z_tail, const std::vector<double>& y_tail) {
    // Step 1: undo d regular differences → z_fcast.
    //
    // Integrating once recovers the (d-1)-times-differenced series, not the
    // original, so this runs d times. Each pass needs the last observed value
    // at its own differencing level, which is recovered by differencing z_tail
    // — the last d values of the series — the corresponding number of times.
    std::vector<double> z_fcast = w_fcast;
    for (int level = d - 1; level >= 0; --level) {
      std::vector<double> tail(z_tail);
      for (int i = 0; i < level && tail.size() > 1; ++i) {
        std::vector<double> differenced;
        differenced.reserve(tail.size() - 1);
        for (size_t k = 1; k < tail.size(); ++k) differenced.push_back(tail[k] - tail[k - 1]);
        tail = std::move(differenced);
      }

      double prev = tail.empty() ? 0.0 : tail.back();
      for (double& value : z_fcast) {
        prev += value;
        value = prev;
      }
    }

    // Step 2: undo D seasonal differences → y_fcast
    if (D == 0) return z_fcast;

    int H = static_cast<int>(z_fcast.size());
    std::vector<double> y_fcast;
    y_fcast.reserve(H);
    for (int h = 0; h < H; ++h) {
      // y_fcast[h] = z_fcast[h] + y[T + h − m]
      // For h < m: look up from y_tail  (last m values of raw series, ascending)
      // For h >= m: look up from y_fcast
      double y_prev;
      if (h < m) {
        int idx = static_cast<int>(y_tail.size()) - m + h;
        y_prev = (idx >= 0 && idx < static_cast<int>(y_tail.size())) ? y_tail[idx] : 0.0;
      } else {
        y_prev = y_fcast[h - m];
      }
      y_fcast.push_back(z_fcast[h] + y_prev);
    }
    return y_fcast;
  }

  // =========================================================================
  // Polynomial convolution: SARIMA AR/MA expansion
  //
  // Returns coefficients of φ(B)·Φ(B^m) after stripping the lag-0 term (=1)
  // and negating, yielding AR coefficients [φ₁_eff, φ₂_eff, ...] for the
  // recursion y(t) = Σ φ_eff_i · y(t-i) + ...
  // =========================================================================

  static std::vector<double> expand_arma_poly(
      const std::vector<double>& phi_ns,  // nonseasonal coefficients [φ₁,..,φ_p]
      const std::vector<double>& phi_s,   // seasonal coefficients [Φ₁,..,Φ_P] at lag m
      int m) {
    if (phi_s.empty()) return phi_ns;

    // A(L) = 1 − φ₁L − ... − φ_pL^p
    int p_ns = static_cast<int>(phi_ns.size());
    std::vector<double> A(p_ns + 1, 0.0);
    A[0] = 1.0;
    for (int i = 0; i < p_ns; ++i) A[i + 1] = -phi_ns[i];

    // C(L) = 1 − Φ₁L^m − ... − Φ_PL^{Pm}
    int P_s = static_cast<int>(phi_s.size());
    std::vector<double> C(P_s * m + 1, 0.0);
    C[0] = 1.0;
    for (int i = 0; i < P_s; ++i) C[(i + 1) * m] = -phi_s[i];

    // Convolve
    std::vector<double> result(A.size() + C.size() - 1, 0.0);
    for (int j = 0; j < static_cast<int>(A.size()); ++j)
      for (int k = 0; k < static_cast<int>(C.size()); ++k) result[j + k] += A[j] * C[k];

    // Extract AR coefficients: skip lag-0 (=1), negate
    std::vector<double> phi_expanded;
    phi_expanded.reserve(result.size() - 1);
    for (size_t i = 1; i < result.size(); ++i) phi_expanded.push_back(-result[i]);

    return phi_expanded;
  }

  // =========================================================================
  // OutlierEffect: correct indexed training values before fitting
  //
  // Reads <OutlierEffect> children from <TimeSeries usage="original"> elements
  // under ts_node, then subtracts the estimated effect from the raw values.
  // =========================================================================

  static void apply_outlier_corrections(std::vector<std::pair<int, double>>& pts, const XmlNode& ts_node) {
    for (const auto& ts : ts_node.get_childs("TimeSeries")) {
      if (ts.exists_attribute("usage") && to_lower(ts.get_attribute("usage")) != "original") continue;
      for (const auto& oe : ts.get_childs("OutlierEffect")) {
        if (!oe.exists_attribute("time")) continue;
        int t = static_cast<int>(oe.get_long_attribute("time"));
        double value = oe.exists_attribute("value") ? oe.get_double_attribute("value") : 0.0;
        double delta = oe.exists_attribute("delta") ? oe.get_double_attribute("delta") : 0.0;
        std::string type = oe.exists_attribute("type") ? to_lower(oe.get_attribute("type")) : "additive";

        for (auto& [idx, val] : pts) {
          if (type == "additive") {
            if (idx == t) val -= value;
          } else if (type == "levelshift") {
            if (idx >= t) val -= value;
          } else if (type == "transientchange") {
            if (idx >= t) val -= value * std::pow(delta, static_cast<double>(idx - t));
          }
        }
      }
    }
  }

  // =========================================================================
  // Parsing helpers
  // =========================================================================

  static std::vector<double> parse_array(const XmlNode& node) {
    std::vector<double> v;
    for (const auto& p : split(node.value(), " "))
      if (!p.empty()) v.push_back(to_double(p));
    return v;
  }

  /**
   * Conformance checks on the KalmanState children that scoring does not read.
   *
   * A document that carries them in the wrong shape is not describing the state
   * space cPMML is about to forecast from, and saying so at load time beats
   * scoring it against a state whose basis is anyone's guess.
   */
  static void check_array_length(const XmlNode& parent, const std::string& element, int expected) {
    if (!parent.exists_child(element)) return;
    const XmlNode node = parent.get_child(element);
    if (!node.exists_child("Array")) throw cpmml::ParsingException(element + " must contain an Array");
    const size_t n = parse_array(node.get_child("Array")).size();
    if (static_cast<int>(n) != expected)
      throw cpmml::ParsingException("TimeSeriesModel: " + element + " has " + std::to_string(n) +
                                    " entries, expected max(p,q) = " + std::to_string(expected));
  }

  static void check_matrix_shape(const XmlNode& parent, const std::string& element, int expected) {
    if (!parent.exists_child(element)) return;
    const Eigen::MatrixXd mat = load_matrix(parent.get_child(element), element);
    if (mat.rows() != expected || mat.cols() != expected)
      throw cpmml::ParsingException("TimeSeriesModel: " + element + " is " + std::to_string(mat.rows()) + "x" +
                                    std::to_string(mat.cols()) + ", expected max(p,q) = " +
                                    std::to_string(expected) + " square");
  }

  // Collect all (index, value) pairs from <TimeSeries usage="original">
  static std::vector<std::pair<int, double>> collect_timeseries(const XmlNode& ts_node) {
    std::vector<std::pair<int, double>> pts;
    for (const auto& ts : ts_node.get_childs("TimeSeries")) {
      if (ts.exists_attribute("usage") && to_lower(ts.get_attribute("usage")) != "original") continue;
      for (const auto& tv : ts.get_childs("TimeValue")) {
        int idx = static_cast<int>(tv.get_long_attribute("index"));
        double val = tv.get_double_attribute("value");
        pts.push_back({idx, val});
      }
    }
    return pts;
  }

  // =========================================================================
  // parse_ets
  // =========================================================================

  void parse_ets(const XmlNode& node) {
    ets.trend = "none";
    ets.seasonal = "none";

    if (node.exists_attribute("transformation"))
      ets.transform = parse_transform(to_lower(node.get_attribute("transformation")));
    if (node.exists_attribute("RMSE")) ets.rmse = node.get_double_attribute("RMSE");

    // Level (required by spec)
    XmlNode lv = node.get_child("Level");
    ets.alpha = lv.exists_attribute("alpha") ? lv.get_double_attribute("alpha") : 0.0;
    ets.l = lv.get_double_attribute("smoothedValue");

    // Trend_ExpoSmooth (optional)
    ets.b = 0.0;
    ets.phi = 1.0;
    ets.gamma = 0.0;
    if (node.exists_child("Trend_ExpoSmooth")) {
      XmlNode tr = node.get_child("Trend_ExpoSmooth");
      ets.trend = tr.exists_attribute("trend") ? to_lower(tr.get_attribute("trend")) : "additive";
      // Brown's polynomial models read their smoothed coefficients from the
      // Array sub-element rather than smoothedValue, and forecast by a
      // different rule than the Gardner trends below. Scoring one as though it
      // had no trend at all would silently return a flat line.
      if (ets.trend == "polynomial_exponential")
        throw cpmml::ParsingException(
            "ExponentialSmoothing trend=\"polynomial_exponential\" (Brown's polynomial model) is not supported");
      if (ets.trend != "additive" && ets.trend != "damped_additive" && ets.trend != "multiplicative" &&
          ets.trend != "damped_multiplicative")
        throw cpmml::ParsingException("ExponentialSmoothing trend=\"" + ets.trend + "\" is not supported");
      ets.gamma = tr.exists_attribute("gamma") ? tr.get_double_attribute("gamma") : 0.0;
      ets.phi = tr.exists_attribute("phi") ? tr.get_double_attribute("phi") : 1.0;
      ets.b = tr.get_double_attribute("smoothedValue");
    }

    // Seasonality_ExpoSmooth (optional)
    if (node.exists_child("Seasonality_ExpoSmooth")) {
      XmlNode se = node.get_child("Seasonality_ExpoSmooth");
      ets.seasonal = se.exists_attribute("type") ? to_lower(se.get_attribute("type")) : "additive";
      ets.delta = se.exists_attribute("delta") ? se.get_double_attribute("delta") : 0.0;
      ets.period = static_cast<int>(se.get_long_attribute("period"));
      ets.phase = se.exists_attribute("phase") ? static_cast<int>(se.get_long_attribute("phase")) : ets.period;
      if (se.exists_child("Array")) {
        ets.s = parse_array(se.get_child("Array"));
      } else {
        auto tvs = se.get_childs("TimeValue");
        if (!tvs.empty()) {
          ets.s.resize(tvs.size(), 0.0);
          for (const auto& tv : tvs) {
            int idx = static_cast<int>(tv.get_long_attribute("index")) - 1;
            if (idx >= 0 && idx < static_cast<int>(ets.s.size())) ets.s[idx] = tv.get_double_attribute("value");
          }
        }
      }
    }
  }

  // =========================================================================
  // parse_ssm
  // =========================================================================

  /**
   * Read a <TransitionMatrix>/<MeasurementMatrix> wrapper.
   *
   * Both hold a <Matrix> whose rows are one <Array> each, per GeneralStructure:
   * "If arrays are used, then each array contains elements of one row in the
   * matrix." nbRows/nbCols are optional -- "if one of them is not specified,
   * the respective dimension is implicitly given by the representation" -- so
   * they are validated when present rather than required.
   */
  static Eigen::MatrixXd load_matrix(const XmlNode& wrapper, const std::string& element) {
    if (!wrapper.exists_child("Matrix"))
      throw cpmml::ParsingException(element + " must contain a Matrix element");

    const XmlNode matrix = wrapper.get_child("Matrix");
    if (matrix.exists_child("MatCell"))
      throw cpmml::ParsingException(element + ": sparse Matrix (MatCell) is not supported");

    std::vector<std::vector<double>> rows;
    for (const auto& array : matrix.get_childs("Array")) rows.push_back(parse_array(array));

    if (rows.empty()) throw cpmml::ParsingException(element + ": Matrix has no Array rows");

    const size_t cols = rows.front().size();
    for (const auto& row : rows)
      if (row.size() != cols) throw cpmml::ParsingException(element + ": Matrix rows have differing lengths");

    if (matrix.exists_attribute("nbRows") && static_cast<size_t>(matrix.get_long_attribute("nbRows")) != rows.size())
      throw cpmml::ParsingException(element + ": nbRows disagrees with the number of Array rows");
    if (matrix.exists_attribute("nbCols") && static_cast<size_t>(matrix.get_long_attribute("nbCols")) != cols)
      throw cpmml::ParsingException(element + ": nbCols disagrees with the Array length");

    Eigen::MatrixXd mat(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(cols));
    for (size_t r = 0; r < rows.size(); ++r)
      for (size_t c = 0; c < cols; ++c)
        mat(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = rows[r][c];

    return mat;
  }

  void parse_ssm(const XmlNode& node) {
    if (node.exists_attribute("intercept")) ssm.intercept = node.get_double_attribute("intercept");
    if (node.exists_attribute("variance")) ssm.sigma2 = node.get_double_attribute("variance");

    // PsiVector: MA(∞) weights ψ₀,ψ₁,... for exact h-step variance
    // var(h) = σ² × Σ_{j=0}^{h-1} ψ_j²   (PMML 4.4 spec §TimeSeriesModel)
    if (node.exists_child("PsiVector")) {
      XmlNode pv = node.get_child("PsiVector");
      // @variance on PsiVector overrides RMSE-derived sigma2
      if (pv.exists_attribute("variance")) ssm.sigma2 = pv.get_double_attribute("variance");
      if (pv.exists_child("Array")) ssm.psi = parse_array(pv.get_child("Array"));
    }

    // StateVector holds a REAL-ARRAY; the n attribute belongs to the Array.
    const XmlNode sv = node.get_child("StateVector");
    if (!sv.exists_child("Array")) throw cpmml::ParsingException("StateVector must contain an Array element");
    const std::vector<double> vals = parse_array(sv.get_child("Array"));
    ssm.x.resize(static_cast<Eigen::Index>(vals.size()));
    for (size_t i = 0; i < vals.size(); ++i) ssm.x[static_cast<Eigen::Index>(i)] = vals[i];

    ssm.F = load_matrix(node.get_child("TransitionMatrix"), "TransitionMatrix");
    ssm.G = load_matrix(node.get_child("MeasurementMatrix"), "MeasurementMatrix");

    if (ssm.F.rows() != ssm.F.cols() || ssm.F.cols() != ssm.x.size())
      throw cpmml::ParsingException("StateSpaceModel: TransitionMatrix must be square and match the StateVector");
    if (ssm.G.cols() != ssm.x.size())
      throw cpmml::ParsingException("StateSpaceModel: MeasurementMatrix columns must match the StateVector");
  }

  // =========================================================================
  // parse_arima
  // =========================================================================

  void parse_arima(const XmlNode& node, const XmlNode& ts_node) {
    // --- Nonseasonal component ---
    int p = 0, d = 0, q = 0;
    std::vector<double> phi, theta, residuals;

    if (node.exists_child("NonseasonalComponent")) {
      XmlNode ns = node.get_child("NonseasonalComponent");
      p = ns.exists_attribute("p") ? static_cast<int>(ns.get_long_attribute("p")) : 0;
      d = ns.exists_attribute("d") ? static_cast<int>(ns.get_long_attribute("d")) : 0;
      q = ns.exists_attribute("q") ? static_cast<int>(ns.get_long_attribute("q")) : 0;

      if (ns.exists_child("AR") && ns.get_child("AR").exists_child("Array"))
        phi = parse_array(ns.get_child("AR").get_child("Array"));

      if (ns.exists_child("MA")) {
        XmlNode ma = ns.get_child("MA");
        if (ma.exists_child("MACoefficients") && ma.get_child("MACoefficients").exists_child("Array"))
          theta = parse_array(ma.get_child("MACoefficients").get_child("Array"));
        if (ma.exists_child("Residuals") && ma.get_child("Residuals").exists_child("Array"))
          residuals = parse_array(ma.get_child("Residuals").get_child("Array"));
      }
    }

    // --- Seasonal component ---
    int P = 0, D = 0, Q = 0, m = 1;
    if (node.exists_child("SeasonalComponent")) {
      XmlNode sc = node.get_child("SeasonalComponent");
      P = sc.exists_attribute("P") ? static_cast<int>(sc.get_long_attribute("P")) : 0;
      D = sc.exists_attribute("D") ? static_cast<int>(sc.get_long_attribute("D")) : 0;
      Q = sc.exists_attribute("Q") ? static_cast<int>(sc.get_long_attribute("Q")) : 0;
      m = sc.exists_attribute("period") ? static_cast<int>(sc.get_long_attribute("period")) : 1;

      std::vector<double> phi_s, theta_s;
      if (sc.exists_child("AR") && sc.get_child("AR").exists_child("Array"))
        phi_s = parse_array(sc.get_child("AR").get_child("Array"));
      if (sc.exists_child("MA")) {
        XmlNode ma_s = sc.get_child("MA");
        if (ma_s.exists_child("MACoefficients") && ma_s.get_child("MACoefficients").exists_child("Array"))
          theta_s = parse_array(ma_s.get_child("MACoefficients").get_child("Array"));
      }

      // Expand: φ(B)·Φ(B^m) and θ(B)·Θ(B^m)
      phi = expand_arma_poly(phi, phi_s, m);
      theta = expand_arma_poly(theta, theta_s, m);
    }
    (void)p;
    (void)q;
    (void)P;
    (void)Q;  // orders now encoded in phi/theta sizes

    // --- Shared attributes ---
    double constant_val = node.exists_attribute("constantTerm") ? node.get_double_attribute("constantTerm") : 0.0;
    bool include_const = (constant_val != 0.0);

    Transform tr = Transform::NONE;
    if (node.exists_attribute("transformation")) tr = parse_transform(to_lower(node.get_attribute("transformation")));

    double sigma2_val = 1.0;
    if (node.exists_attribute("RMSE")) {
      double rmse = node.get_double_attribute("RMSE");
      sigma2_val = rmse * rmse;
    }

    std::string method = "conditionalleastsquares";
    if (node.exists_attribute("predictionMethod")) method = to_lower(node.get_attribute("predictionMethod"));

    // --- Collect and pre-process training data ---
    auto pts = collect_timeseries(ts_node);
    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Apply input transformation (log/sqrt) to raw values
    if (tr != Transform::NONE)
      for (auto& [idx, val] : pts) val = apply_transform(val, tr);

    // Apply outlier corrections (subtract outlier effects from observations)
    apply_outlier_corrections(pts, ts_node);

    if (method == "exactleastsquares") {
      // ----- Kalman path -----

      // Try FinalStateVector shortcut first (avoids re-filtering)
      // MaximumLikelihoodStat is a child of ARIMA, not of TimeSeriesModel:
      //   <xs:element name="ARIMA"> … <xs:element ref="MaximumLikelihoodStat" minOccurs="0"/>
      //
      // Its KalmanState carries the state the producer's own filter arrived at,
      // which is authoritative — the producer may have estimated over data the
      // document does not carry. Absent it (or for method="thetaRecursion",
      // whose ThetaRecursionState is a different route to the same exact-LS
      // predictor), we re-derive the state by filtering below.
      bool have_state = false;
      if (node.exists_child("MaximumLikelihoodStat")) {
        XmlNode mls = node.get_child("MaximumLikelihoodStat");
        if (mls.exists_child("KalmanState")) {
          XmlNode ks = mls.get_child("KalmanState");
          if (ks.exists_child("FinalStateVector") && ks.get_child("FinalStateVector").exists_child("Array")) {
            auto fsv = parse_array(ks.get_child("FinalStateVector").get_child("Array"));

            Eigen::VectorXd derived_h;
            build_state_space(phi, theta, kalman.F, kalman.G, derived_h);
            const int mm = static_cast<int>(kalman.F.rows());

            // "The length of the ARRAY should equal to m, the larger of the
            // degree of autoregression and the degree of moving average." This
            // is not pedantry: the state is only meaningful in the basis the
            // spec defines, and the length is the one check that catches a
            // vector written in some other one.
            check_array_length(ks, "FinalStateVector", mm);

            kalman.x_T.resize(mm);
            for (int i = 0; i < mm; ++i) kalman.x_T[i] = fsv[i];

            // HVector and FinalOmega are validated but cannot change the answer,
            // and it is worth being precise about why rather than storing either
            // in a member nothing reads.
            //
            // HVector exists "to define the first m psi weights instead of
            // recalculating at scoring time" — i.e. to save deriving H. But H
            // only drives the filter that *produces* the state; the forecast is
            // G·F^(h−1)·S, with no H in it. The one case that would need H is
            // deriving the state ourselves, and that is exactly the case where
            // the document carries no KalmanState to hold an HVector. So the
            // shortcut is honoured by there being nothing left to shorten.
            //
            // FinalOmega drives the spec's Ω(h) = F·Ω(h−1)·Fᵀ + H·Hᵀ recursion,
            // which propagates covariance in the *differenced* basis: G·Ω(h)·Gᵀ
            // describes w, not y. On an AR(2) with d=1 it flattens out at 1.82
            // where the variance of y grows past 17. arima_variance() folds
            // (1−B)^d (1−B^m)^D into Φ and so answers the question the forecast
            // actually poses, agreeing with statsmodels to 2e-6 relative on a
            // pure AR. See m-S12 and M-S14.
            check_array_length(ks, "HVector", mm);
            check_matrix_shape(ks, "FinalOmega", mm);

            kalman.d = d;
            kalman.D = D;
            kalman.m = m;
            kalman.transform = tr;
            kalman.sigma2 = sigma2_val;
            kalman.phi = phi;
            kalman.theta = theta;
            kalman.mean = arma_mean(phi, constant_val);

            // Build y_tail and z_tail for undifferencing
            setup_arima_tails(pts, d, D, m, kalman.y_tail, kalman.z_tail);

            algorithm = Algorithm::ARIMA_KALMAN;
            have_state = true;
          }
        }
      }

      if (!have_state && !pts.empty()) {
        algorithm = Algorithm::ARIMA_KALMAN;
        kalman.transform = tr;
        kalman.sigma2 = sigma2_val;

        std::vector<double> y_raw_chron;
        y_raw_chron.reserve(pts.size());
        for (auto& [idx, val] : pts) y_raw_chron.push_back(val);
        run_kalman_filter(phi, theta, d, D, m, constant_val, y_raw_chron);
      }

    } else {
      // ----- Conditional LS path -----
      algorithm = Algorithm::ARIMA_CLS;
      arima.p = p;
      arima.d = d;
      arima.q = q;
      arima.D = D;
      arima.m = m;
      arima.phi = phi;
      arima.theta = theta;
      arima.constant = constant_val;
      arima.include_constant = include_const;
      arima.transform = tr;
      arima.sigma2 = sigma2_val;

      // Build y_tail and z_tail for undifferencing
      setup_arima_tails(pts, d, D, m, arima.y_tail, arima.z_tail);

      // Build fully-differenced w_series for CLS history
      std::vector<double> w_series;
      {
        std::vector<double> y_all;
        y_all.reserve(pts.size());
        for (auto& [idx, val] : pts) y_all.push_back(val);
        w_series = y_all;

        // Seasonal differencing
        for (int di = 0; di < D; ++di) {
          if (static_cast<int>(w_series.size()) <= m) break;
          std::vector<double> tmp;
          tmp.reserve(w_series.size() - m);
          for (size_t k = m; k < w_series.size(); ++k) tmp.push_back(w_series[k] - w_series[k - m]);
          w_series = std::move(tmp);
        }
        // Regular differencing
        for (int di = 0; di < d; ++di) {
          if (w_series.size() < 2) break;
          std::vector<double> tmp;
          tmp.reserve(w_series.size() - 1);
          for (size_t k = 1; k < w_series.size(); ++k) tmp.push_back(w_series[k] - w_series[k - 1]);
          w_series = std::move(tmp);
        }
      }

      // y_diff: last p_eff values of w_series, most-recent first
      int p_eff = static_cast<int>(phi.size());
      if (!w_series.empty()) {
        int from_w = std::max(0, static_cast<int>(w_series.size()) - p_eff);
        for (int i = static_cast<int>(w_series.size()) - 1; i >= from_w; --i) arima.y_diff.push_back(w_series[i]);
      }

      // eps: use stored residuals or zero-init
      if (!residuals.empty())
        arima.eps = residuals;
      else
        arima.eps.assign(theta.size(), 0.0);
    }
  }

  // =========================================================================
  // setup_arima_tails — shared by both CLS and Kalman paths
  //
  // Computes y_tail (last max(m*D,1) raw values) and z_tail (last d values
  // of the seasonally-differenced series) needed for undifferencing.
  // =========================================================================

  static void setup_arima_tails(const std::vector<std::pair<int, double>>& pts,  // ascending by index
                                int d, int D, int m, std::vector<double>& y_tail, std::vector<double>& z_tail) {
    // y_tail: last max(m*D, 1) raw values
    int y_need = std::max(m * D, 1);
    {
      int from = std::max(0, static_cast<int>(pts.size()) - y_need);
      for (int i = from; i < static_cast<int>(pts.size()); ++i) y_tail.push_back(pts[i].second);
    }

    // z_tail: last d values of seasonally-differenced series
    if (d > 0 && !pts.empty()) {
      std::vector<double> z;
      z.reserve(pts.size());
      for (auto& [idx, val] : pts) z.push_back(val);

      for (int di = 0; di < D; ++di) {
        if (static_cast<int>(z.size()) <= m) break;
        std::vector<double> tmp;
        tmp.reserve(z.size() - m);
        for (size_t k = m; k < z.size(); ++k) tmp.push_back(z[k] - z[k - m]);
        z = std::move(tmp);
      }
      if (!z.empty()) {
        int from = std::max(0, static_cast<int>(z.size()) - d);
        z_tail.assign(z.begin() + from, z.end());
      }
    }
  }

  // =========================================================================
  // GARCH — conditional mean (ARMAPart recursion, identical to ARIMA CLS)
  // =========================================================================

  std::vector<double> forecast_garch_mean(int horizon) const {
    const auto& g = garch;
    std::vector<double> y(g.y_diff), eps(g.eps_mean);
    std::vector<double> out;
    out.reserve(horizon);
    int p_eff = static_cast<int>(g.phi.size());
    int q_eff = static_cast<int>(g.theta.size());
    for (int h = 1; h <= horizon; ++h) {
      double yh = g.constant_mean;
      for (int i = 0; i < p_eff && i < static_cast<int>(y.size()); ++i) yh += g.phi[i] * y[i];
      for (int j = 0; j < q_eff && j < static_cast<int>(eps.size()); ++j) yh += g.theta[j] * eps[j];
      out.push_back(yh);
      y.insert(y.begin(), yh);
      eps.insert(eps.begin(), 0.0);
    }
    return out;
  }

  // =========================================================================
  // GARCH — conditional variance (multi-step GARCH(gp,gq) recursion)
  //
  // For h=1: σ²(T+1) = ω + Σαᵢε²(T+1-i) + Σβⱼσ²(T+1-j)   [all past known]
  // For h>1: E[ε²(T+k)] = σ²(T+k) for k>0  (future ε² replaced by σ²)
  // =========================================================================

  std::vector<double> variance_garch(int horizon) const {
    const auto& g = garch;
    std::vector<double> vars;
    vars.reserve(horizon);
    for (int h = 1; h <= horizon; ++h) {
      double s2 = g.omega;
      // ARCH terms αᵢ × E[ε²(T+h-i)], i=1..gq
      for (int i = 1; i <= g.gq; ++i) {
        double val;
        if (i >= h) {
          // Past known: eps_sq[i-h] (0-indexed, most-recent first)
          int idx = i - h;
          val = idx < static_cast<int>(g.eps_sq.size()) ? g.eps_sq[idx] : 0.0;
        } else {
          val = vars[h - i - 1];  // already computed σ²(T+h-i)
        }
        if (i - 1 < static_cast<int>(g.alpha.size())) s2 += g.alpha[i - 1] * val;
      }
      // GARCH terms βⱼ × σ²(T+h-j), j=1..gp
      for (int j = 1; j <= g.gp; ++j) {
        double val;
        if (j >= h) {
          int idx = j - h;
          val = idx < static_cast<int>(g.sigma2_hist.size()) ? g.sigma2_hist[idx] : g.omega;
        } else {
          val = vars[h - j - 1];
        }
        if (j - 1 < static_cast<int>(g.beta_garch.size())) s2 += g.beta_garch[j - 1] * val;
      }
      vars.push_back(s2);
    }
    return vars;
  }

  // =========================================================================
  // DynamicRegressor — add transfer-function contributions to base forecasts
  // =========================================================================

  void apply_dynamic_regressors(std::vector<double>& base, int horizon,
                                const std::unordered_map<std::string, std::vector<double>>& regressors) const {
    for (const auto& dr : dynamic_regressors) {
      auto contrib = compute_dr_contribution(dr, horizon, regressors);
      for (int h = 0; h < horizon && h < static_cast<int>(contrib.size()); ++h) base[h] += contrib[h];
    }
  }

  // Compute v(T+1)..v(T+H) via the IIR transfer function N(B)/D(B).
  std::vector<double> compute_dr_contribution(
      const DynamicRegressorState& dr, int horizon,
      const std::unordered_map<std::string, std::vector<double>>& regressors) const {
    // --- resolve future x values ---
    std::vector<double> x_future;
    if (dr.user_supplied) {
      auto it = regressors.find(dr.field);
      if (it == regressors.end())
        throw cpmml::ParsingException("DynamicRegressor field \"" + dr.field +
                                      "\" requires future values (userSupplied)");
      if (static_cast<int>(it->second.size()) < horizon)
        throw cpmml::ParsingException("DynamicRegressor field \"" + dr.field + "\": not enough future values provided");
      x_future = it->second;
    } else {
      // constant: x(T+h) = x(T) for all h
      double x_const = dr.x_history.empty() ? 0.0 : dr.x_history[0];
      x_future.assign(horizon, x_const);
    }
    if (dr.transform != Transform::NONE)
      for (auto& v : x_future) v = apply_transform(v, dr.transform);

    // --- build chronological x array: [x(T-n+1),..,x(T), x(T+1),..,x(T+H)] ---
    int n_hist = static_cast<int>(dr.x_history.size());
    std::vector<double> x_all;
    x_all.reserve(n_hist + horizon);
    for (int i = n_hist - 1; i >= 0; --i) x_all.push_back(dr.x_history[i]);  // ascending
    for (double xf : x_future) x_all.push_back(xf);
    // x_all[n_hist - 1] = x(T),  x_all[n_hist + h - 1] = x(T+h)

    // --- build v_chron: [v(T-n_vhist+1),..,v(T)] then extend with future v ---
    int n_vhist = static_cast<int>(dr.v_history.size());
    std::vector<double> v_chron;
    v_chron.reserve(n_vhist + horizon);
    for (int i = n_vhist - 1; i >= 0; --i) v_chron.push_back(dr.v_history[i]);

    int r = static_cast<int>(dr.numerator.size());
    int s = static_cast<int>(dr.denominator.size());

    std::vector<double> contrib;
    contrib.reserve(horizon);
    for (int h = 1; h <= horizon; ++h) {
      double vt = 0.0;
      // Numerator: βᵢ × x(T+h-delay-i)
      for (int i = 0; i < r; ++i) {
        int x_idx = n_hist + h - 1 - dr.delay - i;  // index into x_all
        if (x_idx >= 0 && x_idx < static_cast<int>(x_all.size())) vt += dr.numerator[i] * x_all[x_idx];
      }
      // Denominator: −δⱼ × v(T+h-1-j)  (v_chron is ascending, last = most recent)
      for (int j = 0; j < s; ++j) {
        int v_idx = static_cast<int>(v_chron.size()) - 1 - j;
        if (v_idx >= 0) vt -= dr.denominator[j] * v_chron[v_idx];
      }
      v_chron.push_back(vt);
      contrib.push_back(vt);
    }
    return contrib;
  }

  // =========================================================================
  // parse_garch
  // =========================================================================

  void parse_garch(const XmlNode& node, const XmlNode& ts_node) {
    // --- ARMAPart (conditional mean) ---
    if (node.exists_child("ARMAPart")) {
      XmlNode ap = node.get_child("ARMAPart");
      garch.p_mean = ap.exists_attribute("p") ? static_cast<int>(ap.get_long_attribute("p")) : 0;
      garch.q_mean = ap.exists_attribute("q") ? static_cast<int>(ap.get_long_attribute("q")) : 0;
      garch.constant_mean = ap.exists_attribute("constant") ? ap.get_double_attribute("constant") : 0.0;

      if (ap.exists_child("AR") && ap.get_child("AR").exists_child("Array"))
        garch.phi = parse_array(ap.get_child("AR").get_child("Array"));

      if (ap.exists_child("MA")) {
        XmlNode ma = ap.get_child("MA");
        if (ma.exists_child("MACoefficients") && ma.get_child("MACoefficients").exists_child("Array"))
          garch.theta = parse_array(ma.get_child("MACoefficients").get_child("Array"));
        if (ma.exists_child("Residuals") && ma.get_child("Residuals").exists_child("Array"))
          garch.eps_mean = parse_array(ma.get_child("Residuals").get_child("Array"));
      }
    }

    // Past y values for AR mean recursion (from TimeSeries, descending)
    auto pts = collect_timeseries(ts_node);
    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    int p_eff = static_cast<int>(garch.phi.size());
    for (int i = static_cast<int>(pts.size()) - 1; i >= 0 && static_cast<int>(garch.y_diff.size()) < p_eff; --i)
      garch.y_diff.push_back(pts[i].second);
    if (garch.eps_mean.empty()) garch.eps_mean.assign(garch.q_mean, 0.0);

    // --- GARCHPart (conditional variance) ---
    if (node.exists_child("GARCHPart")) {
      XmlNode gp_node = node.get_child("GARCHPart");
      garch.gp = gp_node.exists_attribute("gp") ? static_cast<int>(gp_node.get_long_attribute("gp")) : 0;
      garch.gq = gp_node.exists_attribute("gq") ? static_cast<int>(gp_node.get_long_attribute("gq")) : 0;
      garch.omega = gp_node.exists_attribute("constant") ? gp_node.get_double_attribute("constant") : 0.0;

      if (gp_node.exists_child("ResidualSquareCoefficients")) {
        XmlNode rsc = gp_node.get_child("ResidualSquareCoefficients");
        if (rsc.exists_child("MACoefficients") && rsc.get_child("MACoefficients").exists_child("Array"))
          garch.alpha = parse_array(rsc.get_child("MACoefficients").get_child("Array"));
        if (rsc.exists_child("Residuals") && rsc.get_child("Residuals").exists_child("Array")) {
          // Residuals stores ε values; square them for the variance equation
          for (double e : parse_array(rsc.get_child("Residuals").get_child("Array"))) garch.eps_sq.push_back(e * e);
        }
      }

      if (gp_node.exists_child("VarianceCoefficients")) {
        XmlNode vc = gp_node.get_child("VarianceCoefficients");
        if (vc.exists_child("MACoefficients") && vc.get_child("MACoefficients").exists_child("Array"))
          garch.beta_garch = parse_array(vc.get_child("MACoefficients").get_child("Array"));
        if (vc.exists_child("PastVariances") && vc.get_child("PastVariances").exists_child("Array"))
          garch.sigma2_hist = parse_array(vc.get_child("PastVariances").get_child("Array"));
      }
    }

    if (garch.alpha.empty()) garch.alpha.assign(garch.gq, 0.0);
    if (garch.beta_garch.empty()) garch.beta_garch.assign(garch.gp, 0.0);
    if (garch.eps_sq.empty()) garch.eps_sq.assign(garch.gq, 0.0);
    if (garch.sigma2_hist.empty()) garch.sigma2_hist.assign(garch.gp, 0.0);
  }

  // =========================================================================
  // parse_dynamic_regressors — reads all DynamicRegressor children of ts_node
  // =========================================================================

  void parse_dynamic_regressors(const XmlNode& ts_node) {
    for (const auto& dr_node : ts_node.get_childs("DynamicRegressor")) {
      DynamicRegressorState dr;
      dr.field = dr_node.get_attribute("field");
      dr.delay = dr_node.exists_attribute("delay") ? static_cast<int>(dr_node.get_long_attribute("delay")) : 0;
      if (dr_node.exists_attribute("transformation"))
        dr.transform = parse_transform(to_lower(dr_node.get_attribute("transformation")));

      std::string fvm = "constant";
      if (dr_node.exists_attribute("futureValuesMethod")) fvm = to_lower(dr_node.get_attribute("futureValuesMethod"));
      dr.user_supplied = (fvm == "usersupplied");

      if (dr_node.exists_child("Numerator") && dr_node.get_child("Numerator").exists_child("Array"))
        dr.numerator = parse_array(dr_node.get_child("Numerator").get_child("Array"));
      if (dr_node.exists_child("Denominator") && dr_node.get_child("Denominator").exists_child("Array"))
        dr.denominator = parse_array(dr_node.get_child("Denominator").get_child("Array"));

      if (dr_node.exists_child("RegressorValues") && dr_node.get_child("RegressorValues").exists_child("Array")) {
        dr.x_history = parse_array(dr_node.get_child("RegressorValues").get_child("Array"));
        if (dr.transform != Transform::NONE)
          for (auto& v : dr.x_history) v = apply_transform(v, dr.transform);
      }

      // Compute v_history by running the IIR filter over x_history (zero initial conditions)
      {
        int n = static_cast<int>(dr.x_history.size());
        int r = static_cast<int>(dr.numerator.size());
        int s = static_cast<int>(dr.denominator.size());
        std::vector<double> x_chron(dr.x_history.rbegin(), dr.x_history.rend());
        std::vector<double> v_chron;
        v_chron.reserve(n);
        for (int t = 0; t < n; ++t) {
          double vt = 0.0;
          for (int i = 0; i < r; ++i) {
            int x_eff = t - dr.delay - i;
            if (x_eff >= 0 && x_eff < n) vt += dr.numerator[i] * x_chron[x_eff];
          }
          for (int j = 0; j < s; ++j) {
            int v_idx = t - 1 - j;
            if (v_idx >= 0) vt -= dr.denominator[j] * v_chron[v_idx];
          }
          v_chron.push_back(vt);
        }
        dr.v_history.assign(v_chron.rbegin(), v_chron.rend());
      }

      dynamic_regressors.push_back(std::move(dr));
    }
  }
};

#endif
