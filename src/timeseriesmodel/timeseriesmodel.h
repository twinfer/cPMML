
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TIMESERIESMODEL_H
#define CPMML_TIMESERIESMODEL_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
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
 *   ARIMA exactLS         — Kalman filter on companion state-space form;
 *                           P₀ solved via Kronecker–Lyapunov equation
 *
 * Extra PMML features supported:
 *   - transformation="logarithmic|squareroot" on ETS/SSM/ARIMA
 *   - StateSpaceModel/@intercept
 *   - MaximumLikelihoodStat/KalmanState/FinalStateVector shortcut (skip re-filter)
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
    std::vector<double> s;  // seasonal components [s(T-m+1),..,s(T)]
    Transform transform = Transform::NONE;
    double rmse = 1.0;  // in-sample RMSE for variance
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
    Eigen::MatrixXd F;           // companion transition (n×n)
    Eigen::RowVectorXd H;        // observation (1×n)
    Eigen::VectorXd x_T;         // terminal filtered state
    Eigen::MatrixXd P_T;         // terminal filtered covariance (for variance)
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
    std::vector<double> alpha;        // ARCH coefficients α₁,..,α_{gp}
    std::vector<double> beta_garch;   // GARCH coefficients β₁,..,β_{gq}
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
        vars = variance_arima_cls(horizon);
        break;
      case Algorithm::ARIMA_KALMAN:
        vars = variance_arima_kalman(horizon);
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

      double seas = has_seas ? e.s[static_cast<size_t>((h - 1) % m)] : (e.seasonal == "multiplicative" ? 1.0 : 0.0);

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

  std::vector<double> variance_ets(int horizon) const {
    // Approximate h-step variance: σ² × h (linear growth, conservative)
    double s2 = ets.rmse * ets.rmse;
    std::vector<double> v;
    v.reserve(horizon);
    for (int h = 1; h <= horizon; ++h) v.push_back(s2 * h);
    return v;
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
    // If PsiVector was present in the PMML, use it: var(h) = σ² × Σ_{j=0}^{h-1} ψ_j²
    // Weights beyond the stored horizon are treated as 0 (stationary decay assumption).
    // Otherwise fall back to constant σ² per step.
    if (!ssm.psi.empty()) {
      std::vector<double> vars;
      vars.reserve(horizon);
      double cumsum = 0.0;
      for (int h = 0; h < horizon; ++h) {
        if (h < static_cast<int>(ssm.psi.size())) cumsum += ssm.psi[h] * ssm.psi[h];
        vars.push_back(ssm.sigma2 * cumsum);
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
      for (int j = 0; j < q_eff && j < static_cast<int>(eps.size()); ++j) yh += a.theta[j] * eps[j];
      w_out.push_back(yh);
      y.insert(y.begin(), yh);
      eps.insert(eps.begin(), 0.0);
    }

    auto result = undifference_sarima(w_out, a.d, a.D, a.m, a.z_tail, a.y_tail);
    return invert_transform_vec(std::move(result), a.transform);
  }

  std::vector<double> variance_arima_cls(int horizon) const {
    // MA(∞) psi-weight variance: var(h) = σ² × Σ_{j=0}^{h-1} ψ_j²
    const auto& a = arima;
    int p_eff = static_cast<int>(a.phi.size());
    int q_eff = static_cast<int>(a.theta.size());

    std::vector<double> psi(horizon, 0.0);
    psi[0] = 1.0;
    for (int j = 1; j < horizon; ++j) {
      double v = 0.0;
      for (int i = 0; i < p_eff && i < j; ++i) v += a.phi[i] * psi[j - 1 - i];
      if (j <= q_eff) v += a.theta[j - 1];
      psi[j] = v;
    }

    std::vector<double> vars;
    vars.reserve(horizon);
    double cumsum = 0.0;
    for (int h = 0; h < horizon; ++h) {
      cumsum += psi[h] * psi[h];
      vars.push_back(a.sigma2 * cumsum);
    }
    return vars;
  }

  // =========================================================================
  // ARIMA Kalman forecasting: ŷ(T+h) = H · Fʰ · x_T
  // =========================================================================

  std::vector<double> forecast_arima_kalman(int horizon) const {
    std::vector<double> w_out;
    w_out.reserve(horizon);

    Eigen::VectorXd state = kalman.x_T;
    for (int h = 1; h <= horizon; ++h) {
      state = kalman.F * state;
      w_out.push_back((kalman.H * state)(0));
    }

    auto result = undifference_sarima(w_out, kalman.d, kalman.D, kalman.m, kalman.z_tail, kalman.y_tail);
    return invert_transform_vec(std::move(result), kalman.transform);
  }

  std::vector<double> variance_arima_kalman(int horizon) const {
    // Propagate P_T: var(h) = σ² × H · (Σ_{j=1}^{h} F^j · P_T · (F^j)^T) · H^T
    if (kalman.P_T.rows() == 0) return std::vector<double>(horizon, kalman.sigma2);

    std::vector<double> vars;
    vars.reserve(horizon);
    Eigen::MatrixXd Fh = Eigen::MatrixXd::Identity(kalman.F.rows(), kalman.F.cols());
    double cumvar = 0.0;
    for (int h = 1; h <= horizon; ++h) {
      Fh = kalman.F * Fh;
      double step_var = (kalman.H * Fh * kalman.P_T * Fh.transpose() * kalman.H.transpose())(0, 0);
      cumvar += step_var * kalman.sigma2;
      vars.push_back(cumvar);
    }
    return vars;
  }

  // =========================================================================
  // Kalman filter — called once at load time
  //
  // Builds companion F/H, solves P₀ via Kronecker–Lyapunov, filters the
  // differenced training series, stores terminal x_T and P_T.
  // =========================================================================

  void run_kalman_filter(const std::vector<double>& phi, const std::vector<double>& theta, int d, int D, int m,
                         const std::vector<double>& y_raw_chron) {
    kalman.d = d;
    kalman.D = D;
    kalman.m = m;

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

    // --- Build companion matrices ---
    int p = static_cast<int>(phi.size());
    int q = static_cast<int>(theta.size());
    int n = std::max(p, q + 1);
    if (n < 1) n = 1;

    // Transition F [n×n]: top row = AR coefficients, sub-diagonal = shift
    kalman.F = Eigen::MatrixXd::Zero(n, n);
    for (int j = 0; j < p && j < n; ++j) kalman.F(0, j) = phi[j];
    for (int i = 1; i < n; ++i) kalman.F(i, i - 1) = 1.0;

    // Noise vector G_vec [n×1]: [1, θ₁, ..., θ_{n-1}]ᵀ
    Eigen::VectorXd G_vec = Eigen::VectorXd::Zero(n);
    G_vec(0) = 1.0;
    for (int j = 0; j < q && j + 1 < n; ++j) G_vec(j + 1) = theta[j];

    // Observation H [1×n]: [1, 0, ..., 0]
    kalman.H = Eigen::RowVectorXd::Zero(n);
    kalman.H(0) = 1.0;

    // Process noise Q = G_vec · G_vecᵀ  (σ²=1; cancels in point-forecast gain)
    Eigen::MatrixXd Q = G_vec * G_vec.transpose();

    // --- Solve discrete Lyapunov P = F·P·Fᵀ + Q ---
    // Vectorized: (I_{n²} − F⊗F) · vec(P) = vec(Q)
    // colPivHouseholderQr handles near-unit-root cases robustly.
    {
      Eigen::MatrixXd IFF =
          Eigen::MatrixXd::Identity(n * n, n * n) - Eigen::kroneckerProduct(kalman.F, kalman.F).eval();

      Eigen::Map<const Eigen::VectorXd> vecQ(Q.data(), n * n);
      Eigen::VectorXd vecP = IFF.colPivHouseholderQr().solve(vecQ);

      Eigen::MatrixXd P0 = Eigen::Map<Eigen::MatrixXd>(vecP.data(), n, n);
      P0 = (P0 + P0.transpose()) * 0.5;  // symmetrize

      // --- Kalman filter through training data ---
      kalman.x_T = Eigen::VectorXd::Zero(n);
      Eigen::MatrixXd P = P0;
      const Eigen::MatrixXd In = Eigen::MatrixXd::Identity(n, n);

      for (double y_obs : y_series) {
        // Predict
        Eigen::VectorXd x_pred = kalman.F * kalman.x_T;
        Eigen::MatrixXd P_pred = kalman.F * P * kalman.F.transpose() + Q;

        // Innovation (scalar)
        double v = y_obs - (kalman.H * x_pred)(0);
        double S = (kalman.H * P_pred * kalman.H.transpose())(0, 0);
        if (S < 1e-12) S = 1e-12;

        // Kalman gain [n×1]
        Eigen::VectorXd K = P_pred * kalman.H.transpose() / S;

        // Joseph form update (maintains positive semi-definiteness)
        Eigen::MatrixXd IKH = In - K * kalman.H;
        kalman.x_T = x_pred + K * v;
        P = IKH * P_pred * IKH.transpose() + K * (S * K.transpose());
        P = (P + P.transpose()) * 0.5;
      }
      kalman.P_T = P;
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
    // Step 1: undo d regular differences → z_fcast
    std::vector<double> z_fcast;
    if (d == 0) {
      z_fcast = w_fcast;
    } else {
      z_fcast.reserve(w_fcast.size());
      double prev = z_tail.empty() ? 0.0 : z_tail.back();
      for (double w : w_fcast) {
        prev += w;
        z_fcast.push_back(prev);
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

  void parse_ssm(const XmlNode& node) {
    auto load_matrix = [](const XmlNode& m, int rows, int cols) {
      auto vals = parse_array(m);
      Eigen::MatrixXd mat(rows, cols);
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) mat(r, c) = vals[static_cast<size_t>(r * cols + c)];
      return mat;
    };

    if (node.exists_attribute("transformation"))
      ssm.transform = parse_transform(to_lower(node.get_attribute("transformation")));
    if (node.exists_attribute("intercept")) ssm.intercept = node.get_double_attribute("intercept");
    if (node.exists_attribute("RMSE")) {
      double rmse = node.get_double_attribute("RMSE");
      ssm.sigma2 = rmse * rmse;
    }

    // PsiVector: MA(∞) weights ψ₀,ψ₁,... for exact h-step variance
    // var(h) = σ² × Σ_{j=0}^{h-1} ψ_j²   (PMML 4.4 spec §TimeSeriesModel)
    if (node.exists_child("PsiVector")) {
      XmlNode pv = node.get_child("PsiVector");
      // @variance on PsiVector overrides RMSE-derived sigma2
      if (pv.exists_attribute("variance")) ssm.sigma2 = pv.get_double_attribute("variance");
      if (pv.exists_child("Array")) ssm.psi = parse_array(pv.get_child("Array"));
    }

    XmlNode sv = node.get_child("StateVector");
    int n = static_cast<int>(sv.get_long_attribute("n"));
    auto vals = parse_array(sv);
    ssm.x.resize(n);
    for (int i = 0; i < n; ++i) ssm.x[i] = vals[static_cast<size_t>(i)];

    XmlNode tm = node.get_child("TransitionMatrix");
    ssm.F = load_matrix(tm, static_cast<int>(tm.get_long_attribute("rows")),
                        static_cast<int>(tm.get_long_attribute("cols")));

    XmlNode mm = node.get_child("MeasurementMatrix");
    ssm.G = load_matrix(mm, static_cast<int>(mm.get_long_attribute("rows")),
                        static_cast<int>(mm.get_long_attribute("cols")));
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
      bool have_state = false;
      if (ts_node.exists_child("MaximumLikelihoodStat")) {
        XmlNode mls = ts_node.get_child("MaximumLikelihoodStat");
        if (mls.exists_child("KalmanState")) {
          XmlNode ks = mls.get_child("KalmanState");
          if (ks.exists_child("FinalStateVector")) {
            XmlNode fsv_node = ks.get_child("FinalStateVector");
            if (fsv_node.exists_child("Array")) {
              auto fsv = parse_array(fsv_node.get_child("Array"));
              if (!fsv.empty()) {
                int nc = static_cast<int>(fsv.size());
                kalman.x_T.resize(nc);
                for (int i = 0; i < nc; ++i) kalman.x_T[i] = fsv[i];

                // Rebuild F and H from expanded phi/theta
                int p_eff = static_cast<int>(phi.size());
                int q_eff = static_cast<int>(theta.size());
                int n_c = std::max(p_eff, q_eff + 1);
                if (n_c < 1) n_c = 1;
                kalman.F = Eigen::MatrixXd::Zero(n_c, n_c);
                for (int j = 0; j < p_eff && j < n_c; ++j) kalman.F(0, j) = phi[j];
                for (int i = 1; i < n_c; ++i) kalman.F(i, i - 1) = 1.0;
                kalman.H = Eigen::RowVectorXd::Zero(n_c);
                kalman.H(0) = 1.0;

                // Set metadata
                kalman.d = d;
                kalman.D = D;
                kalman.m = m;
                kalman.transform = tr;
                kalman.sigma2 = sigma2_val;
                // P_T left empty — variance will fall back to constant sigma2

                // Build y_tail and z_tail for undifferencing
                setup_arima_tails(pts, d, D, m, kalman.y_tail, kalman.z_tail);

                algorithm = Algorithm::ARIMA_KALMAN;
                have_state = true;
              }
            }
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
        run_kalman_filter(phi, theta, d, D, m, y_raw_chron);
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
      // ARCH terms αᵢ × E[ε²(T+h-i)], i=1..gp
      for (int i = 1; i <= g.gp; ++i) {
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
      // GARCH terms βⱼ × σ²(T+h-j), j=1..gq
      for (int j = 1; j <= g.gq; ++j) {
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

    if (garch.alpha.empty()) garch.alpha.assign(garch.gp, 0.0);
    if (garch.beta_garch.empty()) garch.beta_garch.assign(garch.gq, 0.0);
    if (garch.eps_sq.empty()) garch.eps_sq.assign(garch.gp, 0.0);
    if (garch.sigma2_hist.empty()) garch.sigma2_hist.assign(garch.gq, 0.0);
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
