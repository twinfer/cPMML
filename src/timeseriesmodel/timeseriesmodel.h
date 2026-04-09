
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TIMESERIESMODEL_H
#define CPMML_TIMESERIESMODEL_H

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

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
 *   StateSpaceModel       — explicit F/G/x matrices; ŷ(T+h) = G · Fʰ · x
 *   ARIMA conditionalLS   — scalar AR/MA recursion (default)
 *   ARIMA exactLS         — Kalman filter on companion state-space form;
 *                           P₀ solved via Kronecker–Lyapunov equation
 *
 * Call forecast(horizon) to obtain a vector of h-step-ahead values.
 */
class TimeSeriesModel {
 public:
  enum class Algorithm { ETS, SSM, ARIMA_CLS, ARIMA_KALMAN };
  Algorithm algorithm = Algorithm::ETS;

  // -------------------------------------------------------------------------
  // ETS state
  // -------------------------------------------------------------------------
  struct ETSState {
    std::string trend    = "none";  // none|additive|damped_additive|multiplicative|damped_multiplicative
    std::string seasonal = "none";  // none|additive|multiplicative
    double alpha = 0.0, gamma = 0.0, delta = 0.0, phi = 1.0;
    double l = 0.0, b = 0.0;
    int    period = 1;
    std::vector<double> s;          // seasonal components [s(T-m+1),..,s(T)]
  } ets;

  // -------------------------------------------------------------------------
  // SSM state
  // -------------------------------------------------------------------------
  struct SSMState {
    Eigen::VectorXd x;   // state  (n×1)
    Eigen::MatrixXd F;   // transition (n×n)
    Eigen::MatrixXd G;   // measurement (m×n)
  } ssm;

  // -------------------------------------------------------------------------
  // ARIMA conditional LS state
  // -------------------------------------------------------------------------
  struct ARIMAState {
    int p = 0, d = 0, q = 0;
    double constant = 0.0;
    bool   include_constant = false;
    std::vector<double> phi, theta;
    std::vector<double> y_diff;   // last p differenced obs (most-recent first)
    std::vector<double> eps;      // last q residuals
    std::vector<double> y_raw;    // last d raw obs for undifferencing
  } arima;

  // -------------------------------------------------------------------------
  // ARIMA Kalman (exact LS) state — built at load time
  // -------------------------------------------------------------------------
  struct KalmanState {
    int d = 0;
    std::vector<double> y_raw_last;  // y(T) needed to undifference (size d)
    Eigen::MatrixXd      F;          // companion (n×n)
    Eigen::RowVectorXd   H;          // observation (1×n)
    Eigen::VectorXd      x_T;        // terminal state after filtering (n×1)
  } kalman;

  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  TimeSeriesModel() = default;

  explicit TimeSeriesModel(const XmlNode &ts_node) {
    std::string bf = to_lower(ts_node.get_attribute("bestFit"));
    if (bf == "exponentialsmoothing") {
      algorithm = Algorithm::ETS;
      parse_ets(ts_node.get_child("ExponentialSmoothing"));
    } else if (bf == "statespacemodel") {
      algorithm = Algorithm::SSM;
      parse_ssm(ts_node.get_child("StateSpaceModel"));
    } else if (bf == "arima") {
      parse_arima(ts_node.get_child("ARIMA"), ts_node);  // sets algorithm internally
    } else {
      throw cpmml::ParsingException(
          "TimeSeriesModel: unsupported bestFit=\"" + ts_node.get_attribute("bestFit") +
          "\"; supported: ExponentialSmoothing, StateSpaceModel, ARIMA");
    }
  }

  // -------------------------------------------------------------------------
  // Forecast
  // -------------------------------------------------------------------------

  std::vector<double> forecast(int horizon) const {
    if (horizon <= 0) return {};
    switch (algorithm) {
      case Algorithm::ETS:          return forecast_ets(horizon);
      case Algorithm::SSM:          return forecast_ssm(horizon);
      case Algorithm::ARIMA_CLS:    return forecast_arima_cls(horizon);
      case Algorithm::ARIMA_KALMAN: return forecast_arima_kalman(horizon);
    }
    return {};
  }

 private:
  // =========================================================================
  // ETS forecasting
  // =========================================================================

  std::vector<double> forecast_ets(int horizon) const {
    const auto &e = ets;
    int m = e.period > 0 ? e.period : 1;
    bool has_seas  = e.seasonal != "none" && !e.s.empty();
    bool mult_trend = (e.trend == "multiplicative" || e.trend == "damped_multiplicative");

    std::vector<double> out;
    out.reserve(horizon);

    for (int h = 1; h <= horizon; ++h) {
      double trend_add = 0.0, trend_mult = 1.0;
      if (e.trend == "additive") {
        trend_add = static_cast<double>(h) * e.b;
      } else if (e.trend == "damped_additive") {
        double ps = 0.0, pp = e.phi;
        for (int i = 0; i < h; ++i) { ps += pp; pp *= e.phi; }
        trend_add = ps * e.b;
      } else if (e.trend == "multiplicative") {
        trend_mult = std::pow(e.b, static_cast<double>(h));
      } else if (e.trend == "damped_multiplicative") {
        double ps = 0.0, pp = e.phi;
        for (int i = 0; i < h; ++i) { ps += pp; pp *= e.phi; }
        trend_mult = std::pow(e.b, ps);
      }

      double base = mult_trend ? (e.l * trend_mult) : (e.l + trend_add);

      double seas = has_seas ? e.s[static_cast<size_t>((h - 1) % m)]
                             : (e.seasonal == "multiplicative" ? 1.0 : 0.0);

      double y_hat;
      if (!has_seas || e.seasonal == "none") y_hat = base;
      else if (e.seasonal == "additive")     y_hat = base + seas;
      else                                   y_hat = base * seas;

      out.push_back(y_hat);
    }
    return out;
  }

  // =========================================================================
  // SSM forecasting: ŷ(T+h) = G · Fʰ · x
  // =========================================================================

  std::vector<double> forecast_ssm(int horizon) const {
    std::vector<double> out;
    out.reserve(horizon);
    Eigen::VectorXd state = ssm.x;
    for (int h = 1; h <= horizon; ++h) {
      state = ssm.F * state;
      out.push_back((ssm.G * state)(0, 0));
    }
    return out;
  }

  // =========================================================================
  // ARIMA conditional LS forecasting
  // =========================================================================

  std::vector<double> forecast_arima_cls(int horizon) const {
    const auto &a = arima;
    std::vector<double> y(a.y_diff), eps(a.eps);
    std::vector<double> y_diff_out;
    y_diff_out.reserve(horizon);

    for (int h = 1; h <= horizon; ++h) {
      double yh = a.include_constant ? a.constant : 0.0;
      for (int i = 0; i < a.p && i < static_cast<int>(y.size()); ++i)
        yh += a.phi[i] * y[i];
      for (int j = 0; j < a.q && j < static_cast<int>(eps.size()); ++j)
        yh += a.theta[j] * eps[j];
      y_diff_out.push_back(yh);
      y.insert(y.begin(), yh);
      eps.insert(eps.begin(), 0.0);
    }
    return undifference(y_diff_out, a.d, a.y_raw);
  }

  // =========================================================================
  // ARIMA Kalman (exact LS) forecasting
  //
  // Point forecast is identical to SSM after filtering: ŷ'(T+h) = H · Fʰ · x_T
  // Then undo d-times differencing.
  // =========================================================================

  std::vector<double> forecast_arima_kalman(int horizon) const {
    std::vector<double> y_diff_out;
    y_diff_out.reserve(horizon);

    Eigen::VectorXd state = kalman.x_T;
    for (int h = 1; h <= horizon; ++h) {
      state = kalman.F * state;
      y_diff_out.push_back((kalman.H * state)(0));
    }
    return undifference(y_diff_out, kalman.d, kalman.y_raw_last);
  }

  // =========================================================================
  // Kalman filter — called once at load time
  //
  // Builds companion F/G/H, solves P₀ via Kronecker–Lyapunov, then filters
  // the training series to produce the terminal state x_T.
  // =========================================================================

  void run_kalman_filter(const std::vector<double> &phi,
                         const std::vector<double> &theta,
                         int p, int d, int q,
                         const std::vector<double> &y_raw_chron) {
    // ---- store undifferencing info ----
    kalman.d = d;
    if (!y_raw_chron.empty())
      kalman.y_raw_last.push_back(y_raw_chron.back());  // y(T)

    // ---- difference the training series ----
    std::vector<double> y_series = y_raw_chron;
    for (int di = 0; di < d; ++di) {
      std::vector<double> tmp;
      tmp.reserve(y_series.size() > 0 ? y_series.size() - 1 : 0);
      for (size_t i = 1; i < y_series.size(); ++i)
        tmp.push_back(y_series[i] - y_series[i - 1]);
      y_series = std::move(tmp);
    }
    if (y_series.empty())
      throw cpmml::ParsingException(
          "TimeSeriesModel: ARIMA exactLeastSquares requires TimeSeries training data");

    // ---- build companion matrices ----
    int n = std::max(p, q + 1);
    if (n < 1) n = 1;

    // Transition F [n×n]: top row = AR coefficients (zero-padded), sub-diagonal = shift
    kalman.F = Eigen::MatrixXd::Zero(n, n);
    for (int j = 0; j < p && j < n; ++j)
      kalman.F(0, j) = phi[j];
    for (int i = 1; i < n; ++i)
      kalman.F(i, i - 1) = 1.0;

    // Noise vector G [n×1]: [1, θ₁, θ₂, ..., θ_{n-1}]ᵀ
    Eigen::VectorXd G = Eigen::VectorXd::Zero(n);
    G(0) = 1.0;
    for (int j = 0; j < q && j + 1 < n; ++j)
      G(j + 1) = theta[j];

    // Observation H [1×n]: [1, 0, ..., 0]
    kalman.H = Eigen::RowVectorXd::Zero(n);
    kalman.H(0) = 1.0;

    // Process noise covariance Q = G·Gᵀ  (σ²=1; cancels in point-forecast gain)
    Eigen::MatrixXd Q = G * G.transpose();

    // ---- solve discrete Lyapunov P = F·P·Fᵀ + Q ----
    // Vectorized: (I_{n²} - F⊗F)·vec(P) = vec(Q)
    //
    // colPivHouseholderQr handles:
    //  - Full-rank case (stationary ARMA after differencing)
    //  - Rank-deficient case (would occur for unit-root F, not present here
    //    since we operate on the already-differenced series, but kept for
    //    robustness against near-unit-root processes)
    {
      Eigen::MatrixXd IFF =
          Eigen::MatrixXd::Identity(n * n, n * n) -
          Eigen::kroneckerProduct(kalman.F, kalman.F).eval();

      Eigen::Map<const Eigen::VectorXd> vecQ(Q.data(), n * n);
      Eigen::VectorXd vecP = IFF.colPivHouseholderQr().solve(vecQ);

      Eigen::MatrixXd P0 = Eigen::Map<Eigen::MatrixXd>(vecP.data(), n, n);
      P0 = (P0 + P0.transpose()) * 0.5;  // symmetrize

      // ---- Kalman filter through training data ----
      kalman.x_T = Eigen::VectorXd::Zero(n);
      Eigen::MatrixXd P = P0;
      const Eigen::MatrixXd In = Eigen::MatrixXd::Identity(n, n);

      for (double y_obs : y_series) {
        // Predict
        Eigen::VectorXd x_pred = kalman.F * kalman.x_T;
        Eigen::MatrixXd P_pred = kalman.F * P * kalman.F.transpose() + Q;

        // Innovation (scalar for univariate)
        double v = y_obs - (kalman.H * x_pred)(0);
        double S = (kalman.H * P_pred * kalman.H.transpose())(0, 0);
        if (S < 1e-12) S = 1e-12;  // numerical guard

        // Kalman gain [n×1]
        Eigen::VectorXd K = P_pred * kalman.H.transpose() / S;

        // Update — Joseph form: (I-KH)·P·(I-KH)ᵀ + K·S·Kᵀ
        // Maintains positive semi-definiteness under floating-point errors.
        Eigen::MatrixXd IKH = In - K * kalman.H;
        kalman.x_T = x_pred + K * v;
        P = IKH * P_pred * IKH.transpose() + K * (S * K.transpose());
        P = (P + P.transpose()) * 0.5;  // keep symmetric
      }
    }
  }

  // =========================================================================
  // Shared undifferencing (used by both CLS and Kalman)
  // =========================================================================

  static std::vector<double> undifference(const std::vector<double> &y_diff,
                                          int d,
                                          const std::vector<double> &y_raw_last) {
    if (d == 0) return y_diff;

    // d == 1: cumulative sum from last raw observation
    std::vector<double> out;
    out.reserve(y_diff.size());
    double last = y_raw_last.empty() ? 0.0 : y_raw_last[0];
    for (double dy : y_diff) {
      last += dy;
      out.push_back(last);
    }
    return out;
  }

  // =========================================================================
  // Parsing helpers
  // =========================================================================

  static std::vector<double> parse_array(const XmlNode &node) {
    std::vector<double> v;
    for (const auto &p : split(node.value(), " "))
      if (!p.empty()) v.push_back(to_double(p));
    return v;
  }

  // Collect all (index, value) pairs from <TimeSeries usage="original">
  static std::vector<std::pair<int, double>> collect_timeseries(const XmlNode &ts_node) {
    std::vector<std::pair<int, double>> pts;
    for (const auto &ts : ts_node.get_childs("TimeSeries")) {
      if (ts.exists_attribute("usage") &&
          to_lower(ts.get_attribute("usage")) != "original")
        continue;
      for (const auto &tv : ts.get_childs("TimeValue")) {
        int    idx = static_cast<int>(tv.get_long_attribute("index"));
        double val = tv.get_double_attribute("value");
        pts.push_back({idx, val});
      }
    }
    return pts;
  }

  void parse_ets(const XmlNode &node) {
    // Spec: trend type lives on Trend_ExpoSmooth/@trend (NOT ExponentialSmoothing/@trend)
    // Spec: seasonal type lives on Seasonality_ExpoSmooth/@type (NOT ExponentialSmoothing/@seasonality)
    ets.trend    = "none";
    ets.seasonal = "none";

    // Level (required by spec)
    XmlNode lv = node.get_child("Level");
    ets.alpha = lv.exists_attribute("alpha") ? lv.get_double_attribute("alpha") : 0.0;
    ets.l     = lv.get_double_attribute("smoothedValue");

    // Trend_ExpoSmooth (optional)
    ets.b = 0.0; ets.phi = 1.0; ets.gamma = 0.0;
    if (node.exists_child("Trend_ExpoSmooth")) {
      XmlNode tr = node.get_child("Trend_ExpoSmooth");
      ets.trend = tr.exists_attribute("trend") ? to_lower(tr.get_attribute("trend")) : "additive";
      ets.gamma = tr.exists_attribute("gamma") ? tr.get_double_attribute("gamma") : 0.0;
      ets.phi   = tr.exists_attribute("phi")   ? tr.get_double_attribute("phi")   : 1.0;
      ets.b     = tr.get_double_attribute("smoothedValue");
    }

    // Seasonality_ExpoSmooth (optional)
    // Seasonal components stored as Array child or as TimeValue children (index-ordered)
    if (node.exists_child("Seasonality_ExpoSmooth")) {
      XmlNode se = node.get_child("Seasonality_ExpoSmooth");
      ets.seasonal = se.exists_attribute("type") ? to_lower(se.get_attribute("type")) : "additive";
      ets.delta    = se.exists_attribute("delta") ? se.get_double_attribute("delta") : 0.0;
      ets.period   = static_cast<int>(se.get_long_attribute("period"));
      if (se.exists_child("Array")) {
        ets.s = parse_array(se.get_child("Array"));
      } else {
        // Fallback: TimeValue children (1-indexed by convention)
        auto tvs = se.get_childs("TimeValue");
        if (!tvs.empty()) {
          ets.s.resize(tvs.size(), 0.0);
          for (const auto &tv : tvs) {
            int idx = static_cast<int>(tv.get_long_attribute("index")) - 1;
            if (idx >= 0 && idx < static_cast<int>(ets.s.size()))
              ets.s[idx] = tv.get_double_attribute("value");
          }
        }
      }
    }
  }

  void parse_ssm(const XmlNode &node) {
    auto load_matrix = [](const XmlNode &m, int rows, int cols) {
      auto vals = parse_array(m);
      Eigen::MatrixXd mat(rows, cols);
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
          mat(r, c) = vals[r * cols + c];
      return mat;
    };

    XmlNode sv = node.get_child("StateVector");
    int n = static_cast<int>(sv.get_long_attribute("n"));
    auto vals = parse_array(sv);
    ssm.x.resize(n);
    for (int i = 0; i < n; ++i) ssm.x[i] = vals[i];

    XmlNode tm = node.get_child("TransitionMatrix");
    ssm.F = load_matrix(tm, static_cast<int>(tm.get_long_attribute("rows")),
                            static_cast<int>(tm.get_long_attribute("cols")));

    XmlNode mm = node.get_child("MeasurementMatrix");
    ssm.G = load_matrix(mm, static_cast<int>(mm.get_long_attribute("rows")),
                            static_cast<int>(mm.get_long_attribute("cols")));
  }

  void parse_arima(const XmlNode &node, const XmlNode &ts_node) {
    // Spec: p, d, q are attributes of NonseasonalComponent (NOT of ARIMA)
    // Spec: constantTerm (NOT constant) on ARIMA; predictionMethod on ARIMA
    int p = 0, d = 0, q = 0;
    std::vector<double> phi, theta, residuals;

    if (node.exists_child("NonseasonalComponent")) {
      XmlNode ns = node.get_child("NonseasonalComponent");
      p = ns.exists_attribute("p") ? static_cast<int>(ns.get_long_attribute("p")) : 0;
      d = ns.exists_attribute("d") ? static_cast<int>(ns.get_long_attribute("d")) : 0;
      q = ns.exists_attribute("q") ? static_cast<int>(ns.get_long_attribute("q")) : 0;

      if (ns.exists_child("AR") && ns.get_child("AR").exists_child("Array"))
        phi = parse_array(ns.get_child("AR").get_child("Array"));

      // Spec: MA/MACoefficients/Array for coefficients; MA/Residuals/Array for last q residuals
      if (ns.exists_child("MA")) {
        XmlNode ma = ns.get_child("MA");
        if (ma.exists_child("MACoefficients") &&
            ma.get_child("MACoefficients").exists_child("Array"))
          theta = parse_array(ma.get_child("MACoefficients").get_child("Array"));
        if (ma.exists_child("Residuals") &&
            ma.get_child("Residuals").exists_child("Array"))
          residuals = parse_array(ma.get_child("Residuals").get_child("Array"));
      }
    }

    // Spec: constantTerm attribute on ARIMA (default 0)
    double constant_val = node.exists_attribute("constantTerm")
                              ? node.get_double_attribute("constantTerm") : 0.0;
    bool include_const = (constant_val != 0.0);

    std::string method = "conditionalleastsquares";
    if (node.exists_attribute("predictionMethod"))
      method = to_lower(node.get_attribute("predictionMethod"));

    // Collect training series from <TimeSeries usage="original">, ascending order
    auto pts = collect_timeseries(ts_node);
    std::sort(pts.begin(), pts.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    if (method == "exactleastsquares" && !pts.empty()) {
      // ---- Kalman path ----
      algorithm = Algorithm::ARIMA_KALMAN;
      std::vector<double> y_raw_chron;
      y_raw_chron.reserve(pts.size());
      for (auto &[idx, val] : pts) y_raw_chron.push_back(val);
      run_kalman_filter(phi, theta, p, d, q, y_raw_chron);

    } else {
      // ---- Conditional LS path ----
      algorithm = Algorithm::ARIMA_CLS;
      arima.p = p; arima.d = d; arima.q = q;
      arima.phi   = phi;
      arima.theta = theta;
      arima.constant         = constant_val;
      arima.include_constant = include_const;

      // Sort descending (most-recent first) for CLS history
      std::reverse(pts.begin(), pts.end());
      for (int i = 0; i < d && i < static_cast<int>(pts.size()); ++i)
        arima.y_raw.push_back(pts[i].second);

      if (d == 0) {
        for (int i = 0; i < p && i < static_cast<int>(pts.size()); ++i)
          arima.y_diff.push_back(pts[i].second);
      } else if (d == 1) {
        for (int i = 0; i < p && i + 1 < static_cast<int>(pts.size()); ++i)
          arima.y_diff.push_back(pts[i].second - pts[i + 1].second);
      }

      // Use stored residuals from MA/Residuals if present, else default 0
      if (!residuals.empty())
        arima.eps = residuals;
      else
        arima.eps.assign(q, 0.0);
    }
  }
};

#endif
