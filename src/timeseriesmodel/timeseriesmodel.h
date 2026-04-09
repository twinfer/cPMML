
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

#include "cPMML.h"
#include "core/xmlnode.h"
#include "utils/utils.h"

/**
 * @class TimeSeriesModel
 *
 * Standalone (non-InternalModel) implementation of
 * <a href="http://dmg.org/pmml/v4-4/TimeSeriesModel.html">PMML TimeSeriesModel</a>.
 *
 * Supported algorithms (bestFit attribute):
 *   ExponentialSmoothing — ETS models (additive/damped/multiplicative trend ×
 *                          none/additive/multiplicative seasonality)
 *   StateSpaceModel      — explicit F/G/x matrices; h-step = G · Fʰ · x
 *   ARIMA                — conditional least squares (p,d,q) only
 *
 * The model state is fully embedded in the PMML file; no input sample is
 * needed.  Call forecast(horizon) to obtain a vector of h-step-ahead values.
 */
class TimeSeriesModel {
 public:
  enum class Algorithm { ETS, SSM, ARIMA_CLS };
  Algorithm algorithm = Algorithm::ETS;

  // -------------------------------------------------------------------------
  // ETS (ExponentialSmoothing) state
  // -------------------------------------------------------------------------
  struct ETSState {
    // Trend type as parsed from the PMML trend="" attribute
    std::string trend    = "none";   // none | additive | damped_additive
                                     //       multiplicative | damped_multiplicative
    std::string seasonal = "none";   // none | additive | multiplicative

    double alpha = 0.0;  // level smoothing
    double gamma = 0.0;  // trend smoothing
    double delta = 0.0;  // seasonal smoothing
    double phi   = 1.0;  // damping factor (damped variants only)

    double l = 0.0;  // last smoothed level
    double b = 0.0;  // last smoothed trend (0 when trend == "none")
    int    period = 1;

    // Seasonal components in PMML chronological order:
    //   s[0] = s(T-m+1), ..., s[m-1] = s(T)
    // For h-step ahead: s(T+h) = s[(h-1) % m]  (same cyclic position)
    std::vector<double> s;
  } ets;

  // -------------------------------------------------------------------------
  // StateSpaceModel state
  // -------------------------------------------------------------------------
  struct SSMState {
    Eigen::VectorXd x;  // state vector  (p × 1)
    Eigen::MatrixXd F;  // transition    (p × p)
    Eigen::MatrixXd G;  // measurement   (m × p)
  } ssm;

  // -------------------------------------------------------------------------
  // ARIMA (conditional least squares) state
  // -------------------------------------------------------------------------
  struct ARIMAState {
    int p = 0, d = 0, q = 0;
    double  constant         = 0.0;
    bool    include_constant = false;

    std::vector<double> phi;    // AR coefficients  φ₁ … φₚ
    std::vector<double> theta;  // MA coefficients  θ₁ … θ_q

    // y_diff[0] = y'(T), y_diff[1] = y'(T-1), ...
    // where y'(t) is the d-times-differenced series.
    std::vector<double> y_diff;

    // eps[0] = ε(T), eps[1] = ε(T-1), ...  (last q residuals; default 0)
    std::vector<double> eps;

    // Raw values for undifferencing (d > 0): raw[0] = y(T), raw[1] = y(T-1)
    std::vector<double> y_raw;
  } arima;

  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  TimeSeriesModel() = default;

  /**
   * @param ts_node  The <TimeSeriesModel> XmlNode.
   */
  explicit TimeSeriesModel(const XmlNode &ts_node) {
    std::string best_fit = ts_node.get_attribute("bestFit");
    std::string bf_lower = to_lower(best_fit);

    if (bf_lower == "exponentialsmoothing") {
      algorithm = Algorithm::ETS;
      parse_ets(ts_node.get_child("ExponentialSmoothing"));
    } else if (bf_lower == "statespacemodel") {
      algorithm = Algorithm::SSM;
      parse_ssm(ts_node.get_child("StateSpaceModel"));
    } else if (bf_lower == "arima") {
      algorithm = Algorithm::ARIMA_CLS;
      parse_arima(ts_node.get_child("ARIMA"), ts_node);
    } else {
      throw cpmml::ParsingException("TimeSeriesModel: unsupported bestFit=\"" + best_fit +
                                    "\"; supported: ExponentialSmoothing, StateSpaceModel, ARIMA");
    }
  }

  // -------------------------------------------------------------------------
  // Forecast
  // -------------------------------------------------------------------------

  std::vector<double> forecast(int horizon) const {
    if (horizon <= 0) return {};
    switch (algorithm) {
      case Algorithm::ETS:       return forecast_ets(horizon);
      case Algorithm::SSM:       return forecast_ssm(horizon);
      case Algorithm::ARIMA_CLS: return forecast_arima(horizon);
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
    bool has_seas = e.seasonal != "none" && !e.s.empty();
    bool mult_trend = (e.trend == "multiplicative" || e.trend == "damped_multiplicative");

    std::vector<double> out;
    out.reserve(horizon);

    for (int h = 1; h <= horizon; ++h) {
      // ---- trend component ----
      double trend_add  = 0.0;  // for additive/none trend: base = l + trend_add
      double trend_mult = 1.0;  // for multiplicative trend: base = l * trend_mult

      if (e.trend == "additive") {
        trend_add = static_cast<double>(h) * e.b;
      } else if (e.trend == "damped_additive") {
        double phi_sum = 0.0, phi_pow = e.phi;
        for (int i = 0; i < h; ++i) { phi_sum += phi_pow; phi_pow *= e.phi; }
        trend_add = phi_sum * e.b;
      } else if (e.trend == "multiplicative") {
        trend_mult = std::pow(e.b, static_cast<double>(h));
      } else if (e.trend == "damped_multiplicative") {
        double phi_sum = 0.0, phi_pow = e.phi;
        for (int i = 0; i < h; ++i) { phi_sum += phi_pow; phi_pow *= e.phi; }
        trend_mult = std::pow(e.b, phi_sum);
      }
      // trend == "none": trend_add = 0, trend_mult = 1 → base = l

      double base = mult_trend ? (e.l * trend_mult) : (e.l + trend_add);

      // ---- seasonal component ----
      // s(T+h) = s(T+h-m) = s[(h-1) % m] given PMML stores [s(T-m+1),..,s(T)]
      double seas;
      if (!has_seas) {
        seas = (e.seasonal == "multiplicative") ? 1.0 : 0.0;
      } else {
        seas = e.s[static_cast<size_t>((h - 1) % m)];
      }

      // ---- combine ----
      double y_hat;
      if (e.seasonal == "none" || !has_seas)
        y_hat = base;
      else if (e.seasonal == "additive")
        y_hat = base + seas;
      else  // multiplicative
        y_hat = base * seas;

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

  std::vector<double> forecast_arima(int horizon) const {
    const auto &a = arima;

    // Working buffers (index 0 = most recent)
    std::vector<double> y(a.y_diff);   // y'(T), y'(T-1), ...
    std::vector<double> eps(a.eps);    // ε(T), ε(T-1), ...

    std::vector<double> y_diff_out;
    y_diff_out.reserve(horizon);

    for (int h = 1; h <= horizon; ++h) {
      double y_hat = a.include_constant ? a.constant : 0.0;

      // AR part
      for (int i = 0; i < a.p && i < static_cast<int>(y.size()); ++i)
        y_hat += a.phi[i] * y[i];

      // MA part (future residuals assumed 0 for conditional LS)
      for (int j = 0; j < a.q && j < static_cast<int>(eps.size()); ++j)
        y_hat += a.theta[j] * eps[j];

      y_diff_out.push_back(y_hat);

      // Advance history
      y.insert(y.begin(), y_hat);
      eps.insert(eps.begin(), 0.0);
    }

    // Undo differencing
    if (a.d == 0) return y_diff_out;

    // d == 1: integrate from last raw observation y(T)
    std::vector<double> out;
    out.reserve(horizon);
    double last = a.y_raw.empty() ? 0.0 : a.y_raw[0];
    for (double dy : y_diff_out) {
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
    auto parts = split(node.value(), " ");
    for (const auto &p : parts)
      if (!p.empty()) v.push_back(to_double(p));
    return v;
  }

  void parse_ets(const XmlNode &node) {
    ets.trend    = node.exists_attribute("trend")      ? to_lower(node.get_attribute("trend"))      : "none";
    ets.seasonal = node.exists_attribute("seasonality") ? to_lower(node.get_attribute("seasonality")) : "none";

    // Level (required)
    XmlNode lv = node.get_child("Level");
    ets.alpha = lv.get_double_attribute("alpha");
    ets.l     = lv.get_double_attribute("smoothedValue");

    // Trend (optional)
    ets.b   = 0.0;
    ets.phi = 1.0;
    if (ets.trend != "none" && node.exists_child("Trend")) {
      XmlNode tr = node.get_child("Trend");
      ets.gamma = tr.exists_attribute("gamma") ? tr.get_double_attribute("gamma") : 0.0;
      ets.phi   = tr.exists_attribute("phi")   ? tr.get_double_attribute("phi")   : 1.0;
      ets.b     = tr.get_double_attribute("smoothedValue");
    }

    // Seasonality (optional)
    if (ets.seasonal != "none" && node.exists_child("Seasonality")) {
      XmlNode se = node.get_child("Seasonality");
      ets.delta  = se.exists_attribute("delta")  ? se.get_double_attribute("delta") : 0.0;
      ets.period = static_cast<int>(se.get_long_attribute("period"));
      if (se.exists_child("Array"))
        ets.s = parse_array(se.get_child("Array"));
    }
  }

  void parse_ssm(const XmlNode &node) {
    // StateVector
    {
      XmlNode sv = node.get_child("StateVector");
      int n = static_cast<int>(sv.get_long_attribute("n"));
      auto vals = parse_array(sv);
      ssm.x.resize(n);
      for (int i = 0; i < n; ++i) ssm.x[i] = vals[i];
    }
    // TransitionMatrix
    {
      XmlNode tm = node.get_child("TransitionMatrix");
      int rows = static_cast<int>(tm.get_long_attribute("rows"));
      int cols = static_cast<int>(tm.get_long_attribute("cols"));
      auto vals = parse_array(tm);
      ssm.F.resize(rows, cols);
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
          ssm.F(r, c) = vals[r * cols + c];
    }
    // MeasurementMatrix
    {
      XmlNode mm = node.get_child("MeasurementMatrix");
      int rows = static_cast<int>(mm.get_long_attribute("rows"));
      int cols = static_cast<int>(mm.get_long_attribute("cols"));
      auto vals = parse_array(mm);
      ssm.G.resize(rows, cols);
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
          ssm.G(r, c) = vals[r * cols + c];
    }
  }

  void parse_arima(const XmlNode &node, const XmlNode &ts_node) {
    arima.p = node.exists_attribute("p") ? static_cast<int>(node.get_long_attribute("p")) : 0;
    arima.d = node.exists_attribute("d") ? static_cast<int>(node.get_long_attribute("d")) : 0;
    arima.q = node.exists_attribute("q") ? static_cast<int>(node.get_long_attribute("q")) : 0;
    arima.constant         = node.exists_attribute("constant")         ? node.get_double_attribute("constant")      : 0.0;
    arima.include_constant = node.exists_attribute("includeConstant")  ? node.get_bool_attribute("includeConstant") : false;

    // Coefficients from <NonseasonalComponent>
    if (node.exists_child("NonseasonalComponent")) {
      XmlNode ns = node.get_child("NonseasonalComponent");
      if (ns.exists_child("AR") && ns.get_child("AR").exists_child("Array"))
        arima.phi   = parse_array(ns.get_child("AR").get_child("Array"));
      if (ns.exists_child("MA") && ns.get_child("MA").exists_child("Array"))
        arima.theta = parse_array(ns.get_child("MA").get_child("Array"));
    }

    // History from <TimeSeries usage="original"> TimeValue elements
    // Collect all (index, value) pairs, sort descending (most recent first)
    std::vector<std::pair<int, double>> orig;
    for (const auto &ts : ts_node.get_childs("TimeSeries")) {
      if (ts.exists_attribute("usage") && to_lower(ts.get_attribute("usage")) != "original")
        continue;
      for (const auto &tv : ts.get_childs("TimeValue")) {
        int    idx = static_cast<int>(tv.get_long_attribute("index"));
        double val = tv.get_double_attribute("value");
        orig.push_back({idx, val});
      }
    }
    std::sort(orig.begin(), orig.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    // Last d raw values (for undifferencing)
    for (int i = 0; i < arima.d && i < static_cast<int>(orig.size()); ++i)
      arima.y_raw.push_back(orig[i].second);

    // Build differenced history
    if (arima.d == 0) {
      for (int i = 0; i < arima.p && i < static_cast<int>(orig.size()); ++i)
        arima.y_diff.push_back(orig[i].second);
    } else if (arima.d == 1) {
      // y'(t) = y(t) - y(t-1); need p+1 raw values for p differenced values
      int need = arima.p + 1;
      for (int i = 0; i < need - 1 && i + 1 < static_cast<int>(orig.size()); ++i)
        arima.y_diff.push_back(orig[i].second - orig[i + 1].second);
    }

    // Default residuals to 0
    arima.eps.assign(arima.q, 0.0);
  }
};

#endif
