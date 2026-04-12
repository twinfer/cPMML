
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TIMESERIESEVALUATOR_H
#define CPMML_TIMESERIESEVALUATOR_H

#include <vector>

#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "timeseriesmodel.h"

/**
 * @class TimeSeriesEvaluator
 *
 * Implementation of InternalEvaluator wrapping TimeSeriesModel.
 *
 * evaluate() expects:
 *   - "horizon" (int or string) — required, number of forecast steps
 *   - "_variance" (string "true" or int 1) — optional, include variance estimates
 *   - Any vector<double> entry — treated as a regressor future-values series
 */
class TimeSeriesEvaluator : public InternalEvaluator {
 public:
  explicit TimeSeriesEvaluator(const XmlNode& node)
      : InternalEvaluator(node), ts_model(node.get_child("TimeSeriesModel")) {}

  TimeSeriesModel ts_model;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    // Extract horizon (required)
    int horizon = 0;
    if (auto it = arguments.find("horizon"); it != arguments.end()) {
      if (std::holds_alternative<int>(it->second))
        horizon = std::get<int>(it->second);
      else if (std::holds_alternative<std::string>(it->second))
        horizon = std::stoi(std::get<std::string>(it->second));
    }
    if (horizon <= 0)
      throw cpmml::ParsingException("TimeSeriesModel requires 'horizon' > 0 in evaluate() arguments");

    // Check for "_variance" flag
    bool with_variance = false;
    if (auto it = arguments.find("_variance"); it != arguments.end()) {
      if (std::holds_alternative<std::string>(it->second))
        with_variance = (std::get<std::string>(it->second) == "true");
      else if (std::holds_alternative<int>(it->second))
        with_variance = (std::get<int>(it->second) != 0);
    }

    // Extract regressors (any key that is a vector<double>)
    std::unordered_map<std::string, std::vector<double>> regressors;
    for (const auto& [k, v] : arguments) {
      if (k == "horizon" || k == "_variance") continue;
      if (std::holds_alternative<std::vector<double>>(v))
        regressors[k] = std::get<std::vector<double>>(v);
    }

    auto result = std::make_unique<InternalScore>();
    result->empty = false;

    if (with_variance) {
      result->forecast_with_variance_values = ts_model.forecast_with_variance(horizon, regressors);
      for (const auto& [pt, var] : result->forecast_with_variance_values)
        result->forecast_values.push_back(pt);
    } else {
      result->forecast_values = ts_model.forecast(horizon, regressors);
    }

    return result;
  }

  inline bool validate(const Input&) const override {
    return true;  // no sample input to validate
  }

  inline std::string mining_function_name() const override { return "TIMESERIES"; }
};

#endif
