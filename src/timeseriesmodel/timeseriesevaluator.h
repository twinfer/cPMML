
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
 * score() and predict() throw ParsingException — time series models do not
 * score individual samples; use model.forecast(horizon) instead.
 */
class TimeSeriesEvaluator : public InternalEvaluator {
 public:
  explicit TimeSeriesEvaluator(const XmlNode &node)
      : InternalEvaluator(node),
        ts_model(node.get_child("TimeSeriesModel")) {}

  TimeSeriesModel ts_model;

  inline bool validate(const std::unordered_map<std::string, std::string> &) override {
    return true;  // no sample input to validate
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string> &) const override {
    throw cpmml::ParsingException(
        "TimeSeriesModel does not support score(); use model.forecast(horizon)");
  }

  inline std::string predict(
      const std::unordered_map<std::string, std::string> &) const override {
    throw cpmml::ParsingException(
        "TimeSeriesModel does not support predict(); use model.forecast(horizon)");
  }

  inline std::vector<double> forecast(int horizon) const override {
    return ts_model.forecast(horizon);
  }
};

#endif
