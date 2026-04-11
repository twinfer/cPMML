
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#include "cPMML.h"
#include "core/internal_evaluator.h"
#include "core/internal_score.h"
#include "core/modelbuilder.h"
#include "utils/csvreader.h"
#include "utils/utils.h"

namespace cpmml {
Model::Model(const std::string& model_filepath) : evaluator(ModelBuilder::build(model_filepath, false)) {}

Model::Model(const std::string& model_filepath, const bool zipped = false)
    : evaluator(ModelBuilder::build(model_filepath, zipped)) {}

bool Model::validate(const std::unordered_map<std::string, std::string>& sample) const {
  return evaluator->validate(sample);
}

Prediction Model::score(const std::unordered_map<std::string, std::string>& sample) const {
  return Prediction(evaluator->score(sample));
}

std::string Model::predict(const std::unordered_map<std::string, std::string>& sample) const {
  return evaluator->predict(sample);
}

std::vector<double> Model::forecast(int horizon) const { return evaluator->forecast(horizon); }

std::vector<double> Model::forecast(int horizon,
                                    const std::unordered_map<std::string, std::vector<double>>& regressors) const {
  return evaluator->forecast(horizon, regressors);
}

std::vector<std::pair<double, double>> Model::forecast_with_variance(int horizon) const {
  return evaluator->forecast_with_variance(horizon);
}

std::vector<std::pair<double, double>> Model::forecast_with_variance(
    int horizon, const std::unordered_map<std::string, std::vector<double>>& regressors) const {
  return evaluator->forecast_with_variance(horizon, regressors);
}

std::string Model::output_name() const { return evaluator->output_name(); }
}  // namespace cpmml
