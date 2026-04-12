
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#include "cPMML.h"
#include "core/internal_score.h"

namespace cpmml {
Result::Result(const std::shared_ptr<InternalScore>& score) : score_(score) {}

std::string Result::as_string() const { return score_->score; }

double Result::as_double() const { return score_->double_score; }

std::unordered_map<std::string, double> Result::distribution() const { return score_->probabilities; }
std::unordered_map<std::string, double> Result::num_outputs() const { return score_->num_outputs; }
std::unordered_map<std::string, std::string> Result::str_outputs() const { return score_->str_outputs; }

std::vector<double> Result::series() const { return score_->forecast_values; }
std::vector<std::pair<double, double>> Result::series_with_variance() const {
  return score_->forecast_with_variance_values;
}

bool Result::empty() const { return score_ == nullptr || score_->empty; }
}  // namespace cpmml
