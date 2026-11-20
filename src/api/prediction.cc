
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#include "cPMML.h"
#include "core/internal_score.h"
#include "utils/utils.h"

namespace cpmml {
Result::Result(const std::shared_ptr<InternalScore>& score) : score_(score) {}

// Result is default-constructible and its default holds no score, so every
// accessor has to answer for that. empty() already did; the rest dereferenced
// a null shared_ptr, which made `cpmml::Result r; r.as_string();` — the first
// thing a caller writes when declaring a result before a branch assigns it —
// undefined. Each returns what "no prediction" means for its own type, which
// is the same thing an empty non-default Result returns.

std::string Result::as_string() const { return score_ ? score_->score : std::string(); }

// The public contract (design §5.5): double_min() when the prediction is no
// number. Inside the library "no number" is typed (InternalScore::double_score).
double Result::as_double() const { return score_ && score_->double_score ? *score_->double_score : double_min(); }

std::unordered_map<std::string, double> Result::distribution() const {
  return score_ ? score_->probabilities : std::unordered_map<std::string, double>();
}
std::unordered_map<std::string, double> Result::num_outputs() const {
  return score_ ? score_->num_outputs : std::unordered_map<std::string, double>();
}
std::unordered_map<std::string, std::string> Result::str_outputs() const {
  return score_ ? score_->str_outputs : std::unordered_map<std::string, std::string>();
}

std::vector<double> Result::series() const { return score_ ? score_->forecast_values : std::vector<double>(); }
std::vector<std::pair<double, double>> Result::series_with_variance() const {
  return score_ ? score_->forecast_with_variance_values : std::vector<std::pair<double, double>>();
}

bool Result::empty() const { return score_ == nullptr || score_->empty; }
}  // namespace cpmml
