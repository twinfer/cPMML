/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_RESIDUAL_H_
#define CPMML_SRC_EXPRESSION_RESIDUAL_H_

#include "outputexpression.h"

/**
 * @class Residual
 *
 * OutputExpression for feature="residual".
 *
 * For regression models:  residual = actual_target_value - predicted_value
 * For classification:     residual = X - probability(predicted_class),
 *                         where X = 1 if actual == predicted, else 0
 *
 * The actual target value is read from the internal sample at target_index,
 * which must be populated from the raw input before OutputExpression evaluation.
 */
class Residual : public OutputExpression {
 public:
  size_t target_index;

  Residual() : target_index(std::numeric_limits<size_t>::max()) {}

  Residual(const XmlNode& /*node*/, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
           const DataType& output_type, const std::string& target_name)
      : OutputExpression(output_index, output_type, indexer),
        target_index(target_name.empty() ? std::numeric_limits<size_t>::max() : indexer->get_index(target_name)) {}

  inline virtual double eval_double(const Sample& sample, const InternalScore& score) const override {
    if (target_index == std::numeric_limits<size_t>::max()) return double_min();
    const Value& actual_val = sample[target_index].value;
    if (actual_val.is_missing()) return double_min();

    // Classification residual (JPMML semantics): X − p(predicted), where
    // X = 1 if the actual value equals the predicted category, 0 otherwise.
    // A score is treated as classification only when the predicted category
    // has a probability entry; anything else falls through to regression.
    auto it = score.probabilities.find(score.score);
    if (it == score.probabilities.end() && !score.raw_score.empty()) it = score.probabilities.find(score.raw_score);
    if (it != score.probabilities.end()) {
      const double x = (actual_val.text() == score.score || actual_val.text() == score.raw_score) ? 1.0 : 0.0;
      return x - it->second;
    }

    // Regression: actual_value - predicted_value
    if (is_double_min(score.double_score)) return double_min();
    return actual_val.as_double() - score.double_score;
  }
};

#endif  // CPMML_SRC_EXPRESSION_RESIDUAL_H_
