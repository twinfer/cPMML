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
 * For classification:     residual = probability(actual_target_class) - 1.0
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

  inline virtual double eval_double(Sample& sample, const InternalScore& score) const override {
    if (target_index == std::numeric_limits<size_t>::max()) return double_min();
    const Value& actual_val = sample[target_index].value;
    if (actual_val.missing) return double_min();

    if (!score.probabilities.empty()) {
      // Classification: probability(actual_class) - 1.0
      const std::string actual_str = actual_val.svalue;
      auto it = score.probabilities.find(actual_str);
      if (it == score.probabilities.end()) return double_min();
      return it->second - 1.0;
    } else {
      // Regression: actual_value - predicted_value
      return actual_val.value - score.double_score;
    }
  }
};

#endif  // CPMML_SRC_EXPRESSION_RESIDUAL_H_
