/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_CONFIDENCE_H_
#define CPMML_SRC_EXPRESSION_CONFIDENCE_H_

#include "outputexpression.h"

/**
 * @class Confidence
 *
 * OutputExpression for feature="confidence" outside association models, where
 * the confidence belongs to whatever the model matched — a RuleSet's SimpleRule
 * today — and is carried in InternalScore::confidence.
 *
 * AssociationModel keeps its own handling: there confidence is a property of a
 * ranked rule and needs the algorithm/rank attributes, which is why
 * OutputExpressionBuilder branches on `algorithm` exactly as it already does
 * for entityId.
 */
class Confidence : public OutputExpression {
 public:
  Confidence() = default;

  Confidence(const std::shared_ptr<Indexer>& indexer, const size_t& output_index, const DataType& output_type)
      : OutputExpression(output_index, output_type, indexer) {}

  inline std::optional<double> eval_double(const Sample& /*sample*/, const InternalScore& score) const override {
    return score.confidence;
  };

  inline std::string eval_str(const Sample& sample, const InternalScore& score) const override {
    const std::optional<double> value = eval_double(sample, score);
    return value ? java::double_text(*value) : "";
  };
};

#endif  // CPMML_SRC_EXPRESSION_CONFIDENCE_H_
