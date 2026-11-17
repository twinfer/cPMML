/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_REASONCODE_H_
#define CPMML_SRC_EXPRESSION_REASONCODE_H_

#include "outputexpression.h"

/**
 * @class ReasonCode
 *
 * OutputExpression for Scorecard's feature="reasonCode". The spec ranks the
 * codes by accumulated points missed, largest first, and the `rank` attribute
 * names a position in that ranking — rank="1" is the code that cost the most
 * points.
 *
 * The ranking is the whole content of the feature, which is why it has to reach
 * the result as an ordered list. Writing the codes into a map keyed by code, as
 * ScorecardModel used to, throws away the only thing an OutputField asks for.
 */
class ReasonCode : public OutputExpression {
 public:
  int rank = 1;

  ReasonCode() = default;

  ReasonCode(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
             const DataType& output_type)
      : OutputExpression(output_index, output_type, indexer),
        rank(static_cast<int>(node.get_long_attribute("rank", 1))) {}

  inline std::string eval_str(const Sample& /*sample*/, const InternalScore& score) const override {
    const int index = rank - 1;
    if (index < 0 || index >= static_cast<int>(score.ranked_reason_codes.size())) return "";
    return score.ranked_reason_codes[index];
  };

  // A reason code is a label, so there is no numeric reading of it. Returning
  // the points missed instead would silently answer a different question.
  inline double eval_double(const Sample& /*sample*/, const InternalScore& /*score*/) const override {
    return double_min();
  };
};

#endif  // CPMML_SRC_EXPRESSION_REASONCODE_H_
