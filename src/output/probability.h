/*******************************************************************************

 * Copyright 2019 AMADEUS. All rights reserved.

 * Author: Paolo Iannino

 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_PROBABILITY_H_
#define CPMML_SRC_EXPRESSION_PROBABILITY_H_

#include "core/target.h"
#include "outputexpression.h"

/**
 * @class Probability
 *
 * Class representing <a
 * href="https://dmg.org/pmml/v4-4-1/Output.html">PMML OutputField
 * feature="probability"</a>.
 *
 * value "corresponds to one of the target categories of the model"; without it
 * "the probability of the predicted categorical value should be returned". The
 * probabilities are keyed by the raw class (never its displayValue), and value
 * is compared with them as a value of the target's dataType, so value="4" and a
 * class written "4.0" are one class of a numeric target (see Category).
 *
 * A class the model gave no probability to -- declared or not, displayValue or
 * not -- has probability 0.0 whenever the model produced a distribution. The
 * spec does not say what to return there; 0.0 is what JPMML 1.7.7 returns for
 * every model type. With no prediction, or no distribution at all (a
 * regression), there is no value.
 */
class Probability : public OutputExpression {
 public:
  size_t index;
  std::string target_value;
  Category category;

  Probability() : index(std::numeric_limits<size_t>::max()) {}

  Probability(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
              const DataType& output_type, const DataType& target_type)
      : OutputExpression(output_index, output_type, indexer),
        target_value(node.get_attribute("value", "")),
        category(target_type) {}

  inline virtual double eval_double(const Sample& /*sample*/, const InternalScore& score) const override {
    if (score.empty || score.probabilities.empty()) return double_min();

    // Without a value attribute: the predicted class, as the model named it
    // before Targets turned it into its displayValue.
    const std::string& key =
        !target_value.empty() ? target_value : (score.raw_score.empty() ? score.score : score.raw_score);
    const auto it = category.find(score.probabilities, key);
    return it != score.probabilities.cend() ? it->second : 0.0;
  };
};

#endif  // CPMML_SRC_EXPRESSION_PROBABILITY_H_
