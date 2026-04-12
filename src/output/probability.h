/*******************************************************************************

 * Copyright 2019 AMADEUS. All rights reserved.

 * Author: Paolo Iannino

 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_PROBABILITY_H_
#define CPMML_SRC_EXPRESSION_PROBABILITY_H_

#include "outputexpression.h"

/**
 * @class Probability
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_FieldRef">PMML
 * Probability</a>.
 *
 * It is used to simply reference (and rename) the probabilities associated with
 * the raw prediction value provided by the model.
 */
class Probability : public OutputExpression {
 public:
  size_t index;
  std::string target_value;

  Probability() : index(std::numeric_limits<size_t>::max()) {}

  Probability(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
              const DataType& output_type)
      : OutputExpression(output_index, output_type, indexer),
        target_value(node.exists_attribute("value") ? node.get_attribute("value") : "") {}

  inline virtual double eval_double(const Sample& sample, const InternalScore& score) const override {
    // When no value attribute is specified, return the probability of the
    // predicted (winning) class.
    const std::string& key = target_value.empty() ? score.score : target_value;
    auto it = score.probabilities.find(key);
    if (it != score.probabilities.cend()) return it->second;

    return double_min();
  };
};

#endif  // CPMML_SRC_EXPRESSION_PROBABILITY_H_
