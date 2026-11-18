
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_SCOREDISTRIBUTION_H
#define CPMML_SCOREDISTRIBUTION_H

#include <optional>
#include <string>

#include "core/value.h"
#include "core/xmlnode.h"

/**
 * @class ScoreDistribution
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/TreeModel.html#xsdElement_ScoreDistribution">PMML
 * ScoreDistribution</a>.
 *
 * It contains additional information for the score associated to a node. For
 * instance:
 * - Probability distribution.
 * - Confidence.
 */
class ScoreDistribution {
 public:
  Value value;
  std::string value_string;
  // pmml-4-4.xsd ScoreDistribution: value and recordCount are use="required";
  // confidence and probability are optional PROB-NUMBERs without a default.
  double record_count = 0.0;
  std::optional<double> confidence;
  std::optional<double> probability;

  ScoreDistribution() = default;

  ScoreDistribution(XmlNode node, DataType dataType)
      : value(Value(node.required_attribute("value"), dataType)),
        value_string(node.required_attribute("value")),
        record_count(node.required_double_attribute("recordCount")),
        confidence(node.optional_double_attribute("confidence")),
        probability(node.optional_double_attribute("probability")) {}

  static std::vector<ScoreDistribution> to_score_distributions(const std::vector<XmlNode>& score_distribution_nodes,
                                                               const DataType& target_type) {
    std::vector<ScoreDistribution> result;  // initialization just for perf improvement

    for (const auto& node : score_distribution_nodes) result.push_back(ScoreDistribution(node, target_type));

    return result;
  }
};

#endif
