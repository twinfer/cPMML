/*******************************************************************************

 * Copyright 2019 AMADEUS. All rights reserved.

 * Author: Paolo Iannino

 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_ENTITYID_H_
#define CPMML_SRC_EXPRESSION_ENTITYID_H_

#include "outputexpression.h"

/**
 * @class EntityId
 *
 * OutputExpression for feature="entityId". Returns the id attribute of the
 * matched leaf node, stored in InternalScore::entity_id.
 */
class EntityId : public OutputExpression {
 public:
  int rank;

  EntityId() : rank(1) {}

  EntityId(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
           const DataType& output_type)
      : OutputExpression(output_index, output_type, indexer),
        rank(node.exists_attribute("rank") ? static_cast<int>(node.get_long_attribute("rank")) : 1) {}

  inline virtual std::string eval_str(const Sample& /*sample*/, const InternalScore& score) const override {
    // Ranked entity IDs (KNN clustering: neighbor1, neighbor2, ...)
    if (!score.ranked_entity_ids.empty()) {
      const int idx = rank - 1;
      if (idx >= 0 && idx < static_cast<int>(score.ranked_entity_ids.size()))
        return score.ranked_entity_ids[idx];
      return "";
    }
    return score.entity_id;
  };

  // Entity ids are strings, but an OutputField may declare dataType="integer"
  // (common for cluster/node ids) — convert when possible.
  inline double eval_double(const Sample& sample, const InternalScore& score) const override {
    try {
      const std::string id = eval_str(sample, score);
      return id.empty() ? double_min() : to_double(id);
    } catch (const cpmml::ParsingException&) {
      return double_min();
    }
  };
};

#endif  // CPMML_SRC_EXPRESSION_ENTITYID_H_
