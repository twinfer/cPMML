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
  EntityId() {}

  EntityId(const XmlNode& /*node*/, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
           const DataType& output_type)
      : OutputExpression(output_index, output_type, indexer) {}

  inline virtual std::string eval_str(Sample& /*sample*/, const InternalScore& score) const override {
    return score.entity_id;
  };
};

#endif  // CPMML_SRC_EXPRESSION_ENTITYID_H_
