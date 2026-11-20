/*******************************************************************************

 * Copyright 2019 AMADEUS. All rights reserved.

 * Author: Paolo Iannino

 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_ENTITYID_H_
#define CPMML_SRC_EXPRESSION_ENTITYID_H_

#include "bind/load_scope.h"
#include "fn/cast.h"
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
        rank(static_cast<int>(node.get_long_attribute("rank", 1))) {}

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
  // (common for cluster/node ids; Output.html: "A clusterId field, for
  // example, can have integer as its dataType"). §3.4: the OutputField casts
  // the id to its dataType, the strict XSD reading of the text (§2.6), and
  // an id that is no value of it makes the record invalid (Q2, decided (b);
  // JPMML 1.7.7: NumberFormatException). OutputField::add_output does that
  // itself (Source::Text, naming the field); this is the number a modelChain
  // step writes for the next segments (InternalModel::
  // _write_outputs_to_sample). It used to catch the ParsingException and
  // give no value, silently (design commit P4.7).
  inline std::optional<double> eval_double(const Sample& sample, const InternalScore& score) const override {
    const std::string id = eval_str(sample, score);
    if (id.empty()) return std::nullopt;
    const cpmml::fn::DeclaredCast cast(cpmml::bind::TypeId::String, numeric_type());
    double d = 0;
    if (!cast.parse_text(id, d))
      throw cpmml::InvalidValueException("OutputField entityId: " + cast.failure(std::string_view(id)));
    return d;
  };

 private:
  // The number or boolean type the id is read in: the OutputField's, a
  // string field's number being a double, as before.
  cpmml::bind::TypeId numeric_type() const {
    const cpmml::bind::TypeId t = cpmml::bind::type_id(output_type);
    return t == cpmml::bind::TypeId::String ? cpmml::bind::TypeId::Double : t;
  }
};

#endif  // CPMML_SRC_EXPRESSION_ENTITYID_H_
