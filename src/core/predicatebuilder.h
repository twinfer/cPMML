
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_PREDICATEBUILDER_H
#define CPMML_PREDICATEBUILDER_H

#define SET_THRESHOLD 150

#include "datadictionary.h"
#include "miningschema.h"
#include "predicatetype.h"

/**
 * @class PredicateBuilder
 *
 * Factory class to create Predicate objects.
 */
class PredicateBuilder {
 public:
  std::shared_ptr<Indexer> indexer;

  PredicateBuilder() = default;

  explicit PredicateBuilder(const std::shared_ptr<Indexer>& indexer) : indexer(indexer) {}

  // The five element names of the PMML PREDICATE group. Lookup must be by exact
  // name: pattern matching on "Predicate" misses <True/> and <False/>.
  static const std::unordered_set<std::string>& predicate_names() {
    static const std::unordered_set<std::string> names = {"SimplePredicate", "CompoundPredicate", "SimpleSetPredicate",
                                                          "True", "False"};
    return names;
  }

  // Build the predicate of an element that contains one PREDICATE child among
  // other children (Segment, Node, SimpleRule, Attribute, ...). Returns an
  // empty predicate when no PREDICATE child exists.
  inline Predicate build_from_parent(const XmlNode& parent) const {
    return build(parent.get_child_bylist(predicate_names()));
  }

  inline Predicate build(XmlNode node) const {
    if (node.is_empty()) return Predicate();

    Predicate result;
    PredicateType predicatetype(node.name());
    switch (predicatetype.value) {
      case PredicateType::PredicateTypeValue::TRUE:
        return Predicate("true");
      case PredicateType::PredicateTypeValue::FALSE:
        return Predicate("false");
      case PredicateType::PredicateTypeValue::SIMPLE: {
        const std::string op = node.get_attribute("operator");
        const std::string lower_op = to_lower(op);
        // isMissing / isNotMissing have no value attribute
        if (lower_op == "ismissing" || lower_op == "isnotmissing")
          return Predicate(indexer->get_index(node.get_attribute("field")), op, Value());
        return Predicate(indexer->get_index(node.get_attribute("field")), op,
                         Value(node.get_attribute("value"), indexer->get_type(node.get_attribute("field"))));
      }
      case PredicateType::PredicateTypeValue::SIMPLESET: {
        std::vector<std::string> values = split(node.get_child("Array").value(), " ");
        if (values.size() > SET_THRESHOLD)
          return Predicate(indexer->get_index(node.get_attribute("field")), node.get_attribute("booleanOperator"),
                           Value::createUnorderedValues(values, indexer->get_type(node.get_attribute("field"))));
        return Predicate(indexer->get_index(node.get_attribute("field")), node.get_attribute("booleanOperator"),
                         Value::createValues(values, indexer->get_type(node.get_attribute("field"))));
      }
      case PredicateType::PredicateTypeValue::COMPOUND:
        std::vector<Predicate> predicates;
        for (const auto& child : node.get_childs()) predicates.push_back(build(child));

        return Predicate(predicates, node.get_attribute("booleanOperator"));
    }

    return Predicate();
  }
};

#endif
