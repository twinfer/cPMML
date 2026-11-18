
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
  // other children (tree Node, Segment, SimpleRule, CompoundRule, scorecard
  // Attribute). The PREDICATE is required in every one of them, so a parent
  // without one is a load error. Element names are case-sensitive: a
  // misspelt <simplePredicate> used to be skipped, leaving an empty predicate
  // that matches every record; JPMML 1.7.7 refuses the document.
  inline Predicate build_from_parent(const XmlNode& parent) const {
    XmlNode predicate = parent.get_child_bylist(predicate_names());
    if (!predicate.is_empty()) return build(predicate);

    std::string found;
    for (const auto& child : parent.get_childs()) {
      if (child.name() == "Extension") continue;
      if (!found.empty()) found += ", ";
      found += child.name();
    }
    throw cpmml::ParsingException("missing predicate element in " + parent.name() +
                                  "; allowed elements are SimplePredicate, CompoundPredicate, SimpleSetPredicate, "
                                  "True, False; found " +
                                  (found.empty() ? std::string("no child elements") : found));
  }

  inline Predicate build(XmlNode node) const {
    if (node.is_empty()) return Predicate();

    using Op = PredicateOpType::PredicateOpTypeValue;
    switch (PredicateType(node).value) {
      case PredicateType::PredicateTypeValue::TRUE:
        return Predicate(Op::TRUE);
      case PredicateType::PredicateTypeValue::FALSE:
        return Predicate(Op::FALSE);
      case PredicateType::PredicateTypeValue::SIMPLE: {
        const Op op = read_enum(node, "operator", PredicateOpType::SIMPLE_OPERATOR_TABLE);
        // isMissing / isNotMissing have no value attribute
        const std::string field = node.required_attribute("field");
        if (op == Op::IS_MISSING || op == Op::IS_NOT_MISSING) return Predicate(indexer->get_index(field), op, Value());
        // Every other operator compares with value (Predicates.html).
        return Predicate(indexer->get_index(field), op,
                         Value(node.required_attribute("value"), indexer->get_type(field)));
      }
      case PredicateType::PredicateTypeValue::SIMPLESET: {
        const Op op = read_enum(node, "booleanOperator", PredicateOpType::SET_OPERATOR_TABLE);
        std::vector<std::string> values = parse_array(node.get_child("Array").value());
        const std::string field = node.required_attribute("field");
        if (values.size() > SET_THRESHOLD)
          return Predicate(indexer->get_index(field), op,
                           Value::createUnorderedValues(values, indexer->get_type(field)));
        return Predicate(indexer->get_index(field), op, Value::createValues(values, indexer->get_type(field)));
      }
      case PredicateType::PredicateTypeValue::COMPOUND: {
        const Op op = read_enum(node, "booleanOperator", PredicateOpType::COMPOUND_OPERATOR_TABLE);
        std::vector<Predicate> predicates;
        for (const auto& child : node.get_childs())
          if (child.name() != "Extension") predicates.push_back(build(child));

        return Predicate(predicates, op);
      }
    }

    return Predicate();
  }
};

#endif
