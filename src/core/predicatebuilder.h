
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_PREDICATEBUILDER_H
#define CPMML_PREDICATEBUILDER_H

#define SET_THRESHOLD 150

#include <algorithm>
#include <string>

#include "bind/load_scope.h"
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
        // scope.resolve (P2.1): the field's declared type even when it is
        // declared after this predicate's model element builds it (B12).
        const cpmml::bind::Binding* binding = nullptr;
        const auto [index, datatype] =
            cpmml::bind::scope_resolve(*indexer, node, node.required_attribute("field"), &binding);
        if (op == Op::IS_MISSING || op == Op::IS_NOT_MISSING) return Predicate(index, op, Value());
        // Every other operator compares with value (Predicates.html), typed
        // (P6.3, §2.7). An ordering of an ordinal field with Values, of any
        // type, follows their order (P6.12; DataDictionary.html: "For
        // ordinal fields, if a sequence of Value elements is given, the
        // sequence defines the ascending order on the respective field").
        // A value outside the Values has no place in that order: a load
        // error (D47, design items P6.9, P6.12; the binder's
        // E-ORDINAL-VALUE-UNKNOWN), a number placed by its value.
        const Value value = Value::document(node.required_attribute("value"), datatype);
        std::shared_ptr<const cpmml::compare::OrdinalOrder> ordinal;
        // (The optype test first: most predicates are of no ordinal field.)
        if (op != Op::EQUAL && op != Op::NOT_EQUAL && binding->type.optype == cpmml::bind::OpType::Ordinal)
          if (auto declared = cpmml::bind::ordinal_declaration(binding)) {
            if (!declared->contains(value))
              throw cpmml::ParsingException(
                  cpmml::bind::ordinal_value_message(node,
                                                     "SimplePredicate field=\"" + node.required_attribute("field") +
                                                         "\" operator=\"" + node.required_attribute("operator") + "\"",
                                                     "value ", node.required_attribute("value"),
                                                     node.required_attribute("field"), declared->values));
            ordinal = std::make_shared<const cpmml::compare::OrdinalOrder>(std::move(declared->members));
          }
        return Predicate::simple(index, op, value, std::move(ordinal));
      }
      case PredicateType::PredicateTypeValue::SIMPLESET: {
        const Op op = read_enum(node, "booleanOperator", PredicateOpType::SET_OPERATOR_TABLE);
        std::vector<std::string> values = parse_array(node.get_child("Array").value());
        const auto [index, datatype] = cpmml::bind::scope_resolve(*indexer, node, node.required_attribute("field"));
        // §2.7 / D26: a NaN member "can match nothing", so it is a load error,
        // for any number of members (the linear path must give the hashed
        // path's answers). JPMML 1.7.7 loads it.
        for (const auto& v : values)
          if (cpmml::bind::is_nan_key(v, datatype))
            throw cpmml::ParsingException(std::string(cpmml::bind::code::kKeyNaN) + ": SimpleSetPredicate field=\"" +
                                          node.required_attribute("field") +
                                          "\": a NaN member matches no input (§2.7, D26: a NaN key is a load error)");
        std::vector<Value> members;
        members.reserve(values.size());
        for (const auto& v : values) members.push_back(Value::document(v, datatype));
        return Predicate::simple_set(index, op, members, values.size() > SET_THRESHOLD);
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
