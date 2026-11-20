
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_DATAFIELD_H
#define CPMML_DATAFIELD_H

#include <sstream>
#include <string>

#include "closure.h"
#include "datatype.h"
#include "indexer.h"
#include "intervalbuilder.h"
#include "optype.h"
#include "predicate.h"
#include "property.h"
#include "value.h"
#include "xmlnode.h"

/**
 * @class DataField
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/DataDictionary.html">PMML DataField</a>.
 *
 * It defines a feature available to the model, along with the values it can
 * assume. The constraints on the admissible values are enforced through the
 * class Predicate.
 */
class DataField {
 public:
  size_t n_values = 1;
  std::string name;
  // INTEGER for dataType="integer" (P2.4), whose lexical space is the XSD
  // integer's: an input "3.5" or "3.0" is not an integer
  // (MiningField::parse_input).
  DataType datatype;
  size_t index = std::numeric_limits<size_t>::max();
  OpType optype;
  // <Value property="missing"> declares which raw input tokens *stand for* a
  // missing value — "?", "NA", "-999". They are markers, not a replacement, so
  // they are kept as the literal strings the document wrote and matched before
  // any conversion: "-999" parses perfectly well as a double and would sail
  // through as data otherwise. This used to be a single Value that nothing read.
  std::set<std::string> missing_markers;
  // The <Value property="valid"> categories in document order. The constraint
  // predicate below holds the same set, but as a set and behind a Predicate, so
  // it answers "is this allowed" and not "which categories are there" -- and
  // the second question has an answer only the DataDictionary knows. It is what
  // GeneralRegression needs to recover a target category that has no PCells of
  // its own.
  std::vector<std::string> valid_values;
  Predicate constraints;

  DataField() = default;

  DataField(const std::string& name, const DataType& datatype)
      : n_values(1), name(name), datatype(datatype), index(std::numeric_limits<size_t>::max()) {}

  DataField(const XmlNode& node, const std::shared_ptr<Indexer>& indexer)
      : name(node.required_attribute("name")),
        datatype(DataType::read(node)),
        index(indexer->get_or_set(name, datatype).first),
        optype(OpType::read(node)) {
    auto values = node.get_childs("Value");
    std::vector<Predicate> tmp_contraints;
    std::vector<Value> allowed_values;
    std::vector<Value> forbidden_values;
    // Document text (§3.4, design commit P2.2): a valid or invalid
    // Value@value is a value of the field, read by the K1 rule, which
    // Value::document(text, datatype) is in the engine's storage (bind/load_scope.h:
    // an integer field's "4.0" is the double 4, rule 2); a missing marker is
    // matched as raw text first (§2.8), so it may be anything. Interval
    // margins are NUMBER (xs:double), whatever the field's type.
    for (const auto& value : values) {
      auto property = Property(value);
      switch (property.value) {
        case Property::PropertyValue::VALID:
          allowed_values.push_back(Value::document(value.required_attribute("value"), datatype));
          valid_values.push_back(value.required_attribute("value"));
          break;
        case Property::PropertyValue::MISSING:
          missing_markers.insert(value.required_attribute("value"));
          break;
        case Property::PropertyValue::INVALID:
          forbidden_values.push_back(Value::document(value.required_attribute("value"), datatype));
          break;
      }
    }
    using Op = PredicateOpType::PredicateOpTypeValue;
    // The members are one per LegacyCompare key, as the std::set<Value> they
    // were (switch 2): n_values counts the distinct ones.
    std::size_t n_allowed = 0;
    if (allowed_values.size() > 0) {
      tmp_contraints.push_back(Predicate(index, Op::IS_IN, allowed_values, false));
      n_allowed = tmp_contraints.back().values.size();
    }
    if (forbidden_values.size() > 0) tmp_contraints.push_back(Predicate(index, Op::IS_NOT_IN, forbidden_values, false));

    auto intervals = node.get_childs("Interval");
    for (const auto& interval : intervals)
      tmp_contraints.push_back(IntervalBuilder::build(interval, index, DataType::DataTypeValue::DOUBLE));

    if (tmp_contraints.size() > 0) constraints = Predicate(tmp_contraints, Op::AND);

    n_values = n_allowed > 0 ? n_allowed : 1;
  };

  // A missing value is not an invalid one: PMML keeps the two apart, and what
  // happens to a missing input is for missingValueTreatment and the model to
  // decide. Without this, a field declaring its valid <Value>s or an <Interval>
  // rejected the whole record, because the isIn/interval constraint answers
  // UNKNOWN for a missing operand and UNKNOWN does not pass.
  inline bool validate(const Sample& sample) const {
    if (constraints.is_empty || sample[index].value.is_missing()) return true;
    return constraints(sample);
  }

  static std::unordered_map<std::string, DataField> to_datafields(const std::vector<XmlNode>& nodes,
                                                                  const std::shared_ptr<Indexer>& indexer) {
    std::unordered_map<std::string, DataField> result;
    for (const auto& node : nodes) result[node.required_attribute("name")] = DataField(node, indexer);

    return result;
  }
};

#endif
