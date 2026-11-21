
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_DATAFIELD_H
#define CPMML_DATAFIELD_H

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

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
  // The two NaN flags (P6.8) sit in optype's padding, so the members after
  // them keep their offsets and DataField its size (the per-record reads of
  // MiningSchema::prepare touch them).
  // Whether a <Value property="missing"> of a float or double field reads as
  // NaN ("NaN", " NaN " after the whiteSpace collapse): every NaN input is
  // then missing, as a valid <Value value="NaN"/> makes every NaN input valid
  // (`nan_valid`), one rule for both (P6.8, decision (7a)).
  bool nan_missing = false;
  // Whether a NaN input is a valid value of the field (P6.8, B39), decided
  // here and not by `constraints`, whose comparisons call NaN indeterminate
  // (D9) or, in a member lookup, find it: NaN is one value of the XSD value
  // space (XML Schema Part 2: Datatypes Second Edition §3.2.5: "not-a-number
  // (NaN)"), and §3.2.5 (§3.2.4 float alike): "NaN equals itself but is
  // incomparable with (neither greater than nor less than) any other value in
  // the value space", so it is a member of a Value list only when the list
  // names NaN (any NaN input, the same rule as `nan_missing`), and inside no
  // Interval. So it is valid unless a valid Value list leaves it out, an
  // invalid Value names it, or an Interval bounds the field
  // (DataDictionary.html: "If intervals are present, any data that is
  // outside the intervals will be considered invalid").
  bool nan_valid = true;
  // <Value property="missing"> declares the values that *stand for* a missing
  // value: "?", "NA", "-999" (DataDictionary.html: "It is possible to
  // explicitly define values which are interpreted as missing values"). They
  // are markers, not a replacement, and an input matches one in two ways
  // (§2.8, P6.4, B6):
  //   * by its raw text, before any conversion (a marker such as "?" on a
  //     double field is no value of the field, so text is all it can match);
  //   * by value, for a field that is not a string: the marker read in the
  //     field's type (Value::document, the K1 rule: an integer field's "-999.0"
  //     is the double -999, D22) against the input read in it, so on a double
  //     field "-999.0" and "-9.99e2" are the marker -999. DataDictionary.html:
  //     "The dataType reuses names and semantics of atomic types in XML
  //     Schema", whose link is XML Schema Part 2: Datatypes Second Edition
  //     (W3C Recommendation 28 October 2004), the one edition this rule
  //     cites. Its §3.2.5 (double; §3.2.4 float alike): "This datatype
  //     differs from that of [IEEE 754-1985] in that there is only one NaN
  //     and only one zero. This makes the equality and ordering of values in
  //     the data space differ from that of [IEEE 754-1985] only in that for
  //     schema purposes NaN = NaN." So -0 is the marker 0 and a NaN marker
  //     matches any NaN input (decided 2026-11-20 after P6.7, (7a); P6.8).
  std::set<std::string, std::less<>> missing_markers;  // transparent: inputs are looked up as views
  // The markers that are values of the field's (non-string) type, read in it;
  // a marker that is not one ("?" on a double field) is matched as text only.
  // A NaN marker is kept apart, in `nan_missing`: NaN == NaN is false in the
  // engine's (IEEE) comparison, but the marker list is a schema Value list,
  // where NaN = NaN (`missing_markers`).
  std::vector<Value> missing_values;
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
    // matched as raw text first (§2.8), so it may be anything, and then, when
    // it is a value of the field's type, by that value (P6.4). Interval
    // margins are NUMBER (xs:double), whatever the field's type.
    for (const auto& value : values) {
      auto property = Property(value);
      switch (property.value) {
        case Property::PropertyValue::VALID:
          allowed_values.push_back(Value::document(value.required_attribute("value"), datatype));
          valid_values.push_back(value.required_attribute("value"));
          break;
        case Property::PropertyValue::MISSING: {
          const std::string text = value.required_attribute("value");
          missing_markers.insert(text);
          if (datatype.value != DataType::DataTypeValue::STRING) {
            // Not a silent fallback: a marker outside the field's lexical
            // space is still a marker, of text (§2.8), as it always was.
            try {
              Value marker = Value::document(text, datatype);
              if (is_nan(marker))
                nan_missing = true;
              else
                missing_values.push_back(std::move(marker));
            } catch (const cpmml::ParsingException&) {
            }
          }
          break;
        }
        case Property::PropertyValue::INVALID:
          forbidden_values.push_back(Value::document(value.required_attribute("value"), datatype));
          break;
      }
    }
    using Op = PredicateOpType::PredicateOpTypeValue;
    auto intervals = node.get_childs("Interval");
    // NaN is decided by `nan_valid` (validate, is_valid_value), never by the
    // member sets: a NaN member breaks the ordering of their std::set<double>
    // (NaN < x and x < NaN are both false, so NaN is "equivalent" to every
    // key), which made any input a member of a valid list that names NaN and
    // of an invalid one (P6.8, B39). So a NaN Value is taken out of both
    // lists before they are built; a valid list of NaN only leaves every
    // other value invalid (Op::FALSE).
    const auto drop_nan = [](std::vector<Value>& values) {
      const auto nan = std::remove_if(values.begin(), values.end(), [](const Value& v) { return is_nan(v); });
      const bool had = nan != values.end();
      values.erase(nan, values.end());
      return had;
    };
    const bool allowed_had_values = !allowed_values.empty();
    const bool nan_allowed = drop_nan(allowed_values);
    const bool nan_forbidden = drop_nan(forbidden_values);
    nan_valid = (!allowed_had_values || nan_allowed) && !nan_forbidden && intervals.empty();
    // The members are one per ContainerKey (a number itself, a string its
    // pool id), as the std::set<Value> they were: n_values counts the
    // distinct ones, NaN one of them.
    std::size_t n_allowed = nan_allowed ? 1 : 0;
    if (allowed_values.size() > 0) {
      tmp_contraints.push_back(Predicate(index, Op::IS_IN, allowed_values, false));
      n_allowed += tmp_contraints.back().values.size();
    } else if (allowed_had_values) {
      tmp_contraints.push_back(Predicate(PredicateOpType(Op::FALSE)));
    }
    if (forbidden_values.size() > 0) tmp_contraints.push_back(Predicate(index, Op::IS_NOT_IN, forbidden_values, false));

    for (const auto& interval : intervals)
      tmp_contraints.push_back(IntervalBuilder::build(interval, index, DataType::DataTypeValue::DOUBLE));

    // An invalid Value NaN alone leaves no member set; the constraints are
    // then not empty, so that validate and is_valid_value ask `nan_valid`,
    // and every other value passes them (Op::TRUE).
    if (!nan_valid && tmp_contraints.empty()) tmp_contraints.push_back(Predicate(PredicateOpType(Op::TRUE)));
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
    if (is_nan(sample[index].value)) [[unlikely]]
      return nan_valid;
    return constraints(sample);
  }

  // Whether `value`, a value of the field (not missing), is valid.
  inline bool is_valid_value(const Value& value) const {
    if (constraints.is_empty) return true;
    if (is_nan(value)) [[unlikely]]
      return nan_valid;
    return constraints(value);
  }

  // A double or float NaN (an input reads one since P6.8, B39).
  static bool is_nan(const Value& value) {
    if (const auto* d = std::get_if<cpmml::types::Double>(&value.typed())) return d->v != d->v;
    if (const auto* f = std::get_if<cpmml::types::Float>(&value.typed())) return f->v != f->v;
    return false;
  }

  static std::unordered_map<std::string, DataField> to_datafields(const std::vector<XmlNode>& nodes,
                                                                  const std::shared_ptr<Indexer>& indexer) {
    std::unordered_map<std::string, DataField> result;
    for (const auto& node : nodes) result[node.required_attribute("name")] = DataField(node, indexer);

    return result;
  }
};

#endif
