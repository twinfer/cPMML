
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_PREDICATE_H
#define CPMML_PREDICATE_H

#include "datadictionary.h"
#include "predicateoptype.h"
#include "sample.h"
#include "value.h"

/**
 * @class Predicate
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/TreeModel.html#xsdGroup_PREDICATE">PMML
 * Predicate</a>.
 *
 * It is used to enforce a constraint between values, between set of values or
 * between recursively defined set of other constraints. See also
 * PredicateOpType, TreeModel, DataDictionary.
 */
class Predicate {
 public:
  size_t feature = std::numeric_limits<size_t>::max();
  bool is_empty = true;
  PredicateOpType predicatetype = PredicateOpType::PredicateOpTypeValue::TRUE;
  Value value;
  bool is_set_predicate = false;
  std::set<Value> values;
  bool is_hash_set = false;
  std::unordered_set<Value, Value::ValueHash> values_hash;
  bool is_compound_predicate = false;
  std::vector<Predicate> predicates;

  Predicate() = default;

  explicit Predicate(PredicateOpType predicatetype) : is_empty(false), predicatetype(predicatetype) {}

  Predicate(const size_t feature, PredicateOpType predicatetype, const Value& value)
      : feature(feature), is_empty(false), predicatetype(predicatetype), value(value) {}

  Predicate(const size_t feature, PredicateOpType predicatetype, const std::set<Value>& values)
      : feature(feature), is_empty(false), predicatetype(predicatetype), is_set_predicate(true), values(values) {}

  Predicate(const size_t feature, PredicateOpType predicatetype,
            const std::unordered_set<Value, Value::ValueHash>& values_hash)
      : feature(feature), is_empty(false), predicatetype(predicatetype), is_hash_set(true), values_hash(values_hash) {}

  Predicate(const std::vector<Predicate>& predicates, PredicateOpType predicatetype)
      : is_empty(false), predicatetype(predicatetype), is_compound_predicate(true), predicates(predicates) {}

  // PMML predicates use three-valued logic: a comparison against a missing
  // value is UNKNOWN, not false. UNKNOWN propagates through compound
  // predicates and is what surrogates key on.
  enum class Result { TRUE_, FALSE_, UNKNOWN };

  // A predicate that resolves to UNKNOWN does not fire (RuleSet rules,
  // Scorecard attributes, ensemble segments).
  inline bool operator()(const Sample& other) const { return evaluate(other) == Result::TRUE_; }

  inline bool operator()(const Value& other) const { return evaluate(other) == Result::TRUE_; }

  inline Result evaluate(const Sample& other) const {
    switch (predicatetype.value) {
      case PredicateOpType::PredicateOpTypeValue::TRUE:
        return Result::TRUE_;
      case PredicateOpType::PredicateOpTypeValue::FALSE:
        return Result::FALSE_;
      case PredicateOpType::PredicateOpTypeValue::AND:
        return AND(other);
      case PredicateOpType::PredicateOpTypeValue::OR:
        return OR(other);
      case PredicateOpType::PredicateOpTypeValue::XOR:
        return XOR(other);
      case PredicateOpType::PredicateOpTypeValue::SURROGATE:
        return SURROGATE(other);
      case PredicateOpType::PredicateOpTypeValue::IS_MISSING:
        return to_result(other[feature].value.missing);
      case PredicateOpType::PredicateOpTypeValue::IS_NOT_MISSING:
        return to_result(!other[feature].value.missing);
      default:
        try {
          return evaluate(other[feature].value);
        } catch (const std::out_of_range& exception) {
#ifdef DEBUG
          throw cpmml::MissingValueException("feature \"" + other[feature].name +
                                              "\" not found in sample: " + other.to_string());
#else
          throw cpmml::MissingValueException("missing feature in sample");
#endif
        }
    }
  }

  inline Result evaluate(const Value& other) const {
    switch (predicatetype.value) {
      case PredicateOpType::PredicateOpTypeValue::TRUE:
        return Result::TRUE_;
      case PredicateOpType::PredicateOpTypeValue::FALSE:
        return Result::FALSE_;
      case PredicateOpType::PredicateOpTypeValue::IS_MISSING:
        return to_result(other.missing);
      case PredicateOpType::PredicateOpTypeValue::IS_NOT_MISSING:
        return to_result(!other.missing);
      case PredicateOpType::PredicateOpTypeValue::AND:
        return AND(other);
      case PredicateOpType::PredicateOpTypeValue::OR:
        return OR(other);
      case PredicateOpType::PredicateOpTypeValue::XOR:
        return XOR(other);
      case PredicateOpType::PredicateOpTypeValue::SURROGATE:
        return SURROGATE(other);
      default:
        break;
    }

    if (other.missing) return Result::UNKNOWN;  // comparison with missing → UNKNOWN

    switch (predicatetype.value) {
      case PredicateOpType::PredicateOpTypeValue::EQUAL:
        return to_result(other == value);
      case PredicateOpType::PredicateOpTypeValue::NOT_EQUAL:
        return to_result(other != value);
      case PredicateOpType::PredicateOpTypeValue::GREATER_THAN:
        return to_result(other > value);
      case PredicateOpType::PredicateOpTypeValue::GREATER_OR_EQUAL:
        return to_result(other >= value);
      case PredicateOpType::PredicateOpTypeValue::LESS_THAN:
        return to_result(other < value);
      case PredicateOpType::PredicateOpTypeValue::LESS_OR_EQUAL:
        return to_result(other <= value);
      case PredicateOpType::PredicateOpTypeValue::IS_IN:
        if (is_hash_set) return to_result(other.is_in(values_hash));
        return to_result(other.is_in(values));
      case PredicateOpType::PredicateOpTypeValue::IS_NOT_IN:
        if (is_hash_set) return to_result(other.is_not_in(values_hash));
        return to_result(other.is_not_in(values));
      default:
        return Result::FALSE_;
    }
  }

  static inline Result to_result(const bool b) { return b ? Result::TRUE_ : Result::FALSE_; }

  // Three-valued AND: any FALSE → FALSE; else any UNKNOWN → UNKNOWN.
  template <typename T>
  inline Result AND(const T& other) const {
    bool unknown = false;
    for (const auto& p : predicates) {
      const Result r = p.evaluate(other);
      if (r == Result::FALSE_) return Result::FALSE_;
      if (r == Result::UNKNOWN) unknown = true;
    }
    return unknown ? Result::UNKNOWN : Result::TRUE_;
  }

  // Three-valued OR: any TRUE → TRUE; else any UNKNOWN → UNKNOWN.
  template <typename T>
  inline Result OR(const T& other) const {
    bool unknown = false;
    for (const auto& p : predicates) {
      const Result r = p.evaluate(other);
      if (r == Result::TRUE_) return Result::TRUE_;
      if (r == Result::UNKNOWN) unknown = true;
    }
    return unknown ? Result::UNKNOWN : Result::FALSE_;
  }

  // Three-valued XOR (parity): any UNKNOWN → UNKNOWN; else TRUE for an odd
  // number of TRUE operands.
  template <typename T>
  inline Result XOR(const T& other) const {
    bool acc = false;
    for (const auto& p : predicates) {
      const Result r = p.evaluate(other);
      if (r == Result::UNKNOWN) return Result::UNKNOWN;
      acc ^= (r == Result::TRUE_);
    }
    return to_result(acc);
  }

  // Surrogate: the first predicate whose result is not UNKNOWN wins.
  template <typename T>
  inline Result SURROGATE(const T& other) const {
    for (const auto& p : predicates) {
      const Result r = p.evaluate(other);
      if (r != Result::UNKNOWN) return r;
    }
    return Result::UNKNOWN;
  }
};

#endif
