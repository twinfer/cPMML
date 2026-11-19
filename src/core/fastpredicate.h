
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_FASTPREDICATE_H
#define CPMML_FASTPREDICATE_H

#include "predicate.h"
#include "predicateoptype.h"
#include "sample.h"

/**
 * Compiled counterpart of Predicate, used on the TreeModel hot path.
 *
 * Like Predicate these are three-valued: a comparison against a missing value
 * is UNKNOWN, not false. The tree traversal needs that distinction to apply
 * missingValueStrategy, and treating missing as "compares less than anything"
 * (the previous behaviour, inherited from the Value sentinel) silently took
 * the wrong branch.
 */
using PredicateResult = Predicate::Result;
using PredicateFunction = std::function<PredicateResult(const Sample&)>;

inline PredicateResult to_predicate_result(const bool value) {
  return value ? PredicateResult::TRUE_ : PredicateResult::FALSE_;
}

constexpr PredicateResult _false(const Sample& other) { return PredicateResult::FALSE_; }

constexpr PredicateResult _true(const Sample& other) { return PredicateResult::TRUE_; }

inline PredicateResult equal(const Sample& other, const Value& value, const size_t& feature) {
  if (other[feature].value.is_missing()) return PredicateResult::UNKNOWN;
  return to_predicate_result(other[feature].cvalue() == value);
}

inline PredicateResult not_equal(const Sample& other, const Value& value, const size_t& feature) {
  if (other[feature].value.is_missing()) return PredicateResult::UNKNOWN;
  return to_predicate_result(other[feature].cvalue() != value);
}

inline PredicateResult greater_than(const Sample& other, const Value& value, const size_t& feature) {
  if (other[feature].value.is_missing()) return PredicateResult::UNKNOWN;
  return to_predicate_result(other[feature].cvalue() > value);
}

inline PredicateResult greater_orequal(const Sample& other, const Value& value, const size_t& feature) {
  if (other[feature].value.is_missing()) return PredicateResult::UNKNOWN;
  return to_predicate_result(other[feature].cvalue() >= value);
}

inline PredicateResult less_than(const Sample& other, const Value& value, const size_t& feature) {
  if (other[feature].value.is_missing()) return PredicateResult::UNKNOWN;
  return to_predicate_result(other[feature].cvalue() < value);
}

inline PredicateResult less_orequal(const Sample& other, const Value& value, const size_t& feature) {
  if (other[feature].value.is_missing()) return PredicateResult::UNKNOWN;
  return to_predicate_result(other[feature].cvalue() <= value);
}

inline PredicateResult is_in(const Sample& other, const std::set<Value>& values, const size_t& feature) {
  if (other[feature].value.is_missing()) return PredicateResult::UNKNOWN;
  return to_predicate_result(other[feature].cvalue().is_in(values));
}

inline PredicateResult is_notin(const Sample& other, const std::set<Value>& values, const size_t& feature) {
  if (other[feature].value.is_missing()) return PredicateResult::UNKNOWN;
  return to_predicate_result(other[feature].cvalue().is_not_in(values));
}

inline PredicateResult is_in_hash(const Sample& other, const std::unordered_set<Value, Value::ValueHash>& values,
                                  const size_t& feature) {
  if (other[feature].value.is_missing()) return PredicateResult::UNKNOWN;
  return to_predicate_result(other[feature].cvalue().is_in(values));
}

inline PredicateResult is_notin_hash(const Sample& other, const std::unordered_set<Value, Value::ValueHash>& values,
                                     const size_t& feature) {
  if (other[feature].value.is_missing()) return PredicateResult::UNKNOWN;
  return to_predicate_result(other[feature].cvalue().is_not_in(values));
}

inline PredicateResult is_missing(const Sample& other, const size_t& feature) {
  return to_predicate_result(other[feature].value.is_missing());
}

inline PredicateResult is_not_missing(const Sample& other, const size_t& feature) {
  return to_predicate_result(!other[feature].value.is_missing());
}

// Three-valued AND: any FALSE → FALSE; else any UNKNOWN → UNKNOWN.
inline PredicateResult _and(const Sample& other, const std::vector<PredicateFunction>& predicates) {
  bool unknown = false;
  for (const auto& predicate : predicates) {
    const PredicateResult result = predicate(other);
    if (result == PredicateResult::FALSE_) return PredicateResult::FALSE_;
    if (result == PredicateResult::UNKNOWN) unknown = true;
  }

  return unknown ? PredicateResult::UNKNOWN : PredicateResult::TRUE_;
}

// Three-valued OR: any TRUE → TRUE; else any UNKNOWN → UNKNOWN.
inline PredicateResult _or(const Sample& other, const std::vector<PredicateFunction>& predicates) {
  bool unknown = false;
  for (const auto& predicate : predicates) {
    const PredicateResult result = predicate(other);
    if (result == PredicateResult::TRUE_) return PredicateResult::TRUE_;
    if (result == PredicateResult::UNKNOWN) unknown = true;
  }

  return unknown ? PredicateResult::UNKNOWN : PredicateResult::FALSE_;
}

// Parity XOR: true when an odd number of operands are true; any UNKNOWN → UNKNOWN.
inline PredicateResult _xor(const Sample& other, const std::vector<PredicateFunction>& predicates) {
  bool acc = false;
  for (const auto& predicate : predicates) {
    const PredicateResult result = predicate(other);
    if (result == PredicateResult::UNKNOWN) return PredicateResult::UNKNOWN;
    acc ^= (result == PredicateResult::TRUE_);
  }

  return to_predicate_result(acc);
}

// Surrogate: the first predicate whose result is not UNKNOWN wins.
// TreeModel.html: "The cascaded predicates are applied when the primary
// predicate evaluates to UNKNOWN" and "surrogate(a,b) is equivalent to if not
// unknown(a) then a else b". An error is not UNKNOWN: an exception from an
// operand ends the record, as it does outside a surrogate (Predicate::SURROGATE
// agrees). The operands compiled here report a missing input as UNKNOWN and
// throw nothing per record (load errors cover the rest).
inline PredicateResult surrogate(const Sample& other, const std::vector<PredicateFunction>& predicates) {
  for (const auto& predicate : predicates) {
    const PredicateResult result = predicate(other);
    if (result != PredicateResult::UNKNOWN) return result;
  }

  return PredicateResult::UNKNOWN;
}

inline std::vector<PredicateFunction> to_function(const std::vector<Predicate>& predicates);

inline PredicateFunction to_function(const Predicate& predicate) {
  switch (predicate.predicatetype.value) {
    case PredicateOpType::PredicateOpTypeValue::TRUE:
      return std::bind(_true, std::placeholders::_1);
    case PredicateOpType::PredicateOpTypeValue::FALSE:
      return std::bind(_false, std::placeholders::_1);
    case PredicateOpType::PredicateOpTypeValue::EQUAL:
      return std::bind(equal, std::placeholders::_1, predicate.value, predicate.feature);
    case PredicateOpType::PredicateOpTypeValue::NOT_EQUAL:
      return std::bind(not_equal, std::placeholders::_1, predicate.value, predicate.feature);
    case PredicateOpType::PredicateOpTypeValue::GREATER_THAN:
      return std::bind(greater_than, std::placeholders::_1, predicate.value, predicate.feature);
    case PredicateOpType::PredicateOpTypeValue::GREATER_OR_EQUAL:
      return std::bind(greater_orequal, std::placeholders::_1, predicate.value, predicate.feature);
    case PredicateOpType::PredicateOpTypeValue::LESS_THAN:
      return std::bind(less_than, std::placeholders::_1, predicate.value, predicate.feature);
    case PredicateOpType::PredicateOpTypeValue::LESS_OR_EQUAL:
      return std::bind(less_orequal, std::placeholders::_1, predicate.value, predicate.feature);
    case PredicateOpType::PredicateOpTypeValue::IS_IN:
      // Large sets are stored in values_hash (see PredicateBuilder's
      // SET_THRESHOLD); binding the wrong member would evaluate against an
      // empty set.
      if (predicate.is_hash_set)
        return std::bind(is_in_hash, std::placeholders::_1, predicate.values_hash, predicate.feature);
      return std::bind(is_in, std::placeholders::_1, predicate.values, predicate.feature);
    case PredicateOpType::PredicateOpTypeValue::IS_NOT_IN:
      if (predicate.is_hash_set)
        return std::bind(is_notin_hash, std::placeholders::_1, predicate.values_hash, predicate.feature);
      return std::bind(is_notin, std::placeholders::_1, predicate.values, predicate.feature);
    case PredicateOpType::PredicateOpTypeValue::AND:
      return std::bind(_and, std::placeholders::_1, to_function(predicate.predicates));
    case PredicateOpType::PredicateOpTypeValue::OR:
      return std::bind(_or, std::placeholders::_1, to_function(predicate.predicates));
    case PredicateOpType::PredicateOpTypeValue::XOR:
      return std::bind(_xor, std::placeholders::_1, to_function(predicate.predicates));
    case PredicateOpType::PredicateOpTypeValue::SURROGATE:
      return std::bind(surrogate, std::placeholders::_1, to_function(predicate.predicates));
    case PredicateOpType::PredicateOpTypeValue::IS_MISSING:
      return std::bind(is_missing, std::placeholders::_1, predicate.feature);
    case PredicateOpType::PredicateOpTypeValue::IS_NOT_MISSING:
      return std::bind(is_not_missing, std::placeholders::_1, predicate.feature);
  }

  return std::bind(_false, std::placeholders::_1);
}

inline std::vector<PredicateFunction> to_function(const std::vector<Predicate>& predicates) {
  std::vector<PredicateFunction> result;

  for (const auto& predicate : predicates) result.push_back(to_function(predicate));

  return result;
}

#endif
