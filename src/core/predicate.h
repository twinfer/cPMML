/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_PREDICATE_H
#define CPMML_PREDICATE_H

#include <algorithm>
#include <set>
#include <unordered_set>
#include <vector>

#include "datadictionary.h"
#include "predicateoptype.h"
#include "sample.h"
#include "value.h"

// PMML predicates use three-valued logic: a comparison against a missing
// value is UNKNOWN, not false. UNKNOWN propagates through compound predicates
// and is what surrogates key on.
enum class PredicateTruth { TRUE_, FALSE_, UNKNOWN };

namespace cpmml::predicate {

using Op = PredicateOpType::PredicateOpTypeValue;

inline PredicateTruth to_result(const bool b) { return b ? PredicateTruth::TRUE_ : PredicateTruth::FALSE_; }

// The members of a SimpleSetPredicate (or of a DataField's valid or invalid
// Values), by the double LegacyCompare compares them by (switch 2,
// core/value.h): ordered, or hashed past PredicateBuilder's SET_THRESHOLD, as
// the std::set<Value> and std::unordered_set<Value> they replace were (the same
// membership: double equality, no NaN member, D26).
class LegacyKeySet {
 public:
  LegacyKeySet() = default;
  LegacyKeySet(const std::vector<Value>& members, const bool hashed) : hashed_(hashed) {
    for (const Value& m : members) {
      if (hashed)
        hash_.insert(cpmml::legacy::LegacyCompare::constant_key(m));
      else
        ordered_.insert(cpmml::legacy::LegacyCompare::constant_key(m));
    }
  }
  inline bool contains(const double key) const { return hashed_ ? hash_.count(key) != 0 : ordered_.count(key) != 0; }

 private:
  bool hashed_ = false;
  std::set<double> ordered_;
  std::unordered_set<double> hash_;
};

// A comparison of an operand x (not missing) with the predicate's value c
// (Predicates.html SimplePredicate), by LegacyCompare's double (switch 2).
inline PredicateTruth compare(const Op op, const double x, const double c) {
  switch (op) {
    case Op::EQUAL:
      return to_result(x == c);
    case Op::NOT_EQUAL:
      return to_result(!(x == c));
    case Op::GREATER_THAN:
      return to_result(x > c);
    case Op::GREATER_OR_EQUAL:
      return to_result(x >= c);
    case Op::LESS_THAN:
      return to_result(x < c);
    case Op::LESS_OR_EQUAL:
      return to_result(x <= c);
    default:
      return PredicateTruth::FALSE_;
  }
}

// CompoundPredicate over n operands, evaluated in order by `eval(i)`
// (TreeModel.html: three-valued and, or, xor; surrogate "the first predicate
// whose result is not UNKNOWN"). An error is not UNKNOWN: an exception from an
// operand ends the record.
template <class Eval>
inline PredicateTruth combine(const Op op, const std::size_t n, Eval&& eval) {
  bool flag = false;
  switch (op) {
    case Op::AND:  // any FALSE -> FALSE; else any UNKNOWN -> UNKNOWN
      for (std::size_t i = 0; i < n; i++) {
        const PredicateTruth r = eval(i);
        if (r == PredicateTruth::FALSE_) return PredicateTruth::FALSE_;
        if (r == PredicateTruth::UNKNOWN) flag = true;
      }
      return flag ? PredicateTruth::UNKNOWN : PredicateTruth::TRUE_;
    case Op::OR:  // any TRUE -> TRUE; else any UNKNOWN -> UNKNOWN
      for (std::size_t i = 0; i < n; i++) {
        const PredicateTruth r = eval(i);
        if (r == PredicateTruth::TRUE_) return PredicateTruth::TRUE_;
        if (r == PredicateTruth::UNKNOWN) flag = true;
      }
      return flag ? PredicateTruth::UNKNOWN : PredicateTruth::FALSE_;
    case Op::XOR:  // parity; any UNKNOWN -> UNKNOWN
      for (std::size_t i = 0; i < n; i++) {
        const PredicateTruth r = eval(i);
        if (r == PredicateTruth::UNKNOWN) return PredicateTruth::UNKNOWN;
        flag ^= (r == PredicateTruth::TRUE_);
      }
      return to_result(flag);
    case Op::SURROGATE:
      for (std::size_t i = 0; i < n; i++) {
        const PredicateTruth r = eval(i);
        if (r != PredicateTruth::UNKNOWN) return r;
      }
      return PredicateTruth::UNKNOWN;
    default:
      throw std::logic_error("predicate::combine: not a compound operator");
  }
}

}  // namespace cpmml::predicate

/**
 * @class Predicate
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/TreeModel.html#xsdGroup_PREDICATE">PMML
 * Predicate</a>, as the load builds it (PredicateBuilder, DataField
 * constraints, IntervalBuilder, Discretize bins, RuleSet's conjunctions). A
 * TreeModel compiles its nodes' predicates into BoundPredicates
 * (fastpredicate.h); the other models evaluate this tree, with the same
 * comparisons (cpmml::predicate).
 */
class Predicate {
 public:
  using Op = cpmml::predicate::Op;
  using Result = PredicateTruth;

  size_t feature = std::numeric_limits<size_t>::max();
  bool is_empty = true;
  PredicateOpType predicatetype = PredicateOpType::PredicateOpTypeValue::TRUE;
  Value value;
  double key = 0;  // value's LegacyCompare key
  // A set predicate: its members as read, in the order of their keys, one per
  // key (the iteration order of the std::set<Value> they were); values is
  // filled for an ordered set only (is_set_predicate), as before.
  bool is_set_predicate = false;
  std::vector<Value> values;
  bool is_hash_set = false;
  cpmml::predicate::LegacyKeySet keys;
  bool is_compound_predicate = false;
  std::vector<Predicate> predicates;

  Predicate() = default;

  explicit Predicate(PredicateOpType predicatetype) : is_empty(false), predicatetype(predicatetype) {}

  Predicate(const size_t feature, PredicateOpType predicatetype, const Value& value)
      : feature(feature),
        is_empty(false),
        predicatetype(predicatetype),
        value(value),
        key(cpmml::legacy::LegacyCompare::constant_key(value)) {}

  // A set of members (SimpleSetPredicate, a DataField's Values): hashed or
  // ordered (PredicateBuilder's SET_THRESHOLD).
  Predicate(const size_t feature, PredicateOpType predicatetype, const std::vector<Value>& members, const bool hashed)
      : feature(feature),
        is_empty(false),
        predicatetype(predicatetype),
        is_set_predicate(!hashed),
        values(hashed ? std::vector<Value>() : by_key(members)),
        is_hash_set(hashed),
        keys(members, hashed) {}

  Predicate(const std::vector<Predicate>& predicates, PredicateOpType predicatetype)
      : is_empty(false), predicatetype(predicatetype), is_compound_predicate(true), predicates(predicates) {}

  // A predicate that resolves to UNKNOWN does not fire (RuleSet rules,
  // Scorecard attributes, ensemble segments).
  inline bool operator()(const Sample& other) const {
    if (predicatetype.value == Op::TRUE) return true;  // most Segments: <True/>
    return evaluate(other) == Result::TRUE_;
  }

  inline bool operator()(const Value& other) const { return evaluate(other) == Result::TRUE_; }

  // Over a sample: each operand is its field's slot.
  inline Result evaluate(const Sample& other) const {
    return evaluate_with([&other](const Predicate& p) -> const Value& { return other[p.feature].value; });
  }

  // Over one value: every operand is `other` (a field's constraints).
  inline Result evaluate(const Value& other) const {
    return evaluate_with([&other](const Predicate&) -> const Value& { return other; });
  }

  template <class Operand>
  inline Result evaluate_with(const Operand& operand) const {
    using cpmml::predicate::to_result;
    switch (predicatetype.value) {
      case Op::TRUE:
        return Result::TRUE_;
      case Op::FALSE:
        return Result::FALSE_;
      case Op::AND:
      case Op::OR:
      case Op::XOR:
      case Op::SURROGATE:
        return cpmml::predicate::combine(predicatetype.value, predicates.size(),
                                         [&](std::size_t i) { return predicates[i].evaluate_with(operand); });
      case Op::IS_MISSING:
        return to_result(operand(*this).is_missing());
      case Op::IS_NOT_MISSING:
        return to_result(!operand(*this).is_missing());
      default:
        break;
    }

    const Value& x = operand(*this);
    if (x.is_missing()) return Result::UNKNOWN;  // comparison with missing → UNKNOWN
    const double k = cpmml::legacy::LegacyCompare::key(x);
    switch (predicatetype.value) {
      case Op::IS_IN:
        return to_result(keys.contains(k));
      case Op::IS_NOT_IN:
        return to_result(!keys.contains(k));
      default:
        return cpmml::predicate::compare(predicatetype.value, k, key);
    }
  }

  static inline Result to_result(const bool b) { return cpmml::predicate::to_result(b); }

 private:
  // The members ordered by key, the first of equal keys kept: a std::set<Value>'s
  // iteration (SVM reads the target's valid Values in that order).
  static std::vector<Value> by_key(const std::vector<Value>& members) {
    std::vector<Value> sorted = members;
    std::stable_sort(sorted.begin(), sorted.end(), [](const Value& a, const Value& b) {
      return cpmml::legacy::LegacyCompare::less_than(a, b);
    });
    sorted.erase(std::unique(sorted.begin(), sorted.end(),
                             [](const Value& a, const Value& b) {
                               return !cpmml::legacy::LegacyCompare::less_than(a, b) &&
                                      !cpmml::legacy::LegacyCompare::less_than(b, a);
                             }),
                 sorted.end());
    return sorted;
  }
};

#endif
