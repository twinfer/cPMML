/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_PREDICATE_H
#define CPMML_PREDICATE_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "compare.h"
#include "core/types/keys.h"
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

inline PredicateTruth to_tri_result(const types::Tri t) {
  return t == types::Tri::True ? PredicateTruth::TRUE_
                               : (t == types::Tri::False ? PredicateTruth::FALSE_ : PredicateTruth::UNKNOWN);
}

inline types::Relation relation_of(const Op op) {
  switch (op) {
    case Op::EQUAL:
      return types::Relation::Equal;
    case Op::NOT_EQUAL:
      return types::Relation::NotEqual;
    case Op::LESS_THAN:
      return types::Relation::Less;
    case Op::LESS_OR_EQUAL:
      return types::Relation::LessOrEqual;
    case Op::GREATER_THAN:
      return types::Relation::Greater;
    case Op::GREATER_OR_EQUAL:
      return types::Relation::GreaterOrEqual;
    default:
      throw std::logic_error("predicate::relation_of: not a comparison operator");
  }
}

// A SimplePredicate's comparison of an operand x (not missing) with its value
// c, typed (§2.7, P6.3; core/compare.h): in the common type, ±INF against the
// same infinity and NaN UNKNOWN (D9, B44), an ordinal field's declared order
// (any type, P6.12). The value is read in the field's type (§3.4), so a pair of no common
// type is a binder bug, never an answer. The operand is read as the value's
// type when that is string or boolean (compare::operand: a number that
// carries a text is that text; a boolean result held as the Double 1 or 0,
// fn::adapter::boolean, is that boolean); a number as it is held.
inline PredicateTruth typed_compare(const Op op, const Value& x, const Value& c,
                                    const cpmml::compare::OrdinalOrder* ordinal) {
  const types::TypeId value_type = types::type_of(c.typed());
  const types::TypeId as = value_type == types::TypeId::String || value_type == types::TypeId::Boolean
                               ? value_type
                               : types::TypeId::Missing;
  const auto t = cpmml::compare::relate(relation_of(op), cpmml::compare::operand(x, as, 0), c.typed(), ordinal);
  if (!t)
    throw std::logic_error("SimplePredicate: the operand and the value have no common type (" +
                           std::string(cpmml::compare::type_name(types::type_of(x.typed()))) + ", " +
                           std::string(cpmml::compare::type_name(types::type_of(c.typed()))) + ")");
  return to_tri_result(*t);
}

// The members of a SimpleSetPredicate (or of a DataField's valid or invalid
// Values), by their ContainerKey (a number itself, a string its pool id):
// ordered, or hashed past PredicateBuilder's SET_THRESHOLD, as the
// std::set<Value> and std::unordered_set<Value> they replace were (the same
// membership: double equality, no NaN member, D26). contains() is that
// membership (a DataField's Values: the identity of a value of the field).
// member() is SimpleSetPredicate's (P6.3, §2.7 "Hash containers"): a probe of
// the members' kind is looked up, a ±INF probe the members hold and a NaN
// probe are UNKNOWN (types::decide_nonfinite: isIn is the or of equalities,
// each indeterminate for them, D9); a string against numbers, or a number
// against strings, is compared member by member by text (D2).
class KeySet {
 public:
  KeySet() = default;
  KeySet(const std::vector<Value>& members, const bool hashed) : hashed_(hashed) {
    for (const Value& m : members) {
      if (hashed)
        hash_.insert(cpmml::ContainerKey::of_constant(m));
      else
        ordered_.insert(cpmml::ContainerKey::of_constant(m));
      nonfinite_.note(m.typed());
      if (const auto* i = std::get_if<types::Integer>(&m.typed()); i && !exact_key(i->v)) big_integers_ = true;
    }
    if (!members.empty()) type_ = types::type_of(members.front().typed());
    numbers_ = kind(type_) == 0;
    members_ = members;
  }
  inline bool contains(const double key) const { return hashed_ ? hash_.count(key) != 0 : ordered_.count(key) != 0; }

  // SimpleSetPredicate membership of x (not missing): TRUE, FALSE or UNKNOWN.
  // Inline: a string among strings and a finite double among numbers, the
  // sets a tree is made of; every other probe out of line.
  inline PredicateTruth member(const Value& x) const {
    const types::Value& v = x.typed();
    if (const auto* s = std::get_if<types::String>(&v); s && type_ == types::TypeId::String)
      return to_result(contains(s->rep->id == types::StringRep::kUnpooled ? -1.0 : static_cast<double>(s->rep->id)));
    if (const auto* d = std::get_if<types::Double>(&v); d && numbers_ && std::isfinite(d->v))
      return to_result(contains(d->v));
    return member_other(x);
  }

 private:
  [[gnu::noinline]] PredicateTruth member_other(const Value& x) const {
    const types::TypeId t = types::type_of(x.typed());
    if (kind(t) == kind(type_) || members_.empty()) {
      if (t == types::TypeId::Double || t == types::TypeId::Float) {
        if (const auto d = types::decide_nonfinite(x.typed(), nonfinite_))
          return *d == types::Lookup::Indeterminate ? PredicateTruth::UNKNOWN : PredicateTruth::FALSE_;
      } else if (t == types::TypeId::Integer && type_ == types::TypeId::Float) {
        // integer x float compares in float (§2.5).
        const double f = static_cast<double>(static_cast<float>(std::get<types::Integer>(x.typed()).v));
        return to_result(contains(f));
      } else if (t == types::TypeId::Integer && type_ == types::TypeId::Integer &&
                 (big_integers_ || !exact_key(std::get<types::Integer>(x.typed()).v))) {
        // Two integers compare exactly (§2.5, types::numeric_order); a double
        // key holds an integer exactly only below 2^53 in magnitude, so a set
        // or probe past it is compared member by member.
        return linear(x);
      }
      return to_result(contains(cpmml::ContainerKey::of(x)));
    }
    return linear(x);
  }

  // Member by member: the or of the typed equalities (D2 for another kind).
  PredicateTruth linear(const Value& x) const {
    bool unknown = false;
    for (const Value& m : members_) {
      const PredicateTruth r = typed_compare(Op::EQUAL, x, m, nullptr);
      if (r == PredicateTruth::TRUE_) return r;
      unknown |= r == PredicateTruth::UNKNOWN;
    }
    return unknown ? PredicateTruth::UNKNOWN : PredicateTruth::FALSE_;
  }

  // An integer its double key holds exactly (|v| <= 2^53).
  static constexpr bool exact_key(const std::int64_t v) {
    constexpr std::int64_t kExact = std::int64_t{1} << 53;
    return v >= -kExact && v <= kExact;
  }

  // 0 number, 1 boolean, 2 string, 3 none.
  static constexpr int kind(const types::TypeId t) {
    switch (t) {
      case types::TypeId::Integer:
      case types::TypeId::Float:
      case types::TypeId::Double:
        return 0;
      case types::TypeId::Boolean:
        return 1;
      case types::TypeId::String:
        return 2;
      default:
        return 3;
    }
  }
  bool hashed_ = false;
  std::set<double> ordered_;
  std::unordered_set<double> hash_;
  types::TypeId type_ = types::TypeId::Missing;
  bool numbers_ = false;       // the members are numbers
  bool big_integers_ = false;  // an integer member past 2^53 in magnitude (no exact double key)
  types::NonFiniteKeys nonfinite_;
  std::vector<Value> members_;
};

// A comparison of an operand x (not missing) with a bound c, by the double
// each is (IEEE): an Interval's or a Discretize bin's margins (pmml-4-4.xsd
// NUMBER; D44: an explicit INF margin bounds as written), not a built-in
// comparison.
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
  double key = 0;  // value's ContainerKey (an Interval's or a bin's margin: the number)
  // A SimplePredicate or SimpleSetPredicate (PredicateBuilder): typed
  // comparisons (P6.3, §2.7). Otherwise (an Interval's or Discretize bin's
  // margins, a DataField's Values) the double each is, as before.
  bool typed = false;
  // A typed comparison whose value is a finite number: the double fast path.
  bool finite_number = false;
  // An ordering of an ordinal field with Values (any type, P6.12): its
  // declared order.
  std::shared_ptr<const cpmml::compare::OrdinalOrder> ordinal;
  // A set predicate: its members as read, in the order of their keys, one per
  // key (the iteration order of the std::set<Value> they were); values is
  // filled for an ordered set only (is_set_predicate), as before.
  bool is_set_predicate = false;
  std::vector<Value> values;
  bool is_hash_set = false;
  cpmml::predicate::KeySet keys;
  bool is_compound_predicate = false;
  std::vector<Predicate> predicates;

  Predicate() = default;

  explicit Predicate(PredicateOpType predicatetype) : is_empty(false), predicatetype(predicatetype) {}

  Predicate(const size_t feature, PredicateOpType predicatetype, const Value& value)
      : feature(feature),
        is_empty(false),
        predicatetype(predicatetype),
        value(value),
        key(cpmml::ContainerKey::of_constant(value)) {}

  // A SimplePredicate (Predicates.html): `value` read in the field's type.
  static Predicate simple(const size_t feature, PredicateOpType op, const Value& value,
                          std::shared_ptr<const cpmml::compare::OrdinalOrder> ordinal = nullptr) {
    Predicate p(feature, op, value);
    p.typed = true;
    const cpmml::types::TypeId t = cpmml::types::type_of(value.typed());
    // (An integer from 2^53 on may have no exact double: the typed path compares
    // two integers exactly.)
    p.finite_number = (t == cpmml::types::TypeId::Integer || t == cpmml::types::TypeId::Float || t == cpmml::types::TypeId::Double) &&
                      std::isfinite(value.as_double()) &&
                      !(t == cpmml::types::TypeId::Integer && std::fabs(value.as_double()) >= 9007199254740992.0) &&
                      !ordinal;  // an ordinal field's Values order it (P6.12): the typed path
    p.ordinal = std::move(ordinal);
    return p;
  }
  // A SimpleSetPredicate.
  static Predicate simple_set(const size_t feature, PredicateOpType op, const std::vector<Value>& members,
                              const bool hashed) {
    Predicate p(feature, op, members, hashed);
    p.typed = true;
    return p;
  }

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
    if (typed) return evaluate_typed(x);
    const double k = cpmml::ContainerKey::of(x);
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

  // The typed comparison (P6.3) of x (not missing): a number against a finite
  // number in double (the promoted type of integer or float and double,
  // exact; NaN is UNKNOWN, D9), every other pair by predicate::typed_compare.
  inline Result evaluate_typed(const Value& x) const {
    switch (predicatetype.value) {
      case Op::IS_IN:
        return keys.member(x);
      case Op::IS_NOT_IN: {
        const Result r = keys.member(x);
        return r == Result::UNKNOWN ? r : (r == Result::TRUE_ ? Result::FALSE_ : Result::TRUE_);
      }
      default:
        break;
    }
    if (finite_number) {
      const cpmml::types::TypeId t = cpmml::types::type_of(x.typed());
      const cpmml::types::TypeId c = cpmml::types::type_of(value.typed());
      if (t == cpmml::types::TypeId::Double || t == cpmml::types::TypeId::Float ||
          (t == cpmml::types::TypeId::Integer && c != cpmml::types::TypeId::Float &&
           (c == cpmml::types::TypeId::Double || std::fabs(x.as_double()) <= 9007199254740992.0))) {
        const double d = x.as_double();
        if (d != d) return Result::UNKNOWN;  // NaN (D9)
        return cpmml::predicate::compare(predicatetype.value, d, key);
      }
    }
    return cpmml::predicate::typed_compare(predicatetype.value, x, value, ordinal.get());
  }

 private:
  // The members ordered by key, the first of equal keys kept: a std::set<Value>'s
  // iteration (SVM reads the target's valid Values in that order).
  static std::vector<Value> by_key(const std::vector<Value>& members) {
    std::vector<Value> sorted = members;
    std::stable_sort(sorted.begin(), sorted.end(), [](const Value& a, const Value& b) {
      return cpmml::ContainerKey::of(a) < cpmml::ContainerKey::of(b);
    });
    sorted.erase(std::unique(sorted.begin(), sorted.end(),
                             [](const Value& a, const Value& b) {
                               return cpmml::ContainerKey::of(a) == cpmml::ContainerKey::of(b);
                             }),
                 sorted.end());
    return sorted;
  }
};

#endif
