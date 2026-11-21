/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_FASTPREDICATE_H
#define CPMML_FASTPREDICATE_H

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "predicate.h"
#include "predicateoptype.h"
#include "sample.h"

/**
 * Bound predicates (docs/design/typed-core.md §6.4, P5): a TreeModel stores
 * the predicates of its nodes as flat arrays, one BoundPredicate per
 * predicate, and evaluates them with one switch over the operator; no
 * std::function, no bound Value copy (review N9).
 *
 * Like Predicate these are three-valued: a comparison against a missing value
 * is UNKNOWN, not false. The tree traversal needs that distinction to apply
 * missingValueStrategy.
 *
 * The comparisons are typed (P6.3, §2.7, core/compare.h), with one switch
 * over the operator and, inline, the two cases a tree is made of: a number
 * operand against a finite number value, compared as doubles (the promoted
 * type of integer or float and double; exact, but for integer x float, which
 * compares in float, and integers beyond 2^53, which take the typed path),
 * NaN UNKNOWN (D9); and equal / notEqual of a string operand with a string
 * value, by rep (canonical reps, P6.1). `type` is the type of the
 * predicate's value. Every other pair (a ±INF or NaN value, D9 and B44; a
 * boolean; an ordering of strings; an ordering of an ordinal field that
 * lists Values, of any type, in their declared order, P6.12; a string
 * against a number, D2) goes out of line to predicate::typed_compare,
 * as do predicates whose op carries kTyped. A SimpleSetPredicate is
 * predicate::KeySet::member.
 */
using PredicateResult = PredicateTruth;

struct BoundPredicate {
  std::uint8_t op;              // PredicateOpType::PredicateOpTypeValue
  cpmml::types::TypeId type;    // the type of its value (an ordered set's first member; else Missing)
  std::uint8_t kind;            // what `operand` indexes (Kind)
  std::uint32_t slot;           // the sample slot compared; a compound's number of operands
  std::uint32_t operand;        // constants_, sets_, or a compound's first operand in children_

  enum Kind : std::uint8_t { Constant, Simple, Set, Compound };
  // Set on `op` for a comparison evaluated out of line (typed_compare).
  static constexpr std::uint8_t kTyped = 0x80;
};

class BoundPredicates {
 public:
  using Op = PredicateOpType::PredicateOpTypeValue;

  // Compiles `p` and returns its index.
  std::uint32_t add(const Predicate& p) {
    BoundPredicate b{static_cast<std::uint8_t>(p.predicatetype.value), cpmml::types::TypeId::Missing,
                     BoundPredicate::Constant, 0, 0};
    switch (p.predicatetype.value) {
      case Op::TRUE:
      case Op::FALSE:
        break;
      case Op::AND:
      case Op::OR:
      case Op::XOR:
      case Op::SURROGATE: {
        std::vector<std::uint32_t> operands;
        operands.reserve(p.predicates.size());
        for (const Predicate& child : p.predicates) operands.push_back(add(child));
        b.kind = BoundPredicate::Compound;
        b.slot = narrow(operands.size());
        b.operand = narrow(children_.size());
        children_.insert(children_.end(), operands.begin(), operands.end());
        break;
      }
      case Op::IS_MISSING:
      case Op::IS_NOT_MISSING:
        b.kind = BoundPredicate::Simple;
        b.slot = narrow(p.feature);
        break;
      case Op::IS_IN:
      case Op::IS_NOT_IN:
        if (!p.typed) throw std::logic_error("BoundPredicates: a SimpleSetPredicate that is not typed (PredicateBuilder)");
        b.kind = BoundPredicate::Set;
        b.slot = narrow(p.feature);
        b.operand = narrow(sets_.size());
        b.type = p.values.empty() ? cpmml::types::TypeId::Missing : cpmml::types::type_of(p.values.front().typed());
        sets_.push_back(p.keys);
        break;
      default: {
        if (!p.typed) throw std::logic_error("BoundPredicates: a SimplePredicate that is not typed (PredicateBuilder)");
        b.kind = BoundPredicate::Simple;
        b.slot = narrow(p.feature);
        b.operand = narrow(constants_.size());
        b.type = cpmml::types::type_of(p.value.typed());
        constants_.push_back(p.key);
        values_.push_back(p.value);
        ordinals_.push_back(p.ordinal);
        // Predicate's constructor took ContainerKey::of_constant: a string
        // value is pooled.
        const auto* str = std::get_if<cpmml::types::String>(&p.value.typed());
        reps_.push_back(str ? str->rep : nullptr);
        const bool string_equality = str && (p.predicatetype.value == Op::EQUAL || p.predicatetype.value == Op::NOT_EQUAL);
        if (!p.finite_number && !string_equality) b.op |= BoundPredicate::kTyped;
        break;
      }
    }
    code_.push_back(b);
    return narrow(code_.size() - 1);
  }

  // One switch over the operator. The comparisons and the sets are inline
  // (a tree's predicates are almost all SimplePredicates and
  // SimpleSetPredicates); the compounds and the missing tests are out of
  // line.
  [[gnu::always_inline]] inline PredicateResult evaluate(const std::uint32_t i, const Sample& sample) const {
    const BoundPredicate& p = code_[i];
    switch (static_cast<Op>(p.op)) {
      case Op::LESS_OR_EQUAL:
        return simple(p, sample, [](double x, double c) { return x <= c; });
      case Op::GREATER_THAN:
        return simple(p, sample, [](double x, double c) { return x > c; });
      case Op::LESS_THAN:
        return simple(p, sample, [](double x, double c) { return x < c; });
      case Op::GREATER_OR_EQUAL:
        return simple(p, sample, [](double x, double c) { return x >= c; });
      case Op::EQUAL:
        if (p.type == cpmml::types::TypeId::String) return string_equality(p, sample, true);
        return simple(p, sample, [](double x, double c) { return x == c; });
      case Op::NOT_EQUAL:
        if (p.type == cpmml::types::TypeId::String) return string_equality(p, sample, false);
        return simple<true>(p, sample, [](double x, double c) { return x == c; });
      case Op::IS_IN:
        return member(p, sample, true);
      case Op::IS_NOT_IN:
        return member(p, sample, false);
      case Op::TRUE:
        return PredicateResult::TRUE_;
      case Op::FALSE:
        return PredicateResult::FALSE_;
      default:
        return evaluate_other(p, sample);
    }
  }

  std::size_t size() const { return code_.size(); }

 private:
  // The operand as a double, for a comparison with a finite number value of
  // type `value_type` in double (exact, the promoted type: §2.5); false for
  // any other operand (missing, a boolean, a string, an integer against a
  // float or beyond 2^53), which takes the typed path.
  [[gnu::always_inline]] static inline bool number_of(const Value& x, const cpmml::types::TypeId value_type,
                                                      double& d) {
    using cpmml::types::TypeId;
    const cpmml::types::Value& v = x.typed();
    switch (static_cast<TypeId>(v.index())) {
      case TypeId::Double:
        d = std::get<cpmml::types::Double>(v).v;
        return true;
      case TypeId::Float:
        d = static_cast<double>(std::get<cpmml::types::Float>(v).v);
        return true;
      case TypeId::Integer: {
        if (value_type == TypeId::Float) return false;
        const std::int64_t i = std::get<cpmml::types::Integer>(v).v;
        if (value_type == TypeId::Integer && (i > (std::int64_t{1} << 53) || i < -(std::int64_t{1} << 53)))
          return false;
        d = static_cast<double>(i);
        return true;
      }
      default:
        return false;
    }
  }

  // A comparison of a number operand with the finite number value: TRUE or
  // FALSE, NaN UNKNOWN (D9; it compares false with everything, so only the
  // false branch asks). kNegate: notEqual, the negation of `compare` (==).
  // Any other operand: the typed path (UNKNOWN for a missing one).
  template <bool kNegate = false, class Compare>
  [[gnu::always_inline]] inline PredicateResult simple(const BoundPredicate& p, const Sample& sample,
                                                       Compare compare) const {
    const Value& x = sample[p.slot].value;
    double d = 0;
    if (!number_of(x, p.type, d)) [[unlikely]] {
      if (x.is_missing()) return PredicateResult::UNKNOWN;
      return typed(p, static_cast<Op>(p.op), x);
    }
    if (compare(d, constants_[p.operand])) return kNegate ? PredicateResult::FALSE_ : PredicateResult::TRUE_;
    if (d != d) [[unlikely]]
      return PredicateResult::UNKNOWN;
    return kNegate ? PredicateResult::TRUE_ : PredicateResult::FALSE_;
  }

  // equal / notEqual with a string value (the op x type case of §6.4): a
  // string operand compares its rep with the value's. That is equality of
  // their texts (P6.1): the value is pooled (ContainerKey::of_constant) and
  // an operand's rep is canonical (core/value.h), so the reps are one iff the
  // texts are equal (types::string_equal's pointer case: a rep the pool
  // lacks has a text the pool, and so the value, does not). An operand of any
  // other type takes the typed path (a number by its text, D2; B18).
  [[gnu::always_inline]] inline PredicateResult string_equality(const BoundPredicate& p, const Sample& sample,
                                                                const bool equal) const {
    const Value& x = sample[p.slot].value;
    const auto* s = std::get_if<cpmml::types::String>(&x.typed());
    if (!s) [[unlikely]]
      return typed(p, static_cast<Op>(p.op), x);
    return (s->rep == reps_[p.operand]) == equal ? PredicateResult::TRUE_ : PredicateResult::FALSE_;
  }

  // SimpleSetPredicate: predicate::KeySet::member; UNKNOWN for a missing
  // value.
  [[gnu::always_inline]] inline PredicateResult member(const BoundPredicate& p, const Sample& sample,
                                                       const bool is_in) const {
    const Value& x = sample[p.slot].value;
    if (x.is_missing()) return PredicateResult::UNKNOWN;
    const PredicateResult r = sets_[p.operand].member(x);
    if (r == PredicateResult::UNKNOWN || is_in) return r;
    return r == PredicateResult::TRUE_ ? PredicateResult::FALSE_ : PredicateResult::TRUE_;
  }

  // The typed comparison, out of line (predicate::typed_compare).
  [[gnu::noinline]] PredicateResult typed(const BoundPredicate& p, const Op op, const Value& x) const {
    if (x.is_missing()) return PredicateResult::UNKNOWN;
    return cpmml::predicate::typed_compare(op, x, values_[p.operand], ordinals_[p.operand].get());
  }

  [[gnu::noinline]] PredicateResult evaluate_other(const BoundPredicate& p, const Sample& sample) const {
    using cpmml::predicate::to_result;
    switch (static_cast<Op>(p.op)) {
      case Op::IS_MISSING:
        return to_result(sample[p.slot].value.is_missing());
      case Op::IS_NOT_MISSING:
        return to_result(!sample[p.slot].value.is_missing());
      case Op::AND:
      case Op::OR:
      case Op::XOR:
      case Op::SURROGATE: {
        const std::uint32_t* operands = children_.data() + p.operand;
        return cpmml::predicate::combine(static_cast<Op>(p.op), p.slot,
                                         [&](std::size_t k) { return evaluate(operands[k], sample); });
      }
      default:
        if (p.op & BoundPredicate::kTyped)
          return typed(p, static_cast<Op>(p.op & ~BoundPredicate::kTyped), sample[p.slot].value);
        throw std::logic_error("BoundPredicates: an operator without a case");
    }
  }

  static std::uint32_t narrow(const std::size_t n) {
    if (n >= std::numeric_limits<std::uint32_t>::max())
      throw cpmml::ParsingException("too many predicates or fields for a BoundPredicate");
    return static_cast<std::uint32_t>(n);
  }

  std::vector<BoundPredicate> code_;
  std::vector<std::uint32_t> children_;
  std::vector<double> constants_;      // the SimplePredicate values' ContainerKeys (a number itself)
  std::vector<const cpmml::types::StringRep*> reps_;  // their pooled reps, read for a string value (P6.1)
  std::vector<Value> values_;          // the values, for the typed path
  std::vector<std::shared_ptr<const cpmml::compare::OrdinalOrder>> ordinals_;  // an ordinal field's declared order (P6.12)
  std::vector<cpmml::predicate::KeySet> sets_;
};

#endif
