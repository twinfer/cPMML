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
 * A comparison reads the operand's LegacyCompare key (switch 2, core/value.h):
 * a number itself, a string its code. `type` is the type of the predicate's
 * value; equal and notEqual with a string value read it to compare a string
 * operand's code as an integer (the same answer, switch 2). P6.3 turns this
 * into the typed comparisons, one switch over op x type.
 */
using PredicateResult = PredicateTruth;

struct BoundPredicate {
  std::uint8_t op;              // PredicateOpType::PredicateOpTypeValue
  cpmml::types::TypeId type;    // the type of its value (an ordered set's first member; else Missing)
  std::uint8_t kind;            // what `operand` indexes (Kind)
  std::uint32_t slot;           // the sample slot compared; a compound's number of operands
  std::uint32_t operand;        // constants_, sets_, or a compound's first operand in children_

  enum Kind : std::uint8_t { Constant, Simple, Set, Compound };
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
        b.kind = BoundPredicate::Set;
        b.slot = narrow(p.feature);
        b.operand = narrow(sets_.size());
        b.type = p.values.empty() ? cpmml::types::TypeId::Missing : cpmml::types::type_of(p.values.front().typed());
        sets_.push_back(p.keys);
        break;
      default:
        b.kind = BoundPredicate::Simple;
        b.slot = narrow(p.feature);
        b.operand = narrow(constants_.size());
        b.type = cpmml::types::type_of(p.value.typed());
        constants_.push_back(p.key);
        codes_.push_back(p.value.legacy_code());
        break;
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
        return simple(p, sample, [](double x, double c) { return !(x == c); });
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
  // A comparison of the slot's LegacyCompare key with the constant
  // (cpmml::predicate::compare, switch 2): UNKNOWN for a missing value.
  template <class Compare>
  [[gnu::always_inline]] inline PredicateResult simple(const BoundPredicate& p, const Sample& sample,
                                                       Compare compare) const {
    const Value& x = sample[p.slot].value;
    if (x.is_missing()) return PredicateResult::UNKNOWN;
    return compare(cpmml::legacy::LegacyCompare::key(x), constants_[p.operand]) ? PredicateResult::TRUE_
                                                                                 : PredicateResult::FALSE_;
  }

  // equal / notEqual with a string value (the op x type case of §6.4): a
  // string operand compares its code with the value's, which is the
  // comparison of their LegacyCompare keys (switch 2: the code -> key map is
  // one to one); an operand of any other type compares its key (B18).
  [[gnu::always_inline]] inline PredicateResult string_equality(const BoundPredicate& p, const Sample& sample,
                                                                const bool equal) const {
    const Value& x = sample[p.slot].value;
    if (x.is_missing()) return PredicateResult::UNKNOWN;
    const bool same = std::holds_alternative<cpmml::types::String>(x.typed())
                          ? x.legacy_code() == codes_[p.operand]
                          : cpmml::legacy::LegacyCompare::key(x) == constants_[p.operand];
    return same == equal ? PredicateResult::TRUE_ : PredicateResult::FALSE_;
  }

  // SimpleSetPredicate: membership of the slot's LegacyCompare key; UNKNOWN
  // for a missing value.
  [[gnu::always_inline]] inline PredicateResult member(const BoundPredicate& p, const Sample& sample,
                                                       const bool is_in) const {
    const Value& x = sample[p.slot].value;
    if (x.is_missing()) return PredicateResult::UNKNOWN;
    return sets_[p.operand].contains(cpmml::legacy::LegacyCompare::key(x)) == is_in ? PredicateResult::TRUE_
                                                                                     : PredicateResult::FALSE_;
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
  std::vector<double> constants_;      // LegacyCompare keys of the SimplePredicate values (switch 2)
  std::vector<std::uint32_t> codes_;   // their legacy codes, read for a string value (switch 2)
  std::vector<cpmml::predicate::LegacyKeySet> sets_;
};

#endif
