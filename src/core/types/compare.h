/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TYPES_COMPARE_H
#define CPMML_TYPES_COMPARE_H

/**
 * Design note -- comparison (§2.7, D1, D2, D9, B44).
 *
 * Every comparison comes from one ordering, so equal and lessOrEqual cannot
 * disagree (L4): equal is `equivalent`, lessOrEqual is `less or equivalent`,
 * and so on (relate). The ordering is std::optional<std::partial_ordering>:
 *
 *   nullopt     no common type (the binder refuses it at load; comparable()
 *               is the binder's question), or a Missing/Invalid operand
 *   unordered   indeterminate: the caller gives missing (a built-in) or
 *               UNKNOWN (a predicate)
 *
 * The rules, by operands:
 *
 *   number × number     in the promoted type (cast.h), either order (D1;
 *                       Functions.html least-restrictive rule)
 *   ±INF × the same INF unordered. Transformations.html: "infinity is not
 *                       single number, but a class of numbers ... the first
 *                       infinity is not equal to the second"; BuiltinFunctions
 *                       .html (lessThan ...): "By definition, INF is taken as
 *                       greater than any finite number or -INF and -INF is
 *                       taken as less than any finite number or INF . But any
 *                       attempt to either with itself is indeterminate and will
 *                       result in a missing value being returned." (D9, B44)
 *   NaN × anything      unordered (IEEE). An extrapolation, D9: the spec says
 *                       only "All arithmetic expressions involving NaN will
 *                       evaluate to NaN"; chosen so NaN never silently
 *                       compares false.
 *   string × number     equality only, the number as Java text, either order
 *                       (D2; string is the least restrictive type). Ordering:
 *                       no common type (DataDictionary.html "Categorical field
 *                       values can only be tested for equality"; the binder's
 *                       E-TYPE-ORDER-MIXED)
 *   string × string     text; ordering by UTF-8 code units (= code points).
 *                       Which orderings of strings are allowed (declared
 *                       Values order, the Q6 warnings) is the binder's.
 *   boolean × boolean   false < true
 *   boolean × other     no common type. Functions.html: "boolean dataTypes can
 *                       not be mixed with other types in a defined function
 *                       without explicitly defining the expected output
 *                       dataType" (Q7)
 *   any × missing       nullopt → missing / UNKNOWN. BuiltinFunctions.html:
 *                       "Except as noted below, any missing inputs to a
 *                       built-in function will result in a missing value
 *                       being returned."
 */

#include <cmath>
#include <compare>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/types/cast.h"
#include "core/types/cold.h"
#include "core/types/java_text.h"
#include "core/types/value.h"

namespace cpmml::types {

// Equality (equal, notEqual, isIn, isNotIn) or ordering (lessThan, ...).
enum class Purpose : std::uint8_t { Equality, Ordering };

enum class Relation : std::uint8_t { Equal, NotEqual, Less, LessOrEqual, Greater, GreaterOrEqual };

constexpr Purpose purpose_of(Relation r) {
  return (r == Relation::Equal || r == Relation::NotEqual) ? Purpose::Equality : Purpose::Ordering;
}

// Three-valued: a predicate's TRUE/FALSE/UNKNOWN, a built-in's true/false/missing.
enum class Tri : std::uint8_t { False, True, Unknown };

// ---------------------------------------------------------------- typed orderings (the hot path's)
template <PmmlNumeric A, PmmlNumeric B>
inline std::partial_ordering numeric_order(A a, B b) {
  using C = Promoted<A, B>;
  const C x = widen<C>(a);
  const C y = widen<C>(b);
  if constexpr (!std::is_same_v<C, Integer>)
    if (std::isinf(x.v) && x.v == y.v) return std::partial_ordering::unordered;
  return compare(x, y);
}

// A string against a number's Java text: only `equivalent` or not is
// meaningful (equality purpose).
template <PmmlNumeric N>
inline std::partial_ordering text_order(String s, N n, std::string& scratch) {
  scratch.clear();
  java_text(n, scratch);
  return s.text().compare(scratch) <=> 0;
}

// The relation from the ordering.
constexpr Tri relate(Relation r, std::optional<std::partial_ordering> o) {
  if (!o || *o == std::partial_ordering::unordered) return Tri::Unknown;
  const std::partial_ordering x = *o;
  bool holds = false;
  switch (r) {
    case Relation::Equal:
      holds = x == std::partial_ordering::equivalent;
      break;
    case Relation::NotEqual:
      holds = x != std::partial_ordering::equivalent;
      break;
    case Relation::Less:
      holds = x == std::partial_ordering::less;
      break;
    case Relation::LessOrEqual:
      holds = x == std::partial_ordering::less || x == std::partial_ordering::equivalent;
      break;
    case Relation::Greater:
      holds = x == std::partial_ordering::greater;
      break;
    case Relation::GreaterOrEqual:
      holds = x == std::partial_ordering::greater || x == std::partial_ordering::equivalent;
      break;
  }
  return holds ? Tri::True : Tri::False;
}

// Whether two static types have a common type for `purpose` (the binder's
// load-time check; the sentinels are not static types).
constexpr bool comparable(Purpose purpose, TypeId a, TypeId b) {
  auto numeric = [](TypeId t) { return t == TypeId::Integer || t == TypeId::Float || t == TypeId::Double; };
  if (a == TypeId::Missing || a == TypeId::Invalid || b == TypeId::Missing || b == TypeId::Invalid) return false;
  if (numeric(a) && numeric(b)) return true;
  if (a == b) return true;  // string × string, boolean × boolean
  if (purpose == Purpose::Equality) return (a == TypeId::String && numeric(b)) || (numeric(a) && b == TypeId::String);
  return false;
}

// ---------------------------------------------------------------- the visitor
struct Comparison {
  Purpose purpose;
  std::string* scratch;
  using Result = std::optional<std::partial_ordering>;

  template <Alternative A, Alternative B>
    requires(Sentinel<A> || Sentinel<B>)
  Result operator()(const A&, const B&) const {
    return std::nullopt;
  }
  template <PmmlNumeric A, PmmlNumeric B>
  Result operator()(const A& a, const B& b) const {
    return numeric_order(a, b);
  }
  Result operator()(const String& a, const String& b) const {
    if (purpose == Purpose::Equality) return string_equal(a, b) ? std::partial_ordering::equivalent : compare(a, b);
    return compare(a, b);
  }
  Result operator()(const Boolean& a, const Boolean& b) const { return compare(a, b); }
  template <PmmlNumeric N>
  Result operator()(const String& a, const N& b) const {
    if (purpose != Purpose::Equality) return std::nullopt;
    return text_order(a, b, *scratch);
  }
  template <PmmlNumeric N>
  Result operator()(const N& a, const String& b) const {
    if (purpose != Purpose::Equality) return std::nullopt;
    return 0 <=> text_order(b, a, *scratch);
  }
  template <Alternative A, Alternative B>
    requires((Logical<A> && (PmmlNumeric<B> || Textual<B>)) || ((PmmlNumeric<A> || Textual<A>) && Logical<B>))
  Result operator()(const A&, const B&) const {
    return std::nullopt;
  }
};
static_assert(ExhaustiveBinaryVisitor<Comparison> && NoBinaryCatchAll<Comparison>);

namespace cold {

inline std::optional<std::partial_ordering> compare(ColdPath, Purpose purpose, const Value& a, const Value& b,
                                                    std::string& scratch) {
  return std::visit(Comparison{purpose, &scratch}, a, b);
}

// equal, lessThan, ... over two values: Unknown for a sentinel operand or an
// indeterminate comparison. Operands of no common type are a binder bug
// (comparable() is false for them at load) and throw, never a silent answer.
inline Tri relate(ColdPath tag, Relation r, const Value& a, const Value& b, std::string& scratch) {
  const auto o = compare(tag, purpose_of(r), a, b, scratch);
  if (!o && !is_missing(a) && !is_invalid(a) && !is_missing(b) && !is_invalid(b))
    throw std::logic_error("cold::relate: operands have no common type for this comparison");
  return types::relate(r, o);
}

}  // namespace cold
}  // namespace cpmml::types

#endif  // CPMML_TYPES_COMPARE_H
