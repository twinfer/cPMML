/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TYPES_ARITHMETIC_H
#define CPMML_TYPES_ARITHMETIC_H

/**
 * Design note -- the arithmetic kernels' type and value rules that §2 fixes
 * (§2.5, §2.8, D4, D20, D23).
 *
 * Types. Both operands are promoted to the least restrictive of the two
 * (cast.h, Functions.html: "if an integer and a double parameter are used
 * with the "+" function, by default the output dataType will be a double");
 * the kernel runs in that type and returns it, or a sentinel.
 *
 * Integers are exact in int64; overflow is Invalid(Overflow) (§2.5, D4: "the
 * XSD integer is unbounded"; int64 is the representable subset).
 *
 * Division. BuiltinFunctions.html: "If a positive number is divided by zero,
 * INF will be returned. If zero is divided by zero, the result is
 * indeterminate and a missing value will be returned. If a negative number is
 * divided by zero, -INF will be returned." An integer has no INF
 * ("Since INF and -INF can only be floating point numbers"), so integer x/0
 * with x ≠ 0 is Invalid(DivisionByZero) (D20, XQuery err:FOAR0001); integer
 * 0/0 is missing; integer division truncates toward zero (§4.2 note, Java);
 * INT64_MIN / -1 overflows.
 *
 * NaN. Transformations.html: "All arithmetic expressions involving NaN will
 * evaluate to NaN" (D23): a NaN operand gives NaN, checked before every rule
 * below.
 *
 * Infinities. Transformations.html, "A Brief Summary of Operations Involving
 * Infinite Numbers":
 *   "The sum of two infinite numbers of opposite signs is indeterminate and
 *    will return a missing value."
 *   "The difference between two infinite numbers of the same sign is
 *    indeterminate and will return a missing value."
 *   "The product of an infinite number and zero is indeterminate and
 *    therefore will return a missing value."
 *   "An infinite number divided by another of the same sign is indeterminate
 *    and therefore will return a missing value. An infinite number divided by
 *    another of the opposite sign is infinite with the same sign as the
 *    numerator."
 * Every other case the summary lists is IEEE's answer. These rules are not in
 * the design's §2 or §4.2: IEEE gives NaN for all four (and INF/-INF's
 * "infinite" is not IEEE's NaN); JPMML 1.7.7 gives an invalid value for them
 * ("Function "/" returned invalid value", N/A under asMissing). Proposed as
 * D27.
 *
 * A zero divisor is one zero whatever its sign: x/-0.0 is INF for x > 0 and
 * -INF for x < 0, as the division sentence says (JPMML agrees; IEEE does not).
 *
 * floor, ceil. BuiltinFunctions.html: "Since INF and -INF can only be floating
 * point numbers, they will cause floor, ceil, round, and modulo to return an
 * invalid value"; the result is an integer (§4.2, "Functions floor, ceil, and
 * round return an integer"); NaN and a value beyond int64 are invalid too.
 * The remaining functions (round, modulo, pow, the aggregates ...) are P4's
 * kernels (fn/kernels.h).
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "core/types/cast.h"
#include "core/types/cold.h"
#include "core/types/value.h"

namespace cpmml::types {

template <class T>
concept FloatingValue = std::same_as<T, Float> || std::same_as<T, Double>;

// ---------------------------------------------------------------- binary kernels
struct Plus {
  static Value apply(Integer a, Integer b) {
    std::int64_t r = 0;
    if (__builtin_add_overflow(a.v, b.v, &r)) return Invalid{InvalidReason::Overflow};
    return Integer{r};
  }
  template <FloatingValue T>
  static Value apply(T a, T b) {
    if (std::isnan(a.v) || std::isnan(b.v)) return T{a.v + b.v};
    if (std::isinf(a.v) && std::isinf(b.v) && std::signbit(a.v) != std::signbit(b.v)) return Missing{};
    return T{a.v + b.v};
  }
};

struct Minus {
  static Value apply(Integer a, Integer b) {
    std::int64_t r = 0;
    if (__builtin_sub_overflow(a.v, b.v, &r)) return Invalid{InvalidReason::Overflow};
    return Integer{r};
  }
  template <FloatingValue T>
  static Value apply(T a, T b) {
    if (std::isnan(a.v) || std::isnan(b.v)) return T{a.v - b.v};
    if (std::isinf(a.v) && std::isinf(b.v) && std::signbit(a.v) == std::signbit(b.v)) return Missing{};
    return T{a.v - b.v};
  }
};

struct Times {
  static Value apply(Integer a, Integer b) {
    std::int64_t r = 0;
    if (__builtin_mul_overflow(a.v, b.v, &r)) return Invalid{InvalidReason::Overflow};
    return Integer{r};
  }
  template <FloatingValue T>
  static Value apply(T a, T b) {
    if (std::isnan(a.v) || std::isnan(b.v)) return T{a.v * b.v};
    if ((std::isinf(a.v) && b.v == 0) || (a.v == 0 && std::isinf(b.v))) return Missing{};
    return T{a.v * b.v};
  }
};

struct Divide {
  static Value apply(Integer a, Integer b) {
    if (b.v == 0) return a.v == 0 ? Value{Missing{}} : Value{Invalid{InvalidReason::DivisionByZero}};
    if (a.v == std::numeric_limits<std::int64_t>::min() && b.v == -1) return Invalid{InvalidReason::Overflow};
    return Integer{a.v / b.v};
  }
  template <FloatingValue T>
  static Value apply(T a, T b) {
    if (std::isnan(a.v) || std::isnan(b.v)) return T{a.v / b.v};
    if (std::isinf(a.v) && std::isinf(b.v)) {
      if (std::signbit(a.v) == std::signbit(b.v)) return Missing{};
      return a;  // "infinite with the same sign as the numerator"
    }
    if (a.v == 0 && b.v == 0) return Missing{};
    // The sign of the result comes from the numerator alone: the spec's "a
    // positive number ... INF", "a negative number ... -INF", over XSD's value
    // space with "only one zero" (§3.2.4). IEEE would flip it for a -0.0
    // divisor, which D18 (an input -0) and -1*0 both produce.
    if (b.v == 0) return T{a.v > 0 ? std::numeric_limits<decltype(a.v)>::infinity()
                                   : -std::numeric_limits<decltype(a.v)>::infinity()};
    return T{a.v / b.v};
  }
};

// ---------------------------------------------------------------- unary kernels
namespace detail {
template <FloatingValue T, class Round>
Value to_integer(T x, Round round) {
  if (!std::isfinite(x.v)) return Invalid{InvalidReason::Domain};
  const double r = round(x.as_double());
  if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0)) return Invalid{InvalidReason::Overflow};
  return Integer{static_cast<std::int64_t>(r)};
}
}  // namespace detail

struct Floor {
  static Value apply(Integer x) { return x; }
  template <FloatingValue T>
  static Value apply(T x) {
    return detail::to_integer(x, [](double d) { return std::floor(d); });
  }
};

struct Ceil {
  static Value apply(Integer x) { return x; }
  template <FloatingValue T>
  static Value apply(T x) {
    return detail::to_integer(x, [](double d) { return std::ceil(d); });
  }
};

// ---------------------------------------------------------------- the visitors
// A missing operand gives missing (BuiltinFunctions.html "any missing inputs
// to a built-in function will result in a missing value being returned"),
// whatever the other operand holds: the order of the §4.3 Apply wrapper,
// which returns missing before the kernel runs. Otherwise an invalid operand
// stays invalid ("An invalid value is not missing"). (An Apply argument never
// holds Invalid today, §2.1, so the order cannot show in scoring.)
// Non-numeric operands never reach a kernel: the binder refuses them at load
// (§2.5: no arithmetic accepts string; boolean "can not be mixed"), so meeting
// one is a binder bug and throws.
template <class Op>
struct Arithmetic {
  template <Alternative A, Alternative B>
    requires(Sentinel<A> || Sentinel<B>)
  Value operator()(const A& a, const B& b) const {
    if constexpr (std::is_same_v<A, Missing> || std::is_same_v<B, Missing>)
      return Missing{};
    else if constexpr (std::is_same_v<A, Invalid>)
      return a;
    else
      return b;
  }
  template <PmmlNumeric A, PmmlNumeric B>
  Value operator()(const A& a, const B& b) const {
    using C = Promoted<A, B>;
    return Op::apply(widen<C>(a), widen<C>(b));
  }
  template <Alternative A, Alternative B>
    requires(!Sentinel<A> && !Sentinel<B> && !(PmmlNumeric<A> && PmmlNumeric<B>))
  Value operator()(const A&, const B&) const {
    throw std::logic_error("arithmetic over a non-numeric operand: the binder refuses it at load");
  }
};

template <class Op>
struct Unary {
  template <Sentinel T>
  Value operator()(const T& x) const {
    return x;
  }
  template <PmmlNumeric T>
  Value operator()(const T& x) const {
    return Op::apply(x);
  }
  template <Alternative T>
    requires(!Sentinel<T> && !PmmlNumeric<T>)
  Value operator()(const T&) const {
    throw std::logic_error("arithmetic over a non-numeric operand: the binder refuses it at load");
  }
};

static_assert(ExhaustiveBinaryVisitor<Arithmetic<Plus>> && NoBinaryCatchAll<Arithmetic<Plus>>);
static_assert(ExhaustiveVisitor<Unary<Floor>> && NoCatchAll<Unary<Floor>>);

namespace cold {
template <class Op>
Value arithmetic(ColdPath, const Value& a, const Value& b) {
  return std::visit(Arithmetic<Op>{}, a, b);
}
template <class Op>
Value unary(ColdPath, const Value& x) {
  return std::visit(Unary<Op>{}, x);
}
}  // namespace cold

}  // namespace cpmml::types

#endif  // CPMML_TYPES_ARITHMETIC_H
