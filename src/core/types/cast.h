/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TYPES_CAST_H
#define CPMML_TYPES_CAST_H

/**
 * Design note -- numeric promotion (§2.5) and casts (§2.6).
 *
 * Promotion. Functions.html: "When the input parameters have multiple
 * dataTypes, the least restrictive dataType will be inherited by default ...
 * The inheritance precedence for mixed type input parameters is as follows:
 * string double float integer". Over the numbers that is one table, by rank
 * (Promoted). widen() is the implicit cast; it exists only toward less
 * restriction: "implicit casts are allowed in the direction of less
 * restriction, according to the precedence list above".
 *
 * Explicit casts (CastTo<To>), the design's table:
 *
 *   integer → float, double   widen (int64 → float rounds to nearest)
 *   float → double            exact
 *   double → float            IEEE round to nearest; overflow ±INF, underflow 0
 *                             (XSD float value space)
 *   double, float → integer   the value if integral and in int64 range, else
 *                             Invalid(NotIntegral) (NaN, a fraction) or
 *                             Invalid(Overflow) (±INF, beyond int64) --
 *                             Functions.html: "a double expression can not be
 *                             implicitly cast as an integer"
 *   number → string           Java text (Q1, java_text.h)
 *   string → number, boolean  the strict XSD reading (parse.h), else Invalid
 *   boolean → number          true 1, false 0 (XPath; shipped in b4efd7d)
 *   number → boolean          1 → true, 0 → false, else Invalid(NoCast): the
 *                             XSD boolean lexical space {true, false, 1, 0} (D14)
 *   boolean → string          "true" / "false" (XSD canonical)
 *   T → T                     identity
 *
 * Every pair of the seven alternatives is in the table; Missing and Invalid
 * cast to themselves. CastTo is called directly on a scalar where the static
 * type is known (a CastNode<From, To>); cold::cast visits a whole Value.
 */

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/types/cold.h"
#include "core/types/java_text.h"
#include "core/types/parse.h"
#include "core/types/value.h"

namespace cpmml::types {

// ---------------------------------------------------------------- promotion
template <PmmlNumeric A, PmmlNumeric B>
using Promoted = std::conditional_t<(A::rank >= B::rank), A, B>;

template <PmmlNumeric To, PmmlNumeric From>
  requires(To::rank >= From::rank)
constexpr To widen(From x) {
  if constexpr (std::is_same_v<To, From>)
    return x;
  else if constexpr (std::is_same_v<To, Float>)  // From is Integer
    return Float{static_cast<float>(x.v)};
  else
    return Double{x.as_double()};
}

// ---------------------------------------------------------------- explicit casts
template <PmmlParsedType To>
struct CastTo {
  ParseContext* ctx;     // where a String result's rep comes from
  std::string* scratch;  // number → string text

  Value operator()(const Missing&) const { return Missing{}; }
  Value operator()(const Invalid& i) const { return i; }

  template <PmmlNumeric From>
  Value operator()(const From& x) const {
    if constexpr (PmmlNumeric<To>) {
      if constexpr (To::rank >= From::rank) {
        return widen<To>(x);
      } else if constexpr (std::is_same_v<To, Float>) {  // double → float
        return Float{narrow_to_float(x.v)};
      } else {  // float, double → integer
        const double d = x.as_double();
        if (std::isnan(d)) return Invalid{InvalidReason::NotIntegral};
        if (!(d >= -9223372036854775808.0 && d < 9223372036854775808.0)) return Invalid{InvalidReason::Overflow};
        if (std::trunc(d) != d) return Invalid{InvalidReason::NotIntegral};
        return Integer{static_cast<std::int64_t>(d)};
      }
    } else if constexpr (Logical<To>) {
      const double d = x.as_double();
      if (d == 1.0) return Boolean{true};
      if (d == 0.0) return Boolean{false};
      return Invalid{InvalidReason::NoCast};
    } else {
      static_assert(Textual<To>);
      scratch->clear();
      types::java_text(x, *scratch);
      return String{make_string(*ctx, *scratch)};
    }
  }

  template <Logical From>
  Value operator()(const From& x) const {
    if constexpr (PmmlNumeric<To>)
      return (*this)(Integer{x.v ? 1 : 0});
    else if constexpr (Logical<To>)
      return x;
    else
      return String{make_string(*ctx, x.v ? "true" : "false")};
  }

  template <Textual From>
  Value operator()(const From& x) const {
    if constexpr (Textual<To>)
      return x;
    else
      return parse_as<To>(x.text(), *ctx);
  }
};

namespace cold {
template <PmmlParsedType To>
Value cast(ColdPath, const Value& v, ParseContext& ctx, std::string& scratch) {
  static_assert(ExhaustiveVisitor<CastTo<To>> && NoCatchAll<CastTo<To>>);
  return std::visit(CastTo<To>{&ctx, &scratch}, v);
}

// The cast to a type known only at run time.
inline Value cast(ColdPath tag, TypeId to, const Value& v, ParseContext& ctx, std::string& scratch) {
  switch (to) {
    case TypeId::Integer:
      return cast<Integer>(tag, v, ctx, scratch);
    case TypeId::Float:
      return cast<Float>(tag, v, ctx, scratch);
    case TypeId::Double:
      return cast<Double>(tag, v, ctx, scratch);
    case TypeId::Boolean:
      return cast<Boolean>(tag, v, ctx, scratch);
    case TypeId::String:
      return cast<String>(tag, v, ctx, scratch);
    case TypeId::Missing:
    case TypeId::Invalid:
      break;
  }
  throw std::logic_error("cold::cast: Missing and Invalid are not cast targets");
}
}  // namespace cold

}  // namespace cpmml::types

#endif  // CPMML_TYPES_CAST_H
