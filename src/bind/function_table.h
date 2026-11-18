/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BIND_FUNCTION_TABLE_H
#define CPMML_BIND_FUNCTION_TABLE_H

/**
 * Design note -- the built-in function table, typing part (§4.1, §4.2).
 *
 * One row per PMML 4.4 built-in (BuiltinFunctions.html lists them): name,
 * arity, parameter categories, result rule and missing rule. The binder reads
 * it to type an Apply. P4 adds the kernel factory (§4.1 `instantiate`) and
 * moves the table to src/fn/; P1.2 needs only the typing columns, so there is
 * no kernel here and nothing in the engine reads it.
 *
 * Rows cover every function of BuiltinFunctions.html, including those the
 * engine refuses today (and, or, isValid, isNotValid, formatNumber,
 * normalIDF, stdNormalIDF, the date functions): typing them is independent of
 * evaluating them. `engine` records where the engine stands (§4.2 status).
 *
 * Spec basis of the columns:
 *
 *   arity       the pseudo-declarations' ParameterFields; "a variable number
 *               of <FieldRef/>" for the aggregates; "two or more parameters"
 *               (concat), "two or more boolean values" (and, or)
 *   params      a ParameterField's dataType where the pseudo-declaration
 *               gives one (not, if: boolean; substring: integer startPos and
 *               length; stringLength, uppercase, ...: string; the
 *               distribution functions, expm1, hypot, ln1p, rint and the
 *               trigonometric functions: double). Transformations.html: "If
 *               the ParameterField definition includes a dataType, the
 *               Constant will inherit the dataType specified in the
 *               ParameterField" (review S2). Numeric: the arithmetic
 *               functions, whose ParameterFields are optype="continuous".
 *   result      Functions.html: "functions are assumed to inherit the
 *               dataType of the current input parameters unless otherwise
 *               specified ... the least restrictive dataType will be
 *               inherited by default" (LeastRestrictive); a declared
 *               dataType is Fixed ("Functions floor, ceil, and round return
 *               an integer"). AtLeastDouble (avg, median) and the double
 *               result of log10/ln/sqrt/exp follow §4.2 (JPMML basis,
 *               shipped). IfBranches is D7.
 *   missing     BuiltinFunctions.html: "Except as noted below, any missing
 *               inputs to a built-in function will result in a missing value
 *               being returned"; the aggregates: "Missing values in the
 *               input to an aggregate function are simply ignored".
 */

#include <array>
#include <cstdint>
#include <string_view>

#include "core/types/value.h"

namespace cpmml::fn {

// Parameter categories (§4.1), plus Temporal for the date functions, which
// the binder cannot check until P10 adds the temporal alternatives, and
// FieldRef for isValid/isNotValid ("ParameterField name=\"field\"": the
// validity of a field, §4.2 "field reference").
enum class Param : std::uint8_t { Any, Comparable, Numeric, Integer, Double, String, Boolean, Temporal, FieldRef };

enum class Result : std::uint8_t {
  Fixed,             // the declared type in `fixed`
  LeastRestrictive,  // §2.5 over the numeric arguments
  AtLeastDouble,     // LeastRestrictive, integer -> double (avg, median)
  IfBranches,        // common type of THEN and ELSE (D7)
};

enum class MissingRule : std::uint8_t {
  Propagate,      // "any missing inputs ... will result in a missing value"
  SkipAggregate,  // "Missing values in the input to an aggregate function are simply ignored"
  Receive,        // isMissing, isNotMissing, isValid, isNotValid
  LazyIf,         // only the condition and the branch taken are inputs
};

// Where the engine stands on the function today (§4.2 "status").
enum class Engine : std::uint8_t { Today, P4, P7, P10, Unplanned };

inline constexpr std::uint8_t kVariadic = 255;

struct Signature {
  std::string_view name;
  std::uint8_t min_args, max_args;
  std::array<Param, 3> params;  // per position
  Param rest;                   // every position after the third
  Result result;
  types::TypeId fixed;  // for Result::Fixed
  MissingRule missing;
  Engine engine;

  constexpr Param param(std::size_t i) const { return i < params.size() ? params[i] : rest; }
};

namespace detail {
using P = Param;
using R = Result;
using M = MissingRule;
using E = Engine;
using T = types::TypeId;
constexpr Signature unary(std::string_view n, P p, R r, T t, E e = E::Today) {
  return {n, 1, 1, {p, p, p}, p, r, t, M::Propagate, e};
}
constexpr Signature binary(std::string_view n, P p, R r, T t, E e = E::Today) {
  return {n, 2, 2, {p, p, p}, p, r, t, M::Propagate, e};
}
constexpr Signature aggregate(std::string_view n, R r) {
  return {n, 1, kVariadic, {P::Numeric, P::Numeric, P::Numeric}, P::Numeric, r, T::Double, M::SkipAggregate, E::Today};
}
}  // namespace detail

// §4.2, in BuiltinFunctions.html's order.
inline constexpr Signature kBuiltins[] = {
    // +, -, * and /
    detail::binary("+", Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::binary("-", Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::binary("*", Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::binary("/", Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    // min, max, sum, avg, median, product
    detail::aggregate("min", Result::LeastRestrictive),
    detail::aggregate("max", Result::LeastRestrictive),
    detail::aggregate("sum", Result::LeastRestrictive),
    detail::aggregate("avg", Result::AtLeastDouble),
    detail::aggregate("median", Result::AtLeastDouble),
    detail::aggregate("product", Result::LeastRestrictive),
    // log10, ln, sqrt, abs, exp, pow, threshold, floor, ceil, round, modulo
    detail::unary("log10", Param::Numeric, Result::Fixed, types::TypeId::Double),
    detail::unary("ln", Param::Numeric, Result::Fixed, types::TypeId::Double),
    detail::unary("sqrt", Param::Numeric, Result::Fixed, types::TypeId::Double),
    detail::unary("abs", Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::unary("exp", Param::Numeric, Result::Fixed, types::TypeId::Double),
    detail::binary("pow", Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::binary("threshold", Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::unary("floor", Param::Numeric, Result::Fixed, types::TypeId::Integer),
    detail::unary("ceil", Param::Numeric, Result::Fixed, types::TypeId::Integer),
    detail::unary("round", Param::Numeric, Result::Fixed, types::TypeId::Integer),
    detail::binary("modulo", Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    // isMissing, isNotMissing, isValid, isNotValid
    {"isMissing",
     1,
     1,
     {Param::Any, Param::Any, Param::Any},
     Param::Any,
     Result::Fixed,
     types::TypeId::Boolean,
     MissingRule::Receive,
     Engine::Today},
    {"isNotMissing",
     1,
     1,
     {Param::Any, Param::Any, Param::Any},
     Param::Any,
     Result::Fixed,
     types::TypeId::Boolean,
     MissingRule::Receive,
     Engine::Today},
    {"isValid",
     1,
     1,
     {Param::FieldRef, Param::FieldRef, Param::FieldRef},
     Param::FieldRef,
     Result::Fixed,
     types::TypeId::Boolean,
     MissingRule::Receive,
     Engine::P7},
    {"isNotValid",
     1,
     1,
     {Param::FieldRef, Param::FieldRef, Param::FieldRef},
     Param::FieldRef,
     Result::Fixed,
     types::TypeId::Boolean,
     MissingRule::Receive,
     Engine::P7},
    // equal, notEqual, lessThan, lessOrEqual, greaterThan, greaterOrEqual
    detail::binary("equal", Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    detail::binary("notEqual", Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    detail::binary("lessThan", Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    detail::binary("lessOrEqual", Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    detail::binary("greaterThan", Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    detail::binary("greaterOrEqual", Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    // and, or
    {"and",
     2,
     kVariadic,
     {Param::Boolean, Param::Boolean, Param::Boolean},
     Param::Boolean,
     Result::Fixed,
     types::TypeId::Boolean,
     MissingRule::Propagate,
     Engine::P4},
    {"or",
     2,
     kVariadic,
     {Param::Boolean, Param::Boolean, Param::Boolean},
     Param::Boolean,
     Result::Fixed,
     types::TypeId::Boolean,
     MissingRule::Propagate,
     Engine::P4},
    // not
    detail::unary("not", Param::Boolean, Result::Fixed, types::TypeId::Boolean),
    // isIn, isNotIn
    {"isIn",
     2,
     kVariadic,
     {Param::Comparable, Param::Comparable, Param::Comparable},
     Param::Comparable,
     Result::Fixed,
     types::TypeId::Boolean,
     MissingRule::Propagate,
     Engine::Today},
    {"isNotIn",
     2,
     kVariadic,
     {Param::Comparable, Param::Comparable, Param::Comparable},
     Param::Comparable,
     Result::Fixed,
     types::TypeId::Boolean,
     MissingRule::Propagate,
     Engine::Today},
    // if
    {"if",
     2,
     3,
     {Param::Boolean, Param::Any, Param::Any},
     Param::Any,
     Result::IfBranches,
     types::TypeId::Double,
     MissingRule::LazyIf,
     Engine::Today},
    // string functions
    detail::unary("uppercase", Param::String, Result::Fixed, types::TypeId::String),
    detail::unary("lowercase", Param::String, Result::Fixed, types::TypeId::String),
    detail::unary("stringLength", Param::String, Result::Fixed, types::TypeId::Integer),
    {"substring",
     3,
     3,
     {Param::String, Param::Integer, Param::Integer},
     Param::Integer,
     Result::Fixed,
     types::TypeId::String,
     MissingRule::Propagate,
     Engine::Today},
    detail::unary("trimBlanks", Param::String, Result::Fixed, types::TypeId::String),
    {"concat",
     2,
     kVariadic,
     {Param::Any, Param::Any, Param::Any},
     Param::Any,
     Result::Fixed,
     types::TypeId::String,
     MissingRule::Propagate,
     Engine::Today},
    {"replace",
     3,
     3,
     {Param::String, Param::String, Param::String},
     Param::String,
     Result::Fixed,
     types::TypeId::String,
     MissingRule::Propagate,
     Engine::Today},
    {"matches",
     2,
     2,
     {Param::String, Param::String, Param::String},
     Param::String,
     Result::Fixed,
     types::TypeId::Boolean,
     MissingRule::Propagate,
     Engine::Today},
    {"formatNumber",
     2,
     2,
     {Param::Numeric, Param::String, Param::String},
     Param::String,
     Result::Fixed,
     types::TypeId::String,
     MissingRule::Propagate,
     Engine::P4},
    {"formatDatetime",
     2,
     2,
     {Param::Temporal, Param::String, Param::String},
     Param::String,
     Result::Fixed,
     types::TypeId::String,
     MissingRule::Propagate,
     Engine::P10},
    {"dateDaysSinceYear",
     2,
     2,
     {Param::Temporal, Param::Integer, Param::Integer},
     Param::Integer,
     Result::Fixed,
     types::TypeId::Integer,
     MissingRule::Propagate,
     Engine::P10},
    {"dateSecondsSinceYear",
     2,
     2,
     {Param::Temporal, Param::Integer, Param::Integer},
     Param::Integer,
     Result::Fixed,
     types::TypeId::Integer,
     MissingRule::Propagate,
     Engine::P10},
    detail::unary("dateSecondsSinceMidnight", Param::Temporal, Result::Fixed, types::TypeId::Integer, Engine::P10),
    // distributions
    {"normalCDF",
     3,
     3,
     {Param::Double, Param::Double, Param::Double},
     Param::Double,
     Result::Fixed,
     types::TypeId::Double,
     MissingRule::Propagate,
     Engine::Today},
    {"normalPDF",
     3,
     3,
     {Param::Double, Param::Double, Param::Double},
     Param::Double,
     Result::Fixed,
     types::TypeId::Double,
     MissingRule::Propagate,
     Engine::Today},
    detail::unary("stdNormalCDF", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("stdNormalPDF", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("erf", Param::Double, Result::Fixed, types::TypeId::Double),
    {"normalIDF",
     3,
     3,
     {Param::Double, Param::Double, Param::Double},
     Param::Double,
     Result::Fixed,
     types::TypeId::Double,
     MissingRule::Propagate,
     Engine::Unplanned},
    detail::unary("stdNormalIDF", Param::Double, Result::Fixed, types::TypeId::Double, Engine::Unplanned),
    // expm1, hypot, ln1p, rint
    detail::unary("expm1", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::binary("hypot", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("ln1p", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("rint", Param::Double, Result::Fixed, types::TypeId::Double),
    // sin, asin, sinh, cos, acos, cosh, tan, atan, tanh
    detail::unary("sin", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("asin", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("sinh", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("cos", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("acos", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("cosh", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("tan", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("atan", Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("tanh", Param::Double, Result::Fixed, types::TypeId::Double),
};

// The row of a built-in, or nullptr. Names are case-sensitive.
constexpr const Signature* find(std::string_view name) {
  for (const auto& s : kBuiltins)
    if (s.name == name) return &s;
  return nullptr;
}

// Names that are not built-ins but are often meant as one; the engine gives
// the same hints (core/builtinfunctions.h).
constexpr std::string_view hint_for(std::string_view name) {
  if (name == "log") return "the natural logarithm is \"ln\" and the base-10 logarithm is \"log10\"";
  if (name == "ceiling") return "it is \"ceil\"";
  return {};
}

}  // namespace cpmml::fn

#endif  // CPMML_BIND_FUNCTION_TABLE_H
