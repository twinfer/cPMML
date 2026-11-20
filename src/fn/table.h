/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_FN_TABLE_H
#define CPMML_FN_TABLE_H

/**
 * Design note -- the built-in function table (§4.1, §4.2).
 *
 * One row per PMML 4.4 built-in (BuiltinFunctions.html lists them): name,
 * arity, parameter categories, result rule, missing rule and the kernel. It
 * is the only table: the binder reads it to type an Apply (bind/binder.h,
 * bind/typing.h, bind/load_scope.h), and the engine's Apply (fn/apply_node.h)
 * reads the same row to check the name and the arity and to pick its kernel
 * (fn/kernels.h), once, at load (design commit P4.1; until then the binder's
 * typing table lived in bind/function_table.h and the engine had its own in
 * core/builtinfunctions.h).
 *
 * Rows cover every function of BuiltinFunctions.html, including those the
 * engine refuses today (isValid, isNotValid, formatNumber,
 * normalIDF, stdNormalIDF, the date functions): typing them is independent of
 * evaluating them. `engine` records where the engine stands (§4.2 status);
 * `op` names the kernel, and a row has one exactly when the engine evaluates
 * it today (checked below).
 *
 * Row order: the rows the engine evaluates come first, in the order of the
 * table core/builtinfunctions.h had, because the load error for a function
 * the engine refuses lists them in that order ('unsupported value "and" for
 * Apply@function; cPMML supports +, -, *, ...'), and that text is a cell of
 * the fixture dumps that P4.1 keeps unchanged; the functions added since
 * (and, or: design item P4.8) follow them, so that list only grows at its
 * end. Then the rows the engine refuses, in that table's order too.
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
 *   op          the kernel fn/kernels.h instantiates for the row (§4.1
 *               `instantiate`; as built in P4.1 an id that kernels.h
 *               switches on, so that this header, which the binder includes,
 *               does not depend on the engine's Value); Op::None where the
 *               engine refuses the function.
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

// The kernel of a row (fn/kernels.h). One per function the engine evaluates.
enum class Op : std::uint8_t {
  None,  // the engine refuses the function (Engine is not Today)
  Plus,
  Minus,
  Times,
  Divide,
  Max,
  Min,
  Sum,
  Avg,
  Exp,
  Ln,
  Log10,
  Sqrt,
  Abs,
  Round,
  Floor,
  Ceil,
  Pow,
  If,
  IsMissing,
  IsNotMissing,
  Equal,
  NotEqual,
  LessThan,
  LessOrEqual,
  GreaterThan,
  GreaterOrEqual,
  IsIn,
  IsNotIn,
  Replace,
  Matches,
  Uppercase,
  Lowercase,
  Substring,
  TrimBlanks,
  Concat,
  StringLength,
  Median,
  Product,
  Threshold,
  Modulo,
  Not,
  Expm1,
  Ln1p,
  Hypot,
  Rint,
  Sin,
  Asin,
  Sinh,
  Cos,
  Acos,
  Cosh,
  Tan,
  Atan,
  Tanh,
  Erf,
  NormalCDF,
  NormalPDF,
  StdNormalCDF,
  StdNormalPDF,
  And,  // P4.8
  Or,   // P4.8
};

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
  Op op;

  constexpr Param param(std::size_t i) const { return i < params.size() ? params[i] : rest; }
};

namespace detail {
using P = Param;
using R = Result;
using M = MissingRule;
using E = Engine;
using T = types::TypeId;
constexpr Signature unary(std::string_view n, Op op, P p, R r, T t) {
  return {n, 1, 1, {p, p, p}, p, r, t, M::Propagate, E::Today, op};
}
constexpr Signature binary(std::string_view n, Op op, P p, R r, T t) {
  return {n, 2, 2, {p, p, p}, p, r, t, M::Propagate, E::Today, op};
}
constexpr Signature aggregate(std::string_view n, Op op, R r) {
  return {n,  1, kVariadic, {P::Numeric, P::Numeric, P::Numeric}, P::Numeric, r, T::Double, M::SkipAggregate,
          E::Today, op};
}
constexpr Signature variadic(std::string_view n, Op op, std::uint8_t min, P p, T t, E e = E::Today) {
  return {n, min, kVariadic, {p, p, p}, p, R::Fixed, t, M::Propagate, e, op};
}
constexpr Signature fixed(std::string_view n, Op op, std::uint8_t arity, std::array<P, 3> ps, P rest, T t,
                          E e = E::Today) {
  return {n, arity, arity, ps, rest, R::Fixed, t, M::Propagate, e, op};
}
constexpr Signature receive(std::string_view n, Op op, P p, E e = E::Today) {
  return {n, 1, 1, {p, p, p}, p, R::Fixed, T::Boolean, M::Receive, e, op};
}
}  // namespace detail

// §4.2. The rows the engine evaluates, in the engine's historical order (see
// the note above), then the rows it refuses.
inline constexpr Signature kBuiltins[] = {
    // +, -, * and /
    detail::binary("+", Op::Plus, Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::binary("-", Op::Minus, Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::binary("*", Op::Times, Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::binary("/", Op::Divide, Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    // max, min, sum, avg
    detail::aggregate("max", Op::Max, Result::LeastRestrictive),
    detail::aggregate("min", Op::Min, Result::LeastRestrictive),
    detail::aggregate("sum", Op::Sum, Result::LeastRestrictive),
    detail::aggregate("avg", Op::Avg, Result::AtLeastDouble),
    // exp, ln, log10, sqrt, abs, round, floor, ceil, pow
    detail::unary("exp", Op::Exp, Param::Numeric, Result::Fixed, types::TypeId::Double),
    detail::unary("ln", Op::Ln, Param::Numeric, Result::Fixed, types::TypeId::Double),
    detail::unary("log10", Op::Log10, Param::Numeric, Result::Fixed, types::TypeId::Double),
    detail::unary("sqrt", Op::Sqrt, Param::Numeric, Result::Fixed, types::TypeId::Double),
    detail::unary("abs", Op::Abs, Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::unary("round", Op::Round, Param::Numeric, Result::Fixed, types::TypeId::Integer),
    detail::unary("floor", Op::Floor, Param::Numeric, Result::Fixed, types::TypeId::Integer),
    detail::unary("ceil", Op::Ceil, Param::Numeric, Result::Fixed, types::TypeId::Integer),
    detail::binary("pow", Op::Pow, Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    // if
    {"if",
     2,
     3,
     {Param::Boolean, Param::Any, Param::Any},
     Param::Any,
     Result::IfBranches,
     types::TypeId::Double,
     MissingRule::LazyIf,
     Engine::Today,
     Op::If},
    // isMissing, isNotMissing
    detail::receive("isMissing", Op::IsMissing, Param::Any),
    detail::receive("isNotMissing", Op::IsNotMissing, Param::Any),
    // equal, notEqual, lessThan, lessOrEqual, greaterThan, greaterOrEqual
    detail::binary("equal", Op::Equal, Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    detail::binary("notEqual", Op::NotEqual, Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    detail::binary("lessThan", Op::LessThan, Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    detail::binary("lessOrEqual", Op::LessOrEqual, Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    detail::binary("greaterThan", Op::GreaterThan, Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    detail::binary("greaterOrEqual", Op::GreaterOrEqual, Param::Comparable, Result::Fixed, types::TypeId::Boolean),
    // isIn, isNotIn
    detail::variadic("isIn", Op::IsIn, 2, Param::Comparable, types::TypeId::Boolean),
    detail::variadic("isNotIn", Op::IsNotIn, 2, Param::Comparable, types::TypeId::Boolean),
    // replace, matches
    detail::fixed("replace", Op::Replace, 3, {Param::String, Param::String, Param::String}, Param::String,
                  types::TypeId::String),
    detail::fixed("matches", Op::Matches, 2, {Param::String, Param::String, Param::String}, Param::String,
                  types::TypeId::Boolean),
    // uppercase, lowercase, substring, trimBlanks, concat, stringLength
    detail::unary("uppercase", Op::Uppercase, Param::String, Result::Fixed, types::TypeId::String),
    detail::unary("lowercase", Op::Lowercase, Param::String, Result::Fixed, types::TypeId::String),
    detail::fixed("substring", Op::Substring, 3, {Param::String, Param::Integer, Param::Integer}, Param::Integer,
                  types::TypeId::String),
    detail::unary("trimBlanks", Op::TrimBlanks, Param::String, Result::Fixed, types::TypeId::String),
    detail::variadic("concat", Op::Concat, 2, Param::Any, types::TypeId::String),
    detail::unary("stringLength", Op::StringLength, Param::String, Result::Fixed, types::TypeId::Integer),
    // median, product, threshold, modulo, not
    detail::aggregate("median", Op::Median, Result::AtLeastDouble),
    detail::aggregate("product", Op::Product, Result::LeastRestrictive),
    detail::binary("threshold", Op::Threshold, Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::binary("modulo", Op::Modulo, Param::Numeric, Result::LeastRestrictive, types::TypeId::Double),
    detail::unary("not", Op::Not, Param::Boolean, Result::Fixed, types::TypeId::Boolean),
    // expm1, ln1p, hypot, rint
    detail::unary("expm1", Op::Expm1, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("ln1p", Op::Ln1p, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::binary("hypot", Op::Hypot, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("rint", Op::Rint, Param::Double, Result::Fixed, types::TypeId::Double),
    // sin, asin, sinh, cos, acos, cosh, tan, atan, tanh
    detail::unary("sin", Op::Sin, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("asin", Op::Asin, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("sinh", Op::Sinh, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("cos", Op::Cos, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("acos", Op::Acos, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("cosh", Op::Cosh, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("tan", Op::Tan, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("atan", Op::Atan, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("tanh", Op::Tanh, Param::Double, Result::Fixed, types::TypeId::Double),
    // erf, normalCDF, normalPDF, stdNormalCDF, stdNormalPDF
    detail::unary("erf", Op::Erf, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::fixed("normalCDF", Op::NormalCDF, 3, {Param::Double, Param::Double, Param::Double}, Param::Double,
                  types::TypeId::Double),
    detail::fixed("normalPDF", Op::NormalPDF, 3, {Param::Double, Param::Double, Param::Double}, Param::Double,
                  types::TypeId::Double),
    detail::unary("stdNormalCDF", Op::StdNormalCDF, Param::Double, Result::Fixed, types::TypeId::Double),
    detail::unary("stdNormalPDF", Op::StdNormalPDF, Param::Double, Result::Fixed, types::TypeId::Double),
    // and, or (design item P4.8): "Evaluate the results of two or more
    // boolean values"; no missing rule of their own, so "any missing inputs
    // ... will result in a missing value being returned" (Propagate)
    detail::variadic("and", Op::And, 2, Param::Boolean, types::TypeId::Boolean),
    detail::variadic("or", Op::Or, 2, Param::Boolean, types::TypeId::Boolean),

    // ---- refused by the engine (load error), typed by the binder
    // isValid, isNotValid
    detail::receive("isValid", Op::None, Param::FieldRef, Engine::P7),
    detail::receive("isNotValid", Op::None, Param::FieldRef, Engine::P7),
    // normalIDF, stdNormalIDF
    detail::fixed("normalIDF", Op::None, 3, {Param::Double, Param::Double, Param::Double}, Param::Double,
                  types::TypeId::Double, Engine::Unplanned),
    detail::fixed("stdNormalIDF", Op::None, 1, {Param::Double, Param::Double, Param::Double}, Param::Double,
                  types::TypeId::Double, Engine::Unplanned),
    // formatNumber, formatDatetime, dateDaysSinceYear, dateSecondsSinceYear,
    // dateSecondsSinceMidnight
    detail::fixed("formatNumber", Op::None, 2, {Param::Numeric, Param::String, Param::String}, Param::String,
                  types::TypeId::String, Engine::P4),
    detail::fixed("formatDatetime", Op::None, 2, {Param::Temporal, Param::String, Param::String}, Param::String,
                  types::TypeId::String, Engine::P10),
    detail::fixed("dateDaysSinceYear", Op::None, 2, {Param::Temporal, Param::Integer, Param::Integer},
                  Param::Integer, types::TypeId::Integer, Engine::P10),
    detail::fixed("dateSecondsSinceYear", Op::None, 2, {Param::Temporal, Param::Integer, Param::Integer},
                  Param::Integer, types::TypeId::Integer, Engine::P10),
    detail::fixed("dateSecondsSinceMidnight", Op::None, 1, {Param::Temporal, Param::Temporal, Param::Temporal},
                  Param::Temporal, types::TypeId::Integer, Engine::P10),
};

// A row has a kernel exactly when the engine evaluates the function, and the
// rows the engine evaluates come first (the order of the load error's list).
constexpr bool kernels_match_engine() {
  bool refused_seen = false;
  for (const auto& s : kBuiltins) {
    if ((s.engine == Engine::Today) != (s.op != Op::None)) return false;
    if (s.engine != Engine::Today) refused_seen = true;
    else if (refused_seen) return false;
  }
  return true;
}
static_assert(kernels_match_engine(), "fn::kBuiltins: a row has a kernel exactly when the engine evaluates it");

// No two rows share a name or a kernel.
constexpr bool rows_distinct() {
  for (std::size_t i = 0; i < std::size(kBuiltins); i++)
    for (std::size_t j = i + 1; j < std::size(kBuiltins); j++) {
      if (kBuiltins[i].name == kBuiltins[j].name) return false;
      if (kBuiltins[i].op != Op::None && kBuiltins[i].op == kBuiltins[j].op) return false;
    }
  return true;
}
static_assert(rows_distinct(), "fn::kBuiltins: names and kernels are distinct");

// The row of a built-in, or nullptr. Names are case-sensitive.
constexpr const Signature* find(std::string_view name) {
  for (const auto& s : kBuiltins)
    if (s.name == name) return &s;
  return nullptr;
}

// Names that are not built-ins but are often meant as one; the binder and the
// engine's load (fn/apply_node.h) give the same hints.
constexpr std::string_view hint_for(std::string_view name) {
  if (name == "log") return "the natural logarithm is \"ln\" and the base-10 logarithm is \"log10\"";
  if (name == "ceiling") return "it is \"ceil\"";
  return {};
}

}  // namespace cpmml::fn

#endif  // CPMML_FN_TABLE_H
