/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_FN_KERNELS_H
#define CPMML_FN_KERNELS_H

/**
 * Design note -- the kernels of the built-in functions (§4.1, §7 P4).
 *
 * A kernel is the function itself: it takes its arguments, already present
 * (or, for an aggregate, present or missing; for isMissing, either), and
 * returns its result, or reports the result invalid. It never throws for
 * data ("Kernels never throw for data; they return Invalid", §4.3): the
 * Apply that calls it (fn/apply_node.h) decides what an invalid result
 * becomes. One kernel per row of fn/table.h, resolved once, at load
 * (`instantiate`); scoring calls it through a function pointer, with no table
 * lookup and, for the numeric kernels, no allocation.
 *
 * Two layers:
 *
 *   scalar   the computation over the P1 scalar types (core/types/value.h):
 *            `types::Double plus(types::Double, types::Double)`, ...;
 *   adapter  how a kernel receives its operands from the engine's storage,
 *            the old `::Value` (§7 P4: "Kernels take P1 scalar types;
 *            ApplyNode unpacks its arguments ... and packs the result").
 *
 * P4.1 reproduces today's results exactly (dump diff empty), including the
 * departures from the spec that P4.2-P4.6 fix one at a time. Each departure
 * is behind a named switch in `legacy` below; the code a switch keeps is in
 * an `if constexpr (legacy::X)` branch, or names the switch in a comment
 * where the legacy rule is the absence of a check. The commit that fixes the
 * bug deletes the switch and its code. A switch cannot simply be set to
 * false: the static_asserts below stop that, because the spec rule that
 * replaces the legacy one is that commit's to write and to test.
 *
 * Operands. Today's engine carries every number as a double, an integer
 * included (D4), so a numeric operand is read as a `types::Double` of the
 * double the Value holds, whatever its static type, and narrowed to a float
 * first when its static type is float (the binder's static type, fixed at
 * load; what core/builtinfunctions.h did since P2.4). It is not
 * `unpack<types::Integer>`: that makes 3.5 in an integer slot invalid (B17,
 * P4.7), which today it is not. Comparisons delegate to `LegacyCompare`,
 * today's Value operators (the double: a number, a boolean's 1 or 0, a
 * string's Indexer code), until P6. The string edits (uppercase, lowercase,
 * substring, trimBlanks) delegate to `LegacyStringEdit`, the Value's own
 * in-place edit, which keeps the argument's string code (B2, P6) and its
 * Value: decided 2026-11-19 after P3, "a kernel never round-trips a string
 * through unpack/pack ... an argument string that the kernel passes through
 * or returns must keep the old Value it came in as; only strings the kernel
 * builds are packed" (concat, replace).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <regex>
#include <string>
#include <vector>

#include "core/types/value.h"
#include "core/value.h"
#include "fn/table.h"

namespace cpmml::fn {

// ---------------------------------------------------------------- legacy switches
// Today's behaviour where it departs from the spec, one switch per bug, each
// removed by the commit named (design §7 P4, §1.2). The behaviour a switch
// keeps is the base build's, measured by the fixture dump (P4.1: no cell
// changes).
namespace legacy {
// B41 (P4.2): built-ins outside their domain keep scoring: sqrt of a
// negative and ln/log10 of a non-positive give NaN/-INF, modulo by 0 gives
// NaN, substring with startPos < 1 returns its whole argument (spec:
// invalid, then invalidValueTreatment).
inline constexpr bool B41 = true;
// B21 (P4.4): 0/0 gives NaN (spec: missing); floor, ceil, round and modulo
// of +-INF give a number (spec: "Since INF and -INF can only be floating
// point numbers, they will cause floor, ceil, round, and modulo to return an
// invalid value").
inline constexpr bool B21 = true;
// D20 (P4.4): integer x/0 with x != 0 gives +-INF (D20: invalid; XQuery
// err:FOAR0001).
inline constexpr bool D20 = true;
// D4 (P4.5): integer arithmetic is computed in double and carried as one:
// no int64 kernel, no overflow check (D4: "integers are int64; overflow
// invalid"), floor/ceil/round give a double.
inline constexpr bool D4 = true;
// B23 (P4.6): concat reads each argument's text, which a number does not
// have, so `concat(d, "|")` is "|" (spec: the arguments are cast to string,
// Q1/D17).
inline constexpr bool B23 = true;
// B2 (P6): uppercase, lowercase, substring and trimBlanks edit a copy of
// their argument's Value, which keeps the argument's string code
// (LegacyStringEdit).
inline constexpr bool B2 = true;
// B1, B18, B44 (P6): comparisons compare the double each Value holds
// (LegacyCompare).
inline constexpr bool LegacyCompare = true;

static_assert(B41, "B41's fix removes this switch together with the code it keeps");
static_assert(B21 && D20 && D4 && B23, "P4.4-P4.6 remove these switches together with the code they keep");
static_assert(B2 && LegacyCompare, "P6 removes these switches together with the code they keep");
}  // namespace legacy

// ---------------------------------------------------------------- bound call
// What a kernel needs to know about its call, fixed at load from the
// arguments' static types (bind::expression_type, the binder's answer) and
// the result type (bind::apply_result_type, the table's result rule). Set by
// ApplyNode for the numeric functions and the comparisons; all false for the
// others (core/builtinfunctions.h's `typed()` flags, unchanged).
struct Bound {
  // A float argument is an xs:float, so it is narrowed before use (the
  // engine carries it as a double).
  bool narrow_arguments = false;
  std::vector<std::uint8_t> float_argument;  // per argument; read only when narrow_arguments
  bool float_result = false;      // the result is a float: narrowed after
  bool integer_division = false;  // "/" of two integers: truncated toward zero
  bool integer_power = false;     // pow of two integers: a negative exponent has no integer result
};

enum class Status : std::uint8_t { Ok, Invalid };

// A kernel over the engine's storage: `n` arguments at `args`, the result in
// `out` (which the caller default-constructs, i.e. missing).
using KernelFn = Status (*)(const Bound& bound, const ::Value* args, std::size_t n, ::Value& out);

// ---------------------------------------------------------------- scalar kernels
// Over the P1 scalar types. Spec text per function: BuiltinFunctions.html.
// Every expression is written as core/builtinfunctions.h wrote it, operation
// for operation, so that the compiler forms the same floating-point
// operations (contraction included) and the results are bit-identical.
namespace scalar {
using types::Double;

// "+ ... returns the sum of a and b"; "The functions -, *, and / are defined
// in the same way".
inline Double plus(Double a, Double b) { return {a.v + b.v}; }
inline Double minus(Double a, Double b) { return {a.v - b.v}; }
inline Double times(Double a, Double b) { return {a.v * b.v}; }
// Integer division truncates toward zero (§4.2; Java and XQuery's idiv).
// legacy::B21: 0/0 is NaN; legacy::D20: an integer x/0 is +-INF.
inline Double divide(Double a, Double b, bool integer) {
  double r = a.v / b.v;
  if (integer && b.v != 0.0) r = std::trunc(r);
  return {r};
}

// log10, ln, sqrt, abs, exp (legacy::B41: no domain check).
inline Double log10(Double x) { return {std::log10(x.v)}; }
inline Double ln(Double x) { return {std::log(x.v)}; }
inline Double sqrt(Double x) { return {std::sqrt(x.v)}; }
inline Double abs(Double x) { return {std::abs(x.v)}; }
inline Double exp(Double x) { return {std::exp(x.v)}; }
// "Functions floor, ceil, and round return an integer"; legacy::D4: the
// integer is carried as a double; legacy::B21: +-INF and NaN pass through.
inline Double round(Double x) { return {std::round(x.v)}; }
inline Double floor(Double x) { return {std::floor(x.v)}; }
inline Double ceil(Double x) { return {std::ceil(x.v)}; }
// pow of two integers with a negative exponent has no integer value (D8):
// invalid, so that invalidValueTreatment decides.
inline types::Value pow(Double a, Double b, bool integer) {
  if (integer && b.v < 0) return types::Invalid{types::InvalidReason::Domain};
  return Double{std::pow(a.v, b.v)};
}
// "Returns one if x>y and zero otherwise."
inline Double threshold(Double a, Double b) { return {a.v > b.v ? 1.0 : 0.0}; }
// "the remainder after x is divided by y, computed according to XQuery
// rules", a - floor(a/b)*b (the sign of the divisor). Its examples:
// modulo(11,3)=2, modulo(-17.2,0.5)=0.3, modulo(9,-7)=-5, modulo(-4,-9)=-4.
// legacy::B41: modulo by 0 is NaN; legacy::B21: an INF operand gives NaN.
inline Double modulo(Double a, Double b) { return {a.v - std::floor(a.v / b.v) * b.v}; }

// Further mathematical functions (dataType="double").
inline Double expm1(Double x) { return {std::expm1(x.v)}; }
inline Double ln1p(Double x) { return {std::log1p(x.v)}; }
inline Double hypot(Double a, Double b) { return {std::hypot(a.v, b.v)}; }
inline Double rint(Double x) { return {std::rint(x.v)}; }
inline Double sin(Double x) { return {std::sin(x.v)}; }
inline Double asin(Double x) { return {std::asin(x.v)}; }
inline Double sinh(Double x) { return {std::sinh(x.v)}; }
inline Double cos(Double x) { return {std::cos(x.v)}; }
inline Double acos(Double x) { return {std::acos(x.v)}; }
inline Double cosh(Double x) { return {std::cosh(x.v)}; }
inline Double tan(Double x) { return {std::tan(x.v)}; }
inline Double atan(Double x) { return {std::atan(x.v)}; }
inline Double tanh(Double x) { return {std::tanh(x.v)}; }

// Distribution functions. Phi(x) = 1/2 erfc(-x/sqrt 2): erfc rather than
// 1 - 1/2 erfc(x/sqrt 2), which loses every digit in the left tail.
inline Double erf(Double x) { return {std::erf(x.v)}; }
inline Double std_normal_cdf(Double x) { return {0.5 * std::erfc(-x.v / std::sqrt(2.0))}; }
inline Double std_normal_pdf(Double x) {
  const double v = x.v;
  return {std::exp(-0.5 * v * v) / std::sqrt(2.0 * M_PI)};
}
// sigma must be positive: invalid otherwise.
inline types::Value normal_cdf(Double x, Double mu, Double sigma) {
  if (!(sigma.v > 0.0)) return types::Invalid{types::InvalidReason::Domain};
  return Double{0.5 * std::erfc(-(x.v - mu.v) / (sigma.v * std::sqrt(2.0)))};
}
inline types::Value normal_pdf(Double x, Double mu, Double sigma) {
  if (!(sigma.v > 0.0)) return types::Invalid{types::InvalidReason::Domain};
  const double z = (x.v - mu.v) / sigma.v;
  return Double{std::exp(-0.5 * z * z) / (sigma.v * std::sqrt(2.0 * M_PI))};
}

// The aggregates' combining steps.
inline Double max(Double a, Double b) { return {b.v > a.v ? b.v : a.v}; }
inline Double min(Double a, Double b) { return {b.v < a.v ? b.v : a.v}; }
inline Double sum(Double a, Double b) { return {a.v + b.v}; }
inline Double product(Double a, Double b) { return {a.v * b.v}; }
// The specification does not say what an even number of arguments gives; the
// mean of the two central values is the usual reading, and what the ensemble
// median does. `values` is sorted in place.
inline Double median(std::vector<double>& values) {
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  return {(values.size() % 2 == 0) ? (values[mid - 1] + values[mid]) / 2.0 : values[mid]};
}

// Strings, over the text of a String (xs:string). concat: "Concatenates
// two or more input values" (legacy::B23: a number has no text here).
inline void append(std::string& out, std::string_view text) { out.append(text); }
// stringLength: "Returns the length of a string".
inline types::Integer string_length(std::string_view text) { return {static_cast<std::int64_t>(text.size())}; }
}  // namespace scalar

// ---------------------------------------------------------------- legacy delegates
// Comparisons as today (legacy::LegacyCompare): the Value operators.
struct LegacyCompare {
  static bool equal(const ::Value& a, const ::Value& b) { return a == b; }
  static bool not_equal(const ::Value& a, const ::Value& b) { return a != b; }
  static bool less_than(const ::Value& a, const ::Value& b) { return a < b; }
  static bool less_or_equal(const ::Value& a, const ::Value& b) { return a <= b; }
  static bool greater_than(const ::Value& a, const ::Value& b) { return a > b; }
  static bool greater_or_equal(const ::Value& a, const ::Value& b) { return a >= b; }
};

// The string edits as today (legacy::B2): the argument's Value, edited in
// place, keeping its string code.
struct LegacyStringEdit {
  static void uppercase(::Value& v) { v.uppercase(); }
  static void lowercase(::Value& v) { v.lowercase(); }
  static void trim_blanks(::Value& v) { v.trim_blanks(); }
};

// ---------------------------------------------------------------- adapters
namespace adapter {

// Argument i as a number: the double the Value holds, narrowed to a float
// when the argument's static type is float.
inline scalar::Double operand(const Bound& b, const ::Value* args, std::size_t i) {
  const double d = args[i].as_double();
  if (b.narrow_arguments && b.float_argument[i]) return {static_cast<double>(types::narrow_to_float(d))};
  return {d};
}

// A number result in the engine's storage; a float result is narrowed.
inline Status number(const Bound& b, scalar::Double r, ::Value& out) {
  out = ::Value(b.float_result ? static_cast<double>(types::narrow_to_float(r.v)) : r.v);
  return Status::Ok;
}
inline Status number(const Bound& b, const types::Value& r, ::Value& out) {
  if (const auto* d = std::get_if<types::Double>(&r)) return number(b, *d, out);
  if (types::is_invalid(r)) return Status::Invalid;
  throw std::logic_error("fn::adapter::number: a numeric kernel returned neither a double nor invalid");
}
inline Status boolean(bool r, ::Value& out) {
  out = ::Value(r ? 1.0 : 0.0);
  return Status::Ok;
}

template <scalar::Double (*F)(scalar::Double)>
Status unary(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  return number(b, F(operand(b, args, 0)), out);
}
template <scalar::Double (*F)(scalar::Double, scalar::Double)>
Status binary(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  return number(b, F(operand(b, args, 0), operand(b, args, 1)), out);
}
template <types::Value (*F)(scalar::Double, scalar::Double, scalar::Double)>
Status ternary(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  return number(b, F(operand(b, args, 0), operand(b, args, 1), operand(b, args, 2)), out);
}
inline Status divide(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  return number(b, scalar::divide(operand(b, args, 0), operand(b, args, 1), b.integer_division), out);
}
inline Status pow(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  return number(b, scalar::pow(operand(b, args, 0), operand(b, args, 1), b.integer_power), out);
}

// The aggregates: "Missing values in the input to an aggregate function are
// simply ignored ... If all inputs are missing, the result evaluates to a
// missing value" (`out` stays missing).
template <scalar::Double (*Combine)(scalar::Double, scalar::Double)>
Status fold(const Bound& b, const ::Value* args, std::size_t n, ::Value& out) {
  bool any = false;
  scalar::Double result{0.0};
  for (std::size_t i = 0; i < n; i++) {
    if (args[i].is_missing()) continue;
    const scalar::Double x = operand(b, args, i);
    result = any ? Combine(result, x) : x;
    any = true;
  }
  return any ? number(b, result, out) : Status::Ok;
}
inline Status avg(const Bound& b, const ::Value* args, std::size_t n, ::Value& out) {
  bool any = false;
  scalar::Double result{0.0};
  std::size_t count = 0;
  for (std::size_t i = 0; i < n; i++) {
    if (args[i].is_missing()) continue;
    const scalar::Double x = operand(b, args, i);
    result = any ? scalar::sum(result, x) : x;
    any = true;
    count++;
  }
  if (!any) return Status::Ok;
  return number(b, scalar::Double{result.v / static_cast<double>(count)}, out);
}
inline Status median(const Bound& b, const ::Value* args, std::size_t n, ::Value& out) {
  std::vector<double> values;
  values.reserve(n);
  for (std::size_t i = 0; i < n; i++)
    if (!args[i].is_missing()) values.push_back(operand(b, args, i).v);
  if (values.empty()) return Status::Ok;
  return number(b, scalar::median(values), out);
}

// isMissing, isNotMissing: they receive their argument missing or not.
inline Status is_missing(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  return boolean(args[0].is_missing(), out);
}
inline Status is_not_missing(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  return boolean(!args[0].is_missing(), out);
}

// `if` (design commit P2.8b4, moved here by P4.3; §4.2 "if laziness", §4.3
// LazyIf; D3, D24), the one lazy function: its inputs are not all its
// arguments. BuiltinFunctions.html, if: "Implements IF-THEN-ELSE logic. The
// ELSE part is optional. If the ELSE part is absent and the boolean value is
// false then a missing value is returned." Read as a conditional, the inputs
// are the condition and the branch it takes (§4.2: "Only the condition and
// the branch taken are inputs (so mapMissingTo applies when either is
// missing)"), evaluated in that order; the other branch is not evaluated, so
// its missing value is not an input. ApplyNode (MissingRule::LazyIf, §4.3
// step 1) evaluates the condition, asks `if_taken` for the argument it
// takes, evaluates that one only, and calls `if_then_else` with the inputs
// it evaluated: the condition, then the branch taken if there is one.
//
// The argument a present condition takes: 1 (THEN) when the double it holds
// is not 0, else 2 (ELSE); `n` (none) when that is ELSE and the call has no
// ELSE part.
inline std::size_t if_taken(const ::Value& condition, std::size_t n) {
  const std::size_t taken = condition.as_double() != 0.0 ? 1 : 2;
  return taken < n ? taken : n;
}
// The result over the inputs [condition, branch taken]: the branch's value as
// it came in (a string keeps its Value, decided 2026-11-19 after P3), missing
// or not; missing (`out` as the caller made it) when the condition is
// missing or no branch is taken ("If the ELSE part is absent and the boolean
// value is false then a missing value is returned"). The Apply's attributes
// are ApplyNode's: a missing condition or branch is a missing input, so
// mapMissingTo is returned before this is called; a missing result then
// gives defaultValue. A false condition without ELSE is a missing result,
// not a missing input: defaultValue applies, mapMissingTo does not. JPMML
// 1.7.7 agrees on every case but the missing condition without
// mapMissingTo, which it refuses (MissingArgumentException; D3).
inline Status if_then_else(const Bound&, const ::Value* args, std::size_t n, ::Value& out) {
  if (n == 2) out = args[1];
  return Status::Ok;
}

// The comparisons (legacy::LegacyCompare), over arguments narrowed as the
// numeric functions' are.
inline ::Value compared(const Bound& b, const ::Value* args, std::size_t i) {
  ::Value v = args[i];
  if (b.float_argument[i]) v.set_double(static_cast<double>(types::narrow_to_float(v.as_double())));
  return v;
}
template <bool (*C)(const ::Value&, const ::Value&)>
Status compare(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  if (!b.narrow_arguments) return boolean(C(args[0], args[1]), out);
  return boolean(C(compared(b, args, 0), compared(b, args, 1)), out);
}
inline bool is_in_values(const Bound& b, const ::Value* args, std::size_t n) {
  if (!b.narrow_arguments) {
    for (std::size_t i = 1; i < n; i++)
      if (LegacyCompare::equal(args[i], args[0])) return true;
    return false;
  }
  const ::Value x = compared(b, args, 0);
  for (std::size_t i = 1; i < n; i++)
    if (LegacyCompare::equal(compared(b, args, i), x)) return true;
  return false;
}
inline Status is_in(const Bound& b, const ::Value* args, std::size_t n, ::Value& out) {
  return boolean(is_in_values(b, args, n), out);
}
inline Status is_not_in(const Bound& b, const ::Value* args, std::size_t n, ::Value& out) {
  return boolean(!is_in_values(b, args, n), out);
}
// not: "Returns the boolean negation" of its boolean argument, read as today:
// any non-zero double is true.
inline Status logical_not(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  return boolean(!(args[0].as_double() != 0.0), out);
}

// String edits (legacy::B2).
template <void (*Edit)(::Value&)>
Status string_edit(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  out = args[0];
  Edit(out);
  return Status::Ok;
}
// substring(input, startPos, length): "startPos ... the first character is
// at position 1"; the arguments are read as today (a double cast to size_t,
// legacy::D4; legacy::B41: a startPos below 1 returns the input). A startPos
// beyond the end has no substring: invalid (today std::string::substr threw
// std::out_of_range, which Apply caught as an invalid value).
inline Status substring(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  const auto start_pos = static_cast<std::size_t>(args[1].as_double());  // 1-based
  const auto length = static_cast<std::size_t>(args[2].as_double());
  out = args[0];
  if (start_pos >= 1) {
    if (start_pos - 1 > out.text().size()) return Status::Invalid;
    out.substr(start_pos - 1, length);
  }
  return Status::Ok;
}
// concat builds a string: packed as the engine builds one (Value(text,
// STRING): the active Indexer's code of the text).
inline Status concat(const Bound&, const ::Value* args, std::size_t n, ::Value& out) {
  std::string result;
  for (std::size_t i = 0; i < n; i++) scalar::append(result, args[i].text());  // legacy::B23
  out = ::Value(result, DataType::DataTypeValue::STRING);
  return Status::Ok;
}
inline Status string_length(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  // legacy::D4: the integer is carried as a double.
  out = ::Value(static_cast<double>(scalar::string_length(args[0].text()).v));
  return Status::Ok;
}
// replace(input, pattern, replacement) and matches(input, pattern): "The
// pattern is a regular expression", here std::regex (ECMAScript). A pattern
// std::regex refuses, or a match it cannot complete, is an invalid result
// (today the std::regex_error reached Apply's catch).
inline Status replace(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  try {
    out = args[0].replace(args[1].text(), args[2].text());
  } catch (const std::regex_error&) {
    return Status::Invalid;
  }
  return Status::Ok;
}
inline Status matches(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  try {
    return boolean(args[0].matches(args[1].text()), out);
  } catch (const std::regex_error&) {
    return Status::Invalid;
  }
}

}  // namespace adapter

// ---------------------------------------------------------------- the factory
// The kernel of a row (§4.1 `instantiate`), at load. The switch has no
// default, so a new Op that is not given a kernel is a compiler warning
// (-Wswitch), and the logic_error below catches it at load.
inline KernelFn instantiate(Op op) {
  namespace a = adapter;
  namespace s = scalar;
  switch (op) {
    case Op::Plus: return a::binary<s::plus>;
    case Op::Minus: return a::binary<s::minus>;
    case Op::Times: return a::binary<s::times>;
    case Op::Divide: return a::divide;
    case Op::Max: return a::fold<s::max>;
    case Op::Min: return a::fold<s::min>;
    case Op::Sum: return a::fold<s::sum>;
    case Op::Avg: return a::avg;
    case Op::Exp: return a::unary<s::exp>;
    case Op::Ln: return a::unary<s::ln>;
    case Op::Log10: return a::unary<s::log10>;
    case Op::Sqrt: return a::unary<s::sqrt>;
    case Op::Abs: return a::unary<s::abs>;
    case Op::Round: return a::unary<s::round>;
    case Op::Floor: return a::unary<s::floor>;
    case Op::Ceil: return a::unary<s::ceil>;
    case Op::Pow: return a::pow;
    case Op::If: return a::if_then_else;  // inputs: MissingRule::LazyIf (if_taken)
    case Op::IsMissing: return a::is_missing;
    case Op::IsNotMissing: return a::is_not_missing;
    case Op::Equal: return a::compare<LegacyCompare::equal>;
    case Op::NotEqual: return a::compare<LegacyCompare::not_equal>;
    case Op::LessThan: return a::compare<LegacyCompare::less_than>;
    case Op::LessOrEqual: return a::compare<LegacyCompare::less_or_equal>;
    case Op::GreaterThan: return a::compare<LegacyCompare::greater_than>;
    case Op::GreaterOrEqual: return a::compare<LegacyCompare::greater_or_equal>;
    case Op::IsIn: return a::is_in;
    case Op::IsNotIn: return a::is_not_in;
    case Op::Replace: return a::replace;
    case Op::Matches: return a::matches;
    case Op::Uppercase: return a::string_edit<LegacyStringEdit::uppercase>;
    case Op::Lowercase: return a::string_edit<LegacyStringEdit::lowercase>;
    case Op::Substring: return a::substring;
    case Op::TrimBlanks: return a::string_edit<LegacyStringEdit::trim_blanks>;
    case Op::Concat: return a::concat;
    case Op::StringLength: return a::string_length;
    case Op::Median: return a::median;
    case Op::Product: return a::fold<s::product>;
    case Op::Threshold: return a::binary<s::threshold>;
    case Op::Modulo: return a::binary<s::modulo>;
    case Op::Not: return a::logical_not;
    case Op::Expm1: return a::unary<s::expm1>;
    case Op::Ln1p: return a::unary<s::ln1p>;
    case Op::Hypot: return a::binary<s::hypot>;
    case Op::Rint: return a::unary<s::rint>;
    case Op::Sin: return a::unary<s::sin>;
    case Op::Asin: return a::unary<s::asin>;
    case Op::Sinh: return a::unary<s::sinh>;
    case Op::Cos: return a::unary<s::cos>;
    case Op::Acos: return a::unary<s::acos>;
    case Op::Cosh: return a::unary<s::cosh>;
    case Op::Tan: return a::unary<s::tan>;
    case Op::Atan: return a::unary<s::atan>;
    case Op::Tanh: return a::unary<s::tanh>;
    case Op::Erf: return a::unary<s::erf>;
    case Op::NormalCDF: return a::ternary<s::normal_cdf>;
    case Op::NormalPDF: return a::ternary<s::normal_pdf>;
    case Op::StdNormalCDF: return a::unary<s::std_normal_cdf>;
    case Op::StdNormalPDF: return a::unary<s::std_normal_pdf>;
    case Op::None: break;
  }
  throw std::logic_error("fn::instantiate: no kernel for this function");
}

// The functions whose numeric arguments and result ApplyNode types (the
// arithmetic, aggregate, mathematical and distribution functions), and the
// comparisons, whose float arguments it narrows. The others (if, isMissing,
// isNotMissing, not, the string functions) read their arguments as they are.
constexpr bool compares(Op op) {
  switch (op) {
    case Op::Equal:
    case Op::NotEqual:
    case Op::LessThan:
    case Op::LessOrEqual:
    case Op::GreaterThan:
    case Op::GreaterOrEqual:
    case Op::IsIn:
    case Op::IsNotIn:
      return true;
    default:
      return false;
  }
}
constexpr bool numeric(Op op) {
  switch (op) {
    case Op::None:
    case Op::If:
    case Op::IsMissing:
    case Op::IsNotMissing:
    case Op::Not:
    case Op::Replace:
    case Op::Matches:
    case Op::Uppercase:
    case Op::Lowercase:
    case Op::Substring:
    case Op::TrimBlanks:
    case Op::Concat:
    case Op::StringLength:
      return false;
    default:
      return !compares(op);
  }
}

}  // namespace cpmml::fn

#endif  // CPMML_FN_KERNELS_H
