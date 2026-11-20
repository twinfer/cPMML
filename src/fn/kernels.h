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
 * P4.1 reproduced the base build's results exactly (dump diff empty),
 * including the departures from the spec that P4.2-P4.6 fix one at a time. Each departure
 * is behind a named switch in `legacy` below; the code a switch keeps is in
 * an `if constexpr (legacy::X)` branch, or names the switch in a comment
 * where the legacy rule is the absence of a check. The commit that fixes the
 * bug deletes the switch and its code. A switch cannot simply be set to
 * false: the static_asserts below stop that, because the spec rule that
 * replaces the legacy one is that commit's to write and to test.
 *
 * Operands. Today's engine stores every number as a double, an integer
 * included, until P5. A numeric operand of a call whose result is a double
 * or a float is read as a `types::Double` of the double the Value holds,
 * whatever its static type, and narrowed to a float first when its static
 * type is float (the binder's static type, fixed at load; what
 * core/builtinfunctions.h did since P2.4). A call whose result is an
 * integer (P4.5, D4) has its own kernel, chosen at load (`instantiate`):
 * it reads each operand as the int64 the stored double holds and computes
 * in int64, an overflow being invalid; see "Integer kernels" below. It is
 * not `unpack<types::Integer>`: that makes 3.5 in an integer slot invalid
 * (B17, P4.7), which today it is not (legacy::B17). Comparisons delegate
 * to `LegacyCompare`, today's Value operators (the double: a number, a boolean's 1 or 0, a
 * string's Indexer code), until P6. The string edits (uppercase, lowercase,
 * substring, trimBlanks) delegate to `LegacyStringEdit`, the Value's own
 * in-place edit, which keeps the argument's string code (B2, P6) and its
 * Value: decided 2026-11-19 after P3, "a kernel never round-trips a string
 * through unpack/pack ... an argument string that the kernel passes through
 * or returns must keep the old Value it came in as; only strings the kernel
 * builds are packed" (concat, replace).
 */

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/types/arithmetic.h"
#include "core/types/java_text.h"
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
// B17 (P4.7): an integer slot can hold a number that is no integer (an
// integer DerivedField over a double keeps its fraction, `dint` = 3.5; or
// ±INF, NaN). An integer kernel that meets such an operand computes the
// whole call as before P4.5, in double (adapter::integer_operand).
inline constexpr bool B17 = true;
// B2 (P6): uppercase, lowercase, substring and trimBlanks edit a copy of
// their argument's Value, which keeps the argument's string code
// (LegacyStringEdit).
inline constexpr bool B2 = true;
// B1, B18, B44 (P6): comparisons compare the double each Value holds
// (LegacyCompare).
inline constexpr bool LegacyCompare = true;

static_assert(B17, "P4.7 removes this switch together with the code it keeps");
static_assert(B2 && LegacyCompare, "P6 removes these switches together with the code they keep");
}  // namespace legacy

namespace scalar {
struct NumberPattern;  // formatNumber's pattern, below
}

// ---------------------------------------------------------------- bound call
// What a kernel needs to know about its call, fixed at load from the
// arguments' static types (bind::expression_type, the binder's answer) and
// the result type (bind::apply_result_type, the table's result rule). Set by
// ApplyNode for the numeric functions, the comparisons, concat and
// formatNumber; all false (empty) for the others (core/builtinfunctions.h's `typed()` flags,
// unchanged).
struct Bound {
  // A float argument is an xs:float, so it is narrowed before use (the
  // engine carries it as a double).
  bool narrow_arguments = false;
  std::vector<std::uint8_t> float_argument;  // per argument; read only when narrow_arguments
  bool float_result = false;  // the result is a float: narrowed after
  // An integer result is not a flag here: it is a kernel of its own,
  // instantiate(op, true) (P4.5).
  // concat (P4.6, B23): the static type each argument is cast to string
  // from, one per argument (adapter::text_type). Empty for every other call.
  std::vector<types::TypeId> argument_type;
  // formatNumber (P4.9): the pattern, parsed at load when it is a Constant
  // (null otherwise: parsed per record). argument_type holds the input's and
  // the pattern's static types.
  std::shared_ptr<const scalar::NumberPattern> number_pattern;
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
// Division by zero (P4.4, B21, D20), "Division by Zero": "If a positive
// number is divided by zero, INF will be returned. If zero is divided by
// zero, the result is indeterminate and a missing value will be returned. If
// a negative number is divided by zero, -INF will be returned." So 0/0 is a
// missing result (then defaultValue), for integers too (§4.2); a zero
// divisor of either sign gives INF signed by the numerator (XSD's "only one
// zero", core/types/arithmetic.h; IEEE would give -INF for 1/-0.0); NaN/0 is
// NaN (Transformations.html: "All arithmetic expressions involving NaN will
// evaluate to NaN", D23). An integer x/0 has no value: "INF and -INF can only
// be floating point numbers", so it is invalid (D20; XQuery err:FOAR0001).
// `divide` is the quotient by a divisor that is not zero, a Double on the
// per-record path; `divide_by_zero` the rest (adapter::divide picks one).
inline Double divide(Double a, Double b, bool integer) {
  double r = a.v / b.v;
  if (integer) r = std::trunc(r);
  return {r};
}
inline types::Value divide_by_zero(Double a, bool integer) {
  if (a.v == 0.0) return types::Missing{};
  if (integer) return types::Invalid{types::InvalidReason::DivisionByZero};
  if (std::isnan(a.v)) return a;
  return Double{a.v > 0.0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity()};
}

// log10, ln, sqrt, abs, exp: "Semantics are as usual", and outside the
// domain the spec gives the value, so none of them is invalid (P4.4, B41
// settled): "The logarithm of a negative number is undefined, so NaN should
// be returned in such a case. The logarithm of INF is to be taken as INF.
// The logarithm of 0 is to be taken as -INF. The square root of a INF is to
// be taken as INF. The square root of a negative number is to be taken as
// NaN since complex numbers are not supported by PMML. The exponential of
// INF is to be taken as INF. The exponential of -INF is to be taken as 0."
// NaN is a value (Transformations.html: "This is the value returned from
// such meaningless expressions as the logarithm of a negative number. It
// should not be used as a generic missing value"); IEEE 754, which the C
// library follows, gives each of these (log of -0 is -INF, as of 0).
inline Double log10(Double x) { return {std::log10(x.v)}; }
inline Double ln(Double x) { return {std::log(x.v)}; }
inline Double sqrt(Double x) { return {std::sqrt(x.v)}; }
inline Double abs(Double x) { return {std::abs(x.v)}; }
inline Double exp(Double x) { return {std::exp(x.v)}; }
// "Functions floor, ceil, and round return an integer obtained by rounding
// the numeric argument down, up, and to the closest integer respectively".
// "Since INF and -INF can only be floating point numbers, they will cause
// floor, ceil, round, and modulo to return an invalid value" (P4.4, B21); NaN
// is no integer either, so it is invalid too (§4.2's row: "±INF, NaN"). The
// integer is an int64 (P4.5, D4: "integers are int64; overflow invalid"), so
// a rounded value outside [-2^63, 2^63) is invalid too (§4.2's row: "out of
// int64 range"; core/types/arithmetic.h detail::to_integer, the same test).
// `to_integer` is the integer x rounds to by R, false when there is none
// (the comparison is false for NaN and ±INF as well); outside it the result
// is invalid (adapter::to_integer_unary). The integer is stored as the
// double it is (exact: a rounded double within the range is an integer
// double), but +0 for a -0.0 (XSD's integer has one zero).
inline double round_half_away(double x) { return std::round(x); }
inline double round_down(double x) { return std::floor(x); }
inline double round_up(double x) { return std::ceil(x); }
template <double (*R)(double)>
bool to_integer(Double x, types::Integer& out) {
  const double r = R(x.v);
  if (!(r >= -9223372036854775808.0 && r < 9223372036854775808.0)) return false;
  out = {static_cast<std::int64_t>(r)};
  return true;
}
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
// Invalid (P4.4): an infinite operand, "INF and -INF ... will cause floor,
// ceil, round, and modulo to return an invalid value" (B21); and a zero
// divisor (B41), which has no remainder in any reading the spec names: "The
// PMML standard follows the semantics of Python when using the % operator"
// (Python raises ZeroDivisionError for x % 0 and x % 0.0); XQuery, for
// integer operands, "A dynamic error is raised [err:FOAR0001] for xs:integer
// and xs:decimal operands, if $arg2 is zero"; and the formula's a/0 is +-INF,
// whose floor is invalid by the sentence above. (XQuery gives NaN for a
// double divisor of 0; the PMML sentences are preferred.) A NaN operand gives
// NaN first ("All arithmetic expressions involving NaN will evaluate to
// NaN", D23; XQuery: "If either operand is NaN, the result is NaN").
// `modulo_domain` is that domain; outside it the result is invalid
// (adapter::checked_binary).
inline bool modulo_domain(Double a, Double b) {
  if (std::isnan(a.v) || std::isnan(b.v)) return true;
  return b.v != 0.0 && !std::isinf(a.v) && !std::isinf(b.v);
}
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

// ---- Integer kernels (P4.5, D4). A call whose result type is integer
// (Functions.html: "functions are assumed to inherit the dataType of the
// current input parameters unless otherwise specified. When the input
// parameters have multiple dataTypes, the least restrictive dataType will be
// inherited by default"; §2.5's table: integer with integer only) computes
// in int64, exactly (XML Schema Part 2 §3.3.13: "This results in the
// standard mathematical concept of the integer numbers. The value space of
// integer is the infinite set {...,-2,-1,0,1,2,...}"; int64 is the
// representable subset, Q10). A result outside int64 is invalid (D4,
// InvalidReason::Overflow; XQuery F&O 3.1 §4.2, which the spec's modulo
// cites: "For xs:integer operations, implementations that support
// limited-precision integer operations must select from the following
// options: They may choose to always raise an error [err:FOAR0002] ...").
// + - * / are core/types/arithmetic.h's Plus, Minus, Times, Divide over
// Integer (overflow; / truncates toward zero, §4.2; 0/0 missing; x/0
// invalid, D20; INT64_MIN / -1 overflows, §4.2's row).
inline types::Value plus(types::Integer a, types::Integer b) { return types::Plus::apply(a, b); }
inline types::Value minus(types::Integer a, types::Integer b) { return types::Minus::apply(a, b); }
inline types::Value times(types::Integer a, types::Integer b) { return types::Times::apply(a, b); }
inline types::Value divide(types::Integer a, types::Integer b) { return types::Divide::apply(a, b); }
// abs: "Semantics are as usual"; |INT64_MIN| is 2^63, beyond int64 (§4.2's
// row: "abs(INT64_MIN)").
inline types::Value abs(types::Integer x) {
  if (x.v == std::numeric_limits<std::int64_t>::min()) return types::Invalid{types::InvalidReason::Overflow};
  return types::Integer{x.v < 0 ? -x.v : x.v};
}
// pow: "returns the number x raised to the power y; if x and y are both zero
// then pow will return one". A negative exponent has no integer value (D8,
// invalid); otherwise exact, by squaring, invalid on overflow (§4.2's row:
// "overflow"). The base is squared only while bits of the exponent remain,
// and those bits multiply the result by at least that square, so an
// overflowing square is an overflowing result (|result| >= 1 unless x = 0,
// whose square does not overflow).
inline types::Value pow(types::Integer a, types::Integer b) {
  if (b.v < 0) return types::Invalid{types::InvalidReason::Domain};
  std::int64_t result = 1, base = a.v;
  for (std::int64_t e = b.v; e > 0;) {
    if ((e & 1) && __builtin_mul_overflow(result, base, &result))
      return types::Invalid{types::InvalidReason::Overflow};
    e >>= 1;
    if (e > 0 && __builtin_mul_overflow(base, base, &base)) return types::Invalid{types::InvalidReason::Overflow};
  }
  return types::Integer{result};
}
// modulo, over integers: "returns the remainder after x is divided by y,
// computed according to XQuery rules ... The PMML standard follows the
// semantics of Python when using the % operator", i.e. a - floor(a/b)*b
// computed exactly: the truncated remainder, moved by b when its sign is not
// the divisor's (Python: "The modulo operator always yields a result with the
// same sign as its second operand (or zero)"). A divisor of 0 is invalid
// (P4.4, B41: "A dynamic error is raised [err:FOAR0001] for xs:integer ...
// operands, if $arg2 is zero"). The remainder by -1 is 0 (INT64_MIN % -1 is
// undefined in C++, so it is not computed).
inline types::Value modulo(types::Integer a, types::Integer b) {
  if (b.v == 0) return types::Invalid{types::InvalidReason::DivisionByZero};
  if (b.v == -1) return types::Integer{0};
  std::int64_t r = a.v % b.v;
  if (r != 0 && ((r < 0) != (b.v < 0))) r += b.v;
  return types::Integer{r};
}
// legacy::B17: an integer call with an operand that holds no integer (an
// integer slot keeping a fraction, ±INF or NaN) is computed as before P4.5,
// in double: these are the double kernels as an integer call ran them.
template <Double (*F)(Double, Double)>
types::Value b17(Double a, Double b) {
  return F(a, b);
}
template <Double (*F)(Double)>
types::Value b17(Double x) {
  return F(x);
}
inline types::Value b17_divide(Double a, Double b) {
  if (b.v == 0.0) return divide_by_zero(a, true);
  return divide(a, b, true);
}
inline types::Value b17_pow(Double a, Double b) { return pow(a, b, true); }
inline types::Value b17_modulo(Double a, Double b) {
  if (!modulo_domain(a, b)) return types::Invalid{types::InvalidReason::DivisionByZero};
  return modulo(a, b);
}

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

// Strings, over the text of a String (xs:string). concat: "Returns a string
// as a result of the concatenation of two or more parameters", "See XQuery
// fn:concat", whose arguments are each cast to xs:string (P4.6, B23). The
// text of that cast is Java's (Q1 = (a), D17; §2.6 "number -> string | Java
// text"; core/types/java_text.h): Long.toString of an integer ("2000"),
// Float.toString of a float ("0.1", "1.0E8"), Double.toString of a double
// ("2.0", "1.0E-4", "1.0E7"), "true"/"false" of a boolean (XSD canonical);
// XQuery would write the whole double 2.0 as "2" (the departure, D17).
inline void append(std::string& out, std::string_view text) { out.append(text); }
inline void append(std::string& out, types::Integer x) { types::java_text(x, out); }
inline void append(std::string& out, types::Float x) { types::java_text(x, out); }
inline void append(std::string& out, types::Double x) { types::java_text(x, out); }
inline void append(std::string& out, types::Boolean x) { types::java_text(x, out); }
// stringLength: "Returns the length of a string".
inline types::Integer string_length(std::string_view text) { return {static_cast<std::int64_t>(text.size())}; }

// formatNumber(input, pattern) (design item P4.9). BuiltinFunctions.html:
// "Formats numbers according to a pattern. The pattern uses the Posix
// descriptors as used, e.g., in the C function printf"; its example:
// formatNumber(Num, "%3d") with Num=2 "is the string \"  2\"" ("a string of
// length 3 with leading blanks"). The descriptors are POSIX fprintf()'s
// (IEEE Std 1003.1, fprintf), in the POSIX locale ("In the POSIX locale ...
// the radix character shall default to a <period>"): "Each conversion
// specification is introduced by the '%' character or by the character
// sequence "%n$", after which the following appear in sequence: Zero or more
// flags ... An optional minimum field width ... An optional precision ... An
// optional length modifier ... A conversion specifier character". The text
// outside the conversions is copied; "%%" is a '%'.
//
// What a pattern may hold, the rest being a pattern formatNumber cannot
// apply (a load error when the pattern is a Constant, else an invalid
// result; adapter::format_number):
//   - the numeric conversions d i o u x X f F e E g G a A ("%c", "%s", "%p",
//     "%n" convert no number);
//   - the flags - + <space> # 0, with POSIX's meaning; '#' with d, i or u is
//     refused ("For other conversion specifiers, the behavior is
//     undefined"), and so is the <apostrophe> (thousands' grouping), whose
//     characters are the locale's, which PMML does not name;
//   - a width and a precision written as decimal digits (up to INT_MAX, C's
//     own limit); '*' is refused: "an argument of type int supplies the
//     field width or precision", and formatNumber has one argument;
//   - no length modifier (hh h l ll j z t L): each names the C type of the
//     argument ("applies to a short ... argument"), which a PMML value does
//     not have;
//   - one conversion, or any number of "%1$" conversions (all of the input:
//     "%n$" with n = 1, the only argument); another n, two unnumbered
//     conversions ("If there are insufficient arguments for the format, the
//     behavior is undefined") or numbered and unnumbered ones mixed are
//     refused. No conversion is the text alone ("If the format is exhausted
//     while arguments remain, the excess arguments shall be evaluated but are
//     otherwise ignored").
// The value is the input as a number of its static type, a float narrowed
// first (a float "is promoted to double" by C's default argument promotions,
// so %.10f of the float 0.1 is 0.1000000015; JPMML 1.7.7 agrees). The integer
// conversions take a value that is an integer (d, i: "The int argument shall
// be converted to a signed decimal"; o, u, x, X: "The unsigned argument"):
// the input's value, of whatever type, when it is integral, within the int64
// range (D4), and, for o u x X, not negative; another value (3.5, -3 for %x,
// INF, NaN) has no integer to convert and the result is invalid. The other
// conversions take the value as a double ("The double argument"). The low-
// order digit is rounded from the exact binary value, to nearest, ties to
// even ("rounded in an implementation-defined manner"; IEEE 754's default
// rounding, as std::to_chars, which is "as if by printf in the C locale",
// and the C libraries of glibc and macOS do), so %.2f of 0.125 is 0.12 and
// of 1.005 (1.00499999999999989...) 1.00. An infinity is "inf" ("[-]inf"),
// a NaN "nan" with no sign, '+' and <space> included (PMML's NaN has none;
// "[-]nan", the sign implementation-defined; macOS's C library agrees,
// glibc writes "+nan" for '+'), "INF"/"NAN"
// for F, E, G and A, padded with blanks ("except when converting an
// infinity or NaN"). %a writes the significand std::to_chars gives (a
// normal number "1.h...", a subnormal "0.h...": "which shall be non-zero if
// the argument is a normalized floating-point number and is otherwise
// unspecified").
struct NumberConversion {
  char specifier = 'd';
  bool left = false, plus = false, space = false, alternate = false, zero = false;
  int width = 0;
  int precision = -1;  // -1: none
};
struct NumberPattern {
  std::vector<std::string> text;            // text[k] precedes conversion k; one more than conversions
  std::vector<NumberConversion> conversion;
};

// Parses `pattern` into `out`; the reason it cannot be applied, or empty.
inline std::string parse_number_pattern(std::string_view pattern, NumberPattern& out) {
  out = NumberPattern{};
  out.text.emplace_back();
  bool numbered = false, unnumbered = false;
  const auto digits = [&](std::size_t& i, int& value) {
    long long v = 0;
    while (i < pattern.size() && pattern[i] >= '0' && pattern[i] <= '9') {
      v = v * 10 + (pattern[i++] - '0');
      if (v > std::numeric_limits<int>::max()) return false;
    }
    value = static_cast<int>(v);
    return true;
  };
  std::size_t i = 0;
  while (i < pattern.size()) {
    const char c = pattern[i++];
    if (c != '%') {
      out.text.back() += c;
      continue;
    }
    if (i < pattern.size() && pattern[i] == '%') {
      out.text.back() += '%';
      i++;
      continue;
    }
    NumberConversion conv;
    // "%n$": digits then '$'.
    std::size_t j = i;
    while (j < pattern.size() && pattern[j] >= '0' && pattern[j] <= '9') j++;
    if (j > i && j < pattern.size() && pattern[j] == '$') {
      if (pattern.substr(i, j - i) != "1")
        return "argument \"%" + std::string(pattern.substr(i, j - i)) +
               "$\" does not exist: formatNumber has one input";
      numbered = true;
      i = j + 1;
    } else {
      unnumbered = true;
    }
    for (; i < pattern.size(); i++) {
      const char f = pattern[i];
      if (f == '-') conv.left = true;
      else if (f == '+') conv.plus = true;
      else if (f == ' ') conv.space = true;
      else if (f == '#') conv.alternate = true;
      else if (f == '0') conv.zero = true;
      else if (f == '\'') return "the thousands' grouping flag (') is not supported: its characters are the locale's";
      else break;
    }
    if (i < pattern.size() && pattern[i] == '*')
      return "a width given by an argument (*) is not supported: formatNumber has one input";
    if (!digits(i, conv.width)) return "the field width is beyond C's int range";
    if (i < pattern.size() && pattern[i] == '.') {
      i++;
      if (i < pattern.size() && pattern[i] == '*')
        return "a precision given by an argument (.*) is not supported: formatNumber has one input";
      if (!digits(i, conv.precision)) return "the precision is beyond C's int range";
    }
    if (i == pattern.size()) return "the conversion at the end is incomplete";
    const char s = pattern[i++];
    if (std::string_view("hljztLqI").find(s) != std::string_view::npos)
      return std::string("the length modifier '") + s + "' is not supported: the input is a PMML number, not a C type";
    if (std::string_view("diouxXfFeEgGaA").find(s) == std::string_view::npos)
      return std::string("'%") + s + "' is not a numeric conversion (d i o u x X f F e E g G a A)";
    if (conv.alternate && (s == 'd' || s == 'i' || s == 'u'))
      return std::string("the flag '#' is undefined for '%") + s + "'";
    conv.specifier = s;
    out.conversion.push_back(conv);
    out.text.emplace_back();
  }
  if (numbered && unnumbered) return "numbered (%1$) and unnumbered conversions are mixed";
  if (unnumbered && out.conversion.size() > 1)
    return "it has " + std::to_string(out.conversion.size()) + " conversions: formatNumber has one input";
  return {};
}

namespace posix {
// The converted text `body` with its `prefix` (sign, 0x), padded to the width.
inline void pad(std::string& out, const NumberConversion& c, std::string_view prefix, std::string_view body,
                bool zero) {
  const std::size_t size = prefix.size() + body.size();
  const std::size_t fill = c.width > 0 && static_cast<std::size_t>(c.width) > size ? c.width - size : 0;
  if (c.left) {
    out.append(prefix).append(body).append(fill, ' ');
  } else if (zero) {
    out.append(prefix).append(fill, '0').append(body);
  } else {
    out.append(fill, ' ').append(prefix).append(body);
  }
}
inline std::string to_chars(double a, std::chars_format format, int precision) {
  // The longest fixed text of a double is 309 integer digits; then the
  // precision.
  std::string buffer(static_cast<std::size_t>(precision < 0 ? 0 : precision) + 400, '\0');
  const auto r = precision < 0 ? std::to_chars(buffer.data(), buffer.data() + buffer.size(), a, format)
                               : std::to_chars(buffer.data(), buffer.data() + buffer.size(), a, format, precision);
  if (r.ec != std::errc()) throw std::logic_error("formatNumber: std::to_chars failed");
  buffer.resize(static_cast<std::size_t>(r.ptr - buffer.data()));
  return buffer;
}
inline void upper(std::string& s) {
  for (char& ch : s)
    if (ch >= 'a' && ch <= 'z') ch = static_cast<char>(ch - 'a' + 'A');
}
// d i o u x X over the integer `v`; false when the value is no integer the
// conversion takes.
inline bool format_integer(std::string& out, const NumberConversion& c, double v) {
  if (!(v >= -9223372036854775808.0 && v < 9223372036854775808.0) || v != std::trunc(v)) return false;
  const std::int64_t n = static_cast<std::int64_t>(v);
  const bool is_signed = c.specifier == 'd' || c.specifier == 'i';
  if (!is_signed && n < 0) return false;
  const std::uint64_t magnitude = n < 0 ? std::uint64_t{0} - static_cast<std::uint64_t>(n) : static_cast<std::uint64_t>(n);
  const int base = c.specifier == 'o' ? 8 : (c.specifier == 'x' || c.specifier == 'X') ? 16 : 10;
  char buffer[24];
  std::string body(buffer, std::to_chars(buffer, buffer + sizeof(buffer), magnitude, base).ptr);
  if (c.specifier == 'X') upper(body);
  if (c.precision == 0 && magnitude == 0) body.clear();  // "converting zero with an explicit precision of zero"
  if (c.precision > 0 && body.size() < static_cast<std::size_t>(c.precision))
    body.insert(0, static_cast<std::size_t>(c.precision) - body.size(), '0');
  if (c.alternate && c.specifier == 'o' && (body.empty() || body[0] != '0')) body.insert(0, 1, '0');
  std::string_view prefix;
  if (is_signed) prefix = n < 0 ? "-" : c.plus ? "+" : c.space ? " " : "";
  else if (c.alternate && magnitude != 0 && c.specifier == 'x') prefix = "0x";
  else if (c.alternate && magnitude != 0 && c.specifier == 'X') prefix = "0X";
  pad(out, c, prefix, body, c.zero && !c.left && c.precision < 0);
  return true;
}
// f F e E g G a A over the double `v`.
inline void format_double(std::string& out, const NumberConversion& c, double v) {
  const char s = c.specifier;
  const bool uppercase = s == 'F' || s == 'E' || s == 'G' || s == 'A';
  const bool negative = std::signbit(v);
  const std::string_view sign = std::isnan(v) ? "" : negative ? "-" : c.plus ? "+" : c.space ? " " : "";
  if (!std::isfinite(v)) {
    pad(out, c, sign, std::isnan(v) ? (uppercase ? "NAN" : "nan") : (uppercase ? "INF" : "inf"), false);
    return;
  }
  const double a = std::abs(v);
  const int p = c.precision < 0 ? 6 : c.precision;
  std::string body;
  std::string_view base;
  if (s == 'f' || s == 'F') {
    body = to_chars(a, std::chars_format::fixed, p);
    if (c.alternate && p == 0) body += '.';
  } else if (s == 'e' || s == 'E') {
    body = to_chars(a, std::chars_format::scientific, p);
    if (c.alternate && p == 0) body.insert(body.find('e'), 1, '.');
  } else if (s == 'g' || s == 'G') {
    // "Let P equal the precision if non-zero, 6 if the precision is omitted,
    // or 1 if the precision is zero. Then, if a conversion with style E
    // would have an exponent of X: If P > X>=-4, the conversion shall be with
    // style f (or F) and precision P -(X+1). Otherwise, the conversion shall
    // be with style e (or E) and precision P -1. Finally, unless the '#' flag
    // is used, any trailing zeros shall be removed from the fractional
    // portion of the result and the decimal-point character shall be removed
    // if there is no fractional portion remaining."
    const int P = c.precision < 0 ? 6 : c.precision == 0 ? 1 : c.precision;
    body = to_chars(a, std::chars_format::scientific, P - 1);
    const std::size_t e = body.find('e');
    const int X = std::stoi(body.substr(e + 1));
    if (P > X && X >= -4) body = to_chars(a, std::chars_format::fixed, P - (X + 1));
    const std::size_t end = body.find('e') == std::string::npos ? body.size() : body.find('e');
    if (!c.alternate) {
      std::size_t point = body.find('.');
      if (point != std::string::npos && point < end) {
        std::size_t last = end;
        while (last > point + 1 && body[last - 1] == '0') last--;
        if (last == point + 1) last = point;
        body.erase(last, end - last);
      }
    } else if (body.find('.') == std::string::npos) {
      body.insert(end, 1, '.');
    }
  } else {  // a, A
    body = to_chars(a, std::chars_format::hex, c.precision);
    if (c.alternate && body.find('.') == std::string::npos) body.insert(body.find('p'), 1, '.');
    base = uppercase ? "0X" : "0x";
  }
  if (uppercase) upper(body);
  const std::string prefix = std::string(sign) + std::string(base);
  pad(out, c, prefix, body, c.zero && !c.left);
}
}  // namespace posix

// The pattern applied to `v`; false when a conversion has no integer to
// convert (the result is invalid).
inline bool format_number(const NumberPattern& pattern, double v, std::string& out) {
  out = pattern.text[0];
  for (std::size_t k = 0; k < pattern.conversion.size(); k++) {
    const NumberConversion& c = pattern.conversion[k];
    if (std::string_view("diouxX").find(c.specifier) != std::string_view::npos) {
      if (!posix::format_integer(out, c, v)) return false;
    } else {
      posix::format_double(out, c, v);
    }
    out += pattern.text[k + 1];
  }
  return true;
}
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
// A kernel that may have no value: invalid, or missing (`out` stays as the
// caller made it, i.e. missing; then defaultValue, §4.3 step 5).
inline Status number(const Bound& b, const types::Value& r, ::Value& out) {
  if (const auto* d = std::get_if<types::Double>(&r)) return number(b, *d, out);
  if (types::is_invalid(r)) return Status::Invalid;
  if (types::is_missing(r)) return Status::Ok;
  throw std::logic_error("fn::adapter::number: a numeric kernel returned neither a double, invalid nor missing");
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
// The same for a function with a domain (floor, ceil, round, modulo):
// invalid outside it.
// floor, ceil, round: the integer the argument rounds to, invalid when there
// is none (scalar::to_integer).
template <double (*R)(double)>
Status to_integer_unary(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  types::Integer r{};
  if (!scalar::to_integer<R>(operand(b, args, 0), r)) return Status::Invalid;
  out = ::Value(static_cast<double>(r.v));
  return Status::Ok;
}
template <scalar::Double (*F)(scalar::Double, scalar::Double), bool (*Domain)(scalar::Double, scalar::Double)>
Status checked_binary(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  const scalar::Double x = operand(b, args, 0);
  const scalar::Double y = operand(b, args, 1);
  if (!Domain(x, y)) return Status::Invalid;
  return number(b, F(x, y), out);
}
template <types::Value (*F)(scalar::Double, scalar::Double, scalar::Double)>
Status ternary(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  return number(b, F(operand(b, args, 0), operand(b, args, 1), operand(b, args, 2)), out);
}
inline Status divide(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  const scalar::Double x = operand(b, args, 0);
  const scalar::Double y = operand(b, args, 1);
  if (y.v == 0.0) return number(b, scalar::divide_by_zero(x, false), out);
  return number(b, scalar::divide(x, y, false), out);
}
inline Status pow(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  return number(b, scalar::pow(operand(b, args, 0), operand(b, args, 1), false), out);
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

// ---- Integer kernels (P4.5, D4): the kernel of a call whose result type is
// integer (instantiate(op, true)). Each operand is the int64 its stored
// double holds; the result is computed in int64 (scalar's integer kernels)
// and stored as a double, the only number the old Value has until P5. That
// store rounds a result beyond 2^53 to the nearest double (ties to even), so
// it is exact in the computation and rounded once, when stored: sum(2^53, 1,
// 1) is 2^53 + 2 where double arithmetic gives 2^53, modulo(2^60, 7) is 1
// where a - floor(a/b)*b in double gives 64. An int64 result within 512 of
// 2^63 is stored as the double 2^63, which an integer kernel reading it back
// finds outside int64 (`integer_operand`: Overflow), so a later integer
// call over it is invalid, as the cast of that double to an integer is
// (§2.6); no input or kernel can hold the int64 it was.
enum class IntegerOperand : std::uint8_t { Integer, Overflow, NotInteger };
// The int64 the Value holds: NotInteger when it holds no integer (a
// fraction, ±INF, NaN: legacy::B17, which P4.7's casts make impossible in an
// integer slot), Overflow for an integer double outside [-2^63, 2^63) (the
// double 2^63 above, or an integer slot over a double, B17), else Integer
// (a -0.0 is the integer 0).
inline IntegerOperand integer_operand(const ::Value& v, types::Integer& out) {
  const double d = v.as_double();
  if (!std::isfinite(d) || std::trunc(d) != d) return IntegerOperand::NotInteger;
  if (!(d >= -9223372036854775808.0 && d < 9223372036854775808.0)) return IntegerOperand::Overflow;
  out = {static_cast<std::int64_t>(d)};
  return IntegerOperand::Integer;
}
// An integer kernel's result in the engine's storage: an Integer as its
// double, invalid, or missing (`out` stays missing).
inline Status integer(const types::Value& r, ::Value& out) {
  if (const auto* i = std::get_if<types::Integer>(&r)) {
    out = ::Value(static_cast<double>(i->v));
    return Status::Ok;
  }
  if (types::is_invalid(r)) return Status::Invalid;
  if (types::is_missing(r)) return Status::Ok;
  throw std::logic_error("fn::adapter::integer: an integer kernel returned neither an integer, invalid nor missing");
}
// A call with an operand that holds no integer is computed whole by
// `Legacy`, the double kernel it ran before P4.5 (legacy::B17); otherwise an
// operand outside int64 makes the result invalid, and the int64 kernel `I`
// computes it.
template <types::Value (*I)(types::Integer), types::Value (*Legacy)(scalar::Double)>
Status integer_unary(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  types::Integer x{};
  switch (integer_operand(args[0], x)) {
    case IntegerOperand::NotInteger: return number(b, Legacy(operand(b, args, 0)), out);  // legacy::B17
    case IntegerOperand::Overflow: return Status::Invalid;
    case IntegerOperand::Integer: break;
  }
  return integer(I(x), out);
}
template <types::Value (*I)(types::Integer, types::Integer), types::Value (*Legacy)(scalar::Double, scalar::Double)>
Status integer_binary(const Bound& b, const ::Value* args, std::size_t, ::Value& out) {
  types::Integer x{}, y{};
  const IntegerOperand rx = integer_operand(args[0], x);
  const IntegerOperand ry = integer_operand(args[1], y);
  if (rx == IntegerOperand::NotInteger || ry == IntegerOperand::NotInteger)  // legacy::B17
    return number(b, Legacy(operand(b, args, 0), operand(b, args, 1)), out);
  if (rx == IntegerOperand::Overflow || ry == IntegerOperand::Overflow) return Status::Invalid;
  return integer(I(x, y), out);
}
// The aggregates over integers (sum, product, max, min): the present
// operands, missing ones skipped, all missing missing (as `fold`). The
// result is the exact aggregate, invalid when it is outside int64 (the value
// space is the integers, int64 its representable subset, D4), so an
// intermediate value outside int64 does not by itself make it invalid: the
// sum is accumulated in 128 bits (exact for any number of arguments a call
// can have); so is the product while its magnitude is at most 2^63 (the
// product of two such values fits in 128 bits), and a product of nonzero
// integers never shrinks in magnitude, so once beyond 2^63 it stays outside
// int64, unless a later factor is 0, which makes it 0. (product(INT64_MIN,
// -1, -1) is INT64_MIN although INT64_MIN * -1 is not an int64.) An operand
// outside int64 makes the result invalid; one that holds no integer makes
// the whole call the double fold it was (legacy::B17).
enum class Aggregate : std::uint8_t { Sum, Product, Max, Min };
template <Aggregate A, scalar::Double (*Legacy)(scalar::Double, scalar::Double)>
Status integer_fold(const Bound& b, const ::Value* args, std::size_t n, ::Value& out) {
  bool any = false, operand_overflow = false, overflow = false, zero = false;
  constexpr __int128 kTwo63 = static_cast<__int128>(1) << 63;
  __int128 wide = A == Aggregate::Product ? 1 : 0;
  std::int64_t acc = 0;
  for (std::size_t i = 0; i < n; i++) {
    if (args[i].is_missing()) continue;
    types::Integer x{};
    switch (integer_operand(args[i], x)) {
      case IntegerOperand::NotInteger: return fold<Legacy>(b, args, n, out);  // legacy::B17
      case IntegerOperand::Overflow: operand_overflow = true; continue;
      case IntegerOperand::Integer: break;
    }
    if constexpr (A == Aggregate::Sum) {
      wide += x.v;
    } else if constexpr (A == Aggregate::Product) {
      if (x.v == 0) {
        zero = true;
      } else if (!overflow) {
        wide *= x.v;
        if (wide > kTwo63 || wide < -kTwo63) overflow = true;
      }
    } else if constexpr (A == Aggregate::Max) {
      acc = (!any || x.v > acc) ? x.v : acc;
    } else {
      acc = (!any || x.v < acc) ? x.v : acc;
    }
    any = true;
  }
  if (operand_overflow) return Status::Invalid;
  if (!any) return Status::Ok;
  if constexpr (A == Aggregate::Sum || A == Aggregate::Product) {
    if (A == Aggregate::Product && zero) {
      acc = 0;
    } else {
      if (overflow || wide < std::numeric_limits<std::int64_t>::min() ||
          wide > std::numeric_limits<std::int64_t>::max())
        return Status::Invalid;
      acc = static_cast<std::int64_t>(wide);
    }
  }
  out = ::Value(static_cast<double>(acc));
  return Status::Ok;
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
// not: "Negates input boolean value" (BuiltinFunctions.html, not). The
// argument is a boolean (`<ParameterField name="x" dataType="boolean"/>`;
// ApplyNode refuses any other static type at load, B53, design item P4.8),
// held as the double 1 or 0; it is true when that double is not 0.
inline Status logical_not(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  return boolean(!(args[0].as_double() != 0.0), out);
}
// and, or (design item P4.8). BuiltinFunctions.html: "Evaluate the results of
// two or more boolean values. and: true only if all input values are true,
// false otherwise. or: true if a single input value is true, false only if
// all input values are false." The inputs are booleans (ApplyNode refuses any
// other static type at load), each true when the double it holds is not 0.
// They are never missing here: the functions give no missing rule of their
// own, so "any missing inputs to a built-in function will result in a
// missing value being returned" (MissingRule::Propagate; ApplyNode returns
// mapMissingTo or a missing result before the kernel), and no three-valued
// logic applies: and(false, missing) is missing, not false. The spec's
// three-valued tables are CompoundPredicate's (TreeModel.html), not these
// functions'.
inline Status logical_and(const Bound&, const ::Value* args, std::size_t n, ::Value& out) {
  for (std::size_t i = 0; i < n; i++)
    if (args[i].as_double() == 0.0) return boolean(false, out);
  return boolean(true, out);
}
inline Status logical_or(const Bound&, const ::Value* args, std::size_t n, ::Value& out) {
  for (std::size_t i = 0; i < n; i++)
    if (args[i].as_double() != 0.0) return boolean(true, out);
  return boolean(false, out);
}

// String edits (legacy::B2).
template <void (*Edit)(::Value&)>
Status string_edit(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  out = args[0];
  Edit(out);
  return Status::Ok;
}
// substring(input, startPos, length) (P4.4, B41): "startPos and length must
// be positive integers. The first character of a string is located at
// position 1 (not position 0)", so a startPos or length below 1, or one that
// is no integer (INF, NaN: "INF and -INF can only be floating point
// numbers"), is outside the domain: invalid. Within it the pseudo-
// declaration's "See XQuery fn:substring" applies: "The characters returned
// do not extend beyond $sourceString", so a startPos past the end gives the
// empty string and a length past the end stops at it (P4.1 made a startPos
// past the end invalid). The arguments are read as today: the double each
// holds, truncated (a non-integral startPos or length, which P4.4 left open,
// is not P4.5's: these are no integer-typed arithmetic).
inline Status substring(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  const double start_pos = args[1].as_double();  // 1-based
  const double length = args[2].as_double();
  if (!(start_pos >= 1.0 && length >= 1.0) || std::isinf(start_pos) || std::isinf(length)) return Status::Invalid;
  out = args[0];
  const auto size = static_cast<double>(out.text().size());
  const double start = std::min(start_pos - 1.0, size);
  out.substr(static_cast<std::size_t>(start), static_cast<std::size_t>(std::min(length, size)));
  return Status::Ok;
}
// concat (P4.6, B23): each argument cast to string (scalar::append), then
// joined. The cast is read from the argument's static type, fixed at load
// (Bound::argument_type), since the old Value carries no type:
//   - a string is its text, as it came in (no unpack/pack, decided
//     2026-11-19 after P3), a numeric_text value included (§9, decided
//     2026-11-18 after P2.1: "string readers get the text");
//   - a number or boolean that carries a text has the text the load gave it
//     (bind::slot_constant, element_constant, number_constant: a document
//     constant in a string slot "keeps its text", so concat(FieldRef d
//     mapMissingTo="5", "z") stays "5z", as base and JPMML 1.7.7 give);
//   - any other number or boolean is written as Java writes its type:
//     integer the int64 the double holds (an integer slot that keeps a
//     fraction, ±INF or NaN, legacy::B17, is written as the double it holds,
//     as java::text writes such an integer; P4.7 makes it unreachable), float
//     the double narrowed to float, double the double, boolean 1 -> true,
//     else false (a boolean slot holds 1 or 0).
// A missing argument never reaches the kernel (MissingRule::Propagate:
// "any missing inputs to a built-in function will result in a missing value
// being returned"). The result is a string the kernel built: packed as the
// engine builds one (Value(text, STRING): the active Indexer's code).
inline void append_text(std::string& out, const ::Value& v, types::TypeId type) {
  if (type == types::TypeId::String || !v.text().empty()) return scalar::append(out, v.text());
  const double d = v.as_double();
  switch (type) {
    case types::TypeId::Integer:
      if (java::is_long(d)) return scalar::append(out, types::Integer{static_cast<std::int64_t>(d)});
      return scalar::append(out, types::Double{d});  // legacy::B17
    case types::TypeId::Float:
      return scalar::append(out, types::Float{static_cast<float>(d)});
    case types::TypeId::Double:
      return scalar::append(out, types::Double{d});
    case types::TypeId::Boolean:
      return scalar::append(out, types::Boolean{d != 0});
    case types::TypeId::String:
    case types::TypeId::Missing:
    case types::TypeId::Invalid:
      break;
  }
  throw std::logic_error("concat: a sentinel is not a static type");
}
inline Status concat(const Bound& bound, const ::Value* args, std::size_t n, ::Value& out) {
  if (bound.argument_type.size() != n) throw std::logic_error("concat: the call was not bound to its argument types");
  std::string result;
  for (std::size_t i = 0; i < n; i++) append_text(result, args[i], bound.argument_type[i]);
  out = ::Value(result, DataType::DataTypeValue::STRING);
  return Status::Ok;
}
// The static type concat casts an argument from: the binder's
// (bind::expression_type), or Double where it does not know one (arithmetic
// over a boolean or string operand, Q7, reported by the binder; a FieldRef
// to an OutputField whose dataType neither the document nor its feature
// gives): the engine computed a double there, and a string OutputField over
// the same expression writes it as one (TransformedValue::kind_of, P2.4).
inline types::TypeId text_type(const std::optional<types::TypeId>& type) {
  if (!type) return types::TypeId::Double;
  if (*type == types::TypeId::Missing || *type == types::TypeId::Invalid)
    throw std::logic_error("concat: a sentinel is not a static type");
  return *type;
}
// formatNumber (P4.9; scalar::format_number). The input is read as a number
// of its static type (Bound::argument_type[0]): the double the Value holds, a
// float narrowed first. The pattern is the one parsed at load when it is a
// Constant (Bound::number_pattern; a pattern formatNumber cannot apply is
// then a load error), else the argument's text, cast to string by its static
// type as concat casts (append_text), parsed here: a pattern that cannot be
// applied is an invalid result, as is a value an integer conversion cannot
// take. The result is a string the kernel built: packed (Value(text,
// STRING)). A missing input or pattern never reaches the kernel
// (MissingRule::Propagate).
inline Status format_number(const Bound& bound, const ::Value* args, std::size_t n, ::Value& out) {
  if (bound.argument_type.size() != n)
    throw std::logic_error("formatNumber: the call was not bound to its argument types");
  double v = args[0].as_double();
  if (bound.argument_type[0] == types::TypeId::Float) v = static_cast<double>(types::narrow_to_float(v));
  std::string result;
  if (bound.number_pattern) {
    if (!scalar::format_number(*bound.number_pattern, v, result)) return Status::Invalid;
  } else {
    std::string text;
    append_text(text, args[1], bound.argument_type[1]);
    scalar::NumberPattern pattern;
    if (!scalar::parse_number_pattern(text, pattern).empty() || !scalar::format_number(pattern, v, result))
      return Status::Invalid;
  }
  out = ::Value(result, DataType::DataTypeValue::STRING);
  return Status::Ok;
}
inline Status string_length(const Bound&, const ::Value* args, std::size_t, ::Value& out) {
  // The integer stored as a double (the old Value's only number, until P5).
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
//
// `integer_result`: the call's result type is integer (bind::
// apply_result_type, the table's result rule over the arguments' static
// types; ApplyNode passes it). The arithmetic and the aggregates whose
// result is the least restrictive type of their arguments then get their
// int64 kernel (P4.5, D4). The other rows are unchanged: floor, ceil and
// round always return an integer and have one kernel; threshold gives 0 or
// 1, the same in int64 and in double; avg, median and the functions
// declared double never have an integer result.
inline KernelFn instantiate(Op op, bool integer_result = false) {
  namespace a = adapter;
  namespace s = scalar;
  if (integer_result) {
    switch (op) {
      case Op::Plus: return a::integer_binary<s::plus, s::b17<s::plus>>;
      case Op::Minus: return a::integer_binary<s::minus, s::b17<s::minus>>;
      case Op::Times: return a::integer_binary<s::times, s::b17<s::times>>;
      case Op::Divide: return a::integer_binary<s::divide, s::b17_divide>;
      case Op::Pow: return a::integer_binary<s::pow, s::b17_pow>;
      case Op::Modulo: return a::integer_binary<s::modulo, s::b17_modulo>;
      case Op::Abs: return a::integer_unary<s::abs, s::b17<s::abs>>;
      case Op::Max: return a::integer_fold<a::Aggregate::Max, s::max>;
      case Op::Min: return a::integer_fold<a::Aggregate::Min, s::min>;
      case Op::Sum: return a::integer_fold<a::Aggregate::Sum, s::sum>;
      case Op::Product: return a::integer_fold<a::Aggregate::Product, s::product>;
      default: break;
    }
  }
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
    case Op::Round: return a::to_integer_unary<s::round_half_away>;
    case Op::Floor: return a::to_integer_unary<s::round_down>;
    case Op::Ceil: return a::to_integer_unary<s::round_up>;
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
    case Op::Modulo: return a::checked_binary<s::modulo, s::modulo_domain>;
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
    case Op::And: return a::logical_and;
    case Op::Or: return a::logical_or;
    case Op::FormatNumber: return a::format_number;
    case Op::None: break;
  }
  throw std::logic_error("fn::instantiate: no kernel for this function");
}

// The functions whose numeric arguments and result ApplyNode types (the
// arithmetic, aggregate, mathematical and distribution functions), and the
// comparisons, whose float arguments it narrows. The others (if, isMissing,
// isNotMissing, not, and, or, the string functions, formatNumber) read their
// arguments as they are (formatNumber narrows its float input itself, by
// Bound::argument_type).
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
    case Op::And:
    case Op::Or:
    case Op::Replace:
    case Op::Matches:
    case Op::Uppercase:
    case Op::Lowercase:
    case Op::Substring:
    case Op::TrimBlanks:
    case Op::Concat:
    case Op::StringLength:
    case Op::FormatNumber:
      return false;
    default:
      return !compares(op);
  }
}

}  // namespace cpmml::fn

#endif  // CPMML_FN_KERNELS_H
