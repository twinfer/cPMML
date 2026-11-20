/*******************************************************************************
 * The built-in function table and its kernels (src/fn/table.h, kernels.h,
 * apply_node.h; docs/design/typed-core.md §4, commit P4.1).
 *
 *   - One table: every row the engine evaluates has a kernel, every row it
 *     refuses has none and stays a load error, with the engine's messages.
 *   - The scalar kernels against BuiltinFunctions.html's own examples and
 *     definitions (modulo's four examples, threshold, the integer division
 *     rule of §4.2, pow of integers with a negative exponent, D8; normalCDF's
 *     sigma), never against cPMML's output.
 *   - The legacy switches (kernels.h `legacy`): what each keeps is pinned
 *     here by the symptom the design's bug table records for it (§1.2: B2
 *     "keep the argument's code"), so that the commit removing a switch sees
 *     these checks fail and replaces them with the spec's. B17's switch is
 *     gone (P4.7): an integer operand that holds no int64 is invalid.
 *   - concat's casts of its arguments to string (P4.6, B23; Java text, Q1,
 *     D17), kernel by kernel and through a whole document.
 *   - Division by zero and the functions' domains (P4.4: B21, D20, B41),
 *     kernel by kernel and through a whole document.
 *   - The int64 kernels of integer calls (P4.5, D4), exactness and
 *     overflow, kernel by kernel and through a whole document.
 *   - formatNumber (P4.9): POSIX printf patterns against the C library's
 *     snprintf, the patterns it cannot apply, float narrowing, and through a
 *     whole document (load errors included).
 *   - The explicit casts of a declaration (P4.7, fn/cast.h: B17, B24; §2.6,
 *     §3.4, §5.1, Q2), rule by rule and through whole documents: a
 *     DerivedField, an OutputField, the target and an entityId.
 *   - ArgBuffer, inline and on the heap.
 *
 * Usage: fn_table.exe   (exit 0 when every check holds)
 *******************************************************************************/

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/internal_evaluator.h"  // first: the expression headers do not compile as the first of their cycle
#include "core/indexer.h"
#include "core/value.h"
#include "cPMML.h"
#include "fn/apply_node.h"
#include "fn/cast.h"
#include "fn/kernels.h"
#include "fn/table.h"

namespace fn = cpmml::fn;
namespace t = cpmml::types;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, rule)                                                                                  \
  do {                                                                                                     \
    ++checks;                                                                                              \
    if (!(cond)) {                                                                                         \
      ++failures;                                                                                          \
      std::cerr << __FILE__ << ":" << __LINE__ << ": FAILED " #cond "\n    rule: " << (rule) << std::endl; \
    }                                                                                                      \
  } while (0)

template <class E, class F>
static std::string message_of(F&& f) {
  try {
    f();
  } catch (const E& e) {
    return e.what();
  } catch (...) {
    return "<another exception>";
  }
  return "<no exception>";
}

static bool near(double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(1.0, std::abs(b)); }

// Calls the kernel of `name` over `args` with `bound`; `integer`: the call's
// result type is integer (its int64 kernel, P4.5).
static constexpr bool kInteger = true;
static fn::Status call(const char* name, const std::vector<Value>& args, Value& out, const fn::Bound& bound = {},
                       bool integer = false) {
  const fn::Signature* sig = fn::find(name);
  if (!sig) throw std::logic_error(std::string("no row ") + name);
  return fn::instantiate(sig->op, integer)(bound, args.data(), args.size(), out);
}
static double number(const char* name, const std::vector<Value>& args, const fn::Bound& bound = {},
                     bool integer = false) {
  Value out;
  if (call(name, args, out, bound, integer) != fn::Status::Ok || out.is_missing())
    throw std::logic_error(std::string(name) + ": no number");
  return out.as_double();
}
static bool invalid(const char* name, const std::vector<Value>& args, const fn::Bound& bound = {},
                    bool integer = false) {
  Value out;
  return call(name, args, out, bound, integer) == fn::Status::Invalid;
}
static std::vector<Value> nums(std::initializer_list<double> xs) {
  std::vector<Value> v;
  for (double x : xs) v.emplace_back(x);
  return v;
}

static void table() {
  // §4.2 status column: a kernel exactly for the functions the engine
  // evaluates ("✓"), none for the others ("✗"); `if` included (P4.3).
  int evaluated = 0;
  for (const auto& s : fn::kBuiltins) {
    const std::string rule = "row " + std::string(s.name) + ": a kernel exactly when Engine::Today";
    if (s.engine == fn::Engine::Today) {
      evaluated++;
      CHECK(fn::instantiate(s.op) != nullptr, rule);
    } else {
      CHECK(s.op == fn::Op::None, rule);
      CHECK(message_of<std::logic_error>([&] { fn::instantiate(s.op); }) ==
                "fn::instantiate: no kernel for this function",
            rule);
    }
  }
  // The 59 functions core/builtinfunctions.h evaluated, and and or (P4.8),
  // formatNumber (P4.9); the 8 others it refused.
  CHECK(evaluated == 62, "every function cPMML evaluated before P4.1, and, or and formatNumber, has a kernel");
  CHECK(std::size(fn::kBuiltins) == 70, "BuiltinFunctions.html lists 70 functions");

  // The load errors the engine gave (core/builtinfunctions.h from_string),
  // word for word: they are LOAD-ERROR cells of the fixture dumps.
  CHECK(message_of<cpmml::ParsingException>([] { fn::engine_row("log"); }) ==
            "function \"log\" is not a PMML built-in: the natural logarithm is \"ln\" and the base-10 logarithm "
            "is \"log10\"",
        "log: the hint");
  CHECK(message_of<cpmml::ParsingException>([] { fn::engine_row("ceiling"); }) ==
            "function \"ceiling\" is not a PMML built-in: it is \"ceil\"",
        "ceiling: the hint");
  CHECK(message_of<cpmml::ParsingException>([] { fn::engine_row("Sum"); }) ==
            "invalid value \"Sum\" for Apply@function: not a PMML built-in function, and user-defined functions "
            "(DefineFunction) are unsupported",
        "names are case-sensitive; an unknown name is a load error");
  const std::string supported =
      "+, -, *, /, max, min, sum, avg, exp, ln, log10, sqrt, abs, round, floor, ceil, pow, if, isMissing, "
      "isNotMissing, equal, notEqual, lessThan, lessOrEqual, greaterThan, greaterOrEqual, isIn, isNotIn, replace, "
      "matches, uppercase, lowercase, substring, trimBlanks, concat, stringLength, median, product, threshold, "
      "modulo, not, expm1, ln1p, hypot, rint, sin, asin, sinh, cos, acos, cosh, tan, atan, tanh, erf, normalCDF, "
      "normalPDF, stdNormalCDF, stdNormalPDF, and, or, formatNumber";
  for (const char* refused : {"isValid", "isNotValid", "normalIDF", "stdNormalIDF", "formatDatetime", "dateDaysSinceYear", "dateSecondsSinceYear",
                              "dateSecondsSinceMidnight"})
    CHECK(message_of<cpmml::ParsingException>([&] { fn::engine_row(refused); }) ==
              "unsupported value \"" + std::string(refused) + "\" for Apply@function; cPMML supports " + supported,
          std::string(refused) + ": refused at load, as before P4.1 (and and or are evaluated since P4.8, "
                                 "formatNumber since P4.9, at the end of the list)");
  CHECK(&fn::engine_row("+") == fn::find("+"), "the engine reads the binder's row");
}

static void kernels() {
  // modulo: "the remainder after x is divided by y, computed according to
  // XQuery rules ... modulo(11, 3) = 2, modulo(-17.2, 0.5) = 0.3,
  // modulo(9, -7) = -5, modulo(-4, -9) = -4".
  CHECK(near(number("modulo", nums({11, 3})), 2), "modulo(11,3) = 2");
  CHECK(near(number("modulo", nums({-17.2, 0.5})), 0.3), "modulo(-17.2,0.5) = 0.3");
  CHECK(near(number("modulo", nums({9, -7})), -5), "modulo(9,-7) = -5");
  CHECK(near(number("modulo", nums({-4, -9})), -4), "modulo(-4,-9) = -4");
  // threshold: "Returns one if x>y and zero otherwise".
  CHECK(number("threshold", nums({3, 2})) == 1 && number("threshold", nums({2, 2})) == 0, "threshold");
  // Integer "/": truncation toward zero (§4.2, Java and XQuery idiv).
  CHECK(number("/", nums({7, 2}), {}, kInteger) == 3, "7/2 of integers is 3");
  CHECK(number("/", nums({-7, 2}), {}, kInteger) == -3, "-7/2 of integers is -3 (toward zero)");
  CHECK(number("/", nums({7, 2})) == 3.5, "7/2 of doubles is 3.5");
  // pow of integers with a negative exponent has no integer value (D8).
  CHECK(invalid("pow", nums({7, -2}), {}, kInteger), "pow(7,-2) of integers is invalid (D8)");
  CHECK(number("pow", nums({7, 2}), {}, kInteger) == 49, "pow(7,2) of integers is 49");
  CHECK(number("pow", nums({4, -0.5})) == 0.5, "pow(4,-0.5) of doubles is 0.5");
  // normalCDF(x, mu, sigma): sigma must be positive.
  CHECK(invalid("normalCDF", nums({0, 0, 0})), "normalCDF with sigma 0 is invalid");
  CHECK(invalid("normalPDF", nums({0, 0, -1})), "normalPDF with sigma < 0 is invalid");
  CHECK(near(number("normalCDF", nums({1, 1, 2})), 0.5), "normalCDF(mu, mu, sigma) = 0.5");
  CHECK(near(number("stdNormalCDF", nums({0})), 0.5), "stdNormalCDF(0) = 0.5");
  // The aggregates skip missing values; all missing gives missing.
  std::vector<Value> some = nums({3, 1});
  some.emplace_back();
  some.push_back(Value(4.0));
  some.push_back(Value(2.0));
  CHECK(number("sum", some) == 10, "sum skips the missing value");
  CHECK(number("avg", some) == 2.5, "avg of the four present values");
  CHECK(number("median", some) == 2.5, "median of an even count: the mean of the two central values");
  CHECK(number("max", some) == 4 && number("min", some) == 1 && number("product", some) == 24, "max, min, product");
  {
    Value out;
    const std::vector<Value> none(3);
    CHECK(call("sum", none, out) == fn::Status::Ok && out.is_missing(), "sum of missing values is missing");
  }
  // A float argument is narrowed to an xs:float first, a float result after.
  fn::Bound floats;
  floats.narrow_arguments = true;
  floats.float_argument = {1, 0, 0};
  floats.float_result = true;
  CHECK(number("+", nums({0.1, 0.0}), floats) == static_cast<double>(0.1f), "float 0.1 + 0 is the float 0.1");
  // The comparisons read each operand as its static type (P6.3,
  // Bound::argument_type): a float one is the float, compared in double (D1).
  fn::Bound float_compare;
  float_compare.argument_type = {cpmml::types::TypeId::Float, cpmml::types::TypeId::Double,
                                 cpmml::types::TypeId::Double};
  CHECK(number("equal", nums({0.1, 0.1}), float_compare) == 0, "equal(float 0.1, double 0.1) is false");
  CHECK(number("isIn", nums({0.1, 0.2, 0.1}), float_compare) == 0, "isIn(float 0.1, {0.2, 0.1}) is false");
  CHECK(number("equal", nums({0.1, static_cast<double>(0.1f)}), float_compare) == 1,
        "equal(float 0.1, double 0.10000000149011612) is true");
  fn::Bound float_sum;
  float_sum.narrow_arguments = true;
  float_sum.float_argument = {1, 1};
  float_sum.float_result = true;
  CHECK(number("+", nums({0.1, 0.2}), float_sum) == static_cast<double>(0.1f + 0.2f),
        "float + float is the float sum (narrowed after), not the double sum of the two floats");
  CHECK(number("+", nums({0.1, 0.2}), float_sum) != static_cast<double>(0.1f) + static_cast<double>(0.2f),
        "the float sum differs from the double sum here, so the narrowing is observed");
  // The comparisons: "Return true or false. Result is dependent on applying
  // either function to two input parameters", on equal and on ordered
  // operands.
  struct Cmp {
    const char* name;
    double lt, eq, gt;  // f(1,2), f(2,2), f(3,2)
  };
  for (const Cmp& c : {Cmp{"equal", 0, 1, 0}, Cmp{"notEqual", 1, 0, 1}, Cmp{"lessThan", 1, 0, 0},
                       Cmp{"lessOrEqual", 1, 1, 0}, Cmp{"greaterThan", 0, 0, 1}, Cmp{"greaterOrEqual", 0, 1, 1}})
    CHECK(number(c.name, nums({1, 2})) == c.lt && number(c.name, nums({2, 2})) == c.eq &&
              number(c.name, nums({3, 2})) == c.gt,
          std::string(c.name) + " over 1, 2, 3 against 2");
  CHECK(number("isIn", nums({2, 1, 2})) == 1 && number("isNotIn", nums({2, 1, 2})) == 0 &&
            number("isIn", nums({2, 1, 3})) == 0 && number("isNotIn", nums({2, 1, 3})) == 1,
        "isIn, isNotIn");
  // P6.3 (§2.7, D9, B44): ±INF against the same infinity and NaN against
  // anything are indeterminate, a missing result ("any attempt to either with
  // itself is indeterminate and will result in a missing value being
  // returned"); INF against a finite number or -INF is ordered.
  {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto missing = [](const char* name, const std::vector<Value>& args, const fn::Bound& bound = {}) {
      Value out;
      return call(name, args, out, bound) == fn::Status::Ok && out.is_missing();
    };
    for (const char* f : {"equal", "notEqual", "lessThan", "lessOrEqual", "greaterThan", "greaterOrEqual"}) {
      CHECK(missing(f, nums({inf, inf})) && missing(f, nums({-inf, -inf})), std::string(f) + "(INF, INF) is missing");
      CHECK(missing(f, nums({nan, 1})) && missing(f, nums({1, nan})) && missing(f, nums({nan, nan})),
            std::string(f) + " with NaN is missing");
    }
    CHECK(number("lessThan", nums({-inf, inf})) == 1 && number("greaterThan", nums({inf, 1e308})) == 1 &&
              number("equal", nums({inf, -inf})) == 0,
          "INF against -INF or a finite number is ordered");
    CHECK(missing("isIn", nums({inf, inf, 1})) && missing("isNotIn", nums({inf, inf, 1})),
          "isIn(INF, INF, 1) is missing (the or of an indeterminate and a false)");
    CHECK(missing("isIn", nums({inf, 1, inf})), "isIn(INF, 1, INF) is missing");
    CHECK(number("isIn", nums({inf, 1, 2})) == 0 && number("isNotIn", nums({inf, 1, 2})) == 1,
          "isIn(INF, 1, 2) is false");
    CHECK(number("isIn", nums({1, inf, 1})) == 1, "isIn(1, INF, 1) is true: one equality holds");
    CHECK(missing("isIn", nums({nan, 1, 2})), "isIn(NaN, 1, 2) is missing");
    // D1: integer x float compares in float, either order.
    fn::Bound int_float;
    int_float.argument_type = {cpmml::types::TypeId::Integer, cpmml::types::TypeId::Float};
    const std::vector<Value> big = {Value(cpmml::types::Integer{16777217}), Value(cpmml::types::Float{16777216.0f})};
    CHECK(number("equal", big, int_float) == 1, "equal(integer 16777217, float 16777216) is true (in float)");
    fn::Bound float_int;
    float_int.argument_type = {cpmml::types::TypeId::Float, cpmml::types::TypeId::Integer};
    CHECK(number("equal", {big[1], big[0]}, float_int) == 1, "and in the other order");
    // A boolean operand is read as its static type: a boolean result held as
    // the Double 1 or 0 is that boolean.
    fn::Bound booleans;
    booleans.argument_type = {cpmml::types::TypeId::Boolean, cpmml::types::TypeId::Boolean};
    CHECK(number("equal", {Value(1.0), Value(cpmml::types::Boolean{true})}, booleans) == 1 &&
              number("lessThan", {Value(0.0), Value(cpmml::types::Boolean{true})}, booleans) == 1,
          "false < true; a boolean result 1 is true");
    // No common type at evaluation (static types unknown) is an invalid result.
    CHECK(invalid("equal", {Value(1.0), Value(cpmml::types::Boolean{true})}),
          "a number against a boolean, types unknown: invalid (§2.6 Invalid(NoCast))");
  }
  // and, or (P4.8): "and: true only if all input values are true, false
  // otherwise. or: true if a single input value is true, false only if all
  // input values are false"; not: "Negates input boolean value". Every
  // combination of two and of three booleans, and five (more than ArgBuffer
  // holds inline, though the kernel reads a plain array).
  for (int bits = 0; bits < 8; bits++) {
    const double a = bits & 1, b = (bits >> 1) & 1, c = (bits >> 2) & 1;
    const std::string which = "over " + std::to_string(bits);
    CHECK(number("and", nums({a, b})) == (a && b ? 1 : 0), "and of two " + which);
    CHECK(number("or", nums({a, b})) == (a || b ? 1 : 0), "or of two " + which);
    CHECK(number("and", nums({a, b, c})) == (a && b && c ? 1 : 0), "and of three " + which);
    CHECK(number("or", nums({a, b, c})) == (a || b || c ? 1 : 0), "or of three " + which);
  }
  CHECK(number("and", nums({1, 1, 1, 1, 1})) == 1 && number("and", nums({1, 1, 1, 1, 0})) == 0, "and of five");
  CHECK(number("or", nums({0, 0, 0, 0, 0})) == 0 && number("or", nums({0, 0, 0, 0, 1})) == 1, "or of five");
  CHECK(number("not", nums({1})) == 0 && number("not", nums({0})) == 1, "not");
  // The rows: boolean parameters, a boolean result, two or more arguments,
  // and a missing input propagates (no rule of their own).
  for (const char* name : {"and", "or"}) {
    const fn::Signature* sig = fn::find(name);
    CHECK(sig->min_args == 2 && sig->max_args == fn::kVariadic && sig->param(0) == fn::Param::Boolean &&
              sig->param(7) == fn::Param::Boolean && sig->result == fn::Result::Fixed &&
              sig->fixed == t::TypeId::Boolean && sig->missing == fn::MissingRule::Propagate &&
              sig->engine == fn::Engine::Today,
          std::string(name) + ": the row");
  }
}

static void legacy_switches(Indexer& indexer) {
  // B17 (P4.7, legacy::B17 removed): an integer operand that holds no int64
  // (a fraction, ±INF, NaN; P4.7's casts keep integer slots integral) makes
  // the integer call invalid, as §2.6's cast of the double to an integer is
  // ("the value if integral and in range, else Invalid(NotIntegral/
  // Overflow)"); it was computed in double before (integer 3.5 + 1 was 4.5).
  {
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(invalid("+", nums({3.5, 1}), {}, kInteger), "B17: integer 3.5 + 1 is invalid");
    CHECK(invalid("-", nums({1, inf}), {}, kInteger), "B17: integer 1 - INF is invalid");
    CHECK(invalid("*", nums({std::nan(""), 2}), {}, kInteger), "B17: integer NaN * 2 is invalid");
    CHECK(invalid("/", nums({7.5, 2}), {}, kInteger), "B17: integer 7.5/2 is invalid");
    CHECK(invalid("pow", nums({2, 0.5}), {}, kInteger), "B17: integer pow(2, 0.5) is invalid");
    CHECK(invalid("modulo", nums({7.5, 2}), {}, kInteger), "B17: integer modulo(7.5, 2) is invalid");
    CHECK(invalid("abs", nums({-3.5}), {}, kInteger), "B17: integer abs(-3.5) is invalid");
    CHECK(invalid("sum", nums({3.5, 1, 1}), {}, kInteger), "B17: integer sum(3.5, 1, 1) is invalid");
    CHECK(invalid("max", nums({1, -inf}), {}, kInteger), "B17: integer max(1, -INF) is invalid");
    CHECK(invalid("product", nums({0, 2.5}), {}, kInteger), "B17: integer product(0, 2.5) is invalid");
    CHECK(number("+", nums({3, 1}), {}, kInteger) == 4, "integer 3 + 1 is 4");
    CHECK(number("sum", nums({-0.0, 1}), {}, kInteger) == 1, "a -0 operand is the integer 0");
  }
  // B2 (closed by P6.1): uppercase gives the string of its text, not its
  // argument's string code.
  {
    Value out;
    const Value abc = Value::string("abc");
    const Value up = Value::string("ABC");
    const std::vector<Value> args{abc};
    CHECK(call("uppercase", args, out) == fn::Status::Ok && out.text() == "ABC", "uppercase(abc) is ABC");
    CHECK(out.as_double() == up.as_double() && out.as_double() != abc.as_double(),
          "P6.1 (B2): uppercase(abc) has the code of ABC, not its argument's");
    Value eq;
    CHECK(call("equal", {out, up}, eq) == fn::Status::Ok && eq.as_double() == 1.0 &&
              call("equal", {out, abc}, eq) == fn::Status::Ok && eq.as_double() == 0.0,
          "P6.1 (B2): equal(uppercase(abc), ABC) is true, equal(uppercase(abc), abc) false");
  }
  // replace and matches: a pattern std::regex refuses is an invalid result.
  {
    const Value abc = Value::string("abc");
    const Value bad = Value::string("(");
    const Value x = Value::string("x");
    CHECK(invalid("replace", {abc, bad, x}), "replace with pattern ( is invalid");
    CHECK(invalid("matches", {abc, bad}), "matches with pattern ( is invalid");
    Value out;
    CHECK(call("replace", {abc, Value::string("b"), x}, out) == fn::Status::Ok &&
              out.text() == "axc",
          "replace(abc, b, x) is axc");
  }
}

// Functions.html's "Output table for Apply" through a whole document (P4.2,
// B22), every row of it. The invalid result is pow of two integers with a
// negative exponent (D8) and replace with a pattern that is no regular
// expression; Functions.html, Apply:
//   "The optional attribute mapMissingTo defines the output value for the
//   cases when any of the function inputs are missing. This means that if it
//   is specified and any of the input values of the function are missing then
//   the function is not applied at all and the mapMissingTo value is returned
//   instead. ... the optional value defaultValue defines an output value when
//   the function returns a missing value. In other words, when a defaultValue
//   value is provided, the function is applied first and if it produces a
//   missing value the defaultValue is returned instead."
//   "This attribute's default value returnInvalid causes the model to return
//   a value indicating an invalid result. On the other hand, the
//   invalidValueTreatment value asMissing replaces the invalid value with a
//   missing value. Note that, in this case, if a defaultValue is also
//   specified then the default value is returned instead of a missing value.
//   Finally, in the context of an Apply, the value asIs is equivalent to
//   returnInvalid as an invalid value cannot be propagated further."
// The expected values are the spec's (JPMML 1.7.7 refuses pow and replace
// here under every treatment, D8 and D34; where it reaches the table, for
// integer division by zero, it gives the same cells but for asIs, D33).
// `apply` is the expression of the DerivedField p, which the OutputField o
// reads; with `in_output` it is o's own expression instead (p is then 0).
static std::string apply_document(const std::string& apply, const std::string& type, bool in_output) {
  const std::string optype = type == "string" ? "categorical" : "continuous";
  return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="i" optype="continuous" dataType="integer"/>
    <DataField name="e" optype="continuous" dataType="integer"/>
    <DataField name="t" optype="categorical" dataType="string"/>
    <DataField name="r" optype="categorical" dataType="string"/>
    <DataField name="y" optype="continuous" dataType="double"/>
  </DataDictionary>
  <RegressionModel functionName="regression">
    <MiningSchema><MiningField name="i"/><MiningField name="e"/><MiningField name="t"/><MiningField name="r"/>
      <MiningField name="y" usageType="target"/></MiningSchema>
    <LocalTransformations>
      <DerivedField name="p" optype=")" +
         optype + R"(" dataType=")" + type + R"(">)" + (in_output ? std::string("<Constant>0</Constant>") : apply) +
         R"(</DerivedField>
    </LocalTransformations>
    <Output><OutputField name="o" optype=")" +
         optype + R"(" dataType=")" + type + R"(" feature="transformedValue">)" +
         (in_output ? apply : std::string(R"(<FieldRef field="p"/>)")) + R"(</OutputField></Output>
    <RegressionTable intercept="1"/>
  </RegressionModel>
</PMML>
)";
}

static std::string pow_apply(const std::string& attrs) {
  return R"(<Apply function="pow")" + attrs + R"(><FieldRef field="i"/><FieldRef field="e"/></Apply>)";
}

// The OutputField "o" (the DerivedField "p", of type `type`) for inputs i,
// e, t and r (empty: missing), as text: "N/A" for no value, "ERROR: <message>" for an invalid
// record.
static std::string apply_output(const std::string& apply, const std::string& i, const std::string& e,
                                const std::string& t = "abc", const std::string& r = "b",
                                const std::string& type = "double", bool in_output = false) {
  const std::string path = "fn_table_apply.pmml";
  {
    std::ofstream out(path);
    out << apply_document(apply, type, in_output);
  }
  cpmml::Model model(path, false);
  std::remove(path.c_str());
  cpmml::Input input;
  if (!i.empty()) input["i"] = i;
  if (!e.empty()) input["e"] = e;
  if (!t.empty()) input["t"] = t;
  if (!r.empty()) input["r"] = r;
  try {
    const cpmml::Result result = model.evaluate(input);
    const auto num = result.num_outputs();
    if (const auto it = num.find("o"); it != num.end()) {
      std::ostringstream text;
      text << it->second;
      return text.str();
    }
    const auto str = result.str_outputs();
    if (const auto it = str.find("o"); it != str.end() && !it->second.empty()) return it->second;
    return "N/A";
  } catch (const cpmml::InvalidValueException& error) {
    return std::string("ERROR: ") + error.what();
  }
}

// concat (P4.6, B23): BuiltinFunctions.html, concat: "Returns a string as a
// result of the concatenation of two or more parameters", "See XQuery
// fn:concat", whose arguments are each cast to xs:string; the text of a
// number is Java's (Q1 = (a), D17: Long.toString, Float.toString,
// Double.toString; true/false). The expected texts are Java's, written out
// here (java.lang.Double.toString(2000.0) is "2000.0", Float.toString(0.1f)
// "0.1", Float.toString(16777217f) "1.6777216E7", Double.toString(-0.0)
// "-0.0"), not cPMML's.
static std::string concat_text(std::vector<cpmml::types::TypeId> types, const std::vector<Value>& args) {
  fn::Bound bound;
  bound.argument_type = std::move(types);
  Value out;
  if (call("concat", args, out, bound) != fn::Status::Ok || out.is_missing())
    throw std::logic_error("concat: no value");
  return std::string(out.text());
}
static std::string concat_one(cpmml::types::TypeId type, const Value& v) {
  return concat_text({type, cpmml::types::TypeId::String}, {v, Value::string("|")});
}
static void concat_casts(Indexer& indexer) {
  using cpmml::types::TypeId;
  const double inf = std::numeric_limits<double>::infinity();
  struct Case {
    TypeId type;
    double v;
    const char* text;
  };
  for (const Case& c : {
           // Double.toString: "2.0", not XQuery's "2" (D17).
           Case{TypeId::Double, 3, "3.0|"}, Case{TypeId::Double, 2000, "2000.0|"}, Case{TypeId::Double, 0.5, "0.5|"},
           Case{TypeId::Double, 1e7, "1.0E7|"}, Case{TypeId::Double, 1234567, "1234567.0|"},
           Case{TypeId::Double, 1e-4, "1.0E-4|"}, Case{TypeId::Double, 0.001, "0.001|"},
           Case{TypeId::Double, -0.0, "-0.0|"}, Case{TypeId::Double, inf, "Infinity|"},
           Case{TypeId::Double, -inf, "-Infinity|"}, Case{TypeId::Double, std::nan(""), "NaN|"},
           // Long.toString; XSD's integer has one zero.
           Case{TypeId::Integer, 2000, "2000|"}, Case{TypeId::Integer, -3, "-3|"}, Case{TypeId::Integer, -0.0, "0|"},
           Case{TypeId::Integer, 4611686018427387904.0, "4611686018427387904|"},
           // Float.toString of the double narrowed to float.
           Case{TypeId::Float, 0.1, "0.1|"}, Case{TypeId::Float, 1e8, "1.0E8|"}, Case{TypeId::Float, 3, "3.0|"},
           Case{TypeId::Float, 16777217, "1.6777216E7|"}, Case{TypeId::Float, 1e300, "Infinity|"},
           Case{TypeId::Float, 1.1, "1.1|"},
           // XSD's canonical boolean.
           Case{TypeId::Boolean, 1, "true|"}, Case{TypeId::Boolean, 0, "false|"}})
    CHECK(concat_one(c.type, Value(c.v)) == c.text,
          std::string("concat of ") + std::string(cpmml::bind::type_name(c.type)) + " " + std::to_string(c.v) +
              " is " + c.text + " (got " + concat_one(c.type, Value(c.v)) + ")");
  // B17 (P4.7, legacy::B17 removed): an integer argument that holds no
  // int64 has no integer text; the call is invalid, as an integer kernel's
  // is (it was written as the double it held, 3.5 -> "3.5").
  for (const double d : {3.5, inf, -inf, std::nan(""), 9223372036854775808.0}) {
    fn::Bound bound;
    bound.argument_type = {TypeId::Integer, TypeId::String};
    CHECK(invalid("concat", {Value(d), Value::string("|")}, bound),
          "B17: concat of an integer slot holding " + std::to_string(d) + " is invalid");
  }
  // A string is its text, as it came in: an empty one, one that spells a
  // number, and a numeric_text one (§9, decided 2026-11-18: string readers
  // get the text).
  CHECK(concat_one(TypeId::String, Value::string("ab")) == "ab|", "concat of a string");
  CHECK(concat_one(TypeId::String, Value::string("")) == "|", "concat of the empty string");
  CHECK(concat_one(TypeId::String, Value::string("2.50")) == "2.50|",
        "a string that spells a number keeps its text");
  {
    Value five = Value::string("5");
    five.mark_numeric_text();
    CHECK(concat_one(TypeId::String, five) == "5|", "a numeric_text string keeps its text");
  }
  // A number that carries a text from the load (bind::slot_constant: a
  // document constant in a string slot keeps its text) keeps it: "5", not
  // "5.0".
  {
    Value five(5.0);
    five.set_text("5");
    CHECK(concat_one(TypeId::Double, five) == "5|", "a number's load text is kept");
  }
  // Every argument, in order, each by its own type.
  CHECK(concat_text({TypeId::Integer, TypeId::String, TypeId::Double, TypeId::String, TypeId::Float, TypeId::String,
                     TypeId::Boolean},
                    {Value(2000.0), Value::string("|"), Value(2.0),
                     Value::string("|"), Value(0.1),
                     Value::string("|"), Value(1.0)}) == "2000|2.0|0.1|true",
        "concat(integer 2000, double 2.0, float 0.1, boolean true)");
  {
    // The result is a string the kernel built, packed with the pool's code.
    fn::Bound bound;
    bound.argument_type = {TypeId::Integer, TypeId::String};
    Value out;
    CHECK(call("concat", {Value(7.0), Value::string("|")}, out, bound) == fn::Status::Ok &&
              out.text() == "7|" && out.as_double() == Value::string("7|").as_double(),
          "concat packs its text with the pool's code");
  }
  // A call not bound to its arguments' types is a programming error.
  CHECK(message_of<std::logic_error>([] { concat_text({}, {Value(1.0), Value(2.0)}); }) ==
            "concat: the call was not bound to its argument types",
        "an unbound concat throws");
  // The type concat casts from: the binder's; Double where it knows none
  // (TransformedValue::kind_of's rule); a sentinel is no static type.
  CHECK(fn::adapter::text_type(std::nullopt) == TypeId::Double, "an unknown static type is cast as a double");
  CHECK(fn::adapter::text_type(TypeId::Float) == TypeId::Float, "a known static type is kept");
  CHECK(message_of<std::logic_error>([] { fn::adapter::text_type(TypeId::Missing); }) ==
            "concat: a sentinel is not a static type",
        "a sentinel static type throws");
}

static void apply_output_table() {
  const std::string invalid_return =
      R"(ERROR: Apply "pow" in DerivedField "p": invalid result (invalidValueTreatment="returnInvalid"))";
  const std::string invalid_as_is =
      R"(ERROR: Apply "pow" in DerivedField "p": invalid result (invalidValueTreatment="asIs"))";
  const std::vector<std::pair<std::string, std::string>> treatments = {
      {"", invalid_return},
      {R"( invalidValueTreatment="returnInvalid")", invalid_return},
      {R"( invalidValueTreatment="asIs")", invalid_as_is},
      {R"( invalidValueTreatment="asMissing")", "asMissing"},
  };
  const std::vector<std::string> extras = {"", R"( mapMissingTo="5")", R"( defaultValue="7")",
                                           R"( mapMissingTo="5" defaultValue="7")"};
  for (const auto& [treatment, invalid] : treatments)
    for (const std::string& extra : extras) {
      const std::string apply = pow_apply(treatment + extra);
      const std::string rule = "Apply" + treatment + extra + ": ";
      const bool mm = extra.find("mapMissingTo") != std::string::npos;
      const bool dv = extra.find("defaultValue") != std::string::npos;
      // Row 1: "at least one missing", mapMissingTo given: "not computed",
      // map_missing_val, whatever defaultValue and invalidValueTreatment are.
      // Row 3 (no mapMissingTo): the function's missing result, then
      // default_val.
      CHECK(apply_output(apply, "", "2") == (mm   ? "5"
                                             : dv ? "7"
                                                  : "N/A"),
            rule + "a missing input gives mapMissingTo, else missing, then defaultValue");
      CHECK(apply_output(apply, "2", "") == (mm   ? "5"
                                             : dv ? "7"
                                                  : "N/A"),
            rule + "a missing second input gives mapMissingTo, else missing, then defaultValue");
      // Row 2: no input missing, a value: out_val.
      CHECK(apply_output(apply, "2", "2") == "4", rule + "pow(2, 2) is 4 whatever the attributes");
      // Rows 4-6: an invalid result.
      const std::string expected = invalid != "asMissing" ? invalid : dv ? "7" : "N/A";
      CHECK(apply_output(apply, "2", "-2") == expected,
            rule + (invalid != "asMissing"
                        ? "returnInvalid, and asIs (\"equivalent to returnInvalid\"), make the record invalid"
                        : "asMissing gives missing, then defaultValue; never mapMissingTo (no input is missing)"));
    }

  // A string result: replace with the pattern "(" (std::regex refuses it).
  const std::string replace_apply_head = R"(<Apply function="replace")";
  const std::string replace_apply_tail =
      R"(><FieldRef field="t"/><FieldRef field="r"/><Constant dataType="string">x</Constant></Apply>)";
  auto replace_apply = [&](const std::string& attrs) { return replace_apply_head + attrs + replace_apply_tail; };
  CHECK(apply_output(replace_apply(R"( invalidValueTreatment="asMissing" mapMissingTo="five" defaultValue="seven")"),
                     "", "", "abc", "(", "string") == "seven",
        "replace with an invalid pattern under asMissing: defaultValue, not mapMissingTo");
  CHECK(apply_output(replace_apply(R"( invalidValueTreatment="asMissing" mapMissingTo="five")"), "", "", "abc", "(",
                     "string") == "N/A",
        "replace with an invalid pattern under asMissing, mapMissingTo only: missing");
  CHECK(apply_output(replace_apply(R"( invalidValueTreatment="asIs" mapMissingTo="five" defaultValue="seven")"), "", "",
                     "abc", "(", "string") ==
            R"(ERROR: Apply "replace" in DerivedField "p": invalid result (invalidValueTreatment="asIs"))",
        "replace with an invalid pattern under asIs: the record is invalid");
  CHECK(apply_output(replace_apply(R"( invalidValueTreatment="asMissing" mapMissingTo="five" defaultValue="seven")"),
                     "", "", "", "(", "string") == "five",
        "replace of a missing input under asMissing: mapMissingTo (the function is not applied)");
  CHECK(apply_output(replace_apply(R"( invalidValueTreatment="asIs" defaultValue="seven")"), "", "", "abc", "b",
                     "string") == "axc",
        "replace(abc, b, x) is axc under asIs");

  // The error names the function that failed and the field, through an
  // enclosing Apply; the enclosing Apply's own treatment does not catch it
  // (the invalid result is the inner Apply's, which raised it).
  CHECK(apply_output(R"(<Apply function="+" invalidValueTreatment="asMissing" defaultValue="9">)" + pow_apply("") +
                         R"(<Constant dataType="integer">1</Constant></Apply>)",
                     "2", "-2") == invalid_return,
        "a nested invalid pow names pow and the DerivedField; the outer asMissing does not apply to it");
  CHECK(apply_output(R"(<Apply function="+">)" + pow_apply(R"( invalidValueTreatment="asMissing" defaultValue="7")") +
                         R"(<Constant dataType="integer">1</Constant></Apply>)",
                     "2", "-2") == "8",
        "a nested asMissing pow gives defaultValue 7 to the outer +");
  CHECK(apply_output(pow_apply(R"( invalidValueTreatment="asIs")"), "2", "-2", "abc", "b", "double", true) ==
            R"(ERROR: Apply "pow" in OutputField "o": invalid result (invalidValueTreatment="asIs"))",
        "an Apply that is an OutputField's expression names the OutputField");
  CHECK(apply_output(pow_apply(R"( invalidValueTreatment="asMissing" mapMissingTo="5" defaultValue="7")"), "2", "-2",
                     "abc", "b", "double", true) == "7",
        "asMissing in an OutputField's expression: defaultValue, not mapMissingTo");
}

// Division by zero and the domains of the built-ins (P4.4: B21, D20, B41),
// kernel by kernel. BuiltinFunctions.html, each rule's sentence at its check.
static bool missing_result(const char* name, const std::vector<Value>& args, const fn::Bound& bound = {},
                           bool integer = false) {
  Value out;
  return call(name, args, out, bound, integer) == fn::Status::Ok && out.is_missing();
}
static bool text_result(const std::vector<Value>& args, const std::string& text) {
  Value out;
  return call("substring", args, out) == fn::Status::Ok && !out.is_missing() && out.text() == text;
}
// P4.5 (D4, Q10): a call whose result type is integer computes in int64,
// exactly; a result outside int64 is invalid. Functions.html: "functions are
// assumed to inherit the dataType of the current input parameters"; XML
// Schema Part 2 §3.3.13: "The value space of integer is the infinite set
// {...,-2,-1,0,1,2,...}"; D4: "integers are int64; overflow invalid". The
// result is stored as the double nearest to it (the old Value, until P5).
static void integer_kernels() {
  const double p53 = 9007199254740992.0;          // 2^53
  const double p62 = 4611686018427387904.0;       // 2^62
  const double p63 = 9223372036854775808.0;       // 2^63, outside int64
  const double min64 = -9223372036854775808.0;    // INT64_MIN
  const double p32 = 4294967296.0;
  const double p60 = 1152921504606846976.0;
  // + - *: exact, invalid outside int64.
  CHECK(invalid("+", nums({p62, p62}), {}, kInteger), "D4: 2^62 + 2^62 = 2^63 is outside int64: invalid");
  CHECK(number("+", nums({p62, p62})) == p63, "the double call is unchanged: 2^63");
  CHECK(number("+", nums({-p62, -p62}), {}, kInteger) == min64, "-2^62 + -2^62 = INT64_MIN is within int64");
  CHECK(invalid("-", nums({min64, 1}), {}, kInteger), "D4: INT64_MIN - 1 is invalid");
  CHECK(number("-", nums({min64, -1}), {}, kInteger) == -9223372036854775807.0, "INT64_MIN - -1, stored rounded");
  CHECK(invalid("*", nums({p32, p32}), {}, kInteger), "D4: 2^32 * 2^32 = 2^64 is invalid");
  CHECK(number("*", nums({-p62, 2}), {}, kInteger) == min64, "-2^62 * 2 = INT64_MIN is within int64");
  CHECK(number("*", nums({p53 + 2, 3}), {}, kInteger) == static_cast<double>((9007199254740994LL) * 3),
        "(2^53 + 2) * 3, exact then rounded once");
  // "/": truncation toward zero, exact; INT64_MIN / -1 overflows (§4.2).
  CHECK(invalid("/", nums({min64, -1}), {}, kInteger), "§4.2: integer INT64_MIN / -1 is invalid");
  CHECK(number("/", nums({4154830492154097664.0, 485}), {}, kInteger) == 8566660808565149.0,
        "4154830492154097664 / 485 is the exact quotient 8566660808565149");
  CHECK(number("/", nums({4154830492154097664.0, 485})) != 8566660808565149.0 &&
            std::trunc(4154830492154097664.0 / 485) == 8566660808565150.0,
        "... where trunc of the double quotient is 8566660808565150");
  CHECK(number("/", nums({-7, 2}), {}, kInteger) == -3 && number("/", nums({7, -2}), {}, kInteger) == -3,
        "toward zero");
  CHECK(number("/", nums({-0.0, 5}), {}, kInteger) == 0 && !std::signbit(number("/", nums({-0.0, 5}), {}, kInteger)),
        "XSD integer has one zero: -0 / 5 is 0");
  // modulo: Python's %, exact.
  CHECK(number("modulo", nums({p60, 7}), {}, kInteger) == 1, "modulo(2^60, 7) is 1 (double formula: 64)");
  CHECK(number("modulo", nums({p60, 7})) == 64, "the double call is unchanged");
  CHECK(number("modulo", nums({11, 3}), {}, kInteger) == 2 && number("modulo", nums({9, -7}), {}, kInteger) == -5 &&
            number("modulo", nums({-4, -9}), {}, kInteger) == -4 && number("modulo", nums({-7, 2}), {}, kInteger) == 1,
        "modulo's integer examples, and modulo(-7, 2) = 1");
  CHECK(number("modulo", nums({min64, -1}), {}, kInteger) == 0, "modulo(INT64_MIN, -1) is 0 (no C++ overflow)");
  CHECK(invalid("modulo", nums({5, 0}), {}, kInteger), "B41: modulo by 0 is invalid for integers too");
  // abs.
  CHECK(invalid("abs", nums({min64}), {}, kInteger), "§4.2: abs(INT64_MIN) is invalid");
  CHECK(number("abs", nums({-p62}), {}, kInteger) == p62 && number("abs", nums({-3}), {}, kInteger) == 3, "abs");
  // pow: exact, invalid outside int64; D8 for a negative exponent.
  CHECK(invalid("pow", nums({2, 63}), {}, kInteger), "D4: pow(2, 63) is invalid");
  CHECK(number("pow", nums({-2, 63}), {}, kInteger) == min64, "pow(-2, 63) = INT64_MIN is within int64");
  CHECK(invalid("pow", nums({3, 40}), {}, kInteger), "D4: pow(3, 40) > 2^63 is invalid");
  CHECK(invalid("pow", nums({65536, 4}), {}, kInteger) && invalid("pow", nums({4294967296.0, 2}), {}, kInteger),
        "D4: pow(2^16, 4) = pow(2^32, 2) = 2^64 is invalid (the square overflows)");
  CHECK(number("pow", nums({3, 39}), {}, kInteger) == static_cast<double>(4052555153018976267LL),
        "pow(3, 39) = 4052555153018976267, exact then rounded");
  CHECK(number("pow", nums({0, 0}), {}, kInteger) == 1, "\"if x and y are both zero then pow will return one\"");
  CHECK(number("pow", nums({-1, 1e18}), {}, kInteger) == 1 && number("pow", nums({0, 5}), {}, kInteger) == 0 &&
            number("pow", nums({1, p62}), {}, kInteger) == 1,
        "pow with a huge exponent and a base of -1, 0, 1");
  CHECK(invalid("pow", nums({7, -2}), {}, kInteger), "D8: a negative exponent is invalid");
  // The aggregates: the exact aggregate, invalid only when it is outside int64.
  CHECK(number("sum", nums({p53, 1, 1}), {}, kInteger) == p53 + 2, "sum(2^53, 1, 1) is 2^53 + 2 (double: 2^53)");
  CHECK(number("sum", nums({p53, 1, 1})) == p53, "the double call is unchanged");
  CHECK(invalid("sum", nums({p62, p62}), {}, kInteger), "D4: sum(2^62, 2^62) is invalid");
  CHECK(number("sum", nums({p62, p62, -p62}), {}, kInteger) == p62,
        "sum(2^62, 2^62, -2^62) is 2^62: an intermediate outside int64 is not the result");
  CHECK(invalid("product", nums({p32, p32, 1}), {}, kInteger), "D4: product(2^32, 2^32, 1) is invalid");
  CHECK(number("product", nums({p32, p32, 0}), {}, kInteger) == 0, "product(2^32, 2^32, 0) is 0");
  CHECK(number("product", nums({p32, -p32 / 2, 1}), {}, kInteger) == min64, "product = INT64_MIN is within int64");
  CHECK(number("product", nums({min64, -1, -1}), {}, kInteger) == min64,
        "product(INT64_MIN, -1, -1) is INT64_MIN: the partial 2^63 is not the result");
  CHECK(invalid("product", nums({min64, -1}), {}, kInteger), "product(INT64_MIN, -1) = 2^63 is invalid");
  {
    std::vector<Value> some = nums({p53, 1});
    some.emplace_back();
    some.push_back(Value(1.0));
    CHECK(number("sum", some, {}, kInteger) == p53 + 2, "integer sum skips the missing value");
    Value out;
    const std::vector<Value> none(3);
    CHECK(call("sum", none, out, {}, kInteger) == fn::Status::Ok && out.is_missing(), "all missing: missing");
  }
  CHECK(number("max", nums({-3, 2, -0.0}), {}, kInteger) == 2 && number("min", nums({-3, 2}), {}, kInteger) == -3,
        "max, min");
  // An operand outside int64 (the double 2^63, which is where an int64 near
  // 2^63 is stored) makes an integer call invalid.
  CHECK(invalid("+", nums({p63, -1}), {}, kInteger) && invalid("sum", nums({p63, -1}), {}, kInteger) &&
            invalid("abs", nums({p63}), {}, kInteger) && invalid("max", nums({p63, 1}), {}, kInteger),
        "an operand outside int64 is invalid");
  // floor, ceil, round: the integer, invalid outside int64 (§4.2: "out of
  // int64 range").
  for (const char* f : {"floor", "ceil", "round"}) {
    CHECK(invalid(f, nums({1e19})) && invalid(f, nums({-1e19})) && invalid(f, nums({p63})),
          std::string(f) + " outside int64 is invalid (D4)");
    CHECK(number(f, nums({min64})) == min64 && number(f, nums({9.2e18})) == 9.2e18,
          std::string(f) + " within int64: the integer");
    CHECK(!std::signbit(number(f, nums({-0.0}))), std::string(f) + "(-0) is the integer 0");
  }
  CHECK(number("floor", nums({-0.5})) == -1 && !std::signbit(number("ceil", nums({-0.5}))) &&
            !std::signbit(number("round", nums({-0.4}))),
        "ceil(-0.5) and round(-0.4) are the integer 0, not -0.0");
  // The rows that have no int64 kernel are the double ones.
  CHECK(number("threshold", nums({3, 2}), {}, kInteger) == 1, "threshold: 0 or 1 either way");
  CHECK(fn::instantiate(fn::Op::Avg, true) == fn::instantiate(fn::Op::Avg, false) &&
            fn::instantiate(fn::Op::Floor, true) == fn::instantiate(fn::Op::Floor, false),
        "avg and floor have one kernel");
  for (fn::Op op : {fn::Op::Plus, fn::Op::Minus, fn::Op::Times, fn::Op::Divide, fn::Op::Pow, fn::Op::Modulo,
                    fn::Op::Abs, fn::Op::Max, fn::Op::Min, fn::Op::Sum, fn::Op::Product})
    CHECK(fn::instantiate(op, true) != fn::instantiate(op, false), "an int64 kernel for each LR arithmetic row");
}

// The same through a whole document: the binder's result type picks the
// kernel; an overflow is an invalid result under the Apply's treatment.
static void integer_in_document() {
  auto app = [](const char* f, const std::string& attrs) {
    return std::string(R"(<Apply function=")") + f + "\" " + attrs +
           R"(><FieldRef field="i"/><FieldRef field="e"/></Apply>)";
  };
  const std::string big = "4611686018427387904";  // 2^62
  CHECK(apply_output(app("+", ""), big, big) ==
            R"(ERROR: Apply "+" in DerivedField "p": invalid result (invalidValueTreatment="returnInvalid"))",
        "D4: integer 2^62 + 2^62 makes the record invalid");
  CHECK(apply_output(app("*", R"(invalidValueTreatment="asMissing" defaultValue="7")"), "4294967296", "4294967296") ==
            "7",
        "D4: integer 2^32 * 2^32 under asMissing: defaultValue");
  CHECK(apply_output(app("sum", R"(invalidValueTreatment="asMissing")"), big, big) == "N/A",
        "D4: integer sum(2^62, 2^62) under asMissing: missing");
  CHECK(apply_output(app("modulo", ""), "1152921504606846976", "7") == "1", "modulo(2^60, 7) = 1 exactly");
  CHECK(apply_output(app("+", ""), "2", "3") == "5", "2 + 3 = 5");
  const std::string mixed =
      R"(<Apply function="+"><FieldRef field="i"/><Constant dataType="double">1</Constant></Apply>)";
  CHECK(apply_output(mixed, big, "") == "4.61169e+18", "integer + double is a double: 2^62 + 1.0 not checked");
  const std::string twice = R"(<Apply function="+"><FieldRef field="i"/><FieldRef field="i"/></Apply>)";
  CHECK(apply_output(twice, big, "", "abc", "b", "double", true) ==
            R"(ERROR: Apply "+" in OutputField "o": invalid result (invalidValueTreatment="returnInvalid"))",
        "D4: in an OutputField too");
  CHECK(apply_output(R"(<Apply function="floor" invalidValueTreatment="asMissing" defaultValue="-1"><Apply function="*"><FieldRef field="i"/><Constant dataType="double">10</Constant></Apply></Apply>)",
                     "1000000000000000000", "") == "-1",
        "D4: floor(1e19) is invalid: asMissing gives defaultValue");
}

static void domains() {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // "/" (B21, D20): "If a positive number is divided by zero, INF will be
  // returned. If zero is divided by zero, the result is indeterminate and a
  // missing value will be returned. If a negative number is divided by zero,
  // -INF will be returned."
  CHECK(number("/", nums({1, 0})) == inf && number("/", nums({-1, 0})) == -inf,
        "\"If a positive number is divided by zero, INF\"; a negative one, -INF");
  CHECK(missing_result("/", nums({0, 0})), "B21: \"If zero is divided by zero ... a missing value will be returned\"");
  CHECK(missing_result("/", nums({-0.0, 0})) && missing_result("/", nums({0, -0.0})), "0/0 of either sign is missing");
  CHECK(number("/", nums({1, -0.0})) == inf && number("/", nums({-1, -0.0})) == -inf,
        "a -0 divisor: INF signed by the numerator (\"a positive number ... INF\"; IEEE would give -INF)");
  CHECK(number("/", nums({inf, 0})) == inf, "INF / 0 is INF (a positive number divided by zero)");
  CHECK(std::isnan(number("/", nums({nan, 0}))), "NaN / 0 is NaN (D23: \"All arithmetic expressions involving NaN\")");
  CHECK(invalid("/", nums({1, 0}), {}, kInteger) && invalid("/", nums({-1, 0}), {}, kInteger),
        "D20: integer x/0 is invalid (\"INF and -INF can only be floating point numbers\")");
  CHECK(missing_result("/", nums({0, 0}), {}, kInteger), "§4.2: integer 0/0 is missing (the same sentence)");
  CHECK(number("/", nums({-7, 2}), {}, kInteger) == -3 && number("/", nums({1, 4})) == 0.25,
        "a non-zero divisor: as before");

  // floor, ceil, round (B21): "Since INF and -INF can only be floating point
  // numbers, they will cause floor, ceil, round, and modulo to return an
  // invalid value"; NaN is no integer either (§4.2).
  for (const char* f : {"floor", "ceil", "round"}) {
    CHECK(invalid(f, nums({inf})) && invalid(f, nums({-inf})), std::string(f) + " of +-INF is invalid (B21)");
    CHECK(invalid(f, nums({nan})), std::string(f) + " of NaN is invalid (§4.2: \"return an integer\")");
  }
  CHECK(number("floor", nums({-2.5})) == -3 && number("ceil", nums({-2.5})) == -2 && number("round", nums({2.5})) == 3,
        "finite: as before");
  // round sends a half up (B54, decided 2026-11-20 after P4.9: as XPath
  // fn:round, "the one that is closest to positive infinity", and Java
  // Math.round), exactly: a fraction just below one half is not a half, and
  // from 2^52 on every double is an integer and is returned as it is.
  {
    const double below_half = std::nextafter(0.5, 0.0);  // 0.49999999999999994
    const double p52 = 4503599627370496.0;
    CHECK(number("round", nums({0.5})) == 1 && number("round", nums({1.5})) == 2 &&
              number("round", nums({2.5})) == 3,
          "round of a positive half goes up");
    CHECK(number("round", nums({-0.5})) == 0 && !std::signbit(number("round", nums({-0.5}))) &&
              number("round", nums({-1.5})) == -1 && number("round", nums({-2.5})) == -2,
          "round of a negative half goes up, towards +INF (B54); round(-0.5) is the integer 0, not -0.0");
    CHECK(number("round", nums({below_half})) == 0 && number("round", nums({-below_half})) == 0 &&
              number("round", nums({std::nextafter(-0.5, -1.0)})) == -1 &&
              number("round", nums({std::nextafter(1.5, 0.0)})) == 1 &&
              number("round", nums({std::nextafter(-1.5, -2.0)})) == -2,
          "round is exact next to a half (no floor(x + 0.5))");
    CHECK(number("round", nums({p52 + 1})) == p52 + 1 && number("round", nums({-(p52 + 1)})) == -(p52 + 1) &&
              number("round", nums({p52 - 0.5})) == p52 && number("round", nums({-(p52 - 0.5)})) == -(p52 - 1),
          "round from 2^52 on returns x; just below 2^52 a half goes up");
    CHECK(number("round", nums({3.7})) == 4 && number("round", nums({-3.7})) == -4 &&
              number("round", nums({-3.2})) == -3,
          "round off a half: the closest integer");
  }

  // modulo: an infinite operand (B21) and a zero divisor (B41) are invalid;
  // NaN gives NaN (D23).
  CHECK(invalid("modulo", nums({inf, 2})) && invalid("modulo", nums({-inf, 2})) && invalid("modulo", nums({5, inf})) &&
            invalid("modulo", nums({5, -inf})),
        "B21: an infinite operand of modulo is invalid");
  CHECK(invalid("modulo", nums({5, 0})) && invalid("modulo", nums({0, 0})) && invalid("modulo", nums({-5, -0.0})),
        "B41: modulo by 0 is invalid (Python's %: ZeroDivisionError; XQuery err:FOAR0001 for integers)");
  CHECK(std::isnan(number("modulo", nums({nan, 0}))) && std::isnan(number("modulo", nums({2, nan}))),
        "a NaN operand of modulo gives NaN (D23)");
  CHECK(number("modulo", nums({11, 3})) == 2 && number("modulo", nums({9, -7})) == -5 &&
            number("modulo", nums({-4, -9})) == -4 && near(number("modulo", nums({-17.2, 0.5})), 0.3),
        "modulo's four examples, as before");

  // ln, log10, sqrt, exp: values, not invalid (B41 settled from the spec):
  // "The logarithm of a negative number is undefined, so NaN should be
  // returned in such a case. The logarithm of INF is to be taken as INF. The
  // logarithm of 0 is to be taken as -INF. The square root of a INF is to be
  // taken as INF. The square root of a negative number is to be taken as NaN
  // ... The exponential of INF is to be taken as INF. The exponential of -INF
  // is to be taken as 0."
  CHECK(std::isnan(number("ln", nums({-1}))) && std::isnan(number("log10", nums({-1}))),
        "the logarithm of a negative number is NaN, a value");
  CHECK(number("ln", nums({0})) == -inf && number("log10", nums({0})) == -inf && number("ln", nums({-0.0})) == -inf,
        "the logarithm of 0 is -INF");
  CHECK(number("ln", nums({inf})) == inf && number("log10", nums({inf})) == inf, "the logarithm of INF is INF");
  CHECK(std::isnan(number("sqrt", nums({-1}))) && number("sqrt", nums({inf})) == inf,
        "sqrt of a negative number is NaN, of INF is INF");
  CHECK(number("exp", nums({inf})) == inf && number("exp", nums({-inf})) == 0, "exp(INF) = INF, exp(-INF) = 0");

  // substring (B41): "startPos and length must be positive integers"; within
  // that, XQuery fn:substring: "The characters returned do not extend beyond
  // $sourceString".
  const Value abc = Value::string("abc");
  CHECK(invalid("substring", {abc, Value(0.0), Value(1.0)}) && invalid("substring", {abc, Value(-1.0), Value(2.0)}),
        "B41: a startPos below 1 is invalid");
  CHECK(invalid("substring", {abc, Value(1.0), Value(0.0)}) && invalid("substring", {abc, Value(2.0), Value(-1.0)}),
        "B41: a length below 1 is invalid (\"must be positive integers\")");
  CHECK(invalid("substring", {abc, Value(inf), Value(1.0)}) && invalid("substring", {abc, Value(1.0), Value(inf)}) &&
            invalid("substring", {abc, Value(nan), Value(1.0)}),
        "an INF or NaN position is no integer: invalid");
  CHECK(text_result({abc, Value(4.0), Value(1.0)}, "") && text_result({abc, Value(5.0), Value(1.0)}, "") &&
            text_result({abc, Value(1e300), Value(1.0)}, ""),
        "a startPos past the end gives the empty string (\"do not extend beyond\"; P4.1: invalid)");
  CHECK(text_result({abc, Value(2.0), Value(5.0)}, "bc") && text_result({abc, Value(1.0), Value(1e300)}, "abc"),
        "a length past the end stops at the end");
  CHECK(text_result({abc, Value(2.0), Value(1.0)}, "b") && text_result({abc, Value(1.0), Value(3.0)}, "abc"),
        "within the string: as before");
}

// The same through a whole document: an invalid result follows the Apply's
// invalidValueTreatment, a missing one gives defaultValue, NaN is a value.
static void domains_in_document() {
  const std::string ri = R"(invalidValueTreatment="returnInvalid")";
  auto err = [](const char* f, const char* t) {
    return std::string("ERROR: Apply \"") + f + "\" in DerivedField \"p\": invalid result (invalidValueTreatment=\"" +
           t + "\")";
  };
  auto div = [](const std::string& attrs) {
    return R"(<Apply function="/" )" + attrs + R"(><FieldRef field="i"/><FieldRef field="e"/></Apply>)";
  };
  // D20: integer x/0 makes the record invalid under returnInvalid and asIs.
  CHECK(apply_output(div(""), "1", "0") == err("/", "returnInvalid"), "D20: integer 1/0, returnInvalid");
  CHECK(apply_output(div(R"(invalidValueTreatment="asIs")"), "-1", "0") == err("/", "asIs"), "D20: integer -1/0, asIs");
  CHECK(apply_output(div(R"(invalidValueTreatment="asMissing" mapMissingTo="5" defaultValue="7")"), "1", "0") == "7",
        "D20: integer 1/0 under asMissing: defaultValue");
  // B21: 0/0 is a missing result, under every treatment.
  CHECK(apply_output(div(ri), "0", "0") == "N/A", "B21: integer 0/0 is missing, not invalid");
  CHECK(apply_output(div(R"(defaultValue="7")"), "0", "0") == "7", "B21: 0/0 then defaultValue");
  CHECK(apply_output(div(R"(mapMissingTo="5")"), "0", "0") == "N/A", "B21: 0/0 is no missing input: not mapMissingTo");
  const std::string double_div =
      R"(<Apply function="/"><FieldRef field="i"/><Constant dataType="double">0</Constant></Apply>)";
  CHECK(apply_output(double_div, "0", "") == "N/A" && apply_output(double_div, "2", "") == "inf",
        "double 0/0 missing, 2/0 INF");
  // B21: floor of INF makes the record invalid; of a missing 0/0, missing.
  const std::string floor_div = R"(<Apply function="floor">)" + double_div + "</Apply>";
  CHECK(apply_output(floor_div, "2", "") == err("floor", "returnInvalid"), "B21: floor(2/0.0) is invalid");
  CHECK(apply_output(floor_div, "0", "") == "N/A", "floor(0/0.0): a missing input, missing");
  // B41: modulo by 0.
  const std::string mod = R"(<Apply function="modulo"><FieldRef field="i"/><FieldRef field="e"/></Apply>)";
  CHECK(apply_output(mod, "5", "0") == err("modulo", "returnInvalid"), "B41: modulo(5, 0) is invalid");
  CHECK(apply_output(mod, "11", "3") == "2", "modulo(11, 3) = 2");
  // ln, sqrt: NaN and -INF are values, under returnInvalid too.
  CHECK(apply_output(R"(<Apply function="ln"><FieldRef field="i"/></Apply>)", "-1", "") == "nan",
        "ln(-1) is NaN, not invalid");
  CHECK(apply_output(R"(<Apply function="log10"><FieldRef field="i"/></Apply>)", "0", "") == "-inf",
        "log10(0) is -INF");
  CHECK(apply_output(R"(<Apply function="sqrt" invalidValueTreatment="asMissing" defaultValue="7"><FieldRef field="i"/></Apply>)",
                     "-1", "") == "nan",
        "sqrt(-1) under asMissing is NaN, not defaultValue: NaN is not invalid");
  // B41: substring, a string result.
  auto sub = [](const std::string& attrs) {
    return R"(<Apply function="substring" )" + attrs +
           R"(><FieldRef field="t"/><FieldRef field="i"/><FieldRef field="e"/></Apply>)";
  };
  CHECK(apply_output(sub(""), "0", "1", "abc", "b", "string") == err("substring", "returnInvalid"),
        "B41: substring(abc, 0, 1) is invalid");
  CHECK(apply_output(sub(R"(invalidValueTreatment="asMissing" defaultValue="dv")"), "-1", "2", "abc", "b", "string") ==
            "dv",
        "B41: substring(abc, -1, 2) under asMissing: defaultValue");
  CHECK(apply_output(sub(R"(invalidValueTreatment="asMissing" defaultValue="dv")"), "5", "1", "abc", "b", "string") ==
            "N/A",
        "substring(abc, 5, 1) is the empty string, not invalid (no defaultValue)");
  CHECK(apply_output(sub(ri), "2", "5", "abc", "b", "string") == "bc", "substring(abc, 2, 5) is bc");
}

// concat through a whole document (P4.6, B23): the Apply binds each argument
// to its static type (bind::expression_type) at load, after Extension*
// (pmml-4-4.xsd Apply, B25). i and e are integer fields, t a string field.
static void concat_in_document() {
  auto cat = [](const std::string& args) { return R"(<Apply function="concat">)" + args + "</Apply>"; };
  const std::string i = R"(<FieldRef field="i"/>)", e = R"(<FieldRef field="e"/>)", t = R"(<FieldRef field="t"/>)";
  const std::string bar = R"(<Constant dataType="string">|</Constant>)";
  auto out = [](const std::string& apply, const std::string& iv, const std::string& ev) {
    return apply_output(apply, iv, ev, "abc", "b", "string");
  };
  CHECK(out(cat(i + bar + t), "7", "2") == "7|abc", "concat(integer 7, |, abc) is 7|abc (B23: base gave |abc)");
  CHECK(out(cat("<Extension/>" + i + bar + t), "-3", "2") == "-3|abc", "an Extension is no argument");
  CHECK(out(cat(R"(<Apply function="/">)" + i + e + "</Apply>" + bar), "7", "2") == "3|",
        "integer 7/2 is the integer 3");
  CHECK(out(cat(R"(<Apply function="avg">)" + i + e + "</Apply>" + bar), "7", "2") == "4.5|", "avg is a double");
  CHECK(out(cat(R"(<Apply function="*">)" + i + R"(<Constant dataType="double">1</Constant></Apply>)" + bar), "7",
            "2") == "7.0|",
        "integer * double is a double: 7.0, not 7 (D17: XQuery writes 7)");
  CHECK(out(cat(R"(<Apply function="greaterThan">)" + i + e + "</Apply>" + bar), "7", "2") == "true|",
        "a comparison is a boolean");
  CHECK(out(cat(R"(<Constant>2.50</Constant><Constant>1e3</Constant><Constant>5</Constant>)"), "7", "2") ==
            "2.51000.05",
        "untyped Constants by their content: double 2.50, double 1e3, integer 5");
  CHECK(out(cat(R"(<FieldRef field="i" mapMissingTo="5"/>)" + bar), "", "2") == "5|",
        "a mapMissingTo inside concat keeps its document text (decided 2026-11-18)");
  CHECK(out(cat(i + bar), "", "2") == "N/A", "a missing argument: a missing result");
}

// formatNumber (design item P4.9). BuiltinFunctions.html: "Formats numbers
// according to a pattern. The pattern uses the Posix descriptors as used,
// e.g., in the C function printf". The oracle is the C library's snprintf
// (POSIX printf), called here with the C argument each conversion takes;
// cPMML formats with std::to_chars and its own flags, width and padding, so
// the two are independent. An integer conversion is given the value as a
// long long ("ll" inserted before the specifier), only when the value is an
// integer the conversion takes.
static std::string printf_text(const std::string& pattern, double v) {
  char buffer[2048];
  const std::size_t at = pattern.find_last_of("diouxXfFeEgGaA");
  const char s = pattern[at];
  if (std::string("diouxX").find(s) != std::string::npos) {
    const std::string format = pattern.substr(0, at) + "ll" + pattern.substr(at);
    std::snprintf(buffer, sizeof(buffer), format.c_str(), static_cast<long long>(v));
  } else {
    std::snprintf(buffer, sizeof(buffer), pattern.c_str(), v);
  }
  return buffer;
}
static std::optional<std::string> format_one(const std::string& pattern, double v) {
  fn::scalar::NumberPattern parsed;
  const std::string reason = fn::scalar::parse_number_pattern(pattern, parsed);
  if (!reason.empty()) throw std::logic_error("pattern " + pattern + ": " + reason);
  std::string out;
  if (!fn::scalar::format_number(parsed, v, out)) return std::nullopt;
  return out;
}
static std::string refusal(const std::string& pattern) {
  fn::scalar::NumberPattern parsed;
  return fn::scalar::parse_number_pattern(pattern, parsed);
}

static void format_number_kernel(Indexer& indexer) {
  const fn::Signature* sig = fn::find("formatNumber");
  CHECK(sig && sig->min_args == 2 && sig->max_args == 2 && sig->param(0) == fn::Param::Numeric &&
            sig->param(1) == fn::Param::String && sig->result == fn::Result::Fixed &&
            sig->fixed == t::TypeId::String && sig->missing == fn::MissingRule::Propagate &&
            sig->engine == fn::Engine::Today && sig->op == fn::Op::FormatNumber,
        "formatNumber: input, pattern dataType=string; result dataType=string; missing propagates");
  // The spec's example: "%3d" of 2 is "  2".
  CHECK(format_one("%3d", 2) == std::optional<std::string>("  2"), "BuiltinFunctions.html: %3d of 2 is \"  2\"");

  // Every flag, width and precision of the integer conversions, against
  // printf.
  const double inf = std::numeric_limits<double>::infinity();
  int compared = 0, agreed = 0;
  std::string first_disagreement;
  const auto against_printf = [&](const std::string& pattern, double v) {
    compared++;
    const auto mine = format_one(pattern, v);
    const std::string c = printf_text(pattern, v);
    if (mine && *mine == c) agreed++;
    else if (first_disagreement.empty())
      first_disagreement = pattern + " of " + std::to_string(v) + ": " + (mine ? *mine : "<invalid>") + " vs " + c;
  };
  for (const char* conversion : {"d", "i", "o", "u", "x", "X"})
    for (const char* flags : {"", "-", "+", " ", "0", "#", "-0", "+0", " 0", "#0", "-#", "+ "})
      for (const char* width : {"", "1", "6"})
        for (const char* precision : {"", ".", ".0", ".1", ".4"}) {
          const std::string c(conversion), f(flags);
          if (f.find('#') != std::string::npos && (c == "d" || c == "i" || c == "u")) continue;
          for (double v : {0.0, 1.0, 7.0, 42.0, 255.0, 1000000.0, 9007199254740993.0, 4611686018427387904.0})
            against_printf("[%" + f + width + precision + c + "]", v);
          if (c == "d" || c == "i")
            for (double v : {-1.0, -42.0, -1000000.0, -9223372036854775808.0})
              against_printf("[%" + f + width + precision + c + "]", v);
        }
  // The floating conversions: flags, width, precision; values whose digits
  // round (0.125, 2.5, 1.005, 0.0005), large and small, negative zero,
  // infinities and NaN (positive: C writes "-nan" for a NaN with its sign bit
  // set, PMML's NaN has no sign). NaN is compared only under flags with no +
  // and no <space>: POSIX leaves the sign of a NaN to the implementation
  // (glibc writes "+nan" and " nan", macOS "nan"); cPMML's rule, no sign, is
  // pinned by the "[%+ 5f]" check below, not by the host's printf.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const char* conversion : {"f", "F", "e", "E", "g", "G", "a", "A"})
    for (const char* flags : {"", "-", "+", " ", "0", "#", "-0", "+0", "#0", "-#", "+ ", " +"})
      for (const char* width : {"", "4", "14"})
        for (const char* precision : {"", ".", ".0", ".1", ".3", ".17"}) {
          const std::string f(flags);
          const bool signed_flag = f.find_first_of("+ ") != std::string::npos;
          for (double v : {0.0, -0.0, 0.5, 2.5, 0.125, 1.005, 0.0005, 1e-5, 1.0 / 3, 123456789.0, 1e10, 1e16, 1e22,
                           1.7976931348623157e308, 2.2250738585072014e-308, -1234.5678, 9.999999, 99.95, inf, -inf,
                           nan}) {
            if (signed_flag && std::isnan(v)) continue;
            against_printf("[%" + f + width + precision + conversion + "]", v);
          }
        }
  CHECK(agreed == compared, "formatNumber agrees with printf on " + std::to_string(compared) +
                                " pattern and value pairs; first disagreement: " + first_disagreement);

  // The text around the conversion, %%, no conversion, %1$.
  CHECK(format_one("rate %.1f%%", 0.25) == std::optional<std::string>("rate 0.2%"), "text and %% (0.25: ties to even)");
  CHECK(format_one("no conversion", 7) == std::optional<std::string>("no conversion"),
        "no conversion: the text alone (\"the excess arguments shall be evaluated but are otherwise ignored\")");
  CHECK(format_one("%%d", 7) == std::optional<std::string>("%d"), "%% is a percent sign, then text");
  CHECK(format_one("%1$.0f|%1$e|%1$x", 10) == std::optional<std::string>("10|1.000000e+01|a"),
        "%1$ conversions all convert the one input");
  CHECK(format_one("", 7) == std::optional<std::string>(""), "the empty pattern is the empty string");
  // Rounding from the binary value, ties to even (printf: "rounded in an
  // implementation-defined manner"; IEEE 754's default; JPMML rounds the
  // decimal half up).
  CHECK(format_one("%.2f", 0.125) == std::optional<std::string>("0.12") &&
            format_one("%.2f", 0.375) == std::optional<std::string>("0.38") &&
            format_one("%.0f", 0.5) == std::optional<std::string>("0") &&
            format_one("%.0f", 1.5) == std::optional<std::string>("2") &&
            format_one("%.2f", 1.005) == std::optional<std::string>("1.00"),
        "ties to even on the exact binary value: 0.12, 0.38, 0, 2; 1.005 is 1.00499999999999989...");
  CHECK(format_one("%g", 100000) == std::optional<std::string>("100000") &&
            format_one("%g", 1000000) == std::optional<std::string>("1e+06") &&
            format_one("%g", 0.0001) == std::optional<std::string>("0.0001") &&
            format_one("%g", 0.00001) == std::optional<std::string>("1e-05") &&
            format_one("%#.3g", 2) == std::optional<std::string>("2.00") &&
            format_one("%.0g", 25) == std::optional<std::string>("2e+01"),
        "%g: style f when P > X >= -4, else e; trailing zeros dropped unless #; 25 to one digit is 2e+01 (a tie)");
  CHECK(format_one("%a", 1) == std::optional<std::string>("0x1p+0") &&
            format_one("%A", 0.5) == std::optional<std::string>("0X1P-1") &&
            format_one("%a", 4.9406564584124654e-324) == std::optional<std::string>("0x0.0000000000001p-1022"),
        "%a: the subnormal's leading digit is 0 (\"otherwise unspecified\"; std::to_chars)");
  CHECK(format_one("[%5f]", inf) == std::optional<std::string>("[  inf]") &&
            format_one("[%05F]", -inf) == std::optional<std::string>("[ -INF]") &&
            format_one("%E", std::numeric_limits<double>::quiet_NaN()) == std::optional<std::string>("NAN") &&
            format_one("%f", -std::numeric_limits<double>::quiet_NaN()) == std::optional<std::string>("nan") &&
            format_one("[%+ 5f]", std::numeric_limits<double>::quiet_NaN()) == std::optional<std::string>("[  nan]") &&
            format_one("[%+f]", inf) == std::optional<std::string>("[+inf]"),
        "inf, INF, nan with no sign (+ and <space> included), blank-padded (\"except when converting an infinity or NaN\")");
  // printf: "If the <space> and '+' flags both appear, the <space> flag shall
  // be ignored", in either order, for every floating conversion.
  CHECK(format_one("[%+ .1f]", 1) == std::optional<std::string>("[+1.0]") &&
            format_one("[% +.1f]", 1) == std::optional<std::string>("[+1.0]") &&
            format_one("[% +.1e]", 1) == std::optional<std::string>("[+1.0e+00]") &&
            format_one("[%+ g]", 1) == std::optional<std::string>("[+1]") &&
            format_one("[% +a]", 1) == std::optional<std::string>("[+0x1p+0]") &&
            format_one("[% +f]", inf) == std::optional<std::string>("[+inf]") &&
            format_one("[% .1f]", 1) == std::optional<std::string>("[ 1.0]"),
        "+ wins over <space> (\"the <space> flag shall be ignored\"); <space> alone gives a blank");
  CHECK(format_one("[%.2f]", -0.0) == std::optional<std::string>("[-0.00]") &&
            format_one("[%d]", -0.0) == std::optional<std::string>("[0]"),
        "-0.0 keeps its sign in f (XSD double has -0, D18); the integer has one zero");
  CHECK(format_one("%.0d", 0) == std::optional<std::string>("") && format_one("%#.0o", 0) == std::optional<std::string>("0") &&
            format_one("%#x", 0) == std::optional<std::string>("0"),
        "zero with precision 0 is no characters; #o gives one 0; # prefixes 0x to a non-zero result only");
  CHECK(format_one("%300.250f", 1e300)->size() == 552, "a wide, precise fixed conversion (buffer)");

  // A value an integer conversion cannot take: invalid.
  CHECK(!format_one("%d", 2.5) && !format_one("%d", inf) && !format_one("%d", -inf) &&
            !format_one("%d", std::numeric_limits<double>::quiet_NaN()) && !format_one("%d", 9223372036854775808.0) &&
            !format_one("%d", 1e300),
        "%d of a fraction, an infinity, NaN, or a value past int64 (D4) has no integer to convert: invalid");
  CHECK(!format_one("%x", -3) && !format_one("%o", -1) && !format_one("%u", -42) && !format_one("%X", -0.5),
        "o u x X take an unsigned argument: a negative value is invalid (JPMML: 32-bit two's complement)");
  CHECK(format_one("%d", -9223372036854775808.0) == std::optional<std::string>("-9223372036854775808") &&
            format_one("%x", 9223372036854775807.0 - 1024) == std::optional<std::string>("7ffffffffffffc00"),
        "the int64 range, both ends");
  CHECK(!format_one("%d|", 2.5) && !format_one("%1$f|%1$d", 2.5), "invalid when any conversion has no integer");

  // Patterns formatNumber cannot apply, each with its reason.
  struct Refused {
    const char* pattern;
    const char* reason;
  };
  for (const Refused& r : {
           Refused{"%", "the conversion at the end is incomplete"},
           Refused{"abc %5.", "the conversion at the end is incomplete"},
           Refused{"%s", "'%s' is not a numeric conversion (d i o u x X f F e E g G a A)"},
           Refused{"%5s", "'%s' is not a numeric conversion (d i o u x X f F e E g G a A)"},
           Refused{"%c", "'%c' is not a numeric conversion (d i o u x X f F e E g G a A)"},
           Refused{"%n", "'%n' is not a numeric conversion (d i o u x X f F e E g G a A)"},
           Refused{"%p", "'%p' is not a numeric conversion (d i o u x X f F e E g G a A)"},
           Refused{"%k", "'%k' is not a numeric conversion (d i o u x X f F e E g G a A)"},
           Refused{"%5%", "'%%' is not a numeric conversion (d i o u x X f F e E g G a A)"},
           Refused{"%ld", "the length modifier 'l' is not supported: the input is a PMML number, not a C type"},
           Refused{"%hhd", "the length modifier 'h' is not supported: the input is a PMML number, not a C type"},
           Refused{"%Lf", "the length modifier 'L' is not supported: the input is a PMML number, not a C type"},
           Refused{"%jd", "the length modifier 'j' is not supported: the input is a PMML number, not a C type"},
           Refused{"%*d", "a width given by an argument (*) is not supported: formatNumber has one input"},
           Refused{"%.*f", "a precision given by an argument (.*) is not supported: formatNumber has one input"},
           Refused{"%'d", "the thousands' grouping flag (') is not supported: its characters are the locale's"},
           Refused{"%#d", "the flag '#' is undefined for '%d'"},
           Refused{"%#i", "the flag '#' is undefined for '%i'"},
           Refused{"%#u", "the flag '#' is undefined for '%u'"},
           Refused{"%d %d", "it has 2 conversions: formatNumber has one input"},
           Refused{"%f%e%g", "it has 3 conversions: formatNumber has one input"},
           Refused{"%2$d", "argument \"%2$\" does not exist: formatNumber has one input"},
           Refused{"%01$d", "argument \"%01$\" does not exist: formatNumber has one input"},
           Refused{"%1$d %d", "numbered (%1$) and unnumbered conversions are mixed"},
           Refused{"%2147483648d", "the field width is beyond C's int range"},
           Refused{"%.2147483648f", "the precision is beyond C's int range"},
       })
    CHECK(refusal(r.pattern) == r.reason,
          std::string("pattern \"") + r.pattern + "\": " + r.reason + " (got \"" + refusal(r.pattern) + "\")");
  CHECK(refusal("%2147483647d").empty() && refusal("%-+ #0e").empty() && refusal("%1$d|%1$x").empty() &&
            refusal("%%%%").empty(),
        "C's int width, every flag of e, numbered conversions, two %%");

  // The kernel over the engine's storage: the input by its static type (a
  // float narrowed), the pattern parsed per record when no Constant was.
  const auto kernel = [&](t::TypeId input, double v, const std::string& pattern, Value& out) {
    fn::Bound bound;
    bound.argument_type = {input, t::TypeId::String};
    return call("formatNumber", {Value(v), Value::string(pattern)}, out, bound);
  };
  {
    Value out;
    CHECK(kernel(t::TypeId::Float, 0.1, "%.10f", out) == fn::Status::Ok && out.text() == "0.1000000015",
          "a float input is narrowed, then promoted to double (the verifier's fix: not 0.1000000000)");
    CHECK(kernel(t::TypeId::Double, 0.1, "%.10f", out) == fn::Status::Ok && out.text() == "0.1000000000",
          "a double input is the double");
    CHECK(kernel(t::TypeId::Float, 16777217, "%d", out) == fn::Status::Ok && out.text() == "16777216",
          "%d of the float 16777217 is 16777216");
    CHECK(kernel(t::TypeId::Integer, 2, "%.2f", out) == fn::Status::Ok && out.text() == "2.00",
          "%.2f of an integer converts its value");
    CHECK(kernel(t::TypeId::Double, 2.5, "%s", out) == fn::Status::Invalid, "a field pattern that cannot be applied is invalid");
    CHECK(kernel(t::TypeId::Double, 2.5, "%d", out) == fn::Status::Invalid, "a field pattern: %d of 2.5 is invalid");
    CHECK(kernel(t::TypeId::Double, 2.5, "%.1f", out) == fn::Status::Ok && out.text() == "2.5" &&
              out.as_double() == Value::string("2.5").as_double(),
          "the result is packed with the pool's code");
    fn::Bound pattern_bound;
    pattern_bound.argument_type = {t::TypeId::Double, t::TypeId::Integer};
    CHECK(call("formatNumber", {Value(2.5), Value(5.0)}, out, pattern_bound) == fn::Status::Ok && out.text() == "5",
          "a number pattern is cast to string as concat casts (Java text of the integer 5): text alone");
    fn::Bound parsed;
    parsed.argument_type = {t::TypeId::Double, t::TypeId::String};
    auto p = std::make_shared<fn::scalar::NumberPattern>();
    fn::scalar::parse_number_pattern("<%+.1e>", *p);
    parsed.number_pattern = p;
    CHECK(call("formatNumber", {Value(2.5), Value::string("ignored %s")}, out, parsed) ==
                  fn::Status::Ok &&
              out.text() == "<+2.5e+00>",
          "a pattern parsed at load is the one applied");
  }
  CHECK(message_of<std::logic_error>([] {
          Value out;
          call("formatNumber", {Value(1.0), Value::string("%d")}, out);
        }) == "formatNumber: the call was not bound to its argument types",
        "an unbound formatNumber throws");
}

static std::string load_error(const std::string& apply, const std::string& type = "string") {
  const std::string path = "fn_table_load.pmml";
  {
    std::ofstream out(path);
    out << apply_document(apply, type, false);
  }
  std::string error = "<loads>";
  try {
    cpmml::Model model(path, false);
  } catch (const cpmml::ParsingException& e) {
    error = e.what();
  }
  std::remove(path.c_str());
  return error;
}

// formatNumber through a whole document (design item P4.9): i and e are
// integer fields, t and r string fields.
static void format_number_in_document() {
  auto fmt = [](const std::string& input, const std::string& pattern, const std::string& attrs = "") {
    return R"(<Apply function="formatNumber")" + attrs + ">" + input + pattern + "</Apply>";
  };
  auto pat = [](const std::string& p) { return R"(<Constant dataType="string">)" + p + "</Constant>"; };
  const std::string i = R"(<FieldRef field="i"/>)", t = R"(<FieldRef field="t"/>)", r = R"(<FieldRef field="r"/>)";
  auto out = [](const std::string& apply, const std::string& iv, const std::string& tv = "abc",
                const std::string& rv = "b") { return apply_output(apply, iv, "2", tv, rv, "string"); };
  CHECK(out(fmt(i, R"(<Constant>%3d</Constant>)"), "2") == "  2",
        "the spec's example: an untyped Constant pattern (a string, S2), %3d of 2");
  CHECK(out(fmt("<Extension/>" + i, "<Extension/>" + pat("[%05.1f]")), "-3") == "[-03.0]", "Extensions are no arguments");
  CHECK(out(fmt(R"(<Constant dataType="float">0.1</Constant>)", pat("%.10f")), "2") == "0.1000000015",
        "a float Constant is narrowed (its static type is float)");
  CHECK(out(fmt(R"(<Apply function="/">)" + i + R"(<FieldRef field="e"/></Apply>)", pat("%.2f")), "7") == "3.00",
        "integer 7/2 is the integer 3");
  CHECK(out(fmt(R"(<Apply function="avg">)" + i + R"(<FieldRef field="e"/></Apply>)", pat("%.2f")), "7") == "4.50",
        "avg is a double");
  CHECK(out(fmt(i, pat("%d")), "") == "N/A", "a missing input: a missing result");
  CHECK(out(fmt(i, pat("%d"), R"( mapMissingTo="none")"), "") == "none", "a missing input: mapMissingTo");
  CHECK(out(fmt(i, R"(<Constant missing="true"/>)", R"( defaultValue="dv")"), "3") == "dv",
        "a missing Constant pattern: a missing result, then defaultValue");
  CHECK(out(fmt(i, t), "255", "%#x") == "0xff", "a pattern read from a field, per record");
  CHECK(out(fmt(i, t, R"( invalidValueTreatment="asMissing" defaultValue="dv")"), "255", "%s") == "dv",
        "a field pattern that cannot be applied is invalid: asMissing, then defaultValue");
  CHECK(out(fmt(i, t), "255", "%s") ==
            R"(ERROR: Apply "formatNumber" in DerivedField "p": invalid result (invalidValueTreatment="returnInvalid"))",
        "a field pattern that cannot be applied is invalid: returnInvalid ends the record");
  CHECK(out(fmt(i, t, R"( invalidValueTreatment="asIs")"), "-3", "%x") ==
            R"(ERROR: Apply "formatNumber" in DerivedField "p": invalid result (invalidValueTreatment="asIs"))",
        "%x of -3 is invalid: asIs is returnInvalid in an Apply");
  CHECK(out(fmt(i, t), "3", "") == "N/A", "an empty pattern field is a missing input: a missing result");

  // Load errors: a Constant pattern formatNumber cannot apply; an input that
  // is no number; a boolean pattern.
  const std::string where = R"(Apply "formatNumber" in RegressionModel)";
  CHECK(load_error(fmt(i, pat("[%5s]"))) ==
            "E-VALUE-MALFORMED: " + where +
                R"(: pattern "[%5s]" cannot be applied: '%s' is not a numeric conversion (d i o u x X f F e E g G a A) )"
                R"((BuiltinFunctions.html: "The pattern uses the Posix descriptors as used, e.g., in the C function printf"))",
        "a Constant pattern that cannot be applied is a load error");
  CHECK(load_error(fmt(i, R"(<Constant>%d %d</Constant>)")).find(
            "cannot be applied: it has 2 conversions: formatNumber has one input") != std::string::npos,
        "an untyped Constant pattern is checked at load too");
  CHECK(load_error(fmt(i, R"(<Constant dataType="integer">5</Constant>)")) == "<loads>",
        "an integer Constant pattern is the text 5 (no conversion)");
  CHECK(load_error(fmt(t, pat("%d"))) ==
            "E-TYPE-PARAM: " + where +
                R"(: argument 1 of formatNumber is string, where a number is expected (BuiltinFunctions.html: "Formats numbers according to a pattern"; §2.5))",
        "a string input is a load error");
  CHECK(load_error(fmt(R"(<Apply function="isMissing">)" + i + "</Apply>", pat("%d"))) ==
            "E-TYPE-BOOLEAN-MIXED: " + where +
                R"(: argument 1 of formatNumber is boolean, where a number is expected (Functions.html: boolean "can not be mixed with other types", Q7))",
        "a boolean input is a load error");
  CHECK(load_error(fmt(i, R"(<Apply function="isMissing">)" + i + "</Apply>")) ==
            "E-TYPE-BOOLEAN-MIXED: " + where +
                R"(: argument 2 of formatNumber is boolean, where a string is expected (Functions.html: boolean "can not be mixed with other types", Q7))",
        "a boolean pattern is a load error");
  CHECK(load_error(fmt(i, "<Extension/>" + pat("%q"))).find(R"(pattern "%q" cannot be applied)") != std::string::npos,
        "the pattern is the second argument after Extension*");
  CHECK(load_error(fmt(i, pat("%d") + pat("%d"))).rfind("E-ARITY: ", 0) == 0, "three arguments: E-ARITY first");
  (void)r;
}

// `if` (P4.3: the rule of P2.8b4's ApplyNode::eval_if, now the table's
// MissingRule::LazyIf and the kernels if_taken / if_then_else).
// BuiltinFunctions.html, if: "Implements IF-THEN-ELSE logic. The ELSE part is
// optional. If the ELSE part is absent and the boolean value is false then a
// missing value is returned." §4.2: "Only the condition and the branch taken
// are inputs (so mapMissingTo applies when either is missing)"; D3: a missing
// condition is a missing input.
static void if_lazy() {
  CHECK(fn::find("if")->missing == fn::MissingRule::LazyIf, "if: MissingRule::LazyIf");
  CHECK(fn::adapter::if_taken(Value(1.0), 3) == 1 && fn::adapter::if_taken(Value(-0.5), 3) == 1,
        "if_taken: a condition that is not 0 takes THEN");
  CHECK(fn::adapter::if_taken(Value(0.0), 3) == 2, "if_taken: a false condition takes ELSE");
  CHECK(fn::adapter::if_taken(Value(0.0), 2) == 2, "if_taken: a false condition without ELSE takes none (n)");
  CHECK(fn::adapter::if_taken(Value(1.0), 2) == 1, "if_taken: THEN without ELSE");
  {
    Value out;
    CHECK(call("if", {Value(1.0), Value::string("abc")}, out) == fn::Status::Ok &&
              out.text() == "abc",
          "if_then_else: the branch taken's value");
    Value none;
    CHECK(call("if", {Value(0.0)}, none) == fn::Status::Ok && none.is_missing(),
          "if_then_else: no branch taken is missing");
  }

  // Through a whole document: o = if(i != 0, e[, ELSE]) for each set of
  // attributes; i and e are integers, the ELSE part a Constant 3 or
  // pow(i, -1), which is invalid for i = 0 (D8) and so shows whether it was
  // evaluated.
  const std::string three = R"(<Constant dataType="integer">3</Constant>)";
  const std::string invalid_else =
      R"(<Apply function="pow"><FieldRef field="i"/><Constant dataType="integer">-1</Constant></Apply>)";
  // The condition is notEqual(i, 0): if's condition is boolean, and an
  // integer one is a load error (P4.3c, B53); it is missing when i is.
  const std::string condition =
      R"(<Apply function="notEqual"><FieldRef field="i"/><Constant dataType="integer">0</Constant></Apply>)";
  auto if_apply = [&condition](const std::string& attrs, const std::string& otherwise) {
    return R"(<Apply function="if")" + attrs + R"(>)" + condition + R"(<FieldRef field="e"/>)" + otherwise + "</Apply>";
  };
  const std::vector<std::string> extras = {"", R"( mapMissingTo="5")", R"( defaultValue="7")",
                                           R"( mapMissingTo="5" defaultValue="7")"};
  for (const std::string& extra : extras) {
    const std::string rule = "if" + extra + ": ";
    const bool mm = extra.find("mapMissingTo") != std::string::npos;
    const bool dv = extra.find("defaultValue") != std::string::npos;
    const std::string missing_input = mm ? "5" : dv ? "7" : "N/A";
    const std::string missing_result = dv ? "7" : "N/A";
    CHECK(apply_output(if_apply(extra, three), "", "2") == missing_input,
          rule + "a missing condition is a missing input (D3): mapMissingTo, else missing, then defaultValue");
    CHECK(apply_output(if_apply(extra, ""), "", "2") == missing_input,
          rule + "a missing condition without ELSE is a missing input");
    CHECK(apply_output(if_apply(extra, three), "1", "2") == "2", rule + "a true condition gives THEN");
    CHECK(apply_output(if_apply(extra, three), "1", "") == missing_input,
          rule + "a missing THEN taken is a missing input: mapMissingTo, else missing, then defaultValue");
    CHECK(apply_output(if_apply(extra, three), "0", "2") == "3", rule + "a false condition gives ELSE");
    CHECK(apply_output(if_apply(extra, three), "0", "") == "3",
          rule + "a missing THEN not taken is not an input: ELSE");
    CHECK(apply_output(if_apply(extra, ""), "0", "2") == missing_result,
          rule + "false without ELSE is a missing result, not a missing input: defaultValue, never mapMissingTo");
    CHECK(apply_output(if_apply(extra, invalid_else), "1", "2") == "2",
          rule + "the ELSE not taken is not evaluated (its invalid pow raises nothing)");
    CHECK(apply_output(if_apply(extra, invalid_else), "0", "2") ==
              R"(ERROR: Apply "pow" in DerivedField "p": invalid result (invalidValueTreatment="returnInvalid"))",
          rule + "the ELSE taken is evaluated (its invalid pow makes the record invalid)");
  }
  // The branch taken keeps its Value: a string passes through as it came in.
  CHECK(apply_output(R"(<Apply function="if" defaultValue="none">)" + condition +
                         R"(<FieldRef field="t"/><FieldRef field="r"/></Apply>)",
                     "1", "", "abc", "b", "string") == "abc",
        "if over strings: THEN");
  CHECK(apply_output(R"(<Apply function="if" defaultValue="none">)" + condition + R"(<FieldRef field="t"/></Apply>)",
                     "0", "", "abc", "b", "string") == "none",
        "if over strings: false without ELSE gives defaultValue");
}

// §2.6's table, as fn::DeclaredCast implements it over the old storage
// (P4.7). The expected values are XML Schema Part 2's value and lexical
// spaces (3.3.13 integer, 3.2.5 double, 3.2.4 float, 3.2.2.1 boolean
// "{true, false, 1, 0}", whiteSpace collapse) and Java's text (Q1, D17),
// written out here, not cPMML's.
static void declared_casts(Indexer& indexer) {
  using TypeId = cpmml::types::TypeId;
  using Op = fn::DeclaredCast::Op;
  const double inf = std::numeric_limits<double>::infinity();
  const TypeId all[] = {TypeId::Integer, TypeId::Float, TypeId::Double, TypeId::Boolean, TypeId::String};
  for (const TypeId t : all) CHECK(fn::DeclaredCast(t, t).op == Op::None, "T -> T is the identity: no work");
  CHECK(fn::DeclaredCast(TypeId::Integer, TypeId::Double).op == Op::None, "integer -> double: the double is the value");
  CHECK(fn::DeclaredCast(TypeId::Float, TypeId::Double).op == Op::None, "float -> double: exact");
  CHECK(fn::DeclaredCast(TypeId::Boolean, TypeId::Integer).op == Op::None, "boolean -> integer: 1, 0 as held");
  CHECK(fn::DeclaredCast(TypeId::Boolean, TypeId::Double).op == Op::None, "boolean -> double: 1, 0 as held");
  // P6.2: §2.6 "double -> float: IEEE round to nearest; overflow -> ±INF,
  // underflow -> 0" (XML Schema Part 2 §3.2.4: an IEEE single).
  for (const TypeId t : {TypeId::Integer, TypeId::Double, TypeId::Boolean})
    CHECK(fn::DeclaredCast(t, TypeId::Float).op == Op::NumberToFloat, "integer, double, boolean -> float");
  {
    const fn::DeclaredCast to_float(TypeId::Double, TypeId::Float);
    Value v(0.1);
    CHECK(to_float.apply(v) && std::holds_alternative<cpmml::types::Float>(v.typed()) &&
              v.as_double() == 0.100000001490116119384765625,
          "double -> float: the nearest float, a Float");
    for (const auto& [d, f] : std::initializer_list<std::pair<double, double>>{
             {1e300, inf}, {-1e300, -inf}, {1e-300, 0.0}, {16777217.0, 16777216.0}}) {
      double x = d;
      CHECK(to_float.apply_number(x) && x == f, "apply_number: " + std::to_string(d) + " rounds to the float " +
                                                    std::to_string(f));
    }
    Value s = Value::string("0.1");
    CHECK(fn::DeclaredCast(TypeId::String, TypeId::Float).apply(s) &&
              std::holds_alternative<cpmml::types::Float>(s.typed()) && s.as_double() == 0.100000001490116119384765625,
          "string -> float: the strict reading, a Float");
  }
  CHECK(fn::DeclaredCast(TypeId::Double, TypeId::Integer).op == Op::NumberToInteger, "double -> integer");
  CHECK(fn::DeclaredCast(TypeId::Float, TypeId::Integer).op == Op::NumberToInteger, "float -> integer");
  CHECK(fn::DeclaredCast(TypeId::Integer, TypeId::Boolean).op == Op::NumberToBoolean, "integer -> boolean");
  CHECK(fn::DeclaredCast(TypeId::Double, TypeId::Boolean).op == Op::NumberToBoolean, "double -> boolean");
  for (const TypeId t : {TypeId::Integer, TypeId::Float, TypeId::Double, TypeId::Boolean}) {
    CHECK(fn::DeclaredCast(t, TypeId::String).op == Op::ToString, "number, boolean -> string");
    CHECK(fn::DeclaredCast(TypeId::String, t).op == Op::StringToNumber, "string -> number, boolean");
  }
  CHECK(fn::DeclaredCast(std::nullopt, TypeId::Integer).from == TypeId::Double,
        "an unknown static type is cast as the double the engine computed");

  // §2.6 "double, float -> integer: the value if integral and in range".
  const fn::DeclaredCast to_int(TypeId::Double, TypeId::Integer);
  auto int_of = [&](double d, double& out) {
    Value v(d);
    const bool ok = to_int.apply(v);
    out = v.as_double();
    return ok;
  };
  double out = 0;
  CHECK(int_of(3, out) && out == 3, "3.0 is the integer 3");
  CHECK(int_of(-0.0, out) && out == 0 && !std::signbit(out), "-0 is the integer 0 (XSD integer has one zero)");
  CHECK(int_of(1e10, out) && out == 1e10, "1e10 is an integer (int64, D4)");
  CHECK(int_of(-9223372036854775808.0, out), "-2^63 is an int64");
  for (const double d : {3.5, -2.5, 3.0000000001, inf, -inf, std::nan(""), 9223372036854775808.0})
    CHECK(!int_of(d, out), "B17: " + std::to_string(d) + " is no integer: the cast fails (Q2)");
  {
    double d = 3.5;
    CHECK(!to_int.apply_number(d) && d == 3.5, "apply_number agrees and leaves the value");
    d = -0.0;
    CHECK(to_int.apply_number(d) && !std::signbit(d), "apply_number: -0 is 0");
  }
  // §2.6 "number -> boolean: 1 -> true, 0 -> false, else Invalid(NoCast)" (D14).
  const fn::DeclaredCast to_bool(TypeId::Integer, TypeId::Boolean);
  for (const double d : {1.0, 0.0, -0.0}) {
    double x = d;
    CHECK(to_bool.apply_number(x) && x == d && !std::signbit(x), "1, 0, -0 are true, false, false");
  }
  for (const double d : {2.0, -1.0, 0.5, inf, std::nan("")}) {
    double x = d;
    CHECK(!to_bool.apply_number(x), "B24: " + std::to_string(d) + " is no boolean: the cast fails (Q2)");
  }
  // §2.6 "number -> string: Java text"; "boolean -> string: true / false".
  auto text_of = [&](TypeId from, double d) {
    Value v(d);
    if (!fn::DeclaredCast(from, TypeId::String).apply(v)) return std::string("<fails>");
    return std::string(v.text());
  };
  CHECK(text_of(TypeId::Integer, 3) == "3", "integer 3: Long.toString");
  CHECK(text_of(TypeId::Integer, -0.0) == "0", "integer -0: 0");
  CHECK(text_of(TypeId::Integer, 1e10) == "10000000000", "integer 1e10: 10000000000");
  CHECK(text_of(TypeId::Integer, 9223372036854775808.0) == "<fails>", "an integer slot holding 2^63 has no text");
  CHECK(text_of(TypeId::Double, 2) == "2.0", "double 2: Double.toString, 2.0 (D17)");
  CHECK(text_of(TypeId::Double, 1e7) == "1.0E7", "double 1e7: 1.0E7");
  CHECK(text_of(TypeId::Double, -0.0) == "-0.0", "double -0: -0.0 (D18)");
  CHECK(text_of(TypeId::Double, inf) == "Infinity", "double INF: Infinity");
  CHECK(text_of(TypeId::Float, 0.1) == "0.1", "float 0.1: Float.toString of the float");
  CHECK(text_of(TypeId::Float, 16777217) == "1.6777216E7", "float 16777217: 1.6777216E7");
  CHECK(text_of(TypeId::Boolean, 1) == "true" && text_of(TypeId::Boolean, 0) == "false", "boolean: true, false");
  {
    Value v(2.0);
    fn::DeclaredCast(TypeId::Double, TypeId::String).apply(v);
    CHECK(v.as_double() == Value::string("2.0").as_double() && !v.has_numeric_text(),
          "the string the cast builds is packed as the engine builds one (its text's code, no mark)");
    Value constant(5.0);
    constant.set_text("5");
    CHECK(fn::DeclaredCast(TypeId::Double, TypeId::String).apply(constant) && constant.text() == "5" &&
              constant.as_double() == 5.0,
          "a number that carries the text the load gave it keeps that Value (decided 2026-11-18)");
  }
  // §2.6 "string -> number, boolean: strict XSD reading of the text".
  struct Parse {
    TypeId to;
    const char* text;
    bool ok;
    double value;
  };
  for (const Parse& c : {Parse{TypeId::Double, "2.50", true, 2.5}, Parse{TypeId::Double, "1e3", true, 1000},
                         Parse{TypeId::Double, " -1 ", true, -1}, Parse{TypeId::Double, "INF", true, inf},
                         Parse{TypeId::Double, "+3", true, 3}, Parse{TypeId::Double, "x", false, 0},
                         Parse{TypeId::Double, "true", false, 0}, Parse{TypeId::Double, "inf", false, 0},
                         Parse{TypeId::Integer, "3", true, 3}, Parse{TypeId::Integer, "+7", true, 7},
                         Parse{TypeId::Integer, "007", true, 7}, Parse{TypeId::Integer, " 3 ", true, 3},
                         Parse{TypeId::Integer, "3.0", false, 0}, Parse{TypeId::Integer, "1e1", false, 0},
                         Parse{TypeId::Integer, "99999999999999999999", false, 0},
                         Parse{TypeId::Float, "0.1", true, static_cast<double>(0.1f)},
                         Parse{TypeId::Boolean, "true", true, 1}, Parse{TypeId::Boolean, "1", true, 1},
                         Parse{TypeId::Boolean, "false", true, 0}, Parse{TypeId::Boolean, "0", true, 0},
                         Parse{TypeId::Boolean, "TRUE", false, 0}, Parse{TypeId::Boolean, "yes", false, 0}}) {
    Value v = Value::string(c.text);
    const bool ok = fn::DeclaredCast(TypeId::String, c.to).apply(v);
    const std::string rule = std::string("string \"") + c.text + "\" -> " +
                             std::string(cpmml::bind::type_name(c.to)) + (c.ok ? " is a value" : " fails (Q2)");
    CHECK(ok == c.ok && (!ok || (v.as_double() == c.value && v.text().empty())), rule);
    double d = 0;
    CHECK(fn::DeclaredCast(TypeId::String, c.to).parse_text(c.text, d) == c.ok, "parse_text: " + rule);
  }
  {
    // Decided 2026-11-20 after P4.9, item 3: "An empty string (or any
    // text-less string) cast to a number or boolean is an invalid record,
    // never its string code." XSD: "" is in no number's or boolean's lexical
    // space. At run time the empty string is a Value with text "" and a
    // string code (here -1, the code of a string not seen at load, as
    // trimBlanks(substring("  x", 1, 2)) gives it, B2).
    for (const TypeId t : {TypeId::Integer, TypeId::Float, TypeId::Double, TypeId::Boolean}) {
      Value empty(-1.0);
      CHECK(!fn::DeclaredCast(TypeId::String, t).apply(empty) && empty.as_double() == -1,
            "the empty string (code -1) is no " + std::string(cpmml::bind::type_name(t)) + " (item 3, D42)");
      Value code(0.0);
      CHECK(!fn::DeclaredCast(TypeId::String, t).apply(code), "the empty string (code 0) is no number or boolean");
      Value blank = Value::string("   ");
      CHECK(!fn::DeclaredCast(TypeId::String, t).apply(blank), "\"   \" collapses to the empty string: no value");
    }
    {
      const Value empty(-1.0);
      CHECK(fn::DeclaredCast(TypeId::String, TypeId::Integer).failure(empty) ==
                "the string is empty, and the empty string is not a value of type integer (§2.6, D42, Q2)",
            "the failure says the string is empty");
      const Value x = Value::string("x");
      CHECK(fn::DeclaredCast(TypeId::String, TypeId::Integer).failure(x) ==
                "the value \"x\" of type string is not a value of type integer (§2.6, Q2)",
            "the failure quotes a malformed text");
    }
    // A document constant the load casts to the field's number slot keeps its
    // text (bind::document_constant), which the cast reads.
    Value five(5.0);
    five.set_text("5");
    CHECK(fn::DeclaredCast(TypeId::String, TypeId::Double).apply(five) && five.as_double() == 5 &&
              five.text().empty(),
          "a load-cast constant is read by its text");
    Value half(2.5);
    half.set_text("2.5");
    CHECK(!fn::DeclaredCast(TypeId::String, TypeId::Integer).apply(half) && half.as_double() == 2.5,
          "... strictly: \"2.5\" is no integer");
    Value marked = Value::string("7");
    marked.mark_numeric_text();
    CHECK(fn::DeclaredCast(TypeId::String, TypeId::Integer).apply(marked) && marked.as_double() == 7,
          "a numeric_text constant is read by its text");
    // NormContinuous mapMissingTo="INF" in a string slot has the text of its
    // Cast<string>, Infinity (bind::number_constant), which is no XSD double:
    // a numeric reader takes the number the load read (decided 2026-11-18).
    Value infinity = Value::string("Infinity");
    infinity.mark_numeric_text();
    CHECK(fn::DeclaredCast(TypeId::String, TypeId::Double).apply(infinity) && infinity.as_double() == inf &&
              !infinity.has_numeric_text(),
          "a numeric_text constant is its number, Infinity included");
    Value plain = Value::string("Infinity");
    CHECK(!fn::DeclaredCast(TypeId::String, TypeId::Double).apply(plain), "the text Infinity is no XSD double");
  }
  // The record errors' reason (Q2).
  CHECK(to_int.failure(3.5) == "the value 3.5 of type double is not a value of type integer (§2.6, Q2)",
        "failure: a number");
  CHECK(to_bool.failure(2.0) == "the value 2 of type integer is not a value of type boolean (§2.6, Q2)",
        "failure: an integer is written as one");
  CHECK(fn::DeclaredCast(TypeId::String, TypeId::Double).failure(Value::string("x")) ==
            "the value \"x\" of type string is not a value of type double (§2.6, Q2)",
        "failure: a string");
  CHECK(message_of<std::logic_error>([] {
          double d = 1;
          fn::DeclaredCast(TypeId::Double, TypeId::String).apply_number(d);
        }) == "DeclaredCast::apply_number: the cast is not between numbers",
        "apply_number of a string cast is a programming error");
  (void)indexer;
}

// The casts through whole documents (P4.7): the record error names the
// DerivedField, OutputField or target (Q2: "InvalidValueException naming the
// field or target"; JPMML 1.7.7: TypeCheckException), and a valid value
// is the cast's.
static std::string score_text(const std::string& body, const std::string& fields,
                              const std::vector<std::pair<std::string, std::string>>& input,
                              const std::string& column) {
  const std::string path = "fn_table_cast.pmml";
  {
    std::ofstream out(path);
    out << R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/><DataDictionary>)"
        << fields << "</DataDictionary>" << body << "</PMML>\n";
  }
  cpmml::Model model(path, false);
  std::remove(path.c_str());
  cpmml::Input in;
  for (const auto& [k, v] : input) in[k] = v;
  try {
    const cpmml::Result result = model.evaluate(in);
    if (column.empty()) return result.as_string();
    const auto str = result.str_outputs();
    if (const auto it = str.find(column); it != str.end()) return it->second;
    const auto num = result.num_outputs();
    if (const auto it = num.find(column); it != num.end()) return java::double_text(it->second);
    return "N/A";
  } catch (const cpmml::InvalidValueException& error) {
    return std::string("ERROR: ") + error.what();
  }
}
static void casts_in_document() {
  const std::string fields = R"(<DataField name="x" optype="continuous" dataType="double"/>
    <DataField name="n" optype="continuous" dataType="integer"/><DataField name="s" optype="categorical" dataType="string"/>)";
  auto model = [](const std::string& td, const std::string& output, const std::string& target = "y",
                  const std::string& targets = "") {
    return "<TransformationDictionary>" + td +
           R"(</TransformationDictionary><RegressionModel functionName="regression"><MiningSchema>
      <MiningField name="x"/><MiningField name="n"/><MiningField name="s"/><MiningField name=")" +
           target + R"(" usageType="target"/></MiningSchema>)" + targets + "<Output>" + output +
           R"(</Output><RegressionTable intercept="0"><NumericPredictor name="x" coefficient="1"/></RegressionTable></RegressionModel>)";
  };
  const std::string y = R"(<DataField name="y" optype="continuous" dataType="double"/>)";
  const std::string yi = R"(<DataField name="y" optype="continuous" dataType="integer"/>)";
  const std::string dint = R"(<DerivedField name="d" optype="continuous" dataType="integer"><FieldRef field="x"/></DerivedField>)";
  const std::string o_d = R"(<OutputField name="o" optype="categorical" dataType="string" feature="transformedValue"><FieldRef field="d"/></OutputField>)";
  CHECK(score_text(model(dint, o_d), fields + y, {{"x", "3"}}, "o") == "3", "integer DerivedField over 3.0 is 3");
  CHECK(score_text(model(dint, o_d), fields + y, {{"x", "3.5"}}, "o") ==
            "ERROR: DerivedField \"d\": the value 3.5 of type double is not a value of type integer (§2.6, Q2)",
        "B17: integer DerivedField over 3.5 is an invalid record naming the field");
  // An unused DerivedField is evaluated too (today's eager DAG, Q11 is P7d).
  const std::string dsd = R"(<DerivedField name="d" optype="continuous" dataType="double"><FieldRef field="s"/></DerivedField>)";
  CHECK(score_text(model(dsd, o_d), fields + y, {{"x", "1"}, {"s", "2.50"}}, "o") == "2.5",
        "double DerivedField over the text 2.50 is 2.5 (B38 part)");
  CHECK(score_text(model(dsd, o_d), fields + y, {{"x", "1"}, {"s", "x"}}, "o") ==
            "ERROR: DerivedField \"d\": the value \"x\" of type string is not a value of type double (§2.6, Q2)",
        "double DerivedField over the text x is an invalid record");
  const std::string ob = R"(<OutputField name="o" optype="categorical" dataType="boolean" feature="transformedValue"><FieldRef field="n"/></OutputField>)";
  CHECK(score_text(model("", ob), fields + y, {{"x", "1"}, {"n", "1"}}, "o") == "1.0", "boolean OutputField over 1");
  CHECK(score_text(model("", ob), fields + y, {{"x", "1"}, {"n", "2"}}, "o") ==
            "ERROR: OutputField \"o\": the value 2 of type integer is not a value of type boolean (§2.6, Q2)",
        "B24: integer 2 into a boolean OutputField is an invalid record naming the field");
  const std::string oi = R"(<OutputField name="o" optype="continuous" dataType="integer" feature="predictedValue"/>)";
  CHECK(score_text(model("", oi), fields + y, {{"x", "-4"}}, "o") == "-4.0", "integer predictedValue -4");
  CHECK(score_text(model("", oi), fields + y, {{"x", "2.5"}}, "o") ==
            "ERROR: OutputField \"o\": the value 2.5 of type double is not a value of type integer (§2.6, Q2)",
        "B24: a non-integral prediction into an integer OutputField is an invalid record");
  const std::string ois = R"(<OutputField name="o" optype="continuous" dataType="integer" feature="transformedValue"><FieldRef field="s"/></OutputField>)";
  CHECK(score_text(model("", ois), fields + y, {{"x", "1"}, {"s", "07"}}, "o") == "7.0",
        "integer OutputField over the text 07 is 7");
  CHECK(score_text(model("", ois), fields + y, {{"x", "1"}, {"s", "7.0"}}, "o") ==
            "ERROR: OutputField \"o\": the value \"7.0\" of type string is not a value of type integer (§2.6, Q2)",
        "integer OutputField over the text 7.0 is an invalid record (D42)");
  // An OutputField without dataType declares nothing, so nothing is cast.
  const std::string untyped = R"(<OutputField name="o" optype="continuous" feature="transformedValue"><FieldRef field="x"/></OutputField>)";
  CHECK(score_text(model("", untyped), fields + yi, {{"x", "2"}}, "o") == "2.0",
        "an OutputField without dataType is not cast");
  const std::string yb0 = R"(<DataField name="y" optype="categorical" dataType="boolean"/>)";
  const std::string untyped_n = R"(<OutputField name="o" optype="continuous" feature="transformedValue"><FieldRef field="n"/></OutputField>)";
  CHECK(score_text(model("", untyped_n), fields + yb0, {{"x", "1"}, {"n", "2"}}, "o") == "2.0",
        "an OutputField without dataType is not cast, whatever type the engine infers for it (here the target's)");
  // entityId declared integer (Q2): an id that is no integer is an invalid
  // record naming the field; it gave no value, silently, before P4.7.
  {
    const std::string tree = R"(<TreeModel functionName="classification"><MiningSchema><MiningField name="x"/>
      <MiningField name="c" usageType="target"/></MiningSchema><Output>
      <OutputField name="eid" optype="categorical" dataType="integer" feature="entityId"/></Output>
      <Node id="r"><True/><Node id="7" score="a"><SimplePredicate field="x" operator="lessThan" value="0"/></Node>
      <Node id="n2" score="b"><True/></Node></Node></TreeModel>)";
    const std::string c = R"(<DataField name="c" optype="categorical" dataType="string"><Value value="a"/><Value value="b"/></DataField>)";
    CHECK(score_text(tree, fields + c, {{"x", "-1"}}, "eid") == "7.0", "entityId 7 as an integer");
    CHECK(score_text(tree, fields + c, {{"x", "1"}}, "eid") ==
              "ERROR: OutputField \"eid\": the value \"n2\" of type string is not a value of type integer (§2.6, Q2)",
          "entityId n2 as an integer is an invalid record naming the field");
  }
  // §5.1: the target cast, after the Target.
  CHECK(score_text(model("", "", "y"), fields + yi, {{"x", "3"}}, "") == "3", "integer target: 3");
  CHECK(score_text(model("", "", "y"), fields + yi, {{"x", "3.5"}}, "") ==
            "ERROR: target \"y\": the predicted value 3.5 of type double is not a value of type integer (§2.6, Q2)",
        "B24: a non-integral regression prediction of an integer target is an invalid record naming the target");
  const std::string round = R"(<Targets><Target field="y" castInteger="round"/></Targets>)";
  CHECK(score_text(model("", "", "y", round), fields + yi, {{"x", "3.5"}}, "") == "4",
        "castInteger comes first (Targets.html order), so the cast has an integer");
  // P6.11 (B61, D50): castInteger of ±INF, NaN or a value whose integer is
  // outside int64 has no integer: an invalid record naming the target, for a
  // double target too, in the Target's every path (the score, a defaultValue,
  // a selectAll segment's text).
  {
    const std::string yd0 = R"(<DataField name="y" optype="continuous" dataType="double"/>)";
    const std::string why = " has no integer value (an integer has no INF or NaN and is an int64; §2.6, D4, D50, Q2)";
    auto cast = [](const std::string& kind, const std::string& extra = "") {
      return R"(<Targets><Target field="y" castInteger=")" + kind + "\"" + extra + "/></Targets>";
    };
    CHECK(score_text(model("", "", "y", cast("round")), fields + yd0, {{"x", "INF"}}, "") ==
              "ERROR: target \"y\": castInteger=\"round\" of the value Infinity" + why,
          "castInteger round of INF is an invalid record naming the target");
    CHECK(score_text(model("", "", "y", cast("ceiling")), fields + yd0, {{"x", "-INF"}}, "") ==
              "ERROR: target \"y\": castInteger=\"ceiling\" of the value -Infinity" + why,
          "castInteger ceiling of -INF is an invalid record");
    CHECK(score_text(model("", "", "y", cast("floor")), fields + yd0, {{"x", "1e19"}}, "") ==
              "ERROR: target \"y\": castInteger=\"floor\" of the value 1.0E19" + why,
          "castInteger floor of 1e19 (beyond int64) is an invalid record");
    CHECK(score_text(model("", "", "y", cast("round", R"( rescaleFactor="0")")), fields + yd0, {{"x", "INF"}}, "") ==
              "ERROR: target \"y\": castInteger=\"round\" of the value NaN" + why,
          "castInteger round of NaN (INF * rescaleFactor 0) is an invalid record");
    CHECK(score_text(model("", "", "y", cast("round", R"( rescaleConstant="1e19")")), fields + yd0, {{"x", "1"}}, "") ==
              "ERROR: target \"y\": castInteger=\"round\" of the value 1.0E19" + why,
          "castInteger after rescaleConstant: the rescaled value is cast");
    CHECK(score_text(model("", "", "y", cast("round")), fields + yd0, {{"x", "-9223372036854775808"}}, "") ==
              "-9.223372036854776E18",
          "castInteger round of -2^63, the least int64, is that integer");
    CHECK(score_text(model("", "", "y", cast("round")), fields + yd0, {{"x", "9223372036854775808"}}, "") ==
              "ERROR: target \"y\": castInteger=\"round\" of the value 9.223372036854776E18" + why,
          "castInteger round of 2^63 is outside int64");
    CHECK(score_text(model("", "", "y", cast("round", R"( min="0" max="10")")), fields + yd0, {{"x", "INF"}}, "") ==
              "10.0",
          "max comes before castInteger (Targets.html order): INF is clamped to 10 first");
    // An integer target: castInteger refuses INF before the target cast.
    CHECK(score_text(model("", "", "y", cast("round")), fields + yi, {{"x", "INF"}}, "") ==
              "ERROR: target \"y\": castInteger=\"round\" of the value Infinity" + why,
          "integer target: castInteger round of INF is an invalid record");
    // No Target@field: the message names the MiningSchema's target.
    CHECK(score_text(model("", "", "y", R"(<Targets><Target castInteger="round"/></Targets>)"), fields + yd0,
                     {{"x", "INF"}}, "") == "ERROR: target \"y\": castInteger=\"round\" of the value Infinity" + why,
          "a Target without field names the MiningSchema's target");
    // defaultValue goes through castInteger too (no prediction: x missing).
    const std::string dflt =
        R"(<Targets><Target field="y" castInteger="round"><TargetValue defaultValue="INF"/></Target></Targets>)";
    CHECK(score_text(model("", "", "y", dflt), fields + yd0, {{"n", "1"}}, "") ==
              "ERROR: target \"y\": castInteger=\"round\" of the value Infinity" + why,
          "a defaultValue INF under castInteger is an invalid record");
    const std::string seg_round =
        R"(<MiningModel functionName="regression"><MiningSchema><MiningField name="x"/><MiningField name="y" usageType="target"/></MiningSchema>
      <Segmentation multipleModelMethod="selectAll"><Segment id="1"><True/><RegressionModel functionName="regression">
      <MiningSchema><MiningField name="x"/><MiningField name="y" usageType="target"/></MiningSchema>
      <Targets><Target field="y" castInteger="round"/></Targets>
      <RegressionTable intercept="0"><NumericPredictor name="x" coefficient="1"/></RegressionTable></RegressionModel></Segment></Segmentation></MiningModel>)";
    CHECK(score_text(seg_round, fields + yd0, {{"x", "2.5"}}, "") == "[3.0]", "selectAll segment, castInteger: [3.0]");
    CHECK(score_text(seg_round, fields + yd0, {{"x", "INF"}}, "") ==
              "ERROR: target \"y\": castInteger=\"round\" of the value Infinity" + why,
          "selectAll segment, castInteger round of INF is an invalid record");
  }
  const std::string yb = R"(<DataField name="y" optype="categorical" dataType="boolean"/>)";
  CHECK(score_text(model("", "", "y"), fields + yb, {{"x", "1"}}, "") == "true", "boolean target: 1 is true");
  CHECK(score_text(model("", "", "y"), fields + yb, {{"x", "2"}}, "").rfind("ERROR: target \"y\"", 0) == 0,
        "boolean target: 2 is an invalid record");
  const std::string yd = R"(<DataField name="y" optype="continuous" dataType="double"/>)";
  CHECK(score_text(model("", "", "y"), fields + yd, {{"x", "3.5"}}, "") == "3.5", "a double target is not cast");
  // A selectAll segment's prediction is read as text (InternalModel::finish
  // of a string, Segment::predict): the target cast applies there too.
  const std::string select_all =
      R"(<MiningModel functionName="regression"><MiningSchema><MiningField name="x"/><MiningField name="y" usageType="target"/></MiningSchema>
      <Segmentation multipleModelMethod="selectAll"><Segment id="1"><True/><RegressionModel functionName="regression">
      <MiningSchema><MiningField name="x"/><MiningField name="y" usageType="target"/></MiningSchema>
      <RegressionTable intercept="0"><NumericPredictor name="x" coefficient="1"/></RegressionTable></RegressionModel></Segment></Segmentation></MiningModel>)";
  CHECK(score_text(select_all, fields + yi, {{"x", "3"}}, "") == "[3]", "selectAll segment, integer target: [3]");
  CHECK(score_text(select_all, fields + yi, {{"x", "3.5"}}, "") ==
            "ERROR: target \"y\": the predicted value 3.5 of type double is not a value of type integer (§2.6, Q2)",
        "selectAll segment, integer target: 3.5 is an invalid record");
}

static void arg_buffer() {
  for (std::size_t n : {std::size_t{0}, std::size_t{3}, fn::ArgBuffer::kInline, fn::ArgBuffer::kInline + 3}) {
    fn::ArgBuffer args(n);
    for (std::size_t i = 0; i < n; i++)
      args.emplace([&] { return Value::string("text number " + std::to_string(i) + " longer than the SSO buffer"); });
    bool ok = args.size() == n;
    for (std::size_t i = 0; i < n; i++)
      ok = ok && args.data()[i].text() == "text number " + std::to_string(i) + " longer than the SSO buffer";
    CHECK(ok, "ArgBuffer holds " + std::to_string(n) + " arguments in order");
  }
}

int main() {
  Indexer indexer;
  // The strings these checks make are a document's being loaded: interned in
  // its pool, codes in the order they are made (design P5).
  const cpmml::StringScope strings(cpmml::StringScope::Loading{}, std::make_shared<cpmml::DocumentStrings>());
  table();
  kernels();
  legacy_switches(indexer);
  concat_casts(indexer);
  apply_output_table();
  if_lazy();
  domains();
  domains_in_document();
  concat_in_document();
  format_number_kernel(indexer);
  format_number_in_document();
  integer_kernels();
  integer_in_document();
  declared_casts(indexer);
  casts_in_document();
  arg_buffer();
  std::cout << checks - failures << "/" << checks << " checks passed" << std::endl;
  return failures == 0 ? 0 : 1;
}
