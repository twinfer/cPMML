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
 *     here by the symptom the design's bug table records for it (§1.2: B21
 *     "0/0 gives NaN ...; integer 1/0 gives INF; floor/ceil/round/modulo of
 *     INF are not invalid"; B23 "concat drops number arguments"; B2 "keep
 *     the argument's code"), so that the commit removing a switch sees these
 *     checks fail and replaces them with the spec's.
 *   - ArgBuffer, inline and on the heap.
 *
 * Usage: fn_table.exe   (exit 0 when every check holds)
 *******************************************************************************/

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
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

// Calls the kernel of `name` over `args` with `bound`.
static fn::Status call(const char* name, const std::vector<Value>& args, Value& out, const fn::Bound& bound = {}) {
  const fn::Signature* sig = fn::find(name);
  if (!sig) throw std::logic_error(std::string("no row ") + name);
  return fn::instantiate(sig->op)(bound, args.data(), args.size(), out);
}
static double number(const char* name, const std::vector<Value>& args, const fn::Bound& bound = {}) {
  Value out;
  if (call(name, args, out, bound) != fn::Status::Ok || out.is_missing())
    throw std::logic_error(std::string(name) + ": no number");
  return out.as_double();
}
static bool invalid(const char* name, const std::vector<Value>& args, const fn::Bound& bound = {}) {
  Value out;
  return call(name, args, out, bound) == fn::Status::Invalid;
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
  // The 59 functions core/builtinfunctions.h evaluated, the 11 it refused.
  CHECK(evaluated == 59, "every function cPMML evaluated before P4.1 is in the table with a kernel");
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
      "normalPDF, stdNormalCDF, stdNormalPDF";
  for (const char* refused : {"and", "or", "isValid", "isNotValid", "normalIDF", "stdNormalIDF", "formatNumber",
                              "formatDatetime", "dateDaysSinceYear", "dateSecondsSinceYear",
                              "dateSecondsSinceMidnight"})
    CHECK(message_of<cpmml::ParsingException>([&] { fn::engine_row(refused); }) ==
              "unsupported value \"" + std::string(refused) + "\" for Apply@function; cPMML supports " + supported,
          std::string(refused) + ": refused at load, as before P4.1");
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
  fn::Bound integer_division;
  integer_division.integer_division = true;
  CHECK(number("/", nums({7, 2}), integer_division) == 3, "7/2 of integers is 3");
  CHECK(number("/", nums({-7, 2}), integer_division) == -3, "-7/2 of integers is -3 (toward zero)");
  CHECK(number("/", nums({7, 2})) == 3.5, "7/2 of doubles is 3.5");
  // pow of integers with a negative exponent has no integer value (D8).
  fn::Bound integer_power;
  integer_power.integer_power = true;
  CHECK(invalid("pow", nums({7, -2}), integer_power), "pow(7,-2) of integers is invalid (D8)");
  CHECK(number("pow", nums({7, 2}), integer_power) == 49, "pow(7,2) of integers is 49");
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
  CHECK(number("equal", nums({0.1, 0.1}), floats) == 0, "equal(float 0.1, double 0.1) is false");
  CHECK(number("isIn", nums({0.1, 0.2, 0.1}), floats) == 0, "isIn(float 0.1, {0.2, 0.1}) is false");
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
}

static void legacy_switches(Indexer& indexer) {
  // B21 (P4.4): 0/0 gives NaN; floor/ceil/round/modulo of INF are numbers.
  const double inf = std::numeric_limits<double>::infinity();
  CHECK(std::isnan(number("/", nums({0, 0}))), "legacy::B21: 0/0 is NaN");
  CHECK(std::isnan(number("modulo", nums({inf, 2}))), "legacy::B21: modulo(INF, 2) is NaN");
  CHECK(number("floor", nums({inf})) == inf && number("round", nums({-inf})) == -inf, "legacy::B21: floor(INF)");
  // D20 (P4.4): integer 1/0 gives INF.
  fn::Bound integer_division;
  integer_division.integer_division = true;
  CHECK(number("/", nums({1, 0}), integer_division) == inf, "legacy::D20: integer 1/0 is INF");
  // D4 (P4.5): floor of 3.5 is the double 3; integer arithmetic does not
  // overflow at int64.
  CHECK(number("floor", nums({3.5})) == 3, "floor(3.5) = 3");
  CHECK(number("*", nums({9.2233720368547758e18, 2})) == 1.8446744073709552e19, "legacy::D4: no int64 overflow");
  // B41 (P4.2): outside the domain, scoring goes on.
  CHECK(std::isnan(number("sqrt", nums({-1}))), "legacy::B41: sqrt(-1) is NaN");
  CHECK(std::isnan(number("modulo", nums({1, 0}))), "legacy::B41: modulo(1, 0) is NaN");
  // B23 (P4.6): concat reads each argument's text; a number has none.
  {
    Value out;
    const std::vector<Value> args{Value(3.0), Value("|", DataType::DataTypeValue::STRING)};
    CHECK(call("concat", args, out) == fn::Status::Ok && out.text() == "|", "legacy::B23: concat(3, \"|\") is \"|\"");
    CHECK(out.as_double() == indexer.get_string_value("|"), "concat packs its text with the Indexer's code");
  }
  // B2 (P6): uppercase keeps its argument's string code.
  {
    Value out;
    const Value abc("abc", DataType::DataTypeValue::STRING);
    const std::vector<Value> args{abc};
    CHECK(call("uppercase", args, out) == fn::Status::Ok && out.text() == "ABC", "uppercase(abc) is ABC");
    CHECK(out.as_double() == abc.as_double(), "legacy::B2: uppercase keeps the argument's code");
  }
  // substring: startPos beyond the end is invalid; below 1 the whole input
  // (legacy::B41).
  {
    Value out;
    const Value abc("abc", DataType::DataTypeValue::STRING);
    CHECK(call("substring", {abc, Value(2.0), Value(5.0)}, out) == fn::Status::Ok && out.text() == "bc",
          "substring(abc, 2, 5) is bc");
    CHECK(call("substring", {abc, Value(4.0), Value(1.0)}, out) == fn::Status::Ok && out.text().empty(),
          "substring(abc, 4, 1) is empty");
    CHECK(invalid("substring", {abc, Value(5.0), Value(1.0)}), "substring(abc, 5, 1): no such position");
    CHECK(call("substring", {abc, Value(0.0), Value(1.0)}, out) == fn::Status::Ok && out.text() == "abc",
          "legacy::B41: substring(abc, 0, 1) is abc");
  }
  // replace and matches: a pattern std::regex refuses is an invalid result.
  {
    const Value abc("abc", DataType::DataTypeValue::STRING);
    const Value bad("(", DataType::DataTypeValue::STRING);
    const Value x("x", DataType::DataTypeValue::STRING);
    CHECK(invalid("replace", {abc, bad, x}), "replace with pattern ( is invalid");
    CHECK(invalid("matches", {abc, bad}), "matches with pattern ( is invalid");
    Value out;
    CHECK(call("replace", {abc, Value("b", DataType::DataTypeValue::STRING), x}, out) == fn::Status::Ok &&
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
    CHECK(call("if", {Value(1.0), Value("abc", DataType::DataTypeValue::STRING)}, out) == fn::Status::Ok &&
              out.text() == "abc",
          "if_then_else: the branch taken's value");
    Value none;
    CHECK(call("if", {Value(0.0)}, none) == fn::Status::Ok && none.is_missing(),
          "if_then_else: no branch taken is missing");
  }

  // Through a whole document: o = if(i, e[, ELSE]) for each set of
  // attributes; i and e are integers, the ELSE part a Constant 3 or
  // pow(i, -1), which is invalid for i = 0 (D8) and so shows whether it was
  // evaluated.
  const std::string three = R"(<Constant dataType="integer">3</Constant>)";
  const std::string invalid_else =
      R"(<Apply function="pow"><FieldRef field="i"/><Constant dataType="integer">-1</Constant></Apply>)";
  auto if_apply = [](const std::string& attrs, const std::string& otherwise) {
    return R"(<Apply function="if")" + attrs + R"(><FieldRef field="i"/><FieldRef field="e"/>)" + otherwise +
           "</Apply>";
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
  CHECK(apply_output(R"(<Apply function="if" defaultValue="none"><FieldRef field="i"/><FieldRef field="t"/>)"
                     R"(<FieldRef field="r"/></Apply>)",
                     "1", "", "abc", "b", "string") == "abc",
        "if over strings: THEN");
  CHECK(apply_output(R"(<Apply function="if" defaultValue="none"><FieldRef field="i"/><FieldRef field="t"/></Apply>)",
                     "0", "", "abc", "b", "string") == "none",
        "if over strings: false without ELSE gives defaultValue");
}

static void arg_buffer() {
  for (std::size_t n : {std::size_t{0}, std::size_t{3}, fn::ArgBuffer::kInline, fn::ArgBuffer::kInline + 3}) {
    fn::ArgBuffer args(n);
    for (std::size_t i = 0; i < n; i++)
      args.emplace([&] { return Value("text number " + std::to_string(i) + " longer than the SSO buffer",
                                      DataType::DataTypeValue::STRING); });
    bool ok = args.size() == n;
    for (std::size_t i = 0; i < n; i++)
      ok = ok && args.data()[i].text() == "text number " + std::to_string(i) + " longer than the SSO buffer";
    CHECK(ok, "ArgBuffer holds " + std::to_string(n) + " arguments in order");
  }
}

int main() {
  Indexer indexer;
  const Value::ScopedIndexer scope(&indexer);
  table();
  kernels();
  legacy_switches(indexer);
  apply_output_table();
  if_lazy();
  arg_buffer();
  std::cout << checks - failures << "/" << checks << " checks passed" << std::endl;
  return failures == 0 ? 0 : 1;
}
