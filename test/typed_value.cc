/*******************************************************************************
 * The type library of the typed core (src/core/types/, docs/design/typed-core.md
 * §2, design commit P1.1): one check per rule, each citing the spec sentence or
 * the design's D-rule it encodes. Expected values come from the spec, the XSD,
 * or Java (JPMML's oracle for number text) -- never from cPMML.
 *
 * Usage: typed_value.exe   (exit 0 when every check holds)
 *******************************************************************************/

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/types/arithmetic.h"
#include "core/types/cast.h"
#include "core/types/cold.h"
#include "core/types/compare.h"
#include "core/types/java_text.h"
#include "core/types/keys.h"
#include "core/types/parse.h"
#include "core/types/string_pool.h"
#include "core/types/value.h"

namespace cpmml::types::testing {
struct ColdPathIssuer {
  static cold::ColdPath make() { return {}; }
};
}  // namespace cpmml::types::testing

using namespace cpmml::types;

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

static const double kInf = std::numeric_limits<double>::infinity();
static const double kNaN = std::numeric_limits<double>::quiet_NaN();
static const float kInfF = std::numeric_limits<float>::infinity();
static const cold::ColdPath kCold = testing::ColdPathIssuer::make();

template <class T>
static bool holds(const Value& v) {
  return std::holds_alternative<T>(v);
}
static bool invalid_because(const Value& v, InvalidReason r) {
  const auto* i = std::get_if<Invalid>(&v);
  return i && i->reason == r;
}
static bool is_double(const Value& v, double d) {
  const auto* x = std::get_if<Double>(&v);
  return x && (x->v == d || (std::isnan(x->v) && std::isnan(d))) && std::signbit(x->v) == std::signbit(d);
}
static bool is_float(const Value& v, float f) {
  const auto* x = std::get_if<Float>(&v);
  return x && (x->v == f || (std::isnan(x->v) && std::isnan(f))) && std::signbit(x->v) == std::signbit(f);
}
static bool is_integer(const Value& v, std::int64_t i) {
  const auto* x = std::get_if<Integer>(&v);
  return x && x->v == i;
}
static bool is_text(const Value& v, std::string_view t) {
  const auto* x = std::get_if<String>(&v);
  return x && x->text() == t;
}

// ============================================================ §2.1 the variant
static_assert(sizeof(Value) == 16 && std::is_trivially_copyable_v<Value>, "§2.1: 16 bytes, trivially copyable");
static_assert(index_matches<Missing>() && index_matches<Invalid>() && index_matches<Integer>() &&
                  index_matches<Float>() && index_matches<Double>() && index_matches<Boolean>() &&
                  index_matches<String>(),
              "§2.1: the variant index is the TypeId");
static_assert(std::is_same_v<decltype(Integer::v), std::int64_t>, "D4: integers are int64");
static_assert(std::is_same_v<decltype(Float::v), float>, "§2.1 / L5: Float holds a C++ float");

// ============================================================ §2.2 concepts and exhaustiveness
static_assert(PmmlNumeric<Integer> && PmmlNumeric<Float> && PmmlNumeric<Double>, "§2.2 numeric types");
static_assert(PmmlParsedType<Boolean> && PmmlParsedType<String> && !PmmlNumeric<Boolean> && !PmmlNumeric<String>,
              "§2.2 parsed, non-numeric types");
static_assert(Sentinel<Missing> && Sentinel<Invalid> && !PmmlParsedType<Missing>, "§2.2 sentinels");
static_assert(!Temporal<Integer> && !Temporal<String> && !Temporal<Boolean> && !Temporal<Missing>,
              "§2.2: Temporal is empty until P10");
static_assert(categories<Integer> == 1 && categories<String> == 1 && categories<Missing> == 1 &&
                  categories<NotAnAlternative> == 0,
              "§2.2: every alternative in exactly one category");
static_assert(Integer::rank < Float::rank && Float::rank < Double::rank,
              "Functions.html: \"The inheritance precedence ... string double float integer\"");
// Every library visitor is exhaustive and has no unconstrained catch-all (§2.2).
static_assert(ExhaustiveVisitor<cold::JavaText> && NoCatchAll<cold::JavaText>);
static_assert(ExhaustiveVisitor<CastTo<Integer>> && ExhaustiveVisitor<CastTo<Float>> &&
              ExhaustiveVisitor<CastTo<Double>> && ExhaustiveVisitor<CastTo<Boolean>> &&
              ExhaustiveVisitor<CastTo<String>> && NoCatchAll<CastTo<String>>);
static_assert(ExhaustiveBinaryVisitor<Comparison> && NoBinaryCatchAll<Comparison>);
static_assert(ExhaustiveBinaryVisitor<Arithmetic<Divide>> && ExhaustiveVisitor<Unary<Ceil>>);
static_assert(ExhaustiveVisitor<KeyHashOf>);
// §2.4 / §6.4: a ColdPath cannot be made outside its issuers.
static_assert(!std::is_default_constructible_v<cold::ColdPath>, "§6.4: ColdPath's constructor is private");
static_assert(std::is_trivially_copyable_v<cold::ColdPath>);

// ============================================================ §2.5 promotion
static_assert(std::is_same_v<Promoted<Integer, Integer>, Integer> && std::is_same_v<Promoted<Integer, Float>, Float> &&
                  std::is_same_v<Promoted<Integer, Double>, Double> &&
                  std::is_same_v<Promoted<Float, Integer>, Float> && std::is_same_v<Promoted<Float, Float>, Float> &&
                  std::is_same_v<Promoted<Float, Double>, Double> &&
                  std::is_same_v<Promoted<Double, Integer>, Double> &&
                  std::is_same_v<Promoted<Double, Float>, Double> && std::is_same_v<Promoted<Double, Double>, Double>,
              "§2.5 table; Functions.html \"the least restrictive dataType will be inherited\"");
template <class To, class From>
concept CanWiden = requires(From x) { widen<To>(x); };
static_assert(CanWiden<Double, Integer> && CanWiden<Float, Integer> && CanWiden<Double, Float>);
static_assert(!CanWiden<Integer, Double> && !CanWiden<Integer, Float> && !CanWiden<Float, Double>,
              "Functions.html: \"implicit casts are allowed in the direction of less restriction\"");

static void test_promotion() {
  const char* rule =
      "§2.6: integer → float widens, int64 → float rounds to nearest (Java (float)16777217L = 1.6777216E7)";
  CHECK(widen<Float>(Integer{16777217}).v == 16777216.0f, rule);
  CHECK(widen<Double>(Float{0.1f}).v == static_cast<double>(0.1f), "§2.6: float → double is exact");
}

// ============================================================ §2.3 strings
static void test_strings() {
  StringPool pool;
  const StringRep* abc = pool.intern("abc");
  CHECK(pool.intern("abc") == abc, "§2.3: one rep per text in a document's pool");
  CHECK(pool.intern("ABC") != abc && pool.size() == 2, "§2.3: distinct texts get distinct reps");
  CHECK(abc->id != StringRep::kUnpooled && abc->hash == std::hash<std::string_view>{}("abc"),
        "§2.3: pooled id; hash of the text");
  CHECK(pool.find("abc") == abc && pool.find("zz") == nullptr, "§2.3: find is a read-only lookup");
  const StringRep* empty = pool.intern("");
  CHECK(empty->text.empty() && empty != abc, "B37: the empty text is a string");
  pool.freeze();
  bool threw = false;
  try {
    pool.intern("late");
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw, "§2.3: the pool is frozen after load; interning then is an error, not a silent insert");

  RecordArena arena;
  const String pooled{abc}, upper{pool.find("ABC")};
  const String a1{arena.make("abc")}, a2{arena.make("abc")}, yy{arena.make("yy")};
  CHECK(a1.rep->id == StringRep::kUnpooled && a1.rep->hash == abc->hash, "§2.3: arena reps hash like pooled ones");
  CHECK(string_equal(pooled, a1) && string_equal(a1, pooled),
        "XSD string: equality is equality of text (pooled vs arena)");
  CHECK(string_equal(a1, a2), "XSD string: two runtime strings of one text are equal");
  CHECK(!string_equal(a1, yy), "B1: two strings the document never names are unequal when their texts differ");
  CHECK(!string_equal(pooled, upper), "§2.3: two pooled reps of one pool are equal iff the same pointer");
  const String made_upper{arena.make("ABC")};  // what uppercase("abc") makes
  CHECK(string_equal(made_upper, upper), "B2: a string made while scoring equals the pooled one of the same text");
}

// ============================================================ §2.6 lexical forms
static void test_parse() {
  StringPool pool;
  RecordArena arena;
  ParseContext run{nullptr, &pool, &arena};

  // integer: XSD §3.3.13.1 "a finite-length sequence of decimal digits (#x30-#x39) with an optional leading sign"
  const char* xi = "XSD §3.3.13.1 integer lexical space";
  CHECK(is_integer(parse_as<Integer>("42", run), 42), xi);
  CHECK(is_integer(parse_as<Integer>("+100000", run), 100000), xi);
  CHECK(is_integer(parse_as<Integer>("-0", run), 0), xi);
  CHECK(is_integer(parse_as<Integer>("007", run), 7), xi);
  CHECK(is_integer(parse_as<Integer>(" \t12\n", run), 12), "XSD whiteSpace collapse for non-string atomic types");
  for (const char* bad : {"", "+", "-", "+-1", "1.0", "1e3", "12a", "0x10", " ", "1 2"})
    CHECK(invalid_because(parse_as<Integer>(bad, run), InvalidReason::Malformed), xi);
  CHECK(is_integer(parse_as<Integer>("9223372036854775807", run), std::numeric_limits<std::int64_t>::max()), "D4");
  CHECK(is_integer(parse_as<Integer>("-9223372036854775808", run), std::numeric_limits<std::int64_t>::min()), "D4");
  CHECK(invalid_because(parse_as<Integer>("9223372036854775808", run), InvalidReason::Overflow),
        "D4: integers are int64; overflow invalid");
  CHECK(invalid_because(parse_as<Integer>("-9223372036854775809", run), InvalidReason::Overflow), "D4");
  CHECK(!parse(std::type_identity<Integer>{}, "1.5", run).has_value(), "§2.2: parse() is the strict reading");

  // double: XSD §3.2.5.1 "The special values ... have lexical representations INF, -INF and NaN"
  const char* xd = "XSD §3.2.5.1 double lexical space";
  CHECK(is_double(parse_as<Double>("1.5", run), 1.5), xd);
  CHECK(is_double(parse_as<Double>(".5", run), 0.5) && is_double(parse_as<Double>("5.", run), 5.0), xd);
  CHECK(is_double(parse_as<Double>("12.78e-2", run), 0.1278), xd);
  CHECK(is_double(parse_as<Double>("INF", run), kInf) && is_double(parse_as<Double>("-INF", run), -kInf), xd);
  CHECK(is_double(parse_as<Double>("NaN", run), kNaN), "B39: the XSD double value space contains NaN");
  CHECK(is_double(parse_as<Double>("-0", run), -0.0),
        "D18 / XSD: \"Lexical representations for zero may take a positive or negative sign\"");
  CHECK(is_double(parse_as<Double>("1e400", run), kInf), "lexical.h: a decimal beyond the range rounds to INF");
  CHECK(is_double(parse_as<Double>("+INF", run), kInf),
        "D30: XSD 1.1 §3.3.5.2 \"the literals 'INF', '+INF', '-INF', and 'NaN'\"");
  for (const char* bad :
       {"", "inf", "Infinity", "+inf", "+NaN", "-NaN", "nan", "NAN", "1d", "0x1p3", ".", "e5", "1e", "1,5", "abc"})
    CHECK(invalid_because(parse_as<Double>(bad, run), InvalidReason::Malformed),
          "D11: inputs and attributes use the strict, case-sensitive XSD forms");

  // float: XSD §3.2.4 "maps to the normalized value in the value space of float that is closest to d"
  const char* xf = "XSD §3.2.4: the float closest to the decimal (Java Float.parseFloat)";
  CHECK(is_float(parse_as<Float>("0.1", run), 0.1f), xf);
  CHECK(is_float(parse_as<Float>("1.00000005960464477550", run), 1.00000012f),
        "XSD §3.2.4: one rounding, decimal → float (Java: 1.0000001; through a double it would be 1.0)");
  CHECK(is_float(parse_as<Float>("3.4e39", run), kInfF), "B4: 3.4e39 is not a finite float (Java: Infinity)");
  CHECK(is_float(parse_as<Float>("-3.4e39", run), -kInfF), "B4");
  CHECK(is_float(parse_as<Float>("1e-50", run), 0.0f), "XSD float: underflow to 0 (Java: 0.0)");
  CHECK(is_float(parse_as<Float>("1e-45", run), std::numeric_limits<float>::denorm_min()), "Java: 1.4E-45");
  CHECK(is_float(parse_as<Float>("NaN", run), std::numeric_limits<float>::quiet_NaN()), "XSD §3.2.4.1 NaN");
  CHECK(invalid_because(parse_as<Float>("inf", run), InvalidReason::Malformed), "D11");
  CHECK(is_float(parse_as<Float>("+INF", run), kInfF), "D30: XSD 1.1 §3.3.4.2 '+INF'");
  CHECK(invalid_because(parse_as<Float>("+inf", run), InvalidReason::Malformed), "D11, D30");

  // boolean: XSD §3.2.2.1 "{true, false, 1, 0}"
  const char* xb = "XSD §3.2.2.1: boolean literals {true, false, 1, 0}";
  CHECK(std::get<Boolean>(parse_as<Boolean>("true", run)).v && std::get<Boolean>(parse_as<Boolean>("1", run)).v, xb);
  CHECK(!std::get<Boolean>(parse_as<Boolean>("false", run)).v && !std::get<Boolean>(parse_as<Boolean>("0", run)).v, xb);
  CHECK(std::get<Boolean>(parse_as<Boolean>(" true ", run)).v, "XSD whiteSpace collapse");
  for (const char* bad : {"TRUE", "True", "yes", "2", "", "1.0", "t"})
    CHECK(invalid_because(parse_as<Boolean>(bad, run), InvalidReason::Malformed), xb);

  // string: XSD "for string the value of whiteSpace is preserve"
  CHECK(is_text(parse_as<String>(" a b ", run), " a b "), "XSD string whiteSpace preserve");
  CHECK(is_text(parse_as<String>("", run), ""), "B37: an empty cell of a string field is the empty string");
  StringPool load_pool;
  ParseContext load{&load_pool, nullptr, nullptr};
  const Value at_load = parse_as<String>("x", load);
  CHECK(std::get<String>(at_load).rep == load_pool.find("x"), "§2.3: document text is interned at load");
  load_pool.freeze();
  ParseContext scoring{nullptr, &load_pool, &arena};
  CHECK(std::get<String>(parse_as<String>("x", scoring)).rep == load_pool.find("x"),
        "§2.3: an input naming a pooled string gets the pooled rep");
  CHECK(std::get<String>(parse_as<String>("unseen", scoring)).rep->id == StringRep::kUnpooled,
        "§2.3: an unseen input gets an arena rep of its own text");

  // Constant text: Transformations.html "These three constants are case insensitive and only meaningful as
  // floating point values (datatype equal to "float" or "double")"
  const char* tc = "Transformations.html Constant: NaN, INF, -INF case insensitive for float/double (D11)";
  CHECK(is_double(parse_constant_as<Double>("inf", run), kInf), tc);
  CHECK(is_double(parse_constant_as<Double>("-Inf", run), -kInf), tc);
  CHECK(is_double(parse_constant_as<Double>("nAn", run), kNaN), tc);
  CHECK(
      is_float(parse_constant_as<Float>("INF", run), kInfF) && is_float(parse_constant_as<Float>("-inf", run), -kInfF),
      tc);
  const char* tp = "D30: +INF, INF's XSD 1.1 spelling, is case insensitive in Constant text too";
  CHECK(is_double(parse_constant_as<Double>("+INF", run), kInf) &&
            is_double(parse_constant_as<Double>("+inf", run), kInf),
        tp);
  CHECK(is_float(parse_constant_as<Float>("+Inf", run), kInfF), tp);
  CHECK(invalid_because(parse_constant_as<Double>("+nan", run), InvalidReason::Malformed), "D30: no +NaN");
  CHECK(invalid_because(parse_constant_as<Integer>("+INF", run), InvalidReason::Malformed), "D30: float/double only");
  CHECK(is_text(parse_constant_as<String>("+INF", run), "+INF"), "D30: a string keeps its text");
  CHECK(is_double(parse_constant_as<Double>("2.5", run), 2.5), "Constant text otherwise strict");
  CHECK(invalid_because(parse_constant_as<Double>("Infinity", run), InvalidReason::Malformed),
        "Constant: only INF/-INF/NaN");
  CHECK(invalid_because(parse_constant_as<Integer>("inf", run), InvalidReason::Malformed),
        "Transformations.html: \"No other data type will be accepted for these constants\"");
  CHECK(is_text(parse_constant_as<String>("INF", run), "INF"),
        "Transformations.html: \"if datatype is set to ... \"string\", then the literal strings will be signified\"");

  // run-time type
  CHECK(is_integer(cold::parse_as(kCold, TypeId::Integer, "5", run), 5), "§2.4 ParseAs by TypeId");
  CHECK(is_double(cold::parse_as(kCold, TypeId::Double, "5", run), 5.0), "§2.4 ParseAs by TypeId");
  bool threw = false;
  try {
    cold::parse_as(kCold, TypeId::Missing, "5", run);
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw, "no silent fallback: Missing has no lexical space");
}

// ============================================================ §2.6 casts
static void test_casts() {
  StringPool pool;
  RecordArena arena;
  ParseContext run{nullptr, &pool, &arena};
  std::string scratch;
  auto cast = [&](TypeId to, const Value& v) { return cold::cast(kCold, to, v, run, scratch); };

  CHECK(is_double(cast(TypeId::Double, Integer{3}), 3.0), "§2.6 integer → double: widen");
  CHECK(is_float(cast(TypeId::Float, Integer{16777217}), 16777216.0f), "§2.6 integer → float rounds to nearest");
  CHECK(is_double(cast(TypeId::Double, Float{0.1f}), static_cast<double>(0.1f)), "§2.6 float → double exact");
  CHECK(is_float(cast(TypeId::Float, Double{0.1}), 0.1f), "§2.6 double → float: IEEE nearest");
  CHECK(is_float(cast(TypeId::Float, Double{1e300}), kInfF), "§2.6 double → float overflow → INF (JPMML: Infinity)");
  CHECK(is_float(cast(TypeId::Float, Double{-1e300}), -kInfF), "§2.6 overflow → -INF");
  CHECK(is_float(cast(TypeId::Float, Double{3.4028235677973366e38}), kInfF),
        "IEEE: the halfway point to INF rounds to INF (Java: Infinity)");
  CHECK(is_float(cast(TypeId::Float, Double{3.4028235677973362e38}), std::numeric_limits<float>::max()),
        "IEEE: below the halfway point, FLT_MAX (Java: 3.4028235E38)");
  CHECK(is_float(cast(TypeId::Float, Double{1e-300}), 0.0f), "§2.6 double → float underflow → 0");
  CHECK(is_integer(cast(TypeId::Integer, Double{3.0}), 3), "§2.6 double → integer when integral");
  CHECK(invalid_because(cast(TypeId::Integer, Double{3.5}), InvalidReason::NotIntegral),
        "§2.6 / B17: Functions.html \"a double expression can not be implicitly cast as an integer\"");
  CHECK(invalid_because(cast(TypeId::Integer, Float{2.5f}), InvalidReason::NotIntegral), "§2.6 float → integer");
  CHECK(invalid_because(cast(TypeId::Integer, Double{1e19}), InvalidReason::Overflow), "§2.6 / D4 out of int64 range");
  CHECK(invalid_because(cast(TypeId::Integer, Double{9223372036854775808.0}), InvalidReason::Overflow), "D4: 2^63");
  CHECK(is_integer(cast(TypeId::Integer, Double{-9223372036854775808.0}), std::numeric_limits<std::int64_t>::min()),
        "D4: -2^63");
  CHECK(invalid_because(cast(TypeId::Integer, Double{kInf}), InvalidReason::Overflow), "§2.6 INF is no integer");
  CHECK(invalid_because(cast(TypeId::Integer, Double{kNaN}), InvalidReason::NotIntegral), "§2.6 NaN is no integer");
  CHECK(is_text(cast(TypeId::String, Integer{7}), "7"), "§2.6 / Q1 number → string: Java Long.toString");
  CHECK(is_text(cast(TypeId::String, Double{2.0}), "2.0"), "Q1: Double.toString(2.0)");
  CHECK(is_text(cast(TypeId::String, Double{1e7}), "1.0E7"), "Q1: Double.toString(1e7)");
  CHECK(is_text(cast(TypeId::String, Float{0.1f}), "0.1"), "Q1: Float.toString(0.1f)");
  CHECK(is_integer(cast(TypeId::Integer, String{arena.make(" 42 ")}), 42), "§2.6 string → number: strict XSD reading");
  CHECK(invalid_because(cast(TypeId::Double, String{arena.make("abc")}), InvalidReason::Malformed),
        "§2.6 string → number: else Invalid(Malformed)");
  CHECK(invalid_because(cast(TypeId::Integer, String{arena.make("2.0")}), InvalidReason::Malformed),
        "§2.6 string → integer is the integer lexical space");
  CHECK(std::get<Boolean>(cast(TypeId::Boolean, String{arena.make("1")})).v, "§2.6 string → boolean: XSD");
  CHECK(invalid_because(cast(TypeId::Boolean, String{arena.make("TRUE")}), InvalidReason::Malformed), "§2.6 / D11");
  CHECK(is_integer(cast(TypeId::Integer, Boolean{true}), 1) && is_double(cast(TypeId::Double, Boolean{false}), 0.0),
        "§2.6 boolean → number: true 1, false 0 (XPath)");
  CHECK(std::get<Boolean>(cast(TypeId::Boolean, Integer{1})).v &&
            !std::get<Boolean>(cast(TypeId::Boolean, Double{0.0})).v,
        "§2.6 / D14 number → boolean: 1 → true, 0 → false");
  CHECK(!std::get<Boolean>(cast(TypeId::Boolean, Double{-0.0})).v, "D14: -0 is 0");
  CHECK(invalid_because(cast(TypeId::Boolean, Integer{2}), InvalidReason::NoCast), "§2.6 / D14: 2 is no boolean");
  CHECK(invalid_because(cast(TypeId::Boolean, Double{kNaN}), InvalidReason::NoCast), "D14: NaN is no boolean");
  CHECK(invalid_because(cast(TypeId::Boolean, Double{0.5}), InvalidReason::NoCast), "D14");
  CHECK(is_text(cast(TypeId::String, Boolean{true}), "true"), "§2.6 boolean → string: XSD canonical");
  const Value s = String{arena.make("x")};
  CHECK(std::get<String>(cast(TypeId::String, s)).rep == std::get<String>(s).rep, "§2.6 T → T identity");
  CHECK(holds<Missing>(cast(TypeId::Integer, Missing{})), "a missing value stays missing through a cast");
  CHECK(invalid_because(cast(TypeId::Integer, Invalid{InvalidReason::Domain}), InvalidReason::Domain),
        "an invalid value stays invalid through a cast");
  // Typed (hot path) use, no ColdPath needed.
  CHECK(is_integer(CastTo<Integer>{&run, &scratch}(Double{4.0}), 4), "§2.4 CastNode<From, To> calls CastTo directly");
}

// ============================================================ §2.7 comparison
static void test_compare() {
  StringPool pool;
  RecordArena arena;
  std::string scratch;
  auto rel = [&](Relation r, const Value& a, const Value& b) { return cold::relate(kCold, r, a, b, scratch); };
  auto str = [&](std::string_view t) { return Value{String{arena.make(t)}}; };
  using R = Relation;

  // numeric × numeric, in the promoted type, either order (D1)
  const Value f = Float{0.1f}, d = Double{0.1};
  CHECK(rel(R::Equal, f, d) == Tri::False && rel(R::Equal, d, f) == Tri::False,
        "D1: equal(float 0.1, double 0.1) compared in double, either order (JPMML: true one way)");
  CHECK(rel(R::LessOrEqual, f, d) == Tri::False && rel(R::Greater, f, d) == Tri::True,
        "L4: lessOrEqual agrees with equal (one ordering)");
  CHECK(rel(R::Equal, Integer{3}, Double{3.5}) == Tri::False && rel(R::Less, Integer{3}, Double{3.5}) == Tri::True,
        "§2.7: integer against double in double");
  CHECK(rel(R::Equal, Integer{3}, Double{3.0}) == Tri::True && rel(R::Equal, Double{3.0}, Integer{3}) == Tri::True,
        "D1");
  CHECK(rel(R::Equal, Integer{16777217}, Float{16777216.0f}) == Tri::True,
        "§2.5: integer against float compares in float (the promoted type)");
  // INF
  CHECK(rel(R::Equal, Double{kInf}, Double{kInf}) == Tri::Unknown,
        "B44 / D9: BuiltinFunctions.html \"any attempt to [compare] either with itself is indeterminate and will "
        "result in a missing value\"");
  CHECK(rel(R::NotEqual, Double{kInf}, Double{kInf}) == Tri::Unknown, "B44: notEqual too");
  CHECK(rel(R::LessOrEqual, Double{-kInf}, Double{-kInf}) == Tri::Unknown, "B44: -INF with itself");
  CHECK(rel(R::Equal, Float{kInfF}, Double{kInf}) == Tri::Unknown, "B44: float INF against double INF");
  CHECK(rel(R::Less, Double{-kInf}, Double{kInf}) == Tri::True,
        "BuiltinFunctions.html: \"-INF is taken as less than any finite number or INF\"");
  CHECK(rel(R::Greater, Double{kInf}, Double{1e308}) == Tri::True,
        "BuiltinFunctions.html: \"INF is taken as greater than any finite number\"");
  // NaN
  CHECK(rel(R::Equal, Double{kNaN}, Double{1.0}) == Tri::Unknown &&
            rel(R::NotEqual, Double{kNaN}, Double{1.0}) == Tri::Unknown,
        "D9 (extrapolation): NaN never silently compares false");
  CHECK(rel(R::Equal, Double{kNaN}, Double{kNaN}) == Tri::Unknown, "D9: NaN against NaN");
  // string × number: equality as text, either order (D2)
  CHECK(rel(R::Equal, str("3"), Integer{3}) == Tri::True && rel(R::Equal, Integer{3}, str("3")) == Tri::True,
        "D2 / B18: string × number equality is text, either order");
  CHECK(rel(R::Equal, str("3.0"), Integer{3}) == Tri::False && rel(R::Equal, Integer{3}, str("3.0")) == Tri::False,
        "D2: the integer's text is \"3\"");
  CHECK(rel(R::Equal, str("2.0"), Double{2.0}) == Tri::True && rel(R::Equal, str("2"), Double{2.0}) == Tri::False,
        "D2 / Q1: Double.toString(2.0) is \"2.0\"");
  CHECK(rel(R::Equal, str("0.1"), Float{0.1f}) == Tri::True, "D2 / Q1: Float.toString(0.1f) is \"0.1\"");
  CHECK(rel(R::NotEqual, Integer{4}, str("3")) == Tri::True, "D2 notEqual");
  // string × number ordering: no common type
  CHECK(!cold::compare(kCold, Purpose::Ordering, str("10"), Integer{3}, scratch).has_value(),
        "§2.7: DataDictionary.html \"Categorical field values can only be tested for equality\" (E-TYPE-ORDER-MIXED)");
  bool threw = false;
  try {
    rel(R::Less, str("10"), Integer{3});
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw, "no silent fallback: an ordering the binder refuses is not answered at run time");
  // strings
  CHECK(rel(R::Equal, Value{String{pool.intern("ab")}}, str("ab")) == Tri::True, "§2.3 text equality");
  CHECK(rel(R::Less, str("B"), str("a")) == Tri::True && rel(R::Less, str("a"), str("ab")) == Tri::True,
        "§2.7 string ordering by code units");
  CHECK(rel(R::Greater, str("\xC3\xA9"), str("z")) == Tri::True, "UTF-8 code units order as code points (é > z)");
  // booleans
  CHECK(rel(R::Less, Boolean{false}, Boolean{true}) == Tri::True, "§2.7 boolean: false < true (XSD)");
  CHECK(!cold::compare(kCold, Purpose::Equality, Boolean{true}, Integer{1}, scratch).has_value(),
        "Q7: Functions.html \"boolean dataTypes can not be mixed with other types\"");
  CHECK(!cold::compare(kCold, Purpose::Equality, str("true"), Boolean{true}, scratch).has_value(), "Q7");
  // missing
  CHECK(rel(R::Equal, Missing{}, Integer{1}) == Tri::Unknown && rel(R::Equal, Integer{1}, Missing{}) == Tri::Unknown,
        "BuiltinFunctions.html \"any missing inputs to a built-in function will result in a missing value\"");
  CHECK(rel(R::Equal, Invalid{InvalidReason::Domain}, Integer{1}) == Tri::Unknown,
        "a sentinel operand gives no answer");
  // the binder's question
  CHECK(comparable(Purpose::Equality, TypeId::String, TypeId::Integer) &&
            !comparable(Purpose::Ordering, TypeId::String, TypeId::Integer),
        "§2.7 string × number: equality yes, ordering no");
  CHECK(comparable(Purpose::Ordering, TypeId::Integer, TypeId::Double) &&
            comparable(Purpose::Ordering, TypeId::String, TypeId::String) &&
            !comparable(Purpose::Equality, TypeId::Boolean, TypeId::Integer) &&
            comparable(Purpose::Ordering, TypeId::Boolean, TypeId::Boolean),
        "§2.7 table");
  // typed (hot path)
  CHECK(numeric_order(Integer{2}, Double{2.5}) == std::partial_ordering::less, "§2.7 typed numeric order");
  CHECK(relate(Relation::Equal, numeric_order(Double{kInf}, Double{kInf})) == Tri::Unknown, "B44 typed");
}

// ============================================================ §2.5 / §2.8 arithmetic
static void test_arithmetic() {
  auto plus = [](const Value& a, const Value& b) { return cold::arithmetic<Plus>(kCold, a, b); };
  auto minus = [](const Value& a, const Value& b) { return cold::arithmetic<Minus>(kCold, a, b); };
  auto times = [](const Value& a, const Value& b) { return cold::arithmetic<Times>(kCold, a, b); };
  auto divide = [](const Value& a, const Value& b) { return cold::arithmetic<Divide>(kCold, a, b); };
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();

  CHECK(is_double(plus(Integer{1}, Double{0.5}), 1.5),
        "Functions.html: \"if an integer and a double parameter are used with the \"+\" function ... the output "
        "dataType will be a double\"");
  CHECK(is_float(plus(Integer{1}, Float{0.5f}), 1.5f), "§2.5: integer + float is float");
  CHECK(is_integer(plus(Integer{2}, Integer{3}), 5), "§2.5: integer + integer is integer");
  CHECK(is_float(times(Float{0.1f}, Float{3.0f}), 0.1f * 3.0f), "§2.5: float arithmetic in float");
  CHECK(invalid_because(plus(Integer{kMax}, Integer{1}), InvalidReason::Overflow),
        "§2.5 / D4: integer overflow is invalid");
  CHECK(invalid_because(minus(Integer{kMin}, Integer{1}), InvalidReason::Overflow), "D4");
  CHECK(invalid_because(times(Integer{kMax}, Integer{2}), InvalidReason::Overflow), "D4");
  // division
  CHECK(is_double(divide(Double{1}, Double{0}), kInf),
        "BuiltinFunctions.html: \"If a positive number is divided by zero, INF will be returned\"");
  CHECK(is_double(divide(Double{-1}, Double{0}), -kInf),
        "BuiltinFunctions.html: \"If a negative number is divided by zero, -INF will be returned\"");
  CHECK(holds<Missing>(divide(Double{0}, Double{0})),
        "B21: \"If zero is divided by zero, the result is indeterminate and a missing value will be returned\"");
  CHECK(holds<Missing>(divide(Integer{0}, Integer{0})), "§4.2: integer 0/0 is missing (same sentence)");
  CHECK(invalid_because(divide(Integer{1}, Integer{0}), InvalidReason::DivisionByZero),
        "D20: integer x/0 is invalid (\"INF and -INF can only be floating point numbers\")");
  CHECK(invalid_because(divide(Integer{kMin}, Integer{-1}), InvalidReason::Overflow), "§4.2: INT64_MIN / -1");
  CHECK(is_integer(divide(Integer{-7}, Integer{2}), -3), "§4.2 note: integer division truncates toward zero (Java)");
  CHECK(is_double(divide(Integer{7}, Double{2}), 3.5), "§2.5: integer / double in double");
  CHECK(is_double(divide(Double{1}, Double{-0.0}), kInf) && is_float(divide(Float{1}, Float{-0.0f}), kInfF),
        "\"If a positive number is divided by zero, INF will be returned\"; XSD §3.2.4: \"only one zero\" (not "
        "IEEE's -INF)");
  CHECK(is_double(divide(Double{-1}, Double{-0.0}), -kInf),
        "\"If a negative number is divided by zero, -INF will be returned\", for a -0.0 divisor too");
  CHECK(is_double(divide(Double{kInf}, Double{-0.0}), kInf) && is_double(divide(Double{-kInf}, Double{-0.0}), -kInf),
        "\"An infinite number divided by a finite number is infinite\", signed by the numerator over one zero");
  CHECK(holds<Missing>(divide(Double{-0.0}, Double{0})) && holds<Missing>(divide(Double{0}, Double{-0.0})),
        "\"If zero is divided by zero ... a missing value will be returned\", either sign");
  // NaN
  CHECK(is_double(plus(Double{kNaN}, Double{1}), kNaN),
        "D23: Transformations.html \"All arithmetic expressions involving NaN will evaluate to NaN\"");
  CHECK(is_double(times(Double{kNaN}, Double{kInf}), kNaN), "D23: NaN before the infinity rules");
  CHECK(is_double(divide(Double{kNaN}, Double{0}), kNaN), "D23");
  // infinities: Transformations.html "A Brief Summary of Operations Involving Infinite Numbers"
  CHECK(is_double(plus(Double{kInf}, Double{1}), kInf),
        "\"The sum of an infinite number and a finite number is infinite\"");
  CHECK(is_double(plus(Double{kInf}, Double{kInf}), kInf),
        "\"The sum of two infinite numbers of the same sign is infinite\"");
  CHECK(holds<Missing>(plus(Double{kInf}, Double{-kInf})),
        "\"The sum of two infinite numbers of opposite signs is indeterminate and will return a missing value\"");
  CHECK(holds<Missing>(minus(Double{kInf}, Double{kInf})),
        "\"The difference between two infinite numbers of the same sign is indeterminate\"");
  CHECK(is_double(minus(Double{-kInf}, Double{kInf}), -kInf),
        "\"The difference between two infinite numbers of opposite signs is infinite\"");
  CHECK(holds<Missing>(times(Double{kInf}, Double{0})) && holds<Missing>(times(Double{-0.0}, Double{-kInf})),
        "\"The product of an infinite number and zero is indeterminate\"");
  CHECK(is_double(times(Double{-kInf}, Double{2}), -kInf), "\"sign being determined by the signs of the factors\"");
  CHECK(holds<Missing>(divide(Double{kInf}, Double{kInf})) && holds<Missing>(divide(Double{-kInf}, Double{-kInf})),
        "\"An infinite number divided by another of the same sign is indeterminate\"");
  CHECK(
      is_double(divide(Double{kInf}, Double{-kInf}), kInf) && is_double(divide(Double{-kInf}, Double{kInf}), -kInf),
      "\"An infinite number divided by another of the opposite sign is infinite with the same sign as the numerator\"");
  CHECK(is_double(divide(Double{-kInf}, Double{2}), -kInf),
        "\"An infinite number divided by a finite number is infinite\"");
  CHECK(holds<Missing>(plus(Float{kInfF}, Double{-kInf})), "infinity rules in the promoted type");
  // sentinels
  CHECK(holds<Missing>(plus(Missing{}, Integer{1})) && holds<Missing>(plus(Integer{1}, Missing{})),
        "BuiltinFunctions.html: \"any missing inputs ... will result in a missing value\" (D24)");
  CHECK(invalid_because(plus(Invalid{InvalidReason::Domain}, Integer{1}), InvalidReason::Domain) &&
            invalid_because(plus(Double{1}, Invalid{InvalidReason::Domain}), InvalidReason::Domain),
        "BuiltinFunctions.html: \"An invalid value is not missing\"");
  CHECK(holds<Missing>(plus(Invalid{InvalidReason::Domain}, Missing{})) &&
            holds<Missing>(plus(Missing{}, Invalid{InvalidReason::Domain})),
        "\"any missing inputs ... will result in a missing value\": missing first, the §4.3 Apply wrapper's order");
  bool threw = false;
  try {
    plus(Boolean{true}, Integer{1});
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw, "Q7 / §2.5: no arithmetic over a boolean; not a silent answer");
  // floor, ceil
  auto floor = [](const Value& x) { return cold::unary<Floor>(kCold, x); };
  auto ceil = [](const Value& x) { return cold::unary<Ceil>(kCold, x); };
  CHECK(is_integer(floor(Double{-2.5}), -3) && is_integer(ceil(Double{-2.5}), -2) && is_integer(floor(Float{2.5f}), 2),
        "BuiltinFunctions.html / §4.2: floor and ceil return an integer");
  CHECK(is_integer(floor(Integer{7}), 7), "floor of an integer");
  CHECK(invalid_because(floor(Double{kInf}), InvalidReason::Domain) &&
            invalid_because(ceil(Double{-kInf}), InvalidReason::Domain),
        "B21: \"INF and -INF ... will cause floor, ceil, round, and modulo to return an invalid value\"");
  CHECK(invalid_because(floor(Double{kNaN}), InvalidReason::Domain), "§4.2: floor(NaN) is invalid");
  CHECK(invalid_because(ceil(Double{1e19}), InvalidReason::Overflow), "D4: out of int64 range");
  CHECK(holds<Missing>(floor(Missing{})), "D24");
  // typed (hot path)
  CHECK(is_double(Divide::apply(Double{1}, Double{4}), 0.25), "§4.1: kernels on scalar types");
}

// ============================================================ §2.9 Java text
static void test_java_text() {
  StringPool pool;
  auto text = [](const Value& v) {
    std::string out;
    return cold::java_text(kCold, v, out) ? out : std::string("<none>");
  };
  CHECK(text(Integer{7}) == "7" && text(Integer{-3}) == "-3", "§2.9 Integer: Long.toString");
  CHECK(text(Integer{std::numeric_limits<std::int64_t>::min()}) == "-9223372036854775808",
        "Long.toString(Long.MIN_VALUE)");
  CHECK(text(Float{0.1f}) == "0.1", "§2.9 Float: Float.toString(0.1f) = 0.1");
  CHECK(text(Double{static_cast<double>(0.1f)}) == "0.10000000149011612", "Double.toString((double) 0.1f)");
  CHECK(text(Double{1e7}) == "1.0E7" && text(Double{2.0}) == "2.0" && text(Double{-0.0}) == "-0.0",
        "§2.9 Double: Double.toString");
  CHECK(text(Double{kInf}) == "Infinity" && text(Double{kNaN}) == "NaN", "Double.toString of INF, NaN");
  CHECK(text(Float{std::numeric_limits<float>::denorm_min()}) == "1.4E-45", "Float.toString(Float.MIN_VALUE)");
  CHECK(text(Boolean{true}) == "true" && text(Boolean{false}) == "false", "§2.9 Boolean");
  CHECK(text(String{pool.intern("a b")}) == "a b", "§2.9 String: its text");
  CHECK(text(Missing{}) == "<none>" && text(Invalid{InvalidReason::Malformed}) == "<none>",
        "a sentinel has no text; the caller decides, nothing is written silently");
}

// ============================================================ §2.7 hash containers
static void test_keys() {
  StringPool pool;
  RecordArena arena;
  std::unordered_map<Value, int, KeyHash, KeyEqual> table;
  table[Double{0.0}] = 1;
  table[Double{kInf}] = 2;
  table[Double{kNaN}] = 3;
  const Value pooled = String{pool.intern("k")};
  table[pooled] = 4;
  const char* id = "§2.7 KeyEqual: same alternative, same payload; XSD \"only one zero\"";
  CHECK(table.count(Double{-0.0}) == 1 && table.at(Double{-0.0}) == 1, id);
  CHECK(table.count(Double{kInf}) == 1, "§2.7: KeyEqual is reflexive for INF (unlike the comparison)");
  CHECK(table.count(Double{kNaN}) == 1 && table.size() == 4, "XSD: \"for schema purposes NaN = NaN\" (reflexive)");
  CHECK(table.count(Value{String{arena.make("k")}}) == 1,
        "§2.3: a pooled and a runtime string of one text are one key");
  CHECK(table.count(Integer{0}) == 0,
        "§2.7: Integer 0 and Double 0 are different alternatives (keys are cast at load)");
  CHECK(KeyHash{}(Double{-0.0}) == KeyHash{}(Double{0.0}) && KeyHash{}(Float{-0.0f}) == KeyHash{}(Float{0.0f}),
        "§2.7: -0.0 and 0.0 hash alike");
  CHECK(KeyEqual{}(Float{-0.0f}, Float{0.0f}), "typed KeyEqual");
  std::unordered_set<Double, KeyHash, KeyEqual> typed{Double{1.0}, Double{-0.0}};
  CHECK(typed.count(Double{0.0}) == 1, "§2.7: typed container keys");

  // A non-finite probe is decided before the lookup, and agrees with the
  // member-by-member comparison (isIn as the OR of equal).
  std::string scratch;
  auto linear = [&](const Value& probe, const std::vector<Value>& keys) {
    bool unknown = false;
    for (const Value& k : keys) {
      const Tri t = cold::relate(kCold, Relation::Equal, probe, k, scratch);
      if (t == Tri::True) return Lookup::Found;
      if (t == Tri::Unknown) unknown = true;
    }
    return unknown ? Lookup::Indeterminate : Lookup::NotFound;
  };
  auto hashed = [&](const Value& probe, const std::vector<Value>& keys) {
    NonFiniteKeys flags;
    std::unordered_set<Value, KeyHash, KeyEqual> set;
    for (const Value& k : keys) {
      flags.note(k);
      set.insert(k);
    }
    if (const auto decided = decide_nonfinite(probe, flags)) return *decided;
    return set.count(probe) ? Lookup::Found : Lookup::NotFound;
  };
  const std::vector<std::vector<Value>> key_sets = {
      {}, {Double{1}, Double{2}}, {Double{kInf}, Double{1}}, {Double{-kInf}}, {Double{kInf}, Double{-kInf}, Double{0}}};
  const std::vector<Value> probes = {Double{1},     Double{0},    Double{-0.0}, Double{kInf},
                                     Double{-kInf}, Double{kNaN}, Double{3}};
  bool agree = true;
  for (const auto& keys : key_sets)
    for (const Value& p : probes)
      if (linear(p, keys) != hashed(p, keys)) agree = false;
  CHECK(agree, "§2.7: the hashed and the linear path give the same answers (INF probe, NaN probe, -0.0)");
  NonFiniteKeys inf_keys;
  inf_keys.note(Double{kInf});
  CHECK(decide_nonfinite(Double{kInf}, inf_keys) == Lookup::Indeterminate &&
            decide_nonfinite(Double{-kInf}, inf_keys) == Lookup::NotFound && !decide_nonfinite(Double{1}, inf_keys),
        "§2.7: indeterminate if the container holds that infinity, not found otherwise");
}

int main() {
  test_promotion();
  test_strings();
  test_parse();
  test_casts();
  test_compare();
  test_arithmetic();
  test_java_text();
  test_keys();
  if (failures) {
    std::cerr << failures << " of " << checks << " checks failed" << std::endl;
    return 1;
  }
  std::cout << "typed_value: " << checks << " checks passed" << std::endl;
  return 0;
}
