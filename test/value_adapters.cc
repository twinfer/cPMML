/*******************************************************************************
 * The engine's Value (src/core/value.h; docs/design/typed-core.md §7, P3 and
 * P5): a types::Value read by the P3 accessors as the old storage was read,
 * the document StringPool and the per-evaluation arena (StringScope), the two
 * compatibility switches, and the adapters: unpack<T>(v, static_type) reads a
 * value as the type the binder fixed (itself when it is stored in that
 * type), and pack(x) is the Value of a typed value.
 * pack(unpack(v, t)) is v for every value the engine builds as a value of t,
 * for every P1 scalar type and missing. Expected values come from the XSD
 * value spaces and the design's cast table (§2.6), never from cPMML.
 *
 * Usage: value_adapters.exe   (exit 0 when every check holds)
 *******************************************************************************/

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/compare.h"
#include "core/types/string_pool.h"
#include "core/value.h"

namespace t = cpmml::types;
using t::TypeId;

// The engine's typed comparisons (core/compare.h, P6.3) over two values as
// they are stored.
struct TypedCompare {
  static t::Tri rel(t::Relation r, const Value& a, const Value& b) {
    const auto x = cpmml::compare::relate(r, a.typed(), b.typed(), nullptr);
    if (!x) throw std::logic_error("TypedCompare: no common type");
    return *x;
  }
  static bool equal(const Value& a, const Value& b) { return rel(t::Relation::Equal, a, b) == t::Tri::True; }
  static bool not_equal(const Value& a, const Value& b) { return rel(t::Relation::NotEqual, a, b) == t::Tri::True; }
  static bool less_than(const Value& a, const Value& b) { return rel(t::Relation::Less, a, b) == t::Tri::True; }
  static bool less_or_equal(const Value& a, const Value& b) {
    return rel(t::Relation::LessOrEqual, a, b) == t::Tri::True;
  }
};

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
static bool throws(F&& f) {
  try {
    f();
  } catch (const E&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

// The call throws E with a message containing `what`.
template <class E, class F>
static bool throws_saying(F&& f, const std::string& what) {
  try {
    f();
  } catch (const E& e) {
    return std::string(e.what()).find(what) != std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

static const double kInf = std::numeric_limits<double>::infinity();
static const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Field for field: the double bit for bit (so -0 and NaN count), the flag, the
// text and the numeric_text mark.
static bool same_fields(const Value& a, const Value& b) {
  return a.is_missing() == b.is_missing() &&
         std::bit_cast<std::uint64_t>(a.as_double()) == std::bit_cast<std::uint64_t>(b.as_double()) &&
         a.text() == b.text() && a.has_numeric_text() == b.has_numeric_text();
}

static bool round_trips(const Value& v, TypeId type, t::ParseContext& ctx) {
  return same_fields(cpmml::pack(cpmml::unpack<t::Value>(v, type, ctx)), v);
}

static bool invalid_because(const t::Value& v, t::InvalidReason r) {
  const auto* i = std::get_if<t::Invalid>(&v);
  return i && i->reason == r;
}

static void accessors() {
  const std::string rule = "§7 P5: a Value is a types::Value; the accessors read it as the old storage was read";
  static_assert(std::is_same_v<decltype(std::declval<const Value&>().text()), std::string_view>,
                "text() is a view of the rep's text: a Value owns no string");
  static_assert(std::is_same_v<decltype(std::declval<const Value&>().as_double()), double>);
  static_assert(std::is_same_v<decltype(std::declval<const Value&>().is_missing()), bool>);
  static_assert(std::is_trivially_copyable_v<Value> && sizeof(Value) <= 32,
                "§6.2: a slot is trivially copyable (a Sample is copied with memcpy)");

  const Value v(2.5);
  CHECK(v.as_double() == 2.5 && !v.is_missing() && v.text().empty() && !v.has_numeric_text(), rule);
  CHECK(std::holds_alternative<t::Double>(v.typed()), rule + " (a number is a Double)");
  const Value missing;
  CHECK(missing.is_missing() && missing.text().empty() && std::holds_alternative<t::Missing>(missing.typed()), rule);
  CHECK(missing.as_double() == double_min(), rule + " (a missing value reads as double_min(), as Value() did)");
  const Value s = Value::string("abc");
  CHECK(s.text() == "abc" && !s.is_missing() && std::holds_alternative<t::String>(s.typed()), rule);

  // set_double writes the number and keeps the text a value carries; a
  // String becomes that number with its text (the old storage wrote its
  // double only).
  Value w(1.0);
  w.set_double(-0.0);
  CHECK(w.as_double() == 0.0 && std::signbit(w.as_double()) && w.text().empty(), rule + " (set_double, -0 kept)");
  w.set_text("abc");
  CHECK(w.text() == "abc" && std::signbit(w.as_double()) && !w.has_numeric_text(),
        rule + " (set_text gives a number a text and leaves the number)");
  w.mark_numeric_text();
  CHECK(w.has_numeric_text() && w.text() == "abc", rule + " (mark_numeric_text)");
  w.set_double(4.0);
  CHECK(w.as_double() == 4.0 && w.text() == "abc" && w.has_numeric_text(), rule + " (set_double keeps the text)");
  Value sw = Value::string("xyz");
  const double code = sw.as_double();
  sw.set_double(7.0);
  CHECK(sw.as_double() == 7.0 && sw.text() == "xyz" && std::holds_alternative<t::Double>(sw.typed()),
        rule + " (set_double on a string: the number, the text kept)");
  Value st = Value::string("xyz");
  st.set_text("other");
  CHECK(st.text() == "other" && st.as_double() != code && std::holds_alternative<t::String>(st.typed()) &&
            std::get<t::String>(st.typed()).rep == std::get<t::String>(Value::string("other").typed()).rep,
        "P6.1 (B2): set_text on a string makes it the string of its new text (its canonical rep and code)");
  CHECK(throws<std::logic_error>([] {
          Value m;
          m.set_double(1.0);
        }),
        "set_double on a missing value is a programming error (no hidden number)");
  CHECK(throws<std::logic_error>([] {
          Value m;
          m.set_text("x");
        }),
        "set_text on a missing value is a programming error");
}

// The typed slots: inputs and document text are parsed into typed values
// (§7 P5), read back by as_double() exactly as the old storage's double.
static void typed_storage() {
  const std::string rule = "§7 P5: integer, boolean and string values are stored typed";
  CHECK(std::get<t::Integer>(Value::integer_slot(7.0).typed()).v == 7, rule);
  CHECK(std::get<t::Integer>(Value::integer_slot(-9007199254740992.0).typed()).v == -9007199254740992, rule);
  CHECK(std::holds_alternative<t::Double>(Value::integer_slot(2.5).typed()),
        rule + " (an integer field's 2.5, K1 rule 2: the double)");
  CHECK(std::holds_alternative<t::Double>(Value::integer_slot(9223372036854775808.0).typed()),
        rule + " (beyond int64: the double the old storage held)");
  const Value negative_zero = Value::integer_slot(-0.0);
  CHECK(std::holds_alternative<t::Double>(negative_zero.typed()) && std::signbit(negative_zero.as_double()),
        rule + " (-0 stays the double -0, as the old storage held it; Q13/D18 is P6's)");
  CHECK(Value::integer_slot(1e15).as_double() == 1e15, rule);

  const Value b = Value::document("true", DataType::DataTypeValue::BOOLEAN);
  CHECK(std::get<t::Boolean>(b.typed()).v && b.as_double() == 1.0, rule + " (a boolean reads as 1)");
  const Value f = Value::document("0", DataType::DataTypeValue::BOOLEAN);
  CHECK(!std::get<t::Boolean>(f.typed()).v && f.as_double() == 0.0, rule + " (false reads as 0)");
  CHECK(throws<cpmml::ParsingException>([] { Value::document("TRUE", DataType::DataTypeValue::BOOLEAN); }),
        "XSD boolean lexical space {true, false, 1, 0}: TRUE is malformed document text");
  CHECK(std::get<t::Integer>(Value::document("4.0", DataType::DataTypeValue::INTEGER).typed()).v == 4,
        rule + " (K1: an integer field's 4.0 is 4)");
  CHECK(std::holds_alternative<t::Double>(Value::document("2.5", DataType::DataTypeValue::INTEGER).typed()), rule);
  CHECK(Value::document("1e1", DataType::DataTypeValue::DOUBLE).as_double() == 10.0, rule);
  CHECK(throws<cpmml::ParsingException>([] { Value::document("x", DataType::DataTypeValue::DOUBLE); }),
        "malformed number text is a ParsingException (a load error)");
}

// The compatibility switches (design P5), one place each; switch 1 is gone
// (P6.2).
static void switches() {
  // P6.2 deleted switch 1: a float is a Float. XML Schema Part 2 §3.2.4:
  // "float is patterned after the IEEE single-precision 32-bit floating
  // point type"; a decimal "maps to the normalized value in the value space
  // of float that is closest to d"; XML Schema 1.1 E.1 floatLexicalMap:
  // floatingPointRound "may nonetheless return zero, or INF or -INF".
  const std::string f_rule = "P6.2: float text is the float closest to it (XSD 3.2.4)";
  const Value f = Value::document("0.1", DataType::DataTypeValue::FLOAT);
  CHECK(std::holds_alternative<t::Float>(f.typed()) && f.as_double() == 0.100000001490116119384765625, f_rule);
  CHECK(Value::document("3.4e39", DataType::DataTypeValue::FLOAT).as_double() ==
            std::numeric_limits<double>::infinity(),
        f_rule + " (beyond the float range: INF)");
  CHECK(Value::document("1e-50", DataType::DataTypeValue::FLOAT).as_double() == 0.0, f_rule + " (underflow: 0)");
  CHECK(Value::document("16777217", DataType::DataTypeValue::FLOAT).as_double() == 16777216.0,
        f_rule + " (2^24 + 1 is no float: the even neighbour)");
  // 0.500000029802322387695312500 is the exact midpoint of the floats 0.5 and
  // 0.50000006; one more digit is above it. Read through a double first, the
  // second literal would round to the midpoint and then to even (0.5): the
  // decimal is rounded once, to the float.
  CHECK(Value::document("0.5000000298023223876953125000001", DataType::DataTypeValue::FLOAT).as_double() ==
            0.500000059604644775390625,
        f_rule + " (rounded once, not through a double)");
  CHECK(throws<cpmml::ParsingException>([] { Value::document("x", DataType::DataTypeValue::FLOAT); }),
        "malformed float text is a ParsingException (a load error)");
  CHECK(std::holds_alternative<t::Float>(cpmml::pack(t::Float{0.1f}).typed()) &&
            cpmml::pack(t::Float{0.1f}).as_double() == 0.100000001490116119384765625,
        "P6.2: pack(Float) is the Float");
  CHECK(::java_text(Value(t::Float{0.1f})) == "0.1", "P6.2: a Float is written as Float.toString writes it");

  // P6.1: two strings compare by text (pooled strings by pointer, any other
  // pair by its texts); an edit is the string of its new text. P6.3 deleted
  // LegacyCompare: a string against a number compares the number's Java text
  // (D2), never the string's code (B18), and strings are ordered by their
  // code units (core/compare.h).
  using C = TypedCompare;
  const Value a = Value::string("abc");
  const Value a2 = Value::string("abc");
  const Value z = Value::string("zzz");
  CHECK(std::get<t::String>(a.typed()).rep == std::get<t::String>(a2.typed()).rep && C::equal(a, a2),
        "switch 2: one pooled rep per text, equal");
  CHECK(!C::equal(a, z) && C::not_equal(a, z), "switch 2: two pooled texts differ");
  Value up = a;
  up.uppercase();
  CHECK(up.text() == "ABC" && !C::equal(up, a) && C::equal(up, Value::string("ABC")) &&
            up.as_double() == Value::string("ABC").as_double(),
        "P6.1 (B2): uppercase gives the string ABC (its code), which equals \"ABC\", not its argument");
  CHECK(!C::equal(Value(cpmml::legacy::code_of(a)), a) && C::not_equal(Value(cpmml::legacy::code_of(a)), a),
        "P6.3 (B18): a string is not the number of its code");
  CHECK(C::equal(Value(2.5), Value::string("2.5")) && C::equal(Value::string("2.5"), Value(2.5)) &&
            !C::equal(Value(3.0), Value::string("3")) && C::equal(Value(t::Integer{3}), Value::string("3")),
        "P6.3 (D2): a number against a string is its Java text, in either order");
  const Value b0 = Value::string("aab, after zzz");  // pooled after zzz: a larger code, a smaller text
  CHECK(C::less_than(b0, z) && !C::less_than(z, b0) && b0.as_double() > z.as_double(),
        "P6.3: strings are ordered by their code units, not by code");
}

static void missing_values(t::ParseContext& ctx) {
  const std::string rule = "§2.8: a missing value is Missing whatever its static type; pack(Missing) is Value()";
  for (const TypeId type : {TypeId::Integer, TypeId::Float, TypeId::Double, TypeId::Boolean, TypeId::String}) {
    CHECK(t::is_missing(cpmml::unpack<t::Value>(Value(), type, ctx)), rule);
    CHECK(round_trips(Value(), type, ctx), rule);
  }
  CHECK(t::is_missing(cpmml::unpack<t::Value>(Value(), TypeId::Double)), rule + " (no ParseContext needed)");
  CHECK(same_fields(cpmml::pack(t::Missing{}), Value()), rule);
  CHECK(throws<std::logic_error>([] { cpmml::unpack<t::Double>(Value(), TypeId::Double); }),
        "a scalar cannot hold Missing: unpack<Double> of a missing value is a programming error");
}

static void doubles(t::ParseContext& ctx) {
  const std::string rule = "§2.6 T -> T identity: a double slot's value is the double, -0, NaN and INF kept";
  const double values[] = {0.0, -0.0, 1.5, -2.25, 0.1, kInf, -kInf, kNaN, std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::denorm_min(), -1e300};
  for (const double d : values) {
    const Value v(d);
    const t::Value x = cpmml::unpack<t::Value>(v, TypeId::Double);
    const auto* dx = std::get_if<t::Double>(&x);
    CHECK(dx && std::bit_cast<std::uint64_t>(dx->v) == std::bit_cast<std::uint64_t>(d), rule);
    CHECK(round_trips(v, TypeId::Double, ctx), rule);
    CHECK(std::bit_cast<std::uint64_t>(cpmml::unpack<t::Double>(v, TypeId::Double).v) ==
              std::bit_cast<std::uint64_t>(d),
          rule + " (scalar form)");
  }
  CHECK(throws_saying<std::logic_error>([] { cpmml::unpack<t::Double>(Value(1.0), TypeId::Float); }, "static type"),
        "unpack<Scalar> with another static type is a programming error");
  CHECK(throws_saying<std::logic_error>([] { cpmml::unpack<t::Integer>(Value(1.0), TypeId::Double); }, "static type"),
        "unpack<Scalar> with another static type is a programming error, even where the double would read");
}

static void floats(t::ParseContext& ctx) {
  const std::string rule = "§2.6 double -> float: IEEE round to nearest; overflow -> ±INF (XSD float value space)";
  const float values[] = {0.0f, -0.0f, 1.5f, 0.1f, std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
  for (const float f : values) {
    const Value v(static_cast<double>(f));
    const t::Value x = cpmml::unpack<t::Value>(v, TypeId::Float);
    const auto* fx = std::get_if<t::Float>(&x);
    CHECK(fx && std::bit_cast<std::uint32_t>(fx->v) == std::bit_cast<std::uint32_t>(f), rule);
    CHECK(round_trips(v, TypeId::Float, ctx), rule + " (a float of a float slot round trips)");
  }
  // A double that is not a float (the engine keeps float inputs as their
  // xs:double reading, B4) reads as its nearest float.
  CHECK(cpmml::unpack<t::Float>(Value(0.1), TypeId::Float).v == 0.1f, rule);
  CHECK(cpmml::unpack<t::Float>(Value(1e300), TypeId::Float).v == std::numeric_limits<float>::infinity(),
        rule + " (overflow -> INF)");
  CHECK(cpmml::unpack<t::Float>(Value(1e-300), TypeId::Float).v == 0.0f, rule + " (underflow -> 0)");
}

static void integers(t::ParseContext& ctx) {
  const std::string rule =
      "§2.6 double -> integer: the value if integral and in int64 range (XSD integer; Functions.html \"a double "
      "expression can not be implicitly cast as an integer\")";
  const std::int64_t values[] = {0,
                                 1,
                                 -1,
                                 7,
                                 9007199254740992,        // 2^53
                                 -9007199254740992,       // -2^53
                                 4611686018427387904,     // 2^62
                                 9223372036854774784,     // the largest double below 2^63
                                 INT64_MIN};              // -2^63
  for (const std::int64_t i : values) {
    const Value v(static_cast<double>(i));
    const t::Value x = cpmml::unpack<t::Value>(v, TypeId::Integer);
    const auto* ix = std::get_if<t::Integer>(&x);
    CHECK(ix && ix->v == i, rule);
    CHECK(round_trips(v, TypeId::Integer, ctx), rule);
    CHECK(cpmml::unpack<t::Integer>(v, TypeId::Integer).v == i, rule + " (scalar form)");
  }
  // XSD integer has one zero: -0 in an integer slot is 0, and packs as +0.
  const t::Value z = cpmml::unpack<t::Value>(Value(-0.0), TypeId::Integer);
  CHECK(std::get_if<t::Integer>(&z) && std::get<t::Integer>(z).v == 0, rule + " (-0 is the integer 0)");
  CHECK(cpmml::pack(z).as_double() == 0.0 && !std::signbit(cpmml::pack(z).as_double()),
        rule + " (pack(unpack(-0)) == -0 as numbers; the integer has no sign of zero)");

  CHECK(invalid_because(cpmml::unpack<t::Value>(Value(2.5), TypeId::Integer), t::InvalidReason::NotIntegral), rule);
  CHECK(invalid_because(cpmml::unpack<t::Value>(Value(kNaN), TypeId::Integer), t::InvalidReason::NotIntegral), rule);
  CHECK(invalid_because(cpmml::unpack<t::Value>(Value(kInf), TypeId::Integer), t::InvalidReason::Overflow), rule);
  CHECK(invalid_because(cpmml::unpack<t::Value>(Value(-kInf), TypeId::Integer), t::InvalidReason::Overflow), rule);
  CHECK(invalid_because(cpmml::unpack<t::Value>(Value(9223372036854775808.0), TypeId::Integer),
                        t::InvalidReason::Overflow),
        rule + " (2^63 is beyond int64, D4)");
  CHECK(throws<std::logic_error>([] { cpmml::unpack<t::Integer>(Value(2.5), TypeId::Integer); }),
        "a scalar cannot hold Invalid: unpack<Integer> of 2.5 is a programming error");
  // pack: an Integer is itself, and reads as its nearest double, as the
  // engine read every integer.
  CHECK(std::get<t::Integer>(cpmml::pack(t::Integer{9007199254740993}).typed()).v == 9007199254740993 &&
            cpmml::pack(t::Integer{9007199254740993}).as_double() == 9007199254740992.0,
        "pack(Integer) is the Integer; as_double() its double (round to nearest even)");
  // unpack of a value stored as its type is that value (std::get).
  const Value big(t::Integer{9007199254740993});
  CHECK(std::get<t::Integer>(cpmml::unpack<t::Value>(big, TypeId::Integer)).v == 9007199254740993,
        "unpack of a value stored as an Integer is that Integer (std::get), not the cast of its double");
}

static void booleans(t::ParseContext& ctx) {
  const std::string rule =
      "§2.6 number -> boolean: 1 -> true, 0 -> false, else Invalid(NoCast) (XSD boolean lexical space {true, false, "
      "1, 0}, D14); boolean -> number: true 1, false 0";
  const Value yes(1.0);
  const Value no(0.0);
  CHECK(std::get<t::Boolean>(cpmml::unpack<t::Value>(yes, TypeId::Boolean)).v, rule);
  CHECK(!std::get<t::Boolean>(cpmml::unpack<t::Value>(no, TypeId::Boolean)).v, rule);
  CHECK(cpmml::unpack<t::Boolean>(yes, TypeId::Boolean).v && !cpmml::unpack<t::Boolean>(no, TypeId::Boolean).v,
        rule + " (scalar form)");
  CHECK(round_trips(yes, TypeId::Boolean, ctx) && round_trips(no, TypeId::Boolean, ctx), rule);
  CHECK(same_fields(cpmml::pack(t::Boolean{true}), Value(1.0)) && same_fields(cpmml::pack(t::Boolean{false}), Value(0.0)),
        rule);
  CHECK(std::get<t::Boolean>(cpmml::pack(t::Boolean{true}).typed()).v, rule + " (pack(Boolean) is the Boolean)");
  CHECK(round_trips(Value(t::Boolean{true}), TypeId::Boolean, ctx), rule);
  CHECK(!std::get<t::Boolean>(cpmml::unpack<t::Value>(Value(-0.0), TypeId::Boolean)).v, rule + " (-0 == 0)");
  CHECK(invalid_because(cpmml::unpack<t::Value>(Value(2.0), TypeId::Boolean), t::InvalidReason::NoCast), rule);
  CHECK(invalid_because(cpmml::unpack<t::Value>(Value(kNaN), TypeId::Boolean), t::InvalidReason::NoCast), rule);
}

static void strings(cpmml::DocumentStrings& document, t::ParseContext& ctx) {
  const std::string rule =
      "§2.3/§9: a String is the text (string readers get the text); pack is Value::string(text), its code the "
      "document pool's id of the text";
  const std::vector<std::string> texts = {"abc", "", " 1", "say \"hi\"", "\xc3\xa9t\xc3\xa9", "5", "true"};
  for (const std::string& text : texts) {
    const Value v = Value::string(text);
    const t::Value x = cpmml::unpack<t::Value>(v, TypeId::String, ctx);
    const auto* sx = std::get_if<t::String>(&x);
    CHECK(sx && sx->text() == text && sx->rep == std::get<t::String>(v.typed()).rep,
          rule + " (unpack of a String is the String itself)");
    CHECK(round_trips(v, TypeId::String, ctx), rule);
    CHECK(cpmml::unpack<t::String>(v, TypeId::String, ctx).text() == text, rule + " (scalar form)");
    CHECK(cpmml::pack(x).as_double() == static_cast<double>(document.pool.find(text)->id), rule);
  }
  CHECK(std::get<t::String>(cpmml::unpack<t::Value>(Value::string("abc"), TypeId::String)).text() == "abc",
        "a value stored as a String is itself: no ParseContext needed");
  CHECK(throws<std::logic_error>([] { cpmml::unpack<t::Value>(Value(5.0), TypeId::String); }),
        "a number in a string slot needs a ParseContext for its text's rep: the two-argument form refuses it");
  // A string of another pool (the binder reads document text with pools of
  // its own) packs as the document's string of its text.
  t::StringPool other;
  const t::String foreign{other.intern("abc")};
  CHECK(cpmml::pack(foreign).as_double() == static_cast<double>(document.pool.find("abc")->id),
        "pack re-pools a String by its text");

  // A numeric_text value (§9, decided after P2.1) reads as its text; the mark
  // is the old storage's, so pack gives the text's code and no mark.
  Value marked = Value::string("5");
  marked.mark_numeric_text();
  CHECK(cpmml::unpack<t::String>(marked, TypeId::String, ctx).text() == "5", rule + " (numeric_text: the text)");
  const Value repacked = cpmml::pack(cpmml::unpack<t::Value>(marked, TypeId::String, ctx));
  CHECK(TypedCompare::equal(repacked, marked) && repacked.text() == "5" &&
            !repacked.has_numeric_text(),
        "numeric_text does not survive pack: documented in core/value.h");

  // P6.1 (B2): uppercase() gives the string of its new text, the code pack
  // of that text gives, so the round trip keeps it.
  Value up = Value::string("abc");
  up.uppercase();
  CHECK(cpmml::unpack<t::String>(up, TypeId::String, ctx).text() == "ABC", rule + " (an edited string: its text)");
  CHECK(cpmml::pack(t::String{other.intern("ABC")}).as_double() == static_cast<double>(document.pool.find("ABC")->id),
        "B2 (P6): pack gives the edited text's own code, not the argument's");
  CHECK(up.as_double() == static_cast<double>(document.pool.find("ABC")->id) &&
            TypedCompare::equal(cpmml::pack(cpmml::unpack<t::Value>(up, TypeId::String, ctx)), up),
        "P6.1 (B2): the edit has the code of ABC, and pack(unpack) of it is equal to it");
}

// The document's pool replaces the Indexer's string map: ids are given in
// the order the load names the strings, a frozen pool is only read, every
// text it lacks gets the unseen code -1 and an arena rep of its own, which
// lives as long as the evaluation.
static void string_scopes() {
  const std::string rule = "§2.3, §7 P5: the document StringPool and the per-evaluation arena";
  auto document = std::make_shared<cpmml::DocumentStrings>();
  {
    const cpmml::StringScope load(cpmml::StringScope::Loading{}, document);
    CHECK(Value::string("first").as_double() == 0.0 && Value::string("second").as_double() == 1.0 &&
              Value::string("first").as_double() == 0.0,
          rule + " (codes in the order the load names the strings, as the Indexer gave them)");
    CHECK(cpmml::StringScope::loading() == document, rule + " (the document being loaded)");
    Value edited = Value::string("second");
    edited.uppercase();
    CHECK(edited.text() == "SECOND" && edited.as_double() == 2.0 && document->pool.find("SECOND") != nullptr,
          rule + " (P6.1: a string edited at load is the string of its text, interned as Value::string interns it)");
    Value first_up = Value::string("first");
    first_up.uppercase();
    first_up.lowercase();
    CHECK(first_up.as_double() == 0.0 &&
              std::get<t::String>(first_up.typed()).rep == std::get<t::String>(Value::string("first").typed()).rep,
          rule + " (P6.1: an edit back to a pooled text is that pooled string)");
    CHECK(Value::string("third").as_double() == 4.0, rule + " (FIRST was interned by the edit before it)");
    Value carried(7.0);
    carried.set_text("a text given at load");
    CHECK(carried.text() == "a text given at load" && document->pool.find("a text given at load") == nullptr &&
              Value::string("fourth").as_double() == 5.0,
          rule + " (a number's text set at load is not interned: the next string's code is unchanged)");
    Value marked = Value::string("5");
    marked.mark_numeric_text();
    marked.trim_blanks();
    CHECK(!marked.has_numeric_text() && marked.text() == "5", rule + " (an edit drops the numeric_text mark)");
  }
  CHECK(throws<std::logic_error>([] { Value::string("x"); }),
        "a string outside every StringScope is a programming error");
  CHECK(throws<std::logic_error>([] { cpmml::StringScope::loading(); }), "no document is being loaded");
  document->pool.freeze();
  std::string_view kept;
  const char* first_made = nullptr;
  {
    const cpmml::StringScope score(cpmml::StringScope::Scoring{}, *document);
    const Value seen = Value::string("second");
    const Value unseen = Value::string("never named");
    const Value unseen2 = Value::string("another");
    CHECK(seen.as_double() == 1.0 && unseen.as_double() == -1.0 && unseen2.as_double() == -1.0,
          rule + " (a frozen pool: a text it lacks has the unseen code -1)");
    using C = TypedCompare;
    CHECK(!C::equal(unseen, unseen2) && C::not_equal(unseen, unseen2) && C::equal(unseen, Value::string("never named")),
          rule + " (P6.1, B1: two strings the pool lacks are equal iff their texts are)");
    Value up_seen = Value::string("SECOND");
    up_seen.lowercase();
    CHECK(std::get<t::String>(up_seen.typed()).rep == std::get<t::String>(seen.typed()).rep && C::equal(up_seen, seen),
          rule + " (P6.1, B2: an edit while scoring to a pooled text is the pooled string)");
    Value up_unseen = Value::string("never named");
    up_unseen.uppercase();
    CHECK(up_unseen.as_double() == -1.0 && !C::equal(up_unseen, unseen) &&
              C::equal(up_unseen, Value::string("NEVER NAMED")) && !C::equal(up_unseen, seen),
          rule + " (P6.1, B2: an edit to a text the pool lacks is compared by that text)");
    CHECK(throws<std::logic_error>([&] { cpmml::ContainerKey::of_constant(unseen); }) &&
              cpmml::ContainerKey::of_constant(seen) == 1.0,
          rule + " (P6.1: a key is a pooled string, never one with the unseen code -1)");
    CHECK(!C::less_than(unseen, unseen2) && C::less_than(unseen2, unseen) && !C::less_than(unseen, unseen) &&
              C::less_or_equal(unseen, Value::string("never named")),
          rule + " (P6.3, L4: two strings the pool lacks are ordered by their texts, as they are compared)");
    CHECK(document->pool.find("never named") == nullptr, rule + " (scoring never interns)");
    first_made = unseen.text().data();
    CHECK(std::get<t::String>(unseen.typed()).rep->id == t::StringRep::kUnpooled && unseen.text() == "never named",
          rule + " (its rep is the arena's, with its own text)");
    {
      const cpmml::StringScope nested(cpmml::StringScope::Scoring{}, *document);
      kept = Value::string("made in the nested scope").text();
    }
    CHECK(kept == "made in the nested scope", rule + " (a nested scope of the document shares the record's arena)");
    const Value later = Value::string("made after the nested scope, long enough to overwrite");
    CHECK(unseen.text() == "never named" && kept == "made in the nested scope" &&
              later.text() == "made after the nested scope, long enough to overwrite",
          rule + " (the nested scope's end empties nothing: the record's strings stay)");
    auto other_document = std::make_shared<cpmml::DocumentStrings>();
    other_document->pool.freeze();
    {
      const cpmml::StringScope other(cpmml::StringScope::Scoring{}, *other_document);
      CHECK(Value::string("second").as_double() == -1.0, rule + " (another document's pool, another arena)");
    }
    CHECK(kept == "made in the nested scope", rule + " (another document's scope leaves this arena alone)");
  }
  {
    // The arena is emptied when the outermost scope ends; the next record's
    // strings reuse it.
    const cpmml::StringScope score(cpmml::StringScope::Scoring{}, *document);
    const Value again = Value::string("never named");
    CHECK(again.text() == "never named" && again.text().data() == first_made,
          rule + " (the outermost scope's arena is reset and reused per evaluation)");
  }
}

static void misuse() {
  CHECK(throws<std::logic_error>([] { cpmml::pack(t::Invalid{t::InvalidReason::Overflow}); }),
        "Invalid has no old storage: pack(Invalid) is a programming error");
  CHECK(throws<std::logic_error>([] { cpmml::unpack<t::Value>(Value(1.0), TypeId::Missing); }),
        "Missing is not a static type");
  CHECK(throws<std::logic_error>([] { cpmml::unpack<t::Value>(Value(1.0), TypeId::Invalid); }),
        "Invalid is not a static type");
}

int main() {
  string_scopes();

  auto document = std::make_shared<cpmml::DocumentStrings>();
  const cpmml::StringScope load(cpmml::StringScope::Loading{}, document);
  t::RecordArena arena;
  t::ParseContext ctx{nullptr, nullptr, &arena};

  accessors();
  typed_storage();
  switches();
  missing_values(ctx);
  doubles(ctx);
  floats(ctx);
  integers(ctx);
  booleans(ctx);
  strings(*document, ctx);
  misuse();

  std::cout << checks - failures << "/" << checks << " checks passed" << std::endl;
  return failures == 0 ? 0 : 1;
}
