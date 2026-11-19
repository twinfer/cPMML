/*******************************************************************************
 * The P3 accessors of the engine's Value and the adapters between it and the
 * P1 typed value (src/core/value.h; docs/design/typed-core.md §7, P3):
 * as_double(), is_missing(), text() and the setters read and write the fields
 * they name; unpack<T>(v, static_type) reads the old storage as a value of the
 * type the binder fixed, and pack(x) is the old storage of a typed value.
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

#include "core/indexer.h"
#include "core/types/string_pool.h"
#include "core/value.h"

namespace t = cpmml::types;
using t::TypeId;

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
  const std::string rule = "§7 P3: each accessor is the field it names";
  static_assert(std::is_same_v<decltype(std::declval<const Value&>().text()), const std::string&>,
                "text() returns the field, not a copy");
  static_assert(std::is_same_v<decltype(std::declval<const Value&>().as_double()), double>);
  static_assert(std::is_same_v<decltype(std::declval<const Value&>().is_missing()), bool>);

  const Value v(2.5);
  CHECK(v.as_double() == 2.5 && !v.is_missing() && v.text().empty() && !v.has_numeric_text(), rule);
  const Value missing;
  CHECK(missing.is_missing() && missing.text().empty(), rule);
  const Value s("abc", DataType::DataTypeValue::STRING);
  CHECK(s.text() == "abc" && !s.is_missing(), rule);

  Value w;
  const std::string& text = w.text();
  w.set_double(-0.0);
  CHECK(w.as_double() == 0.0 && std::signbit(w.as_double()) && w.is_missing(),
        rule + " (set_double writes the double only)");
  w.set_missing(false);
  CHECK(!w.is_missing() && std::signbit(w.as_double()), rule + " (set_missing writes the flag only)");
  w.set_text("abc");
  CHECK(text == "abc" && &w.text() == &text && !w.has_numeric_text(),
        rule + " (set_text writes the text only; text() is a reference to the stored text, not a copy)");
  w.mark_numeric_text();
  CHECK(w.has_numeric_text() && text == "abc", rule + " (mark_numeric_text)");
  w.clear_text();
  CHECK(text.empty() && w.has_numeric_text(), rule + " (clear_text writes the text only, as svalue.clear() did)");
  w.set_missing(true);
  CHECK(w.is_missing() && std::signbit(w.as_double()), rule);
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
  CHECK(cpmml::pack(z) == Value(-0.0) && !std::signbit(cpmml::pack(z).as_double()),
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
  // pack: an Integer beyond 2^53 is the nearest double, as the engine holds it.
  CHECK(cpmml::pack(t::Integer{9007199254740993}).as_double() == 9007199254740992.0,
        "pack(Integer) is its double (round to nearest even)");
}

static void booleans(t::ParseContext& ctx) {
  const std::string rule =
      "§2.6 number -> boolean: 1 -> true, 0 -> false, else Invalid(NoCast) (XSD boolean lexical space {true, false, "
      "1, 0}, D14); boolean -> number: true 1, false 0";
  const Value yes(1.0, DataType::DataTypeValue::BOOLEAN);
  const Value no(0.0, DataType::DataTypeValue::BOOLEAN);
  CHECK(std::get<t::Boolean>(cpmml::unpack<t::Value>(yes, TypeId::Boolean)).v, rule);
  CHECK(!std::get<t::Boolean>(cpmml::unpack<t::Value>(no, TypeId::Boolean)).v, rule);
  CHECK(cpmml::unpack<t::Boolean>(yes, TypeId::Boolean).v && !cpmml::unpack<t::Boolean>(no, TypeId::Boolean).v,
        rule + " (scalar form)");
  CHECK(round_trips(yes, TypeId::Boolean, ctx) && round_trips(no, TypeId::Boolean, ctx), rule);
  CHECK(same_fields(cpmml::pack(t::Boolean{true}), Value(1.0)) && same_fields(cpmml::pack(t::Boolean{false}), Value(0.0)),
        rule);
  CHECK(!std::get<t::Boolean>(cpmml::unpack<t::Value>(Value(-0.0), TypeId::Boolean)).v, rule + " (-0 == 0)");
  CHECK(invalid_because(cpmml::unpack<t::Value>(Value(2.0), TypeId::Boolean), t::InvalidReason::NoCast), rule);
  CHECK(invalid_because(cpmml::unpack<t::Value>(Value(kNaN), TypeId::Boolean), t::InvalidReason::NoCast), rule);
}

static void strings(Indexer& indexer, t::ParseContext& ctx) {
  const std::string rule =
      "§2.3/§9: a String is the text (string readers get the text); pack is Value(text, STRING), its double the "
      "active Indexer's code of the text";
  const std::vector<std::string> texts = {"abc", "", " 1", "say \"hi\"", "\xc3\xa9t\xc3\xa9", "5", "true"};
  for (const std::string& text : texts) {
    const Value v(text, DataType::DataTypeValue::STRING);
    const t::Value x = cpmml::unpack<t::Value>(v, TypeId::String, ctx);
    const auto* sx = std::get_if<t::String>(&x);
    CHECK(sx && sx->text() == text, rule);
    CHECK(round_trips(v, TypeId::String, ctx), rule);
    CHECK(cpmml::unpack<t::String>(v, TypeId::String, ctx).text() == text, rule + " (scalar form)");
    CHECK(cpmml::pack(x).as_double() == indexer.get_string_value(text), rule);
  }
  CHECK(throws<std::logic_error>([] { cpmml::unpack<t::Value>(Value("abc", DataType::DataTypeValue::STRING), TypeId::String); }),
        "a String needs a ParseContext for its rep: the two-argument form refuses it");

  // A numeric_text value (§9, decided after P2.1) reads as its text; the mark
  // is the old storage's, so pack gives the text's code and no mark.
  Value marked("5", DataType::DataTypeValue::STRING);
  marked.mark_numeric_text();
  CHECK(cpmml::unpack<t::String>(marked, TypeId::String, ctx).text() == "5", rule + " (numeric_text: the text)");
  const Value repacked = cpmml::pack(cpmml::unpack<t::Value>(marked, TypeId::String, ctx));
  CHECK(repacked == marked && repacked.text() == "5" && !repacked.has_numeric_text(),
        "numeric_text does not survive pack: documented in core/value.h");

  // B2: uppercase() edits the text in place and keeps the argument's code;
  // unpack reads the text, so pack gives the new text's code.
  Value up("abc", DataType::DataTypeValue::STRING);
  up.uppercase();
  CHECK(cpmml::unpack<t::String>(up, TypeId::String, ctx).text() == "ABC", rule + " (an edited string: its text)");
  CHECK(cpmml::pack(cpmml::unpack<t::Value>(up, TypeId::String, ctx)).as_double() == indexer.get_string_value("ABC"),
        "B2 (P6): pack gives the edited text's own code, not the argument's");

  // A frozen Indexer gives every text it never saw the code -1.
  indexer.freeze();
  const Value unseen("never seen", DataType::DataTypeValue::STRING);
  CHECK(unseen.as_double() == -1.0 && round_trips(unseen, TypeId::String, ctx),
        rule + " (a text the frozen Indexer never saw keeps code -1)");
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
  Indexer indexer;
  const Value::ScopedIndexer scope(&indexer);
  t::RecordArena arena;
  t::ParseContext ctx{nullptr, nullptr, &arena};

  accessors();
  missing_values(ctx);
  doubles(ctx);
  floats(ctx);
  integers(ctx);
  booleans(ctx);
  strings(indexer, ctx);
  misuse();

  std::cout << checks - failures << "/" << checks << " checks passed" << std::endl;
  return failures == 0 ? 0 : 1;
}
