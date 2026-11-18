/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TYPES_VALUE_H
#define CPMML_TYPES_VALUE_H

/**
 * Design note -- the typed value (docs/design/typed-core.md §2.1, §2.2).
 *
 * A PMML value is one alternative of a closed variant: Missing, Invalid, or a
 * value of one PMML dataType. DataDictionary.html: "The dataType reuses names
 * and semantics of atomic types in XML Schema. The representation of the
 * values is defined in XML Schema Part 2: Datatypes." So each alternative is
 * a strong type over its XSD value space:
 *
 *   Integer  xs:integer held in 64 bits; beyond is Invalid(Overflow) (D4)
 *   Float    xs:float, always an IEEE single (never a double that "should be")
 *   Double   xs:double
 *   Boolean  xs:boolean
 *   String   xs:string, a pointer to a StringRep; text is its identity (§2.3)
 *
 * Missing and Invalid are alternatives so that a function can return them
 * without an exception. BuiltinFunctions.html: "An invalid value is not
 * missing."
 *
 * The temporal types (date, time, dateTime, the *Since types) join in P10;
 * until then nothing here can hold one.
 *
 * Exhaustiveness. Every alternative is in exactly one category (Sentinel,
 * PmmlNumeric, Textual, Logical, Temporal); visitors are overload sets over
 * those categories with no unconstrained catch-all, so a new alternative
 * makes every visitor that forgets it fail to compile
 * (test/types_negative/).
 *
 * The per-type operations the concepts require -- parse() (parse.h) and
 * java_text() (java_text.h) -- are declared here, so the concepts see them in
 * every translation unit (a concept's satisfaction must not depend on what
 * was included before it), and defined in their own headers.
 *
 * Nothing in the engine includes this header yet (P1.1).
 */

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

namespace cpmml::types {

// One id per PMML dataType plus the two sentinels, in the variant's
// alternative order (index_matches below checks it).
enum class TypeId : std::uint8_t { Missing, Invalid, Integer, Float, Double, Boolean, String };

// Why a value is invalid. Carried for diagnostics; every reason is handled
// the same way (invalidValueTreatment, or an invalid record).
enum class InvalidReason : std::uint8_t { Malformed, NotIntegral, Overflow, DivisionByZero, Domain, NoCast };

// The text of a String. One per distinct text in a document's StringPool
// (id < kUnpooled), or one per string made while scoring (id == kUnpooled).
// hash is std::hash<std::string_view> of text, the same in pool and arena.
struct StringRep {
  std::string_view text;
  std::uint32_t id;
  std::size_t hash;
  static constexpr std::uint32_t kUnpooled = std::numeric_limits<std::uint32_t>::max();
};

struct Missing {
  static constexpr TypeId type_id = TypeId::Missing;
};

struct Invalid {
  static constexpr TypeId type_id = TypeId::Invalid;
  InvalidReason reason;
};

struct Integer {  // xs:integer, in 64 bits
  static constexpr TypeId type_id = TypeId::Integer;
  static constexpr std::string_view name = "integer";
  static constexpr int rank = 0;  // Functions.html precedence "string double float integer", numbers only
  std::int64_t v;
  constexpr double as_double() const { return static_cast<double>(v); }
};

struct Float {  // xs:float
  static constexpr TypeId type_id = TypeId::Float;
  static constexpr std::string_view name = "float";
  static constexpr int rank = 1;
  float v;
  constexpr double as_double() const { return v; }
};

struct Double {  // xs:double
  static constexpr TypeId type_id = TypeId::Double;
  static constexpr std::string_view name = "double";
  static constexpr int rank = 2;
  double v;
  constexpr double as_double() const { return v; }
};

struct Boolean {  // xs:boolean
  static constexpr TypeId type_id = TypeId::Boolean;
  static constexpr std::string_view name = "boolean";
  bool v;
};

struct String {  // xs:string
  static constexpr TypeId type_id = TypeId::String;
  static constexpr std::string_view name = "string";
  const StringRep* rep;
  constexpr std::string_view text() const { return rep->text; }
};

using Value = std::variant<Missing, Invalid, Integer, Float, Double, Boolean, String>;

constexpr TypeId type_of(const Value& v) { return static_cast<TypeId>(v.index()); }
constexpr bool is_missing(const Value& v) { return std::holds_alternative<Missing>(v); }
constexpr bool is_invalid(const Value& v) { return std::holds_alternative<Invalid>(v); }

// double -> float, IEEE round to nearest even; beyond the float range ±INF
// (§2.6 "overflow → ±INF, underflow → 0"). Spelled out because a C++ cast of
// a finite double outside float's range is undefined behaviour
// ([conv.double]), even where IEEE defines it.
constexpr float narrow_to_float(double d) {
  // FLT_MAX + half an ulp (2^103): from here on the nearest float is INF.
  constexpr double kHalfwayToInf = 340282356779733661637539395458142568448.0;  // 0x1.ffffffp127
  if (d >= kHalfwayToInf) return std::numeric_limits<float>::infinity();
  if (d <= -kHalfwayToInf) return -std::numeric_limits<float>::infinity();
  return static_cast<float>(d);
}

// The variant index is the TypeId.
template <class T>
constexpr bool index_matches() {
  constexpr Value probe{std::in_place_type<T>};
  return probe.index() == static_cast<std::size_t>(T::type_id);
}

// ---------------------------------------------------------------- within one type
// The order of two values of one type. Numbers: IEEE (NaN unordered). The INF
// rule of §2.7 is the comparison's (compare.h), not the type's. Strings:
// the pool shortcut, then code-unit order of the UTF-8 bytes (= Unicode code
// point order). Booleans: false < true.
constexpr std::partial_ordering compare(Missing, Missing) { return std::partial_ordering::equivalent; }
constexpr std::partial_ordering compare(Invalid a, Invalid b) { return a.reason <=> b.reason; }
constexpr std::partial_ordering compare(Integer a, Integer b) { return a.v <=> b.v; }
constexpr std::partial_ordering compare(Float a, Float b) { return a.v <=> b.v; }
constexpr std::partial_ordering compare(Double a, Double b) { return a.v <=> b.v; }
constexpr std::partial_ordering compare(Boolean a, Boolean b) { return a.v <=> b.v; }
constexpr std::partial_ordering compare(String a, String b) {
  if (a.rep == b.rep) return std::partial_ordering::equivalent;
  return a.text().compare(b.text()) <=> 0;
}

// String equality is equality of text. With one pool per document two pooled
// reps are equal exactly when they are the same pointer, so the common case
// reads no text (design §2.3, review S10).
constexpr bool string_equal(String a, String b) {
  if (a.rep == b.rep) return true;
  if (a.rep->id != StringRep::kUnpooled && b.rep->id != StringRep::kUnpooled) return false;
  return a.text() == b.text();
}

// ---------------------------------------------------------------- per-type operations
// Declared here, defined in parse.h and java_text.h (see the note above).
struct ParseContext;

inline std::optional<Integer> parse(std::type_identity<Integer>, std::string_view text, ParseContext& ctx);
inline std::optional<Float> parse(std::type_identity<Float>, std::string_view text, ParseContext& ctx);
inline std::optional<Double> parse(std::type_identity<Double>, std::string_view text, ParseContext& ctx);
inline std::optional<Boolean> parse(std::type_identity<Boolean>, std::string_view text, ParseContext& ctx);
inline std::optional<String> parse(std::type_identity<String>, std::string_view text, ParseContext& ctx);

inline void java_text(const Integer& x, std::string& out);
inline void java_text(const Float& x, std::string& out);
inline void java_text(const Double& x, std::string& out);
inline void java_text(const Boolean& x, std::string& out);
inline void java_text(const String& x, std::string& out);

// ---------------------------------------------------------------- concepts
template <class T>
concept PmmlType = requires(const T& a, const T& b) {
  { T::type_id } -> std::convertible_to<TypeId>;
  { compare(a, b) } -> std::same_as<std::partial_ordering>;
} && index_matches<T>();

template <class T>
concept PmmlParsedType =
    PmmlType<T> && requires(std::string_view text, ParseContext& ctx, const T& a, std::string& out) {
      { parse(std::type_identity<T>{}, text, ctx) } -> std::same_as<std::optional<T>>;
      { java_text(a, out) } -> std::same_as<void>;
      { T::name } -> std::convertible_to<std::string_view>;
    };

template <class T>
concept PmmlNumeric = PmmlParsedType<T> && requires(const T& a) {
  { a.as_double() } -> std::same_as<double>;
  { T::rank } -> std::convertible_to<int>;
};

template <class T>
concept Sentinel = std::same_as<T, Missing> || std::same_as<T, Invalid>;
template <class T>
concept Textual = std::same_as<T, String>;
template <class T>
concept Logical = std::same_as<T, Boolean>;
// Empty until P10.
template <class T>
concept Temporal = PmmlType<T> && !PmmlNumeric<T> && !Textual<T> && !Logical<T> && !Sentinel<T>;

// Every alternative is in exactly one category.
template <class T>
constexpr int categories =
    int(Sentinel<T>) + int(PmmlNumeric<T>) + int(Textual<T>) + int(Logical<T>) + int(Temporal<T>);
template <class... T>
constexpr bool all_categorised(std::variant<T...>*) {
  return ((categories<T> == 1) && ...);
}
static_assert(all_categorised(static_cast<Value*>(nullptr)), "every Value alternative is in exactly one category");

// A type in exactly one category: what visitor overloads are constrained by.
template <class T>
concept Alternative = (categories<T> == 1);

// A visitor that has an overload for every alternative of Value (unary), or
// for every pair of alternatives (binary).
template <class F, class V>
struct handles_every_alternative;
template <class F, class... T>
struct handles_every_alternative<F, std::variant<T...>>
    : std::bool_constant<(std::is_invocable_v<F, const T&> && ...)> {};
template <class F>
concept ExhaustiveVisitor = handles_every_alternative<F, Value>::value;

template <class F, class A, class V>
struct handles_every_second;
template <class F, class A, class... T>
struct handles_every_second<F, A, std::variant<T...>>
    : std::bool_constant<(std::is_invocable_v<F, const A&, const T&> && ...)> {};
template <class F, class V>
struct handles_every_pair;
template <class F, class... T>
struct handles_every_pair<F, std::variant<T...>>
    : std::bool_constant<(handles_every_second<F, T, Value>::value && ...)> {};
template <class F>
concept ExhaustiveBinaryVisitor = handles_every_pair<F, Value>::value;

// A type that is no alternative: a visitor callable with it has an
// unconstrained catch-all, which the design forbids (§2.2).
struct NotAnAlternative {};
template <class F>
concept NoCatchAll = !std::is_invocable_v<F, const NotAnAlternative&>;
template <class F>
concept NoBinaryCatchAll = !std::is_invocable_v<F, const NotAnAlternative&, const Missing&> &&
                           !std::is_invocable_v<F, const Double&, const NotAnAlternative&> &&
                           !std::is_invocable_v<F, const NotAnAlternative&, const NotAnAlternative&>;

// Frames are copied with memcpy (design §2.1, §6.2).
static_assert(sizeof(Value) == 16 && alignof(Value) == 8);
static_assert(std::is_trivially_copyable_v<Value> && std::is_trivially_destructible_v<Value>);

}  // namespace cpmml::types

#endif  // CPMML_TYPES_VALUE_H
