// Standalone sketch of the typed core for cPMML (docs/design/typed-core.md).
// Not part of the library. Compiles with -std=c++20 on Apple clang 21,
// g++-16 and em++ (see build.sh next to it).
#pragma once

#include <bit>
#include <cassert>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory_resource>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

namespace pmml {

// ---------------------------------------------------------------- type ids
// One id per PMML dataType (DataDictionary.html DATATYPE) plus the two
// sentinels. The order is the variant's alternative order.
enum class TypeId : std::uint8_t {
  Missing,
  Invalid,
  Integer,
  Float,
  Double,
  Boolean,
  String,
  Date,
  Time,
  DateTime,
  DateDaysSince0,
  DateDaysSince1960,
  DateDaysSince1970,
  DateDaysSince1980,
  TimeSeconds,
  DateTimeSecondsSince0,
  DateTimeSecondsSince1960,
  DateTimeSecondsSince1970,
  DateTimeSecondsSince1980,
};

// ---------------------------------------------------------------- strings
// Text is the identity of a String. The pool interns every string the
// document names at load; equal pool ids mean equal text, and that is only a
// shortcut: two strings without a common id compare by text.
struct StringRep {
  std::string_view text;
  std::uint32_t id;  // kUnpooled for strings made while scoring
  std::size_t hash;  // std::hash of text, computed once
  static constexpr std::uint32_t kUnpooled = std::numeric_limits<std::uint32_t>::max();
};

class StringPool {  // per model, frozen after load: read-only while scoring
 public:
  explicit StringPool(std::pmr::memory_resource* upstream = std::pmr::get_default_resource()) : arena_(upstream) {}
  const StringRep* intern(std::string_view text) {
    if (auto it = index_.find(std::string(text)); it != index_.end()) return it->second;
    return index_.emplace(std::string(text), make(text, next_id_++)).first->second;
  }
  // Read-only lookup used on inputs: the pooled rep, or nullptr.
  const StringRep* find(std::string_view text) const {
    auto it = index_.find(std::string(text));
    return it == index_.end() ? nullptr : it->second;
  }
  const StringRep* make(std::string_view text, std::uint32_t id) {
    char* bytes = static_cast<char*>(arena_.allocate(text.size() + 1, 1));
    std::memcpy(bytes, text.data(), text.size());
    bytes[text.size()] = '\0';
    void* where = arena_.allocate(sizeof(StringRep), alignof(StringRep));
    return ::new (where) StringRep{std::string_view(bytes, text.size()), id, std::hash<std::string_view>{}(text)};
  }

 private:
  std::pmr::monotonic_buffer_resource arena_;
  std::unordered_map<std::string, const StringRep*> index_;  // sketch; the library uses a transparent hash
  std::uint32_t next_id_ = 0;
};

// Per record: strings computed while scoring (uppercase, concat, casts to
// string) live here until the Result has copied what it returns.
class RecordArena {
 public:
  const StringRep* make(std::string_view text) {
    char* bytes = static_cast<char*>(arena_.allocate(text.size() + 1, 1));
    std::memcpy(bytes, text.data(), text.size());
    void* where = arena_.allocate(sizeof(StringRep), alignof(StringRep));
    return ::new (where)
        StringRep{std::string_view(bytes, text.size()), StringRep::kUnpooled, std::hash<std::string_view>{}(text)};
  }
  void reset() { arena_.release(); }

 private:
  std::byte initial_[1024];
  std::pmr::monotonic_buffer_resource arena_{initial_, sizeof(initial_)};
};

// ---------------------------------------------------------------- the types
enum class InvalidReason : std::uint8_t { Malformed, NotIntegral, Overflow, DivisionByZero, Domain, NoCast };

struct Missing {
  static constexpr TypeId type_id = TypeId::Missing;
};
struct Invalid {
  static constexpr TypeId type_id = TypeId::Invalid;
  InvalidReason reason;
};

// Every PMML value type is a strong type over its XSD value space.
struct Integer {  // xs:integer, held in 64 bits; beyond that is Invalid(Overflow)
  static constexpr TypeId type_id = TypeId::Integer;
  static constexpr std::string_view name = "integer";
  static constexpr int rank = 0;  // Functions.html precedence: integer < float < double < string
  std::int64_t v;
  double as_double() const { return static_cast<double>(v); }
};
struct Float {  // xs:float: an IEEE single, never a double that "should be" one
  static constexpr TypeId type_id = TypeId::Float;
  static constexpr std::string_view name = "float";
  static constexpr int rank = 1;
  float v;
  double as_double() const { return v; }
};
struct Double {  // xs:double
  static constexpr TypeId type_id = TypeId::Double;
  static constexpr std::string_view name = "double";
  static constexpr int rank = 2;
  double v;
  double as_double() const { return v; }
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
  std::string_view text() const { return rep->text; }
};
// The PMML epoch-based types are numbers of days or seconds, but each is its
// own type: dateDaysSince[1960] 0 is not dateDaysSince[1970] 0.
template <TypeId Id, int Year>
struct DaysSince {
  static constexpr TypeId type_id = Id;
  static constexpr int epoch_year = Year;
  std::int64_t days;
};
template <TypeId Id, int Year>
struct SecondsSince {
  static constexpr TypeId type_id = Id;
  static constexpr int epoch_year = Year;
  std::int64_t seconds;
};
struct TimeSeconds {
  static constexpr TypeId type_id = TypeId::TimeSeconds;
  std::int64_t seconds;
};
struct Date {  // xs:date, as days since 1970-01-01 (local time, DataDictionary.html)
  static constexpr TypeId type_id = TypeId::Date;
  std::int32_t days;
};
struct Time {  // xs:time, as nanoseconds since midnight
  static constexpr TypeId type_id = TypeId::Time;
  std::int64_t nanos;
};
struct DateTime {  // xs:dateTime, as nanoseconds since 1970-01-01T00:00
  static constexpr TypeId type_id = TypeId::DateTime;
  std::int64_t nanos;
};
using DateDaysSince0 = DaysSince<TypeId::DateDaysSince0, 0>;
using DateDaysSince1960 = DaysSince<TypeId::DateDaysSince1960, 1960>;
using DateDaysSince1970 = DaysSince<TypeId::DateDaysSince1970, 1970>;
using DateDaysSince1980 = DaysSince<TypeId::DateDaysSince1980, 1980>;
using DateTimeSecondsSince0 = SecondsSince<TypeId::DateTimeSecondsSince0, 0>;
using DateTimeSecondsSince1960 = SecondsSince<TypeId::DateTimeSecondsSince1960, 1960>;
using DateTimeSecondsSince1970 = SecondsSince<TypeId::DateTimeSecondsSince1970, 1970>;
using DateTimeSecondsSince1980 = SecondsSince<TypeId::DateTimeSecondsSince1980, 1980>;

// The temporal alternatives join in P8 (review O1); until then the variant has
// the seven alternatives below and every visitor is exhaustive over them.
#if PMML_WITH_TEMPORAL
using Value = std::variant<Missing, Invalid, Integer, Float, Double, Boolean, String, Date, Time, DateTime,
                           DateDaysSince0, DateDaysSince1960, DateDaysSince1970, DateDaysSince1980, TimeSeconds,
                           DateTimeSecondsSince0, DateTimeSecondsSince1960, DateTimeSecondsSince1970,
                           DateTimeSecondsSince1980>;
#else
using Value = std::variant<Missing, Invalid, Integer, Float, Double, Boolean, String>;
#endif

// The variant index is the TypeId.
template <class T>
constexpr bool index_matches() {
  constexpr Value probe{std::in_place_type<T>};
  return probe.index() == static_cast<std::size_t>(T::type_id);
}

// ---------------------------------------------------------------- per-type operations
// Strict XSD parse (lexical.h rules), Java text (java_format.h rules) and the
// comparison within one type. Free functions found by ADL so the concepts can
// require them.
struct ParseContext {
  StringPool* pool = nullptr;            // load: intern document text
  const StringPool* frozen = nullptr;    // scoring: the model's pool, read-only
  RecordArena* arena = nullptr;          // scoring: text the pool does not hold
};

std::optional<Integer> parse(std::type_identity<Integer>, std::string_view, ParseContext&);
std::optional<Float> parse(std::type_identity<Float>, std::string_view, ParseContext&);
std::optional<Double> parse(std::type_identity<Double>, std::string_view, ParseContext&);
std::optional<Boolean> parse(std::type_identity<Boolean>, std::string_view, ParseContext&);
std::optional<String> parse(std::type_identity<String>, std::string_view, ParseContext&);

void java_text(const Integer& x, std::string& out);
void java_text(const Float& x, std::string& out);
void java_text(const Double& x, std::string& out);
void java_text(const Boolean& x, std::string& out);
void java_text(const String& x, std::string& out);

inline std::partial_ordering compare(Integer a, Integer b) { return a.v <=> b.v; }
inline std::partial_ordering compare(Float a, Float b) { return a.v <=> b.v; }
inline std::partial_ordering compare(Double a, Double b) { return a.v <=> b.v; }
inline std::partial_ordering compare(Boolean a, Boolean b) { return a.v <=> b.v; }
inline std::partial_ordering compare(String a, String b) {
  if (a.rep == b.rep) return std::partial_ordering::equivalent;  // the pool shortcut
  return a.text().compare(b.text()) <=> 0;                        // code-unit order
}
// Equality only. One pool per document (review S10): two pooled reps are equal
// iff they are the same pointer, so no text is read; otherwise compare text.
inline bool string_equal(String a, String b) {
  if (a.rep == b.rep) return true;
  if (a.rep->id != StringRep::kUnpooled && b.rep->id != StringRep::kUnpooled) return false;
  return a.text() == b.text();
}
template <class T>
  requires requires(T t) { t.days; }
std::partial_ordering compare(T a, T b) {
  return a.days <=> b.days;
}
template <class T>
  requires requires(T t) { t.seconds; }
std::partial_ordering compare(T a, T b) {
  return a.seconds <=> b.seconds;
}
inline std::partial_ordering compare(Date a, Date b) { return a.days <=> b.days; }
inline std::partial_ordering compare(Time a, Time b) { return a.nanos <=> b.nanos; }
inline std::partial_ordering compare(DateTime a, DateTime b) { return a.nanos <=> b.nanos; }

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
template <class T>
concept Temporal = PmmlType<T> && !PmmlNumeric<T> && !Textual<T> && !Logical<T> && !Sentinel<T>;

// Every alternative is in exactly one category. A new alternative that is in
// none makes this fail -- and so do the visitors below, which have no
// unconstrained catch-all.
template <class T>
constexpr int categories = int(Sentinel<T>) + int(PmmlNumeric<T>) + int(Textual<T>) + int(Logical<T>) +
                           int(Temporal<T>);
template <class... T>
constexpr bool all_categorised(std::variant<T...>*) {
  return ((categories<T> == 1) && ...);
}
static_assert(all_categorised(static_cast<Value*>(nullptr)));

// ---------------------------------------------------------------- promotion
// Functions.html: "When the input parameters have multiple dataTypes, the
// least restrictive dataType will be inherited ... string, double, float,
// integer". The numeric part of that list is one table, by rank.
template <PmmlNumeric A, PmmlNumeric B>
using Promoted = std::conditional_t<(A::rank >= B::rank), A, B>;

static_assert(std::is_same_v<Promoted<Integer, Float>, Float>);
static_assert(std::is_same_v<Promoted<Float, Double>, Double>);
static_assert(std::is_same_v<Promoted<Integer, Integer>, Integer>);

template <PmmlNumeric To, PmmlNumeric From>
  requires(To::rank >= From::rank)  // implicit casts only toward less restriction
constexpr To widen(From x) {
  if constexpr (std::is_same_v<To, Integer>)
    return x;
  else if constexpr (std::is_same_v<To, Float>)
    return Float{static_cast<float>(x.v)};
  else
    return Double{x.as_double()};
}

// ---------------------------------------------------------------- casts
// cast<To>(x): the explicit conversions a Cast node performs (DerivedField,
// OutputField, Constant, ParameterField-like positions). A conversion the
// spec and XPath casting both lack is Invalid(NoCast); the bind pass refuses
// those at load, so it is reached only through a value-dependent cast.
template <class To>
struct CastTo {
  ParseContext* ctx;
  std::string* scratch;

  Value operator()(Missing) const { return Missing{}; }
  Value operator()(Invalid i) const { return i; }

  // numeric -> numeric
  template <PmmlNumeric From>
    requires PmmlNumeric<To>
  Value operator()(From x) const {
    if constexpr (To::rank >= From::rank) {
      return widen<To>(x);
    } else if constexpr (std::is_same_v<To, Float>) {  // double -> float: IEEE nearest
      return Float{static_cast<float>(x.v)};
    } else {  // float/double -> integer: only an integral value in range
      const double d = x.as_double();
      if (!std::isfinite(d) || std::trunc(d) != d) return Invalid{InvalidReason::NotIntegral};
      if (d < -9223372036854775808.0 || d >= 9223372036854775808.0) return Invalid{InvalidReason::Overflow};
      return Integer{static_cast<std::int64_t>(d)};
    }
  }
  // number -> string: the number's text (open question Q3 in the design doc)
  template <PmmlNumeric From>
    requires Textual<To>
  Value operator()(From x) const {
    scratch->clear();
    java_text(x, *scratch);
    return String{ctx->arena->make(*scratch)};
  }
  // string -> number or boolean: the strict XSD reading of the text
  template <Textual From>
    requires(PmmlNumeric<To> || Logical<To>)
  Value operator()(From x) const {
    if (auto parsed = parse(std::type_identity<To>{}, x.text(), *ctx)) return *parsed;
    return Invalid{InvalidReason::Malformed};
  }
  template <Textual From>
    requires Textual<To>
  Value operator()(From x) const {
    return x;
  }
  // boolean <-> number: through the XSD boolean lexical space {1, 0}
  template <Logical From>
    requires PmmlNumeric<To>
  Value operator()(From x) const {
    return CastTo<To>{ctx, scratch}(Integer{x.v ? 1 : 0});
  }
  template <PmmlNumeric From>
    requires Logical<To>
  Value operator()(From x) const {
    const double d = x.as_double();
    if (d == 1.0) return Boolean{true};
    if (d == 0.0) return Boolean{false};
    return Invalid{InvalidReason::NoCast};
  }
  template <Logical From>
    requires Logical<To>
  Value operator()(From x) const {
    return x;
  }
  template <Logical From>
    requires Textual<To>
  Value operator()(From x) const {
    return String{ctx->arena->make(x.v ? "true" : "false")};
  }
  // temporal: same type only in this sketch (phase 8 adds the date casts)
  template <Temporal From>
  Value operator()(From x) const {
    if constexpr (std::is_same_v<From, To>)
      return x;
    else
      return Invalid{InvalidReason::NoCast};
  }
  template <class From>
    requires(!Temporal<From> && !Sentinel<From> && Temporal<To>)
  Value operator()(From) const {
    return Invalid{InvalidReason::NoCast};
  }
};

// ---------------------------------------------------------------- comparison
// One visitor gives the ordering; equal, notEqual, lessThan ... and the
// predicates, set lookups and MapValues keys all derive from it, so equal and
// lessOrEqual cannot disagree.
//   nullopt   no common type (refused at load; Missing/Invalid operands)
//   unordered NaN, or INF against itself (BuiltinFunctions.html: "any attempt
//             to [compare] either with itself is indeterminate")
struct Comparison {
  ParseContext* ctx;
  std::string* scratch;

  template <class A, class B>
    requires(Sentinel<A> || Sentinel<B>)
  std::optional<std::partial_ordering> operator()(const A&, const B&) const {
    return std::nullopt;
  }
  template <PmmlNumeric A, PmmlNumeric B>
  std::optional<std::partial_ordering> operator()(A a, B b) const {
    using C = Promoted<A, B>;
    const C x = widen<C>(a), y = widen<C>(b);
    if constexpr (!std::is_same_v<C, Integer>)
      if (std::isinf(x.v) && x.v == y.v) return std::partial_ordering::unordered;
    return compare(x, y);
  }
  // string against number: the number becomes text ("string" is the least
  // restrictive type), whichever side it is on
  template <Textual A, PmmlNumeric B>
  std::optional<std::partial_ordering> operator()(A a, B b) const {
    Value t = CastTo<String>{ctx, scratch}(b);
    return compare(a, std::get<String>(t));
  }
  template <PmmlNumeric A, Textual B>
  std::optional<std::partial_ordering> operator()(A a, B b) const {
    auto r = (*this)(b, a);
    if (!r) return r;
    return 0 <=> *r;  // swap
  }
  template <class A, class B>
    requires(!Sentinel<A> && !Sentinel<B> && !(PmmlNumeric<A> && PmmlNumeric<B>) &&
             !(Textual<A> && PmmlNumeric<B>) && !(PmmlNumeric<A> && Textual<B>))
  std::optional<std::partial_ordering> operator()(const A& a, const B& b) const {
    if constexpr (std::is_same_v<A, B>)
      return compare(a, b);
    else
      return std::nullopt;  // boolean vs number, date vs number, ...: no common type
  }
};

inline std::optional<std::partial_ordering> compare_values(const Value& a, const Value& b, ParseContext& ctx,
                                                           std::string& scratch) {
  return std::visit(Comparison{&ctx, &scratch}, a, b);
}

// ---------------------------------------------------------------- arithmetic
// One template per operator over the promotion table. The bind pass has
// already refused non-numeric operands; the sentinel and non-numeric
// overloads are there so that the visit is exhaustive.
struct Plus {
  static Value apply(Integer a, Integer b) {
    std::int64_t r;
    if (__builtin_add_overflow(a.v, b.v, &r)) return Invalid{InvalidReason::Overflow};
    return Integer{r};
  }
  template <class T>
    requires std::floating_point<decltype(T::v)>
  static Value apply(T a, T b) {
    return T{a.v + b.v};
  }
};
struct Divide {
  // BuiltinFunctions.html, Division by Zero: positive/0 is INF, negative/0
  // is -INF, 0/0 is missing. An integer has no INF, so x/0 of integers is
  // invalid ("INF and -INF can only be floating point numbers").
  static Value apply(Integer a, Integer b) {
    if (b.v == 0) return a.v == 0 ? Value{Missing{}} : Value{Invalid{InvalidReason::DivisionByZero}};
    if (a.v == std::numeric_limits<std::int64_t>::min() && b.v == -1) return Invalid{InvalidReason::Overflow};
    return Integer{a.v / b.v};  // truncation toward zero (Java, XQuery idiv)
  }
  template <class T>
    requires std::floating_point<decltype(T::v)>
  static Value apply(T a, T b) {
    if (b.v == 0 && a.v == 0) return Missing{};
    return T{a.v / b.v};
  }
};

template <class Op>
struct Arithmetic {
  template <class A, class B>
    requires(Sentinel<A> || Sentinel<B>)
  Value operator()(const A& a, const B& b) const {
    if constexpr (std::is_same_v<A, Invalid>)
      return a;
    else if constexpr (std::is_same_v<B, Invalid>)
      return b;
    else
      return Missing{};
  }
  template <PmmlNumeric A, PmmlNumeric B>
  Value operator()(A a, B b) const {
    using C = Promoted<A, B>;
    return Op::apply(widen<C>(a), widen<C>(b));
  }
  template <class A, class B>
    requires(!Sentinel<A> && !Sentinel<B> && !(PmmlNumeric<A> && PmmlNumeric<B>))
  Value operator()(const A&, const B&) const {
    return Invalid{InvalidReason::NoCast};  // unreachable after binding
  }
};

inline Value plus(const Value& a, const Value& b) { return std::visit(Arithmetic<Plus>{}, a, b); }
inline Value divide(const Value& a, const Value& b) { return std::visit(Arithmetic<Divide>{}, a, b); }

// ---------------------------------------------------------------- Java text
struct JavaText {
  std::string* out;
  void operator()(Missing) const {}
  void operator()(Invalid) const {}
  template <PmmlParsedType T>
  void operator()(const T& x) const {
    java_text(x, *out);
  }
  template <Temporal T>
  void operator()(const T&) const { /* phase 8: ISO text or the day/second count */ }
};

// ---------------------------------------------------------------- container keys
// Hash containers (MapValues tables, large sets, NaiveBayes counts,
// distributions) need a reflexive equality, which the comparison visitor is
// not (INF vs INF and NaN are unordered). Keys are of one type (the binder
// casts them at load); NaN keys are refused at load; -0.0 and 0.0 are one key.
struct KeyHash {
  std::size_t operator()(const Value& v) const {
    return std::visit(
        [](const auto& x) -> std::size_t {
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, String>) return x.rep->hash;
          else if constexpr (std::is_same_v<T, Double> || std::is_same_v<T, Float>) return std::hash<double>{}(x.v == 0 ? 0.0 : double(x.v));
          else if constexpr (std::is_same_v<T, Integer>) return std::hash<std::int64_t>{}(x.v);
          else if constexpr (std::is_same_v<T, Boolean>) return x.v;
          else return 0;  // Missing/Invalid/temporal: not keys before P8
        },
        v);
  }
};
struct KeyEqual {  // reflexive: same alternative, same value
  bool operator()(const Value& a, const Value& b) const {
    if (a.index() != b.index()) return false;
    return std::visit(
        [&](const auto& x) -> bool {
          using T = std::decay_t<decltype(x)>;
          const T& y = std::get<T>(b);
          if constexpr (std::is_same_v<T, String>) return string_equal(x, y);
          else if constexpr (requires { x.v; }) return x.v == y.v;
          else return true;
        },
        a);
  }
};
// A probe that compares unordered with a key (±INF against the same INF, NaN)
// is decided before the hash lookup, so the hashed and linear paths agree:
// "indeterminate" if the set holds that infinity, "not found" otherwise.
enum class Lookup : std::uint8_t { Found, NotFound, Indeterminate };
inline bool is_nonfinite(const Value& v) {
  if (auto d = std::get_if<Double>(&v)) return !std::isfinite(d->v);
  if (auto f = std::get_if<Float>(&v)) return !std::isfinite(f->v);
  return false;
}

// ---------------------------------------------------------------- the bound fast path
// What the bind pass produces for `SimplePredicate field op constant` when it
// has proved the slot holds Double or Missing: no double dispatch.
enum class Tri : std::uint8_t { False, True, Unknown };
struct DoubleLessOrEqual {
  std::uint32_t slot;
  double constant;
  Tri operator()(const Value* frame) const {
    const Value& v = frame[slot];
    if (const Double* d = std::get_if<Double>(&v)) return d->v <= constant ? Tri::True : Tri::False;
    // A bound Double slot holds Double or Missing. Anything else is a binder
    // bug; in a debug build it stops here instead of taking a wrong branch.
    assert(std::holds_alternative<Missing>(v) && "slot type disagrees with its binding");
    return Tri::Unknown;
  }
};

}  // namespace pmml
