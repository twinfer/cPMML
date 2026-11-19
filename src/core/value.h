
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_VALUE_H
#define CPMML_VALUE_H

#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <regex>

#include "core/types/cast.h"
#include "core/types/parse.h"
#include "core/types/value.h"
#include "datatype.h"
#include "indexer.h"
#include "utils/utils.h"

/**
 * @class Value
 *
 * Internal representation of each value used by the model. Numeric types are
 * stored as double for uniform comparison and arithmetic. String types also
 * carry the original text in svalue and are assigned a stable sequential index
 * (via string_converter) so that equality and set-membership comparisons work
 * correctly through the double field. Lookups into string_converter use
 * std::string_view (C++20 heterogeneous lookup) to avoid temporary allocations.
 */
class Value {
  // The storage is private (design typed-core.md §7, P3.3): every reader
  // and writer outside this class goes through the accessors below, so the
  // compiler proves that nothing else depends on it and P5 can change it.
 private:
  double value = double_min();
  bool missing = true;
  // A number or boolean document constant that is the value of a string slot
  // (design typed-core.md §9, decided 2026-11-18 after P2.1: it "keeps its
  // text as well as its number: string readers get the text (§3.4
  // Cast<string>) ..., numeric readers the number"). The slot is a string, so
  // `value` is svalue's string code, as every string's is, and the equality
  // readers (predicates, MapValues keys, NormDiscrete, equal, isIn) compare
  // the text as the base build did; a numeric reader that knows it reads a
  // number takes text_number(). Set only at load (cpmml::bind::slot_constant,
  // document_constant); the in-place string edits below clear it.
  bool numeric_text = false;
  std::string svalue;

 public:
  // Thread-local pointer to the active model's Indexer for per-model string mapping.
  // Set via ScopedIndexer RAII at model parse/score entry points.
  inline static thread_local class Indexer* active_indexer = nullptr;

  // RAII helper to set/restore the active indexer.
  struct ScopedIndexer {
    Indexer* prev;
    explicit ScopedIndexer(Indexer* idx) : prev(active_indexer) { active_indexer = idx; }
    ~ScopedIndexer() { active_indexer = prev; }
    ScopedIndexer(const ScopedIndexer&) = delete;
    ScopedIndexer& operator=(const ScopedIndexer&) = delete;
  };

  class ValueHash {
   public:
    // Equality compares `value`, so the hash has to be derived from it and from
    // nothing else. Casting the double to size_t is undefined whenever it is
    // negative or out of range, and it collapses every value in [0,1) onto the
    // same bucket besides. std::hash<double> is defined over the whole range and
    // agrees with operator== on the one case that matters here, -0.0 == 0.0.
    size_t operator()(const Value& obj) const { return std::hash<double>{}(obj.value); }
  };

  // Accessors (design typed-core.md §7, P3). Every reader outside this file
  // goes through these instead of the fields, so that P5 can change the
  // storage behind them; each is the field read or write it names, nothing
  // more (no copy, no check).
  inline double as_double() const { return value; }
  inline bool is_missing() const { return missing; }
  inline const std::string& text() const& { return svalue; }
  // The text of a temporary is the field as an xvalue, as `tmp.svalue` was,
  // so a caller that takes it (return f().text();) moves it, not copies it
  // (P3.2: TransformedValue::eval_str).
  inline std::string&& text() && { return std::move(svalue); }
  inline bool has_numeric_text() const { return numeric_text; }

  inline void set_double(const double number) { value = number; }
  inline void set_missing(const bool missing_now) { missing = missing_now; }
  inline void set_text(std::string new_text) { svalue = std::move(new_text); }
  inline void clear_text() { svalue.clear(); }
  inline void mark_numeric_text() { numeric_text = true; }

  Value() = default;
  // 3 cases: uninitialized (and missing), initialized but missing and not
  // missing (in other constructors)
  explicit Value(const std::string& value) : value(infer_value(value)), missing(false) {}
  explicit Value(const double& value) : value(value), missing(false) {}
  Value(const double& value, const DataType& datatype) : value(value), missing(false) {}
  Value(const std::string& value, const DataType& datatype) : value(to_double(value, datatype)), missing(false) {
    if (datatype == DataType::DataTypeValue::STRING) svalue = value;
  }

  inline Value operator+(const Value& other) const { return Value(value + other.value); }
  inline Value operator-(const Value& other) const { return Value(value - other.value); }
  inline Value operator/(const Value& other) const { return Value(value / other.value); }
  inline Value operator*(const Value& other) const { return Value(value * other.value); }
  inline bool operator==(const Value& other) const { return value == other.value; }
  inline bool operator!=(const Value& other) const { return !operator==(other); }
  inline bool operator>(const Value& other) const { return value > other.value; }
  inline bool operator>=(const Value& other) const { return value >= other.value; }
  inline bool operator<(const Value& other) const { return value < other.value; }
  inline bool operator<=(const Value& other) const { return value <= other.value; }
  inline bool operatorand(const Value& other) const { return value and other.value; }
  inline bool operatoror(const Value& other) const { return (bool)value or (bool) other.value; }
  inline bool operatornot() const { return not value; }
  inline Value diff(const Value& other) const { return Value(std::abs(value - other.value)); }
  inline bool is_in(const std::set<Value>& other) const { return other.find(*this) != other.end(); }
  inline bool is_in(const std::unordered_set<Value, ValueHash>& other) const {
    return other.find(*this) != other.end();
  }
  inline bool is_not_in(const std::set<Value>& other) const { return other.find(*this) == other.end(); }
  inline bool is_not_in(const std::unordered_set<Value, ValueHash>& other) const {
    return other.find(*this) == other.end();
  }

  // The number of a numeric_text value: its text, validated at load as a
  // number literal (an XSD double, or Java text: "Infinity") or a boolean
  // literal (1 or 0).
  inline double text_number() const {
    double number = 0;
    if (java::parse(svalue, number)) return number;
    bool flag = false;
    if (lexical::parse_boolean(svalue, flag)) return flag ? 1 : 0;
    throw std::logic_error("Value::text_number: \"" + svalue + "\" is not a number or boolean literal");
  }

  inline void lowercase() {
    numeric_text = false;
    std::transform(svalue.begin(), svalue.end(), svalue.begin(), ::tolower);
  }
  inline void uppercase() {
    numeric_text = false;
    std::transform(svalue.begin(), svalue.end(), svalue.begin(), ::toupper);
  }
  inline void substr(const size_t& start, const size_t& size) {
    numeric_text = false;
    svalue = svalue.substr(start, size);
  }
  inline void trim_blanks() {
    numeric_text = false;
    svalue = ::trim(svalue);
  }
  // std::regex's constructor parses and compiles the pattern. Both of these
  // rebuilt it per record, though the pattern comes from a Constant the
  // document fixes at parse time. The cache is thread_local: building a regex
  // into shared state from several scoring threads is a race, and one model
  // uses a handful of distinct patterns, so per-thread duplication is cheap.
  inline static const std::regex& cached_regex(const std::string& pattern) {
    static thread_local std::unordered_map<std::string, std::regex> cache;
    auto it = cache.find(pattern);
    if (it == cache.end()) it = cache.emplace(pattern, std::regex(pattern)).first;

    return it->second;
  }

  inline Value replace(const std::string& regex, const std::string& replacement) const {
    return Value(std::regex_replace(svalue, cached_regex(regex), replacement), DataType::DataTypeValue::STRING);
  }
  inline bool matches(const std::string& pattern) const { return std::regex_match(svalue, cached_regex(pattern)); }

  // Static members
  template <class CollectionT>
  inline static Value sum(const CollectionT& other) {
    return std::accumulate(other.cbegin(), other.cend(), Value());
  }

  template <class CollectionT>
  inline static Value min(const CollectionT& other) {
    return *std::min_element(other.cbegin(), other.cend());
  }

  template <class CollectionT>
  inline static Value max(const CollectionT& other) {
    return *std::max_element(other.cbegin(), other.cend());
  }

  inline static Value min(const std::set<Value>& other) { return *other.cbegin(); }
  inline static Value max(const std::set<Value>& other) { return *other.crbegin(); }

  // A Constant with no dataType: a number if the text is one, else a string.
  // The int range check is the existing rule (review M-X4 questions it); what
  // changed is the reader -- stod honoured LC_NUMERIC, accepted "inf"/"0x10",
  // and threw out_of_range past DBL_MAX instead of answering "not an int".
  inline static double infer_value(const std::string& value) {
    double double_value = 0.0;
    if (!lexical::parse_double(value, double_value) ||
        !(double_value > std::numeric_limits<int>::min() && double_value < std::numeric_limits<int>::max()))
      return to_double(value, DataType::DataTypeValue::STRING);

    return double_value;
  }

  // Document text (attributes, Array members, table cells) read as `datatype`.
  // Malformed text throws ParsingException, which at load time is a load error.
  // Input values do not come here: MiningField::parse_input reads them and
  // hands a malformed one to invalidValueTreatment.
  inline static double to_double(const std::string& value, const DataType& datatype) {
    switch (datatype.value) {
      case DataType::DataTypeValue::BOOLEAN: {
        bool parsed = false;
        if (!lexical::parse_boolean(value, parsed))
          throw cpmml::ParsingException("\"" + value + "\" is not a valid boolean (true, false, 1 or 0)");
        return parsed ? 1 : 0;
      }
      case DataType::DataTypeValue::FLOAT:
        return ::to_double(value);
      case DataType::DataTypeValue::INTEGER:
        return ::to_double(value);
      case DataType::DataTypeValue::DOUBLE:
        return ::to_double(value);
      case DataType::DataTypeValue::STRING:
        if (active_indexer) return active_indexer->get_string_value(value);

        // String codes are only meaningful within one model's Indexer: two
        // strings compare equal iff they were interned by the same map. There
        // used to be a process-global fallback map here for "contexts without
        // an active indexer", which is not a degraded answer -- a code drawn
        // from a different numbering silently fails to match the model's own,
        // and the map is shared mutable state across scoring threads. Every
        // entry point establishes a ScopedIndexer, so this is unreachable;
        // saying so beats interning into the wrong map.
        throw cpmml::InvalidValueException("no active Indexer: cannot map the string \"" + value +
                                           "\" to a value comparable with this model's");
    }

    return std::numeric_limits<double>::min();
  }

  // These used to strip every double quote from each member, compensating for a
  // splitter that left the delimiters attached to the token. parse_array now
  // consumes the delimiters and honours the spec's \" escape, so stripping here
  // would delete quotes that are part of the value: `"say \"hi\""` is the one
  // member `say "hi"`, and it matched nothing while this ran over it.
  inline static std::set<Value> createValues(const std::vector<std::string>& values, DataType datatype) {
    std::set<Value> result;
    for (const auto& value : values) result.insert(Value(value, datatype));

    return result;
  }

  inline static std::unordered_set<Value, ValueHash> createUnorderedValues(const std::vector<std::string>& values,
                                                                           DataType datatype) {
    std::unordered_set<Value, ValueHash> result;
    for (const auto& value : values) result.insert(Value(value, datatype));

    return result;
  }
};

// A Value as text, as JPMML writes it (utils/java_format.h): nothing for a
// missing value, the text of a string, Double.toString of a number. Value
// carries no type, so an integer, float or boolean value is written as a
// double here; a caller that knows the type writes java::Number{v.as_double(), kind}
// instead (OutputField does, from the expression's type).
inline std::string java_text(const Value& value) {
  if (value.is_missing()) return std::string();
  if (!value.text().empty()) return value.text();
  return java::double_text(value.as_double());
}

#if defined(__cpp_lib_format)
// std::format("{}", value): java_text(value).
template <>
struct std::formatter<Value, char> {
  constexpr auto parse(std::format_parse_context& ctx) {
    auto it = ctx.begin();
    if (it != ctx.end() && *it != '}') throw std::format_error("Value takes no format spec");
    return it;
  }
  template <class FormatContext>
  auto format(const Value& value, FormatContext& ctx) const {
    const std::string text = java_text(value);
    return std::copy(text.begin(), text.end(), ctx.out());
  }
};
#endif

// ---------------------------------------------------------------- adapters
// Between the engine's Value (old storage: a double, a missing flag and a
// text) and the P1 typed value (core/types/value.h), for P4's kernels, which
// take P1 scalar types over the old storage (design typed-core.md §7, P3 and
// P4). Nothing in the engine calls them in P3.
//
// unpack<T>(v, static_type) reads v as a value of static_type, the type the
// binder fixed for the expression that produced it. The old storage keeps
// every number and boolean as a double, so reading it in its static type is
// the explicit cast from that double (§2.6, types::CastTo): Double is the
// double itself (-0, NaN and INF kept); Float its IEEE round to nearest
// ("double -> float ... overflow -> ±INF, underflow -> 0", XSD float value
// space); Integer the value if it is integral and within int64, else
// Invalid(NotIntegral) or Invalid(Overflow) (Functions.html: "a double
// expression can not be implicitly cast as an integer"); Boolean 1 -> true,
// 0 -> false, else Invalid(NoCast) (the XSD boolean lexical space {true,
// false, 1, 0}, D14). A String is the text, and only the text: text() is
// what every string reader reads (§9, decided 2026-11-18 after P2.1: string
// readers get the text of a numeric_text value), and the string code in the
// double is the Indexer's, which a typed String does not carry; its rep is
// made by `ctx` (types::make_string), so a String needs the three-argument
// form. A missing v is Missing whatever its static type.
//
// T is types::Value (any of the above) or the scalar type of static_type
// itself. A scalar cannot hold Missing or Invalid, so unpack<Scalar> of a
// missing v, of a v that is not a value of the type, or with another static
// type is a programming error (std::logic_error), never a default.
//
// pack(x) is the old storage of a typed value, as the engine builds it from
// each type: a number its double (an Integer beyond 2^53 rounds, as every
// double the engine holds does), a boolean 1 or 0, a String Value(text,
// STRING), whose double is the active Indexer's code of the text (and -1,
// no code, for a text the model never saw once its Indexer is frozen), and
// Missing Value(). Invalid has no old storage: the engine raises an
// exception where a value is invalid, so packing one is a programming error.
//
// Round trip: pack(unpack(v, t)) is v, field for field, for every v the
// engine builds as a value of t: a number of a double or integer slot (any
// double, for double), a float of a float slot, 1 or 0 of a boolean slot, a
// string built from its text (Value(text, STRING)), and a missing v built as
// Value(). Two string Values do not round trip, because their double is not
// their text's code: one edited in place by uppercase/lowercase/substring/
// trimBlanks keeps its argument's code (B2, P6), and a numeric_text one
// (§9) keeps its mark, which a String cannot carry (pack gives the text's
// code and no mark).
namespace cpmml {

namespace detail {
// v (not missing) as a value of static_type.
inline types::Value unpack_present(const ::Value& v, const types::TypeId static_type, types::ParseContext* ctx) {
  const types::Double d{v.as_double()};
  switch (static_type) {
    case types::TypeId::Integer:
      return types::CastTo<types::Integer>{nullptr, nullptr}(d);
    case types::TypeId::Float:
      return types::CastTo<types::Float>{nullptr, nullptr}(d);
    case types::TypeId::Double:
      return d;
    case types::TypeId::Boolean:
      return types::CastTo<types::Boolean>{nullptr, nullptr}(d);
    case types::TypeId::String:
      if (!ctx) throw std::logic_error("unpack: a string needs a ParseContext to make its rep");
      return types::String{types::make_string(*ctx, v.text())};
    case types::TypeId::Missing:
    case types::TypeId::Invalid:
      break;
  }
  throw std::logic_error("unpack: Missing and Invalid are not static types");
}

template <class T>
T unpack(const ::Value& v, const types::TypeId static_type, types::ParseContext* ctx) {
  if constexpr (std::is_same_v<T, types::Value>) {
    if (static_type == types::TypeId::Missing || static_type == types::TypeId::Invalid)
      throw std::logic_error("unpack: Missing and Invalid are not static types");
    if (v.is_missing()) return types::Missing{};
    return unpack_present(v, static_type, ctx);
  } else {
    static_assert(types::PmmlParsedType<T>, "unpack<T>: T is types::Value or a P1 scalar type");
    if (static_type != T::type_id)
      throw std::logic_error("unpack<" + std::string(T::name) + ">: the static type is another");
    if (v.is_missing()) throw std::logic_error("unpack<" + std::string(T::name) + ">: the value is missing");
    const types::Value x = unpack_present(v, static_type, ctx);
    if (const T* scalar = std::get_if<T>(&x)) return *scalar;
    throw std::logic_error("unpack<" + std::string(T::name) + ">: " + java::double_text(v.as_double()) +
                           " is not a value of the type");
  }
}
}  // namespace detail

// A number, boolean or missing value (T = types::Value, or Integer, Float,
// Double, Boolean); a String static type needs the form below.
template <class T>
T unpack(const ::Value& v, const types::TypeId static_type) {
  static_assert(!std::is_same_v<T, types::String>, "unpack<String> needs a ParseContext");
  return detail::unpack<T>(v, static_type, nullptr);
}

// Any static type; a String's rep is made by ctx.
template <class T>
T unpack(const ::Value& v, const types::TypeId static_type, types::ParseContext& ctx) {
  return detail::unpack<T>(v, static_type, &ctx);
}

inline ::Value pack(const types::Value& x) {
  if (const auto* i = std::get_if<types::Integer>(&x)) return ::Value(static_cast<double>(i->v));
  if (const auto* f = std::get_if<types::Float>(&x)) return ::Value(static_cast<double>(f->v));
  if (const auto* d = std::get_if<types::Double>(&x)) return ::Value(d->v);
  if (const auto* b = std::get_if<types::Boolean>(&x)) return ::Value(b->v ? 1.0 : 0.0);
  if (const auto* s = std::get_if<types::String>(&x))
    return ::Value(std::string(s->text()), DataType::DataTypeValue::STRING);
  if (types::is_missing(x)) return ::Value();
  throw std::logic_error("pack: an Invalid value has no old storage");
}

}  // namespace cpmml

#endif  // CPMML_VALUE_H
