
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
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <regex>

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
 public:
  double value = double_min();
  bool missing = true;
  std::string svalue;

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

  inline void lowercase() { std::transform(svalue.begin(), svalue.end(), svalue.begin(), ::tolower); }
  inline void uppercase() { std::transform(svalue.begin(), svalue.end(), svalue.begin(), ::toupper); }
  inline void substr(const size_t& start, const size_t& size) { svalue = svalue.substr(start, size); }
  inline void trim_blanks() { svalue = ::trim(svalue); }
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

#endif  // CPMML_VALUE_H
