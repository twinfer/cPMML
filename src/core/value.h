
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

  struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const noexcept { return std::hash<std::string_view>{}(sv); }
  };
  inline static double string_index = 0;
  inline static std::unordered_map<std::string, double, StringHash, std::equal_to<>> string_converter;

  class ValueHash {
   public:
    size_t operator()(const Value& obj) const { return static_cast<size_t>(obj.value); }
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

  inline Value operator+=(const Value& other) const { return Value(svalue + other.svalue); }

  inline void lowercase() { std::transform(svalue.begin(), svalue.end(), svalue.begin(), ::tolower); }
  inline void uppercase() { std::transform(svalue.begin(), svalue.end(), svalue.begin(), ::toupper); }
  inline void substr(const size_t& start, const size_t& size) { svalue = svalue.substr(start, size); }
  inline void trim_blanks() { svalue = ::trim(svalue); }
  inline Value replace(const std::string& regex, const std::string& replacement) const {
    return Value(std::regex_replace(svalue, std::regex(regex), replacement), DataType::DataTypeValue::STRING);
  }
  inline bool matches(const std::string& pattern) { return std::regex_match(svalue, std::regex(pattern)); }

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
  inline static Value max(const std::set<Value>& other) { return *other.cend(); }

  inline static double infer_value(const std::string& value) {
    size_t char_index;
    double double_value;

    try {
      double_value = stod(value, &char_index);
      if (char_index != value.size() || !(double_value > std::numeric_limits<int>::min() &&
                                          double_value < std::numeric_limits<int>::max()))  // second condition is
                                                                                            // checking overflow
        return to_double(value, DataType::DataTypeValue::STRING);
    } catch (const std::invalid_argument& exception) {
      return to_double(value, DataType::DataTypeValue::STRING);
    }

    if ((double_value - int(double_value)) > 0 || value.find('.') != std::string::npos)
      return to_double(value, DataType::DataTypeValue::DOUBLE);

    return to_double(value, DataType::DataTypeValue::INTEGER);
  }

  inline static double to_double(const std::string& value, const DataType& datatype) {
    switch (datatype.value) {
      case DataType::DataTypeValue::BOOLEAN:
        if (to_lower(value) == "true" || value == "1") return 1;  // only case in which tolower is needed on value
        return 0;
      case DataType::DataTypeValue::FLOAT:
        return ::to_double(value);
      case DataType::DataTypeValue::INTEGER:
        return ::to_double(value);
      case DataType::DataTypeValue::DOUBLE:
        return ::to_double(value);
      case DataType::DataTypeValue::STRING: {
        auto it = string_converter.find(std::string_view(value));
        if (it == string_converter.end()) {
          auto [inserted, _] = string_converter.emplace(value, string_index++);
          return inserted->second;
        }
        return it->second;
      }
    }

    return std::numeric_limits<double>::min();
  }

  inline static std::set<Value> createValues(const std::vector<std::string>& values, DataType datatype) {
    std::set<Value> result;
    for (const auto& value : values) result.insert(Value(remove_all(value, '"'), datatype));

    return result;
  }

  inline static std::unordered_set<Value, ValueHash> createUnorderedValues(const std::vector<std::string>& values,
                                                                           DataType datatype) {
    std::unordered_set<Value, ValueHash> result;
    for (const auto& value : values) result.insert(Value(remove_all(value, '"'), datatype));

    return result;
  }
};

#endif  // CPMML_VALUE_H
