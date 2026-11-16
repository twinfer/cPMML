
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_UTILS_H
#define CPMML_UTILS_H

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cPMML.h"
#include "miniz.h"

/**
 * @defgroup Utils
 *
 * Various utility functions, used also during model scoring.
 */
//@{
inline std::string to_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), ::tolower);

  return value;
}

/**
 * Locale-independent numeric parsing.
 *
 * `std::stod`/`std::stol` honour LC_NUMERIC: a host application that installs
 * a comma-decimal locale would silently mis-parse every PMML coefficient
 * ("1.5" -> 1). `std::from_chars` is locale-independent by definition, but is
 * also stricter than `stod` — it rejects leading whitespace and a leading '+',
 * both of which XSD permits — so those are handled here. Trailing characters
 * are ignored, as `stod` ignores them.
 */
inline const char* skip_numeric_prefix(const char* first, const char* const last) {
  while (first != last && std::isspace(static_cast<unsigned char>(*first))) first++;
  if (first != last && *first == '+') first++;

  return first;
}

inline bool try_to_double(const std::string_view value, double& parsed) {
  const char* const last = value.data() + value.size();
  const auto result = std::from_chars(skip_numeric_prefix(value.data(), last), last, parsed);

  return result.ec == std::errc();
}

inline bool try_to_long(const std::string_view value, long& parsed) {
  const char* const last = value.data() + value.size();
  const auto result = std::from_chars(skip_numeric_prefix(value.data(), last), last, parsed);

  return result.ec == std::errc();
}

inline static double to_double(const std::string& value) {
  const char* const last = value.data() + value.size();
  double parsed = 0.0;
  const auto result = std::from_chars(skip_numeric_prefix(value.data(), last), last, parsed);

  if (result.ec == std::errc::result_out_of_range)
    throw cpmml::ParsingException(value + " cannot be converted to double (overflow)");
  if (result.ec != std::errc())
    throw cpmml::ParsingException(value + " cannot be converted to double (invalid argument)");

  return parsed;
}

/**
 * Locale-independent equivalent of `std::to_string(double)`, which formats
 * through `sprintf("%f")` and therefore honours LC_NUMERIC. Same output as
 * `std::to_string` under the classic locale: fixed notation, six decimals.
 */
inline std::string double_to_string(const double value) {
  std::array<char, 512> buffer{};
  const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::fixed, 6);
  if (result.ec != std::errc()) return std::to_string(value);  // unreachable for finite doubles

  return std::string(buffer.data(), result.ptr);
}

template <class T>
T parse_string(const std::string& value) {
  std::istringstream is(to_lower(value));
  T parsed_value;
  is >> parsed_value;
  return parsed_value;
}

template <class T>
std::string to_string(const T& value) {
  std::stringstream buffer;
  buffer.imbue(std::locale::classic());  // never format numbers through the host locale

  buffer << value;

  return buffer.str();
}

/**
 * Join [first, last) into "[a,b,c]", writing each element through `format`.
 *
 * The separator is decided by looking *back* at whether anything has been
 * written, not forward with a lookahead iterator. Every mkstring overload used
 * to open with `auto jt = values.cbegin(); jt++;`, which is undefined on an
 * empty container — there is no position past cbegin() to advance to — and the
 * empty case is reachable from every one of them.
 */
template <class It, class Format>
std::string join_bracketed(It first, It last, const std::string& separator, Format format) {
  std::stringstream buffer;

  buffer << "[";
  for (It it = first; it != last; ++it) {
    if (it != first) buffer << separator;
    buffer << format(*it);
  }
  buffer << "]";

  return buffer.str();
}

template <class T>
std::string mkstring(const std::vector<T>& values, const std::string& separator = ",") {
  return join_bracketed(values.cbegin(), values.cend(), separator, [](const T& v) -> const T& { return v; });
}

template <class T>
std::string mkstring(const std::set<T>& values, const std::string& separator = ",") {
  return join_bracketed(values.cbegin(), values.cend(), separator, [](const T& v) -> const T& { return v; });
}

template <class T, class H>
std::string mkstring(const std::unordered_set<T, H>& values, const std::string& separator = ",") {
  return join_bracketed(values.cbegin(), values.cend(), separator, [](const T& v) -> const T& { return v; });
}

template <class K, class V, class H>
std::string to_string(const std::unordered_map<K, V, H>& values, const std::string& separator = ",") {
  std::stringstream buffer;

  buffer << "{";
  auto it = values.cbegin();
  size_t j = 1;
  for (; it != values.cend(); it++, j++)
    buffer << "\"" << it->first << "\":\"" << it->second << (j == values.size() ? "\"" : "\"" + separator);

  buffer << "}";

  return buffer.str();
}

// Declared before the mkstring overload that calls it. The call names a
// std::pair, so ADL only reaches namespace std and the name has to be visible
// at the point of definition; it went unnoticed because that overload had no
// callers and so was never instantiated.
template <class T, class V>
std::string to_string(const std::pair<T, V>& value) {
  std::stringstream buffer;

  buffer << "{";
  buffer << "\"" << value.first << "\":\"" << value.second << "\"";
  buffer << "}";

  return buffer.str();
}

template <class T, class U>
std::string mkstring(const std::vector<std::pair<T, U>>& values, const std::string& separator = ",") {
  return join_bracketed(values.cbegin(), values.cend(), separator,
                        [](const std::pair<T, U>& v) { return to_string<T, U>(v); });
}

template <class K, class V>
std::vector<V> to_values(const std::unordered_map<K, V>& values) {
  std::vector<V> result;
  result.reserve(values.size());
  for (const auto& value : values) result.push_back(value.second);

  return result;
}

template <class K, class V>
std::vector<V> to_values(const std::map<K, V>& values) {
  std::vector<V> result;
  result.reserve(values.size());
  for (const auto& value : values) result.push_back(value.second);

  return result;
}

template <class K, class V>
std::vector<K> to_keys(const std::unordered_map<K, V>& values) {
  std::vector<K> result;
  result.reserve(values.size());
  for (const auto& value : values) result.push_back(value.first);

  return result;
}

// Interior empty tokens are dropped but the remainder was pushed
// unconditionally, so a trailing separator produced an empty last member --
// which for a PMML Array meant a phantom category. The remainder is now held to
// the same rule, with the empty input still yielding one empty token so callers
// that index parts[0] keep working.
inline std::vector<std::string> split(std::string value, const std::string& separator) {
  size_t pos = 0;
  std::string token;
  std::vector<std::string> tokens;

  while ((pos = value.find(separator)) != std::string::npos) {
    token = value.substr(0, pos);
    if (token.size() > 0) tokens.push_back(token);
    value.erase(0, pos + separator.length());
  }

  if (!value.empty() || tokens.empty()) tokens.push_back(value);

  return tokens;
}

/**
 * The content of a PMML Array, per the spec's own description:
 *
 *   "The content of Array is a blank separated sequence of values, multiple
 *    blanks are as good as one blank. String values may be enclosed within
 *    double quotes ", which are not considered to be part of the value. If a
 *    string value contains the double quote character ", then it must be
 *    escaped by a backslash character \ (that is the same escaping mechanism
 *    as used in C/C++)."
 *
 * so `ab  "a b"   "with \"quotes\" "` is the three values `ab`, `a b` and
 * `with "quotes" `. Splitting on blanks alone tore quoted values apart at every
 * space they contained, which is how a category named "New York" became two
 * categories that matched nothing.
 */
inline std::vector<std::string> parse_array(const std::string& value) {
  std::vector<std::string> tokens;
  size_t pos = 0;

  while (pos < value.size()) {
    while (pos < value.size() && std::isspace(static_cast<unsigned char>(value[pos]))) pos++;
    if (pos >= value.size()) break;

    std::string token;
    if (value[pos] == '"') {
      pos++;  // opening quote, not part of the value
      while (pos < value.size() && value[pos] != '"') {
        // A backslash escapes the next character; the spec names the double
        // quote, and C/C++ escaping makes \\ a literal backslash.
        if (value[pos] == '\\' && pos + 1 < value.size()) pos++;
        token += value[pos++];
      }
      if (pos < value.size()) pos++;  // closing quote
    } else {
      while (pos < value.size() && !std::isspace(static_cast<unsigned char>(value[pos]))) token += value[pos++];
    }
    tokens.push_back(std::move(token));
  }

  return tokens;
}

inline std::string remove_all(std::string value, char to_remove) {
  value.erase(std::remove(value.begin(), value.end(), to_remove), value.end());

  return value;
}

inline bool parse_boolstring(const std::string& value) { return value == "true"; }

// Sentinel value indicating "missing/uninitialized".
// Must NOT be NaN (undefined under -ffast-math) and must NOT be a large magnitude
// value (would corrupt arithmetic when missing values leak through expressions).
// Using min() (~2.2e-308): a tiny positive value, indistinguishable from zero in
// real arithmetic, matching the original behavior.
inline constexpr double double_min() { return std::numeric_limits<double>::min(); }

inline bool is_double_min(double v) { return v == std::numeric_limits<double>::min(); }

// Heterogeneous hash for string-keyed maps: lets find() take a string_view
// without materialising a std::string. Value and Indexer each carried their own
// copy of this.
struct TransparentStringHash {
  using is_transparent = void;
  size_t operator()(std::string_view sv) const noexcept { return std::hash<std::string_view>{}(sv); }
};

static inline std::string& ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isspace(c); }));
  return s;
}

// trim from end
static inline std::string& rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) { return !std::isspace(c); }).base(), s.end());
  return s;
}

// trim from both ends
static inline std::string& trim(std::string& s) { return ltrim(rtrim(s)); }

static inline std::vector<char> read_xml(const std::string& filepath) {
  std::ifstream file(filepath);
  std::vector<char> data((std::istreambuf_iterator<char>(file)), {});
  data.push_back('\0');

  return data;
}

static inline void mz_reader_error(mz_zip_archive* zip_archive, const std::string& message) {
  mz_zip_reader_end(zip_archive);
  throw cpmml::ParsingException("unzip - err: " + message);
}

static inline std::vector<char> read_zip(const std::string& filepath) {
  mz_zip_archive zip_archive;
  size_t inflated_size;
  void* p;

  memset(&zip_archive, 0, sizeof(zip_archive));
  // mz_zip_reader_init_file returns mz_bool, not a stream status. It compared
  // against MZ_STREAM_END, which is 1 and so happened to agree with MZ_TRUE --
  // the right answer for the wrong reason, and wrong the moment miniz renumbers
  // its status enum.
  if (!mz_zip_reader_init_file(&zip_archive, filepath.c_str(), 0))
    throw cpmml::ParsingException("unzip - err: reading archive");
  if (mz_zip_reader_get_num_files(&zip_archive) < 1) mz_reader_error(&zip_archive, "no file in archive");
  if (mz_zip_reader_is_file_a_directory(&zip_archive, 0)) mz_reader_error(&zip_archive, "directory in archive");
  if ((p = mz_zip_reader_extract_to_heap(&zip_archive, 0, &inflated_size, 0)) == NULL)
    mz_reader_error(&zip_archive, "decompressing");
  std::vector<char> data((char*)p, (char*)p + inflated_size);
  data.push_back('\0');

  mz_free(p);
  mz_zip_reader_end(&zip_archive);

  return data;
}

static inline bool file_exists(const std::string& name) {
  std::ifstream f(name.c_str());
  return f.good();
}

static inline std::vector<char> read_file(const std::string& filepath, const bool zipped) {
  if (!file_exists(filepath)) throw cpmml::ParsingException("Input file " + filepath + " does not exist");

  if (zipped) return read_zip(filepath);

  return read_xml(filepath);
}

template <class T>
inline std::string format_num(const T& value) {
  std::stringstream sstr_value;
  std::string str_value;
  std::stringstream res;
  sstr_value.precision(std::numeric_limits<T>::max_digits10);
  sstr_value << value;
  str_value = sstr_value.str();
  if (str_value.find("e") != std::string::npos) return str_value;
  std::vector<std::string> parts = split(str_value, ".");

  std::reverse(parts[0].begin(), parts[0].end());
  for (auto i = 0u; i < parts[0].size(); i++) {
    if (i != 0 && i % 3 == 0) res << ",";

    res << parts[0][i];
  }

  std::string res_str = res.str();
  std::reverse(res_str.begin(), res_str.end());
  return res_str + (parts.size() > 1 ? "." + (parts[1].size() > 3 ? parts[1].substr(0, 3) : parts[1]) : "");
}

inline std::string format_int(const int& value) {
  std::stringstream sstr_value;
  std::string str_value;
  std::stringstream res;
  sstr_value.precision(std::numeric_limits<int>::max_digits10);
  sstr_value << value;
  str_value = sstr_value.str();
  if (str_value.find("e") != std::string::npos) return str_value;

  std::reverse(str_value.begin(), str_value.end());
  for (auto i = 0u; i < str_value.size(); i++) {
    if (i != 0 && i % 3 == 0) res << ",";

    res << str_value[i];
  }

  std::string res_str = res.str();
  std::reverse(res_str.begin(), res_str.end());
  return res_str;
}

//}@
#endif
