/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_JAVA_FORMAT_H
#define CPMML_JAVA_FORMAT_H

/**
 * Design note -- one writer for every value cPMML turns into text.
 *
 * Whenever a value becomes text -- the predicted value (Result::as_string),
 * a string OutputField over a number, a selectAll collection entry, a class
 * label of a numeric target, a prediction handed from a segment to its
 * MiningModel -- it is written here, the way JPMML 1.7.7 writes it, which is
 * the way Java writes the value's type:
 *
 *   double   Double.toString: the shortest decimal that reads back as the same
 *            double (at least two significant digits when one would do, the
 *            closest such pair: 4.9E-324, not 5E-324), plain for
 *            1e-3 <= |x| < 1e7 with at least one digit after the point
 *            ("1.5", "2.0", "0.001"), computerized scientific notation
 *            otherwise ("1.0E7", "1.0E-4", "1.7976931348623157E308");
 *            "NaN", "Infinity", "-Infinity", "0.0", "-0.0".
 *   float    Float.toString: the same rules over the float value, shortest
 *            for a float ("0.1" for 0.1f, where the double is
 *            0.10000000149011612).
 *   integer  Long.toString: "7", "-3".
 *   boolean  "true", "false".
 *
 * Before this, cPMML wrote numbers with std::to_string's six fixed decimals
 * ("1.500000", "0.000000" for 1e-7, three hundred digits for 1e300) in most
 * places and a private Java writer in selectAll collections.
 *
 * The digits come from std::to_chars, whose shortest form is the shortest
 * round-trip decimal, as Java's (JDK 19 and later, the JDK JPMML is run on)
 * is; only the layout and the two-digit rule are Java's own. No locale, no
 * allocation below write_double/write_float.
 *
 * Text written here is read back by parse() (internal hand-offs: a segment's
 * prediction, a Target's input): the XSD forms of lexical.h plus Java's
 * "Infinity" and "-Infinity". Inputs and document text never go through
 * parse(); they stay XSD-strict.
 *
 * C++20 <format>: std::formatter<java::Number> is specialised here, and
 * std::formatter<Value> in core/value.h, so std::format("{}", x) gives the
 * same text. std::format's own "{}" for a double is the shortest round trip
 * too, but in printf's layout ("1e+07", "1.5", "2"), not Java's. Where the
 * standard library has no <format> (__cpp_lib_format undefined) the
 * specialisations are left out and the functions below, which do not depend
 * on it, are the whole interface.
 */

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <version>

#if defined(__cpp_lib_format)
#include <format>
#endif

#include "lexical.h"

namespace java {

// The Java type a number is written as.
enum class Kind : unsigned char { DOUBLE, FLOAT, INTEGER, BOOLEAN };

// A number together with the Java type it is written as:
// std::format("{}", java::Number{0.1, java::Kind::FLOAT}) is "0.1".
struct Number {
  double value;
  Kind kind = Kind::DOUBLE;
};

// Enough for "-1.2345678901234567E-308".
constexpr size_t MAX_TEXT = 32;

namespace detail {

// The shortest digits of `value` (finite, non-zero), no sign, no trailing
// zeros, and the decimal exponent of the first one: 0.00125 is "125", -3.
template <class F>
inline void shortest_digits(const F value, char* digits, int& count, int& exponent) {
  std::array<char, 64> buffer{};
  const auto end = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::scientific).ptr;
  const char* p = buffer.data();
  if (*p == '-') ++p;
  count = 0;
  for (; p < end && *p != 'e'; ++p)
    if (*p != '.') digits[count++] = *p;
  // exponent: e[+-]dd(d)
  ++p;
  const bool negative = *p == '-';
  ++p;
  int e = 0;
  for (; p < end; ++p) e = e * 10 + (*p - '0');
  exponent = negative ? -e : e;
  while (count > 1 && digits[count - 1] == '0') --count;
}

// Java's rule when the shortest decimal has one digit: of the two-digit
// decimals that read back as `value`, the closest. That is the correctly
// rounded two-digit decimal when it reads back, and the one-digit decimal
// (d.0) otherwise.
template <class F>
inline void two_digits(const F value, char* digits, int& count, int& exponent) {
  std::array<char, 64> buffer{};
  const auto end =
      std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::scientific, 1).ptr;
  F back{};
  const char* first = buffer.data();
  std::from_chars(first, end, back);
  if (back != value) return;
  const char* p = first;
  if (*p == '-') ++p;
  char d[2] = {p[0], p[2]};  // "d.de..."
  const char* q = p + 3;     // 'e'
  ++q;
  const bool negative = *q == '-';
  ++q;
  int e = 0;
  for (; q < end; ++q) e = e * 10 + (*q - '0');
  digits[0] = d[0];
  digits[1] = d[1];
  count = d[1] == '0' ? 1 : 2;
  exponent = negative ? -e : e;
}

template <class F>
inline char* write(char* out, const F value) {
  if (std::isnan(value)) {
    for (const char c : std::string_view("NaN")) *out++ = c;
    return out;
  }
  if (std::isinf(value)) {
    for (const char c : std::string_view(value > 0 ? "Infinity" : "-Infinity")) *out++ = c;
    return out;
  }
  if (value == 0) {
    for (const char c : std::string_view(std::signbit(value) ? "-0.0" : "0.0")) *out++ = c;
    return out;
  }

  char digits[24];
  int count = 0;
  int exponent = 0;
  shortest_digits(value, digits, count, exponent);
  if (count == 1) two_digits(value, digits, count, exponent);

  if (value < 0) *out++ = '-';
  const F magnitude = value < 0 ? -value : value;
  if (magnitude >= F(1e-3) && magnitude < F(1e7)) {
    if (exponent >= 0) {
      // integer part: digits[0..exponent], zero-padded
      for (int i = 0; i <= exponent; i++) *out++ = i < count ? digits[i] : '0';
      *out++ = '.';
      if (count > exponent + 1)
        for (int i = exponent + 1; i < count; i++) *out++ = digits[i];
      else
        *out++ = '0';
    } else {
      *out++ = '0';
      *out++ = '.';
      for (int i = 0; i < -exponent - 1; i++) *out++ = '0';
      for (int i = 0; i < count; i++) *out++ = digits[i];
    }
  } else {
    *out++ = digits[0];
    *out++ = '.';
    if (count > 1)
      for (int i = 1; i < count; i++) *out++ = digits[i];
    else
      *out++ = '0';
    *out++ = 'E';
    out = std::to_chars(out, out + 8, exponent).ptr;
  }
  return out;
}

}  // namespace detail

// Double.toString(value) into [out, out + MAX_TEXT); returns the end.
inline char* write_double(char* out, const double value) { return detail::write(out, value); }

// Float.toString(value) into [out, out + MAX_TEXT); returns the end.
inline char* write_float(char* out, const float value) { return detail::write(out, value); }

inline std::string double_text(const double value) {
  char buffer[MAX_TEXT];
  return std::string(buffer, write_double(buffer, value));
}

// The double narrowed to float first, as JPMML holds a float value: 1e300 is
// "Infinity", 0.1 is "0.1".
inline std::string float_text(const double value) {
  char buffer[MAX_TEXT];
  return std::string(buffer, write_float(buffer, static_cast<float>(value)));
}

// Whether `value` has an exact long (Long.toString) form.
inline bool is_long(const double value) {
  return std::trunc(value) == value && value >= -9223372036854775808.0 && value < 9223372036854775808.0;
}

inline std::string integer_text(const long long value) {
  char buffer[MAX_TEXT];
  return std::string(buffer, std::to_chars(buffer, buffer + sizeof(buffer), value).ptr);
}

inline std::string boolean_text(const bool value) { return value ? "true" : "false"; }

// `number` as Java writes a value of its kind. An INTEGER that is not a whole
// number (a regression prediction of an integer target, say) has no integer
// text; it is written as a double ("2.5"). JPMML refuses such a value
// ("Expected integer value, got double value").
inline std::string text(const Number& number) {
  switch (number.kind) {
    case Kind::FLOAT:
      return float_text(number.value);
    case Kind::INTEGER:
      if (is_long(number.value)) return integer_text(static_cast<long long>(number.value));
      return double_text(number.value);
    case Kind::BOOLEAN:
      return boolean_text(number.value != 0);
    case Kind::DOUBLE:
      break;
  }
  return double_text(number.value);
}

// Reads back text written by this header: an XSD double (lexical.h) or Java's
// "Infinity"/"-Infinity". For internal hand-offs only, never for inputs.
inline bool parse(const std::string_view text, double& parsed) {
  if (lexical::parse_double(text, parsed)) return true;
  if (text == "Infinity") return parsed = std::numeric_limits<double>::infinity(), true;
  if (text == "-Infinity") return parsed = -std::numeric_limits<double>::infinity(), true;
  return false;
}

}  // namespace java

#if defined(__cpp_lib_format)
// std::format("{}", java::Number{...}): the Java text above. No format spec
// other than the empty one is accepted.
template <>
struct std::formatter<java::Number, char> {
  constexpr auto parse(std::format_parse_context& ctx) {
    auto it = ctx.begin();
    if (it != ctx.end() && *it != '}') throw std::format_error("java::Number takes no format spec");
    return it;
  }
  template <class FormatContext>
  auto format(const java::Number& number, FormatContext& ctx) const {
    const std::string text = java::text(number);
    return std::copy(text.begin(), text.end(), ctx.out());
  }
};
#endif

#endif  // CPMML_JAVA_FORMAT_H
