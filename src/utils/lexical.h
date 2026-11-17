/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_LEXICAL_H
#define CPMML_LEXICAL_H

/**
 * Design note -- one strict reader for the PMML lexical forms.
 *
 * Every number or boolean cPMML reads from text, whether an attribute of the
 * document, the content of an Array or a cell of a table, or an input value
 * passed to score/predict, goes through the functions here. The rules are the
 * XML Schema 1.0 lexical spaces that PMML's types are defined over:
 *
 *   double/float  [+-]? (D+ (. D*)? | . D+) ([eE] [+-]? D+)?  |  INF | -INF | NaN
 *   integer       [+-]? D+
 *   boolean       true | false | 1 | 0
 *
 * - The whole text must be the value. Leading and trailing XML whitespace
 *   (space, tab, CR, LF) is allowed, as the schema's whiteSpace="collapse"
 *   facet makes it insignificant; anything else left over makes the text
 *   malformed. The previous reader stopped at the first character it could
 *   not use, so "3abc" read as 3.
 * - Locale-independent: the conversion is std::from_chars, never stod/strtod.
 * - Case-sensitive, as the schema is: "INF", not "inf" or "Infinity"; "true",
 *   not "TRUE". Java-isms ("1d", "0x1p3") are not numbers.
 * - A decimal outside the double range is still a well-formed double: it
 *   rounds to +-INF or to +-0, as XSD 1.1 specifies and Java does.
 *
 * - No date or time forms. cPMML carries date, time, dateTime and the
 *   dateDaysSince/dateTimeSecondsSince types as DOUBLE (datatype.h), so an
 *   input of such a field is read as an XSD double: "2020-01-01" is malformed
 *   (an invalid value), where the previous reader took it as 2020.
 *
 * The functions report failure by return value; what a failure means is for
 * the caller: a malformed attribute is a load error (XmlNode's typed getters),
 * a malformed input value is an invalid value (MiningField's
 * invalidValueTreatment).
 */

#include <charconv>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>

namespace lexical {

constexpr bool is_xml_space(const char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

constexpr std::string_view trim(std::string_view text) {
  while (!text.empty() && is_xml_space(text.front())) text.remove_prefix(1);
  while (!text.empty() && is_xml_space(text.back())) text.remove_suffix(1);

  return text;
}

constexpr bool is_digit(const char c) { return c >= '0' && c <= '9'; }

namespace detail {
// Everything but the fast path of parse_double, kept out of line so that the
// fast path stays small enough to inline into the per-input readers.
inline bool parse_double_full(std::string_view text, double& parsed) {
  text = trim(text);
  if (text == "INF") return parsed = std::numeric_limits<double>::infinity(), true;
  if (text == "-INF") return parsed = -std::numeric_limits<double>::infinity(), true;
  if (text == "NaN") return parsed = std::numeric_limits<double>::quiet_NaN(), true;

  // Validate the grammar first: from_chars has its own (it takes "inf", "nan"
  // and rejects a leading '+'), so it is only trusted with text already known
  // to be a decimal.
  size_t i = 0;
  const size_t n = text.size();
  const bool negative = i < n && text[i] == '-';
  if (i < n && (text[i] == '+' || text[i] == '-')) i++;
  const size_t mantissa_begin = i;

  size_t integer_digits = 0, fraction_digits = 0;
  while (i < n && is_digit(text[i])) i++, integer_digits++;
  if (i < n && text[i] == '.') {
    i++;
    while (i < n && is_digit(text[i])) i++, fraction_digits++;
  }
  if (integer_digits + fraction_digits == 0) return false;
  const size_t mantissa_end = i;

  if (i < n && (text[i] == 'e' || text[i] == 'E')) {
    i++;
    if (i < n && (text[i] == '+' || text[i] == '-')) i++;
    size_t exponent_digits = 0;
    while (i < n && is_digit(text[i])) i++, exponent_digits++;
    if (exponent_digits == 0) return false;
  }
  if (i != n) return false;

  // from_chars rejects '+', so it starts after one; a '-' it reads itself.
  const char* first = text.data() + (text[0] == '+' ? 1 : 0);
  const char* last = text.data() + n;
  const auto result = std::from_chars(first, last, parsed);
  if (result.ec == std::errc() && result.ptr == last) return true;
  if (result.ec != std::errc::result_out_of_range) return false;

  // Out of range is overflow or underflow; which one is decided by the
  // magnitude, not by what from_chars left in `parsed` (unspecified). The
  // decimal exponent of the first significant digit says it: >= 0 is a
  // magnitude of at least 1, which can only have overflowed.
  long long exponent = 0;
  const size_t e = text.find_first_of("eE", mantissa_end);
  if (e != std::string_view::npos) {
    size_t j = e + 1;
    const bool exponent_negative = text[j] == '-';
    if (text[j] == '+' || text[j] == '-') j++;
    for (; j < n; j++) {
      exponent = exponent * 10 + (text[j] - '0');
      if (exponent > 1000000) break;  // far beyond any double either way
    }
    if (exponent_negative) exponent = -exponent;
  }
  long long position = -1;  // decimal exponent of the first significant digit
  long long place = static_cast<long long>(integer_digits) - 1;
  for (size_t j = mantissa_begin; j < mantissa_end; j++) {
    if (text[j] == '.') continue;
    if (text[j] != '0') {
      position = place;
      break;
    }
    place--;
  }
  const bool overflow = exponent + position >= 0;
  const double magnitude = overflow ? std::numeric_limits<double>::infinity() : 0.0;
  parsed = negative ? -magnitude : magnitude;

  return true;
}
}  // namespace detail

/**
 * XSD double (and float: same lexical space). Returns false, leaving `parsed`
 * unspecified, when the text is not one.
 */
inline bool parse_double(std::string_view text, double& parsed) {
  // Fast path for the common case, a plain decimal with no surrounding
  // blanks: from_chars on text that starts with a digit, '.', or '-' and then
  // one of those reads exactly the XSD decimal grammar (its other forms --
  // inf, nan -- start with a letter), so if it consumes all of it, the text is
  // well-formed. Anything else takes the full path below.
  if (!text.empty()) {
    const char c0 = text[0];
    const bool decimal_start =
        is_digit(c0) || c0 == '.' || (c0 == '-' && text.size() > 1 && (is_digit(text[1]) || text[1] == '.'));
    if (decimal_start) {
      const auto fast = std::from_chars(text.data(), text.data() + text.size(), parsed);
      if (fast.ec == std::errc() && fast.ptr == text.data() + text.size()) return true;
    }
  }

  return detail::parse_double_full(text, parsed);
}

/**
 * XSD integer, read into a long. False for text that is not one, and for one
 * that does not fit (the caller's message says which).
 */
inline bool parse_long(std::string_view text, long& parsed) {
  text = trim(text);
  if (!text.empty() && text.front() == '+') {
    text.remove_prefix(1);
    if (!text.empty() && text.front() == '-') return false;
  }
  if (text.empty()) return false;
  for (size_t i = text.front() == '-' ? 1 : 0; i < text.size(); i++)
    if (!is_digit(text[i])) return false;

  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);

  return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

/**
 * XSD integer read as a double: the lexical rule of an integer field, whose
 * value cPMML carries as a double. Any length is well-formed, as the XSD type
 * is unbounded; a value beyond 2^53 rounds as a double does.
 */
inline bool parse_integer_as_double(std::string_view text, double& parsed) {
  text = trim(text);
  size_t i = 0;
  if (i < text.size() && (text[i] == '+' || text[i] == '-')) i++;
  if (i == text.size()) return false;
  for (size_t j = i; j < text.size(); j++)
    if (!is_digit(text[j])) return false;

  return parse_double(text, parsed);
}

/** XSD boolean: true, false, 1, 0 -- nothing else, in no other case. */
inline bool parse_boolean(std::string_view text, bool& parsed) {
  text = trim(text);
  if (text == "true" || text == "1") return parsed = true, true;
  if (text == "false" || text == "0") return parsed = false, true;

  return false;
}

}  // namespace lexical

#endif
