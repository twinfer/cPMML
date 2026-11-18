/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TYPES_PARSE_H
#define CPMML_TYPES_PARSE_H

/**
 * Design note -- XSD lexical reading per type (§2.6, D4, D11).
 *
 * The text of a value of type T is read with T's lexical space from XML
 * Schema Part 2 (DataDictionary.html: "The representation of the values is
 * defined in XML Schema Part 2: Datatypes"). The grammar is the one the
 * library already uses (src/utils/lexical.h, reused here, not copied):
 *
 *   integer  §3.3.13.1 "a finite-length sequence of decimal digits (#x30-#x39)
 *            with an optional leading sign". Read into int64; a value beyond
 *            int64 is Invalid(Overflow), not Malformed (D4: "integers are
 *            int64; overflow invalid").
 *   float    §3.2.4.1 mantissa, optional e/E exponent, "INF, -INF and NaN".
 *            The value is "the normalized value in the value space of float
 *            that is closest to d" (§3.2.4): rounded once, straight from the
 *            decimal to a float -- never through a double, which would round
 *            twice (1.00000005960464477550 is 1.0000001f, not 1.0f). Beyond
 *            the float range, ±INF or ±0 (lexical.h's XSD 1.1 rule).
 *   double   §3.2.5.1, the same grammar.
 *   boolean  §3.2.2.1 "{true, false, 1, 0}".
 *   string   any text, kept as is: "for string the value of whiteSpace is
 *            preserve". The empty text is the empty string (B37).
 *
 * Numbers and booleans: leading and trailing XML blanks are dropped ("For all
 * atomic datatypes other than string ... the value of whiteSpace is
 * collapse"). Everything is case-sensitive: TRUE, inf, Infinity, 1d are not
 * in any lexical space.
 *
 * The one exception is Constant text (parse_constant): Transformations.html,
 * Constant: NaN, INF, -INF "These three constants are case insensitive and
 * only meaningful as floating point values (datatype equal to "float" or
 * "double")". Everywhere else the forms stay strict (D11).
 *
 * Inferring the type of a Constant without dataType (§3.4) is the binder's,
 * not a lexical rule; it is not here.
 */

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>

#include "core/types/cold.h"
#include "core/types/string_pool.h"
#include "core/types/value.h"
#include "utils/lexical.h"

namespace cpmml::types {

// Where a String's rep comes from. Load: `pool` set, its text is interned.
// Scoring: `frozen` (the document's pool, read-only) is asked first, so a
// known text gets its pooled rep; otherwise `arena` makes one.
struct ParseContext {
  StringPool* pool = nullptr;
  const StringPool* frozen = nullptr;
  RecordArena* arena = nullptr;
};

// The rep of `text` under `ctx`'s rules.
inline const StringRep* make_string(ParseContext& ctx, std::string_view text) {
  if (ctx.pool) return ctx.pool->intern(text);
  if (ctx.frozen)
    if (const StringRep* pooled = ctx.frozen->find(text)) return pooled;
  if (!ctx.arena) throw std::logic_error("ParseContext without a pool or an arena cannot make a string");
  return ctx.arena->make(text);
}

namespace detail {

enum class IntegerText : std::uint8_t { Ok, Malformed, Overflow };

// XSD integer into int64.
inline IntegerText read_integer(std::string_view text, std::int64_t& out) {
  text = lexical::trim(text);
  bool plus = false;
  if (!text.empty() && text.front() == '+') {
    text.remove_prefix(1);
    plus = true;
  }
  if (text.empty()) return IntegerText::Malformed;
  const std::size_t first_digit = (!plus && text.front() == '-') ? 1 : 0;
  if (first_digit == text.size()) return IntegerText::Malformed;
  for (std::size_t i = first_digit; i < text.size(); i++)
    if (!lexical::is_digit(text[i])) return IntegerText::Malformed;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
  if (result.ec == std::errc::result_out_of_range) return IntegerText::Overflow;
  if (result.ec != std::errc() || result.ptr != text.data() + text.size()) return IntegerText::Malformed;
  return IntegerText::Ok;
}

// XSD float: the grammar and specials from lexical.h, the rounding straight
// to float.
inline bool read_float(std::string_view text, float& out) {
  double wide = 0;
  if (!lexical::parse_double(text, wide)) return false;
  if (!std::isfinite(wide)) {  // INF, -INF, NaN -- or a decimal beyond the double range
    out = narrow_to_float(wide);
    return true;
  }
  text = lexical::trim(text);
  if (text.front() == '+') text.remove_prefix(1);
  const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
  if (result.ec == std::errc() && result.ptr == text.data() + text.size()) return true;
  // Out of the float range: the double already says which side (±INF, ±0).
  out = narrow_to_float(wide);
  return true;
}

// Case-insensitive INF, -INF, NaN of Constant text.
inline std::optional<double> constant_special(std::string_view text) {
  text = lexical::trim(text);
  auto same = [](std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); i++) {
      const char x = (a[i] >= 'A' && a[i] <= 'Z') ? static_cast<char>(a[i] - 'A' + 'a') : a[i];
      if (x != b[i]) return false;
    }
    return true;
  };
  if (same(text, "inf")) return std::numeric_limits<double>::infinity();
  if (same(text, "-inf")) return -std::numeric_limits<double>::infinity();
  if (same(text, "nan")) return std::numeric_limits<double>::quiet_NaN();
  return std::nullopt;
}

}  // namespace detail

// ---------------------------------------------------------------- parse (the concept's operation)
inline std::optional<Integer> parse(std::type_identity<Integer>, std::string_view text, ParseContext&) {
  std::int64_t v = 0;
  if (detail::read_integer(text, v) != detail::IntegerText::Ok) return std::nullopt;
  return Integer{v};
}
inline std::optional<Float> parse(std::type_identity<Float>, std::string_view text, ParseContext&) {
  float v = 0;
  if (!detail::read_float(text, v)) return std::nullopt;
  return Float{v};
}
inline std::optional<Double> parse(std::type_identity<Double>, std::string_view text, ParseContext&) {
  double v = 0;
  if (!lexical::parse_double(text, v)) return std::nullopt;
  return Double{v};
}
inline std::optional<Boolean> parse(std::type_identity<Boolean>, std::string_view text, ParseContext&) {
  bool v = false;
  if (!lexical::parse_boolean(text, v)) return std::nullopt;
  return Boolean{v};
}
inline std::optional<String> parse(std::type_identity<String>, std::string_view text, ParseContext& ctx) {
  return String{make_string(ctx, text)};
}

// ---------------------------------------------------------------- ParseAs<T>
// The text as a value of T, or Invalid with the reason: Overflow for an
// integer beyond int64, Malformed for text outside T's lexical space.
template <PmmlParsedType T>
Value parse_as(std::string_view text, ParseContext& ctx) {
  if constexpr (std::is_same_v<T, Integer>) {
    std::int64_t v = 0;
    switch (detail::read_integer(text, v)) {
      case detail::IntegerText::Ok:
        return Integer{v};
      case detail::IntegerText::Overflow:
        return Invalid{InvalidReason::Overflow};
      case detail::IntegerText::Malformed:
        break;
    }
    return Invalid{InvalidReason::Malformed};
  } else {
    if (auto parsed = parse(std::type_identity<T>{}, text, ctx)) return *parsed;
    return Invalid{InvalidReason::Malformed};
  }
}

// Constant text with dataType T: as parse_as, plus case-insensitive INF,
// -INF and NaN when T is float or double (D11).
template <PmmlParsedType T>
Value parse_constant_as(std::string_view text, ParseContext& ctx) {
  if constexpr (std::is_same_v<T, Float> || std::is_same_v<T, Double>) {
    if (const auto special = detail::constant_special(text)) {
      if constexpr (std::is_same_v<T, Float>)
        return Float{narrow_to_float(*special)};
      else
        return Double{*special};
    }
  }
  return parse_as<T>(text, ctx);
}

namespace cold {
// ParseAs over a type known only at run time (the binder reading a document
// attribute whose type it has just decided). Missing and Invalid are not
// parsed types: asking for them is a programming error.
inline Value parse_as(ColdPath, TypeId type, std::string_view text, ParseContext& ctx) {
  switch (type) {
    case TypeId::Integer:
      return types::parse_as<Integer>(text, ctx);
    case TypeId::Float:
      return types::parse_as<Float>(text, ctx);
    case TypeId::Double:
      return types::parse_as<Double>(text, ctx);
    case TypeId::Boolean:
      return types::parse_as<Boolean>(text, ctx);
    case TypeId::String:
      return types::parse_as<String>(text, ctx);
    case TypeId::Missing:
    case TypeId::Invalid:
      break;
  }
  throw std::logic_error("cold::parse_as: Missing and Invalid have no lexical space");
}
}  // namespace cold

}  // namespace cpmml::types

#endif  // CPMML_TYPES_PARSE_H
