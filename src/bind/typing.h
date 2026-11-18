/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BIND_TYPING_H
#define CPMML_BIND_TYPING_H

/**
 * Design note -- the static typing rules the binder applies (§2.5, §2.6,
 * §3.4). Pure functions over TypeIds and text; the binder (binder.h) calls
 * them and turns their answers into diagnostics.
 *
 * StaticType. A static type is a TypeId of a parsed type, or unknown
 * (nullopt): the type of an element the binder does not type yet, or of an
 * expression that already failed. Unknown never produces a second
 * diagnostic, so one defect is reported once.
 *
 * dataType. DataDictionary.html: "The dataType reuses names and semantics of
 * atomic types in XML Schema." The temporal dataTypes map to Double until P10
 * (§2.1: "until then the binder maps those dataTypes to Double exactly as
 * today"), with an "unsupported" diagnostic from the binder.
 *
 * Promotion (§2.5). Functions.html: "The inheritance precedence for mixed
 * type input parameters is as follows: string double float integer";
 * "boolean dataTypes can not be mixed with other types in a defined function
 * without explicitly defining the expected output dataType".
 *
 * Casts (§2.6). Functions.html: "implicit casts are allowed in the direction
 * of less restriction, according to the precedence list above ... an integer
 * expression can be implicitly cast as a double ... but a double expression
 * can not be implicitly cast as an integer". classify_cast says which kind a
 * declared type's cast of a value is: identity, widening (toward less
 * restriction), lossy (double -> float rounds, never fails), narrowing (may
 * give Invalid: to integer from float/double, string -> number or boolean,
 * number -> boolean), or explicit (boolean -> number or string: always
 * succeeds, outside the precedence list).
 *
 * Document text (§3.4, review K1). An xs:string attribute holding "a value of
 * the field", for a field of numeric type T:
 *   1. the text is a T literal -> a value of T;
 *   2. else it is an xs:double literal -> a Double (the comparison promotes);
 *   3. else -> malformed.
 * For boolean and string fields there is no rule 2.
 *
 * Constant (§3.4). Transformations.html: "A Constant that consists solely of
 * numbers will be treated as an integer, if a decimal point is present among
 * the numbers the Constant will be treated as a double ... A scientific
 * notation expression like 3.14e+5 or like 3.14E-5 should also be treated as
 * a double. The presence of any non-numeric characters (other than e, E, -, +
 * in scientific notation expressions ) will result in the Constant being
 * treated as a string. If the content is empty, then the constant will be
 * interpreted as an empty string, unless dataType is something besides
 * "string", in which case, it will be taken as a missing value of the
 * specified type." Untyped NaN/INF/-INF are strings: "if datatype is set to
 * its default value of "string", then the literal strings will be signified"
 * (B9, D10).
 */

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types/parse.h"
#include "core/types/value.h"
#include "utils/lexical.h"

namespace cpmml::bind {

// Load code may run the generic (std::visit) operations of the type library
// (core/types/cold.h): the binder is one of the three ColdPath issuers.
struct ColdPathIssuer {
  static types::cold::ColdPath make() { return {}; }
};

using types::TypeId;
using StaticType = std::optional<TypeId>;

enum class OpType : std::uint8_t { Categorical, Ordinal, Continuous };

constexpr std::string_view type_name(TypeId t) {
  switch (t) {
    case TypeId::Missing:
      return "missing";
    case TypeId::Invalid:
      return "invalid";
    case TypeId::Integer:
      return "integer";
    case TypeId::Float:
      return "float";
    case TypeId::Double:
      return "double";
    case TypeId::Boolean:
      return "boolean";
    case TypeId::String:
      return "string";
  }
  return "?";
}

inline std::string type_name(StaticType t) { return t ? std::string(type_name(*t)) : std::string("unknown"); }

constexpr bool is_numeric(TypeId t) { return t == TypeId::Integer || t == TypeId::Float || t == TypeId::Double; }
constexpr bool is_numeric(StaticType t) { return t && is_numeric(*t); }

constexpr int numeric_rank(TypeId t) { return t == TypeId::Integer ? 0 : t == TypeId::Float ? 1 : 2; }

// §2.5 over two numeric types.
constexpr TypeId promote(TypeId a, TypeId b) { return numeric_rank(a) >= numeric_rank(b) ? a : b; }

// The least restrictive of two static types (Functions.html precedence
// "string double float integer"), or nullopt when there is none: boolean
// with another type (Q7). Unknown absorbs.
enum class Common : std::uint8_t { Ok, Unknown, BooleanMixed };
struct CommonType {
  Common status;
  StaticType type;
};
constexpr CommonType least_restrictive(StaticType a, StaticType b) {
  if (!a || !b) return {Common::Unknown, std::nullopt};
  if (*a == *b) return {Common::Ok, a};
  if (is_numeric(*a) && is_numeric(*b)) return {Common::Ok, promote(*a, *b)};
  if (*a == TypeId::Boolean || *b == TypeId::Boolean) return {Common::BooleanMixed, std::nullopt};
  return {Common::Ok, TypeId::String};  // string with a number
}

// ---------------------------------------------------------------- dataType, optype
struct DataTypeInfo {
  TypeId type;
  bool temporal;
};

// DATATYPE of pmml-4-4.xsd.
inline std::optional<DataTypeInfo> parse_datatype(std::string_view s) {
  if (s == "string") return DataTypeInfo{TypeId::String, false};
  if (s == "integer") return DataTypeInfo{TypeId::Integer, false};
  if (s == "float") return DataTypeInfo{TypeId::Float, false};
  if (s == "double") return DataTypeInfo{TypeId::Double, false};
  if (s == "boolean") return DataTypeInfo{TypeId::Boolean, false};
  static constexpr std::string_view temporal[] = {"date",
                                                  "time",
                                                  "dateTime",
                                                  "dateDaysSince[0]",
                                                  "dateDaysSince[1960]",
                                                  "dateDaysSince[1970]",
                                                  "dateDaysSince[1980]",
                                                  "timeSeconds",
                                                  "dateTimeSecondsSince[0]",
                                                  "dateTimeSecondsSince[1960]",
                                                  "dateTimeSecondsSince[1970]",
                                                  "dateTimeSecondsSince[1980]"};
  for (auto t : temporal)
    if (s == t) return DataTypeInfo{TypeId::Double, true};
  return std::nullopt;
}

// OPTYPE of pmml-4-4.xsd.
inline std::optional<OpType> parse_optype(std::string_view s) {
  if (s == "categorical") return OpType::Categorical;
  if (s == "ordinal") return OpType::Ordinal;
  if (s == "continuous") return OpType::Continuous;
  return std::nullopt;
}

constexpr std::string_view optype_name(OpType o) {
  switch (o) {
    case OpType::Categorical:
      return "categorical";
    case OpType::Ordinal:
      return "ordinal";
    case OpType::Continuous:
      return "continuous";
  }
  return "?";
}

// ---------------------------------------------------------------- casts
enum class CastKind : std::uint8_t { Identity, Widening, Lossy, Narrowing, Explicit };

constexpr CastKind classify_cast(TypeId from, TypeId to) {
  if (from == to) return CastKind::Identity;
  if (is_numeric(from) && is_numeric(to)) {
    if (numeric_rank(to) > numeric_rank(from)) return CastKind::Widening;
    if (to == TypeId::Float) return CastKind::Lossy;  // double -> float
    return CastKind::Narrowing;                       // float, double -> integer
  }
  if (to == TypeId::String) return from == TypeId::Boolean ? CastKind::Explicit : CastKind::Widening;
  if (from == TypeId::Boolean) return CastKind::Explicit;  // true 1, false 0
  return CastKind::Narrowing;                              // string -> number or boolean; number -> boolean
}

// ---------------------------------------------------------------- document text (K1)
enum class TextRule : std::uint8_t {
  AsType,     // rule 1: a literal of the field's type
  AsDouble,   // rule 2: an xs:double literal, read as Double
  Malformed,  // rule 3
};

struct DocumentValue {
  TextRule rule;
  types::Value value;  // the value read (Invalid when malformed)
};

inline DocumentValue read_document_text(TypeId field_type, std::string_view text, types::ParseContext& ctx) {
  types::Value v = types::cold::parse_as(ColdPathIssuer::make(), field_type, text, ctx);
  if (!types::is_invalid(v)) return {TextRule::AsType, v};
  if (is_numeric(field_type)) {
    double d = 0;
    if (lexical::parse_double(text, d)) return {TextRule::AsDouble, types::Double{d}};
  }
  return {TextRule::Malformed, v};
}

// ---------------------------------------------------------------- Constant
enum class ConstantKind : std::uint8_t { Integer, IntegerOverflow, Double, String };

// The inferred type of an untyped Constant's text. "Solely of numbers":
// [+-]? digits; with a decimal point or an exponent it is a double if it is
// an xs:double literal made of digits, '.', e, E, -, + only (so NaN and INF
// are strings). Leading and trailing XML blanks do not make a number a
// string (xs:integer and xs:double collapse whitespace).
inline ConstantKind infer_constant(std::string_view raw) {
  const std::string_view text = lexical::trim(raw);
  if (text.empty()) return ConstantKind::String;
  for (char c : text)
    if (!(lexical::is_digit(c) || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+'))
      return ConstantKind::String;
  std::int64_t i = 0;
  switch (types::detail::read_integer(text, i)) {
    case types::detail::IntegerText::Ok:
      return ConstantKind::Integer;
    case types::detail::IntegerText::Overflow:
      return ConstantKind::IntegerOverflow;
    case types::detail::IntegerText::Malformed:
      break;
  }
  double d = 0;
  if (lexical::parse_double(text, d)) return ConstantKind::Double;
  return ConstantKind::String;
}

// ---------------------------------------------------------------- Array content
// Array (pmml-4-4.xsd, "ArrayType"): values separated by blanks; a value with
// blanks is in double quotes, and a backslash escapes the next character.
// The engine's tokenizer (utils/utils.h parse_array) reads it the same way.
inline std::vector<std::string> split_array(std::string_view value) {
  std::vector<std::string> tokens;
  std::size_t pos = 0;
  while (pos < value.size()) {
    while (pos < value.size() && lexical::is_xml_space(value[pos])) pos++;
    if (pos >= value.size()) break;
    std::string token;
    if (value[pos] == '"') {
      pos++;
      while (pos < value.size() && value[pos] != '"') {
        if (value[pos] == '\\' && pos + 1 < value.size()) pos++;
        token += value[pos++];
      }
      if (pos < value.size()) pos++;
    } else {
      while (pos < value.size() && !lexical::is_xml_space(value[pos])) token += value[pos++];
    }
    tokens.push_back(std::move(token));
  }
  return tokens;
}

}  // namespace cpmml::bind

#endif  // CPMML_BIND_TYPING_H
