/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BIND_TYPING_H
#define CPMML_BIND_TYPING_H

/**
 * Design note -- the static typing rules the binder applies (§2.5, §2.6,
 * §3.4). Pure functions over TypeIds and text (and, for D21's untyped
 * Discretize and MapValues, the element's place in the document); the binder
 * (binder.h) and the engine's load (load_scope.h) call them, and the binder
 * turns their answers into diagnostics.
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
#include <pugixml.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "fn/table.h"
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

// ---------------------------------------------------------------- ordering (§2.7)
// The built-ins that order their two arguments. BuiltinFunctions.html lists
// them under one heading with the equality tests ("equal, notEqual, lessThan,
// lessOrEqual, greaterThan, greaterOrEqual ... Result is dependent on applying
// either function to two input parameters") and does not name the ordering
// subset; it is these four by what each tests (lessThan: "Check if field A is
// less than field B"). equal, notEqual, isIn and isNotIn only test equality.
constexpr bool is_ordering(std::string_view function) {
  return function == "lessThan" || function == "lessOrEqual" || function == "greaterThan" ||
         function == "greaterOrEqual";
}

// §2.7 (design commit P2.7, review S5): ordering a string against a number,
// in either order, is a load error (E-TYPE-ORDER-MIXED). DataDictionary.html:
// "Categorical field values can only be tested for equality, while ordinal
// field values in addition have an order defined"; the order of an ordinal
// string is its Value sequence ("For ordinal fields, if a sequence of Value
// elements is given, the sequence defines the ascending order"), which is an
// order of strings, and Functions.html's least restrictive type of a string
// and a number is string ("string double float integer"), whose text order
// ("10" < "3") is no order the spec gives the number. §2.7 makes no
// carve-out for a number whose text is one of an ordinal string's Values
// (JPMML 1.7.7 orders that one by the Value sequence; design P2.7 "As
// built"). Unknown types give no answer (the defect that made them unknown
// is reported once).
// Shared by the binder (Binder::check_comparable, report mode) and the
// engine's load (the Apply constructor, fn/apply_node.h).
constexpr bool order_mixed(StaticType a, StaticType b) {
  if (!a || !b) return false;
  return (*a == TypeId::String && is_numeric(*b)) || (is_numeric(*a) && *b == TypeId::String);
}

// Two static types of one kind (both strings, both numbers, both booleans):
// the operands an ordinal field's declared order can rank (design item
// P6.12, core/compare.h OrdinalOrder). Shared by the binder and the engine's
// load (fn/apply_node.h).
constexpr bool same_kind(StaticType a, StaticType b) {
  if (!a || !b) return false;
  return (is_numeric(*a) && is_numeric(*b)) || (*a == *b && (*a == TypeId::String || *a == TypeId::Boolean));
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
// an xs:double literal made of digits, '.', e, E, -, + only (so NaN, INF,
// -INF and +INF are strings, D10, D30). A decimal beyond the double range is
// still such a literal and reads as ±INF ("A scientific notation expression
// like 3.14e+5 or like 3.14E-5 should also be treated as a double"; XSD 1.1
// rounds it): untyped 1E400 is the double Infinity (D31; JPMML 1.7.7 reads
// it as a string). The text is not trimmed: Constant is simpleContent of
// base xs:string (pmml-4-4.xsd, element Constant), whose whiteSpace facet is
// "preserve" (XML Schema Part 2, 4.3.6), and Transformations.html says "The
// presence of any non-numeric characters ... will result in the Constant
// being treated as a string", so " 1" is the string " 1" (JPMML 1.7.7
// agrees: concat gives " 1x", + over a double is refused). Only a declared
// or inherited numeric dataType collapses the blanks (bind_constant).
inline ConstantKind infer_constant(std::string_view text) {
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

// The dataType an untyped Constant argument inherits from the built-in
// parameter it is passed to (§3.4, review S2). Transformations.html: "If the
// ParameterField definition includes a dataType, the Constant will inherit
// the dataType specified in the ParameterField". Only the parameters whose
// pseudo-declaration in BuiltinFunctions.html carries a dataType are
// inherited from: not (`<ParameterField name="x" dataType="boolean"/>`), the
// condition of if (`<ParameterField name="x" dataType="boolean"/>`),
// uppercase, lowercase, stringLength, trimBlanks, replace, matches (every
// `dataType="string"`), substring (input string, startPos and length
// `dataType="integer"`), the pattern of formatNumber and formatDatetime
// (`dataType="string"`), and the distribution, expm1, hypot, ln1p, rint and
// trigonometric functions (`dataType="double"`; normalPDF, normalIDF,
// stdNormalPDF, stdNormalIDF and erf "have similar to above
// pseudo-declarations"). The function table (fn/table.h) gives some
// parameters a type the spec's declarations do not: and and or ("The
// function takes a variable number of fields as parameters", no
// ParameterField) and the referenceYear of dateDaysSinceYear and
// dateSecondsSinceYear (`<ParameterField name="referenceYear"
// optype="continuous"/>`, no dataType). Those types still check the
// arguments (E-TYPE-PARAM) but give an untyped Constant nothing to inherit,
// so it is inferred. The arithmetic, comparison and aggregate parameters and
// the values of concat and if declare none either (Param::Numeric,
// Comparable, Any).
constexpr StaticType parameter_type(const fn::Signature& sig, std::size_t position) {
  if (sig.name == "and" || sig.name == "or" || sig.name == "dateDaysSinceYear" || sig.name == "dateSecondsSinceYear")
    return std::nullopt;
  switch (sig.param(position)) {
    case fn::Param::Integer:
      return TypeId::Integer;
    case fn::Param::Double:
      return TypeId::Double;
    case fn::Param::String:
      return TypeId::String;
    case fn::Param::Boolean:
      return TypeId::Boolean;
    default:
      return std::nullopt;
  }
}

// The type a Constant at `position` (0-based, after the Extensions) of an
// Apply of `function` inherits, or nullopt (no built-in of that name, or a
// parameter with no declared dataType).
constexpr StaticType inherited_constant_type(std::string_view function, std::size_t position) {
  const fn::Signature* sig = fn::find(function);
  return sig ? parameter_type(*sig, position) : std::nullopt;
}

// Constant text read as `type` (a declared or inherited dataType): the
// type's lexical space (XML Schema Part 2; numbers and booleans collapse
// blanks, string keeps them), plus, for float and double, INF, -INF and NaN
// in any case (Transformations.html: "These three constants are case
// insensitive and only meaningful as floating point values (datatype equal to
// "float" or "double")", §2.6, review S3). Invalid when the text is not in
// the space; empty content is the caller's (a missing value, or "").
inline types::Value parse_constant(TypeId type, std::string_view text, types::ParseContext& ctx) {
  switch (type) {
    case TypeId::Float:
      return types::parse_constant_as<types::Float>(text, ctx);
    case TypeId::Double:
      return types::parse_constant_as<types::Double>(text, ctx);
    default:
      return types::cold::parse_as(ColdPathIssuer::make(), type, text, ctx);
  }
}

// ---------------------------------------------------------------- Apply result (§4.1)
// The static type of an Apply of the built-in `sig` over arguments of static
// types `args` (its EXPRESSION children after Extension*), by the table's
// result rule. Functions.html: "functions are assumed to inherit the
// dataType of the current input parameters unless otherwise specified ...
// When the input parameters have multiple dataTypes, the least restrictive
// dataType will be inherited by default ... The inheritance precedence for
// mixed type input parameters is as follows: string double float integer";
// "boolean dataTypes can not be mixed with other types in a defined function
// without explicitly defining the expected output dataType". So:
//   Fixed             the declared type of the pseudo-declaration
//                     (BuiltinFunctions.html: round, floor, ceil and
//                     stringLength return dataType="integer", the
//                     predicates boolean, the string functions string,
//                     exp/ln/log10/sqrt and the further mathematical and
//                     distribution functions double);
//   LeastRestrictive  §2.5 over the arguments, unknown when one is not a
//                     number (a boolean or string operand, Q7, which the
//                     binder reports);
//   AtLeastDouble     the same, integer giving double (avg and median of
//                     integers are not integers);
//   IfBranches        D7: the common type of THEN and ELSE (THEN alone with
//                     no ELSE), by the same inheritance rule: BuiltinFunctions.html
//                     declares no result dataType for `if` (`<DefineFunction
//                     name="if">` with `<ParameterField name="A"/>` and
//                     `<ParameterField name="B"/>` untyped), so it inherits
//                     the least restrictive type of its value arguments, and
//                     the type is static, whichever branch a record takes
//                     (design commit P2.6, B19; JPMML 1.7.7 gives the type of
//                     the branch taken, §8.5 D7); unknown when either branch
//                     is, and `boolean_mixed` when one is boolean and the
//                     other is not (no common type, Q7).
// Shared by the binder (Binder::bind_apply, which reports what it finds) and
// the engine's load (the Apply constructor, fn/apply_node.h, and
// load_scope.h expression_type).
struct ApplyType {
  StaticType type;
  bool boolean_mixed = false;  // IfBranches only: THEN and ELSE have no common type
};

inline ApplyType apply_result_type(const fn::Signature& sig, const std::vector<StaticType>& args) {
  const std::size_t n = args.size();
  switch (sig.result) {
    case fn::Result::Fixed:
      return {sig.fixed};
    case fn::Result::LeastRestrictive:
    case fn::Result::AtLeastDouble: {
      if (n == 0) return {};
      TypeId t = TypeId::Integer;
      for (const StaticType& a : args) {
        if (!is_numeric(a)) return {};
        t = promote(t, *a);
      }
      if (sig.result == fn::Result::AtLeastDouble && t == TypeId::Integer) t = TypeId::Double;
      return {t};
    }
    case fn::Result::IfBranches: {
      if (n < 2) return {};
      if (n == 2) return {args[1]};
      const CommonType c = least_restrictive(args[1], args[2]);
      return {c.type, c.status == Common::BooleanMixed};
    }
  }
  return {};
}

// ---------------------------------------------------------------- untyped Discretize, MapValues (D21)
// pmml-4-4.xsd gives Discretize and MapValues an optional dataType
// (`<xs:attribute name="dataType" type="DATATYPE"/>`, no default) and the
// text of Transformations.html none. §3.4 / D21: "its dataType, else the
// nearest enclosing declared type", where the type that encloses the element
// is the one that applies to its position (design commit P2.8b1; P2.2 took
// it from the enclosing DerivedField or OutputField wherever the element
// stood, so lessThan over an untyped Discretize in a boolean DerivedField
// read its binValue "2" as a boolean, and failed the load):
//   the expression of a DerivedField or OutputField
//                 that field's dataType (`enclosing`): "DerivedField
//                 dataType=X -> Cast<X>(expression)" (§3.4);
//   an argument of an Apply whose built-in parameter has a type
//                 that type: the pseudo-declaration's ParameterField dataType
//                 (Transformations.html, Constant: "If the ParameterField
//                 definition includes a dataType, the Constant will inherit
//                 the dataType specified in the ParameterField"), or the
//                 boolean of and/or ("Evaluate the results of two or more
//                 boolean values"), as the function table types them
//                 (argument_type), when every value of the element is in
//                 that type's lexical space (reads_every_value); when one is
//                 not, the enclosing field's dataType, P2.2's reading, so
//                 that no document that loads on the base build stops
//                 loading here (P2.8b1 re-run: if(MapValues "1"/"2", ...) in
//                 a string field, sin(MapValues "true"/"false") in a boolean
//                 one); the binder still types the argument against the
//                 parameter (E-TYPE-PARAM, E-TYPE-BOOLEAN-MIXED), as on base;
//   an argument of a function whose result inherits its arguments' types
//                 (+, -, *, /, min, max, sum, avg, median, product, abs,
//                 pow, threshold, modulo; the values of if)
//                 the type that applies to the Apply's own position, by the
//                 same rule. Functions.html: "functions are assumed to
//                 inherit the dataType of the current input parameters", so
//                 a type that holds the result holds the arguments: an
//                 untyped MapValues under `/` in an integer DerivedField is
//                 an integer (fixture fix_apply_untyped_mapvalues_integer);
//   any other argument (an untyped parameter of a function of fixed result
//                 type: the comparisons, isIn, isNotIn, isMissing,
//                 isNotMissing, concat, round, floor, ceil, exp, ln, log10,
//                 sqrt, formatNumber's number; or a function that is not a
//                 built-in)
//                 the enclosing field's dataType, as P2.2 typed it, except
//                 in a boolean field. There the boolean is the predicate's
//                 result (comparisons, isIn, isMissing: the only built-ins
//                 of boolean result with an untyped parameter), not its
//                 argument's type, and P2.2's reading refused every text but
//                 "true", "false", "1", "0". In a boolean field an element
//                 every value of which is such a text (reads_every_value,
//                 also when it has no value) keeps that boolean reading, so
//                 no document that loads on the base build changes here
//                 (P2.8b1 re-run: inferring them read equal(MapValues
//                 "true"/"false", FieldRef b) of a boolean b as a string
//                 against a boolean, always false; base and JPMML 1.7.7
//                 compare booleans). Any other element in a boolean field
//                 (one that P2.2 refused) has its values inferred from their
//                 own text by the untyped Constant rule (Transformations.html: "If the dataType is not
//                 specified in the ParameterField definition or Constant
//                 element, it will be inferred by the content of the
//                 element. A Constant that consists solely of numbers will
//                 be treated as an integer, if a decimal point is present
//                 among the numbers the Constant will be treated as a double
//                 ... The presence of any non-numeric characters ... will
//                 result in the Constant being treated as a string"), each
//                 value (output cells or binValues, mapMissingTo,
//                 defaultValue) as a Constant's text and the element's type
//                 the least restrictive of them (Functions.html: "When the
//                 input parameters have multiple dataTypes, the least
//                 restrictive dataType will be inherited by default ...
//                 string double float integer"): lessThan(Discretize
//                 binValue "1"/"2", 2) compares integers. In any other field the field's type is
//                 kept (smallest choice, design commit P2.8b1): inference
//                 there changed documents that load on the base build, e.g.
//                 concat(MapValues "01", "x") in a string field ("01x" on
//                 base and JPMML 1.7.7, which types an untyped MapValues
//                 argument as a string) and equal or lessThan of such a
//                 MapValues against a string field. Unknown (an OutputField
//                 without a dataType) stays unknown, as in P2.2.
// Any other element between the expression and its field (none of the
// EXPRESSION elements but Apply holds another) is passed through.

// The position of `node` among the arguments of the Apply holding it (its
// element siblings after Extension*).
inline std::size_t argument_position(pugi::xml_node node) {
  std::size_t i = 0;
  for (pugi::xml_node s = node.previous_sibling(); s; s = s.previous_sibling())
    if (s.type() == pugi::node_element && std::string_view(s.name()) != "Extension") i++;
  return i;
}

// The type the function table gives the parameter at `position`, when it is
// one type (Comparable, Numeric, Any, Temporal and FieldRef are not).
constexpr StaticType argument_type(const fn::Signature& sig, std::size_t position) {
  switch (sig.param(position)) {
    case fn::Param::Integer:
      return TypeId::Integer;
    case fn::Param::Double:
      return TypeId::Double;
    case fn::Param::String:
      return TypeId::String;
    case fn::Param::Boolean:
      return TypeId::Boolean;
    default:
      return std::nullopt;
  }
}

// Every value of a Discretize or MapValues as the document writes it:
// mapMissingTo, defaultValue, and its binValues or output cells, the texts
// the engine's constructors read (discretize.h, mapvalues.h).
template <class F>
inline void for_each_value_text(pugi::xml_node element, F&& f) {
  if (const pugi::xml_attribute a = element.attribute("mapMissingTo")) f(std::string_view(a.value()));
  if (const pugi::xml_attribute a = element.attribute("defaultValue")) f(std::string_view(a.value()));
  if (std::string_view(element.name()) == "Discretize") {
    for (pugi::xml_node bin : element.children("DiscretizeBin"))
      if (const pugi::xml_attribute a = bin.attribute("binValue")) f(std::string_view(a.value()));
  } else if (const pugi::xml_attribute out = element.attribute("outputColumn")) {
    for (pugi::xml_node row : element.child("InlineTable").children("row"))
      if (const pugi::xml_node cell = row.child(out.value())) f(std::string_view(cell.text().get()));
  }
}

// Whether `text` is in the lexical space of `type`, by the readers the
// engine's document_constant and the binder's strict_value use
// (core/types/parse.h: XSD Part 2, whiteSpace collapse for every type but
// string, case-sensitive). For boolean, §3.2.2.1: "An instance of a datatype
// that is defined as boolean can have the following legal literals {true,
// false, 1, 0}"; every text is a string.
inline bool in_lexical_space(TypeId type, std::string_view text) {
  types::ParseContext ctx;
  switch (type) {
    case TypeId::Integer:
      return types::parse(std::type_identity<types::Integer>{}, text, ctx).has_value();
    case TypeId::Float:
      return types::parse(std::type_identity<types::Float>{}, text, ctx).has_value();
    case TypeId::Double:
      return types::parse(std::type_identity<types::Double>{}, text, ctx).has_value();
    case TypeId::Boolean:
      return types::parse(std::type_identity<types::Boolean>{}, text, ctx).has_value();
    case TypeId::String:
      return true;
    case TypeId::Missing:
    case TypeId::Invalid:
      break;
  }
  return false;
}

// Whether every value of the element is in the lexical space of `type`: the
// element can be read in it. True for an element with no value at all.
inline bool reads_every_value(pugi::xml_node element, TypeId type) {
  bool all = true;
  for_each_value_text(element, [&all, type](std::string_view text) { all = all && in_lexical_space(type, text); });
  return all;
}

// The type of an untyped Discretize's or MapValues' values read as untyped
// Constants' texts (the last case above).
inline TypeId inferred_values_type(pugi::xml_node element) {
  StaticType t;
  for_each_value_text(element, [&t](std::string_view text) {
    TypeId k = TypeId::String;
    switch (infer_constant(text)) {
      case ConstantKind::Integer:
      case ConstantKind::IntegerOverflow:  // a load error of the integer reading (D4)
        k = TypeId::Integer;
        break;
      case ConstantKind::Double:
        k = TypeId::Double;
        break;
      case ConstantKind::String:
        break;
    }
    t = t ? least_restrictive(t, k).type : StaticType(k);  // integer, double, string: never boolean-mixed
  });
  return t.value_or(TypeId::String);
}

// The static type of a Discretize or MapValues without a dataType, by the
// rule above. `enclosing` is the dataType the DerivedField or OutputField
// whose expression holds it declares (unknown when it declares none).
inline StaticType untyped_value_type(pugi::xml_node element, StaticType enclosing) {
  pugi::xml_node n = element;
  for (pugi::xml_node p = n.parent(); p && p.type() == pugi::node_element; n = p, p = p.parent()) {
    const std::string_view name = p.name();
    if (name == "DerivedField" || name == "OutputField") return enclosing;
    if (name != "Apply") continue;
    const fn::Signature* sig = fn::find(p.attribute("function").value());
    const StaticType t = sig ? argument_type(*sig, argument_position(n)) : std::nullopt;
    if (t) return reads_every_value(element, *t) ? t : enclosing;
    if (!sig || sig->result == fn::Result::Fixed)
      return enclosing == TypeId::Boolean && !reads_every_value(element, TypeId::Boolean)
                 ? StaticType(inferred_values_type(element))
                 : enclosing;
  }
  return enclosing;
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
