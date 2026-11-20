/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_FN_CAST_H
#define CPMML_FN_CAST_H

/**
 * Design note -- the explicit cast of a declaration (§2.6, §3.4, §5.1; P4.7).
 *
 * §2.6: the binder "inserts explicit Cast nodes where a declaration fixes a
 * type: a DerivedField's or OutputField's dataType ... the target field".
 * §3.4: "DerivedField dataType=X -> Cast<X>(expression)"; OutputField
 * "predictedValue -> Cast(dataType, prediction); ... transformedValue ->
 * Cast(dataType, expression)". DataDictionary.html: "The dataType reuses
 * names and semantics of atomic types in XML Schema", so the value of a
 * field declared X is a value of the XSD type X, and the cast takes the
 * value of the expression's static type (the binder's, fixed at load) to it:
 *
 *   from -> to                    result                         failure
 *   T -> T, integer/float/boolean
 *     -> double, boolean -> integer the value (no work)            --
 *   float, double -> integer      the value if integral, in int64  NotIntegral, Overflow
 *                                 (a -0 is the integer 0)
 *   integer, float, double
 *     -> boolean                  1 -> true, 0 -> false (D14)      NoCast
 *   integer, float, double,
 *     boolean -> string           Java text (Q1, D17): Long.toString,
 *                                 Float.toString, Double.toString,
 *                                 true/false (XSD canonical)       integer: an integer slot
 *                                                                  holding no int64
 *   string -> integer, float,
 *     double, boolean             the strict XSD reading of the text
 *                                 (lexical.h; whiteSpace collapse    Malformed; the
 *                                 only, D42)                         empty string, Empty
 *
 * ("Functions.html: a double expression can not be implicitly cast as an
 * integer"; XML Schema Part 2 §3.2.2.1 boolean "{true, false, 1, 0}".)
 * double -> float is the caller's (DerivedField narrows every value of a
 * float field, as it did before P4.7; OutputField leaves it, B5, P6/P8).
 *
 * Q2 (decided (b)): "a narrowing cast that fails at run time is an invalid
 * result for the record (InvalidValueException naming the field or target)":
 * apply() says whether the cast has a value, and the caller raises.
 *
 * The old storage (`::Value`, until P5) holds no type, and two of its
 * representations need a rule:
 *   - a number or boolean that carries a text (a document constant that the
 *     load gave its Cast<string> text, bind::slot_constant, element_constant,
 *     number_constant; a numeric_text value, §9 "decided 2026-11-18 after
 *     P2.1") keeps that text when cast to string: it is the text the load
 *     already gave the value's Cast<string> (what fn::adapter::concat does);
 *   - a numeric_text value (a number or boolean document constant in a
 *     string slot) cast to a number or boolean is its number (§9: "numeric
 *     readers the number"), Value::text_number;
 *   - a value of a string expression that has no text is the empty string,
 *     which is in the lexical space of no number and of no boolean (XML
 *     Schema Part 2 §3.3.13 integer: "a finite-length sequence of decimal
 *     digits"; §3.2.5.1 double: "a mantissa followed, optionally, by ... an
 *     exponent", or INF, -INF, NaN; §3.2.2.1 boolean: "{true, false, 1, 0}"):
 *     the cast has no value (decided 2026-11-20 after P4.9, item 3: "an empty
 *     string (or any text-less string) cast to a number or boolean is an
 *     invalid record, never its string code"). At run time the empty string
 *     is a Value whose text is "" and whose double is a string code (the
 *     pool's code of "", -1 for a string not seen at load, or the code of
 *     the string an in-place edit emptied: trimBlanks(substring("  x", 1, 2))
 *     keeps the code of "  x", B2). This is exact because nothing else of a
 *     string type is held without text: a number or boolean document
 *     constant that the load casts to the number or boolean slot of the
 *     field its element gives (bind::document_constant) keeps its document
 *     text, and an `if` of type string gives a number or boolean branch the
 *     text of its Cast<string> (fn::adapter::if_then_else_text, P4.7).
 * A cast never round-trips a string through unpack/pack (decided
 * 2026-11-19 after P3): a string kept is the Value it came in as, and only a
 * string the cast builds is packed (Value::string).
 *
 * Identity costs nothing: Op::None, which the caller tests before calling.
 */

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "core/types/parse.h"
#include "core/types/value.h"
#include "core/value.h"
#include "utils/java_format.h"

namespace cpmml::fn {

class DeclaredCast {
 public:
  using TypeId = types::TypeId;
  enum class Op : std::uint8_t {
    None,             // identity, or a cast with nothing to do over the old storage
    NumberToInteger,  // float, double -> integer
    NumberToBoolean,  // integer, float, double -> boolean
    ToString,         // integer, float, double, boolean -> string
    StringToNumber,   // string -> integer, float, double, boolean
  };

  DeclaredCast() = default;

  // The cast of a value of static type `from` (nullopt: a type the binder
  // does not give, a boolean or string operand of arithmetic, Q7, or a
  // FieldRef to an OutputField without dataType; the engine computed a
  // double there, so it is cast as one, the rule fn::adapter::text_type and
  // TransformedValue::kind_of apply) to the declared type `to`.
  DeclaredCast(std::optional<TypeId> from_type, TypeId to_type)
      : from(from_type.value_or(TypeId::Double)), to(to_type), op(classify(from, to)) {
    if (from == TypeId::Missing || from == TypeId::Invalid || to == TypeId::Missing || to == TypeId::Invalid)
      throw std::logic_error("DeclaredCast: a sentinel is not a static type");
  }

  TypeId from = TypeId::Double;
  TypeId to = TypeId::Double;
  Op op = Op::None;

  // Casts the present value `v` in place; false when it has no value of
  // `to` (Q2), `v` then unchanged.
  inline bool apply(::Value& v) const {
    switch (op) {
      case Op::None:
        return true;
      case Op::NumberToInteger:
        return to_integer(v);
      case Op::NumberToBoolean:
        return to_boolean(v);
      case Op::ToString:
        return to_string(v);
      case Op::StringToNumber:
        return from_string(v);
    }
    return true;
  }

  // The same for a number the caller holds as a double (an OutputField's
  // eval_double): only NumberToInteger and NumberToBoolean have work.
  inline bool apply_number(double& d) const {
    switch (op) {
      case Op::NumberToInteger:
        return integer_of(d);
      case Op::NumberToBoolean:
        return boolean_of(d);
      case Op::None:
        return true;
      case Op::ToString:
      case Op::StringToNumber:
        break;
    }
    throw std::logic_error("DeclaredCast::apply_number: the cast is not between numbers");
  }

  // `text` read strictly as `to` (a number or boolean), the double the old
  // storage holds it as, into `d`; false when it is not in the lexical space.
  inline bool parse_text(std::string_view text, double& d) const {
    switch (to) {
      case TypeId::Integer: {
        std::int64_t i = 0;
        if (types::detail::read_integer(text, i) != types::detail::IntegerText::Ok) return false;
        d = static_cast<double>(i);
        return true;
      }
      case TypeId::Float: {
        float f = 0;
        if (!types::detail::read_float(text, f)) return false;
        d = static_cast<double>(f);
        return true;
      }
      case TypeId::Double:
        return lexical::parse_double(text, d);
      case TypeId::Boolean: {
        bool b = false;
        if (!lexical::parse_boolean(text, b)) return false;
        d = b ? 1.0 : 0.0;
        return true;
      }
      case TypeId::String:
      case TypeId::Missing:
      case TypeId::Invalid:
        break;
    }
    throw std::logic_error("DeclaredCast::parse_text: not a number or boolean type");
  }

  // Why `v` has no value of `to`, for the record error: 'the value 3.5 of
  // type double is not a value of type integer (§2.6, Q2)'.
  [[gnu::cold]] [[gnu::noinline]] std::string failure(const ::Value& v) const {
    if (from == TypeId::String && v.text().empty() && !v.has_numeric_text())
      return "the string is empty, and the empty string is not a value of type " + std::string(type_label(to)) +
             " (§2.6, D42, Q2)";
    return failure_text(from == TypeId::String ? "\"" + std::string(v.text()) + "\"" : value_text(v.as_double()));
  }
  [[gnu::cold]] [[gnu::noinline]] std::string failure(double d) const { return failure_text(value_text(d)); }
  [[gnu::cold]] [[gnu::noinline]] std::string failure(std::string_view text) const {
    return failure_text("\"" + std::string(text) + "\"");
  }

  static constexpr Op classify(TypeId from, TypeId to) {
    if (from == to) return Op::None;
    switch (to) {
      case TypeId::String:
        return Op::ToString;
      case TypeId::Integer:
        if (from == TypeId::String) return Op::StringToNumber;
        return from == TypeId::Boolean ? Op::None : Op::NumberToInteger;  // a boolean is held as 1 or 0
      case TypeId::Boolean:
        return from == TypeId::String ? Op::StringToNumber : Op::NumberToBoolean;
      case TypeId::Float:
      case TypeId::Double:
        // integer, float, boolean -> double, and every number -> float: the
        // double is the value (the float narrowing is the caller's).
        return from == TypeId::String ? Op::StringToNumber : Op::None;
      case TypeId::Missing:
      case TypeId::Invalid:
        break;
    }
    return Op::None;
  }

 private:
  // §2.6 "double, float -> integer: the value if integral and in range"
  // (types::CastTo<Integer>); XSD's integer has one zero.
  static inline bool integer_of(double& d) {
    if (!(std::trunc(d) == d && d >= -9223372036854775808.0 && d < 9223372036854775808.0)) return false;
    d += 0.0;  // -0 -> 0
    return true;
  }
  // §2.6 "number -> boolean: 1 -> true, 0 -> false, else Invalid(NoCast)".
  static inline bool boolean_of(double& d) {
    if (d == 1.0) return true;
    if (d == 0.0) return d = 0.0, true;
    return false;
  }
  static inline bool to_integer(::Value& v) {
    double d = v.as_double();
    if (!integer_of(d)) return false;
    v.set_double(d);
    return true;
  }
  static inline bool to_boolean(::Value& v) {
    double d = v.as_double();
    if (!boolean_of(d)) return false;
    v.set_double(d);
    return true;
  }
  // §2.6 "number -> string: Java text"; "boolean -> string: true / false".
  inline bool to_string(::Value& v) const {
    if (!v.text().empty()) return true;  // the text the load gave it (see the note)
    const double d = v.as_double();
    std::string text;
    switch (from) {
      case TypeId::Integer:
        if (!java::is_long(d)) return false;
        text = java::integer_text(static_cast<long long>(d));
        break;
      case TypeId::Float:
        text = java::float_text(d);
        break;
      case TypeId::Double:
        text = java::double_text(d);
        break;
      case TypeId::Boolean:
        text = java::boolean_text(d != 0);
        break;
      case TypeId::String:
      case TypeId::Missing:
      case TypeId::Invalid:
        throw std::logic_error("DeclaredCast: string, Missing and Invalid are not cast to string");
    }
    v = ::Value::string(text);
    return true;
  }
  // §2.6 "string -> number, boolean: strict XSD reading of the text". A
  // numeric_text value is a document constant whose number the load read
  // (§9, decided 2026-11-18 after P2.1: "numeric readers the number"), and a
  // string value without text is the number it holds (see the note): each
  // is cast as that number.
  inline bool from_string(::Value& v) const {
    double d = 0;
    if (!v.has_numeric_text()) {
      // The empty string has no number (see the note): parse_text("") fails.
      if (!parse_text(v.text(), d)) return false;
      v = ::Value(d);
      return true;
    }
    d = v.text_number();
    switch (to) {
      case TypeId::Integer:
        if (!integer_of(d)) return false;
        break;
      case TypeId::Boolean:
        if (!boolean_of(d)) return false;
        break;
      default:
        break;
    }
    v = ::Value(d);
    return true;
  }

  // The value as Java writes its static type (an integer slot holding no
  // int64 as the double it holds).
  std::string value_text(double d) const {
    switch (from) {
      case TypeId::Integer:
        return java::is_long(d) ? java::integer_text(static_cast<long long>(d)) : java::double_text(d);
      case TypeId::Float:
        return java::float_text(d);
      case TypeId::Boolean:
        return java::boolean_text(d != 0);
      default:
        return java::double_text(d);
    }
  }
  std::string failure_text(const std::string& value) const {
    return "the value " + value + " of type " + std::string(type_label(from)) + " is not a value of type " +
           std::string(type_label(to)) + " (§2.6, Q2)";
  }
  static std::string_view type_label(TypeId t) {
    switch (t) {
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
      case TypeId::Missing:
      case TypeId::Invalid:
        break;
    }
    return "?";
  }
};

}  // namespace cpmml::fn

#endif  // CPMML_FN_CAST_H
