
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_BUILTINFUNCTIONS_H
#define CPMML_BUILTINFUNCTIONS_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <optional>
#include <pugixml.hpp>
#include <utility>
#include <vector>

#include "bind/load_scope.h"
#include "cPMML.h"
#include "enum_parser.h"
#include "value.h"
#include "xmlnode.h"

/**
 * @class BuiltInFunction
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/BuiltinFunctions.html">PMML built-in
 * functions</a>.
 *
 * It defines a set of predefined functions implementing low-level operations.
 * For instance:
 *      - arithmetic functions: min, max, sum, etc.
 *      - boolean functions: equal, isMissing, etc.
 *      - etc.
 */
class BuiltInFunction {  // inputs are mapped by position
 public:
  enum class BuiltInFunctionType {
    PLUS,
    MINUS,
    MUL,
    DIV,
    MAX,
    MIN,
    SUM,
    AVG,
    EXP,
    LN,
    LOG10,
    SQRT,
    ABS,
    ROUND,
    FLOOR,
    CEIL,
    POW,
    IF,
    IS_MISSING,
    IS_NOT_MISSING,
    EQUAL,
    NOT_EQUAL,
    LESS_THAN,
    LESS_THAN_EQUAL,
    GREATER_THAN,
    GREATER_THAN_EQUAL,
    IS_IN,
    IS_NOT_IN,
    REPLACE,
    MATCHES,
    UPPERCASE,
    LOWERCASE,
    SUBSTRING,
    TRIM_BLANKS,
    CONCAT,
    STRING_LENGTH,
    // Arithmetic and boolean members the table never had
    MEDIAN,
    PRODUCT,
    THRESHOLD,
    MODULO,
    NOT,
    // "Further mathematical functions"
    EXPM1,
    LN1P,
    HYPOT,
    RINT,
    SIN,
    ASIN,
    SINH,
    COS,
    ACOS,
    COSH,
    TAN,
    ATAN,
    TANH,
    // Distribution functions
    ERF,
    NORMAL_CDF,
    NORMAL_PDF,
    STD_NORMAL_CDF,
    STD_NORMAL_PDF,
    IDENTITY  // introduced to have "empty" function type
  };

  std::string function_string;
  BuiltInFunctionType function_type = BuiltInFunctionType::IDENTITY;
  std::function<Value(const std::vector<Value>&)> function;

  // Typing of this application, set by the Apply constructor from the
  // binder's static types (design commit P2.4: bind::expression_type, the
  // §3.4 rows and the function table's result rules, typing.h
  // apply_result_type); all false for a function built from its name alone.
  //   narrow_arguments  a float argument of a numeric function or a
  //                     comparison is a 32-bit float (XSD float), so it is
  //                     narrowed before use: the inputs are carried as
  //                     doubles and are not narrowed when read;
  //   float_result      the result is a float, so it is narrowed after;
  //   integer_division  "/" of two integers is an integer: truncated;
  //   integer_power     pow of two integers is an integer, which a negative
  //                     exponent does not give.
  // An argument or result whose type the binder does not know (a boolean or
  // string operand of arithmetic, Q7, reported in report mode and not yet a
  // load error) is neither a float nor an integer, so it sets none of them:
  // the function computes in doubles, as it did before P2.4.
  std::vector<cpmml::bind::StaticType> argument_types;
  bool narrow_arguments = false;
  bool float_result = false;
  bool integer_division = false;
  bool integer_power = false;
  bool typed_call = false;  // any of the four: operator() goes through typed()

  BuiltInFunction() = default;

  explicit BuiltInFunction(const std::string& function_string)
      : function_string(function_string),
        function_type(from_string(function_string)),
        function(get_function(function_type)) {}

  // The function an Apply element names, typed by its arguments' static types.
  explicit BuiltInFunction(const XmlNode& apply) : BuiltInFunction(apply.required_attribute("function")) {
    using cpmml::bind::TypeId;
    // A supported function (from_string refused the others) is a row of the
    // binder's table.
    const cpmml::fn::Signature* sig = cpmml::fn::find(function_string);
    if (!sig) throw std::logic_error("BuiltInFunction: \"" + function_string + "\" is not in the function table");
    // Arity (design commit P2.8b2, §4.2): the wrong number of arguments is a
    // load error, E-ARITY, by the table's min_args/max_args (binder.h
    // arity_fits; BuiltinFunctions.html: "+ ... <ParameterField name="a"/>
    // <ParameterField name="b"/>", "The functions -, *, and / are defined in
    // the same way and have two parameters"). It was checked on every record
    // instead, so that every record failed ("evaluating apply function"), or
    // under invalidValueTreatment="asMissing" was silently missing; and the
    // engine's own counts let concat take one argument and if four. The
    // arguments are the element children after Extension* (pmml-4-4.xsd
    // Apply, B25), as the binder counts them.
    std::size_t n = 0;
    for (pugi::xml_node argument : apply.xml().children())
      if (argument.type() == pugi::node_element && std::strcmp(argument.name(), "Extension") != 0) n++;
    if (!cpmml::bind::arity_fits(*sig, n)) throw cpmml::ParsingException(cpmml::bind::arity_message(apply, *sig, n));
    const bool comparison = compares_numbers(function_type);
    if (!is_numeric(function_type) && !comparison) return;
    argument_types.reserve(n);
    for (pugi::xml_node argument : apply.xml().children())
      if (argument.type() == pugi::node_element && std::strcmp(argument.name(), "Extension") != 0)
        argument_types.push_back(cpmml::bind::expression_type(argument));
    for (const auto& type : argument_types)
      if (type == TypeId::Float) narrow_arguments = true;
    if (comparison) {
      // §2.7 (design commit P2.7): string x number ordering is a load error.
      if (cpmml::bind::is_ordering(function_string))
        for (std::size_t i = 1; i < argument_types.size(); i++)
          if (cpmml::bind::order_mixed(argument_types[0], argument_types[i]))
            throw cpmml::ParsingException(cpmml::bind::order_mixed_message(apply, function_string, *argument_types[0],
                                                                           *argument_types[i]));
      typed_call = narrow_arguments;
      return;
    }
    const cpmml::bind::StaticType result = cpmml::bind::apply_result_type(*sig, argument_types).type;
    float_result = result == TypeId::Float;
    integer_division = function_type == BuiltInFunctionType::DIV && result == TypeId::Integer;
    integer_power = function_type == BuiltInFunctionType::POW && result == TypeId::Integer;
    typed_call = narrow_arguments || float_result || integer_division || integer_power;
  }

  static constexpr EnumEntry<BuiltInFunctionType> TABLE[] = {
      {"+", BuiltInFunctionType::PLUS},
      {"-", BuiltInFunctionType::MINUS},
      {"*", BuiltInFunctionType::MUL},
      {"/", BuiltInFunctionType::DIV},
      {"max", BuiltInFunctionType::MAX},
      {"min", BuiltInFunctionType::MIN},
      {"sum", BuiltInFunctionType::SUM},
      {"avg", BuiltInFunctionType::AVG},
      {"exp", BuiltInFunctionType::EXP},
      {"ln", BuiltInFunctionType::LN},
      {"log10", BuiltInFunctionType::LOG10},
      {"sqrt", BuiltInFunctionType::SQRT},
      {"abs", BuiltInFunctionType::ABS},
      {"round", BuiltInFunctionType::ROUND},
      {"floor", BuiltInFunctionType::FLOOR},
      {"ceil", BuiltInFunctionType::CEIL},
      {"pow", BuiltInFunctionType::POW},
      {"if", BuiltInFunctionType::IF},
      {"isMissing", BuiltInFunctionType::IS_MISSING},
      {"isNotMissing", BuiltInFunctionType::IS_NOT_MISSING},
      {"equal", BuiltInFunctionType::EQUAL},
      {"notEqual", BuiltInFunctionType::NOT_EQUAL},
      {"lessThan", BuiltInFunctionType::LESS_THAN},
      {"lessOrEqual", BuiltInFunctionType::LESS_THAN_EQUAL},
      {"greaterThan", BuiltInFunctionType::GREATER_THAN},
      {"greaterOrEqual", BuiltInFunctionType::GREATER_THAN_EQUAL},
      {"isIn", BuiltInFunctionType::IS_IN},
      {"isNotIn", BuiltInFunctionType::IS_NOT_IN},
      {"replace", BuiltInFunctionType::REPLACE},
      {"matches", BuiltInFunctionType::MATCHES},
      {"uppercase", BuiltInFunctionType::UPPERCASE},
      {"lowercase", BuiltInFunctionType::LOWERCASE},
      {"substring", BuiltInFunctionType::SUBSTRING},
      {"trimBlanks", BuiltInFunctionType::TRIM_BLANKS},
      {"concat", BuiltInFunctionType::CONCAT},
      {"stringLength", BuiltInFunctionType::STRING_LENGTH},
      {"median", BuiltInFunctionType::MEDIAN},
      {"product", BuiltInFunctionType::PRODUCT},
      {"threshold", BuiltInFunctionType::THRESHOLD},
      {"modulo", BuiltInFunctionType::MODULO},
      {"not", BuiltInFunctionType::NOT},
      {"expm1", BuiltInFunctionType::EXPM1},
      {"ln1p", BuiltInFunctionType::LN1P},
      {"hypot", BuiltInFunctionType::HYPOT},
      {"rint", BuiltInFunctionType::RINT},
      {"sin", BuiltInFunctionType::SIN},
      {"asin", BuiltInFunctionType::ASIN},
      {"sinh", BuiltInFunctionType::SINH},
      {"cos", BuiltInFunctionType::COS},
      {"acos", BuiltInFunctionType::ACOS},
      {"cosh", BuiltInFunctionType::COSH},
      {"tan", BuiltInFunctionType::TAN},
      {"atan", BuiltInFunctionType::ATAN},
      {"tanh", BuiltInFunctionType::TANH},
      {"erf", BuiltInFunctionType::ERF},
      {"normalCDF", BuiltInFunctionType::NORMAL_CDF},
      {"normalPDF", BuiltInFunctionType::NORMAL_PDF},
      {"stdNormalCDF", BuiltInFunctionType::STD_NORMAL_CDF},
      {"stdNormalPDF", BuiltInFunctionType::STD_NORMAL_PDF},
      // PMML built-ins cPMML does not implement; see the note at the end of
      // this class.
      {"and", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
      {"or", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
      {"isValid", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
      {"isNotValid", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
      {"normalIDF", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
      {"stdNormalIDF", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
      {"formatNumber", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
      {"formatDatetime", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
      {"dateDaysSinceYear", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
      {"dateSecondsSinceYear", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
      {"dateSecondsSinceMidnight", BuiltInFunctionType::IDENTITY, nullptr, EnumSupport::UNSUPPORTED},
  };

  // Apply@function is required. A name that is not a PMML built-in is either
  // a DefineFunction, which cPMML does not support, or a typo; either way
  // there is no function to apply. Names are case-sensitive, as in JPMML.
  static BuiltInFunctionType from_string(const std::string& builtinfunction) {
    // "log" and "ceiling" are what this table used to hold. Neither is a PMML
    // built-in, and "log" is worse than merely non-conforming: several systems
    // spell log10 that way, so a document meaning log10 would have got a
    // natural logarithm. Point at the real name rather than fail blankly.
    if (builtinfunction == "log")
      throw cpmml::ParsingException(
          "function \"" + builtinfunction +
          "\" is not a PMML built-in: the natural logarithm is \"ln\" and the base-10 logarithm is \"log10\"");
    if (builtinfunction == "ceiling")
      throw cpmml::ParsingException("function \"" + builtinfunction + "\" is not a PMML built-in: it is \"ceil\"");
    if (!enum_find(builtinfunction, TABLE))
      throw cpmml::ParsingException("invalid value \"" + builtinfunction +
                                    "\" for Apply@function: not a PMML built-in function, and user-defined "
                                    "functions (DefineFunction) are unsupported");

    return enum_value(builtinfunction, TABLE, "Apply@function");
  }

  static std::function<Value(const std::vector<Value>&)> get_function(const BuiltInFunctionType& function_type) {
    switch (function_type) {
      case BuiltInFunctionType::PLUS:
        return plus;
      case BuiltInFunctionType::MINUS:
        return minus;
      case BuiltInFunctionType::MUL:
        return mul;
      case BuiltInFunctionType::DIV:
        return div;
      case BuiltInFunctionType::MAX:
        return max;
      case BuiltInFunctionType::MIN:
        return min;
      case BuiltInFunctionType::SUM:
        return sum;
      case BuiltInFunctionType::AVG:
        return avg;
      case BuiltInFunctionType::EXP:
        return exp;
      case BuiltInFunctionType::LN:
        return ln;
      case BuiltInFunctionType::LOG10:
        return log10;
      case BuiltInFunctionType::SQRT:
        return sqrt;
      case BuiltInFunctionType::ABS:
        return abs;
      case BuiltInFunctionType::ROUND:
        return round;
      case BuiltInFunctionType::FLOOR:
        return floor;
      case BuiltInFunctionType::CEIL:
        return ceil;
      case BuiltInFunctionType::POW:
        return pow;
      case BuiltInFunctionType::IF:
        return if_then_else;
      case BuiltInFunctionType::IS_MISSING:
        return is_missing;
      case BuiltInFunctionType::IS_NOT_MISSING:
        return is_notmissing;
      case BuiltInFunctionType::IS_IN:
        return is_in;
      case BuiltInFunctionType::IS_NOT_IN:
        return is_notin;
      case BuiltInFunctionType::EQUAL:
        return equal;
      case BuiltInFunctionType::NOT_EQUAL:
        return not_equal;
      case BuiltInFunctionType::LESS_THAN:
        return less_than;
      case BuiltInFunctionType::LESS_THAN_EQUAL:
        return less_thanorequal;
      case BuiltInFunctionType::GREATER_THAN:
        return greater_than;
      case BuiltInFunctionType::GREATER_THAN_EQUAL:
        return greater_thanorequal;
      case BuiltInFunctionType::REPLACE:
        return replace;
      case BuiltInFunctionType::MATCHES:
        return str_matches;
      case BuiltInFunctionType::UPPERCASE:
        return str_uppercase;
      case BuiltInFunctionType::LOWERCASE:
        return str_lowercase;
      case BuiltInFunctionType::SUBSTRING:
        return str_substring;
      case BuiltInFunctionType::TRIM_BLANKS:
        return str_trim_blanks;
      case BuiltInFunctionType::CONCAT:
        return str_concat;
      case BuiltInFunctionType::STRING_LENGTH:
        return str_length;
      case BuiltInFunctionType::MEDIAN:
        return median;
      case BuiltInFunctionType::PRODUCT:
        return product;
      case BuiltInFunctionType::THRESHOLD:
        return threshold;
      case BuiltInFunctionType::MODULO:
        return modulo;
      case BuiltInFunctionType::NOT:
        return logical_not;
      case BuiltInFunctionType::EXPM1:
        return unary(std::expm1);
      case BuiltInFunctionType::LN1P:
        return unary(std::log1p);
      case BuiltInFunctionType::HYPOT:
        return hypot;
      case BuiltInFunctionType::RINT:
        return unary(std::rint);
      case BuiltInFunctionType::SIN:
        return unary(std::sin);
      case BuiltInFunctionType::ASIN:
        return unary(std::asin);
      case BuiltInFunctionType::SINH:
        return unary(std::sinh);
      case BuiltInFunctionType::COS:
        return unary(std::cos);
      case BuiltInFunctionType::ACOS:
        return unary(std::acos);
      case BuiltInFunctionType::COSH:
        return unary(std::cosh);
      case BuiltInFunctionType::TAN:
        return unary(std::tan);
      case BuiltInFunctionType::ATAN:
        return unary(std::atan);
      case BuiltInFunctionType::TANH:
        return unary(std::tanh);
      case BuiltInFunctionType::ERF:
        return unary(std::erf);
      case BuiltInFunctionType::NORMAL_CDF:
        return normal_cdf;
      case BuiltInFunctionType::NORMAL_PDF:
        return normal_pdf;
      case BuiltInFunctionType::STD_NORMAL_CDF:
        return std_normal_cdf;
      case BuiltInFunctionType::STD_NORMAL_PDF:
        return std_normal_pdf;
      default:
        throw cpmml::ParsingException("unsupported function");
    }
  }

  // The number of inputs was checked at load (E-ARITY, the constructor).
  inline Value operator()(const std::vector<Value>& input) const {
    if (!typed_call) return function(input);

    return typed(input);
  }

  // The function applied in the types of its arguments and result.
  Value typed(const std::vector<Value>& input) const {
    std::vector<Value> narrowed;
    if (narrow_arguments) {
      narrowed = input;
      for (size_t i = 0; i < narrowed.size() && i < argument_types.size(); ++i)
        if (argument_types[i] == cpmml::bind::TypeId::Float && !narrowed[i].missing)
          narrowed[i].value = static_cast<double>(static_cast<float>(narrowed[i].value));
    }
    const std::vector<Value>& arguments = narrow_arguments ? narrowed : input;
    // XSD integer has no value for 7^-2. Invalid, so that the Apply's
    // invalidValueTreatment decides (JPMML refuses it, whatever that says).
    if (integer_power && arguments[1].value < 0)
      throw cpmml::InvalidValueException("pow: an integer raised to a negative integer is not an integer");

    Value result = function(arguments);
    if (result.missing) return result;
    // Truncated toward zero, as Java and XQuery's idiv do; the specification
    // gives the type but not the rounding. Division by zero is left as it was
    // (INF, -INF or NaN): there is no integer to give.
    if (integer_division && arguments[1].value != 0.0) result.value = std::trunc(result.value);
    if (float_result) result.value = static_cast<double>(static_cast<float>(result.value));

    return result;
  }

  // True for functions that must receive missing arguments instead of having
  // Apply short-circuit them (PMML: isMissing/isNotMissing/if).
  inline bool handles_missing() const {
    return function_type == BuiltInFunctionType::IS_MISSING || function_type == BuiltInFunctionType::IS_NOT_MISSING ||
           function_type == BuiltInFunctionType::IF;
  }

  // The aggregates, for which "Missing values in the input to an aggregate
  // function are simply ignored ... If all inputs are missing, the result
  // evaluates to a missing value" (Built-in functions, min, max, sum, avg,
  // median, product). mapMissingTo still comes first: "if it is specified and
  // any of the input values of the function are missing then the function is
  // not applied at all" (Functions, Apply).
  inline bool ignores_missing() const { return is_aggregate(function_type); }

  inline static bool is_aggregate(const BuiltInFunctionType& type) {
    return type == BuiltInFunctionType::MIN || type == BuiltInFunctionType::MAX || type == BuiltInFunctionType::SUM ||
           type == BuiltInFunctionType::AVG || type == BuiltInFunctionType::MEDIAN ||
           type == BuiltInFunctionType::PRODUCT;
  }

  inline static Value plus(const std::vector<Value>& input) { return input[0] + input[1]; }
  inline static Value minus(const std::vector<Value>& input) { return input[0] - input[1]; }
  inline static Value mul(const std::vector<Value>& input) { return input[0] * input[1]; }
  inline static Value div(const std::vector<Value>& input) { return input[0] / input[1]; }
  // The aggregates skip missing arguments (ignores_missing), and are missing
  // only when every argument is. `combine` folds the values present.
  template <class Combine>
  inline static Value fold(const std::vector<Value>& input, Combine combine) {
    bool any = false;
    double result = 0.0;
    for (const auto& v : input) {
      if (v.missing) continue;
      result = any ? combine(result, v.value) : v.value;
      any = true;
    }
    return any ? Value(result) : Value();
  }
  inline static size_t count_present(const std::vector<Value>& input) {
    size_t n = 0;
    for (const auto& v : input) n += v.missing ? 0 : 1;
    return n;
  }
  inline static Value max(const std::vector<Value>& input) {
    return fold(input, [](double a, double b) { return b > a ? b : a; });
  }
  inline static Value min(const std::vector<Value>& input) {
    return fold(input, [](double a, double b) { return b < a ? b : a; });
  }
  inline static Value sum(const std::vector<Value>& input) {
    return fold(input, [](double a, double b) { return a + b; });
  }
  inline static Value avg(const std::vector<Value>& input) {
    Value result = sum(input);
    if (!result.missing) result.value /= static_cast<double>(count_present(input));
    return result;
  }
  inline static Value is_missing(const std::vector<Value>& input) {
    return Value(input[0].missing, DataType::DataTypeValue::BOOLEAN);
  }
  inline static Value is_notmissing(const std::vector<Value>& input) {
    return Value(!input[0].missing, DataType::DataTypeValue::BOOLEAN);
  }
  inline static Value equal(const std::vector<Value>& input) {
    return Value(input[0] == input[1], DataType::DataTypeValue::BOOLEAN);
  }
  inline static Value not_equal(const std::vector<Value>& input) {
    return Value(input[0] != input[1], DataType::DataTypeValue::BOOLEAN);
  }
  inline static Value less_than(const std::vector<Value>& input) {
    return Value(input[0] < input[1], DataType::DataTypeValue::BOOLEAN);
  }
  inline static Value less_thanorequal(const std::vector<Value>& input) {
    return Value(input[0] <= input[1], DataType::DataTypeValue::BOOLEAN);
  }
  inline static Value greater_than(const std::vector<Value>& input) {
    return Value(input[0] > input[1], DataType::DataTypeValue::BOOLEAN);
  }
  inline static Value greater_thanorequal(const std::vector<Value>& input) {
    return Value(input[0] >= input[1], DataType::DataTypeValue::BOOLEAN);
  }
  inline static Value exp(const std::vector<Value>& input) {
    return Value(std::exp(input[0].value), DataType::DataTypeValue::DOUBLE);
  }
  inline static Value ln(const std::vector<Value>& input) {
    return Value(std::log(input[0].value), DataType::DataTypeValue::DOUBLE);
  }
  inline static Value log10(const std::vector<Value>& input) {
    return Value(std::log10(input[0].value), DataType::DataTypeValue::DOUBLE);
  }
  inline static Value sqrt(const std::vector<Value>& input) {
    return Value(std::sqrt(input[0].value), DataType::DataTypeValue::DOUBLE);
  }
  inline static Value abs(const std::vector<Value>& input) {
    return Value(std::abs(input[0].value), DataType::DataTypeValue::DOUBLE);
  }
  inline static Value round(const std::vector<Value>& input) {
    return Value(std::round(input[0].value), DataType::DataTypeValue::DOUBLE);
  }
  inline static Value floor(const std::vector<Value>& input) {
    return Value(std::floor(input[0].value), DataType::DataTypeValue::DOUBLE);
  }
  inline static Value ceil(const std::vector<Value>& input) {
    return Value(std::ceil(input[0].value), DataType::DataTypeValue::DOUBLE);
  }
  inline static Value pow(const std::vector<Value>& input) {
    return Value(std::pow(input[0].value, input[1].value), DataType::DataTypeValue::DOUBLE);
  }
  // PMML if(condition, then-value[, else-value]); without an else part the
  // result is missing when the condition does not hold.
  inline static Value if_then_else(const std::vector<Value>& input) {
    if (input[0].value != 0.0 && !input[0].missing) return input[1];
    return input.size() > 2 ? input[2] : Value();
  }
  inline static Value is_in(const std::vector<Value>& input) {
    auto found = std::find(input.begin() + 1, input.end(), input[0]);
    return Value(found != input.end(), DataType::DataTypeValue::BOOLEAN);
  }
  inline static Value is_notin(const std::vector<Value>& input) {
    return Value(!(is_in(input).value), DataType::DataTypeValue::BOOLEAN);
  }

  inline static Value replace(const std::vector<Value>& input) {
    return input[0].replace(input[1].svalue, input[2].svalue);
  }
  inline static Value str_matches(const std::vector<Value>& input) {
    return Value(input[0].matches(input[1].svalue), DataType::DataTypeValue::BOOLEAN);
  }

  // String built-in functions
  inline static Value str_uppercase(const std::vector<Value>& input) {
    Value v = input[0];
    v.uppercase();
    return v;
  }

  inline static Value str_lowercase(const std::vector<Value>& input) {
    Value v = input[0];
    v.lowercase();
    return v;
  }

  // PMML substring uses 1-based startPos and a character count (length).
  inline static Value str_substring(const std::vector<Value>& input) {
    const size_t start_pos = static_cast<size_t>(input[1].value);  // 1-based
    const size_t length = static_cast<size_t>(input[2].value);
    Value v = input[0];
    if (start_pos >= 1) v.substr(start_pos - 1, length);
    return v;
  }

  inline static Value str_trim_blanks(const std::vector<Value>& input) {
    Value v = input[0];
    v.trim_blanks();
    return v;
  }

  inline static Value str_concat(const std::vector<Value>& input) {
    std::string result;
    for (const auto& v : input) result += v.svalue;
    return Value(result, DataType::DataTypeValue::STRING);
  }

  inline static Value str_length(const std::vector<Value>& input) {
    return Value(static_cast<double>(input[0].svalue.size()), DataType::DataTypeValue::DOUBLE);
  }

  // --- Functions the table was missing entirely ---
  //
  // PMML defines around sixty built-ins; cPMML implemented thirty-six of them
  // and misnamed two. The ones below are the arithmetic, boolean and
  // mathematical members whose definition is a single expression. What is still
  // absent, and why, is at the bottom of this file.

  // Every one-argument numeric built-in is the same wrapper around a different
  // <cmath> function.
  inline static std::function<Value(const std::vector<Value>&)> unary(double (*f)(double)) {
    return [f](const std::vector<Value>& input) {
      return Value(f(input[0].value), DataType::DataTypeValue::DOUBLE);
    };
  }

  // "Returns one if x>y and zero otherwise." Note it returns a number, not a
  // boolean, which is why it is not written with the comparison functions.
  inline static Value threshold(const std::vector<Value>& input) {
    return Value(input[0].value > input[1].value ? 1.0 : 0.0, DataType::DataTypeValue::DOUBLE);
  }

  /**
   * "the remainder after x is divided by y, computed according to XQuery
   *  rules", which the specification then pins down as
   *  a - floor(a/b)*b -- the sign of the divisor, like Python's %, not
   *  std::fmod's sign-of-the-dividend. Its own examples are the test:
   *  modulo(11,3)=2, modulo(-17.2,0.5)=0.3, modulo(9,-7)=-5, modulo(-4,-9)=-4.
   *  std::fmod would give 2, 0.3, 2 and -4 -- so two of the four differ.
   */
  inline static Value modulo(const std::vector<Value>& input) {
    const double a = input[0].value;
    const double b = input[1].value;

    return Value(a - std::floor(a / b) * b, DataType::DataTypeValue::DOUBLE);
  }

  inline static Value hypot(const std::vector<Value>& input) {
    return Value(std::hypot(input[0].value, input[1].value), DataType::DataTypeValue::DOUBLE);
  }

  inline static Value logical_not(const std::vector<Value>& input) {
    return Value(!(input[0].value != 0.0), DataType::DataTypeValue::BOOLEAN);
  }

  inline static Value product(const std::vector<Value>& input) {
    return fold(input, [](double a, double b) { return a * b; });
  }

  // The specification does not say what an even number of arguments gives. The
  // mean of the two central values is the usual reading and is what the
  // ensemble median already does here, so the two agree.
  inline static Value median(const std::vector<Value>& input) {
    std::vector<double> values;
    values.reserve(input.size());
    for (const auto& v : input)
      if (!v.missing) values.push_back(v.value);
    if (values.empty()) return Value();
    std::sort(values.begin(), values.end());

    const size_t mid = values.size() / 2;
    const double result = (values.size() % 2 == 0) ? (values[mid - 1] + values[mid]) / 2.0 : values[mid];

    return Value(result, DataType::DataTypeValue::DOUBLE);
  }

  // Φ(x) = ½·erfc(−x/√2). Written with erfc rather than 1−½erfc(x/√2) because
  // the latter loses every significant digit in the left tail.
  inline static Value std_normal_cdf(const std::vector<Value>& input) {
    return Value(0.5 * std::erfc(-input[0].value / std::sqrt(2.0)), DataType::DataTypeValue::DOUBLE);
  }

  inline static Value std_normal_pdf(const std::vector<Value>& input) {
    const double x = input[0].value;

    return Value(std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI), DataType::DataTypeValue::DOUBLE);
  }

  inline static Value normal_cdf(const std::vector<Value>& input) {
    const double sigma = input[2].value;
    if (!(sigma > 0.0)) throw cpmml::InvalidValueException("normalCDF: sigma must be positive");

    return Value(0.5 * std::erfc(-(input[0].value - input[1].value) / (sigma * std::sqrt(2.0))),
                 DataType::DataTypeValue::DOUBLE);
  }

  inline static Value normal_pdf(const std::vector<Value>& input) {
    const double sigma = input[2].value;
    if (!(sigma > 0.0)) throw cpmml::InvalidValueException("normalPDF: sigma must be positive");

    const double z = (input[0].value - input[1].value) / sigma;

    return Value(std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * M_PI)), DataType::DataTypeValue::DOUBLE);
  }

  // --- Typing ---
  //
  // The functions whose arguments or result the Apply constructor types (the
  // static types themselves are the binder's, P2.4: bind::expression_type
  // and typing.h apply_result_type, §3.4 and §4.1): every numeric function,
  // for a float argument, a float result, an integer "/" or pow.
  static bool is_numeric(const BuiltInFunctionType& type) {
    switch (type) {
      case BuiltInFunctionType::IF:
      case BuiltInFunctionType::IS_MISSING:
      case BuiltInFunctionType::IS_NOT_MISSING:
      case BuiltInFunctionType::EQUAL:
      case BuiltInFunctionType::NOT_EQUAL:
      case BuiltInFunctionType::LESS_THAN:
      case BuiltInFunctionType::LESS_THAN_EQUAL:
      case BuiltInFunctionType::GREATER_THAN:
      case BuiltInFunctionType::GREATER_THAN_EQUAL:
      case BuiltInFunctionType::IS_IN:
      case BuiltInFunctionType::IS_NOT_IN:
      case BuiltInFunctionType::NOT:
      case BuiltInFunctionType::REPLACE:
      case BuiltInFunctionType::MATCHES:
      case BuiltInFunctionType::UPPERCASE:
      case BuiltInFunctionType::LOWERCASE:
      case BuiltInFunctionType::SUBSTRING:
      case BuiltInFunctionType::TRIM_BLANKS:
      case BuiltInFunctionType::CONCAT:
      case BuiltInFunctionType::STRING_LENGTH:
      case BuiltInFunctionType::IDENTITY:
        return false;
      default:
        return true;
    }
  }

  // The comparisons. A float argument is a float here too, so that equal(f,
  // 0.1) and equal(f + 0, 0.1) agree: both compare the float 0.1f, widened to
  // a double ("implicit casts are allowed in the direction of less
  // restriction", Functions.html), with the double 0.1, and are false.
  static bool compares_numbers(const BuiltInFunctionType& type) {
    switch (type) {
      case BuiltInFunctionType::EQUAL:
      case BuiltInFunctionType::NOT_EQUAL:
      case BuiltInFunctionType::LESS_THAN:
      case BuiltInFunctionType::LESS_THAN_EQUAL:
      case BuiltInFunctionType::GREATER_THAN:
      case BuiltInFunctionType::GREATER_THAN_EQUAL:
      case BuiltInFunctionType::IS_IN:
      case BuiltInFunctionType::IS_NOT_IN:
        return true;
      default:
        return false;
    }
  }

  /**
   * Still not implemented, and each for a reason rather than by oversight:
   *
   *   isValid / isNotValid   need the field's DataDictionary Value constraints,
   *                          which a built-in taking a vector<Value> cannot
   *                          reach -- validity is decided in MiningSchema.
   *   normalIDF / stdNormalIDF
   *                          the inverse normal CDF, which needs a rational
   *                          approximation whose accuracy should be pinned by
   *                          an oracle before it is shipped.
   *   formatNumber / formatDatetime / dateDaysSinceYear /
   *   dateSecondsSinceYear / dateSecondsSinceMidnight
   *                          cPMML carries every date and time type as an
   *                          epoch double with no record of which epoch, so
   *                          there is nothing to format or offset from.
   *
   * All five groups fail at load with the name that was asked for, which is the
   * right answer until they exist.
   */
};

#endif
