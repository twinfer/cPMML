
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_BUILTINFUNCTIONS_H
#define CPMML_BUILTINFUNCTIONS_H

#include <cmath>

#include "cPMML.h"
#include "enum_parser.h"
#include "value.h"

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
    LOG,
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
    IDENTITY  // introduced to have "empty" function type
  };

  std::string function_string;
  BuiltInFunctionType function_type = BuiltInFunctionType::IDENTITY;
  int n_args = -1;
  int min_nargs = 0;
  std::function<Value(const std::vector<Value>&)> function;

  BuiltInFunction() = default;

  explicit BuiltInFunction(const std::string& function_string)
      : function_string(function_string),
        function_type(from_string(function_string)),
        n_args(get_nargs(function_type)),
        min_nargs(get_min_nargs(function_type)),
        function(get_function(function_type)) {}

  static constexpr EnumEntry<BuiltInFunctionType> TABLE[] = {
      {"+", BuiltInFunctionType::PLUS, "+"},
      {"-", BuiltInFunctionType::MINUS, "-"},
      {"*", BuiltInFunctionType::MUL, "*"},
      {"/", BuiltInFunctionType::DIV, "/"},
      {"max", BuiltInFunctionType::MAX, "max"},
      {"min", BuiltInFunctionType::MIN, "min"},
      {"sum", BuiltInFunctionType::SUM, "sum"},
      {"avg", BuiltInFunctionType::AVG, "avg"},
      {"exp", BuiltInFunctionType::EXP, "exp"},
      {"log", BuiltInFunctionType::LOG, "log"},
      {"log10", BuiltInFunctionType::LOG10, "log10"},
      {"sqrt", BuiltInFunctionType::SQRT, "sqrt"},
      {"abs", BuiltInFunctionType::ABS, "abs"},
      {"round", BuiltInFunctionType::ROUND, "round"},
      {"floor", BuiltInFunctionType::FLOOR, "floor"},
      {"ceiling", BuiltInFunctionType::CEIL, "ceiling"},
      {"pow", BuiltInFunctionType::POW, "pow"},
      {"if", BuiltInFunctionType::IF, "if"},
      {"ismissing", BuiltInFunctionType::IS_MISSING, "isMissing"},
      {"isnotmissing", BuiltInFunctionType::IS_NOT_MISSING, "isNotMissing"},
      {"equal", BuiltInFunctionType::EQUAL, "equal"},
      {"notequal", BuiltInFunctionType::NOT_EQUAL, "notEqual"},
      {"lessthan", BuiltInFunctionType::LESS_THAN, "lessThan"},
      {"lessorequal", BuiltInFunctionType::LESS_THAN_EQUAL, "lessOrEqual"},
      {"greaterthan", BuiltInFunctionType::GREATER_THAN, "greaterThan"},
      {"greaterorequal", BuiltInFunctionType::GREATER_THAN_EQUAL, "greaterOrEqual"},
      {"isin", BuiltInFunctionType::IS_IN, "isIn"},
      {"isnotin", BuiltInFunctionType::IS_NOT_IN, "isNotIn"},
      {"replace", BuiltInFunctionType::REPLACE, "replace"},
      {"matches", BuiltInFunctionType::MATCHES, "matches"},
      {"uppercase", BuiltInFunctionType::UPPERCASE, "uppercase"},
      {"lowercase", BuiltInFunctionType::LOWERCASE, "lowercase"},
      {"substring", BuiltInFunctionType::SUBSTRING, "substring"},
      {"trimblanks", BuiltInFunctionType::TRIM_BLANKS, "trimBlanks"},
      {"concat", BuiltInFunctionType::CONCAT, "concat"},
      {"stringlength", BuiltInFunctionType::STRING_LENGTH, "stringLength"},
  };

  // Apply/@function is required, and a name that is not a PMML built-in is
  // either a DefineFunction cPMML does not support or a typo. Either way there
  // is no function to apply.
  static BuiltInFunctionType from_string(const std::string& builtinfunction) {
    return enum_value(builtinfunction, TABLE, "function");
  }

  // Exact arity, or -1 for variadic functions (see get_min_nargs for those).
  static int get_nargs(const BuiltInFunctionType& function_type) {
    switch (function_type) {
      case BuiltInFunctionType::PLUS:
      case BuiltInFunctionType::MINUS:
      case BuiltInFunctionType::MUL:
      case BuiltInFunctionType::DIV:
      case BuiltInFunctionType::POW:
      case BuiltInFunctionType::EQUAL:
      case BuiltInFunctionType::NOT_EQUAL:
      case BuiltInFunctionType::LESS_THAN:
      case BuiltInFunctionType::LESS_THAN_EQUAL:
      case BuiltInFunctionType::GREATER_THAN:
      case BuiltInFunctionType::GREATER_THAN_EQUAL:
      case BuiltInFunctionType::MATCHES:
        return 2;
      case BuiltInFunctionType::IF:
        return -1;  // if(condition, then[, else]) — the else part is optional
      case BuiltInFunctionType::IS_MISSING:
      case BuiltInFunctionType::IS_NOT_MISSING:
      case BuiltInFunctionType::EXP:
      case BuiltInFunctionType::LOG:
      case BuiltInFunctionType::LOG10:
      case BuiltInFunctionType::SQRT:
      case BuiltInFunctionType::ABS:
      case BuiltInFunctionType::ROUND:
      case BuiltInFunctionType::FLOOR:
      case BuiltInFunctionType::CEIL:
      case BuiltInFunctionType::UPPERCASE:
      case BuiltInFunctionType::LOWERCASE:
      case BuiltInFunctionType::TRIM_BLANKS:
      case BuiltInFunctionType::STRING_LENGTH:
        return 1;
      case BuiltInFunctionType::REPLACE:
      case BuiltInFunctionType::SUBSTRING:
        return 3;
      default:
        return -1;  // variadic: max, min, sum, avg, concat, isIn, isNotIn
    }
  }

  // Minimum arity for variadic functions (n_args == -1).
  static int get_min_nargs(const BuiltInFunctionType& function_type) {
    switch (function_type) {
      case BuiltInFunctionType::IS_IN:
      case BuiltInFunctionType::IS_NOT_IN:
        return 2;  // isIn(candidate, member, ...)
      case BuiltInFunctionType::MAX:
      case BuiltInFunctionType::MIN:
      case BuiltInFunctionType::SUM:
      case BuiltInFunctionType::AVG:
      case BuiltInFunctionType::CONCAT:
        return 1;
      case BuiltInFunctionType::IF:
        return 2;  // if(condition, then[, else])
      default:
        return 0;
    }
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
      case BuiltInFunctionType::LOG:
        return log;
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
      default:
        throw cpmml::ParsingException("unsupported function");
    }
  }

  inline Value operator()(const std::vector<Value>& input) const {
    if (n_args != -1 && n_args != (int)input.size()) throw cpmml::InvalidValueException("Wrong number of inputs");
    if (n_args == -1 && (int)input.size() < min_nargs) throw cpmml::InvalidValueException("Wrong number of inputs");
    return function(input);
  }

  // True for functions that must receive missing arguments instead of having
  // Apply short-circuit them (PMML: isMissing/isNotMissing/if).
  inline bool handles_missing() const {
    return function_type == BuiltInFunctionType::IS_MISSING || function_type == BuiltInFunctionType::IS_NOT_MISSING ||
           function_type == BuiltInFunctionType::IF;
  }

  inline static Value plus(const std::vector<Value>& input) { return input[0] + input[1]; }
  inline static Value minus(const std::vector<Value>& input) { return input[0] - input[1]; }
  inline static Value mul(const std::vector<Value>& input) { return input[0] * input[1]; }
  inline static Value div(const std::vector<Value>& input) { return input[0] / input[1]; }
  inline static Value max(const std::vector<Value>& input) { return Value::max(input); }
  inline static Value min(const std::vector<Value>& input) { return Value::min(input); }
  inline static Value sum(const std::vector<Value>& input) { return Value::sum(input); }
  inline static Value avg(const std::vector<Value>& input) { return Value::sum(input) / Value(input.size()); }
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
  inline static Value log(const std::vector<Value>& input) {
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
};

#endif
