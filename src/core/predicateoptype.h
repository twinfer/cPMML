
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_PREDICATEOPTYPE_H
#define CPMML_PREDICATEOPTYPE_H

#include <string>

#include "enum_parser.h"

/**
 * @class PredicateOpType
 *
 * Class representing the operator for the condition enforced by a predicate.
 *
 * For instance:
 * - equal
 * - notEqual
 * - isIn
 * - surrogate
 * - etc.
 */
class PredicateOpType {
 public:
  enum class PredicateOpTypeValue {
    EQUAL,
    NOT_EQUAL,
    GREATER_THAN,
    GREATER_OR_EQUAL,
    LESS_THAN,
    LESS_OR_EQUAL,
    IS_IN,
    IS_NOT_IN,
    IS_MISSING,
    IS_NOT_MISSING,
    AND,
    OR,
    XOR,
    SURROGATE,
    TRUE,
    FALSE
  };

  static constexpr EnumEntry<PredicateOpTypeValue> TABLE[] = {
      {"equal", PredicateOpTypeValue::EQUAL, "=="},
      {"notequal", PredicateOpTypeValue::NOT_EQUAL, "!="},
      {"greaterthan", PredicateOpTypeValue::GREATER_THAN, ">"},
      {"greaterorequal", PredicateOpTypeValue::GREATER_OR_EQUAL, ">="},
      {"lessthan", PredicateOpTypeValue::LESS_THAN, "<"},
      {"lessorequal", PredicateOpTypeValue::LESS_OR_EQUAL, "<="},
      {"isin", PredicateOpTypeValue::IS_IN, "isIn"},
      {"isnotin", PredicateOpTypeValue::IS_NOT_IN, "isNotIn"},
      {"ismissing", PredicateOpTypeValue::IS_MISSING, "isMissing"},
      {"isnotmissing", PredicateOpTypeValue::IS_NOT_MISSING, "isNotMissing"},
      {"and", PredicateOpTypeValue::AND, "AND"},
      {"or", PredicateOpTypeValue::OR, "OR"},
      {"xor", PredicateOpTypeValue::XOR, "XOR"},
      {"surrogate", PredicateOpTypeValue::SURROGATE, "SURROGATE"},
      {"true", PredicateOpTypeValue::TRUE, "TRUE"},
      {"false", PredicateOpTypeValue::FALSE, "FALSE"},
  };

  // See PredicateType: default-constructed instances used to hold an
  // indeterminate value.
  PredicateOpTypeValue value = PredicateOpTypeValue::TRUE;

  PredicateOpType() = default;

  explicit PredicateOpType(const std::string& predicatetype) : value(from_string(predicatetype)) {}

  // SimplePredicate/@operator and CompoundPredicate/@booleanOperator are both
  // required.
  static PredicateOpTypeValue from_string(const std::string& predicate_type) {
    return enum_value(predicate_type, TABLE, "predicate operator");
  }

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }
};

#endif
