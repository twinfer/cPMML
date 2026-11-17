
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

  // What to_string() shows; True and False are the elements of that name.
  static constexpr EnumEntry<PredicateOpTypeValue> TABLE[] = {
      {"equal", PredicateOpTypeValue::EQUAL, "=="},
      {"notEqual", PredicateOpTypeValue::NOT_EQUAL, "!="},
      {"greaterThan", PredicateOpTypeValue::GREATER_THAN, ">"},
      {"greaterOrEqual", PredicateOpTypeValue::GREATER_OR_EQUAL, ">="},
      {"lessThan", PredicateOpTypeValue::LESS_THAN, "<"},
      {"lessOrEqual", PredicateOpTypeValue::LESS_OR_EQUAL, "<="},
      {"isIn", PredicateOpTypeValue::IS_IN, "isIn"},
      {"isNotIn", PredicateOpTypeValue::IS_NOT_IN, "isNotIn"},
      {"isMissing", PredicateOpTypeValue::IS_MISSING, "isMissing"},
      {"isNotMissing", PredicateOpTypeValue::IS_NOT_MISSING, "isNotMissing"},
      {"and", PredicateOpTypeValue::AND, "AND"},
      {"or", PredicateOpTypeValue::OR, "OR"},
      {"xor", PredicateOpTypeValue::XOR, "XOR"},
      {"surrogate", PredicateOpTypeValue::SURROGATE, "SURROGATE"},
      {"True", PredicateOpTypeValue::TRUE, "TRUE"},
      {"False", PredicateOpTypeValue::FALSE, "FALSE"},
  };

  // Each predicate element has its own operator enumeration, so a
  // CompoundPredicate with booleanOperator="isIn" is refused rather than read
  // as a set test with no field.
  static constexpr EnumEntry<PredicateOpTypeValue> SIMPLE_OPERATOR_TABLE[] = {
      {"equal", PredicateOpTypeValue::EQUAL},
      {"notEqual", PredicateOpTypeValue::NOT_EQUAL},
      {"lessThan", PredicateOpTypeValue::LESS_THAN},
      {"lessOrEqual", PredicateOpTypeValue::LESS_OR_EQUAL},
      {"greaterThan", PredicateOpTypeValue::GREATER_THAN},
      {"greaterOrEqual", PredicateOpTypeValue::GREATER_OR_EQUAL},
      {"isMissing", PredicateOpTypeValue::IS_MISSING},
      {"isNotMissing", PredicateOpTypeValue::IS_NOT_MISSING},
  };

  static constexpr EnumEntry<PredicateOpTypeValue> SET_OPERATOR_TABLE[] = {
      {"isIn", PredicateOpTypeValue::IS_IN},
      {"isNotIn", PredicateOpTypeValue::IS_NOT_IN},
  };

  static constexpr EnumEntry<PredicateOpTypeValue> COMPOUND_OPERATOR_TABLE[] = {
      {"or", PredicateOpTypeValue::OR},
      {"and", PredicateOpTypeValue::AND},
      {"xor", PredicateOpTypeValue::XOR},
      {"surrogate", PredicateOpTypeValue::SURROGATE},
  };

  // See PredicateType: default-constructed instances used to hold an
  // indeterminate value.
  PredicateOpTypeValue value = PredicateOpTypeValue::TRUE;

  PredicateOpType() = default;

  PredicateOpType(PredicateOpTypeValue value) : value(value) {}

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }
};

#endif
