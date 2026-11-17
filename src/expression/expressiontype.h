
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_EXPRESSIONTYPE_H
#define CPMML_EXPRESSIONTYPE_H

#include <string>

#include "core/enum_parser.h"

/**
 * @class ExpressionType
 *
 * Class representing the type of an <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdGroup_EXPRESSION">PMML
 * EXPRESSION</a> element.
 */
class ExpressionType {
 public:
  enum class ExpressionTypeValue {
    CONSTANT,
    FIELD_REF,
    NORM_CONTINUOUS,
    NORM_DISCRETE,
    DISCREZITE,
    MAP_VALUES,
    TEXT_INDEX,
    APPLY,
  };

  static constexpr EnumEntry<ExpressionTypeValue> TABLE[] = {
      {"Constant", ExpressionTypeValue::CONSTANT},
      {"FieldRef", ExpressionTypeValue::FIELD_REF},
      {"NormContinuous", ExpressionTypeValue::NORM_CONTINUOUS},
      {"NormDiscrete", ExpressionTypeValue::NORM_DISCRETE},
      {"Discretize", ExpressionTypeValue::DISCREZITE},
      {"MapValues", ExpressionTypeValue::MAP_VALUES},
      {"TextIndex", ExpressionTypeValue::TEXT_INDEX},
      {"Apply", ExpressionTypeValue::APPLY},
      {"Aggregate", ExpressionTypeValue::CONSTANT, nullptr, EnumSupport::UNSUPPORTED},
      {"Lag", ExpressionTypeValue::CONSTANT, nullptr, EnumSupport::UNSUPPORTED},
  };

  ExpressionTypeValue value = ExpressionTypeValue::CONSTANT;

  ExpressionType() = default;

  // Read from the element name. Aggregate and Lag are EXPRESSION elements
  // cPMML does not implement.
  explicit ExpressionType(const XmlNode& element)
      : value(read_enum_element(element, TABLE, "expression element in " + element.parent_name())) {}

  inline bool operator==(const ExpressionType& other) const { return value == other.value; };

  inline bool operator!=(const ExpressionType& other) const { return value != other.value; };

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }
};

#endif
