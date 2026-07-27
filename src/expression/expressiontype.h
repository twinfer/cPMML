
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
      {"constant", ExpressionTypeValue::CONSTANT, "Constant"},
      {"fieldref", ExpressionTypeValue::FIELD_REF, "FieldRef"},
      {"normcontinuous", ExpressionTypeValue::NORM_CONTINUOUS, "NormContinuous"},
      {"normdiscrete", ExpressionTypeValue::NORM_DISCRETE, "NormDiscrete"},
      {"discretize", ExpressionTypeValue::DISCREZITE, "Discretize"},
      {"mapvalues", ExpressionTypeValue::MAP_VALUES, "MapValues"},
      {"textindex", ExpressionTypeValue::TEXT_INDEX, "TextIndex"},
      {"apply", ExpressionTypeValue::APPLY, "Apply"},
  };

  ExpressionTypeValue value = ExpressionTypeValue::CONSTANT;

  ExpressionType() = default;

  explicit ExpressionType(const std::string& value) : value(from_string(value)) {}

  inline bool operator==(const ExpressionType& other) const { return value == other.value; };

  inline bool operator!=(const ExpressionType& other) const { return value != other.value; };

  // Comes from the XML element name, so an unknown one is an EXPRESSION element
  // cPMML does not implement (Aggregate, Lag) rather than a typo.
  static ExpressionTypeValue from_string(const std::string& expressiontype) {
    return enum_value(expressiontype, TABLE, "expression type");
  }

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }
};

#endif
