
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_PROPERTY_H
#define CPMML_PROPERTY_H

#include <string>

#include "enum_parser.h"

/**
 * @class Property
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/DataDictionary.html">PMML Property</a>.
 *
 * It used to define the constraints for admissible values in the
 * DataDictionary.
 */
class Property {
 public:
  enum class PropertyValue { VALID, INVALID, MISSING };

  static constexpr EnumEntry<PropertyValue> TABLE[] = {
      {"valid", PropertyValue::VALID, "VALID"},
      {"invalid", PropertyValue::INVALID, "INVALID"},
      {"missing", PropertyValue::MISSING, "MISSING"},
  };

  const PropertyValue value;

  // Value@property, default valid.
  explicit Property(const XmlNode& value_node)
      : value(read_enum(value_node, "property", TABLE, PropertyValue::VALID)) {}

  std::string to_string() const { return enum_display(value, TABLE, "VALID"); }
};

#endif
