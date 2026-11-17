
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_INVALIDVALUETREATMENTMETHOD_H
#define CPMML_INVALIDVALUETREATMENTMETHOD_H

#include <string>

#include "enum_parser.h"

/**
 * @class InvalidValueTreatmentMethod
 *
 * Class representing  <a
 * href="http://dmg.org/pmml/v4-4/MiningSchema.html#xsdType_INVALID-VALUE-TREATMENT-METHOD">PMML
 * INVALID-VALUE-TREATMENT-METHOD</a>.
 *
 * It defines how the model should behave when encountering a feature assuming a
 * value not respecting the constraints defined in the DataDictionary.
 */
class InvalidValueTreatmentMethod {
 public:
  enum class InvalidValueTreatmentMethodValue { RETURN_INVALID, AS_IS, AS_MISSING, AS_VALUE };

  static constexpr EnumEntry<InvalidValueTreatmentMethodValue> TABLE[] = {
      {"returninvalid", InvalidValueTreatmentMethodValue::RETURN_INVALID, "returnInvalid"},
      {"asis", InvalidValueTreatmentMethodValue::AS_IS, "asIs"},
      {"asmissing", InvalidValueTreatmentMethodValue::AS_MISSING, "asMissing"},
      {"asvalue", InvalidValueTreatmentMethodValue::AS_VALUE, "asValue"},
  };

  InvalidValueTreatmentMethodValue value = InvalidValueTreatmentMethodValue::RETURN_INVALID;

  InvalidValueTreatmentMethod() = default;

  explicit InvalidValueTreatmentMethod(const std::string& value) : value(from_string(value)) {}

  inline bool operator==(const InvalidValueTreatmentMethod& other) const { return value == other.value; };

  inline bool operator!=(const InvalidValueTreatmentMethod& other) const { return value != other.value; };

  // MiningField/@invalidValueTreatment defaults to returnInvalid.
  static InvalidValueTreatmentMethodValue from_string(const std::string& invalidvaluetreatmentmethod) {
    return enum_value(invalidvaluetreatmentmethod, TABLE, "invalidValueTreatment",
                      InvalidValueTreatmentMethodValue::RETURN_INVALID);
  }

  std::string to_string() const { return enum_display(value, TABLE, "returnInvalid"); }
};

#endif
