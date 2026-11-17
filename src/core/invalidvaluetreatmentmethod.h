
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
      {"returnInvalid", InvalidValueTreatmentMethodValue::RETURN_INVALID},
      {"asIs", InvalidValueTreatmentMethodValue::AS_IS},
      {"asMissing", InvalidValueTreatmentMethodValue::AS_MISSING},
      {"asValue", InvalidValueTreatmentMethodValue::AS_VALUE},
  };

  InvalidValueTreatmentMethodValue value = InvalidValueTreatmentMethodValue::RETURN_INVALID;

  InvalidValueTreatmentMethod() = default;

  explicit InvalidValueTreatmentMethod(InvalidValueTreatmentMethodValue value) : value(value) {}

  inline bool operator==(const InvalidValueTreatmentMethod& other) const { return value == other.value; };

  inline bool operator!=(const InvalidValueTreatmentMethod& other) const { return value != other.value; };

  // invalidValueTreatment on MiningField and Apply, default returnInvalid.
  static InvalidValueTreatmentMethod read(const XmlNode& node) {
    return InvalidValueTreatmentMethod(
        read_enum(node, "invalidValueTreatment", TABLE, InvalidValueTreatmentMethodValue::RETURN_INVALID));
  }

  std::string to_string() const { return enum_display(value, TABLE, "returnInvalid"); }
};

#endif
