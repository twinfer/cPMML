
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MISSINGVALUETREATMENTMETHOD_H
#define CPMML_MISSINGVALUETREATMENTMETHOD_H

#include <string>

#include "enum_parser.h"

/**
 * @class MissingValueTreatmentMethod
 *
 * Class representing  <a
 * href="http://dmg.org/pmml/v4-4/MiningSchema.html#xsdType_MISSING-VALUE-TREATMENT-METHOD">PMML
 * MISSING-VALUE-TREATMENT-METHOD</a>.
 *
 * It defines how the model should behave when encountering a feature whose
 * value is missing.
 */
class MissingValueTreatmentMethod {
 public:
  enum class MissingValueTreatmentMethodValue { AS_IS, AS_MEAN, AS_MODE, AS_MEDIAN, AS_VALUE };

  static constexpr EnumEntry<MissingValueTreatmentMethodValue> TABLE[] = {
      {"asis", MissingValueTreatmentMethodValue::AS_IS, "asIs"},
      {"asmean", MissingValueTreatmentMethodValue::AS_MEAN, "asMean"},
      {"asmode", MissingValueTreatmentMethodValue::AS_MODE, "asMode"},
      {"asmedian", MissingValueTreatmentMethodValue::AS_MEDIAN, "asMedian"},
      {"asvalue", MissingValueTreatmentMethodValue::AS_VALUE, "asValue"},
  };

  MissingValueTreatmentMethodValue value = MissingValueTreatmentMethodValue::AS_IS;

  MissingValueTreatmentMethod() = default;

  explicit MissingValueTreatmentMethod(const std::string& value) : value(from_string(value)) {}

  inline bool operator==(const MissingValueTreatmentMethod& other) const { return value == other.value; };

  inline bool operator!=(const MissingValueTreatmentMethod& other) const { return value != other.value; };

  // MiningField/@missingValueTreatment is optional; absent means asIs.
  static MissingValueTreatmentMethodValue from_string(const std::string& missingvaluetreatmentmethod) {
    return enum_value(missingvaluetreatmentmethod, TABLE, "missingValueTreatment",
                      MissingValueTreatmentMethodValue::AS_IS);
  }

  std::string to_string() const { return enum_display(value, TABLE, "asIs"); }
};

#endif
