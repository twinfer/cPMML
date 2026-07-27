
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_OUTLIERTREATMENTMETHOD_H
#define CPMML_OUTLIERTREATMENTMETHOD_H

#include <string>

#include "enum_parser.h"

/**
 * @class OutlierTreatmentMethod
 *
 * Class representing  <a
 * href="http://dmg.org/pmml/v4-4/MiningSchema.html#xsdType_OUTLIER-TREATMENT-METHOD">PMML
 * OUTLIER-TREATMENT-METHOD</a>.
 *
 * It defines how the model should behave when encountering a feature assuming a
 * value considered as an outlier.
 */
class OutlierTreatmentMethod {
 public:
  enum class OutlierTreatmentMethodValue {
    AS_IS,
    AS_MISSING_VALUES,
    AS_EXTREME_VALUES,
  };

  static constexpr EnumEntry<OutlierTreatmentMethodValue> TABLE[] = {
      {"asis", OutlierTreatmentMethodValue::AS_IS, "asIs"},
      {"asmissingvalues", OutlierTreatmentMethodValue::AS_MISSING_VALUES, "asMissingValues"},
      {"asextremevalues", OutlierTreatmentMethodValue::AS_EXTREME_VALUES, "asExtremeValues"},
  };

  OutlierTreatmentMethodValue value = OutlierTreatmentMethodValue::AS_IS;

  OutlierTreatmentMethod() = default;

  explicit OutlierTreatmentMethod(const std::string& value) : value(from_string(value)) {}

  inline bool operator==(const OutlierTreatmentMethod& other) const { return value == other.value; };

  inline bool operator!=(const OutlierTreatmentMethod& other) const { return value != other.value; };

  // MiningField/@outliers defaults to asIs. This one used to look the token up
  // without lower-casing it, so it only worked because its own constructor
  // lower-cased first; a direct from_string("asMissingValues") silently
  // returned asIs.
  static OutlierTreatmentMethodValue from_string(const std::string& outliertreatmentmethod) {
    return enum_value(outliertreatmentmethod, TABLE, "outliers", OutlierTreatmentMethodValue::AS_IS);
  }

  std::string to_string() const { return enum_display(value, TABLE, "asIs"); }
};

#endif
