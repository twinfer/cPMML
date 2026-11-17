
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
      {"asIs", OutlierTreatmentMethodValue::AS_IS},
      {"asMissingValues", OutlierTreatmentMethodValue::AS_MISSING_VALUES},
      {"asExtremeValues", OutlierTreatmentMethodValue::AS_EXTREME_VALUES},
  };

  OutlierTreatmentMethodValue value = OutlierTreatmentMethodValue::AS_IS;

  OutlierTreatmentMethod() = default;

  explicit OutlierTreatmentMethod(OutlierTreatmentMethodValue value) : value(value) {}

  inline bool operator==(const OutlierTreatmentMethod& other) const { return value == other.value; };

  inline bool operator!=(const OutlierTreatmentMethod& other) const { return value != other.value; };

  // outliers on MiningField and NormContinuous, default asIs on both.
  static OutlierTreatmentMethod read(const XmlNode& node) {
    return OutlierTreatmentMethod(read_enum(node, "outliers", TABLE, OutlierTreatmentMethodValue::AS_IS));
  }

  std::string to_string() const { return enum_display(value, TABLE, "asIs"); }
};

#endif
