
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
      {"asIs", MissingValueTreatmentMethodValue::AS_IS},
      {"asMean", MissingValueTreatmentMethodValue::AS_MEAN},
      {"asMode", MissingValueTreatmentMethodValue::AS_MODE},
      {"asMedian", MissingValueTreatmentMethodValue::AS_MEDIAN},
      {"asValue", MissingValueTreatmentMethodValue::AS_VALUE},
      // PMML 4.4: a missing input makes the result invalid. Not implemented;
      // scoring it as asIs would be a silent guess.
      {"returnInvalid", MissingValueTreatmentMethodValue::AS_IS, nullptr, EnumSupport::UNSUPPORTED},
  };

  MissingValueTreatmentMethodValue value = MissingValueTreatmentMethodValue::AS_IS;

  MissingValueTreatmentMethod() = default;

  explicit MissingValueTreatmentMethod(MissingValueTreatmentMethodValue value) : value(value) {}

  inline bool operator==(const MissingValueTreatmentMethod& other) const { return value == other.value; };

  inline bool operator!=(const MissingValueTreatmentMethod& other) const { return value != other.value; };

  // MiningField@missingValueTreatment, default asIs.
  static MissingValueTreatmentMethod read(const XmlNode& node) {
    return MissingValueTreatmentMethod(
        read_enum(node, "missingValueTreatment", TABLE, MissingValueTreatmentMethodValue::AS_IS));
  }

  std::string to_string() const { return enum_display(value, TABLE, "asIs"); }
};

#endif
