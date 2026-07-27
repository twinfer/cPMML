
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_FIELDUSAGETYPE_H
#define CPMML_FIELDUSAGETYPE_H

#include <string>

#include "enum_parser.h"

/**
 * @class FieldUsageType
 *
 * Class representing  <a
 * href="http://dmg.org/pmml/v4-4/MiningSchema.html#xsdType_FIELD-USAGE-TYPE">PMML
 * FieldUsageType</a>.
 *
 * It defines the usage of a field in the mining schema.
 */
class FieldUsageType {
 public:
  enum class FieldUsageTypeValue { ACTIVE, TARGET, SUPPLEMENTARY, GROUP, ORDER, FREQUENCY_WEIGHT, ANALYSIS_WEIGHT };

  static constexpr EnumEntry<FieldUsageTypeValue> TABLE[] = {
      {"active", FieldUsageTypeValue::ACTIVE, "ACTIVE"},
      {"target", FieldUsageTypeValue::TARGET, "TARGET"},
      {"predicted", FieldUsageTypeValue::TARGET, "TARGET"},  // before 4.1
      {"supplementary", FieldUsageTypeValue::SUPPLEMENTARY, "SUPPLEMENTARY"},
      {"group", FieldUsageTypeValue::GROUP, "GROUP"},
      {"order", FieldUsageTypeValue::ORDER, "ORDER"},
      {"frequency_weight", FieldUsageTypeValue::FREQUENCY_WEIGHT, "FREQUENCY_WEIGHT"},
      {"frequencyweight", FieldUsageTypeValue::FREQUENCY_WEIGHT, "FREQUENCY_WEIGHT"},
      {"analysis_weight", FieldUsageTypeValue::ANALYSIS_WEIGHT, "ANALYSIS_WEIGHT"},
      {"analysisweight", FieldUsageTypeValue::ANALYSIS_WEIGHT, "ANALYSIS_WEIGHT"},
  };

  FieldUsageTypeValue value = FieldUsageTypeValue::ACTIVE;

  FieldUsageType() = default;

  explicit FieldUsageType(const std::string& field_usage_type) : value(from_string(field_usage_type)) {}

  // MiningField/@usageType defaults to active. The spec spells the weight
  // usages "frequencyWeight" and "analysisWeight"; the underscored forms were
  // the only ones in the table, so a conforming document fell through to ACTIVE
  // -- harmless while nothing distinguishes the two, but now that an
  // unrecognised usageType is a load error the conforming spelling has to be
  // here. Both are accepted.
  static FieldUsageTypeValue from_string(const std::string& field_usage_type) {
    return enum_value(field_usage_type, TABLE, "usageType", FieldUsageTypeValue::ACTIVE);
  }

  inline bool operator==(FieldUsageType other) const { return value == other.value; }
  inline bool operator==(FieldUsageTypeValue other) const { return value == other; }

  std::string to_string() const { return enum_display(value, TABLE, "ACTIVE"); }
};

#endif
