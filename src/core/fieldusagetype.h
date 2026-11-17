
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
      {"frequencyWeight", FieldUsageTypeValue::FREQUENCY_WEIGHT, "FREQUENCY_WEIGHT"},
      {"analysisWeight", FieldUsageTypeValue::ANALYSIS_WEIGHT, "ANALYSIS_WEIGHT"},
  };

  FieldUsageTypeValue value = FieldUsageTypeValue::ACTIVE;

  FieldUsageType() = default;

  explicit FieldUsageType(FieldUsageTypeValue value) : value(value) {}

  // MiningField@usageType, default active. The underscored spellings
  // frequency_weight and analysis_weight this table used to accept are not
  // PMML; JPMML 1.7.7 reads them as an absent usageType.
  static FieldUsageType read(const XmlNode& node) {
    return FieldUsageType(read_enum(node, "usageType", TABLE, FieldUsageTypeValue::ACTIVE));
  }

  inline bool operator==(FieldUsageType other) const { return value == other.value; }
  inline bool operator==(FieldUsageTypeValue other) const { return value == other; }

  std::string to_string() const { return enum_display(value, TABLE, "ACTIVE"); }
};

#endif
