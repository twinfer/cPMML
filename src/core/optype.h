
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_OPTYPE_H
#define CPMML_OPTYPE_H

#include <string>

#include "enum_parser.h"

/**
 * @class OpType
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/DataDictionary.html#xsdType_OPTYPE">PMML
 * OPTYPE</a>: whether a field is categorical, ordinal or continuous.
 */
class OpType {
 public:
  enum class OpTypeValue { CATEGORICAL, ORDINAL, CONTINUOUS, UNDEFINED };

  static constexpr EnumEntry<OpTypeValue> TABLE[] = {
      {"categorical", OpTypeValue::CATEGORICAL, "CATEGORICAL"},
      {"ordinal", OpTypeValue::ORDINAL, "ORDINAL"},
      {"continuous", OpTypeValue::CONTINUOUS, "CONTINUOUS"},
  };

  OpTypeValue value = OpTypeValue::UNDEFINED;

  OpType() = default;
  explicit OpType(const std::string& optype) : value(from_string(optype)) {}

  // optype is required on DataField but optional on MiningField and elsewhere,
  // where its absence means "inherit"; UNDEFINED is that state.
  static OpTypeValue from_string(const std::string& optype) {
    return enum_value(optype, TABLE, "optype", OpTypeValue::UNDEFINED);
  }

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }
};
#endif
