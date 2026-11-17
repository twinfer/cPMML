
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
  explicit OpType(OpTypeValue value) : value(value) {}

  // optype is required on DataField and DerivedField; elsewhere (MiningField,
  // OutputField) its absence means "inherit", and UNDEFINED is that state.
  static OpType read(const XmlNode& node) { return OpType(read_enum(node, "optype", TABLE)); }

  static OpType read_optional(const XmlNode& node) {
    return OpType(read_enum(node, "optype", TABLE, OpTypeValue::UNDEFINED));
  }

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }
};
#endif
