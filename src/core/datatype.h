
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_DATATYPE_H
#define CPMML_DATATYPE_H

#include <string>

#include "enum_parser.h"

/**
 * @class DataType
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/DataDictionary.html#xsdType_DATATYPE">PMML
 * DATATYPE</a>.
 */
class DataType {
 public:
  enum class DataTypeValue {
    STRING,
    INTEGER,
    FLOAT,
    DOUBLE,
    BOOLEAN,
  };

  // Every numeric type is carried as a double, except float, which keeps its
  // float32 quantization. The PMML 4.4 date/time types are epoch-based
  // numerics, so they are doubles too.
  static constexpr EnumEntry<DataTypeValue> TABLE[] = {
      {"string", DataTypeValue::STRING, "STRING"},
      {"integer", DataTypeValue::DOUBLE, "DOUBLE"},
      {"float", DataTypeValue::FLOAT, "FLOAT"},
      {"double", DataTypeValue::DOUBLE, "DOUBLE"},
      {"boolean", DataTypeValue::BOOLEAN, "BOOLEAN"},
      {"date", DataTypeValue::DOUBLE, "DOUBLE"},
      {"time", DataTypeValue::DOUBLE, "DOUBLE"},
      {"dateTime", DataTypeValue::DOUBLE, "DOUBLE"},
      {"dateDaysSince[0]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"dateDaysSince[1960]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"dateDaysSince[1970]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"dateDaysSince[1980]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"timeSeconds", DataTypeValue::DOUBLE, "DOUBLE"},
      {"dateTimeSecondsSince[0]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"dateTimeSecondsSince[1960]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"dateTimeSecondsSince[1970]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"dateTimeSecondsSince[1980]", DataTypeValue::DOUBLE, "DOUBLE"},
  };

  DataTypeValue value = DataTypeValue::STRING;

  DataType() = default;

  DataType(const DataTypeValue& value) : value(value) {}

  bool operator==(const DataType& other) const { return value == other.value; };

  bool operator!=(const DataType& other) const { return value != other.value; };

  // dataType is required on DataField and optional (inherited or inferred)
  // elsewhere; the caller decides by which overload it uses.
  static DataType read(const XmlNode& node) { return read_enum(node, "dataType", TABLE); }

  static DataType read(const XmlNode& node, DataTypeValue when_absent) {
    return read_enum(node, "dataType", TABLE, when_absent);
  }

  // dataType="integer" is carried as DOUBLE; this tells the two apart.
  static bool is_integer(const XmlNode& node) {
    return node.exists_attribute("dataType") && node.get_attribute("dataType") == "integer";
  }

  // INTEGER is unreachable through read() (integer maps to DOUBLE) but is
  // still a constructible enumerator, so it keeps its own spelling.
  std::string to_string() const {
    if (value == DataTypeValue::INTEGER) return "INTEGER";

    return enum_display(value, TABLE, "MISSING");
  }
};

#endif
