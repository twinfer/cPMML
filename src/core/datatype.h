
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
      {"datetime", DataTypeValue::DOUBLE, "DOUBLE"},
      {"datedayssince[0]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"datedayssince[1960]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"datedayssince[1970]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"datedayssince[1980]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"timeseconds", DataTypeValue::DOUBLE, "DOUBLE"},
      {"datetimesecondssince[0]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"datetimesecondssince[1960]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"datetimesecondssince[1970]", DataTypeValue::DOUBLE, "DOUBLE"},
      {"datetimesecondssince[1980]", DataTypeValue::DOUBLE, "DOUBLE"},
  };

  DataTypeValue value = DataTypeValue::STRING;

  DataType() = default;

  DataType(const DataTypeValue& value) : value(value) {}

  DataType(const std::string& value) : value(from_string(value)) {}

  bool operator==(const DataType& other) const { return value == other.value; };

  bool operator!=(const DataType& other) const { return value != other.value; };

  // dataType is required wherever it appears.
  static DataTypeValue from_string(const std::string& data_type) {
    return enum_value(data_type, TABLE, "datatype");
  }

  // INTEGER is unreachable through from_string (integer maps to DOUBLE) but is
  // still a constructible enumerator, so it keeps its own spelling.
  std::string to_string() const {
    if (value == DataTypeValue::INTEGER) return "INTEGER";

    return enum_display(value, TABLE, "MISSING");
  }
};

#endif
