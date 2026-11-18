
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_DATATYPE_H
#define CPMML_DATATYPE_H

#include <string>

#include "enum_parser.h"
#include "utils/java_format.h"

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
    return node.optional_attribute("dataType") == "integer";
  }

  // INTEGER is unreachable through read() (integer maps to DOUBLE) but is
  // still a constructible enumerator, so it keeps its own spelling.
  std::string to_string() const {
    if (value == DataTypeValue::INTEGER) return "INTEGER";

    return enum_display(value, TABLE, "MISSING");
  }
};

// The Java type a number of this dataType is written as (java_format.h):
// integer (`integer`, which DataType carries as DOUBLE) as a long, float as a
// float, boolean as true/false, every other numeric type -- the date and time
// types included, which cPMML carries as doubles -- as a double. A string has
// no number form; it is DOUBLE here and the caller writes its text instead.
inline java::Kind java_kind(const DataType& datatype, const bool integer) {
  if (integer) return java::Kind::INTEGER;
  switch (datatype.value) {
    case DataType::DataTypeValue::INTEGER:
      return java::Kind::INTEGER;
    case DataType::DataTypeValue::FLOAT:
      return java::Kind::FLOAT;
    case DataType::DataTypeValue::BOOLEAN:
      return java::Kind::BOOLEAN;
    default:
      return java::Kind::DOUBLE;
  }
}

#endif
