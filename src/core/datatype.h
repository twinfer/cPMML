
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

  // Each of the five non-temporal dataTypes is its own value (design commit
  // P2.4): DataDictionary.html, "The dataType reuses names and semantics of
  // atomic types in XML Schema", and pmml-4-4.xsd's DATATYPE enumerates
  // "integer" apart from "double", so an integer field is INTEGER, not a
  // double with a side flag. Values are still the old Value, in which every
  // number is a double (float keeps its float32 quantization only where a
  // reader narrows it); INTEGER changes how a value is read (an input "3.5"
  // is not an integer, MiningField::parse_input) and written (java_kind),
  // not how it is stored. The PMML 4.4 date/time types are epoch-based
  // numerics, carried as doubles until P10 (bind/typing.h parse_datatype
  // maps them to Double too).
  static constexpr EnumEntry<DataTypeValue> TABLE[] = {
      {"string", DataTypeValue::STRING, "STRING"},
      {"integer", DataTypeValue::INTEGER, "INTEGER"},
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

  std::string to_string() const { return enum_display(value, TABLE, "MISSING"); }
};

// The Java type a number of this dataType is written as (java_format.h):
// integer as a long, float as a float, boolean as true/false, every other
// numeric type -- the date and time types included, which cPMML carries as
// doubles -- as a double. A string has no number form; it is DOUBLE here and
// the caller writes its text instead.
inline java::Kind java_kind(const DataType& datatype) {
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
