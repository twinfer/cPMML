
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_DATATYPE_H
#define CPMML_DATATYPE_H

#include <map>
#include <string>

#include "utils/utils.h"

/**
 * @class DataType
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/DataDictionary.html#xsdType_DATATYPE">PMML
 * dataTypes</a>.
 *
 * For instance:
 *      - STRING
 *      - INTEGER
 *      - BOOLEAN
 *      - etc.
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

  DataTypeValue value;

  DataType() = default;

  DataType(const DataTypeValue& value) : value(value) {}

  DataType(const std::string& value) : value(from_string(to_lower(value))) {}

  bool operator==(const DataType& other) const { return value == other.value; };

  bool operator!=(const DataType& other) const { return value != other.value; };

  static DataTypeValue from_string(const std::string& data_type) {
    const static std::unordered_map<std::string, DataTypeValue> datatype_converter = {
        {"string", DataTypeValue::STRING},
        {"integer", DataTypeValue::DOUBLE},  // every numeric type is treated as
                                             // a double, except float which
                                             // preserves float32 quantization
        {"float", DataTypeValue::FLOAT},
        {"double", DataTypeValue::DOUBLE},
        {"boolean", DataTypeValue::BOOLEAN},
        // PMML 4.4.1 date/time types — mapped to DOUBLE (epoch-based numeric)
        {"date", DataTypeValue::DOUBLE},
        {"time", DataTypeValue::DOUBLE},
        {"datetime", DataTypeValue::DOUBLE},
        {"datedayssince[0]", DataTypeValue::DOUBLE},
        {"datedayssince[1960]", DataTypeValue::DOUBLE},
        {"datedayssince[1970]", DataTypeValue::DOUBLE},
        {"datedayssince[1980]", DataTypeValue::DOUBLE},
        {"timeseconds", DataTypeValue::DOUBLE},
        {"datetimesecondssince[0]", DataTypeValue::DOUBLE},
        {"datetimesecondssince[1960]", DataTypeValue::DOUBLE},
        {"datetimesecondssince[1970]", DataTypeValue::DOUBLE},
        {"datetimesecondssince[1980]", DataTypeValue::DOUBLE},
    };

    return enum_from_string(datatype_converter, to_lower(data_type), "datatype");
  }

  std::string to_string() const {
    switch (value) {
      case DataTypeValue::STRING:
        return "STRING";
      case DataTypeValue::INTEGER:
        return "INTEGER";
      case DataTypeValue::FLOAT:
        return "FLOAT";
      case DataTypeValue::DOUBLE:
        return "DOUBLE";
      case DataTypeValue::BOOLEAN:
        return "BOOLEAN";
      default:
        return "MISSING";
    }
  }
};

#endif
