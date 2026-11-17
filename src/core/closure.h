
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_CLOSURE_H
#define CPMML_CLOSURE_H

#include <string>

#include "enum_parser.h"

/**
 * @class Closure
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/DataDictionary.html#interval">PMML
 * Closure</a>.
 *
 * It describes the type of boundaries for a range of continous values.
 * For instance:
 *      - closedClosed
 *      - closedOpen
 *      - etc.
 */
class Closure {
 public:
  enum class ClosureValue { CLOSED_CLOSED, OPEN_OPEN, CLOSED_OPEN, OPEN_CLOSED };

  static constexpr EnumEntry<ClosureValue> TABLE[] = {
      {"closedClosed", ClosureValue::CLOSED_CLOSED, "CLOSED_CLOSED"},
      {"openOpen", ClosureValue::OPEN_OPEN, "OPEN_OPEN"},
      {"closedOpen", ClosureValue::CLOSED_OPEN, "CLOSED_OPEN"},
      {"openClosed", ClosureValue::OPEN_CLOSED, "OPEN_CLOSED"},
  };

  const ClosureValue value;

  // Interval@closure is required.
  explicit Closure(const XmlNode& interval) : value(read_enum(interval, "closure", TABLE)) {}

  std::string to_string() const { return enum_display(value, TABLE, "OPEN_OPEN"); }
};

#endif
