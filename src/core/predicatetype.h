
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_PREDICATETYPE_H
#define CPMML_PREDICATETYPE_H

#include <string>

#include "enum_parser.h"

/**
 * @class PredicateType
 *
 * Class representing the type of predicate.
 *
 * For instance:
 * - SimplePredicate
 * - SimpleSetPredicate
 * - CompoundPredicate
 * - True
 * - False
 *
 * See also <a
 * href="http://dmg.org/pmml/v4-4/TreeModel.html#xsdGroup_PREDICATE">PMML
 * Predicate</a>.
 */
class PredicateType {
 public:
  enum class PredicateTypeValue { TRUE, FALSE, SIMPLE, SIMPLESET, COMPOUND };

  static constexpr EnumEntry<PredicateTypeValue> TABLE[] = {
      {"True", PredicateTypeValue::TRUE},
      {"False", PredicateTypeValue::FALSE},
      {"SimplePredicate", PredicateTypeValue::SIMPLE},
      {"SimpleSetPredicate", PredicateTypeValue::SIMPLESET},
      {"CompoundPredicate", PredicateTypeValue::COMPOUND},
  };

  // Default-constructed and then assigned in several places; without an
  // initialiser the default constructor left this indeterminate and merely
  // reading it was undefined.
  PredicateTypeValue value = PredicateTypeValue::TRUE;

  PredicateType() = default;

  // Read from the element name.
  explicit PredicateType(const XmlNode& element)
      : value(read_enum_element(element, TABLE, "predicate element in " + element.parent_name())) {}

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }
};

#endif
