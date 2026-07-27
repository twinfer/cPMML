
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
      {"true", PredicateTypeValue::TRUE, "True"},
      {"false", PredicateTypeValue::FALSE, "False"},
      {"simplepredicate", PredicateTypeValue::SIMPLE, "SimplePredicate"},
      {"simplesetpredicate", PredicateTypeValue::SIMPLESET, "SimpleSetPredicate"},
      {"compoundpredicate", PredicateTypeValue::COMPOUND, "CompoundPredicate"},
  };

  // Default-constructed and then assigned in several places; without an
  // initialiser the default constructor left this indeterminate and merely
  // reading it was undefined.
  PredicateTypeValue value = PredicateTypeValue::TRUE;

  PredicateType() = default;

  explicit PredicateType(const std::string& predicatetype) : value(from_string(predicatetype)) {}

  // Comes from an XML element name, never from an attribute: there is nothing
  // to default to.
  static PredicateTypeValue from_string(const std::string& predicate_type) {
    return enum_value(predicate_type, TABLE, "predicate type");
  }

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }
};

#endif
