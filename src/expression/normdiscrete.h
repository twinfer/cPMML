
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_NORMDISCRETE_H
#define CPMML_NORMDISCRETE_H

#include <cstdlib>
#include <utility>

#include "bind/load_scope.h"
#include "expression.h"

/**
 * @class NormDiscrete
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_NormDiscrete">PMML
 * NormDiscrete</a>.
 *
 * It performs the encoding of a value of a field into 1.0 or 0.0.
 *
 * Transformations.html: "An element (f, v) defines that the unit has value
 * 1.0 if the value of input field f is v, otherwise it is 0." v is a value of
 * f (pmml-4-4.xsd: NormDiscrete@value is xs:string), read by §3.4 in f's type
 * (B10: it was always read as a string, so it never matched an integer,
 * double or boolean field); mapMissingTo is NUMBER (xs:double), not a value
 * of the enclosing DerivedField's type (B11).
 */
class NormDiscrete : public Expression {
 public:
  bool exist_missingreplacement = false;
  std::string field_name;
  size_t index = std::numeric_limits<size_t>::max();
  Value value;
  Value mapmissing_to;

  NormDiscrete() = default;

  NormDiscrete(const XmlNode& node, const unsigned int& output_index, const DataType& output_type,
               const std::shared_ptr<Indexer>& indexer)
      : Expression(output_index, output_type, indexer),
        exist_missingreplacement(node.exists_attribute("mapMissingTo")),
        field_name(node.required_attribute("field")),
        index(indexer->get_or_set(field_name)),
        value(Value(node.required_attribute("value"), cpmml::bind::scope_resolve(*indexer, node, field_name).datatype)),
        mapmissing_to(exist_missingreplacement ? cpmml::bind::number_constant(node, "mapMissingTo", output_type) : Value()) {
    inputs.insert(field_name);
  }

  inline Value eval(const Sample& sample) const override {
    Value input = sample[index].value;
    if (input.missing) return mapmissing_to;
    if (input == value) return Value(1, output_type);

    return Value(0, output_type);
  }
};

#endif
