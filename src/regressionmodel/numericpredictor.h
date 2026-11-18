
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_NUMERICPREDICTOR_H
#define CPMML_NUMERICPREDICTOR_H

#include <cmath>
#include <string>

#include "bind/load_scope.h"
#include "core/datatype.h"
#include "core/sample.h"
#include "core/value.h"
#include "core/xmlnode.h"

/**
 * @class NumericPredictor
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Regression.html#xsdElement_NumericPredictor">PMML
 * NumericPredictor</a>.
 *
 * It contains the parameters (coefficients and exponents) needed to compute the
 * regressed value of a continuous variable.
 */
class NumericPredictor {
 public:
  std::string name;
  size_t index = std::numeric_limits<size_t>::max();
  DataType datatype;
  double coefficient = 0.0;
  double exponent = 1.0;

  std::function<double(const Value&)> numeric_predictor;

  NumericPredictor() = default;

  NumericPredictor(const XmlNode& node,  // NumericPredictor node
                   const std::shared_ptr<Indexer>& indexer)
      : NumericPredictor(node, cpmml::bind::scope_resolve(*indexer, node, node.required_attribute("name"))) {}

  // scope.resolve (P2.1): the field's slot and declared type.
  NumericPredictor(const XmlNode& node, const cpmml::bind::ResolvedField& field)
      : name(node.required_attribute("name")),
        index(field.index),
        datatype(field.datatype),
        coefficient(node.required_double_attribute("coefficient")),
        // pmml-4-4.xsd: <xs:attribute name="exponent" type="INT-NUMBER"
        // default="1"/>, and INT-NUMBER is xs:integer: "2.5" is a load error
        // (B7: it was read as a double, so 4^2.5 gave 32). JPMML 1.7.7 loads
        // it and scores x^1.
        exponent(static_cast<double>(node.get_long_attribute("exponent", 1))) {}

  inline double get_term(const Sample& sample) const {
    const Value& value = sample[index].value;
    return value.missing ? 0 : coefficient * std::pow(value.value, exponent);
  }

  static std::vector<NumericPredictor> to_numericpredictors(const std::vector<XmlNode>& nodes,
                                                            const std::shared_ptr<Indexer>& indexer) {
    std::vector<NumericPredictor> result;

    for (const auto& node : nodes) result.push_back(NumericPredictor(node, indexer));

    return result;
  }
};

#endif
