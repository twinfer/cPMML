
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_DERIVEDFIELD_H
#define CPMML_DERIVEDFIELD_H

#include "expression/expression.h"
#include "expression/expressionbuilder.h"
#include "sample.h"
#include "xmlnode.h"

/**
 * @class DerivedField
 * Class representing a <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_DerivedField">PMML
 * DerivedField</a>.
 *
 * It defines a feature available to the model, which is obtained by
 * transforming other features. The associated transformation is represented by
 * the class Expression.
 *
 * @see Expression.
 */
class DerivedField {
 public:
  std::string name;
  OpType optype;
  DataType datatype;
  size_t index = std::numeric_limits<size_t>::max();
  std::shared_ptr<Expression> expression;

  DerivedField() = default;

  explicit DerivedField(const XmlNode& node, const std::shared_ptr<Indexer>& indexer)
      // pmml-4-4.xsd: DerivedField@name is optional; it is unnamed only inside
      // a NeuralInput/NeuralOutput, and DerivedField::declared requires it
      // everywhere else.
      : name(node.get_attribute("name", "")),
        optype(OpType::read(node)),
        datatype(DataType::read(node)),
        index(indexer->get_or_set(name, datatype).first),
        expression(ExpressionBuilder::build(node.get_child_bylist(expression_names), index, datatype, indexer)) {}

  // A DerivedField of a TransformationDictionary or LocalTransformations: its
  // name is how it is referenced, so an absent one is a load error.
  static DerivedField declared(const XmlNode& node, const std::shared_ptr<Indexer>& indexer) {
    node.required_attribute("name");
    return DerivedField(node, indexer);
  }

  inline void prepare(Sample& sample) const {
    Value v = expression->eval(sample);
    // Quantize to float32 when DerivedField declares dataType="float", to avoid precision issues in case of float32
    // quantization in the original model
    if (datatype.value == DataType::DataTypeValue::FLOAT && !v.missing)
      v.value = static_cast<double>(static_cast<float>(v.value));
    sample.change_value(index, v);
  }

  inline static std::unordered_map<std::string, DerivedField> to_derivedfields(
      const std::vector<XmlNode>& nodes, const std::shared_ptr<Indexer>& indexer) {
    std::unordered_map<std::string, DerivedField> result;
    for (const auto& node : nodes)
      result.insert(std::make_pair(node.required_attribute("name"), declared(node, indexer)));

    return result;
  }
};

#endif
