
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_APPLY_H
#define CPMML_APPLY_H

#include "constant.h"
#include "discretize.h"
#include "expression.h"
#include "expressionbuilder.h"
#include "expressiontype.h"
#include "fieldref.h"
#include "fn/apply_node.h"
#include "mapvalues.h"
#include "normcontinuous.h"
#include "normdiscrete.h"
#include "textindex.h"

/**
 * @class Apply
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Functions.html#xsdElement_Apply">PMML
 * Apply</a>.
 *
 * It allows to perform user-defined transformations made of other PMML
 * Expressions or PMML Built-in functions. The function, its kernel and the
 * Apply's own attributes are cpmml::fn::ApplyNode's (fn/apply_node.h, design
 * §4.3, commit P4.1); this class builds the argument expressions.
 */
class Apply : public cpmml::fn::ApplyNode {
 public:
  Apply() = default;

  Apply(const XmlNode& node, const size_t& output_index, const DataType& output_type,
        const std::shared_ptr<Indexer>& indexer)
      : ApplyNode(node, output_index, output_type, indexer) {
    std::shared_ptr<Expression> expression;
    for (const auto& n : node.get_childs()) {
      // B25 (design commit P2.5): pmml-4-4.xsd Apply is "Extension* then
      // EXPRESSION*" (<xs:element ref="Extension" minOccurs="0"
      // maxOccurs="unbounded"/> <xs:group ref="EXPRESSION" minOccurs="0"
      // maxOccurs="unbounded"/>): an Extension is producer content, not an
      // argument. It was read as an expression element and refused. Skipped
      // wherever it stands, as ApplyNode counts the arguments and Constant
      // finds its parameter position (§3.4 "children after Extension*").
      if (n.name() == "Extension") continue;
      ExpressionType expression_type(n);
      switch (expression_type.value) {
        case ExpressionType::ExpressionTypeValue::CONSTANT:
          expression = std::make_shared<Constant>(n, output_index, output_type, indexer);
          break;  // to be refined in case of Parameter fields
        case ExpressionType::ExpressionTypeValue::FIELD_REF:
          expression = std::make_shared<FieldRef>(n, output_index, output_type, indexer);
          break;
        case ExpressionType::ExpressionTypeValue::NORM_CONTINUOUS:
          expression = std::make_shared<NormContinuous>(n, output_index, output_type, indexer);
          break;
        case ExpressionType::ExpressionTypeValue::NORM_DISCRETE:
          expression = std::make_shared<NormDiscrete>(n, output_index, output_type, indexer);
          break;
        case ExpressionType::ExpressionTypeValue::DISCREZITE:
          expression = std::make_shared<Discretize>(n, output_index, output_type, indexer);
          break;
        case ExpressionType::ExpressionTypeValue::APPLY:
          expression = std::make_shared<Apply>(n, output_index, output_type, indexer);
          break;
        case ExpressionType::ExpressionTypeValue::MAP_VALUES:
          expression = std::make_shared<MapValues>(n, output_index, output_type, indexer);
          break;
        case ExpressionType::ExpressionTypeValue::TEXT_INDEX:
          expression = std::make_shared<TextIndex>(n, output_index, output_type, indexer);
          break;
        default:
          throw cpmml::ParsingException("Expression " + expression_type.to_string() + " type not supported");
      }

      for (const auto& input : expression->inputs) inputs.insert(input);

      add_argument(expression);
    }
  }
};

#endif
