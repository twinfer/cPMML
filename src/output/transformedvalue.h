/*******************************************************************************

 * Copyright 2019 AMADEUS. All rights reserved.

 * Author: Paolo Iannino

 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_TRANSFORMED_VALUE_H_
#define CPMML_SRC_EXPRESSION_TRANSFORMED_VALUE_H_

#include "core/builtinfunctions.h"
#include "expression/fieldref.h"
#include "outputexpression.h"

/**
 * @class TransformedValue
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_FieldRef">PMML
 * TransformedValue</a>.
 *
 * It allows to perform user-defined transformations made of other PMML
 * Expressions or PMML Built-in functions.
 */
class TransformedValue : public OutputExpression {
 public:
  std::shared_ptr<Expression> expression;

  TransformedValue() {}

  TransformedValue(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
                   const DataType& output_type)
      : OutputExpression(output_index, output_type, indexer) {
    XmlNode child = node.get_child_bylist(expression_names);
    if (!child.is_empty()) {
      try {
        expression = ExpressionBuilder::build(child, output_index, output_type, indexer);
      } catch (const cpmml::ParsingException&) {
        // Unsupported function — leave expression null; eval returns empty.
      }
    } else if (node.exists_attribute("value")) {
      // Non-standard: value= references another output field by name.
      // Treat as a FieldRef to that field.
      const std::string ref = node.get_attribute("value");
      expression = std::make_shared<FieldRef>(ref, indexer, output_index, output_type);
    }
    if (expression) Expression::inputs = expression->inputs;
    value_kind = child.is_empty() ? std::nullopt : kind_of(child.xml());
  }

  // The type of the value an expression gives, read from the document at
  // load: nullopt for a string, else the type its number is written as when a
  // string OutputField holds it. Value carries no type of its own; the rules,
  // shared with Apply, are BuiltInFunction::expression_kind's.
  static std::optional<java::Kind> kind_of(const pugi::xml_node& expression) {
    return BuiltInFunction::expression_kind(expression);
  }

  inline Value eval(const Sample& sample) const override {
    if (!expression) return Value();
    return expression->eval(sample);
  }

  // The text of a string value; a number is written by OutputField::add_output
  // through value_kind.
  inline std::string eval_str(const Sample& sample, const InternalScore& /*score*/) const override {
    if (!expression) return "";
    return expression->eval(sample).svalue;
  };

  inline double eval_double(const Sample& sample, const InternalScore& /*score*/) const override {
    if (!expression) return double_min();
    const Value v = expression->eval(sample);
    return v.missing ? double_min() : v.value;
  };
};

#endif  // CPMML_SRC_EXPRESSION_TRANSFORMED_VALUE_H_
