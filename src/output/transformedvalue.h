/*******************************************************************************

 * Copyright 2019 AMADEUS. All rights reserved.

 * Author: Paolo Iannino

 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_TRANSFORMED_VALUE_H_
#define CPMML_SRC_EXPRESSION_TRANSFORMED_VALUE_H_

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
  }

  inline Value eval(Sample& sample) const override {
    if (!expression) return Value();
    return expression->eval(sample);
  }

  inline std::string eval_str(Sample& sample, const InternalScore& /*score*/) const override {
    if (!expression) return "";
#ifndef REGEX_SUPPORT
    const Value v = expression->eval(sample);
    if (!v.missing) {
      // Attempt to reverse-map the encoded double back to its original string.
      const std::string s = Value::double_to_string(v.value);
      if (!s.empty()) return s;
      return std::to_string(v.value);
    }
    return "";
#else
    return expression->eval(sample).svalue;
#endif
  };
};

#endif  // CPMML_SRC_EXPRESSION_TRANSFORMED_VALUE_H_
