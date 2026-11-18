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

  // Design commit P2.5 (no silent fallbacks, B13): an expression that fails to
  // load fails the load, as it does in a DerivedField (§3.6: "Nothing is
  // caught and turned into a silent default"); it was caught here and the
  // OutputField was empty on every record. Output.html: "For this result
  // feature, OutputField must contain an EXPRESSION, unless it is used to
  // refer to a transformed value of a segment model through the segmentID
  // attribute." With neither, the field was empty on every record too; a
  // `value` attribute was read as the name of a field to copy, which the spec
  // does not define ("The attribute value is used in conjunction with result
  // features referring to specific values"). segmentId is not supported
  // (B34, P8.15). JPMML 1.7.7 refuses both: "Required element
  // OutputField/<Expression> is not defined".
  TransformedValue(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
                   const DataType& output_type)
      : OutputExpression(output_index, output_type, indexer) {
    const XmlNode child = node.get_child_bylist(expression_names);
    if (child.is_empty()) {
      const std::string name = node.optional_attribute("name").value_or("");
      if (node.exists_attribute("segmentId"))
        throw cpmml::ParsingException("OutputField " + name +
                                      ": a transformedValue of a segment (segmentId) is not supported");
      throw cpmml::ParsingException("OutputField " + name +
                                    ": feature transformedValue must contain an EXPRESSION (Output.html)");
    }
    expression = ExpressionBuilder::build(child, output_index, output_type, indexer);
    Expression::inputs = expression->inputs;
    value_kind = kind_of(child);
  }

  // The Java type a string OutputField writes the expression's number as
  // (OutputField::add_output), from the expression's static type (design
  // commit P2.4, bind::expression_type): nullopt for a string, whose text is
  // written as it is. Where the binder does not know the type (arithmetic
  // over a boolean or string operand, Q7, reported in report mode and not
  // yet a load error; a FieldRef to an OutputField whose dataType neither
  // the document nor its feature gives) the value is the double the engine
  // computed, and is written as one, as before P2.4.
  static std::optional<java::Kind> kind_of(const XmlNode& expression) {
    const cpmml::bind::StaticType type = cpmml::bind::expression_type(expression);
    if (!type) return java::Kind::DOUBLE;
    switch (*type) {
      case cpmml::bind::TypeId::String:
        return std::nullopt;
      case cpmml::bind::TypeId::Integer:
        return java::Kind::INTEGER;
      case cpmml::bind::TypeId::Float:
        return java::Kind::FLOAT;
      case cpmml::bind::TypeId::Boolean:
        return java::Kind::BOOLEAN;
      case cpmml::bind::TypeId::Double:
        return java::Kind::DOUBLE;
      case cpmml::bind::TypeId::Missing:
      case cpmml::bind::TypeId::Invalid:
        break;
    }
    throw std::logic_error("TransformedValue: a sentinel is not a static type");
  }

  inline Value eval(const Sample& sample) const override {
    return expression->eval(sample);
  }

  // The text of a string value; a number is written by OutputField::add_output
  // through value_kind.
  inline std::string eval_str(const Sample& sample, const InternalScore& /*score*/) const override {
    return expression->eval(sample).svalue;
  };

  inline double eval_double(const Sample& sample, const InternalScore& /*score*/) const override {
    const Value v = expression->eval(sample);
    if (v.missing) return double_min();
    // A number or boolean constant held as a string slot's text (Value::
    // numeric_text): a numeric OutputField reads its number.
    return v.numeric_text ? v.text_number() : v.value;
  };
};

#endif  // CPMML_SRC_EXPRESSION_TRANSFORMED_VALUE_H_
