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
    value_kind = child.is_empty() ? std::nullopt : kind_of(child.xml());
  }

  // The type of the value an expression gives, as JPMML 1.7.7 types it, read
  // from the document at load: nullopt for a string, else the Java type its
  // number is written as when a string OutputField holds it. Value carries no
  // type of its own, which is why this is worked out here.
  //   FieldRef        the dataType of the DataField, DerivedField or
  //                   OutputField it names;
  //   Constant        its dataType; without one, integer if the text is an
  //                   XSD integer, double if a double, else string ("7" is
  //                   written "7", "7.50" is "7.5");
  //   NormContinuous, NormDiscrete  double;
  //   Discretize, MapValues  their dataType, string by default;
  //   Apply           boolean for the predicates (equal ... isNotIn,
  //                   isMissing, matches), the common type of the arguments
  //                   (integer < float < double) for + - * / min max sum abs,
  //                   integer for round ceil floor, integer for
  //                   stringLength, string for the string functions, the
  //                   type of the second argument for if, double otherwise.
  // Arithmetic is still done in double: n / 2 of an integer n=3 is 1.5 in
  // cPMML and 1 in JPMML, written "1.5" here (a value, not a format, gap).
  static std::optional<java::Kind> kind_of(const pugi::xml_node& expression) {
    const std::string element = expression.name();
    if (element == "FieldRef") return field_kind(expression, expression.attribute("field").value());
    if (element == "Constant") {
      if (expression.attribute("dataType")) return declared_kind(expression.attribute("dataType").value());
      const std::string text = expression.child_value();
      long whole = 0;
      double number = 0;
      if (lexical::parse_long(text, whole)) return java::Kind::INTEGER;
      if (lexical::parse_double(text, number)) return java::Kind::DOUBLE;
      return std::nullopt;
    }
    if (element == "NormContinuous" || element == "NormDiscrete") return java::Kind::DOUBLE;
    if (element == "Discretize" || element == "MapValues")
      return declared_kind(expression.attribute("dataType") ? expression.attribute("dataType").value() : "string");
    if (element != "Apply") return java::Kind::DOUBLE;

    static const std::unordered_set<std::string> predicates{
        "equal",       "notEqual", "lessThan",  "lessOrEqual", "greaterThan", "greaterOrEqual", "and", "or",
        "not",         "isIn",     "isNotIn",   "isMissing",   "isNotMissing", "isValid",       "isNotValid",
        "matches"};
    static const std::unordered_set<std::string> common{"+", "-", "*", "/", "min", "max", "sum", "abs"};
    static const std::unordered_set<std::string> whole{"round", "ceil", "floor", "stringLength"};
    static const std::unordered_set<std::string> strings{"concat",    "substring", "uppercase",
                                                         "lowercase", "trimBlanks", "replace", "formatNumber",
                                                         "formatDatetime"};
    const std::string function = expression.attribute("function").value();
    if (predicates.count(function)) return java::Kind::BOOLEAN;
    if (whole.count(function)) return java::Kind::INTEGER;
    if (strings.count(function)) return std::nullopt;
    std::vector<pugi::xml_node> arguments;
    for (const auto& child : expression.children())
      if (expression_names.count(child.name())) arguments.push_back(child);
    if (function == "if") return arguments.size() > 1 ? kind_of(arguments[1]) : std::nullopt;
    if (!common.count(function)) return java::Kind::DOUBLE;
    java::Kind result = java::Kind::INTEGER;
    for (const auto& argument : arguments) {
      const auto kind = kind_of(argument);
      if (!kind || *kind == java::Kind::DOUBLE || *kind == java::Kind::BOOLEAN) return java::Kind::DOUBLE;
      if (*kind == java::Kind::FLOAT) result = java::Kind::FLOAT;
    }
    return result;
  }

  static std::optional<java::Kind> declared_kind(const std::string& datatype) {
    if (datatype == "string") return std::nullopt;
    if (datatype == "integer") return java::Kind::INTEGER;
    if (datatype == "float") return java::Kind::FLOAT;
    if (datatype == "boolean") return java::Kind::BOOLEAN;
    return java::Kind::DOUBLE;
  }

  // The dataType of the field `name` as the document declares it: the
  // nearest DerivedField or OutputField of that name, else the DataField.
  static std::optional<java::Kind> field_kind(const pugi::xml_node& from, const std::string& name) {
    const pugi::xml_node root = from.root().child("PMML");
    for (const char* element : {"DerivedField", "OutputField", "DataField"})
      for (const auto& found : root.select_nodes((std::string("//") + element).c_str()))
        if (name == found.node().attribute("name").value() && found.node().attribute("dataType"))
          return declared_kind(found.node().attribute("dataType").value());
    return java::Kind::DOUBLE;
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
