
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_APPLY_H
#define CPMML_APPLY_H

#include "constant.h"
#include "core/builtinfunctions.h"
#include "discretize.h"
#include "expression.h"
#include "expressionbuilder.h"
#include "expressiontype.h"
#include "fieldref.h"
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
 * Expressions or PMML Built-in functions.
 */
class Apply : public Expression {
 public:
  BuiltInFunction function;
  bool exist_missingreplacement = false;
  Value mapmissing_to;
  bool exist_defaultvalue = false;
  Value defaultValue;
  InvalidValueTreatmentMethod invalidValueTreatmentMethod;
  std::vector<std::shared_ptr<Expression>> expressions;

  Apply() = default;

  Apply(const XmlNode& node, const size_t& output_index, const DataType& output_type,
        const std::shared_ptr<Indexer>& indexer)
      : Expression(output_index, output_type, indexer),
        function(BuiltInFunction(node)),
        exist_missingreplacement(node.exists_attribute("mapMissingTo")),
        mapmissing_to(exist_missingreplacement ? result_constant(node, "mapMissingTo", output_type) : Value()),
        exist_defaultvalue(node.exists_attribute("defaultValue")),
        defaultValue(exist_defaultvalue ? result_constant(node, "defaultValue", output_type) : Value()),
        invalidValueTreatmentMethod(InvalidValueTreatmentMethod::read(node)) {
    // asValue names MiningField/@invalidValueReplacement; an Apply has no
    // replacement to give, so the token is refused rather than read as asIs.
    if (invalidValueTreatmentMethod.value == InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_VALUE)
      throw cpmml::ParsingException("Apply: invalidValueTreatment=\"asValue\" is only defined for a MiningField");
    std::shared_ptr<Expression> expression;
    for (const auto& n : node.get_childs()) {
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

      expressions.push_back(expression);
    }
  }

  // §3.4: Apply "mapMissingTo, defaultValue (xs:string) parsed as the
  // **Apply's** result type", not the enclosing DerivedField's (B11): they
  // stand for the function's result, so they are kept in its representation,
  // as the values the function returns are. The result type is the one the
  // engine gives the Apply today (BuiltInFunction::expression_kind; P2.4 puts
  // the binder's in its place); a string result has no java::Kind. Stored as
  // cpmml::bind::slot_constant says: `concat` with mapMissingTo="5" in a
  // double DerivedField gives the double 5, as does `if` with
  // defaultValue="5" over an untyped MapValues, which expression_kind types as
  // a string; `+` with mapMissingTo="5" in a string DerivedField gives the
  // string "5" that keeps its number 5 (decided 2026-11-18, §9; JPMML 1.7.7
  // and the base build agree).
  static Value result_constant(const XmlNode& node, const std::string& attribute, const DataType& slot) {
    using cpmml::bind::TypeId;
    TypeId type = TypeId::String;
    if (const auto kind = BuiltInFunction::expression_kind(node.xml())) switch (*kind) {
        case java::Kind::INTEGER:
          type = TypeId::Integer;
          break;
        case java::Kind::FLOAT:
          type = TypeId::Float;
          break;
        case java::Kind::DOUBLE:
          type = TypeId::Double;
          break;
        case java::Kind::BOOLEAN:
          type = TypeId::Boolean;
          break;
      }
    return cpmml::bind::slot_constant(node, attribute, node.required_attribute(attribute), type, slot);
  }

  inline Value eval(const Sample& sample) const override {
    std::vector<Value> input;
    input.reserve(expressions.size());
    bool missing_input = false;
    for (const auto& expression : expressions) {
      input.push_back(expression->eval(sample));
      if (input.back().missing) missing_input = true;
    }

    // isMissing/isNotMissing/if must receive their missing arguments; for any
    // other function a missing argument makes the result missing, with
    // mapMissingTo (then defaultValue) taking over when given. The aggregates
    // skip missing arguments unless mapMissingTo is given.
    if (missing_input && !function.handles_missing()) {
      if (exist_missingreplacement) return mapmissing_to;
      if (!function.ignores_missing()) return exist_defaultvalue ? defaultValue : Value();
    }

    Value result;

    try {  // there is no rule in PMML for validation at this level, here
           // invalid is like "division by zero"
      // thus it is captured with an exception
      result = function(input);
    } catch (const std::exception&) {
      switch (invalidValueTreatmentMethod.value) {
        case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::RETURN_INVALID:
          throw cpmml::InvalidValueException("evaluating apply function");
        case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_MISSING:
          if (exist_missingreplacement)
            return mapmissing_to;
          else if (exist_defaultvalue)
            return defaultValue;
          else
            return Value();
        case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_IS:
        case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_VALUE:  // refused at load
          break;
      }
    }

    if (result.missing && exist_defaultvalue) return defaultValue;
    return result;
  }
};

#endif
