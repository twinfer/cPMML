
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
      // B25 (design commit P2.5): pmml-4-4.xsd Apply is "Extension* then
      // EXPRESSION*" (<xs:element ref="Extension" minOccurs="0"
      // maxOccurs="unbounded"/> <xs:group ref="EXPRESSION" minOccurs="0"
      // maxOccurs="unbounded"/>): an Extension is producer content, not an
      // argument. It was read as an expression element and refused. Skipped
      // wherever it stands, as BuiltInFunction counts the arguments and
      // Constant finds its parameter position (§3.4 "children after
      // Extension*").
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

      expressions.push_back(expression);
    }
  }

  // §3.4: Apply "mapMissingTo, defaultValue (xs:string) parsed as the
  // **Apply's** result type", not the enclosing DerivedField's (B11): they
  // stand for the function's result, so they are kept in its representation,
  // as the values the function returns are. The result type is the binder's
  // (design commit P2.4, bind::expression_type: the function table's result
  // rule over the arguments' static types). Where the binder does not know
  // it (arithmetic over a boolean or string operand, Q7, reported in report
  // mode and not yet a load error) the text is read as a double, the type the
  // function computes in. Stored as cpmml::bind::slot_constant says: `concat`
  // with mapMissingTo="5" in a double DerivedField gives the double 5, as
  // does `if` with defaultValue="5" over an untyped MapValues; `+` with
  // mapMissingTo="5" in a string DerivedField gives the string "5" that keeps
  // its number 5 (decided 2026-11-18, §9; JPMML 1.7.7 and the base build
  // agree).
  static Value result_constant(const XmlNode& node, const std::string& attribute, const DataType& slot) {
    using cpmml::bind::TypeId;
    const TypeId type = cpmml::bind::expression_type(node).value_or(TypeId::Double);
    return cpmml::bind::slot_constant(node, attribute, node.required_attribute(attribute), type, slot);
  }

  inline Value eval(const Sample& sample) const override {
    if (function.function_type == BuiltInFunction::BuiltInFunctionType::IF) return eval_if(sample);

    std::vector<Value> input;
    input.reserve(expressions.size());
    bool missing_input = false;
    for (const auto& expression : expressions) {
      input.push_back(expression->eval(sample));
      if (input.back().is_missing()) missing_input = true;
    }

    // isMissing/isNotMissing must receive their missing arguments; for any
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

    if (result.is_missing() && exist_defaultvalue) return defaultValue;
    return result;
  }

  // `if` (design commit P2.8b4; §4.2 "if laziness", §4.3 LazyIf; D3, D24).
  // Functions.html, Apply: "The optional attribute mapMissingTo defines the
  // output value for the cases when any of the function inputs are missing.
  // This means that if it is specified and any of the input values of the
  // function are missing then the function is not applied at all and the
  // mapMissingTo value is returned instead", and "when a defaultValue value is
  // provided, the function is applied first and if it produces a missing
  // value the defaultValue is returned instead". BuiltinFunctions.html, if:
  // "Implements IF-THEN-ELSE logic. The ELSE part is optional. If the ELSE
  // part is absent and the boolean value is false then a missing value is
  // returned." Read as a conditional, the inputs are the condition and the
  // branch it takes (§4.2), evaluated in that order; the other branch is not
  // evaluated, so its missing value is not an input. A missing condition is a
  // missing input (BuiltinFunctions.html: "Except as noted below, any missing
  // inputs to a built-in function will result in a missing value being
  // returned"; D3): mapMissingTo, else a missing result, then defaultValue.
  // A false condition without an ELSE part is a missing result, not a
  // missing input: defaultValue applies and mapMissingTo does not. The base
  // build passed `if` its missing arguments and never read mapMissingTo, and
  // took the ELSE branch for a missing condition (B20). JPMML 1.7.7 agrees on
  // every case but the missing condition without mapMissingTo, which it
  // refuses (MissingArgumentException; D3).
  inline Value eval_if(const Sample& sample) const {
    const Value condition = expressions[0]->eval(sample);
    if (condition.is_missing()) {
      if (exist_missingreplacement) return mapmissing_to;
      return exist_defaultvalue ? defaultValue : Value();
    }
    const size_t taken = condition.as_double() != 0.0 ? 1 : 2;
    if (taken >= expressions.size()) return exist_defaultvalue ? defaultValue : Value();
    const Value result = expressions[taken]->eval(sample);
    if (!result.is_missing()) return result;
    if (exist_missingreplacement) return mapmissing_to;
    return exist_defaultvalue ? defaultValue : result;
  }
};

#endif
