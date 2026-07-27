
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_OUTPUTEXPRESSIONTYPE_H_
#define CPMML_SRC_EXPRESSION_OUTPUTEXPRESSIONTYPE_H_

#include <string>

#include "core/enum_parser.h"

/**
 * @class OutputExpressionType
 *
 * Class representing the <a
 * href="http://dmg.org/pmml/v4-4/Output.html#xsdType_RESULT-FEATURE">PMML
 * RESULT-FEATURE</a> an OutputField asks for.
 */
class OutputExpressionType {
 public:
  enum class OutputExpressionTypeValue {
    PREDICTED_VALUE,
    PREDICTED_DISPLAY_VALUE,
    TRANSFORMED_VALUE,
    PROBABILITY,
    ENTITY_ID,
    RESIDUAL,
    PASS_VALUE,
    // Association model output features (PMML 4.2+)
    RULE_VALUE,  // feature="ruleValue" (PMML 4.0/4.1 style, uses ruleFeature attr)
    ANTECEDENT,  // feature="antecedent"
    CONSEQUENT,  // feature="consequent"
    RULE,        // feature="rule"
    RULE_ID,     // feature="ruleId"
    SUPPORT,     // feature="support"
    CONFIDENCE,  // feature="confidence"
    LIFT,        // feature="lift"
    // Scorecard
    REASON_CODE  // feature="reasonCode", selected by the rank attribute
  };

  static constexpr EnumEntry<OutputExpressionTypeValue> TABLE[] = {
      {"predictedvalue", OutputExpressionTypeValue::PREDICTED_VALUE, "predictedValue"},
      {"predicteddisplayvalue", OutputExpressionTypeValue::PREDICTED_DISPLAY_VALUE, "predictedDisplayValue"},
      {"transformedvalue", OutputExpressionTypeValue::TRANSFORMED_VALUE, "transformedValue"},
      {"probability", OutputExpressionTypeValue::PROBABILITY, "probability"},
      {"entityid", OutputExpressionTypeValue::ENTITY_ID, "entityId"},
      {"residual", OutputExpressionTypeValue::RESIDUAL, "residual"},
      {"rulevalue", OutputExpressionTypeValue::RULE_VALUE, "ruleValue"},
      {"antecedent", OutputExpressionTypeValue::ANTECEDENT, "antecedent"},
      {"consequent", OutputExpressionTypeValue::CONSEQUENT, "consequent"},
      {"rule", OutputExpressionTypeValue::RULE, "rule"},
      {"ruleid", OutputExpressionTypeValue::RULE_ID, "ruleId"},
      {"support", OutputExpressionTypeValue::SUPPORT, "support"},
      {"confidence", OutputExpressionTypeValue::CONFIDENCE, "confidence"},
      {"lift", OutputExpressionTypeValue::LIFT, "lift"},
      {"reasoncode", OutputExpressionTypeValue::REASON_CODE, "reasonCode"},
  };

  OutputExpressionTypeValue value = OutputExpressionTypeValue::PREDICTED_VALUE;

  OutputExpressionType() = default;

  explicit OutputExpressionType(const OutputExpressionTypeValue& value) : value(value) {}

  explicit OutputExpressionType(const std::string& value) : value(from_string(value)) {}

  inline bool operator==(const OutputExpressionType& other) const { return value == other.value; };

  inline bool operator!=(const OutputExpressionType& other) const { return value != other.value; };

  /**
   * OutputField/@feature defaults to predictedValue.
   *
   * This is the one enumeration whose catch-all is deliberate rather than a
   * guess: PASS_VALUE is a real state meaning "a RESULT-FEATURE cPMML does not
   * compute", and the OutputField machinery handles it. So an unrecognised
   * feature stays a value here instead of becoming a load error.
   */
  static OutputExpressionTypeValue from_string(const std::string& expressiontype) {
    if (is_absent_attribute(expressiontype)) return OutputExpressionTypeValue::PREDICTED_VALUE;

    const EnumEntry<OutputExpressionTypeValue>* entry = enum_find(expressiontype, TABLE);
    return entry ? entry->value : OutputExpressionTypeValue::PASS_VALUE;
  }

  std::string to_string() const { return enum_display(value, TABLE, "passValue"); }
};

#endif  // CPMML_SRC_EXPRESSION_OUTPUTEXPRESSIONTYPE_H_
