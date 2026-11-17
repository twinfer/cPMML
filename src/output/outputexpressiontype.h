
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

  // RESULT-FEATURE (4.4.1). The members cPMML does not compute stay in the
  // table as unsupported: an OutputField asking for one used to be answered
  // with the predicted value (a PASS_VALUE catch-all), as was any misspelt
  // feature, so feature="affinity" or feature="Probability" returned a
  // plausible number that was not what the document asked for.
  static constexpr EnumEntry<OutputExpressionTypeValue> TABLE[] = {
      {"predictedValue", OutputExpressionTypeValue::PREDICTED_VALUE},
      {"predictedDisplayValue", OutputExpressionTypeValue::PREDICTED_DISPLAY_VALUE},
      {"transformedValue", OutputExpressionTypeValue::TRANSFORMED_VALUE},
      {"decision", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
      {"probability", OutputExpressionTypeValue::PROBABILITY},
      {"affinity", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
      {"residual", OutputExpressionTypeValue::RESIDUAL},
      {"standardError", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
      {"standardDeviation", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
      {"clusterId", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
      {"clusterAffinity", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
      {"entityId", OutputExpressionTypeValue::ENTITY_ID},
      {"entityAffinity", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
      {"warning", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
      {"ruleValue", OutputExpressionTypeValue::RULE_VALUE},
      {"reasonCode", OutputExpressionTypeValue::REASON_CODE},
      {"antecedent", OutputExpressionTypeValue::ANTECEDENT},
      {"consequent", OutputExpressionTypeValue::CONSEQUENT},
      {"rule", OutputExpressionTypeValue::RULE},
      {"ruleId", OutputExpressionTypeValue::RULE_ID},
      {"confidence", OutputExpressionTypeValue::CONFIDENCE},
      {"support", OutputExpressionTypeValue::SUPPORT},
      {"lift", OutputExpressionTypeValue::LIFT},
      {"leverage", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
      {"confidenceIntervalLower", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
      {"confidenceIntervalUpper", OutputExpressionTypeValue::PREDICTED_VALUE, nullptr, EnumSupport::UNSUPPORTED},
  };

  OutputExpressionTypeValue value = OutputExpressionTypeValue::PREDICTED_VALUE;

  OutputExpressionType() = default;

  explicit OutputExpressionType(const OutputExpressionTypeValue& value) : value(value) {}

  inline bool operator==(const OutputExpressionType& other) const { return value == other.value; };

  inline bool operator!=(const OutputExpressionType& other) const { return value != other.value; };

  // OutputField@feature, default predictedValue.
  static OutputExpressionType read(const XmlNode& output_field) {
    return OutputExpressionType(read_enum(output_field, "feature", TABLE, OutputExpressionTypeValue::PREDICTED_VALUE));
  }

  std::string to_string() const { return enum_display(value, TABLE, "predictedValue"); }
};

#endif  // CPMML_SRC_EXPRESSION_OUTPUTEXPRESSIONTYPE_H_
