/*******************************************************************************

 * Copyright 2019 AMADEUS. All rights reserved.

 * Author: Paolo Iannino

 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_PREDICTEDVALUE_H_
#define CPMML_SRC_EXPRESSION_PREDICTEDVALUE_H_

#include "outputexpression.h"

/**
 * @class PredictedValue
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Output.html#xsdType_RESULT-FEATURE">PMML
 * PredictedValue</a>.
 *
 * It is used to simply reference (and rename) the raw predicion value provided
 * by the model.
 */
class PredictedValue : public OutputExpression {
 public:
  std::string field_name;
  size_t index;

  PredictedValue() : index(std::numeric_limits<size_t>::max()) {}

  // The target's dataType: a boolean prediction is a value of it, true or
  // false, and not the text "true" or "false".
  DataType target_type;

  PredictedValue(const std::string& field_name, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
                 const DataType& output_type, const DataType& target_type)
      : OutputExpression(output_index, output_type, indexer),
        field_name(field_name),
        index(indexer->get_or_set(field_name)),
        target_type(target_type) {}

  inline Value eval(const Sample& sample) const override { return sample[index].value; }

  // The prediction after the Target: rescaled, clamped and cast for a
  // regression, and the class value, not the displayValue the Target names it
  // by, for a classification (raw_score holds the class when score is a
  // displayValue).
  inline virtual std::string eval_str(const Sample& sample, const InternalScore& score) const override {
    return score.raw_score.empty() ? score.score : score.raw_score;
  };

  // The number of that class. Reading double_score, the number of the
  // displayValue, gave a numeric predictedValue no value whenever a
  // TargetValue renamed the class (JPMML 1.7.7: class 1 shown as "one" is
  // predictedValue 1 as integer and 1.0 as double).
  //
  // A boolean is held as 1 or 0 (Value). PMML's dataTypes are XML Schema's
  // (DataDictionary: "The dataType reuses names and semantics of atomic types
  // in XML Schema"), whose boolean is written true, false, 1 or 0:
  //  - the prediction of a boolean target is that value, whatever dataType
  //    the OutputField has: 1 or 0 as an integer or a double (the spec gives
  //    no boolean-to-number cast; this is JPMML 1.7.7's, and XPath's);
  //  - into a boolean OutputField, a prediction of another type that is no
  //    number is read as a boolean literal, so the string class "true" is
  //    true. A text that is no boolean literal ("yes") has no value; a text
  //    that is neither a number nor a boolean has no number either ("true"
  //    of a string target as an integer).
  // These used to read the text as a number only, and "true" is none: a
  // boolean prediction had no value as a boolean, integer or double
  // OutputField, nor did a transformedValue reading it.
  inline std::optional<double> eval_double(const Sample& /*sample*/, const InternalScore& score) const override {
    const std::string& value = score.raw_score.empty() ? score.score : score.raw_score;
    if (target_type.value == DataType::DataTypeValue::BOOLEAN) return boolean_of(value);
    const std::optional<double> number = score.raw_score.empty() || score.raw_score == score.score
                                             ? score.double_score
                                             : InternalScore::value_of(score.raw_score);
    if (!number && output_type.value == DataType::DataTypeValue::BOOLEAN) return boolean_of(value);
    return number;
  };

 private:
  // 1 or 0 for an XSD boolean literal (true, false, 1, 0), else no value.
  static std::optional<double> boolean_of(const std::string& text) {
    bool parsed = false;
    if (lexical::parse_boolean(text, parsed)) return parsed ? 1.0 : 0.0;
    return std::nullopt;
  }
};

/**
 * @class PredictedDisplayValue
 *
 * Returns the display-value-transformed prediction (after Target processing).
 */
class PredictedDisplayValue : public OutputExpression {
 public:
  std::string field_name;
  size_t index;

  PredictedDisplayValue() : index(std::numeric_limits<size_t>::max()) {}

  PredictedDisplayValue(const std::string& field_name, const std::shared_ptr<Indexer>& indexer,
                        const size_t& output_index, const DataType& output_type)
      : OutputExpression(output_index, output_type, indexer),
        field_name(field_name),
        index(indexer->get_or_set(field_name)) {}

  inline Value eval(const Sample& sample) const override { return sample[index].value; }

  inline virtual std::string eval_str(const Sample& sample, const InternalScore& score) const override {
    return score.score;
  };

  inline std::optional<double> eval_double(const Sample& /*sample*/, const InternalScore& score) const override {
    return score.double_score;
  };
};

#endif  // CPMML_SRC_EXPRESSION_PREDICTEDVALUE_H_
