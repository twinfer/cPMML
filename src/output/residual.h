/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_RESIDUAL_H_
#define CPMML_SRC_EXPRESSION_RESIDUAL_H_

#include <limits>
#include <string>
#include <unordered_map>

#include "core/target.h"
#include "outputexpression.h"
#include "utils/lexical.h"

/**
 * @class Residual
 *
 * OutputExpression for feature="residual". Output.html: "For numeric
 * prediction this is the actual value minus the predicted value. For
 * classification this is [actual value = target value] minus the predicted
 * probability for the target value. The attribute value specifies the raw
 * target value. The term [actual value = target value] is defined as 1.0 if
 * the actual value is the same as the target value, and 0.0 otherwise."
 *
 * The target value is the OutputField's value attribute or, without one, the
 * predicted class (as for feature="probability": "When not specified, the
 * probability of the predicted categorical value should be returned"), always
 * the raw class, never its displayValue. "The same" is equality of values of
 * the target's dataType (B47): the actual value is an input of that type
 * (MiningField::parse_input), so the class text is read in that type too and
 * the two compare as values -- an integer or double target compares numbers
 * ("1" and "1.0" are one class of a double target), a float target compares
 * floats, a boolean target compares booleans, and only a string target
 * compares text. A class text that is not a value of the target's type is the
 * same as no actual value. The probability is "the predicted probability for
 * the target value", so it is found by value too: as Probability finds it
 * (Category: a numeric target compares numbers), and for a boolean target the
 * class whose text is the same boolean ("1" is "true", D11). It is 0.0 for a
 * class the distribution does not name. An empty value attribute is no value
 * attribute, as for Probability.
 *
 * A record with no actual value has no residual: "in order for residual values
 * to be computed the input data must include target values".
 */
class Residual : public OutputExpression {
 public:
  size_t target_index;

  Residual() : target_index(std::numeric_limits<size_t>::max()) {}

  Residual(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
           const DataType& output_type, const std::string& target_name, const DataType& target_type)
      : OutputExpression(output_index, output_type, indexer),
        target_index(target_name.empty() ? std::numeric_limits<size_t>::max() : indexer->get_index(target_name)),
        target_type(target_type.value),
        category(target_type),
        target_value(node.get_attribute("value", "")),
        has_target_value(!target_value.empty()) {
    if (has_target_value) target_class = read_class(target_value);
  }

  inline virtual std::optional<double> eval_double(const Sample& sample, const InternalScore& score) const override {
    if (target_index == std::numeric_limits<size_t>::max()) return std::nullopt;
    const Value& actual_val = sample[target_index].value;
    if (actual_val.is_missing()) return std::nullopt;

    // Classification: the model gave a probability distribution. Anything
    // else is a numeric prediction.
    if (!score.empty && !score.probabilities.empty()) {
      const std::string& key =
          has_target_value ? target_value : (score.raw_score.empty() ? score.score : score.raw_score);
      const Class cls = has_target_value ? target_class : read_class(key);
      const auto it = find_probability(score.probabilities, key, cls);
      if (has_target_value || it != score.probabilities.cend()) {
        const double x = same(actual_val, cls, key) ? 1.0 : 0.0;
        return x - (it != score.probabilities.cend() ? it->second : 0.0);
      }
    }

    // Regression: actual_value - predicted_value
    if (!score.double_score) return std::nullopt;
    return actual_val.as_double() - *score.double_score;
  }

 private:
  // A class text read in the target's dataType: `valid` is false when the
  // text is not a value of that type (it then equals no actual value).
  struct Class {
    bool valid = false;
    double number = 0.0;
  };

  DataType::DataTypeValue target_type;
  Category category;
  std::string target_value;
  bool has_target_value;
  Class target_class;

  inline Class read_class(const std::string& text) const {
    Class cls;
    switch (target_type) {
      case DataType::DataTypeValue::STRING:
        cls.valid = true;  // compared as text in same()
        break;
      case DataType::DataTypeValue::BOOLEAN: {
        bool b = false;
        cls.valid = lexical::parse_boolean(text, b);
        cls.number = b ? 1.0 : 0.0;
        break;
      }
      case DataType::DataTypeValue::INTEGER:
      case DataType::DataTypeValue::FLOAT:
      case DataType::DataTypeValue::DOUBLE:
        cls.valid = lexical::parse_double(text, cls.number);
        break;
    }
    return cls;
  }

  // The entry of probabilities for the target value: Category::find, which
  // compares numeric classes as numbers and others as text, and for a boolean
  // target, when no class is spelled `key`, the class read as the same boolean
  // (a distribution keyed "false"/"true" and value="1").
  inline auto find_probability(const std::unordered_map<std::string, double>& probabilities, const std::string& key,
                               const Class& cls) const -> std::unordered_map<std::string, double>::const_iterator {
    auto it = category.find(probabilities, key);
    if (it != probabilities.cend() || target_type != DataType::DataTypeValue::BOOLEAN || !cls.valid) return it;
    for (it = probabilities.cbegin(); it != probabilities.cend(); ++it) {
      bool b = false;
      if (lexical::parse_boolean(it->first, b) && (b ? 1.0 : 0.0) == cls.number) return it;
    }
    return it;
  }

  // [actual value = target value], in the target's dataType.
  inline bool same(const Value& actual, const Class& cls, const std::string& text) const {
    if (!cls.valid) return false;
    switch (target_type) {
      case DataType::DataTypeValue::STRING:
        return actual.text() == text;
      case DataType::DataTypeValue::FLOAT:
        return static_cast<float>(actual.as_double()) == static_cast<float>(cls.number);
      case DataType::DataTypeValue::BOOLEAN:
      case DataType::DataTypeValue::INTEGER:
      case DataType::DataTypeValue::DOUBLE:
        return actual.as_double() == cls.number;
    }
    return false;  // unreachable: the switch is exhaustive
  }
};

#endif  // CPMML_SRC_EXPRESSION_RESIDUAL_H_
