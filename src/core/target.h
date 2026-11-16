/*******************************************************************************

 * Copyright 2019 AMADEUS. All rights reserved.

 * Author: Paolo Iannino

 *******************************************************************************/

#ifndef CPMML_SRC_CORE_TARGET_H_
#define CPMML_SRC_CORE_TARGET_H_

#include <functional>
#include <string>
#include <unordered_map>

#include "internal_score.h"
#include "math/misc.h"
#include "miningfunction.h"
#include "miningschema.h"
#include "optype.h"
#include "transformationdictionary.h"
#include "xmlnode.h"

/**
 * @class Target
 *
 * Class representing <a href="http://dmg.org/pmml/v4-4/Targets.html">PMML
 * Target</a>.
 *
 * It defines transformations for the raw output of a model.
 */
class CastInteger {
 public:
  bool empty = true;
  std::function<double(const double)> cast_function;

  CastInteger() = default;
  explicit CastInteger(const std::string& cast_integer) {
    static std::unordered_map<std::string, std::function<double(const double)>> cast_integer_converter = {
        {"round", _round},
        {"ceiling", _ceil},
        {"floor", _floor},
    };

    auto result = cast_integer_converter.find(cast_integer);
    if (result != cast_integer_converter.cend()) {
      empty = false;
      cast_function = result->second;
    } else {
      empty = true;
      cast_function = _identity;
    }
  }

  // Not constexpr: no instantiation of this can be a constant expression, since
  // it calls std::function::operator(). That makes it ill-formed, no diagnostic
  // required -- a compiler is free to reject it and none is obliged to say why.
  inline double operator()(const double value) const { return cast_function(value); }
};

class TargetValue {
 public:
  std::string value;
  bool has_display_value;
  std::string display_value;
  bool has_prior_probability;
  double prior_probability;
  bool has_default_value;
  double default_value;
  // TODO: Partition

  TargetValue()
      : has_display_value(false),
        has_prior_probability(false),
        prior_probability(double_min()),
        has_default_value(false),
        default_value(double_min()) {}

  TargetValue(const XmlNode& node)
      : value(node.get_attribute("value")),
        has_display_value(node.exists_attribute("displayValue")),
        display_value(node.get_attribute("displayValue")),
        has_prior_probability(node.exists_attribute("priorProbability")),
        prior_probability(node.get_double_attribute("priorProbability")),
        has_default_value(node.exists_attribute("defaultValue")),
        default_value(node.get_double_attribute("defaultValue")) {}

  inline const std::string& label() const { return has_display_value ? display_value : value; }

  static std::vector<TargetValue> to_targetvalues(std::vector<XmlNode> nodes) {
    std::vector<TargetValue> result;
    for (const auto& node : nodes) result.push_back(TargetValue(node));

    return result;
  }
};

class Target {
 public:
  MiningFunction mining_function;
  bool has_field_name = false;
  std::string field_name;
  CastInteger cast;
  bool has_min = false;
  double min = double_min();
  bool has_max = false;
  double max = double_min();
  bool has_rescale_constant = false;
  double rescale_constant = double_min();
  bool has_rescale_factor = false;
  double rescale_factor = double_min();
  OpType optype;
  std::vector<TargetValue> target_values;
  std::function<void(InternalScore&)> transform_score;

  Target() = default;

  explicit Target(const XmlNode& node, const MiningSchema& mining_schema,
                  const TransformationDictionary& transformation_dictionary, const MiningFunction& mining_fuction)
      : mining_function(mining_fuction),
        has_field_name(node.exists_attribute("field")),
        field_name(node.get_attribute("field")),
        cast(node.get_attribute("castInteger")),
        has_min(node.exists_attribute("min")),
        min(node.get_double_attribute("min")),
        has_max(node.exists_attribute("max")),
        max(node.get_double_attribute("max")),
        has_rescale_constant(node.exists_attribute("rescaleConstant")),
        rescale_constant(node.get_double_attribute("rescaleConstant")),
        has_rescale_factor(node.exists_attribute("rescaleFactor")),
        rescale_factor(node.get_double_attribute("rescaleFactor")),
        target_values(TargetValue::to_targetvalues(node.get_childs("TargetValue"))) {}

  // When the model produced no prediction, Targets supply one: the class with
  // the highest priorProbability, or the defaultValue of a continuous target.
  // The spec leaves "the exact rules" to each model; the rules below are the
  // ones JPMML 1.7.7 applies to every model type:
  // - classification: the priors, normalized to sum to 1, become the
  //   probabilities, and the predicted class is the one with the largest prior.
  //   Between equal priors the first TargetValue in document order wins.
  // - regression: defaultValue goes through min/max, rescaleFactor,
  //   rescaleConstant and castInteger like any predicted value (42 with
  //   rescaleFactor 2 and rescaleConstant 1 predicts 85).
  // A Target with neither leaves the score without a prediction.
  inline void operator()(InternalScore& score) const {
    switch (mining_function.value) {
      case MiningFunction::MiningFunctionType::CLASSIFICATION:
        if (score.empty) {
          const TargetValue* best = apply_priors(score);
          if (best == nullptr) break;
          score.empty = false;
          score.raw_score = best->value;
          score.score = best->label();
          try {
            score.double_score = to_double(score.score);
          } catch (const cpmml::ParsingException& e) {
            score.double_score = double_min();
          }
          break;
        }

        for (const auto& target_value : target_values) {
          if (score.score == target_value.value && target_value.has_display_value) {
            score.score = target_value.display_value;
            try {
              score.double_score = to_double(score.score);
            } catch (const cpmml::ParsingException& e) {
              score.double_score = double_min();
            }
          }

          const auto it = score.probabilities.find(target_value.value);
          if (it != score.probabilities.cend()) {
            // Class present in the score: rename its key to the display value.
            if (target_value.has_display_value && target_value.display_value != target_value.value) {
              score.probabilities[target_value.display_value] = it->second;
              score.probabilities.erase(target_value.value);
            }
          } else if (!score.probabilities.empty()) {
            // Class absent from a distribution the model did produce: its
            // probability is 0. Priors apply only when there is no prediction;
            // filling them in here made the probabilities sum past 1.
            score.probabilities[target_value.label()] = 0.0;
          } else if (target_value.has_prior_probability) {
            // A prediction without any distribution: the prior is all there is.
            score.probabilities[target_value.label()] = target_value.prior_probability;
          }
        }
        break;

      case MiningFunction::MiningFunctionType::REGRESSION:
        if (score.empty) {
          const TargetValue* default_value = find_default_value();
          if (default_value == nullptr) break;
          score.empty = false;
          score.double_score = transform(default_value->default_value);
        } else {
          score.double_score = transform(score.double_score);
        }
        // score string representation must be updated accordingly
        score.score = double_to_string(score.double_score);
        break;

      default:
        break;
    }
  }

  inline std::string operator()(const std::string& predicted) const {
    switch (mining_function.value) {
      case MiningFunction::MiningFunctionType::CLASSIFICATION:
        if (predicted.empty()) {
          const TargetValue* best = top_prior();
          return best == nullptr ? predicted : best->label();
        }
        for (const auto& target_value : target_values) {
          if (predicted == target_value.value && target_value.has_display_value) return target_value.display_value;
        }
        break;

      case MiningFunction::MiningFunctionType::REGRESSION:
        // Empty prediction must be checked before conversion: to_double("")
        // throws, which made the defaultValue branch unreachable.
        if (predicted.empty()) {
          const TargetValue* default_value = find_default_value();
          return default_value == nullptr ? predicted : double_to_string(transform(default_value->default_value));
        }
        // score string representation must be updated accordingly
        return double_to_string(transform(to_double(predicted)));

      default:
        break;
    }

    return predicted;
  }

 private:
  // min/max, then rescaleFactor, rescaleConstant and castInteger, in the order
  // the spec prescribes.
  inline double transform(double value) const {
    if (has_min && value < min)
      value = min;
    else if (has_max && value > max)
      value = max;

    if (has_rescale_factor) value *= rescale_factor;
    if (has_rescale_constant) value += rescale_constant;
    if (!cast.empty) value = cast(value);

    return value;
  }

  inline const TargetValue* find_default_value() const {
    for (const auto& target_value : target_values)
      if (target_value.has_default_value) return &target_value;
    return nullptr;
  }

  // The TargetValue with the largest prior; the first one on a tie.
  inline const TargetValue* top_prior() const {
    const TargetValue* best = nullptr;
    for (const auto& target_value : target_values)
      if (target_value.has_prior_probability &&
          (best == nullptr || target_value.prior_probability > best->prior_probability))
        best = &target_value;
    return best;
  }

  // Replaces the probabilities with the normalized priors and returns the
  // TargetValue to predict, or nullptr when no TargetValue carries a prior.
  inline const TargetValue* apply_priors(InternalScore& score) const {
    const TargetValue* best = top_prior();
    if (best == nullptr) return nullptr;

    double total = 0;
    for (const auto& target_value : target_values)
      if (target_value.has_prior_probability) total += target_value.prior_probability;

    score.probabilities.clear();
    for (const auto& target_value : target_values)
      if (target_value.has_prior_probability)
        score.probabilities[target_value.label()] =
            total > 0 ? target_value.prior_probability / total : target_value.prior_probability;

    return best;
  }
};

#endif  // CPMML_SRC_CORE_TARGET_H_
