/*******************************************************************************

 * Copyright 2019 AMADEUS. All rights reserved.

 * Author: Paolo Iannino

 *******************************************************************************/

#ifndef CPMML_SRC_CORE_TARGET_H_
#define CPMML_SRC_CORE_TARGET_H_

#include <charconv>
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
 * @class Category
 *
 * How the classes of a categorical target are compared. PMML dataTypes reuse the
 * XML Schema value spaces, so for a numeric target "4" and "4.0" are one class
 * and a ScoreDistribution, a targetCategory, a TargetValue and a probability
 * OutputField may each spell it either way. Classes of any other dataType
 * compare as written. A spelling that is not a number in full ("4abc") is
 * compared as written even for a numeric target.
 */
class Category {
 public:
  bool numeric = false;

  Category() = default;
  explicit Category(const DataType& datatype)
      : numeric(datatype.value == DataType::DataTypeValue::INTEGER ||
                datatype.value == DataType::DataTypeValue::FLOAT ||
                datatype.value == DataType::DataTypeValue::DOUBLE),
        is_float(datatype.value == DataType::DataTypeValue::FLOAT) {}

  inline bool same(const std::string& a, const std::string& b) const {
    if (a == b) return true;
    if (!numeric) return false;
    double x = 0, y = 0;
    if (!parse(a, x) || !parse(b, y)) return false;
    return is_float ? static_cast<float>(x) == static_cast<float>(y) : x == y;
  }

  // The entry of probabilities naming the class `key`, or end().
  template <typename Map>
  inline auto find(Map& probabilities, const std::string& key) const -> decltype(probabilities.begin()) {
    auto it = probabilities.find(key);
    if (it != probabilities.end() || !numeric) return it;
    for (it = probabilities.begin(); it != probabilities.end(); ++it)
      if (same(it->first, key)) return it;
    return it;
  }

 private:
  bool is_float = false;

  inline static bool parse(const std::string& value, double& parsed) { return lexical::parse_double(value, parsed); }
};

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
  enum class Kind { ROUND, CEILING, FLOOR };

  // Target@castInteger: round, ceiling or floor, absent meaning no cast.
  // Anything else used to mean no cast too.
  static constexpr EnumEntry<Kind> TABLE[] = {
      {"round", Kind::ROUND},
      {"ceiling", Kind::CEILING},
      {"floor", Kind::FLOOR},
  };

  explicit CastInteger(const XmlNode& target) {
    if (!target.exists_attribute("castInteger")) {
      cast_function = _identity;
      return;
    }
    empty = false;
    switch (read_enum(target, "castInteger", TABLE)) {
      case Kind::ROUND:
        cast_function = _round;
        break;
      case Kind::CEILING:
        cast_function = _ceil;
        break;
      case Kind::FLOOR:
        cast_function = _floor;
        break;
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
  Category category;  // compares the TargetValue classes with the model's
  std::function<void(InternalScore&)> transform_score;

  Target() = default;

  explicit Target(const XmlNode& node, const MiningSchema& mining_schema,
                  const TransformationDictionary& transformation_dictionary, const MiningFunction& mining_fuction)
      : mining_function(mining_fuction),
        has_field_name(node.exists_attribute("field")),
        field_name(node.get_attribute("field")),
        cast(node),
        has_min(node.exists_attribute("min")),
        min(node.get_double_attribute("min")),
        has_max(node.exists_attribute("max")),
        max(node.get_double_attribute("max")),
        has_rescale_constant(node.exists_attribute("rescaleConstant")),
        rescale_constant(node.get_double_attribute("rescaleConstant")),
        has_rescale_factor(node.exists_attribute("rescaleFactor")),
        rescale_factor(node.get_double_attribute("rescaleFactor")),
        target_values(TargetValue::to_targetvalues(node.get_childs("TargetValue"))),
        category(mining_schema.target.datatype) {}

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
  //
  // display chooses how the predicted class is named. A model scored on its
  // own reports the displayValue (display = true). A segment inside a
  // MiningModel, the last step of a modelChain included, hands its parent the
  // raw class values (display = false): JPMML 1.7.7 votes, averages
  // probabilities, writes modelChain predictedValue outputs and predicts at the
  // MiningModel level by value, never by displayValue; only the MiningModel's
  // own Targets name its prediction. Priors, defaultValue and the rescaling
  // apply either way. The segment's own predictedDisplayValue OutputFields
  // still report its displayValue (InternalModel::add_outputs).
  //
  // The probabilities stay keyed by the raw class whatever display is: a
  // probability OutputField's value "corresponds to one of the target
  // categories of the model" (PMML 4.4.1 Output), and JPMML 1.7.7 matches it
  // against the raw class, so value="<displayValue>" names no class.
  inline void operator()(InternalScore& score, const bool display = true) const {
    switch (mining_function.value) {
      case MiningFunction::MiningFunctionType::CLASSIFICATION:
        if (score.empty) {
          const TargetValue* best = apply_priors(score);
          if (best == nullptr) break;
          score.empty = false;
          score.raw_score = best->value;
          score.score = display ? best->label() : best->value;
          score.double_score = InternalScore::value_of(score.score);
          break;
        }

        {
          if (const TargetValue* shown = display ? display_of(score.score) : nullptr) {
            // Keep the raw class: it names the probabilities and the
            // predictedValue OutputFields.
            if (score.raw_score.empty()) score.raw_score = score.score;
            score.score = shown->display_value;
            score.double_score = InternalScore::value_of(score.score);
          }
          for (const auto& target_value : target_values) {
            // The model may spell a numeric class differently ("4" for "4.0").
            if (category.find(score.probabilities, target_value.value) != score.probabilities.end()) continue;
            if (!score.probabilities.empty()) {
              // Class absent from a distribution the model did produce: its
              // probability is 0. Priors apply only when there is no prediction;
              // filling them in here made the probabilities sum past 1.
              score.probabilities[target_value.value] = 0.0;
            } else if (target_value.has_prior_probability) {
              // A prediction without any distribution: the prior is all there is.
              score.probabilities[target_value.value] = target_value.prior_probability;
            }
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
        score.score = java::double_text(score.double_score);
        break;

      default:
        break;
    }
  }

  inline std::string operator()(const std::string& predicted, const bool display = true) const {
    switch (mining_function.value) {
      case MiningFunction::MiningFunctionType::CLASSIFICATION:
        if (predicted.empty()) {
          const TargetValue* best = top_prior();
          if (best == nullptr) return predicted;
          return display ? best->label() : best->value;
        }
        if (!display) break;
        if (const TargetValue* shown = display_of(predicted)) return shown->display_value;
        break;

      case MiningFunction::MiningFunctionType::REGRESSION:
        // Empty prediction must be checked before conversion: to_double("")
        // throws, which made the defaultValue branch unreachable.
        if (predicted.empty()) {
          const TargetValue* default_value = find_default_value();
          return default_value == nullptr ? predicted : java::double_text(transform(default_value->default_value));
        }
        // score string representation must be updated accordingly
        {
          double value = 0;
          if (!java::parse(predicted, value))
            throw cpmml::ParsingException("\"" + predicted + "\" is not a valid double");
          return java::double_text(transform(value));
        }

      default:
        break;
    }

    return predicted;
  }

  // Whether `text`, a classification prediction after this Target, is a
  // displayValue it put there (display) rather than a class value.
  inline bool is_display_value(const std::string& text, const bool display) const {
    if (!display || mining_function.value != MiningFunction::MiningFunctionType::CLASSIFICATION) return false;
    for (const auto& target_value : target_values)
      if (target_value.has_display_value && target_value.display_value == text) return true;
    return false;
  }

  // The first TargetValue whose displayValue names the class `predicted` of a
  // classification model, or nullptr: no TargetValue renames it, or the model
  // is not a classifier.
  inline const TargetValue* display_of(const std::string& predicted) const {
    if (mining_function.value != MiningFunction::MiningFunctionType::CLASSIFICATION) return nullptr;
    for (const auto& target_value : target_values)
      if (target_value.has_display_value && category.same(predicted, target_value.value)) return &target_value;
    return nullptr;
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
        score.probabilities[target_value.value] =
            total > 0 ? target_value.prior_probability / total : target_value.prior_probability;

    return best;
  }
};

#endif  // CPMML_SRC_CORE_TARGET_H_
