/*******************************************************************************

 * Copyright 2019 AMADEUS. All rights reserved.

 * Author: Paolo Iannino

 *******************************************************************************/

#ifndef CPMML_SRC_CORE_TARGET_H_
#define CPMML_SRC_CORE_TARGET_H_

#include <charconv>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "fn/kernels.h"
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

  // castInteger="round" (P4.11, B57). Targets.html: "If a regression model
  // should predict integers, use the attribute castInteger to control how
  // decimal places should be handled: round: round to nearest integer, e.g.,
  // 2.718 becomes 3, -2.89 becomes -3". A half has two nearest integers;
  // it goes up, as the built-in round does (B54, decided 2026-11-20 after
  // P4.9, §9; decided 2026-11-20 for castInteger too, P4.11): -2.5 becomes
  // -2, -0.5 becomes 0, 2.5 becomes 3. One kernel for both,
  // fn::scalar::round_half_up, exact near a half (0.49999999999999994
  // becomes 0, -0.5000000000000001 becomes -1) and returning a double
  // from 2^52 on as it is. The result is "an integer", whose value space has
  // one zero, so -0.4, -0.5 and -0.0 become +0 (as the built-in round,
  // which stores +0, P4.10). INF and NaN come back from round_half_up as
  // they went in; `holds_integer` then refuses them (P6.11, B61).
  static double round_to_integer(const double a) {
    const double r = cpmml::fn::scalar::round_half_up(a);
    return r == 0.0 ? 0.0 : r;
  }

  // Whether `r`, the result of the cast, is an integer the engine holds
  // (P6.11, B61). Targets.html: "If a regression model should predict
  // integers, use the attribute castInteger to control how decimal places
  // should be handled", so the result is an integer, and DataDictionary.html
  // "The dataType reuses names and semantics of atomic types in XML Schema":
  // XML Schema Part 2 §3.3.13, "The value space of integer is the infinite set
  // {...,-2,-1,0,1,2,...}", which has no INF or NaN (they are values of
  // float and double only, §3.2.4, §3.2.5); BuiltinFunctions.html says the
  // same of its floor, ceil and round: "Since INF and -INF can only
  // be floating point numbers, they will cause floor, ceil, round, and modulo
  // to return an invalid value" (D38). The integer is an int64 (D4: "integers
  // are int64; overflow invalid", Q10), so a result outside [-2^63, 2^63) has
  // none either, the same test as the built-ins' fn::scalar::to_integer. The
  // comparison is false for NaN. Every finite double from 2^52 on is an
  // integer, so the cast of 1e19 is 1e19 itself, an integer XSD has and
  // int64 does not. JPMML 1.7.7 saturates round (Math.round: INF and 1e19
  // give 9.223372036854776E18, NaN gives 0) and passes ceiling and floor
  // through as doubles (Infinity, NaN, 1.0E19): D50.
  static inline bool holds_integer(const double r) { return r >= -9223372036854775808.0 && r < 9223372036854775808.0; }

  explicit CastInteger(const XmlNode& target) {
    if (!target.exists_attribute("castInteger")) {
      cast_function = _identity;
      return;
    }
    empty = false;
    kind = read_enum(target, "castInteger", TABLE);
    switch (kind) {
      case Kind::ROUND:
        cast_function = round_to_integer;
        break;
      case Kind::CEILING:
        cast_function = _ceil;
        break;
      case Kind::FLOOR:
        cast_function = _floor;
        break;
    }
  }

  // The attribute's value, for messages.
  std::string_view kind_name() const {
    for (const auto& entry : TABLE)
      if (entry.value == kind) return entry.token;
    return "";
  }

  // Not constexpr: no instantiation of this can be a constant expression, since
  // it calls std::function::operator(). That makes it ill-formed, no diagnostic
  // required -- a compiler is free to reject it and none is obliged to say why.
  inline double operator()(const double value) const { return cast_function(value); }

 private:
  Kind kind = Kind::ROUND;
};

class TargetValue {
 public:
  std::string value;
  bool has_display_value;
  std::string display_value;
  // pmml-4-4.xsd TargetValue: value, displayValue, priorProbability and
  // defaultValue are all optional without a default; absent is "not stated".
  std::optional<double> prior_probability;
  std::optional<double> default_value;
  // TODO: Partition

  TargetValue() : has_display_value(false) {}

  TargetValue(const XmlNode& node)
      : value(node.get_attribute("value", "")),
        has_display_value(node.exists_attribute("displayValue")),
        display_value(node.get_attribute("displayValue", "")),
        prior_probability(node.optional_double_attribute("priorProbability")),
        default_value(node.optional_double_attribute("defaultValue")) {}

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
  // pmml-4-4.xsd Target: min and max are optional without a default;
  // rescaleConstant defaults to 0 and rescaleFactor to 1. The has_ flags keep
  // applying the rescale only when it is written, as before (x + 0 turns -0.0
  // into 0.0).
  std::optional<double> min;
  std::optional<double> max;
  bool has_rescale_constant = false;
  double rescale_constant = 0.0;
  bool has_rescale_factor = false;
  double rescale_factor = 1.0;
  OpType optype;
  std::vector<TargetValue> target_values;
  Category category;  // compares the TargetValue classes with the model's
  std::function<void(InternalScore&)> transform_score;
  // The target the messages name: Target@field, else the MiningSchema's target.
  std::string target_name;

  Target() = default;

  explicit Target(const XmlNode& node, const MiningSchema& mining_schema,
                  const TransformationDictionary& transformation_dictionary, const MiningFunction& mining_fuction)
      : mining_function(mining_fuction),
        has_field_name(node.exists_attribute("field")),
        field_name(node.get_attribute("field", "")),
        cast(node),
        min(node.optional_double_attribute("min")),
        max(node.optional_double_attribute("max")),
        has_rescale_constant(node.exists_attribute("rescaleConstant")),
        rescale_constant(node.get_double_attribute("rescaleConstant", 0.0)),
        has_rescale_factor(node.exists_attribute("rescaleFactor")),
        rescale_factor(node.get_double_attribute("rescaleFactor", 1.0)),
        target_values(TargetValue::to_targetvalues(node.get_childs("TargetValue"))),
        category(mining_schema.target.datatype),
        target_name(has_field_name ? field_name : mining_schema.target.name) {}

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
            } else if (target_value.prior_probability) {
              // A prediction without any distribution: the prior is all there is.
              score.probabilities[target_value.value] = *target_value.prior_probability;
            }
          }
        }
        break;

      case MiningFunction::MiningFunctionType::REGRESSION:
        if (score.empty) {
          const TargetValue* default_value = find_default_value();
          if (default_value == nullptr) break;
          score.empty = false;
          score.double_score = transform(*default_value->default_value);
        } else {
          score.double_score = transform(score.number());
        }
        // score string representation must be updated accordingly
        score.score = java::double_text(*score.double_score);
        // A number has no raw value apart from its displayed one: the
        // predictedValue of a regression is the value after min/max,
        // rescaleFactor, rescaleConstant and castInteger, whatever its
        // dataType (JPMML 1.7.7: 0 * 3 + 0.5 gives predictedValue 0.5 as
        // double and as string). A string predictedValue used to report the
        // value from before the Target.
        score.raw_score.clear();
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
          return default_value == nullptr ? predicted : java::double_text(transform(*default_value->default_value));
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
    if (min && value < *min)
      value = *min;
    else if (max && value > *max)
      value = *max;

    if (has_rescale_factor) value *= rescale_factor;
    if (has_rescale_constant) value += rescale_constant;
    if (!cast.empty) {
      const double cast_value = cast(value);
      if (!CastInteger::holds_integer(cast_value)) raise_cast_invalid(value);
      value = cast_value;
    }

    return value;
  }

  // castInteger of ±INF, NaN or a value whose integer is outside int64: no
  // integer, an invalid result for the record (Q2, decided (b)), naming the
  // target, as the target cast does (P6.11, B61, D50).
  [[noreturn]] [[gnu::cold]] [[gnu::noinline]] void raise_cast_invalid(const double value) const {
    throw cpmml::InvalidValueException("target \"" + target_name + "\": castInteger=\"" +
                                       std::string(cast.kind_name()) + "\" of the value " + java::double_text(value) +
                                       " has no integer value (an integer has no INF or NaN and is an int64; " +
                                       "§2.6, D4, D50, Q2)");
  }

  inline const TargetValue* find_default_value() const {
    for (const auto& target_value : target_values)
      if (target_value.default_value) return &target_value;
    return nullptr;
  }

  // The TargetValue with the largest prior; the first one on a tie.
  inline const TargetValue* top_prior() const {
    const TargetValue* best = nullptr;
    for (const auto& target_value : target_values)
      if (target_value.prior_probability &&
          (best == nullptr || *target_value.prior_probability > *best->prior_probability))
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
      if (target_value.prior_probability) total += *target_value.prior_probability;

    score.probabilities.clear();
    for (const auto& target_value : target_values)
      if (target_value.prior_probability)
        score.probabilities[target_value.value] =
            total > 0 ? *target_value.prior_probability / total : *target_value.prior_probability;

    return best;
  }
};

#endif  // CPMML_SRC_CORE_TARGET_H_
