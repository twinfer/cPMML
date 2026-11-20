
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_SCORE_H
#define CPMML_SCORE_H

#include <optional>
#include <stdexcept>
#include <string>

#include "core/value.h"
#include "utils/utils.h"

/**
 * @class InternalScore
 *
 * Abstract class internally representing the prediction produced by the scoring
 * process.
 *
 * It contains both double and literal representations of the score, as well as
 * the associated probabilities and all values produced by Output.
 */
namespace cpmml {
class InternalScore {
 public:
  // Per-rule data stored for association model output features
  struct MatchedRule {
    std::string rule_id;
    std::string antecedent;   // human-readable antecedent itemset string
    std::string consequent;   // human-readable consequent itemset string
    double support    = 0.0;
    double confidence = 0.0;
    double lift       = 1.0;
  };

  bool empty = true;
  std::string score;
  // The class value when `score` is the displayValue a Target names it by
  // (predictedValue, probability and residual read the class); empty or equal
  // to `score` otherwise. Never the prediction from before the Target.
  std::string raw_score;
  // The number the prediction is; none (std::nullopt) when it is no number
  // (a class label) or there is no prediction. Typed, so that no number, the
  // double_min() it used to be included, stands for "none" (P6.4).
  std::optional<double> double_score;
  std::unordered_map<std::string, double> probabilities;
  std::unordered_map<std::string, double> num_outputs;
  std::unordered_map<std::string, std::string> str_outputs;
  std::string entity_id;
  std::vector<std::string> ranked_entity_ids;  // KNN clustering: neighbor IDs in distance order
  // OutputField feature="confidence" outside association models, where the
  // confidence belongs to the matched rule or node rather than to a rule set.
  // Populated by RuleSetModel; none when the model states none (P6.4: it was
  // the marker double_min()).
  std::optional<double> confidence;
  // Scorecard: reason codes ranked by accumulated points missed, largest first.
  // OutputField feature="reasonCode" rank="N" names a position in this list.
  std::vector<std::string> ranked_reason_codes;

  // Association model: matched rules per algorithm
  std::unordered_map<std::string, std::vector<MatchedRule>> matched_rules_by_algorithm;

  // Time series forecast data
  std::vector<double> forecast_values;
  std::vector<std::pair<double, double>> forecast_with_variance_values;

  InternalScore() = default;

  explicit InternalScore(const double& score) : empty(false), score(java::double_text(score)), double_score(score) {}

  explicit InternalScore(const std::string& score) : empty(false), score(score), double_score(value_of(score)) {}

  InternalScore(const double& score, const std::unordered_map<std::string, double>& probabilities)
      : empty(false), score(java::double_text(score)), double_score(score), probabilities(probabilities) {}

  InternalScore(const std::string& score, const std::unordered_map<std::string, double>& probabilities)
      : empty(false), score(score), double_score(value_of(score)), probabilities(probabilities) {}

  // The number a prediction's text stands for, or none when it is not one
  // (a class label). The text is cPMML's own (java_format.h) or the
  // document's, so java::parse, which also reads "Infinity".
  static std::optional<double> value_of(const std::string& text) {
    double parsed = 0;
    if (java::parse(text, parsed)) return parsed;
    return std::nullopt;
  }

  // The number of a prediction that is one (a regression's, a segment's in a
  // numeric ensemble). Where there is none the caller is wrong: no silent
  // stand-in (P6.4; it read the marker double_min() as a number).
  double number() const {
    if (!double_score) throw std::logic_error("InternalScore: the prediction \"" + score + "\" is no number");
    return *double_score;
  }

  InternalScore(const InternalScore&) = default;

  InternalScore(InternalScore&&) = default;

  InternalScore& operator=(const InternalScore&) = default;

  InternalScore& operator=(InternalScore&&) = default;

  virtual ~InternalScore() = default;
};
}  // namespace cpmml

// The public header declares this as cpmml::InternalScore, so that including
// cPMML.h no longer introduces the name into the consumer's global namespace.
// Internal code keeps spelling it unqualified.
using cpmml::InternalScore;

#endif
