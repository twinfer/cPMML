/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_ASSOCIATIONOUTPUT_H_
#define CPMML_SRC_EXPRESSION_ASSOCIATIONOUTPUT_H_

#include <algorithm>
#include <string>
#include <vector>

#include "outputexpression.h"

/**
 * @class AssociationOutput
 *
 * OutputExpression for association model result features.
 *
 * Handles both PMML 4.2+ direct features (feature="antecedent", "consequent",
 * "rule", "ruleId", "support", "confidence", "lift") and the PMML 4.0/4.1
 * ruleValue style (feature="ruleValue" + ruleFeature attribute).
 *
 * Attributes:
 *   algorithm  — which matching algorithm (read from OutputField, but scoring
 *                uses the matched rules already computed by the model)
 *   rank       — 0 = all matching rules, 1 = best, 2 = second-best, etc.
 *   rankBasis  — sort field: confidence (default), support, lift
 *   rankOrder  — descending (default) or ascending
 *   isMultiValued — "1" when rank=0, output is pipe-delimited list
 *   ruleFeature   — for feature="ruleValue": which sub-feature to extract
 */
class AssociationOutput : public OutputExpression {
 public:
  enum class RuleFeature { RULE_ID, ANTECEDENT, CONSEQUENT, RULE, SUPPORT, CONFIDENCE, LIFT };
  enum class RankBasis { CONFIDENCE, SUPPORT, LIFT };

  RuleFeature rule_feature;
  std::string algorithm_key;  // lowercase algorithm name for matched_rules_by_algorithm lookup
  int rank;
  RankBasis rank_basis;
  bool descending;
  bool multi_valued;

  AssociationOutput()
      : algorithm_key("recommendation"), rank(1), rank_basis(RankBasis::CONFIDENCE), descending(true), multi_valued(false) {}

  AssociationOutput(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
                    const DataType& output_type, RuleFeature feature)
      : OutputExpression(output_index, output_type, indexer),
        rule_feature(feature),
        algorithm_key(to_lower(node.exists_attribute("algorithm") ? node.get_attribute("algorithm") : "recommendation")),
        rank(node.exists_attribute("rank") ? static_cast<int>(node.get_long_attribute("rank")) : 1),
        rank_basis(parse_rank_basis(node.exists_attribute("rankBasis") ? node.get_attribute("rankBasis") : "confidence")),
        descending(to_lower(node.exists_attribute("rankOrder") ? node.get_attribute("rankOrder") : "descending") ==
                   "descending"),
        multi_valued(node.exists_attribute("isMultiValued") && node.get_attribute("isMultiValued") == "1") {}

  inline virtual std::string eval_str(const Sample& /*sample*/, const InternalScore& score) const override {
    auto it = score.matched_rules_by_algorithm.find(algorithm_key);

    if (rank == 0 || multi_valued) {
      // Multi-valued: format as [elem1, elem2, ...] 
      if (it == score.matched_rules_by_algorithm.end() || it->second.empty()) return "[]";

      auto sorted = it->second;
      sort_rules(sorted);

      std::string result = "[";
      for (size_t i = 0; i < sorted.size(); i++) {
        if (i > 0) result += ", ";
        result += extract_feature(sorted[i]);
      }
      result += "]";
      return result;
    }

    // Single-valued (rank >= 1)
    if (it == score.matched_rules_by_algorithm.end() || it->second.empty()) return "";

    auto sorted = it->second;
    sort_rules(sorted);

    const int idx = rank - 1;
    if (idx >= static_cast<int>(sorted.size())) return "";
    return extract_feature(sorted[idx]);
  }

  inline virtual double eval_double(const Sample& /*sample*/, const InternalScore& score) const override {
    auto it = score.matched_rules_by_algorithm.find(algorithm_key);
    if (it == score.matched_rules_by_algorithm.end() || it->second.empty()) return double_min();

    auto sorted = it->second;
    sort_rules(sorted);

    const int idx = (rank == 0 || multi_valued) ? 0 : rank - 1;
    if (idx >= static_cast<int>(sorted.size())) return double_min();

    const auto& mr = sorted[idx];
    switch (rule_feature) {
      case RuleFeature::SUPPORT:
        return mr.support;
      case RuleFeature::CONFIDENCE:
        return mr.confidence;
      case RuleFeature::LIFT:
        return mr.lift;
      default:
        return double_min();
    }
  }

 private:
  std::string extract_feature(const InternalScore::MatchedRule& mr) const {
    switch (rule_feature) {
      case RuleFeature::RULE_ID:
        return mr.rule_id;
      case RuleFeature::ANTECEDENT:
        return mr.antecedent;
      case RuleFeature::CONSEQUENT:
        return mr.consequent;
      case RuleFeature::RULE:
        return "{" + mr.antecedent + "}->{" + mr.consequent + "}";
      case RuleFeature::SUPPORT:
        return std::to_string(mr.support);
      case RuleFeature::CONFIDENCE:
        return std::to_string(mr.confidence);
      case RuleFeature::LIFT:
        return std::to_string(mr.lift);
    }
    return "";
  }

  void sort_rules(std::vector<InternalScore::MatchedRule>& rules) const {
    auto key = [this](const InternalScore::MatchedRule& r) -> double {
      switch (rank_basis) {
        case RankBasis::SUPPORT:
          return r.support;
        case RankBasis::LIFT:
          return r.lift;
        default:
          return r.confidence;
      }
    };

    if (descending)
      std::sort(rules.begin(), rules.end(), [&](const auto& a, const auto& b) { return key(a) > key(b); });
    else
      std::sort(rules.begin(), rules.end(), [&](const auto& a, const auto& b) { return key(a) < key(b); });
  }

  static RankBasis parse_rank_basis(const std::string& s) {
    const std::string lower = to_lower(s);
    if (lower == "support") return RankBasis::SUPPORT;
    if (lower == "lift") return RankBasis::LIFT;
    return RankBasis::CONFIDENCE;
  }

  static RuleFeature parse_rule_feature(const std::string& s) {
    const std::string lower = to_lower(s);
    if (lower == "antecedent") return RuleFeature::ANTECEDENT;
    if (lower == "consequent") return RuleFeature::CONSEQUENT;
    if (lower == "rule") return RuleFeature::RULE;
    if (lower == "ruleid") return RuleFeature::RULE_ID;
    if (lower == "support") return RuleFeature::SUPPORT;
    if (lower == "confidence") return RuleFeature::CONFIDENCE;
    if (lower == "lift") return RuleFeature::LIFT;
    return RuleFeature::RULE_ID;
  }

 public:
  // Factory for feature="ruleValue" (reads ruleFeature attribute from node)
  static std::shared_ptr<AssociationOutput> from_rule_value(const XmlNode& node,
                                                             const std::shared_ptr<Indexer>& indexer,
                                                             const size_t& output_index,
                                                             const DataType& output_type) {
    RuleFeature rf = parse_rule_feature(node.exists_attribute("ruleFeature") ? node.get_attribute("ruleFeature") : "ruleId");
    return std::make_shared<AssociationOutput>(node, indexer, output_index, output_type, rf);
  }
};

#endif  // CPMML_SRC_EXPRESSION_ASSOCIATIONOUTPUT_H_
