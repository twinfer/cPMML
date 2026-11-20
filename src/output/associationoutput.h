/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SRC_EXPRESSION_ASSOCIATIONOUTPUT_H_
#define CPMML_SRC_EXPRESSION_ASSOCIATIONOUTPUT_H_

#include <algorithm>
#include <string>
#include <vector>

#include "core/enum_parser.h"
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
 *   algorithm  — which matching algorithm's rules to read (default
 *                exclusiveRecommendation, as the spec and JPMML have it)
 *   rank       — 0 = all matching rules, 1 = best, 2 = second-best, etc.
 *   rankBasis  — sort field: confidence (default), support, lift
 *   rankOrder  — descending (default) or ascending
 *   isMultiValued — "1" when rank=0, output is pipe-delimited list
 *   ruleFeature   — for feature="ruleValue": which sub-feature to extract
 *                   (default consequent)
 */
class AssociationOutput : public OutputExpression {
 public:
  enum class RuleFeature { RULE_ID, ANTECEDENT, CONSEQUENT, RULE, SUPPORT, CONFIDENCE, LIFT };
  enum class RankBasis { CONFIDENCE, SUPPORT, LIFT };

  // OutputField@algorithm, mapped to the key InternalScore files the matched
  // rules under (AssociationModel::score).
  static constexpr EnumEntry<const char*> ALGORITHM_TABLE[] = {
      {"recommendation", "recommendation"},
      {"exclusiveRecommendation", "exclusiverecommendation"},
      {"ruleAssociation", "ruleassociation"},
  };

  static constexpr EnumEntry<RankBasis> RANK_BASIS_TABLE[] = {
      {"confidence", RankBasis::CONFIDENCE},
      {"support", RankBasis::SUPPORT},
      {"lift", RankBasis::LIFT},
      {"leverage", RankBasis::CONFIDENCE, nullptr, EnumSupport::UNSUPPORTED},
      {"affinity", RankBasis::CONFIDENCE, nullptr, EnumSupport::UNSUPPORTED},
  };

  static constexpr EnumEntry<bool> RANK_ORDER_TABLE[] = {
      {"descending", true},
      {"ascending", false},
  };

  // RULE-FEATURE.
  static constexpr EnumEntry<RuleFeature> RULE_FEATURE_TABLE[] = {
      {"antecedent", RuleFeature::ANTECEDENT},
      {"consequent", RuleFeature::CONSEQUENT},
      {"rule", RuleFeature::RULE},
      {"ruleId", RuleFeature::RULE_ID},
      {"confidence", RuleFeature::CONFIDENCE},
      {"support", RuleFeature::SUPPORT},
      {"lift", RuleFeature::LIFT},
      {"leverage", RuleFeature::RULE_ID, nullptr, EnumSupport::UNSUPPORTED},
      {"affinity", RuleFeature::RULE_ID, nullptr, EnumSupport::UNSUPPORTED},
  };

  RuleFeature rule_feature;
  std::string algorithm_key;  // lowercase algorithm name for matched_rules_by_algorithm lookup
  int rank;
  RankBasis rank_basis;
  bool descending;
  bool multi_valued;

  AssociationOutput()
      : algorithm_key("exclusiverecommendation"),
        rank(1),
        rank_basis(RankBasis::CONFIDENCE),
        descending(true),
        multi_valued(false) {}

  AssociationOutput(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
                    const DataType& output_type, RuleFeature feature)
      : OutputExpression(output_index, output_type, indexer),
        rule_feature(feature),
        // The spec default is exclusiveRecommendation; this used to default to
        // recommendation, so an OutputField without algorithm returned rules
        // whose consequent the basket already holds.
        algorithm_key(read_enum(node, "algorithm", ALGORITHM_TABLE, "exclusiverecommendation")),
        rank(static_cast<int>(node.get_long_attribute("rank", 1))),
        rank_basis(read_enum(node, "rankBasis", RANK_BASIS_TABLE, RankBasis::CONFIDENCE)),
        descending(read_enum(node, "rankOrder", RANK_ORDER_TABLE, true)),
        multi_valued(node.get_attribute("isMultiValued", "0") == "1") {}

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

  inline virtual std::optional<double> eval_double(const Sample& /*sample*/, const InternalScore& score) const override {
    auto it = score.matched_rules_by_algorithm.find(algorithm_key);
    if (it == score.matched_rules_by_algorithm.end() || it->second.empty()) return std::nullopt;

    auto sorted = it->second;
    sort_rules(sorted);

    const int idx = (rank == 0 || multi_valued) ? 0 : rank - 1;
    if (idx >= static_cast<int>(sorted.size())) return std::nullopt;

    const auto& mr = sorted[idx];
    switch (rule_feature) {
      case RuleFeature::SUPPORT:
        return mr.support;
      case RuleFeature::CONFIDENCE:
        return mr.confidence;
      case RuleFeature::LIFT:
        return mr.lift;
      default:
        return std::nullopt;
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
        return java::double_text(mr.support);
      case RuleFeature::CONFIDENCE:
        return java::double_text(mr.confidence);
      case RuleFeature::LIFT:
        return java::double_text(mr.lift);
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

 public:
  // Factory for feature="ruleValue" (reads ruleFeature attribute from node)
  static std::shared_ptr<AssociationOutput> from_rule_value(const XmlNode& node,
                                                             const std::shared_ptr<Indexer>& indexer,
                                                             const size_t& output_index,
                                                             const DataType& output_type) {
    // The spec default is consequent; this used to default to ruleId.
    const RuleFeature rf = read_enum(node, "ruleFeature", RULE_FEATURE_TABLE, RuleFeature::CONSEQUENT);
    return std::make_shared<AssociationOutput>(node, indexer, output_index, output_type, rf);
  }
};

#endif  // CPMML_SRC_EXPRESSION_ASSOCIATIONOUTPUT_H_
