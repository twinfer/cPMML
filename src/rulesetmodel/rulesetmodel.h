
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_RULESETMODEL_H
#define CPMML_RULESETMODEL_H

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/internal_score.h"
#include "core/predicate.h"
#include "core/predicatebuilder.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"

/**
 * @class RuleSetModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/RuleSet.html">PMML RuleSetModel</a>.
 *
 * Supports all three RuleSelectionMethod criteria:
 * - firstHit: first matching rule wins
 * - weightedMax: matching rule with highest weight wins
 * - weightedSum: class with highest total weight across all matching rules wins
 */
class RuleSetModel : public InternalModel {
 public:
  enum class SelectionCriterion { FIRST_HIT, WEIGHTED_MAX, WEIGHTED_SUM };

  struct SimpleRule {
    std::string id;
    std::string score;
    double confidence = 1.0;
    double weight = 1.0;
    Predicate predicate;
  };

  // --- Members ---

  SelectionCriterion criterion = SelectionCriterion::FIRST_HIT;
  std::string default_score;
  double default_confidence = 0.0;
  std::vector<SimpleRule> rules;

  // --- Constructors ---

  RuleSetModel() = default;

  RuleSetModel(const XmlNode& node, const DataDictionary& data_dictionary,
               const TransformationDictionary& transformation_dictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer) {
    XmlNode ruleset_node = node.get_child("RuleSet");
    default_score = ruleset_node.exists_attribute("defaultScore") ? ruleset_node.get_attribute("defaultScore") : "";
    default_confidence = ruleset_node.exists_attribute("defaultConfidence")
                             ? to_double(ruleset_node.get_attribute("defaultConfidence"))
                             : 0.0;

    // Parse selection criterion (take first method if multiple present)
    if (ruleset_node.exists_child("RuleSelectionMethod")) {
      std::string crit = to_lower(ruleset_node.get_child("RuleSelectionMethod").get_attribute("criterion"));
      if (crit == "weightedmax")
        criterion = SelectionCriterion::WEIGHTED_MAX;
      else if (crit == "weightedsum")
        criterion = SelectionCriterion::WEIGHTED_SUM;
      else
        criterion = SelectionCriterion::FIRST_HIT;
    }

    parse_rules(ruleset_node, indexer);
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    return std::make_unique<InternalScore>(evaluate(sample));
  }

  inline std::string predict_raw(const Sample& sample) const override { return evaluate(sample); }

 private:
  void parse_rules(const XmlNode& ruleset_node, const std::shared_ptr<Indexer>& indexer) {
    PredicateBuilder pb(indexer);
    collect_rules(ruleset_node, pb, Predicate());
  }

  // Recursively flatten CompoundRule/SimpleRule hierarchy into a flat list of rules.
  // Each SimpleRule's predicate is AND-ed with all ancestor CompoundRule predicates.
  void collect_rules(const XmlNode& node, PredicateBuilder& pb, const Predicate& parent_pred) {
    for (const auto& child : node.get_childs()) {
      if (child.name() == "SimpleRule") {
        SimpleRule rule;
        rule.id = child.exists_attribute("id") ? child.get_attribute("id") : "";
        rule.score = child.get_attribute("score");
        rule.confidence = child.exists_attribute("confidence") ? to_double(child.get_attribute("confidence")) : 1.0;
        rule.weight = child.exists_attribute("weight") ? to_double(child.get_attribute("weight")) : 1.0;

        // Build this rule's own predicate
        Predicate rule_pred = pb.build(child.get_child_bypattern("Predicate"));
        if (rule_pred.is_empty) {
          for (const auto& c : child.get_childs()) {
            Predicate p = pb.build(c);
            if (!p.is_empty) {
              rule_pred = p;
              break;
            }
          }
        }

        // AND with ancestor CompoundRule predicate (if any)
        rule.predicate = (!parent_pred.is_empty && !rule_pred.is_empty)
                             ? Predicate({parent_pred, rule_pred}, "and")
                             : (rule_pred.is_empty ? parent_pred : rule_pred);
        rules.push_back(std::move(rule));

      } else if (child.name() == "CompoundRule") {
        // Build this compound rule's own predicate
        Predicate compound_pred = pb.build(child.get_child_bypattern("Predicate"));
        if (compound_pred.is_empty) {
          for (const auto& c : child.get_childs()) {
            Predicate p = pb.build(c);
            if (!p.is_empty) {
              compound_pred = p;
              break;
            }
          }
        }

        // Combine with parent predicate and recurse
        Predicate combined = (!parent_pred.is_empty && !compound_pred.is_empty)
                                 ? Predicate({parent_pred, compound_pred}, "and")
                                 : (compound_pred.is_empty ? parent_pred : compound_pred);
        collect_rules(child, pb, combined);
      }
    }
  }

  std::string evaluate(const Sample& sample) const {
    switch (criterion) {
      case SelectionCriterion::FIRST_HIT:
        return first_hit(sample);
      case SelectionCriterion::WEIGHTED_MAX:
        return weighted_max(sample);
      case SelectionCriterion::WEIGHTED_SUM:
        return weighted_sum(sample);
    }
    return default_score;
  }

  std::string first_hit(const Sample& sample) const {
    for (const auto& rule : rules)
      if (!rule.predicate.is_empty && rule.predicate(sample)) return rule.score;
    return default_score;
  }

  std::string weighted_max(const Sample& sample) const {
    double best = -1.0;
    std::string result = default_score;
    for (const auto& rule : rules) {
      if (!rule.predicate.is_empty && rule.predicate(sample) && rule.weight > best) {
        best = rule.weight;
        result = rule.score;
      }
    }
    return result;
  }

  std::string weighted_sum(const Sample& sample) const {
    std::map<std::string, double> sums;
    for (const auto& rule : rules)
      if (!rule.predicate.is_empty && rule.predicate(sample)) sums[rule.score] += rule.weight;
    if (sums.empty()) return default_score;
    return std::max_element(sums.begin(), sums.end(), [](const auto& a, const auto& b) { return a.second < b.second; })
        ->first;
  }
};

#endif
