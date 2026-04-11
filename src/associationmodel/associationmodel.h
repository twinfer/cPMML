/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_ASSOCIATIONMODEL_H
#define CPMML_ASSOCIATIONMODEL_H

#include <algorithm>
#include <ranges>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/internal_score.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"

/**
 * @class AssociationModel
 *
 * Implementation of InternalModel representing a
 * <a href="http://dmg.org/pmml/v4-4-1/AssociationRules.html">PMML AssociationModel</a>.
 *
 * Performs recommendation scoring by matching input items against preloaded
 * association rules. Supports all three rule-selection algorithms:
 * - recommendation:          antecedent ⊆ input (default)
 * - exclusiveRecommendation: antecedent ⊆ input, consequent ∩ input = ∅
 * - ruleAssociation:         antecedent ⊆ input, consequent ⊆ input
 *
 * score_raw() returns the consequent value of the highest-confidence matching
 * rule as the primary score. All matching consequents → confidence values are
 * stored in InternalScore::probabilities for multi-output access.
 */
class AssociationModel : public InternalModel {
 public:
  // -------------------------------------------------------------------------
  // Algorithm
  // -------------------------------------------------------------------------
  enum class Algorithm { RECOMMENDATION, EXCLUSIVE_RECOMMENDATION, RULE_ASSOCIATION };

  // -------------------------------------------------------------------------
  // Data structures
  // -------------------------------------------------------------------------
  struct Item {
    std::string id;
    std::string value;         // required — original item value in input data
    std::string field;         // optional (PMML 4.3+) — DataDictionary field name
    std::string category;      // optional (PMML 4.3+) — categorical match value
    std::string mapped_value;  // optional — display/output substitute for value
    double weight = 1.0;       // optional — item weight (e.g. price)
  };

  struct AssociationRule {
    std::string id;
    std::string antecedent;  // references an Itemset id
    std::string consequent;  // references an Itemset id
    double support    = 0.0;
    double confidence = 0.0;
    double lift       = 1.0;
    double leverage   = 0.0;
    double affinity   = 0.0;
  };

  // -------------------------------------------------------------------------
  // Members
  // -------------------------------------------------------------------------
  Algorithm algorithm = Algorithm::RECOMMENDATION;
  std::unordered_map<std::string, Item>                              items;    // item id → Item
  std::unordered_map<std::string, std::unordered_set<std::string>>  itemsets; // itemset id → set<item id>
  std::vector<AssociationRule>                                       rules;

  // -------------------------------------------------------------------------
  // Constructors
  // -------------------------------------------------------------------------
  AssociationModel() = default;

  AssociationModel(const XmlNode& node, const DataDictionary& data_dictionary,
                   const TransformationDictionary& transformation_dictionary,
                   const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer) {
    // Derive algorithm from the first OutputField that carries it.
    // The PMML spec puts "algorithm" on OutputField, not on the model itself,
    // so we read it here once and apply it globally for the primary score.
    if (node.exists_child("Output")) {
      for (const auto& of_node : node.get_child("Output").get_childs("OutputField")) {
        if (of_node.exists_attribute("algorithm")) {
          algorithm = algorithm_from_string(of_node.get_attribute("algorithm"));
          break;
        }
      }
    }

    parse_items(node);
    parse_itemsets(node);
    parse_rules(node);
  }

  // -------------------------------------------------------------------------
  // Scoring
  // -------------------------------------------------------------------------

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    const auto active   = build_active_items(sample);
    const auto matching = get_matching_rules(active);

    if (matching.empty()) return std::make_unique<InternalScore>();

    // std::ranges::minmax_element finds best (max confidence) and worst (min)
    // rule in a single O(3n/2) pass — cheaper than two separate scans when
    // both ends are needed (e.g. for confidence-range reporting in outputs).
    const auto [min_it, max_it] =
        std::ranges::minmax_element(matching, {}, &AssociationRule::confidence);

    // Primary score: consequent value of the highest-confidence rule
    const std::string best = consequent_value(*max_it);

    // Probabilities map: consequent value → best confidence across all
    // matching rules with that consequent (available via Prediction::distribution())
    std::unordered_map<std::string, double> probs;
    for (const auto& r : matching) {
      const std::string cons = consequent_value(r);
      auto it = probs.find(cons);
      if (it == probs.end() || r.confidence > it->second)
        probs[cons] = r.confidence;
    }

    auto result = std::make_unique<InternalScore>(best, probs);
    result->double_score = max_it->confidence;
    return result;
  }

  inline std::string predict_raw(const Sample& sample) const override {
    const auto active   = build_active_items(sample);
    const auto matching = get_matching_rules(active);

    if (matching.empty()) return "";

    return consequent_value(
        *std::ranges::max_element(matching, {}, &AssociationRule::confidence));
  }

 private:
  // -------------------------------------------------------------------------
  // Parsing
  // -------------------------------------------------------------------------

  void parse_items(const XmlNode& node) {
    for (const auto& n : node.get_childs("Item")) {
      Item item;
      item.id           = n.get_attribute("id");
      item.value        = n.get_attribute("value");
      item.field        = n.exists_attribute("field")        ? n.get_attribute("field")        : "";
      item.category     = n.exists_attribute("category")     ? n.get_attribute("category")     : "";
      item.mapped_value = n.exists_attribute("mappedValue")  ? n.get_attribute("mappedValue")  : "";
      item.weight       = n.exists_attribute("weight")       ? to_double(n.get_attribute("weight")) : 1.0;
      items[item.id]    = std::move(item);
    }
  }

  void parse_itemsets(const XmlNode& node) {
    for (const auto& n : node.get_childs("Itemset")) {
      std::string id = n.get_attribute("id");
      std::unordered_set<std::string> refs;
      for (const auto& ref : n.get_childs("ItemRef"))
        refs.insert(ref.get_attribute("itemRef"));
      itemsets[id] = std::move(refs);
    }
  }

  void parse_rules(const XmlNode& node) {
    for (const auto& n : node.get_childs("AssociationRule")) {
      AssociationRule rule;
      rule.id         = n.exists_attribute("id")       ? n.get_attribute("id")       : "";
      rule.antecedent = n.get_attribute("antecedent");
      rule.consequent = n.get_attribute("consequent");
      rule.support    = to_double(n.get_attribute("support"));
      rule.confidence = to_double(n.get_attribute("confidence"));
      rule.lift       = n.exists_attribute("lift")      ? to_double(n.get_attribute("lift"))      : 1.0;
      rule.leverage   = n.exists_attribute("leverage")  ? to_double(n.get_attribute("leverage"))  : 0.0;
      rule.affinity   = n.exists_attribute("affinity")  ? to_double(n.get_attribute("affinity"))  : 0.0;
      rules.push_back(std::move(rule));
    }
  }

  // -------------------------------------------------------------------------
  // Runtime helpers
  // -------------------------------------------------------------------------

  // Build the set of item IDs that are "present" in the given sample.
  //
  // Two conventions are supported:
  //   PMML 4.3+ (field attribute set):
  //     item is active when sample[field] == category (or value if no category)
  //   Binary encoding (no field attribute):
  //     item is active when the field named item.value is non-zero
  std::unordered_set<std::string> build_active_items(const Sample& sample) const {
    std::unordered_set<std::string> active;
    for (const auto& [id, item] : items) {
      if (!item.field.empty()) {
        if (!indexer->contains(item.field)) continue;
        const size_t   idx = indexer->get_index(item.field);
        const DataType dt  = indexer->get_type(item.field);
        if (sample[idx].value.missing) continue;
        const std::string& match = !item.category.empty() ? item.category : item.value;
        if (sample[idx].value == Value(match, dt))
          active.insert(id);
      } else {
        // Binary encoding: field name == item value, item present when field != 0
        if (!indexer->contains(item.value)) continue;
        const size_t idx = indexer->get_index(item.value);
        if (!sample[idx].value.missing && sample[idx].value.value != 0.0)
          active.insert(id);
      }
    }
    return active;
  }

  bool antecedent_matches(const AssociationRule& rule,
                          const std::unordered_set<std::string>& active) const {
    const auto& ant = itemsets.at(rule.antecedent);
    return std::ranges::all_of(ant, [&](const std::string& id) {
      return active.contains(id);
    });
  }

  bool consequent_check(const AssociationRule& rule,
                        const std::unordered_set<std::string>& active) const {
    const auto& cons = itemsets.at(rule.consequent);
    switch (algorithm) {
      case Algorithm::EXCLUSIVE_RECOMMENDATION:
        // consequent must not overlap with the input basket
        return std::ranges::none_of(cons, [&](const std::string& id) {
          return active.contains(id);
        });
      case Algorithm::RULE_ASSOCIATION:
        // entire consequent must already be in the input basket
        return std::ranges::all_of(cons, [&](const std::string& id) {
          return active.contains(id);
        });
      default:  // RECOMMENDATION
        return true;
    }
  }

  std::vector<AssociationRule> get_matching_rules(
      const std::unordered_set<std::string>& active) const {
    std::vector<AssociationRule> matching;
    for (const auto& rule : rules)
      if (antecedent_matches(rule, active) && consequent_check(rule, active))
        matching.push_back(rule);
    return matching;
  }

  // Returns a deterministic string representation of a rule's consequent.
  // For single-item consequents (the common case) this is just the item value
  // (or mappedValue when set). For multi-item consequents the values are
  // sorted and joined with commas to ensure a stable key.
  std::string consequent_value(const AssociationRule& rule) const {
    const auto& cons_ids = itemsets.at(rule.consequent);
    if (cons_ids.empty()) return "";

    std::vector<std::string> vals;
    vals.reserve(cons_ids.size());
    for (const auto& item_id : cons_ids) {
      const auto& item = items.at(item_id);
      vals.push_back(!item.mapped_value.empty() ? item.mapped_value : item.value);
    }

    std::ranges::sort(vals);

    std::string result;
    for (const auto& v : vals) {
      if (!result.empty()) result += ',';
      result += v;
    }
    return result;
  }

  static Algorithm algorithm_from_string(const std::string& s) {
    const std::string lower = to_lower(s);
    if (lower == "exclusiverecommendation") return Algorithm::EXCLUSIVE_RECOMMENDATION;
    if (lower == "ruleassociation")         return Algorithm::RULE_ASSOCIATION;
    return Algorithm::RECOMMENDATION;
  }
};

#endif  // CPMML_ASSOCIATIONMODEL_H
