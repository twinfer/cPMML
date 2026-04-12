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
  bool is_transactional = false;       // true when MiningSchema has a group field
  std::string item_field_name;         // the active item field (transactional schema)
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

    // Detect transactional schema: group field present → item field is the
    // single active field that receives a collection of values.
    for (const auto& mf : mining_schema.miningfields) {
      if (mf.field_usage_type == FieldUsageType::FieldUsageTypeValue::GROUP) {
        is_transactional = true;
      } else if (mf.field_usage_type == FieldUsageType::FieldUsageTypeValue::ACTIVE) {
        item_field_name = mf.name;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Scoring
  // -------------------------------------------------------------------------

  using InternalModel::score;  // prevent name hiding of score(string map) overload

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    const auto active   = build_active_items(sample);
    return score_from_active(active);
  }

  // Variant-aware score override: extracts collection values for transactional models.
  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, FieldValue>& sample) const override {
    if (!is_transactional) {
      // Non-transactional: flatten to string map and use normal path
      return InternalModel::score(sample);
    }

    // Transactional: extract collection from the item field, build active items
    // from the collection values, then score. MiningSchema preparation uses
    // only the string fields (group field is skipped).
    Sample internal_sample = base_sample;
    std::unordered_map<std::string, std::string> flat;
    std::vector<std::string> basket;
    for (const auto& [k, v] : sample) {
      if (k == item_field_name) {
        if (std::holds_alternative<std::vector<std::string>>(v)) {
          basket = std::get<std::vector<std::string>>(v);
        } else {
          // Single string → single-element basket
          basket = {std::get<std::string>(v)};
        }
      } else if (std::holds_alternative<std::string>(v)) {
        flat[k] = std::get<std::string>(v);
      }
    }
    mining_schema.prepare(internal_sample, flat);

    if (!transformation_dictionary.empty)
      for (const auto& derivedfield_name : derivedfields_dag)
        transformation_dictionary[derivedfield_name].prepare(internal_sample);

    auto active = build_active_items_from_basket(basket);
    auto result = score_from_active(active);
    result->raw_score = result->score;
    target(*result);
    output.add_output(internal_sample, *result);
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
    int rule_index = 0;
    for (const auto& n : node.get_childs("AssociationRule")) {
      ++rule_index;
      AssociationRule rule;
      rule.id         = n.exists_attribute("id") ? n.get_attribute("id") : std::to_string(rule_index);
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

  // Build active items from an explicit basket (transactional schema).
  // Each string in the basket is matched against Item::value.
  std::unordered_set<std::string> build_active_items_from_basket(
      const std::vector<std::string>& basket) const {
    std::unordered_set<std::string> basket_set(basket.begin(), basket.end());
    std::unordered_set<std::string> active;
    for (const auto& [id, item] : items) {
      if (basket_set.contains(item.value))
        active.insert(id);
    }
    return active;
  }

  // Human-readable string for an itemset (comma-joined item values, sorted).
  std::string itemset_string(const std::string& itemset_id) const {
    const auto& item_ids = itemsets.at(itemset_id);
    std::vector<std::string> vals;
    vals.reserve(item_ids.size());
    for (const auto& iid : item_ids) vals.push_back(items.at(iid).value);
    std::ranges::sort(vals);
    std::string result;
    for (const auto& v : vals) {
      if (!result.empty()) result += ", ";
      result += v;
    }
    return result;
  }

  // Shared scoring logic: given active items, find matching rules and produce score.
  std::unique_ptr<InternalScore> score_from_active(
      const std::unordered_set<std::string>& active) const {
    const auto matching = get_matching_rules(active);

    if (matching.empty()) return std::make_unique<InternalScore>();

    const auto [min_it, max_it] =
        std::ranges::minmax_element(matching, {}, &AssociationRule::confidence);

    const std::string best = consequent_value(*max_it);

    std::unordered_map<std::string, double> probs;
    for (const auto& r : matching) {
      const std::string cons = consequent_value(r);
      auto it = probs.find(cons);
      if (it == probs.end() || r.confidence > it->second)
        probs[cons] = r.confidence;
    }

    auto result = std::make_unique<InternalScore>(best, probs);
    result->double_score = max_it->confidence;

    // Populate matched rules per algorithm for OutputField evaluation
    populate_matched_rules(*result, active);

    return result;
  }

  bool antecedent_matches(const AssociationRule& rule,
                          const std::unordered_set<std::string>& active) const {
    const auto& ant = itemsets.at(rule.antecedent);
    return std::ranges::all_of(ant, [&](const std::string& id) {
      return active.contains(id);
    });
  }

  bool consequent_check(const AssociationRule& rule,
                        const std::unordered_set<std::string>& active,
                        Algorithm algo) const {
    const auto& cons = itemsets.at(rule.consequent);
    switch (algo) {
      case Algorithm::EXCLUSIVE_RECOMMENDATION:
        return std::ranges::none_of(cons, [&](const std::string& id) {
          return active.contains(id);
        });
      case Algorithm::RULE_ASSOCIATION:
        return std::ranges::all_of(cons, [&](const std::string& id) {
          return active.contains(id);
        });
      default:  // RECOMMENDATION
        return true;
    }
  }

  std::vector<AssociationRule> get_matching_rules(
      const std::unordered_set<std::string>& active) const {
    return get_matching_rules(active, algorithm);
  }

  std::vector<AssociationRule> get_matching_rules(
      const std::unordered_set<std::string>& active, Algorithm algo) const {
    std::vector<AssociationRule> matching;
    for (const auto& rule : rules)
      if (antecedent_matches(rule, active) && consequent_check(rule, active, algo))
        matching.push_back(rule);
    return matching;
  }

  std::vector<InternalScore::MatchedRule> to_matched_rules(
      const std::vector<AssociationRule>& matching) const {
    std::vector<InternalScore::MatchedRule> result;
    result.reserve(matching.size());
    for (const auto& r : matching) {
      InternalScore::MatchedRule mr;
      mr.rule_id    = r.id;
      mr.antecedent = itemset_string(r.antecedent);
      mr.consequent = itemset_string(r.consequent);
      mr.support    = r.support;
      mr.confidence = r.confidence;
      mr.lift       = r.lift;
      result.push_back(std::move(mr));
    }
    return result;
  }

  void populate_matched_rules(InternalScore& score,
                              const std::unordered_set<std::string>& active) const {
    // Populate for each algorithm that OutputFields may reference
    score.matched_rules_by_algorithm["recommendation"] =
        to_matched_rules(get_matching_rules(active, Algorithm::RECOMMENDATION));
    score.matched_rules_by_algorithm["exclusiverecommendation"] =
        to_matched_rules(get_matching_rules(active, Algorithm::EXCLUSIVE_RECOMMENDATION));
    score.matched_rules_by_algorithm["ruleassociation"] =
        to_matched_rules(get_matching_rules(active, Algorithm::RULE_ASSOCIATION));
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
