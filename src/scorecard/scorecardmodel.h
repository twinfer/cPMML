
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SCORECARDMODEL_H
#define CPMML_SCORECARDMODEL_H

#include <algorithm>
#include <limits>
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
 * @class ScorecardModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/Scorecard.html">PMML Scorecard</a>.
 *
 * Scoring: initialScore + Σ partialScore of first matching Attribute per Characteristic.
 * Reason codes ranked by pointsBelow (baseline - partial) or pointsAbove (partial - baseline).
 */
class ScorecardModel : public InternalModel {
 public:
  struct Attribute {
    Predicate predicate;
    double partial_score;
    std::string reason_code;  // may be empty → inherits from Characteristic
  };

  struct Characteristic {
    std::string name;
    std::string reason_code;
    double baseline_score;
    std::vector<Attribute> attributes;
  };

  // --- Members ---

  double initial_score;
  bool use_reason_codes;
  bool points_below;  // true=pointsBelow, false=pointsAbove
  double model_baseline;

  std::vector<Characteristic> characteristics;
  std::vector<std::string> classes;

  // --- Constructors ---

  ScorecardModel() = default;

  ScorecardModel(const XmlNode &node, const DataDictionary &data_dictionary,
                  const TransformationDictionary &transformation_dictionary,
                  const std::shared_ptr<Indexer> &indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        initial_score(node.exists_attribute("initialScore")
                          ? to_double(node.get_attribute("initialScore"))
                          : 0.0),
        use_reason_codes(node.exists_attribute("useReasonCodes")
                             ? node.get_bool_attribute("useReasonCodes")
                             : true),
        points_below(to_lower(node.get_attribute("reasonCodeAlgorithm")) != "pointsabove"),
        model_baseline(node.exists_attribute("baselineScore")
                           ? to_double(node.get_attribute("baselineScore"))
                           : 0.0) {
    classes.push_back(mining_schema.target.name);
    parse_characteristics(node, indexer);
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample &sample) const override {
    auto [val, reason_codes] = compute(sample);
    auto score = std::make_unique<InternalScore>(val);
    for (const auto &[code, diff] : reason_codes)
      score->num_outputs[code] = diff;
    return score;
  }

  inline std::string predict_raw(const Sample &sample) const override {
    return std::to_string(compute(sample).first);
  }

 private:
  void parse_characteristics(const XmlNode &node, const std::shared_ptr<Indexer> &indexer) {
    PredicateBuilder pb(indexer);
    for (const auto &ch_node : node.get_child("Characteristics").get_childs("Characteristic")) {
      Characteristic ch;
      ch.name         = ch_node.get_attribute("name");
      ch.reason_code  = ch_node.get_attribute("reasonCode");
      ch.baseline_score = ch_node.exists_attribute("baselineScore")
                              ? to_double(ch_node.get_attribute("baselineScore"))
                              : model_baseline;

      for (const auto &attr_node : ch_node.get_childs("Attribute")) {
        Attribute attr;
        attr.partial_score = to_double(attr_node.get_attribute("partialScore"));
        attr.reason_code   = attr_node.exists_attribute("reasonCode")
                                 ? attr_node.get_attribute("reasonCode")
                                 : "";
        // Predicate is first child element of Attribute
        attr.predicate = pb.build(attr_node.get_child_bypattern("Predicate"));
        // Fallback: check for True/False/Simple/Compound children directly
        if (attr.predicate.is_empty) {
          for (const auto &child : attr_node.get_childs()) {
            Predicate p = pb.build(child);
            if (!p.is_empty) { attr.predicate = p; break; }
          }
        }
        ch.attributes.push_back(std::move(attr));
      }

      characteristics.push_back(std::move(ch));
    }
  }

  std::pair<double, std::vector<std::pair<std::string, double>>> compute(const Sample &sample) const {
    double total = initial_score;
    std::vector<std::pair<std::string, double>> reason_codes;

    for (const auto &ch : characteristics) {
      for (const auto &attr : ch.attributes) {
        if (attr.predicate.is_empty || attr.predicate(sample)) {
          total += attr.partial_score;

          if (use_reason_codes) {
            const std::string code = attr.reason_code.empty() ? ch.reason_code : attr.reason_code;
            if (!code.empty() && code != "null") {
              const double diff = points_below
                                      ? (ch.baseline_score - attr.partial_score)
                                      : (attr.partial_score - ch.baseline_score);
              reason_codes.emplace_back(code, diff);
            }
          }
          break;  // first matching attribute wins
        }
      }
    }

    // Sort reason codes: highest differential first
    std::sort(reason_codes.begin(), reason_codes.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    return {total, reason_codes};
  }
};

#endif
