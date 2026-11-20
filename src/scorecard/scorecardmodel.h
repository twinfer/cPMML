
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SCORECARDMODEL_H
#define CPMML_SCORECARDMODEL_H

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/internal_score.h"
#include "core/predicate.h"
#include "core/predicatebuilder.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"
#include "expression/expression.h"
#include "expression/expressionbuilder.h"

/**
 * @class ScorecardModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/Scorecard.html">PMML Scorecard</a>.
 *
 * Scoring: initialScore + Σ partialScore of first matching Attribute per Characteristic.
 * Supports both static partialScore and ComplexPartialScore (expression-computed).
 * Reason codes ranked by pointsBelow (baseline - partial) or pointsAbove (partial - baseline).
 */
class ScorecardModel : public InternalModel {
 public:
  struct Attribute {
    Predicate predicate;
    double partial_score;
    std::shared_ptr<Expression> complex_score;  // non-null → use expression instead of partial_score
    std::string reason_code;                    // may be empty → inherits from Characteristic
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

  // reasonCodeAlgorithm, default pointsBelow; anything but a case-folded
  // "pointsabove" used to mean pointsBelow.
  static constexpr EnumEntry<bool> REASON_CODE_ALGORITHM_TABLE[] = {
      {"pointsAbove", false},
      {"pointsBelow", true},
  };

  ScorecardModel() = default;

  ScorecardModel(const XmlNode& node, const DataDictionary& data_dictionary,
                 const TransformationDictionary& transformation_dictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        initial_score(node.get_double_attribute("initialScore", 0.0)),
        use_reason_codes(node.get_bool_attribute("useReasonCodes", true)),
        points_below(read_enum(node, "reasonCodeAlgorithm", REASON_CODE_ALGORITHM_TABLE, true)),
        model_baseline(node.get_double_attribute("baselineScore", 0.0)) {
    classes.push_back(mining_schema.target.name);
    parse_characteristics(node, indexer);
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    const Scored scored = compute(sample);
    if (!scored.valid) return std::make_unique<InternalScore>();

    auto score = std::make_unique<InternalScore>(scored.total);
    // Ranked, so an OutputField feature="reasonCode" rank="N" can name the Nth.
    // The codes also stay in num_outputs keyed by name, which is how they were
    // reachable before this and is what any existing consumer reads.
    for (const auto& [code, diff] : scored.reason_codes) {
      score->ranked_reason_codes.push_back(code);
      score->num_outputs[code] = diff;
    }
    return score;
  }

  inline std::string predict_raw(const Sample& sample) const override {
    const Scored scored = compute(sample);
    return scored.valid ? java::double_text(scored.total) : "";
  }

 private:
  void parse_characteristics(const XmlNode& node, const std::shared_ptr<Indexer>& indexer) {
    PredicateBuilder pb(indexer);
    const DataType double_type(DataType::DataTypeValue::DOUBLE);
    const size_t dummy_index = 0;

    for (const auto& ch_node : node.get_child("Characteristics").get_childs("Characteristic")) {
      Characteristic ch;
      // pmml-4-4.xsd: Characteristic@name and @reasonCode, and Attribute@reasonCode,
      // are optional xs:string without a default; absent is the empty text,
      // i.e. "no reason code" (a reason code may be the text "null").
      ch.name = ch_node.get_attribute("name", "");
      ch.reason_code = ch_node.get_attribute("reasonCode", "");
      ch.baseline_score = ch_node.get_double_attribute("baselineScore", model_baseline);

      for (const auto& attr_node : ch_node.get_childs("Attribute")) {
        Attribute attr;
        attr.partial_score =
            attr_node.get_double_attribute("partialScore", 0.0);
        attr.reason_code = attr_node.get_attribute("reasonCode", "");

        // ComplexPartialScore: expression-computed partial score
        if (attr_node.exists_child("ComplexPartialScore")) {
          const XmlNode cps = attr_node.get_child("ComplexPartialScore");
          const std::vector<XmlNode> expr_children = cps.get_childs();
          if (!expr_children.empty())
            attr.complex_score = ExpressionBuilder::build(expr_children[0], dummy_index, double_type, indexer);
        }

        attr.predicate = pb.build_from_parent(attr_node);
        ch.attributes.push_back(std::move(attr));
      }

      characteristics.push_back(std::move(ch));
    }
  }

  struct Scored {
    bool valid = false;  // false when some Characteristic matched no Attribute or a ComplexPartialScore is missing
    double total = 0.0;
    std::vector<std::pair<std::string, double>> reason_codes;  // highest points-missed first
  };

  /**
   * The spec's reason-code procedure, in its own order:
   *
   *   "Initialize points missed for each unique reason code to zero... For each
   *    characteristic, compute the point differential... Add the differential to
   *    the running total for that reason code... Rank the total points, P1...Pn
   *    from largest to smallest, and return the corresponding reason codes."
   *
   * Accumulation comes *before* ranking, so a code used by two characteristics
   * is one entry holding the sum. This used to emit one entry per characteristic
   * and then write them into an unordered_map keyed by code, which lost the rank
   * the sort had just established and let the second use of a code overwrite the
   * first instead of adding to it.
   */
  Scored compute(const Sample& sample) const {
    Scored result;
    result.valid = true;
    result.total = initial_score;

    std::vector<std::pair<std::string, double>> points_missed;  // insertion-ordered, so ties stay stable

    for (const auto& ch : characteristics) {
      bool matched = false;
      for (const auto& attr : ch.attributes) {
        if (attr.predicate.is_empty || attr.predicate(sample)) {
          matched = true;
          double ps = attr.partial_score;
          if (attr.complex_score) {
            // Scorecard.html: "Whenever element ComplexPartialScore is used,
            // the actual partial score is the value returned by the
            // EXPRESSION". A missing value is no number, and the page gives
            // no replacement, so the score has no value (P6.4, B49; JPMML
            // 1.7.7: UndefinedResultException). It added as_double() of the
            // missing Value, the marker double_min(), as if it were 0.
            const Value v = attr.complex_score->eval(sample);
            if (v.is_missing()) {
              result.valid = false;
              return result;
            }
            ps = v.as_double();
          }
          result.total += ps;

          if (use_reason_codes) {
            const std::string code = attr.reason_code.empty() ? ch.reason_code : attr.reason_code;
            if (!code.empty()) {
              const double diff = points_below ? (ch.baseline_score - ps) : (ps - ch.baseline_score);
              auto it = std::find_if(points_missed.begin(), points_missed.end(),
                                     [&code](const auto& entry) { return entry.first == code; });
              if (it != points_missed.end())
                it->second += diff;
              else
                points_missed.emplace_back(code, diff);
            }
          }
          break;  // first matching attribute wins
        }
      }

      // "If not even a single Attribute/PREDICATE evaluates to TRUE for a given
      // Characteristic, the scorecard as a whole returns an invalid value." It
      // used to contribute 0 and carry on, which is a different scorecard.
      if (!matched) {
        result.valid = false;
        return result;
      }
    }

    std::stable_sort(points_missed.begin(), points_missed.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    result.reason_codes = std::move(points_missed);
    return result;
  }
};

#endif
