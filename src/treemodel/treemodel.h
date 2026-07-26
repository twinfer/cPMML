
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_TREEMODEL_H
#define CPMML_TREEMODEL_H

#include <sstream>
#include <string>
#include <vector>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/miningfield.h"
#include "core/miningfunction.h"
#include "core/miningschema.h"
#include "core/string_view.h"
#include "core/xmlnode.h"
#include "node.h"

/**
 * @class TreeModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/TreeModel.html">PMML TreeModel</a>.
 *
 * Through this class are represented Decision Tree models, both for
 * classification and regression.
 */
class TreeModel : public InternalModel {
 public:
  // What to do when a child's predicate evaluates to UNKNOWN because an input
  // it needs is missing. "none" (the default) treats UNKNOWN as false.
  enum class MissingValueStrategy { NONE, LAST_PREDICTION, NULL_PREDICTION, DEFAULT_CHILD };

  // What to do when no child evaluates to TRUE.
  enum class NoTrueChildStrategy { RETURN_NULL_PREDICTION, RETURN_LAST_PREDICTION };

  MissingValueStrategy missing_value_strategy = MissingValueStrategy::NONE;
  NoTrueChildStrategy no_true_child_strategy = NoTrueChildStrategy::RETURN_NULL_PREDICTION;
  Node root_node;

  TreeModel() = default;

  TreeModel(const XmlNode& node, const DataDictionary& data_dictionary, const PredicateBuilder& predicate_builder,
            const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, indexer),
        missing_value_strategy(to_missing_value_strategy(node.get_attribute("missingValueStrategy", "none"))),
        no_true_child_strategy(to_no_true_child_strategy(node.get_attribute("noTrueChildStrategy", "returnNullPrediction"))),
        root_node(Node(node.get_child("Node"), true, predicate_builder, target_field.datatype)) {};

  TreeModel(const XmlNode& node, const DataDictionary& data_dictionary,
            const TransformationDictionary& transformationDictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformationDictionary, indexer),
        missing_value_strategy(to_missing_value_strategy(node.get_attribute("missingValueStrategy", "none"))),
        no_true_child_strategy(to_no_true_child_strategy(node.get_attribute("noTrueChildStrategy", "returnNullPrediction"))),
        root_node(Node(node.get_child("Node"), true, PredicateBuilder(indexer), target_field.datatype)) {};

  static MissingValueStrategy to_missing_value_strategy(const std::string& value) {
    if (value == "none") return MissingValueStrategy::NONE;
    if (value == "lastPrediction") return MissingValueStrategy::LAST_PREDICTION;
    if (value == "nullPrediction") return MissingValueStrategy::NULL_PREDICTION;
    if (value == "defaultChild") return MissingValueStrategy::DEFAULT_CHILD;
    // weightedConfidence/aggregateNodes would need a different result than a
    // single node's score; scoring them as "none" would be silently wrong.
    throw cpmml::ParsingException("TreeModel missingValueStrategy=\"" + value + "\" is not supported");
  }

  static NoTrueChildStrategy to_no_true_child_strategy(const std::string& value) {
    if (value == "returnNullPrediction") return NoTrueChildStrategy::RETURN_NULL_PREDICTION;
    if (value == "returnLastPrediction") return NoTrueChildStrategy::RETURN_LAST_PREDICTION;
    throw cpmml::ParsingException("TreeModel noTrueChildStrategy=\"" + value + "\" is not supported");
  }

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    const Node* predicting_node = walk(sample, root_node);
    if (predicting_node == nullptr || !predicting_node->has_score()) return std::make_unique<TreeScore>();

    return std::make_unique<TreeScore>(predicting_node->score);
  };

  inline std::string predict_raw(const Sample& sample) const override {
    const Node* predicting_node = walk(sample, root_node);

    return predicting_node != nullptr ? predicting_node->simple_score : "";
  };

  /**
   * Single traversal shared by scoring and prediction: returns the node whose
   * score is the prediction, or nullptr when the tree yields none.
   *
   * missingValuePenalty is deliberately not honoured: it scales the
   * *confidence* of a prediction, which cPMML does not expose as an output
   * distinct from probability, and the reference implementation leaves
   * probabilities untouched (see the fix_tree_mvs_penalty fixture).
   */
  inline const Node* walk(const Sample& sample, const Node& current_node) const {
#ifdef DEBUG
    static int depth = 0;
    std::cout << std::string(depth, '\t') << current_node.predicate << (current_node.leaf ? " !!! MATCH !!!" : "")
              << std::endl;
#endif

    if (current_node.leaf) return &current_node;

    for (const auto& child : current_node.children) {
      const PredicateResult result = child.evaluate(sample);

      if (result == PredicateResult::TRUE_) {
#ifdef DEBUG
        depth++;
#endif
        const Node* selected = walk(sample, child);
#ifdef DEBUG
        depth--;
#endif
        return selected;
      }

      if (result == PredicateResult::FALSE_) continue;

      // UNKNOWN: an input this predicate needs is missing.
      switch (missing_value_strategy) {
        case MissingValueStrategy::NONE:
          continue;  // UNKNOWN behaves as false; keep scanning the siblings
        case MissingValueStrategy::LAST_PREDICTION:
          return &current_node;
        case MissingValueStrategy::NULL_PREDICTION:
          return nullptr;
        case MissingValueStrategy::DEFAULT_CHILD: {
          const Node* default_child = find_default_child(current_node);

          return default_child != nullptr ? walk(sample, *default_child) : nullptr;
        }
      }
    }

    // No child evaluated TRUE.
    if (no_true_child_strategy == NoTrueChildStrategy::RETURN_LAST_PREDICTION) return &current_node;

    return nullptr;
  }

  static inline const Node* find_default_child(const Node& current_node) {
    if (current_node.default_child.empty()) return nullptr;

    for (const auto& child : current_node.children)
      if (child.id == current_node.default_child) return &child;

    return nullptr;
  }
};

#endif
