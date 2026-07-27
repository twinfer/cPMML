
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_KNNMODEL_H
#define CPMML_KNNMODEL_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/datadictionary.h"
#include "core/fieldusagetype.h"
#include "core/internal_model.h"
#include "core/miningfunction.h"
#include "core/transformationdictionary.h"
#include "core/value.h"
#include "core/xmlnode.h"
#include "math/distance.h"
#include "regressionmodel/regressionscore.h"

/**
 * @class NearestNeighborModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/KNN.html">PMML NearestNeighborModel</a>.
 *
 * Supports:
 * - All 8 PMML distance/similarity metrics (euclidean, cityBlock, chebyshev,
 *   minkowski, simpleMatching, jaccard, tanimoto, binarySimilarity)
 * - Per-field compare functions: absDiff, gaussSim, delta, equal
 * - Per-field weights (KNNInput fieldWeight)
 * - Classification: majorityVote, weightedMajorityVote
 * - Regression: average, median, weightedAverage
 * - Inline training data from InlineTable
 */
class NearestNeighborModel : public InternalModel {
 public:
  // --- Enumerations (shared via math/distance.h) ---

  // --- Nested types ---

  struct KnnInput {
    std::string field_name;
    size_t field_index;
    CompareFunc compare_func;
    double field_weight;
  };

  // --- Members ---

  int k;
  DistanceMetric metric;
  double minkowski_p;
  std::string categorical_method;  // to_lower of categoricalScoringMethod
  std::string continuous_method;   // to_lower of continuousScoringMethod
  std::string instance_id_variable;  // instanceIdVariable attribute (for clustering)
  bool is_clustering = false;
  bool uses_derived_inputs = false;  // KNNInputs reference DerivedFields

  std::vector<KnnInput> knn_inputs;
  Eigen::MatrixXd training_features;  // [n_instances x n_features]
  std::vector<std::string> training_targets;
  std::vector<std::string> training_instance_ids;  // per-instance ID (instanceIdVariable)
  std::vector<std::string> classes;
  std::vector<size_t> class_train_counts;  // aligned to classes; the spec's first tie-break

  // --- Constructors ---

  NearestNeighborModel() = default;

  NearestNeighborModel(const XmlNode& node, const DataDictionary& data_dictionary,
                       const TransformationDictionary& transformation_dictionary,
                       const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        k(static_cast<int>(node.get_long_attribute("numberOfNeighbors"))),
        metric(DistanceMetric::EUCLIDEAN),
        minkowski_p(2.0),
        categorical_method(to_lower(node.get_attribute("categoricalScoringMethod") == "null"
                                        ? "majorityvote"
                                        : node.get_attribute("categoricalScoringMethod"))),
        continuous_method(to_lower(node.get_attribute("continuousScoringMethod") == "null"
                                       ? "average"
                                       : node.get_attribute("continuousScoringMethod"))),
        instance_id_variable(node.exists_attribute("instanceIdVariable") ? node.get_attribute("instanceIdVariable") : ""),
        is_clustering(mining_function.value == MiningFunction::MiningFunctionType::CLUSTERING) {
    parse_metric(node);
    parse_knn_inputs(node, indexer);
    parse_training_instances(node);

    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION) {
      std::unordered_set<std::string> seen;
      for (const auto& t : training_targets)
        if (seen.insert(t).second) classes.push_back(t);

      // "the category with the largest number of cases in the training data" --
      // the first tie-break, so the counts have to survive parsing.
      class_train_counts.assign(classes.size(), 0);
      for (const auto& t : training_targets) {
        const auto it = std::find(classes.cbegin(), classes.cend(), t);
        if (it != classes.cend()) class_train_counts[static_cast<size_t>(std::distance(classes.cbegin(), it))]++;
      }
    } else if (!is_clustering) {
      classes.push_back(mining_schema.target.name);
    }
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    const Eigen::VectorXd query = build_query(sample);
    auto neighbors = find_k_nearest(query);

    if (is_clustering) {
      // Clustering: return nearest neighbor instance IDs
      auto result = std::make_unique<InternalScore>();
      if (!neighbors.empty()) {
        const std::string& best_id = training_instance_ids[neighbors[0].second];
        result->score = best_id;
        result->entity_id = best_id;
        result->empty = false;
      }
      // Populate ranked entity IDs for entityId rank=N output fields
      for (const auto& [dist, idx] : neighbors)
        result->ranked_entity_ids.push_back(training_instance_ids[idx]);
      return result;
    }

    if (mining_function.value != MiningFunction::MiningFunctionType::CLASSIFICATION) {
      const double val = to_double(regress(neighbors));
      return std::make_unique<InternalScore>(val);
    }

    // PMML defines no probability distribution for a NearestNeighborModel --
    // jpmml returns an AffinityDistribution and refuses a probability request
    // outright -- so what cPMML reports here is an extension. It is at least
    // made an honest one: the same ballot that elects the winner, normalized by
    // the weight actually cast. It used to be unweighted counts divided by k,
    // which is two separate defects: under weightedMajorityVote the reported
    // winner could be the class with the *lower* probability, and whenever
    // fewer than k training instances existed the distribution summed to less
    // than 1.
    const std::vector<double> votes = vote(neighbors);
    const double total = std::accumulate(votes.cbegin(), votes.cend(), 0.0);

    std::vector<double> probabilities(classes.size(), 0.0);
    if (total > 0.0)
      for (size_t i = 0; i < votes.size(); i++) probabilities[i] = votes[i] / total;

    const size_t best = argmax(votes);
    return std::make_unique<RegressionScore>(classes[best], probabilities[best], classes, probabilities);
  }

  inline std::string predict_raw(const Sample& sample) const override {
    const Eigen::VectorXd query = build_query(sample);
    auto neighbors = find_k_nearest(query);

    if (is_clustering) {
      return neighbors.empty() ? "" : training_instance_ids[neighbors[0].second];
    }
    if (mining_function.value != MiningFunction::MiningFunctionType::CLASSIFICATION) return regress(neighbors);

    return classify(neighbors);
  }

 private:
  // --- Parsing ---

  static DistanceMetric parse_metric_type(const XmlNode& measure) {
    if (measure.exists_child("euclidean")) return DistanceMetric::EUCLIDEAN;
    if (measure.exists_child("cityBlock")) return DistanceMetric::CITYBLOCK;
    if (measure.exists_child("chebychev")) return DistanceMetric::CHEBYSHEV;
    if (measure.exists_child("minkowski")) return DistanceMetric::MINKOWSKI;
    if (measure.exists_child("simpleMatching")) return DistanceMetric::SIMPLE_MATCHING;
    if (measure.exists_child("jaccard")) return DistanceMetric::JACCARD;
    if (measure.exists_child("tanimoto")) return DistanceMetric::TANIMOTO;
    if (measure.exists_child("binarySimilarity")) return DistanceMetric::BINARY_SIMILARITY;
    return DistanceMetric::EUCLIDEAN;
  }

  void parse_metric(const XmlNode& node) {
    if (!node.exists_child("ComparisonMeasure")) return;
    const XmlNode cm = node.get_child("ComparisonMeasure");
    metric = parse_metric_type(cm);
    if (metric == DistanceMetric::MINKOWSKI && cm.exists_child("minkowski"))
      if (cm.get_child("minkowski").exists_attribute("p"))
        minkowski_p = to_double(cm.get_child("minkowski").get_attribute("p"));
  }

  void parse_knn_inputs(const XmlNode& node, const std::shared_ptr<Indexer>& indexer) {
    for (const auto& ki : node.get_child("KNNInputs").get_childs("KNNInput")) {
      KnnInput input;
      input.field_name = ki.get_attribute("field");
      input.field_index = indexer->get_index(input.field_name);
      input.compare_func = parse_compare_func(ki.get_attribute("compareFunction"));
      input.field_weight = ki.exists_attribute("fieldWeight") ? to_double(ki.get_attribute("fieldWeight")) : 1.0;
      knn_inputs.push_back(std::move(input));
    }
  }

  void parse_training_instances(const XmlNode& node) {
    const XmlNode ti = node.get_child("TrainingInstances");

    // Build field → column mapping from InstanceFields
    std::unordered_map<std::string, std::string> field_to_col;
    for (const auto& f : ti.get_child("InstanceFields").get_childs("InstanceField")) {
      const std::string fname = f.get_attribute("field");
      const std::string col = (f.get_attribute("column") == "null") ? fname : f.get_attribute("column");
      field_to_col[fname] = col;
    }

    // Instance ID column (for clustering models with instanceIdVariable)
    std::string id_col;
    if (!instance_id_variable.empty())
      id_col = field_to_col.count(instance_id_variable) ? field_to_col.at(instance_id_variable) : instance_id_variable;

    // Target column (skip for clustering which has no target)
    // The target column is the InstanceField declaring the mining schema's
    // target. This used to switch to a *different* target field whenever the
    // declared one was a string -- so a classification model with a string
    // target and any second numeric target field silently trained on the wrong
    // column. A model with several targets now falls back only when the
    // declared one has no InstanceField at all, which is the case that
    // fallback was reaching for.
    std::string target_col;
    if (!is_clustering && !mining_schema.target.empty) {
      std::string primary_target_field = mining_schema.target.name;
      if (!field_to_col.count(primary_target_field)) {
        for (const auto& mf : mining_schema.miningfields) {
          if (mf.field_usage_type == FieldUsageType::FieldUsageTypeValue::TARGET &&
              field_to_col.count(mf.name)) {
            primary_target_field = mf.name;
            break;
          }
        }
      }
      target_col = field_to_col.count(primary_target_field) ? field_to_col.at(primary_target_field) : primary_target_field;
    }

    // Check if KNNInputs reference DerivedFields (not directly in InstanceFields)
    uses_derived_inputs = false;
    for (const auto& ki : knn_inputs) {
      if (!field_to_col.count(ki.field_name)) {
        uses_derived_inputs = true;
        break;
      }
    }

    // Build ordered feature columns for direct (non-derived) KNNInputs
    std::vector<std::string> feat_cols;
    if (!uses_derived_inputs) {
      feat_cols.reserve(knn_inputs.size());
      for (const auto& ki : knn_inputs)
        feat_cols.push_back(field_to_col.at(ki.field_name));
    }

    // Read rows from InlineTable
    std::vector<Eigen::VectorXd> rows;
    for (const auto& xml_row : ti.get_child("InlineTable").get_childs("row")) {
      if (uses_derived_inputs) {
        // Apply LocalTransformations to compute derived features from raw training data.
        std::unordered_map<std::string, std::string> raw_input;
        for (const auto& [field, col] : field_to_col) {
          if (xml_row.exists_child(col))
            raw_input[field] = xml_row.get_child(col).value();
        }
        Sample train_sample = base_sample;
        mining_schema.prepare(train_sample, raw_input);
        if (!transformation_dictionary.empty)
          for (const auto& df_name : derivedfields_dag)
            transformation_dictionary[df_name].prepare(train_sample);

        // Extract KNNInput values from the transformed sample
        Eigen::VectorXd fv(static_cast<Eigen::Index>(knn_inputs.size()));
        for (size_t i = 0; i < knn_inputs.size(); i++)
          fv[static_cast<Eigen::Index>(i)] = train_sample[knn_inputs[i].field_index].value.value;
        rows.push_back(fv);
      } else {
        // Direct KNNInputs: read feature columns directly from InlineTable
        Eigen::VectorXd fv(static_cast<Eigen::Index>(feat_cols.size()));
        for (size_t i = 0; i < feat_cols.size(); i++)
          fv[static_cast<Eigen::Index>(i)] = Value(xml_row.get_child(feat_cols[i]).value()).value;
        rows.push_back(fv);
      }

      // Read target value (classification/regression)
      if (!target_col.empty() && xml_row.exists_child(target_col))
        training_targets.push_back(xml_row.get_child(target_col).value());
      else
        training_targets.push_back("");

      // Read instance ID (clustering)
      if (!id_col.empty() && xml_row.exists_child(id_col))
        training_instance_ids.push_back(xml_row.get_child(id_col).value());
      else
        training_instance_ids.push_back(std::to_string(training_instance_ids.size() + 1));
    }

    // Build training matrix
    if (!rows.empty()) {
      training_features.resize(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(knn_inputs.size()));
      for (size_t i = 0; i < rows.size(); i++) training_features.row(static_cast<Eigen::Index>(i)) = rows[i];
    }
  }

  // --- Query vector ---

  Eigen::VectorXd build_query(const Sample& sample) const {
    Eigen::VectorXd q(static_cast<Eigen::Index>(knn_inputs.size()));
    for (size_t i = 0; i < knn_inputs.size(); i++) {
      const Value& v = sample[knn_inputs[i].field_index].value;
      if (v.missing) throw cpmml::MissingValueException("missing value for KNN input");
      q[static_cast<Eigen::Index>(i)] = v.value;
    }
    return q;
  }

  // --- Distance computation (uses shared math/distance.h) ---

  double compute_row_distance(const Eigen::VectorXd& query, Eigen::Index row) const {
    const Eigen::VectorXd train = training_features.row(row);
    const size_t n = knn_inputs.size();
    return compute_distance(
        metric, n, minkowski_p, [&](size_t i) { return query[i]; }, [&](size_t i) { return train[i]; },
        [&](size_t i) { return knn_inputs[i].field_weight; }, [&](size_t i) { return knn_inputs[i].compare_func; });
  }

  // --- K-nearest neighbor search ---

  std::vector<std::pair<double, int>> find_k_nearest(const Eigen::VectorXd& query) const {
    const int n = static_cast<int>(training_features.rows());
    std::vector<std::pair<double, int>> distances(n);
    for (int i = 0; i < n; i++) distances[i] = {compute_row_distance(query, i), i};
    const int kk = std::min(k, n);
    std::partial_sort(distances.begin(), distances.begin() + kk, distances.end());
    return {distances.begin(), distances.begin() + kk};
  }

  // --- Classification voting ---

  // One ballot, aligned to `classes`, used both to pick the winner and to
  // report the distribution -- so the two cannot disagree.
  std::vector<double> vote(const std::vector<std::pair<double, int>>& neighbors) const {
    if (neighbors.empty()) throw cpmml::MissingValueException("KNN: no neighbors available for classification");

    const bool weighted = (categorical_method == "weightedmajorityvote");
    std::vector<double> votes(classes.size(), 0.0);
    for (const auto& [dist, idx] : neighbors) {
      const double w = weighted ? (dist > 0.0 ? 1.0 / dist : 1e10) : 1.0;
      const auto it = std::find(classes.cbegin(), classes.cend(), training_targets[idx]);
      if (it != classes.cend()) votes[static_cast<size_t>(std::distance(classes.cbegin(), it))] += w;
    }

    return votes;
  }

  /**
   * The spec's tie-break, which cPMML did not implement at all:
   *
   *   "In case of a tie, the category with the largest number of cases in the
   *    training data is the winner. If multiple categories are tied on the
   *    largest number of cases in the training data, then the category with the
   *    smallest data value (in lexical order) among the tied categories is the
   *    winner."
   *
   * Ties were previously resolved by max_element over an unordered_map, i.e. by
   * hash bucket layout -- not merely non-conformant but not even stable across
   * runs of the same binary.
   */
  size_t argmax(const std::vector<double>& votes) const {
    size_t best = 0;
    for (size_t i = 1; i < votes.size(); i++) {
      if (votes[i] > votes[best]) {
        best = i;
      } else if (votes[i] == votes[best]) {
        if (class_train_counts[i] > class_train_counts[best] ||
            (class_train_counts[i] == class_train_counts[best] && classes[i] < classes[best]))
          best = i;
      }
    }

    return best;
  }

  std::string classify(const std::vector<std::pair<double, int>>& neighbors) const {
    return classes[argmax(vote(neighbors))];
  }

  // --- Regression aggregation ---

  std::string regress(const std::vector<std::pair<double, int>>& neighbors) const {
    if (neighbors.empty()) throw cpmml::MissingValueException("KNN: no neighbors available for regression");
    if (continuous_method == "median") {
      std::vector<double> vals;
      for (const auto& [dist, idx] : neighbors) vals.push_back(to_double(training_targets[idx]));
      std::sort(vals.begin(), vals.end());
      const size_t mid = vals.size() / 2;
      const double med = (vals.size() % 2 == 0) ? (vals[mid - 1] + vals[mid]) / 2.0 : vals[mid];
      return double_to_string(med);
    }

    const bool weighted = (continuous_method == "weightedaverage");
    double sum = 0.0, weight_sum = 0.0;
    for (const auto& [dist, idx] : neighbors) {
      const double w = weighted ? (dist > 0.0 ? 1.0 / dist : 1e10) : 1.0;
      sum += w * to_double(training_targets[idx]);
      weight_sum += w;
    }
    return double_to_string(weight_sum > 0.0 ? sum / weight_sum : 0.0);
  }
};

#endif
