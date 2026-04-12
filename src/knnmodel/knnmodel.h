
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
      return std::make_unique<RegressionScore>(std::to_string(val), val, classes, std::vector<double>{val});
    }

    const std::string winner = classify(neighbors);
    std::vector<double> scores(classes.size(), 0.0);
    for (const auto& [dist, idx] : neighbors) {
      auto it = std::find(classes.begin(), classes.end(), training_targets[idx]);
      if (it != classes.end()) scores[std::distance(classes.begin(), it)] += 1.0;
    }
    for (auto& s : scores) s /= static_cast<double>(k);
    return std::make_unique<RegressionScore>(winner, 1.0, classes, scores);
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
    std::string target_col;
    if (!is_clustering && !mining_schema.target.empty) {
      std::string primary_target_field = mining_schema.target.name;
      if (indexer->get_type(primary_target_field).value == DataType::DataTypeValue::STRING) {
        for (const auto& mf : mining_schema.miningfields) {
          if (mf.field_usage_type == FieldUsageType::FieldUsageTypeValue::TARGET &&
              indexer->get_type(mf.name).value != DataType::DataTypeValue::STRING) {
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
    for (size_t i = 0; i < knn_inputs.size(); i++)
      q[static_cast<Eigen::Index>(i)] = sample[knn_inputs[i].field_index].value.value;
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

  std::string classify(const std::vector<std::pair<double, int>>& neighbors) const {
    std::unordered_map<std::string, double> votes;
    const bool weighted = (categorical_method == "weightedmajorityvote");
    for (const auto& [dist, idx] : neighbors) {
      const double w = weighted ? (dist > 0.0 ? 1.0 / dist : 1e10) : 1.0;
      votes[training_targets[idx]] += w;
    }
    return std::max_element(votes.begin(), votes.end(),
                            [](const auto& a, const auto& b) { return a.second < b.second; })
        ->first;
  }

  // --- Regression aggregation ---

  std::string regress(const std::vector<std::pair<double, int>>& neighbors) const {
    if (continuous_method == "median") {
      std::vector<double> vals;
      for (const auto& [dist, idx] : neighbors) vals.push_back(to_double(training_targets[idx]));
      std::sort(vals.begin(), vals.end());
      const size_t mid = vals.size() / 2;
      const double med = (vals.size() % 2 == 0) ? (vals[mid - 1] + vals[mid]) / 2.0 : vals[mid];
      return std::to_string(med);
    }

    const bool weighted = (continuous_method == "weightedaverage");
    double sum = 0.0, weight_sum = 0.0;
    for (const auto& [dist, idx] : neighbors) {
      const double w = weighted ? (dist > 0.0 ? 1.0 / dist : 1e10) : 1.0;
      sum += w * to_double(training_targets[idx]);
      weight_sum += w;
    }
    return std::to_string(weight_sum > 0.0 ? sum / weight_sum : 0.0);
  }
};

#endif
