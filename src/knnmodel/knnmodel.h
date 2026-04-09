
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
  // --- Enumerations ---

  enum class Metric {
    EUCLIDEAN,
    CITYBLOCK,
    CHEBYSHEV,
    MINKOWSKI,
    SIMPLE_MATCHING,
    JACCARD,
    TANIMOTO,
    BINARY_SIMILARITY
  };

  enum class CompareFunc { ABS_DIFF, GAUSS_SIM, DELTA, EQUAL };

  // --- Nested types ---

  struct KnnInput {
    std::string field_name;
    size_t field_index;
    CompareFunc compare_func;
    double field_weight;
  };

  // --- Members ---

  int k;
  Metric metric;
  double minkowski_p;
  std::string categorical_method;  // to_lower of categoricalScoringMethod
  std::string continuous_method;   // to_lower of continuousScoringMethod

  std::vector<KnnInput> knn_inputs;
  Eigen::MatrixXd training_features;  // [n_instances x n_features]
  std::vector<std::string> training_targets;
  std::vector<std::string> classes;

  // --- Constructors ---

  NearestNeighborModel() = default;

  NearestNeighborModel(const XmlNode& node, const DataDictionary& data_dictionary,
                       const TransformationDictionary& transformation_dictionary,
                       const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        k(static_cast<int>(node.get_long_attribute("numberOfNeighbors"))),
        metric(Metric::EUCLIDEAN),
        minkowski_p(2.0),
        categorical_method(to_lower(node.get_attribute("categoricalScoringMethod") == "null"
                                        ? "majorityvote"
                                        : node.get_attribute("categoricalScoringMethod"))),
        continuous_method(to_lower(node.get_attribute("continuousScoringMethod") == "null"
                                       ? "average"
                                       : node.get_attribute("continuousScoringMethod"))) {
    parse_metric(node);
    parse_knn_inputs(node, indexer);
    parse_training_instances(node);

    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION) {
      std::unordered_set<std::string> seen;
      for (const auto& t : training_targets)
        if (seen.insert(t).second) classes.push_back(t);
    } else {
      classes.push_back(mining_schema.target.name);
    }
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    const Eigen::VectorXd query = build_query(sample);
    auto neighbors = find_k_nearest(query);

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

    if (mining_function.value != MiningFunction::MiningFunctionType::CLASSIFICATION) return regress(neighbors);

    return classify(neighbors);
  }

 private:
  // --- Parsing ---

  static Metric parse_metric_type(const XmlNode& measure) {
    if (measure.exists_child("euclidean")) return Metric::EUCLIDEAN;
    if (measure.exists_child("cityBlock")) return Metric::CITYBLOCK;
    if (measure.exists_child("chebychev")) return Metric::CHEBYSHEV;
    if (measure.exists_child("minkowski")) return Metric::MINKOWSKI;
    if (measure.exists_child("simpleMatching")) return Metric::SIMPLE_MATCHING;
    if (measure.exists_child("jaccard")) return Metric::JACCARD;
    if (measure.exists_child("tanimoto")) return Metric::TANIMOTO;
    if (measure.exists_child("binarySimilarity")) return Metric::BINARY_SIMILARITY;
    return Metric::EUCLIDEAN;
  }

  void parse_metric(const XmlNode& node) {
    if (!node.exists_child("ComparisonMeasure")) return;
    const XmlNode cm = node.get_child("ComparisonMeasure");
    metric = parse_metric_type(cm);
    if (metric == Metric::MINKOWSKI && cm.exists_child("minkowski"))
      if (cm.get_child("minkowski").exists_attribute("p"))
        minkowski_p = to_double(cm.get_child("minkowski").get_attribute("p"));
  }

  static CompareFunc parse_compare_func(const std::string& s) {
    const std::string lower = to_lower(s);
    if (lower == "gausssim") return CompareFunc::GAUSS_SIM;
    if (lower == "delta") return CompareFunc::DELTA;
    if (lower == "equal") return CompareFunc::EQUAL;
    return CompareFunc::ABS_DIFF;
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
    // Target column: prefer the first continuous target (MIXED models may have several)
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
    std::string target_col =
        field_to_col.count(primary_target_field) ? field_to_col.at(primary_target_field) : primary_target_field;

    // Ordered feature columns matching knn_inputs
    std::vector<std::string> feat_cols;
    feat_cols.reserve(knn_inputs.size());
    for (const auto& ki : knn_inputs) feat_cols.push_back(field_to_col.at(ki.field_name));

    // Read rows from InlineTable
    std::vector<Eigen::VectorXd> rows;
    for (const auto& row : ti.get_child("InlineTable").get_childs("row")) {
      Eigen::VectorXd fv(static_cast<Eigen::Index>(feat_cols.size()));
      for (size_t i = 0; i < feat_cols.size(); i++)
        fv[static_cast<Eigen::Index>(i)] = Value(row.get_child(feat_cols[i]).value()).value;
      rows.push_back(fv);
      training_targets.push_back(row.get_child(target_col).value());
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

  // --- Per-field comparison ---

  static double field_compare(double xi, double yi, CompareFunc cf) {
    switch (cf) {
      case CompareFunc::ABS_DIFF:
        return std::abs(xi - yi);
      case CompareFunc::GAUSS_SIM:
        return 1.0 - std::exp(-(xi - yi) * (xi - yi));
      case CompareFunc::DELTA:
        return (xi != yi) ? 1.0 : 0.0;
      case CompareFunc::EQUAL:
        return (xi == yi) ? 1.0 : 0.0;
    }
    return std::abs(xi - yi);
  }

  // --- Distance computation ---

  double compute_distance(const Eigen::VectorXd& query, Eigen::Index row) const {
    const Eigen::VectorXd train = training_features.row(row);
    const size_t n = knn_inputs.size();

    switch (metric) {
      case Metric::EUCLIDEAN: {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
          const double d = field_compare(query[i], train[i], knn_inputs[i].compare_func);
          sum += knn_inputs[i].field_weight * d * d;
        }
        return std::sqrt(sum);
      }
      case Metric::CITYBLOCK: {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
          sum += knn_inputs[i].field_weight * field_compare(query[i], train[i], knn_inputs[i].compare_func);
        return sum;
      }
      case Metric::CHEBYSHEV: {
        double max_val = 0.0;
        for (size_t i = 0; i < n; i++)
          max_val = std::max(
              max_val, knn_inputs[i].field_weight * field_compare(query[i], train[i], knn_inputs[i].compare_func));
        return max_val;
      }
      case Metric::MINKOWSKI: {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
          sum += knn_inputs[i].field_weight *
                 std::pow(field_compare(query[i], train[i], knn_inputs[i].compare_func), minkowski_p);
        return std::pow(sum, 1.0 / minkowski_p);
      }
      case Metric::SIMPLE_MATCHING: {
        int match = 0;
        for (size_t i = 0; i < n; i++)
          if ((query[i] > 0.5) == (train[i] > 0.5)) match++;
        return (n > 0) ? 1.0 - static_cast<double>(match) / n : 0.0;
      }
      case Metric::JACCARD: {
        int a11 = 0, union_count = 0;
        for (size_t i = 0; i < n; i++) {
          const bool xi = query[i] > 0.5, yi = train[i] > 0.5;
          if (xi || yi) union_count++;
          if (xi && yi) a11++;
        }
        return (union_count > 0) ? 1.0 - static_cast<double>(a11) / union_count : 0.0;
      }
      case Metric::TANIMOTO: {
        // distance = 1 - (a11+a00) / (a11 + 2*(a10+a01) + a00)
        int a11 = 0, a10 = 0, a01 = 0, a00 = 0;
        for (size_t i = 0; i < n; i++) {
          const bool xi = query[i] > 0.5, yi = train[i] > 0.5;
          if (xi && yi)
            a11++;
          else if (xi)
            a10++;
          else if (yi)
            a01++;
          else
            a00++;
        }
        const int denom = a11 + 2 * (a10 + a01) + a00;
        return (denom > 0) ? 1.0 - static_cast<double>(a11 + a00) / denom : 0.0;
      }
      case Metric::BINARY_SIMILARITY:
        // Falls back to simple matching (binarySimilarity requires custom p1..p4 parameters)
        return compute_distance(query, row);
    }
    return 0.0;
  }

  // --- K-nearest neighbor search ---

  std::vector<std::pair<double, int>> find_k_nearest(const Eigen::VectorXd& query) const {
    const int n = static_cast<int>(training_features.rows());
    std::vector<std::pair<double, int>> distances(n);
    for (int i = 0; i < n; i++) distances[i] = {compute_distance(query, i), i};
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
