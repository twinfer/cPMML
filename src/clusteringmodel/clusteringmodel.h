
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_CLUSTERINGMODEL_H
#define CPMML_CLUSTERINGMODEL_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/internal_score.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"

/**
 * @class ClusteringModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/ClusteringModel.html">PMML ClusteringModel</a>.
 *
 * Supports centerBased clustering with all PMML distance metrics.
 * Predicted value is the cluster id (or 1-based index if id absent) or name.
 * The target_field in InternalModel is a placeholder; predicted value is
 * the cluster assignment, not a supervised label.
 */
class ClusteringModel : public InternalModel {
 public:
  enum class Metric { EUCLIDEAN, SQUARED_EUCLIDEAN, CITYBLOCK, CHEBYSHEV, MINKOWSKI };
  enum class CompareFunc { ABS_DIFF, GAUSS_SIM, DELTA, EQUAL };

  struct ClusteringField {
    std::string field_name;
    size_t field_index;
    CompareFunc compare_func;
    double field_weight;
  };

  struct Cluster {
    std::string id;
    std::string name;
    Eigen::VectorXd centroid;
  };

  // --- Members ---

  Metric metric;
  double minkowski_p;
  bool is_similarity;  // true → pick max; false → pick min distance

  std::vector<ClusteringField> fields;
  std::vector<Cluster> clusters;

  // --- Constructors ---

  ClusteringModel() = default;

  ClusteringModel(const XmlNode& node, const DataDictionary& data_dictionary,
                  const TransformationDictionary& transformation_dictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        metric(Metric::EUCLIDEAN),
        minkowski_p(2.0),
        is_similarity(false) {
    parse_comparison_measure(node);
    parse_clustering_fields(node, indexer);
    parse_clusters(node);
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    const std::string winner = find_cluster(sample);
    return std::make_unique<InternalScore>(winner);
  }

  inline std::string predict_raw(const Sample& sample) const override { return find_cluster(sample); }

 private:
  void parse_comparison_measure(const XmlNode& node) {
    if (!node.exists_child("ComparisonMeasure")) return;
    const XmlNode cm = node.get_child("ComparisonMeasure");
    is_similarity = (to_lower(cm.get_attribute("kind")) == "similarity");

    if (cm.exists_child("squaredEuclidean")) {
      metric = Metric::SQUARED_EUCLIDEAN;
      return;
    }
    if (cm.exists_child("cityBlock")) {
      metric = Metric::CITYBLOCK;
      return;
    }
    if (cm.exists_child("chebychev")) {
      metric = Metric::CHEBYSHEV;
      return;
    }
    if (cm.exists_child("minkowski")) {
      metric = Metric::MINKOWSKI;
      const XmlNode mn = cm.get_child("minkowski");
      if (mn.exists_attribute("p")) minkowski_p = to_double(mn.get_attribute("p"));
      return;
    }
    // default: euclidean
  }

  static CompareFunc parse_compare_func(const std::string& s) {
    const std::string lower = to_lower(s);
    if (lower == "gausssim") return CompareFunc::GAUSS_SIM;
    if (lower == "delta") return CompareFunc::DELTA;
    if (lower == "equal") return CompareFunc::EQUAL;
    return CompareFunc::ABS_DIFF;
  }

  void parse_clustering_fields(const XmlNode& node, const std::shared_ptr<Indexer>& indexer) {
    for (const auto& cf : node.get_childs("ClusteringField")) {
      ClusteringField f;
      f.field_name = cf.get_attribute("field");
      f.field_index = indexer->get_index(f.field_name);
      f.compare_func = parse_compare_func(cf.get_attribute("compareFunction"));
      f.field_weight = cf.exists_attribute("fieldWeight") ? to_double(cf.get_attribute("fieldWeight")) : 1.0;
      fields.push_back(std::move(f));
    }
  }

  void parse_clusters(const XmlNode& node) {
    int idx = 1;
    for (const auto& cl : node.get_childs("Cluster")) {
      Cluster c;
      c.id = cl.exists_attribute("id") ? cl.get_attribute("id") : std::to_string(idx);
      c.name = cl.exists_attribute("name") ? cl.get_attribute("name") : c.id;

      // Parse Array: space-separated real values
      const XmlNode arr = cl.get_child("Array");
      const std::vector<std::string> vals = split(arr.value(), " ");
      c.centroid.resize(static_cast<Eigen::Index>(vals.size()));
      for (size_t i = 0; i < vals.size(); i++) c.centroid[static_cast<Eigen::Index>(i)] = to_double(vals[i]);

      clusters.push_back(std::move(c));
      idx++;
    }
  }

  Eigen::VectorXd build_query(const Sample& sample) const {
    Eigen::VectorXd q(static_cast<Eigen::Index>(fields.size()));
    for (size_t i = 0; i < fields.size(); i++)
      q[static_cast<Eigen::Index>(i)] = sample[fields[i].field_index].value.value;
    return q;
  }

  static double field_compare(double xi, double ci, CompareFunc cf) {
    switch (cf) {
      case CompareFunc::ABS_DIFF:
        return std::abs(xi - ci);
      case CompareFunc::GAUSS_SIM:
        return 1.0 - std::exp(-(xi - ci) * (xi - ci));
      case CompareFunc::DELTA:
        return (xi != ci) ? 1.0 : 0.0;
      case CompareFunc::EQUAL:
        return (xi == ci) ? 1.0 : 0.0;
    }
    return std::abs(xi - ci);
  }

  double compute_distance(const Eigen::VectorXd& q, const Eigen::VectorXd& c) const {
    const size_t n = fields.size();
    switch (metric) {
      case Metric::EUCLIDEAN: {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
          const double d = field_compare(q[i], c[i], fields[i].compare_func);
          sum += fields[i].field_weight * d * d;
        }
        return std::sqrt(sum);
      }
      case Metric::SQUARED_EUCLIDEAN: {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
          const double d = field_compare(q[i], c[i], fields[i].compare_func);
          sum += fields[i].field_weight * d * d;
        }
        return sum;
      }
      case Metric::CITYBLOCK: {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
          sum += fields[i].field_weight * field_compare(q[i], c[i], fields[i].compare_func);
        return sum;
      }
      case Metric::CHEBYSHEV: {
        double mx = 0.0;
        for (size_t i = 0; i < n; i++)
          mx = std::max(mx, fields[i].field_weight * field_compare(q[i], c[i], fields[i].compare_func));
        return mx;
      }
      case Metric::MINKOWSKI: {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
          sum += fields[i].field_weight * std::pow(field_compare(q[i], c[i], fields[i].compare_func), minkowski_p);
        return std::pow(sum, 1.0 / minkowski_p);
      }
    }
    return 0.0;
  }

  std::string find_cluster(const Sample& sample) const {
    const Eigen::VectorXd q = build_query(sample);
    double best_val = is_similarity ? -std::numeric_limits<double>::max() : std::numeric_limits<double>::max();
    size_t best_idx = 0;

    for (size_t i = 0; i < clusters.size(); i++) {
      const double d = compute_distance(q, clusters[i].centroid);
      if (is_similarity ? (d > best_val) : (d < best_val)) {
        best_val = d;
        best_idx = i;
      }
    }

    return clusters[best_idx].id;
  }
};

#endif
