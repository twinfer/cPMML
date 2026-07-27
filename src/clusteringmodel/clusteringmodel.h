
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
#include "math/distance.h"

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
  struct ClusteringField {
    std::string field_name;
    size_t field_index;
    CompareFunc compare_func;
    double field_weight;
    double similarity_scale = 1.0;  // gaussSim similarityScale
  };

  struct Cluster {
    std::string id;
    std::string name;
    Eigen::VectorXd centroid;
  };

  // --- Members ---

  DistanceMetric metric;
  double minkowski_p;
  bool is_similarity;  // kind="similarity"; compute_distance flips similarities
                       // to dissimilarities, so selection always minimizes
  CompareFunc default_compare_func = CompareFunc::ABS_DIFF;  // model-level ComparisonMeasure/@compareFunction

  std::vector<ClusteringField> fields;
  std::vector<Cluster> clusters;

  // --- Constructors ---

  ClusteringModel() = default;

  ClusteringModel(const XmlNode& node, const DataDictionary& data_dictionary,
                  const TransformationDictionary& transformation_dictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        metric(DistanceMetric::EUCLIDEAN),
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

  // Distance (dissimilarity) to the nearest cluster centroid, with the full
  // prepare/transform pipeline applied. Used by AnomalyDetection
  // clusterMeanDist, where the score is the distance itself, not the cluster id.
  double min_distance(const std::unordered_map<std::string, std::string>& sample) const {
    Value::ScopedIndexer _scope(indexer.get());
    Sample internal_sample = base_sample;
    mining_schema.prepare(internal_sample, sample);
    if (!transformation_dictionary.empty)
      for (const auto& derivedfield_name : derivedfields_dag)
        transformation_dictionary[derivedfield_name].prepare(internal_sample);

    const Eigen::VectorXd q = build_query(internal_sample);
    double best = std::numeric_limits<double>::max();
    for (const auto& cluster : clusters) best = std::min(best, compute_distance_to(q, cluster.centroid));
    return best;
  }

 private:
  void parse_comparison_measure(const XmlNode& node) {
    if (!node.exists_child("ComparisonMeasure")) return;
    const XmlNode cm = node.get_child("ComparisonMeasure");
    is_similarity = (to_lower(cm.get_attribute("kind")) == "similarity");
    // Model-wide default compare function that ClusteringFields inherit.
    default_compare_func = parse_compare_func(to_lower(cm.get_attribute("compareFunction")));

    if (cm.exists_child("squaredEuclidean")) { metric = DistanceMetric::SQUARED_EUCLIDEAN; return; }
    if (cm.exists_child("cityBlock")) { metric = DistanceMetric::CITYBLOCK; return; }
    if (cm.exists_child("chebychev")) { metric = DistanceMetric::CHEBYSHEV; return; }
    if (cm.exists_child("minkowski")) {
      metric = DistanceMetric::MINKOWSKI;
      const XmlNode mn = cm.get_child("minkowski");
      if (mn.exists_attribute("p")) minkowski_p = to_double(mn.get_attribute("p"));
      return;
    }
    // Similarity measures (aggregated as 1−s dissimilarities in compute_distance)
    if (cm.exists_child("simpleMatching")) { metric = DistanceMetric::SIMPLE_MATCHING; return; }
    if (cm.exists_child("jaccard")) { metric = DistanceMetric::JACCARD; return; }
    if (cm.exists_child("tanimoto")) { metric = DistanceMetric::TANIMOTO; return; }
    if (cm.exists_child("binarySimilarity")) { metric = DistanceMetric::BINARY_SIMILARITY; return; }
    // default: euclidean
  }

  void parse_clustering_fields(const XmlNode& node, const std::shared_ptr<Indexer>& indexer) {
    for (const auto& cf : node.get_childs("ClusteringField")) {
      ClusteringField f;
      f.field_name = cf.get_attribute("field");
      f.field_index = indexer->get_index(f.field_name);
      f.compare_func = parse_compare_func(to_lower(cf.get_attribute("compareFunction")), default_compare_func);
      f.field_weight = cf.exists_attribute("fieldWeight") ? to_double(cf.get_attribute("fieldWeight")) : 1.0;
      f.similarity_scale = cf.exists_attribute("similarityScale") ? to_double(cf.get_attribute("similarityScale")) : 1.0;
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
      const std::vector<std::string> vals = parse_array(arr.value());
      c.centroid.resize(static_cast<Eigen::Index>(vals.size()));
      for (size_t i = 0; i < vals.size(); i++) c.centroid[static_cast<Eigen::Index>(i)] = to_double(vals[i]);

      clusters.push_back(std::move(c));
      idx++;
    }
  }

  Eigen::VectorXd build_query(const Sample& sample) const {
    Eigen::VectorXd q(static_cast<Eigen::Index>(fields.size()));
    for (size_t i = 0; i < fields.size(); i++) {
      const Value& v = sample[fields[i].field_index].value;
      if (v.missing) throw cpmml::MissingValueException("missing value for clustering field");
      q[static_cast<Eigen::Index>(i)] = v.value;
    }
    return q;
  }

  double compute_distance_to(const Eigen::VectorXd& q, const Eigen::VectorXd& c) const {
    const size_t n = fields.size();
    return compute_distance(
        metric, n, minkowski_p, [&](size_t i) { return q[i]; }, [&](size_t i) { return c[i]; },
        [&](size_t i) { return fields[i].field_weight; }, [&](size_t i) { return fields[i].compare_func; },
        [&](size_t i) { return fields[i].similarity_scale; });
  }

  std::string find_cluster(const Sample& sample) const {
    const Eigen::VectorXd q = build_query(sample);
    // compute_distance is uniformly a dissimilarity (similarity measures are
    // flipped to 1−s), so the closest cluster is always the minimum.
    double best_val = std::numeric_limits<double>::max();
    size_t best_idx = 0;

    for (size_t i = 0; i < clusters.size(); i++) {
      const double d = compute_distance_to(q, clusters[i].centroid);
      if (d < best_val) {
        best_val = d;
        best_idx = i;
      }
    }

    return clusters[best_idx].id;
  }
};

#endif
