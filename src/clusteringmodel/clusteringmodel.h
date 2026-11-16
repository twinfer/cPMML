
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
  // Shared with NearestNeighborModel: a ClusteringField and a KNNInput declare
  // the same thing. See math/distance.h.
  using ClusteringField = ComparisonField;

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
  // MissingValueWeights: Qi per ClusteringField, in field order. Without the
  // element every Qi is 1.
  std::vector<double> missing_value_weights;

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
    parse_missing_value_weights(node);
    parse_clusters(node);

    // A ClusteringModel with no Cluster has nothing to assign a case to. This
    // was detected on the first score instead, so a document that could never
    // produce an answer loaded successfully and failed one record at a time.
    if (clusters.empty()) throw cpmml::ParsingException("ClusteringModel: no Cluster elements");
  }

  // --- Scoring ---

  // A missing ClusteringField is left out of the distance, not a reason to
  // give up: the ComparisonMeasure aggregations sum over the non-missing
  // fields only (SumNM) and scale by AdjustM = sum[Qi] / sum[nonmissing(Xi)*Qi],
  // Qi from MissingValueWeights or 1. JPMML 1.7.7 does exactly this, for every
  // distance measure, with and without MissingValueWeights. AdjustM is the
  // same for every cluster, so it never changes which cluster wins; it only
  // changes the distance (see nearest()).
  //
  // The model gives no prediction when AdjustM is undefined, that is when the
  // present fields' Qi sum to 0 (every ClusteringField missing, or only fields
  // with Qi = 0 present), and for the binary measures (simpleMatching,
  // jaccard, tanimoto, binarySimilarity) with any field missing, which have
  // no SumNM form in the spec. JPMML throws in all of these ("Undefined
  // result", or a NullPointerException for the binary measures). This
  // replaces a MissingValueException for any missing ClusteringField.
  inline bool has_missing_input(const Sample& sample) const {
    size_t present = 0;
    double present_weight = 0.0;
    for (size_t i = 0; i < fields.size(); i++) {
      if (sample[fields[i].field_index].value.missing) continue;
      present++;
      present_weight += missing_value_weight(i);
    }
    if (present == fields.size()) return false;  // AdjustM = 1, per the spec

    return is_binary_metric() || !(present_weight > 0.0);
  }

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    if (has_missing_input(sample)) return std::make_unique<InternalScore>();
    const std::string winner = find_cluster(sample);
    return std::make_unique<InternalScore>(winner);
  }

  inline std::string predict_raw(const Sample& sample) const override {
    if (has_missing_input(sample)) return "";
    return find_cluster(sample);
  }

  struct Nearest {
    size_t index = 0;
    double distance = std::numeric_limits<double>::max();
  };

  // The nearest cluster and the dissimilarity to it, with the full
  // prepare/transform pipeline applied. AnomalyDetection clusterMeanDist needs
  // both: the distance is the numerator of the anomaly score and the index
  // selects that cluster's entry in MeanClusterDistances. This used to return
  // the distance alone, which is why the divisor could not be applied.
  // With a ClusteringField missing the distance carries AdjustM. Where the
  // model gives no prediction (has_missing_input) there is no nearest cluster
  // and this throws MissingValueException, as it did for any missing field.
  Nearest nearest(const std::unordered_map<std::string, std::string>& sample) const {
    Value::ScopedIndexer _scope(indexer.get());
    Sample internal_sample = base_sample;
    mining_schema.prepare(internal_sample, sample);
    if (!transformation_dictionary.empty)
      for (const auto& derivedfield_name : derivedfields_dag)
        transformation_dictionary[derivedfield_name].prepare(internal_sample);

    if (has_missing_input(internal_sample))
      throw cpmml::MissingValueException("ClusteringModel: no distance defined for the missing ClusteringFields");
    return nearest_of(internal_sample);
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
      minkowski_p = parse_minkowski_p(cm.get_child("minkowski"));
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

  // MissingValueWeights holds one Qi per ClusteringField. JPMML rejects an
  // array of another length when it first needs it; this refuses it at load.
  void parse_missing_value_weights(const XmlNode& node) {
    if (!node.exists_child("MissingValueWeights")) return;
    const XmlNode arr = node.get_child("MissingValueWeights").get_child("Array");
    for (const auto& v : parse_array(arr.value())) missing_value_weights.push_back(to_double(v));
    if (missing_value_weights.size() != fields.size())
      throw cpmml::ParsingException("ClusteringModel: MissingValueWeights has " +
                                    std::to_string(missing_value_weights.size()) + " values for " +
                                    std::to_string(fields.size()) + " ClusteringFields");
  }

  double missing_value_weight(size_t i) const {
    return missing_value_weights.empty() ? 1.0 : missing_value_weights[i];
  }

  bool is_binary_metric() const {
    return metric == DistanceMetric::SIMPLE_MATCHING || metric == DistanceMetric::JACCARD ||
           metric == DistanceMetric::TANIMOTO || metric == DistanceMetric::BINARY_SIMILARITY;
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

  // The query's values, which ClusteringFields have one, and AdjustM. A
  // missing entry's value is never read.
  struct Query {
    Eigen::VectorXd values;
    std::vector<size_t> present;
    double adjust = 1.0;
  };

  // Callers have checked has_missing_input, so the present Qi sum to > 0.
  Query build_query(const Sample& sample) const {
    Query query;
    query.values = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(fields.size()));
    double total_weight = 0.0;
    double present_weight = 0.0;
    for (size_t i = 0; i < fields.size(); i++) {
      total_weight += missing_value_weight(i);
      const Value& value = sample[fields[i].field_index].value;
      if (value.missing) continue;
      query.values[static_cast<Eigen::Index>(i)] = value.value;
      query.present.push_back(i);
      present_weight += missing_value_weight(i);
    }
    if (query.present.size() < fields.size()) query.adjust = total_weight / present_weight;

    return query;
  }

  // SumNM over the present ClusteringFields, then AdjustM where each
  // aggregation puts it: inside the root for euclidean and minkowski, outside
  // for squaredEuclidean, cityBlock and chebychev. With no field missing
  // AdjustM is 1 and this is the plain distance.
  double compute_distance_to(const Query& q, const Eigen::VectorXd& c) const {
    const std::vector<size_t>& present = q.present;
    const double distance = compute_distance(
        metric, present.size(), minkowski_p, [&](size_t j) { return q.values[present[j]]; },
        [&](size_t j) { return c[present[j]]; }, [&](size_t j) { return fields[present[j]].field_weight; },
        [&](size_t j) { return fields[present[j]].compare_func; },
        [&](size_t j) { return fields[present[j]].similarity_scale; });
    if (present.size() == fields.size()) return distance;

    switch (metric) {
      case DistanceMetric::EUCLIDEAN:
        return distance * std::sqrt(q.adjust);
      case DistanceMetric::MINKOWSKI:
        return distance * std::pow(q.adjust, 1.0 / minkowski_p);
      default:
        return distance * q.adjust;
    }
  }

  // clusters is non-empty: the constructor refuses a model without one.
  Nearest nearest_of(const Sample& sample) const {
    const Query q = build_query(sample);
    // compute_distance is uniformly a dissimilarity (similarity measures are
    // flipped to 1−s), so the closest cluster is always the minimum.
    Nearest best;
    for (size_t i = 0; i < clusters.size(); i++) {
      const double d = compute_distance_to(q, clusters[i].centroid);
      if (d < best.distance) {
        best.distance = d;
        best.index = i;
      }
    }

    return best;
  }

  std::string find_cluster(const Sample& sample) const { return clusters[nearest_of(sample).index].id; }
};

#endif
