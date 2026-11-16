
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
  BinarySimilarityParams binary_params;  // read only for binarySimilarity
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
    const Nearest best = nearest_of(sample);
    if (!best.defined) return std::make_unique<InternalScore>();
    return std::make_unique<InternalScore>(clusters[best.index].id);
  }

  inline std::string predict_raw(const Sample& sample) const override {
    if (has_missing_input(sample)) return "";
    const Nearest best = nearest_of(sample);
    return best.defined ? clusters[best.index].id : "";
  }

  // defined is false when the measure is undefined against some cluster (a
  // binary measure on a value other than 0/1, or with a zero denominator):
  // the model gives no prediction, where JPMML 1.7.7 throws.
  struct Nearest {
    size_t index = 0;
    double distance = std::numeric_limits<double>::max();
    bool defined = true;
  };

  // The nearest cluster and the dissimilarity to it, with the full
  // prepare/transform pipeline applied. AnomalyDetection clusterMeanDist needs
  // both: the distance is the numerator of the anomaly score and the index
  // selects that cluster's entry in MeanClusterDistances. This used to return
  // the distance alone, which is why the divisor could not be applied.
  // With a ClusteringField missing the distance carries AdjustM. Where the
  // model gives no prediction there is no nearest cluster and defined is
  // false, exactly where score_raw and predict_raw give none: AdjustM
  // undefined or a binary measure with a field missing (has_missing_input),
  // or a measure undefined against some cluster (nearest_of). This used to
  // throw MissingValueException and EvaluationException for those, so an
  // AnomalyDetectionModel clusterMeanDist record with every ClusteringField
  // missing aborted scoring instead of giving no prediction.
  Nearest nearest(const std::unordered_map<std::string, std::string>& sample) const {
    Value::ScopedIndexer _scope(indexer.get());
    Sample internal_sample = base_sample;
    mining_schema.prepare(internal_sample, sample);
    if (!transformation_dictionary.empty)
      for (const auto& derivedfield_name : derivedfields_dag)
        transformation_dictionary[derivedfield_name].prepare(internal_sample);

    if (has_missing_input(internal_sample)) return Nearest{0, std::numeric_limits<double>::quiet_NaN(), false};
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
    // Similarity measures (ranked as 1−s dissimilarities, binary_measure_similarity)
    if (cm.exists_child("simpleMatching")) { metric = DistanceMetric::SIMPLE_MATCHING; return; }
    if (cm.exists_child("jaccard")) { metric = DistanceMetric::JACCARD; return; }
    if (cm.exists_child("tanimoto")) { metric = DistanceMetric::TANIMOTO; return; }
    if (cm.exists_child("binarySimilarity")) {
      metric = DistanceMetric::BINARY_SIMILARITY;
      binary_params = parse_binary_similarity(cm.get_child("binarySimilarity"));
      return;
    }
    // default: euclidean
  }

  // Only center fields (isCenterField="true", the default) are components of
  // the cluster centers: "Only center fields correspond to the entries in the
  // center vectors in order" (ClusteringModel, 4.4.1). A field with
  // isCenterField="false" is not part of the distance, so it is not kept;
  // fields, MissingValueWeights and every Cluster Array are then indexed by
  // center field. Every ClusteringField used to be kept, so a non-center field
  // shifted the fields after it onto the wrong center entries and read past
  // the end of the Array. JPMML 1.7.7 validates the Array and
  // MissingValueWeights lengths against the center fields too, but then
  // throws IndexOutOfBoundsException when scoring such a model.
  void parse_clustering_fields(const XmlNode& node, const std::shared_ptr<Indexer>& indexer) {
    for (const auto& cf : node.get_childs("ClusteringField")) {
      if (cf.exists_attribute("isCenterField")) {
        const std::string is_center = cf.get_attribute("isCenterField");
        if (is_center == "false") continue;
        if (is_center != "true")
          throw cpmml::ParsingException("ClusteringModel: invalid isCenterField \"" + is_center +
                                        "\" on ClusteringField \"" + cf.get_attribute("field") + "\"");
      }
      ClusteringField f;
      f.field_name = cf.get_attribute("field");
      f.field_index = indexer->get_index(f.field_name);
      f.compare_func = parse_compare_func(to_lower(cf.get_attribute("compareFunction")), default_compare_func);
      f.field_weight = cf.exists_attribute("fieldWeight") ? to_double(cf.get_attribute("fieldWeight")) : 1.0;
      f.similarity_scale = cf.exists_attribute("similarityScale") ? to_double(cf.get_attribute("similarityScale")) : 1.0;
      fields.push_back(std::move(f));
    }
    // With no center field there is no distance: every cluster would tie at 0
    // and the first would always win. JPMML 1.7.7 throws "Undefined result" on
    // every record of such a model.
    if (fields.empty()) throw cpmml::ParsingException("ClusteringModel: no center ClusteringField");
  }

  // MissingValueWeights holds one Qi per center ClusteringField. JPMML rejects an
  // array of another length when it first needs it; this refuses it at load.
  void parse_missing_value_weights(const XmlNode& node) {
    if (!node.exists_child("MissingValueWeights")) return;
    const XmlNode arr = node.get_child("MissingValueWeights").get_child("Array");
    for (const auto& v : parse_array(arr.value())) missing_value_weights.push_back(to_double(v));
    if (missing_value_weights.size() != fields.size())
      throw cpmml::ParsingException("ClusteringModel: MissingValueWeights has " +
                                    std::to_string(missing_value_weights.size()) + " values for " +
                                    std::to_string(fields.size()) + " center ClusteringFields");
  }

  double missing_value_weight(size_t i) const {
    return missing_value_weights.empty() ? 1.0 : missing_value_weights[i];
  }

  bool is_binary_metric() const { return ::is_binary_metric(metric); }

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
      // One entry per center ClusteringField, as JPMML 1.7.7 requires
      // (InvalidElementException). A shorter Array used to be read past its
      // end, a longer one had its extra entries ignored.
      if (vals.size() != fields.size())
        throw cpmml::ParsingException("ClusteringModel: Cluster \"" + c.id + "\" has " + std::to_string(vals.size()) +
                                      " center values for " + std::to_string(fields.size()) +
                                      " center ClusteringFields");

      clusters.push_back(std::move(c));
      idx++;
    }
  }

  // Callers have checked has_missing_input, so the present Qi sum to > 0.
  Query build_query(const Sample& sample) const {
    return ::build_query(sample, fields, [&](size_t i) { return missing_value_weight(i); });
  }

  // SumNM over the present ClusteringFields, then AdjustM where each
  // aggregation puts it (apply_adjust).
  //
  // The binary measures are similarities s, ranked as the dissimilarity 1 - s;
  // they are only reached with every field present (has_missing_input), and
  // are NaN where s is undefined (binary_measure_similarity). binarySimilarity
  // used to be scored as simpleMatching here, its c/d parameters unread.
  double compute_distance_to(const Query& q, const Eigen::VectorXd& c) const {
    const std::vector<size_t>& present = q.present;
    if (is_binary_metric())
      return 1.0 - binary_measure_similarity(
                       metric, binary_params, present.size(), [&](size_t j) { return q.values[present[j]]; },
                       [&](size_t j) { return c[present[j]]; });
    const double distance = compute_distance(
        metric, present.size(), minkowski_p, [&](size_t j) { return q.values[present[j]]; },
        [&](size_t j) { return c[present[j]]; }, [&](size_t j) { return fields[present[j]].field_weight; },
        [&](size_t j) { return fields[present[j]].compare_func; },
        [&](size_t j) { return fields[present[j]].similarity_scale; });
    return apply_adjust(metric, distance, q.adjust, minkowski_p);
  }

  // clusters is non-empty: the constructor refuses a model without one.
  Nearest nearest_of(const Sample& sample) const {
    const Query q = build_query(sample);
    // compute_distance is uniformly a dissimilarity (similarity measures are
    // flipped to 1−s), so the closest cluster is always the minimum.
    // An undefined measure against any cluster leaves the record unscored,
    // as JPMML evaluates every cluster and throws on the first.
    Nearest best;
    for (size_t i = 0; i < clusters.size(); i++) {
      const double d = compute_distance_to(q, clusters[i].centroid);
      if (std::isnan(d)) return Nearest{0, std::numeric_limits<double>::quiet_NaN(), false};
      if (d < best.distance) {
        best.distance = d;
        best.index = i;
      }
    }

    return best;
  }
};

#endif
