
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
#include "core/traininginstances.h"
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
 * - All 9 PMML ComparisonMeasure measures (euclidean, squaredEuclidean,
 *   cityBlock, chebychev, minkowski, simpleMatching, jaccard, tanimoto,
 *   binarySimilarity)
 * - Per-field compare functions: absDiff, gaussSim, delta, equal, defaulting
 *   to the ComparisonMeasure compareFunction
 * - Per-field weights (KNNInput fieldWeight)
 * - Classification: majorityVote, weightedMajorityVote
 * - Regression: average, median, weightedAverage
 * - Inline training data from InlineTable
 */
class NearestNeighborModel : public InternalModel {
 public:
  // --- Enumerations (shared via math/distance.h) ---

  // --- Nested types ---

  // Shared with ClusteringModel: a KNNInput and a ClusteringField declare the
  // same thing, bar the similarityScale only ClusteringField carries. See
  // math/distance.h.
  using KnnInput = ComparisonField;

  // --- Members ---

  int k;
  DistanceMetric metric;
  double minkowski_p;
  BinarySimilarityParams binary_params;  // read only for binarySimilarity
  CompareFunc default_compare_func = CompareFunc::ABS_DIFF;  // ComparisonMeasure/@compareFunction
  std::string categorical_method;  // to_lower of categoricalScoringMethod
  std::string continuous_method;   // to_lower of continuousScoringMethod
  std::string instance_id_variable;  // instanceIdVariable attribute (for clustering)
  double threshold = 0.001;          // threshold attribute: added to the distance for weighted methods
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
        k(static_cast<int>(node.required_long_attribute("numberOfNeighbors"))),
        metric(DistanceMetric::EUCLIDEAN),
        minkowski_p(2.0),
        categorical_method(to_lower(node.get_attribute("categoricalScoringMethod") == "null"
                                        ? "majorityvote"
                                        : node.get_attribute("categoricalScoringMethod"))),
        continuous_method(to_lower(node.get_attribute("continuousScoringMethod") == "null"
                                       ? "average"
                                       : node.get_attribute("continuousScoringMethod"))),
        instance_id_variable(node.exists_attribute("instanceIdVariable") ? node.get_attribute("instanceIdVariable") : ""),
        threshold(node.get_double_attribute("threshold", 0.001)),
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

  // A missing KNNInput is left out of the distance, not a reason to give up:
  // the ComparisonMeasure aggregations sum over the non-missing fields only
  // (SumNM) and scale by AdjustM = sum[Qi] / sum[nonmissing(Xi)*Qi]
  // (ClusteringModel spec). A NearestNeighborModel has no MissingValueWeights,
  // so every Qi is 1. JPMML 1.7.7 does exactly this. AdjustM is the same for
  // every training instance, so it changes no ranking, but it does change the
  // 1/(d + threshold) weights. With every KNNInput missing, AdjustM divides by
  // zero, and the model gives no prediction. JPMML throws "Undefined result"
  // there. The binary measures (simpleMatching, jaccard, tanimoto,
  // binarySimilarity) have no SumNM form in the spec, and JPMML fails with a
  // NullPointerException, so with any input missing they give no prediction
  // too. This replaces a MissingValueException for any missing input.
  inline bool has_missing_input(const Sample& sample) const {
    size_t present = 0;
    for (const auto& input : knn_inputs)
      if (!sample[input.field_index].value.missing) present++;
    if (present == 0) return true;
    return present < knn_inputs.size() && is_binary_metric();
  }

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    if (has_missing_input(sample)) return std::make_unique<InternalScore>();
    const Query query = build_query(sample);
    std::vector<std::pair<double, int>> neighbors;
    if (!find_k_nearest(query, neighbors)) return std::make_unique<InternalScore>();

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
    if (has_missing_input(sample)) return "";
    const Query query = build_query(sample);
    std::vector<std::pair<double, int>> neighbors;
    if (!find_k_nearest(query, neighbors)) return "";

    if (is_clustering) {
      return neighbors.empty() ? "" : training_instance_ids[neighbors[0].second];
    }
    if (mining_function.value != MiningFunction::MiningFunctionType::CLASSIFICATION) return regress(neighbors);

    return classify(neighbors);
  }

 private:
  // --- Parsing ---

  // ComparisonMeasure holds exactly one measure element (an xs:choice of the
  // nine below), besides any Extension. This used to have no squaredEuclidean
  // case and to return euclidean for anything it did not recognise -- so
  // <squaredEuclidean/> scored with the root taken, which ranks the same but
  // changes 1/(d + threshold) weights and AdjustM -- and to take euclidean
  // for a model with no ComparisonMeasure at all. A missing, unknown or second
  // measure is now a load error. JPMML 1.7.7 refuses the first two ("Required
  // element ComparisonMeasure/<Measure> is not defined") and a missing
  // ComparisonMeasure; it skips an unknown element next to a known one and
  // takes the last of two measures, both schema violations cPMML refuses.
  static DistanceMetric parse_metric_type(const XmlNode& measure) {
    static const std::pair<const char*, DistanceMetric> measures[] = {
        {"euclidean", DistanceMetric::EUCLIDEAN},
        {"squaredEuclidean", DistanceMetric::SQUARED_EUCLIDEAN},
        {"cityBlock", DistanceMetric::CITYBLOCK},
        {"chebychev", DistanceMetric::CHEBYSHEV},
        {"minkowski", DistanceMetric::MINKOWSKI},
        {"simpleMatching", DistanceMetric::SIMPLE_MATCHING},
        {"jaccard", DistanceMetric::JACCARD},
        {"tanimoto", DistanceMetric::TANIMOTO},
        {"binarySimilarity", DistanceMetric::BINARY_SIMILARITY},
    };

    bool found = false;
    DistanceMetric result = DistanceMetric::EUCLIDEAN;
    for (const auto& child : measure.get_childs()) {
      const std::string name = child.name();
      if (name.empty() || name == "Extension") continue;  // text, or an Extension
      const auto it = std::find_if(std::cbegin(measures), std::cend(measures),
                                   [&](const auto& m) { return name == m.first; });
      if (it == std::cend(measures))
        throw cpmml::ParsingException("NearestNeighborModel: unknown ComparisonMeasure measure <" + name + ">");
      if (found)
        throw cpmml::ParsingException("NearestNeighborModel: ComparisonMeasure has more than one measure (<" + name +
                                      "> after another)");
      found = true;
      result = it->second;
    }
    if (!found) throw cpmml::ParsingException("NearestNeighborModel: ComparisonMeasure has no measure element");

    return result;
  }

  void parse_metric(const XmlNode& node) {
    if (!node.exists_child("ComparisonMeasure"))
      throw cpmml::ParsingException("NearestNeighborModel: required element ComparisonMeasure is missing");
    const XmlNode cm = node.get_child("ComparisonMeasure");
    metric = parse_metric_type(cm);
    default_compare_func = parse_compare_function(cm, CompareFunc::ABS_DIFF, "ComparisonMeasure");
    if (metric == DistanceMetric::MINKOWSKI) minkowski_p = parse_minkowski_p(cm.get_child("minkowski"));
    if (metric == DistanceMetric::BINARY_SIMILARITY)
      binary_params = parse_binary_similarity(cm.get_child("binarySimilarity"));
  }

  // compareFunction on ComparisonMeasure is the default (absDiff when absent)
  // for every KNNInput that does not declare its own. KNNInput used to ignore
  // the ComparisonMeasure value and fall back to absDiff, so a model declaring
  // <ComparisonMeasure compareFunction="delta"> compared categorical codes by
  // their difference; it also took any unknown value as absDiff. An unknown
  // value is now a load error. JPMML 1.7.7 reads the ComparisonMeasure default
  // the same way, but takes an unknown value (at either level) as absDiff.
  static CompareFunc parse_compare_function(const XmlNode& node, CompareFunc fallback, const std::string& where) {
    if (!node.exists_attribute("compareFunction")) return fallback;
    const std::string raw = node.get_attribute("compareFunction");
    const std::string s = to_lower(raw);
    if (s != "absdiff" && s != "gausssim" && s != "delta" && s != "equal")
      throw cpmml::ParsingException("NearestNeighborModel: unknown compareFunction \"" + raw + "\" on " + where);
    return parse_compare_func(s);
  }

  void parse_knn_inputs(const XmlNode& node, const std::shared_ptr<Indexer>& indexer) {
    for (const auto& ki : node.get_child("KNNInputs").get_childs("KNNInput")) {
      KnnInput input;
      input.field_name = ki.get_attribute("field");
      input.field_index = indexer->get_index(input.field_name);
      input.compare_func =
          parse_compare_function(ki, default_compare_func, "KNNInput \"" + input.field_name + "\"");
      input.field_weight = ki.get_double_attribute("fieldWeight", 1.0);
      knn_inputs.push_back(std::move(input));
    }
  }

  void parse_training_instances(const XmlNode& node) {
    const XmlNode ti = node.get_child("TrainingInstances");

    // Build field → column mapping from InstanceFields
    std::unordered_map<std::string, std::string> field_to_col;
    for (const auto& f : ti.get_child("InstanceFields").get_childs("InstanceField")) {
      field_to_col[f.get_attribute("field")] = instance_field_column(f);
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
        // Direct KNNInputs: read feature columns directly from InlineTable,
        // with the field's declared dataType -- the one its query value is
        // read with. The type used to be guessed from the text, so a string
        // field holding "10" stored the number 10 for the training instance
        // while a query "10" got the field's string code, and the two never
        // compared equal. JPMML 1.7.7 parses the cell with the field's type.
        Eigen::VectorXd fv(static_cast<Eigen::Index>(feat_cols.size()));
        for (size_t i = 0; i < feat_cols.size(); i++)
          fv[static_cast<Eigen::Index>(i)] =
              Value(xml_row.get_child(feat_cols[i]).value(), indexer->get_type(knn_inputs[i].field_name)).value;
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

  // Every Qi is 1: a NearestNeighborModel has no MissingValueWeights.
  Query build_query(const Sample& sample) const {
    return ::build_query(sample, knn_inputs, [](size_t) { return 1.0; });
  }

  bool is_binary_metric() const { return ::is_binary_metric(metric); }

  // "The weights are proportional to the inverse of the distance", with the
  // model's threshold ("a very small positive number to be used for weighted
  // scoring methods to avoid numerical problems when distance ... is zero",
  // default 0.001) added to the distance first, as JPMML does: 1/(d +
  // threshold). This used to be 1/d, capped at 1e10 for d = 0, and ignored
  // threshold. The difference is large only at small distances, but a
  // missing input scales every distance by AdjustM, which 1/d cancels and
  // 1/(d + threshold) does not. A model declaring threshold="0" keeps the
  // old cap for a neighbour exactly on the query.
  double neighbor_weight(double distance, bool weighted) const {
    if (!weighted) return 1.0;

    const double shifted = distance + threshold;
    return shifted > 0.0 ? 1.0 / shifted : 1e10;
  }

  // --- Distance computation (uses shared math/distance.h) ---

  // SumNM over the present KNNInputs, then AdjustM (all Qi = 1) applied where
  // each aggregation puts it (apply_adjust).
  //
  // The binary measures are similarities s, ranked as the dissimilarity 1 - s.
  // They are only reached with every input present (has_missing_input), and
  // are NaN where s is undefined (binary_measure_similarity).
  //
  // compareFunction="equal" is the spec's c(x,y) = 1 if x = y, 0 otherwise,
  // fed into the measure as it stands, as JPMML 1.7.7 does: under a distance
  // measure a matching value adds 1. field_compare in math/distance.h flips it
  // to 0-if-equal, which made it the same as delta and inverted every equal
  // field. ClusteringModel shares field_compare and is left as it is here; for
  // KNN an equal field is passed to compute_distance as |c(x,y) - 0| under
  // absDiff, which is c(x,y) itself.
  double compute_row_distance(const Query& query, Eigen::Index row) const {
    const Eigen::VectorXd train = training_features.row(row);
    const std::vector<size_t>& present = query.present;
    if (is_binary_metric())
      return 1.0 - binary_measure_similarity(
                       metric, binary_params, present.size(), [&](size_t j) { return query.values[present[j]]; },
                       [&](size_t j) { return train[present[j]]; });
    const auto is_equal = [&](size_t j) { return knn_inputs[present[j]].compare_func == CompareFunc::EQUAL; };
    const double distance = compute_distance(
        metric, present.size(), minkowski_p,
        [&](size_t j) {
          const double x = query.values[present[j]];
          return is_equal(j) ? (x == train[present[j]] ? 1.0 : 0.0) : x;
        },
        [&](size_t j) { return is_equal(j) ? 0.0 : train[present[j]]; },
        [&](size_t j) { return knn_inputs[present[j]].field_weight; },
        [&](size_t j) { return is_equal(j) ? CompareFunc::ABS_DIFF : knn_inputs[present[j]].compare_func; });
    return apply_adjust(metric, distance, query.adjust, minkowski_p);
  }

  // --- K-nearest neighbor search ---

  // False, leaving `neighbors` empty, when the distance to some training
  // instance is undefined (NaN: a binary measure on a value other than 0/1 or
  // with a zero denominator), which is no prediction. A NaN would also break
  // partial_sort's ordering.
  bool find_k_nearest(const Query& query, std::vector<std::pair<double, int>>& neighbors) const {
    const int n = static_cast<int>(training_features.rows());
    std::vector<std::pair<double, int>> distances(n);
    for (int i = 0; i < n; i++) {
      distances[i] = {compute_row_distance(query, i), i};
      if (std::isnan(distances[i].first)) return false;
    }
    const int kk = std::min(k, n);
    std::partial_sort(distances.begin(), distances.begin() + kk, distances.end());
    neighbors.assign(distances.begin(), distances.begin() + kk);
    return true;
  }

  // --- Classification voting ---

  // One ballot, aligned to `classes`, used both to pick the winner and to
  // report the distribution -- so the two cannot disagree.
  std::vector<double> vote(const std::vector<std::pair<double, int>>& neighbors) const {
    if (neighbors.empty()) throw cpmml::MissingValueException("KNN: no neighbors available for classification");

    const bool weighted = (categorical_method == "weightedmajorityvote");
    std::vector<double> votes(classes.size(), 0.0);
    for (const auto& [dist, idx] : neighbors) {
      const double w = neighbor_weight(dist, weighted);
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
      const double w = neighbor_weight(dist, weighted);
      sum += w * to_double(training_targets[idx]);
      weight_sum += w;
    }
    return double_to_string(weight_sum > 0.0 ? sum / weight_sum : 0.0);
  }
};

#endif
