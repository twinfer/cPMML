
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_ANOMALYDETECTIONEVALUATOR_H
#define CPMML_ANOMALYDETECTIONEVALUATOR_H

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "clusteringmodel/clusteringmodel.h"
#include "core/internal_evaluator.h"
#include "core/internal_model.h"
#include "core/internal_score.h"
#include "core/miningschema.h"
#include "core/xmlnode.h"
#include "ensemblemodel/ensemblemodel.h"
#include "svmmodel/svmmodel.h"
#include "utils/java_format.h"

/**
 * @class AnomalyDetectionEvaluator
 *
 * Implementation of InternalEvaluator for <a
 * href="http://dmg.org/pmml/v4-4/AnomalyDetectionModel.html">PMML AnomalyDetectionModel</a>.
 *
 * Supported inner model types (algorithmType):
 * - iforest (MiningModel):       score = 2^(-E[h]/c(n))
 * - ocsvm (SupportVectorMachineModel): score = raw SVM decision value
 * - clusterMeanDist (ClusteringModel): score = d(case, nearest centre) divided
 *   by that cluster's entry in MeanClusterDistances
 *
 * For iForest, c(n) = 2*(ln(n-1)+Euler) - 2*(n-1)/n  (Liu et al., 2008).
 * Higher scores indicate greater anomalousness.
 */
class AnomalyDetectionEvaluator : public InternalEvaluator {
 public:
  enum class Algorithm { IFOREST, OCSVM, CLUSTER_MEAN_DIST };

  std::unique_ptr<InternalModel> inner;
  const ClusteringModel* clustering = nullptr;  // set for clusterMeanDist (non-owning view of inner)
  double sample_data_size = 256.0;
  Algorithm algorithm = Algorithm::IFOREST;
  std::vector<double> mean_cluster_distances;
  // The AnomalyDetectionModel's own MiningSchema, applied to the record
  // before the inner model sees it. See apply_mining_schema().
  MiningSchema mining_schema;

  explicit AnomalyDetectionEvaluator(const XmlNode& pmml_root) : InternalEvaluator(pmml_root) {
    XmlNode ad = pmml_root.get_child("AnomalyDetectionModel");
    // There was a `meanThreshold` member here, read from an attribute of the
    // same name and used by nothing. AnomalyDetectionModel has no such
    // attribute — the schema allows only modelName, algorithmName,
    // functionName, algorithmType, sampleDataSize and isScorable — so it was
    // not a dropped feature but an invented one. Removed rather than wired up.
    sample_data_size = ad.get_double_attribute("sampleDataSize", 256.0);
    algorithm = read_enum(ad, "algorithmType", ALGORITHM_TYPE_TABLE);
    // The schema requires it, as it does of every model element.
    if (!ad.exists_child("MiningSchema"))
      throw cpmml::ParsingException("AnomalyDetectionModel: MiningSchema is required");
    mining_schema = MiningSchema(ad.get_child("MiningSchema"), data_dictionary);

    if (ad.exists_child("MiningModel")) {
      inner = std::make_unique<EnsembleModel>(ad.get_child("MiningModel"), data_dictionary, transformation_dictionary,
                                              indexer);
    } else if (ad.exists_child("SupportVectorMachineModel")) {
      inner = std::make_unique<SupportVectorMachineModel>(ad.get_child("SupportVectorMachineModel"), data_dictionary,
                                                          transformation_dictionary, indexer);
    } else if (ad.exists_child("ClusteringModel")) {
      auto cm = std::make_unique<ClusteringModel>(ad.get_child("ClusteringModel"), data_dictionary,
                                                  transformation_dictionary, indexer);
      clustering = cm.get();
      inner = std::move(cm);
    } else {
      throw cpmml::ParsingException("AnomalyDetectionModel: unsupported or missing inner model");
    }

    parse_mean_cluster_distances(ad);
  }

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    const auto flat = apply_mining_schema(flatten_input(arguments));

    // The anomaly score is a function of the inner model's result, so where
    // the inner model gives no prediction there is no score either: the
    // AnomalyDetectionModel page says nothing of missing values, and the
    // inner models' own rules decide (a ClusteringModel with AdjustM
    // undefined, a MiningModel whose segments give no prediction, an SVM with
    // a FieldRef missing). Each of these used to throw -- MissingValueException
    // from nearest(), or a ParsingException from to_double("") -- so one such
    // record aborted the batch. JPMML 1.7.7 cannot load AnomalyDetectionModel;
    // on the inner models alone it gives no prediction for the forest and
    // throws for the other two.
    switch (algorithm) {
      case Algorithm::CLUSTER_MEAN_DIST: {
        // clusterMeanDist scores with the distance to the nearest cluster —
        // inner->predict would return the cluster *id*, which is not a score.
        const ClusteringModel::Nearest n = clustering->nearest(flat);
        if (!n.defined) return std::make_unique<InternalScore>();
        return std::make_unique<InternalScore>(cluster_mean_dist_score(n));
      }
      case Algorithm::IFOREST: {
        const std::string avg_path = inner->predict(flat);
        if (avg_path.empty()) return std::make_unique<InternalScore>();
        return std::make_unique<InternalScore>(iforest_score(from_java_text(avg_path)));
      }
      case Algorithm::OCSVM: {
        const std::string decision = inner->predict(flat);
        if (decision.empty()) return std::make_unique<InternalScore>();
        return std::make_unique<InternalScore>(from_java_text(decision));
      }
    }

    throw cpmml::EvaluationException("AnomalyDetectionModel: unreachable algorithmType");
  }

  // A record the outer MiningSchema refuses (an invalid value under
  // returnInvalid) does not validate.
  inline bool validate(const Input& arguments) const override {
    std::unordered_map<std::string, std::string> flat;
    try {
      flat = apply_mining_schema(flatten_input(arguments));
    } catch (const cpmml::InvalidValueException&) {
      return false;
    }
    return inner->validate(flat);
  }

  inline std::string get_target_name() const override { return inner->target_field.name; }
  inline std::string output_name() const override { return inner->output_name(); }
  inline std::string mining_function_name() const override { return "CLASSIFICATION"; }

 private:
  /**
   * The outer MiningSchema, applied before the inner model's own.
   *
   * It used to be read by nobody: the inner model was handed the raw record,
   * so a missingValueReplacement, invalidValueTreatment or outlier treatment
   * on an AnomalyDetectionModel MiningField was never used. JPMML 1.7.7
   * cannot load AnomalyDetectionModel, but for a model nested in another (a
   * MiningModel Segment) it applies the outer MiningField first and the inner
   * one to the result: an outer replacement wins over an inner one, a field
   * the outer schema leaves missing takes the inner replacement, an outer asIs
   * leaves an invalid value for the inner MiningField to judge again, and a
   * value clamped by asExtremeValues is what the inner model sees.
   *
   * Per field, with MiningField's own readers and treatments (those of
   * MiningSchema::prepare): absent, empty or a DataDictionary missing marker
   * is missing; text that is not a value of the dataType, or a NaN, goes to
   * invalidValueTreatment; so does a value outside the DataField's Values or
   * Intervals, whether or not the attribute is written (its default is
   * returnInvalid, InvalidValueException, as for every other model); only a
   * valid value takes the outlier treatment. MiningSchema::prepare itself is
   * not called: it treats outliers after an asIs or asMissing invalid
   * treatment, and it leaves an invalid value with no invalidValueTreatment
   * attribute to MiningSchema::validate, which refuses asIs as well -- JPMML
   * does neither here.
   *
   * The inner model reads the record as text again, so a value the outer
   * schema changed is written back as text (java_format.h; a string as
   * itself, a number as a double, an integer as a long, so that it reads back
   * as the same value), missing as empty. A value it leaves alone is passed on
   * exactly as given.
   *
   * Runs in a StringScope of the document: reading a string value makes a
   * string Value, and there is no StringScope outside a scoring entry point.
   */
  std::unordered_map<std::string, std::string> apply_mining_schema(
      std::unordered_map<std::string, std::string> flat) const {
    const cpmml::StringScope strings_scope(cpmml::StringScope::Scoring{}, *strings);
    using Invalid = InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue;

    for (const MiningField& field : mining_schema.miningfields) {
      if (field.field_usage_type.value == FieldUsageType::FieldUsageTypeValue::TARGET ||
          field.field_usage_type.value == FieldUsageType::FieldUsageTypeValue::GROUP)
        continue;

      const auto found = flat.find(field.name);
      const auto put = [&](const Value& value) {
        if (value.is_missing()) {
          if (found != flat.end()) found->second.clear();
        } else {
          flat[field.name] = input_text(field, value);
        }
      };
      // asIs keeps the input as given, for the inner MiningField to judge.
      const auto treat_invalid = [&](const Value& value) {
        if (field.invalidvalue_treatmentmethod.value != Invalid::AS_IS) put(field.handle_invalid(value));
      };

      if (found == flat.end() || found->second.empty() || field.is_missing_marker(found->second)) {
        put(field.handle_missing());
        continue;
      }

      Value value;
      switch (field.parse_input(found->second, value)) {
        case MiningField::InputStatus::MALFORMED:
          put(field.handle_malformed(found->second));  // throws for returnInvalid and asIs
          continue;
        case MiningField::InputStatus::INVALID:
          treat_invalid(value);
          continue;
        case MiningField::InputStatus::VALID:
          break;
      }

      if (field.is_invalid(value))
        treat_invalid(value);
      else if (field.is_outlier(value))
        put(field.handle_outlier(value));
    }
    return flat;
  }

  // A value of `field` as input text that reads back as the same value.
  static std::string input_text(const MiningField& field, const Value& value) {
    if (field.datatype.value == DataType::DataTypeValue::STRING) return std::string(value.text());
    // A float is written as the double it widens to, which reads back as the
    // same float (MiningField::parse_input rounds it to the nearest float).
    const java::Kind kind = java_kind(field.datatype);
    return java::text({value.as_double(), kind == java::Kind::FLOAT ? java::Kind::DOUBLE : kind});
  }

  // The enumeration is iforest | ocsvm | clusterMeanDist | other. Everything
  // that was not "iforest" used to be scored as a raw inner prediction, so
  // clusterMeanDist and an unrecognised or absent algorithmType were all the
  // same branch.
  // ALGORITHM-TYPE, required. "other" names an algorithm the document does
  // not define, so there is no scoring procedure to apply. The spec spells
  // the first "iforest"; "iForest" used to be accepted through case-folding.
  static constexpr EnumEntry<Algorithm> ALGORITHM_TYPE_TABLE[] = {
      {"iforest", Algorithm::IFOREST},
      {"ocsvm", Algorithm::OCSVM},
      {"clusterMeanDist", Algorithm::CLUSTER_MEAN_DIST},
      {"other", Algorithm::IFOREST, nullptr, EnumSupport::UNSUPPORTED},
  };

  /**
   * "The MeanClusterDistances element contains an array of non-negative real
   *  values, it is required when the algorithm type is clusterMeanDist." One
   *  entry per cluster, in cluster order.
   */
  void parse_mean_cluster_distances(const XmlNode& ad) {
    const bool present = ad.exists_child("MeanClusterDistances");

    if (algorithm != Algorithm::CLUSTER_MEAN_DIST) {
      if (present)
        throw cpmml::ParsingException("AnomalyDetectionModel: MeanClusterDistances is only for clusterMeanDist");
      return;
    }

    if (clustering == nullptr)
      throw cpmml::ParsingException("AnomalyDetectionModel: clusterMeanDist requires a ClusteringModel");
    if (!present)
      throw cpmml::ParsingException("AnomalyDetectionModel: clusterMeanDist requires MeanClusterDistances");

    for (const auto& raw : parse_array(ad.get_child("MeanClusterDistances").get_child("Array").value()))
      mean_cluster_distances.push_back(to_double(raw));

    if (mean_cluster_distances.size() != clustering->clusters.size())
      throw cpmml::ParsingException("AnomalyDetectionModel: MeanClusterDistances has " +
                                    std::to_string(mean_cluster_distances.size()) + " entries for " +
                                    std::to_string(clustering->clusters.size()) + " clusters");

    for (const double d : mean_cluster_distances)
      if (d < 0.0)
        throw cpmml::ParsingException("AnomalyDetectionModel: MeanClusterDistances must be non-negative, got " +
                                      java::double_text(d));
  }

  /**
   * "its anomaly index defined as the ratio of the distances/similarities (with
   *  special treatment for the average distance of 0)."
   *
   * The spec's own worked example is this arithmetic and nothing else: a case
   * at distance 0.048 from the first cluster, whose mean distance is 0.165,
   * scores 0.048/0.165 = 0.291; one at 0.457 from the third, mean 0.210, scores
   * 2.176 and is flagged. cPMML returned the bare 0.048 and 0.457, so every
   * clusterMeanDist score was off by an unapplied per-cluster divisor -- and
   * off by a *different* divisor per cluster, so it was not even a monotone
   * rescaling of the right answer.
   *
   * The "special treatment" is left unspecified, so this takes the reading the
   * ratio forces: against a cluster whose members all sat exactly on its
   * centre, any distance at all is unboundedly anomalous, and no distance is
   * not anomalous.
   */
  double cluster_mean_dist_score(const ClusteringModel::Nearest& n) const {
    const double mean = mean_cluster_distances[n.index];
    if (mean > 0.0) return n.distance / mean;

    return n.distance > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
  }

  double iforest_score(double avg_path) const {
    double c_n = compute_c_n(sample_data_size);
    return std::pow(2.0, -avg_path / c_n);
  }

  // Expected path length in a random binary search tree of n samples
  // Liu et al. (2008), Eq. 1
  static double compute_c_n(double n) {
    if (n <= 1.0) return 1.0;
    const double euler = 0.5772156649015328;
    double H = std::log(n - 1.0) + euler;  // harmonic number approx
    return 2.0 * H - 2.0 * (n - 1.0) / n;
  }
};

#endif
