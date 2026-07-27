
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_ANOMALYDETECTIONEVALUATOR_H
#define CPMML_ANOMALYDETECTIONEVALUATOR_H

#include <cmath>
#include <memory>
#include <string>

#include "clusteringmodel/clusteringmodel.h"
#include "core/internal_evaluator.h"
#include "core/internal_model.h"
#include "core/internal_score.h"
#include "core/xmlnode.h"
#include "ensemblemodel/ensemblemodel.h"
#include "svmmodel/svmmodel.h"

/**
 * @class AnomalyDetectionEvaluator
 *
 * Implementation of InternalEvaluator for <a
 * href="http://dmg.org/pmml/v4-4/AnomalyDetectionModel.html">PMML AnomalyDetectionModel</a>.
 *
 * Supported inner model types (algorithmType):
 * - iforest (MiningModel):       score = 2^(-E[h]/c(n))
 * - ocsvm (SupportVectorMachineModel): score = raw SVM decision value
 * - clusterMeanDist (ClusteringModel): score = raw mean cluster distance
 *
 * For iForest, c(n) = 2*(ln(n-1)+Euler) - 2*(n-1)/n  (Liu et al., 2008).
 * Higher scores indicate greater anomalousness.
 */
class AnomalyDetectionEvaluator : public InternalEvaluator {
 public:
  std::unique_ptr<InternalModel> inner;
  const ClusteringModel* clustering = nullptr;  // set for clusterMeanDist (non-owning view of inner)
  double sample_data_size = 256.0;
  bool is_iforest = true;

  explicit AnomalyDetectionEvaluator(const XmlNode& pmml_root) : InternalEvaluator(pmml_root) {
    XmlNode ad = pmml_root.get_child("AnomalyDetectionModel");
    // There was a `meanThreshold` member here, read from an attribute of the
    // same name and used by nothing. AnomalyDetectionModel has no such
    // attribute — the schema allows only modelName, algorithmName,
    // functionName, algorithmType, sampleDataSize and isScorable — so it was
    // not a dropped feature but an invented one. Removed rather than wired up.
    sample_data_size = ad.get_double_attribute("sampleDataSize", 256.0);
    std::string algo = to_lower(ad.get_attribute("algorithmType"));
    is_iforest = (algo == "iforest");

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
  }

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    auto flat = flatten_input(arguments);
    // clusterMeanDist scores with the distance to the nearest cluster —
    // inner->predict would return the cluster *id*, which is not a score.
    double raw = clustering ? clustering->min_distance(flat) : to_double(inner->predict(flat));
    double s = is_iforest ? iforest_score(raw) : raw;
    return std::make_unique<InternalScore>(s);
  }

  inline bool validate(const Input& arguments) const override {
    return inner->validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return inner->target_field.name; }
  inline std::string output_name() const override { return inner->output_name(); }
  inline std::string mining_function_name() const override { return "CLASSIFICATION"; }

 private:
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
