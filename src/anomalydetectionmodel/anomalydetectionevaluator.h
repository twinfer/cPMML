
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_ANOMALYDETECTIONEVALUATOR_H
#define CPMML_ANOMALYDETECTIONEVALUATOR_H

#include <cmath>
#include <memory>
#include <string>

#include "core/internal_evaluator.h"
#include "core/internal_score.h"
#include "core/xmlnode.h"
#include "ensemblemodel/ensemblemodel.h"

/**
 * @class AnomalyDetectionEvaluator
 *
 * Implementation of InternalEvaluator for <a
 * href="http://dmg.org/pmml/v4-4/AnomalyDetectionModel.html">PMML AnomalyDetectionModel</a>.
 *
 * Wraps an inner MiningModel (typically an isolation forest ensemble) and
 * converts the raw average path length to a normalised anomaly score in [0,1]:
 *
 *   score = 2^( -E[h(x)] / c(n) )
 *
 * where c(n) = 2*(ln(n-1)+Euler) - 2*(n-1)/n  (Liu et al., 2008).
 * Higher scores indicate greater anomalousness.
 */
class AnomalyDetectionEvaluator : public InternalEvaluator {
 public:
  std::unique_ptr<EnsembleModel> inner;
  double mean_threshold = 0.5;
  double sample_data_size = 256.0;
  bool is_iforest = true;

  explicit AnomalyDetectionEvaluator(const XmlNode &pmml_root)
      : InternalEvaluator(pmml_root) {
    XmlNode ad = pmml_root.get_child("AnomalyDetectionModel");
    mean_threshold = ad.get_double_attribute("meanThreshold");
    sample_data_size = ad.get_double_attribute("sampleDataSize");
    is_iforest = to_lower(ad.get_attribute("algorithmType")) == "iforest";

    if (!ad.exists_child("MiningModel"))
      throw cpmml::ParsingException("AnomalyDetectionModel requires a MiningModel inner model");

    inner = std::make_unique<EnsembleModel>(
        ad.get_child("MiningModel"), data_dictionary, transformation_dictionary, indexer);
  }

  inline bool validate(const std::unordered_map<std::string, std::string> &sample) override {
    return inner->validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string> &sample) const override {
    double raw = to_double(inner->predict(sample));
    double s = is_iforest ? iforest_score(raw) : raw;
    return std::make_unique<InternalScore>(s);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string> &sample) const override {
    double raw = to_double(inner->predict(sample));
    double s = is_iforest ? iforest_score(raw) : raw;
    return std::to_string(s);
  }

  inline std::string get_target_name() const override { return inner->target_field.name; }

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
