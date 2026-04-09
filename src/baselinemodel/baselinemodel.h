
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BASELINEMODEL_H
#define CPMML_BASELINEMODEL_H

#include <cmath>
#include <string>
#include <unordered_map>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/internal_score.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"

/**
 * @class BaselineModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/BaselineModel.html">PMML BaselineModel</a>.
 *
 * Computes a statistical test statistic comparing an observed field value to a
 * reference baseline distribution. Supported test statistics:
 * - zScore  : (x - mean) / sqrt(variance) using GaussianDistribution baseline
 * - logProb : log P(x) using DiscreteDistribution / CountTable baseline
 */
class BaselineModel : public InternalModel {
 public:
  enum class TestStat { ZSCORE, LOG_PROB };

  struct BaselineDist {
    // Gaussian
    double mean = 0.0;
    double variance = 1.0;
    // Discrete: category encoded value → count
    std::unordered_map<double, double> counts;
    double total = 0.0;
    bool is_gaussian = true;
  };

  // --- Members ---

  std::string test_field;
  size_t test_field_idx = 0;
  TestStat stat = TestStat::ZSCORE;
  BaselineDist baseline;

  // --- Constructors ---

  BaselineModel() = default;

  BaselineModel(const XmlNode &node, const DataDictionary &data_dictionary,
                const TransformationDictionary &transformation_dictionary,
                const std::shared_ptr<Indexer> &indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer) {
    XmlNode td = node.get_child("TestDistributions");
    test_field = td.get_attribute("field");
    test_field_idx = indexer->get_index(test_field);

    std::string stat_str = to_lower(td.get_attribute("testStatistic"));
    stat = (stat_str == "zscore") ? TestStat::ZSCORE : TestStat::LOG_PROB;

    XmlNode base_node = td.get_child("Baseline");

    if (base_node.exists_child("GaussianDistribution")) {
      XmlNode g = base_node.get_child("GaussianDistribution");
      baseline.mean = g.get_double_attribute("mean");
      baseline.variance = g.get_double_attribute("variance");
      baseline.is_gaussian = true;
    } else {
      // Discrete distribution: CountTable or DiscreteDistribution
      baseline.is_gaussian = false;
      XmlNode table = base_node.exists_child("CountTable")
                          ? base_node.get_child("CountTable")
                          : base_node.get_child("DiscreteDistribution");
      for (const auto &fv : table.get_childs("FieldValue")) {
        double key = Value(fv.get_attribute("value")).value;
        double count = to_double(fv.get_attribute("count"));
        baseline.counts[key] = count;
        baseline.total += count;
      }
    }
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample &sample) const override {
    return std::make_unique<InternalScore>(compute(sample));
  }

  inline std::string predict_raw(const Sample &sample) const override {
    return std::to_string(compute(sample));
  }

 private:
  double compute(const Sample &sample) const {
    const double x = sample[test_field_idx].value.value;
    switch (stat) {
      case TestStat::ZSCORE:
        return (x - baseline.mean) / std::sqrt(baseline.variance);
      case TestStat::LOG_PROB: {
        auto it = baseline.counts.find(x);
        double p = (it != baseline.counts.end() && baseline.total > 0)
                       ? it->second / baseline.total
                       : 1.0 / (baseline.total + 1.0);
        return std::log(p);
      }
    }
    return 0.0;
  }
};

#endif
