
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
 * reference baseline distribution.
 *
 * Supported test statistics:
 * - zScore / zValue : (x - mean) / sqrt(variance)
 *     GaussianDistribution baseline: mean and variance attributes
 *     PoissonDistribution baseline: mean attribute (variance = mean)
 *     UniformDistribution baseline: lower, upper; mean=(L+U)/2, var=(U-L)²/12
 * - chiSquare       : (x - expected)² / expected  (CountTable / NormalizedCountTable)
 * - logProb         : log P(x) using DiscreteDistribution / CountTable baseline
 */
class BaselineModel : public InternalModel {
 public:
  enum class TestStat { ZSCORE, CHI_SQUARE, LOG_PROB };

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
    if (stat_str == "zscore" || stat_str == "zvalue")
      stat = TestStat::ZSCORE;
    else if (stat_str == "chisquare")
      stat = TestStat::CHI_SQUARE;
    else
      stat = TestStat::LOG_PROB;

    XmlNode base_node = td.get_child("Baseline");

    if (base_node.exists_child("GaussianDistribution")) {
      XmlNode g = base_node.get_child("GaussianDistribution");
      baseline.mean = g.get_double_attribute("mean");
      baseline.variance = g.get_double_attribute("variance");
      baseline.is_gaussian = true;
    } else if (base_node.exists_child("PoissonDistribution")) {
      XmlNode p = base_node.get_child("PoissonDistribution");
      baseline.mean = p.get_double_attribute("mean");
      baseline.variance = baseline.mean;  // Poisson: variance = mean
      baseline.is_gaussian = true;
    } else if (base_node.exists_child("UniformDistribution")) {
      XmlNode u = base_node.get_child("UniformDistribution");
      double lower = u.get_double_attribute("lower");
      double upper = u.get_double_attribute("upper");
      baseline.mean = (lower + upper) / 2.0;
      baseline.variance = (upper - lower) * (upper - lower) / 12.0;
      baseline.is_gaussian = true;
    } else {
      // Discrete distribution: CountTable, NormalizedCountTable, or DiscreteDistribution
      baseline.is_gaussian = false;
      std::string table_tag = "DiscreteDistribution";
      if (base_node.exists_child("CountTable")) table_tag = "CountTable";
      else if (base_node.exists_child("NormalizedCountTable")) table_tag = "NormalizedCountTable";
      XmlNode table = base_node.get_child(table_tag);
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
      case TestStat::CHI_SQUARE: {
        // (x - expected)^2 / expected; for continuous: (x - mean)^2 / variance
        if (baseline.is_gaussian) {
          return (x - baseline.mean) * (x - baseline.mean) / baseline.variance;
        } else {
          // For count data: sum of (observed - expected)^2 / expected across cells
          // Single-cell form: use the count for x as observed vs proportional expected
          auto it = baseline.counts.find(x);
          if (it == baseline.counts.end() || baseline.total <= 0) return 0.0;
          double expected = it->second;
          double diff = x - expected;
          return (expected > 0) ? diff * diff / expected : 0.0;
        }
      }
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
