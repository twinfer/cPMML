
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BASELINEMODEL_H
#define CPMML_BASELINEMODEL_H

#include <cmath>
#include <string>
#include <unordered_map>

#include "bind/load_scope.h"
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
 * Only zValue is implemented -- see parse_test_statistic for why the other four
 * cannot be, one record at a time:
 *
 *   "the value of defects is converted to a standard score or z-value using the
 *    formula score = (defects - mean) / sqrt (variance)"
 *
 * so the baseline has to supply a mean and a variance:
 *   GaussianDistribution : mean and variance attributes
 *   PoissonDistribution  : mean attribute (variance = mean)
 *   UniformDistribution  : lower, upper; mean=(L+U)/2, var=(U-L)^2/12
 *
 * A CountTable, NormalizedCountTable, DiscreteDistribution or AnyDistribution
 * baseline supplies neither, so it is rejected too.
 */
class BaselineModel : public InternalModel {
 public:
  struct BaselineDist {
    double mean = 0.0;
    double variance = 1.0;
  };

  // --- Members ---

  std::string test_field;
  size_t test_field_idx = 0;
  BaselineDist baseline;

  // --- Constructors ---

  BaselineModel() = default;

  BaselineModel(const XmlNode& node, const DataDictionary& data_dictionary,
                const TransformationDictionary& transformation_dictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer) {
    XmlNode td = node.get_child("TestDistributions");
    test_field = td.required_attribute("field");
    test_field_idx = cpmml::bind::scope_resolve(*indexer, td, test_field).index;  // scope.resolve (P2.1)

    parse_test_statistic(td);

    XmlNode base_node = td.get_child("Baseline");

    if (base_node.exists_child("GaussianDistribution")) {
      XmlNode g = base_node.get_child("GaussianDistribution");
      baseline.mean = g.required_double_attribute("mean");
      baseline.variance = g.required_double_attribute("variance");
    } else if (base_node.exists_child("PoissonDistribution")) {
      XmlNode p = base_node.get_child("PoissonDistribution");
      baseline.mean = p.required_double_attribute("mean");
      baseline.variance = baseline.mean;  // Poisson: variance = mean
    } else if (base_node.exists_child("UniformDistribution")) {
      XmlNode u = base_node.get_child("UniformDistribution");
      double lower = u.required_double_attribute("lower");
      double upper = u.required_double_attribute("upper");
      baseline.mean = (lower + upper) / 2.0;
      baseline.variance = (upper - lower) * (upper - lower) / 12.0;
    } else {
      throw cpmml::ParsingException(
          "BaselineModel: testStatistic=\"zValue\" needs a baseline supplying a mean and a variance "
          "(GaussianDistribution, PoissonDistribution or UniformDistribution)");
    }

    // sqrt(0) is 0 and the division that follows is 1/0. A degenerate baseline
    // yields no z-value at all, so it is refused at load rather than turned
    // into an infinity halfway through a batch.
    if (!(baseline.variance > 0.0))
      throw cpmml::ParsingException("BaselineModel: baseline variance must be positive, got " +
                                    java::double_text(baseline.variance));
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    return std::make_unique<InternalScore>(compute(sample));
  }

  inline std::string predict_raw(const Sample& sample) const override { return java::double_text(compute(sample)); }

 private:
  /**
   * PMML 4.4 enumerates testStatistic as zValue, chiSquareIndependence,
   * chiSquareDistribution, CUSUM and scalarProduct. cPMML recognised neither
   * chi-square token, and anything it did not recognise -- including CUSUM and
   * scalarProduct -- fell through to a logProb branch that the spec does not
   * define at all.
   *
   * The four statistics beyond zValue are rejected rather than approximated,
   * because each needs an *aggregated observed distribution* that single-record
   * scoring cannot form:
   *
   *   chiSquareDistribution  "a reference to a field describing the sampled
   *                           distribution and a discrete distribution
   *                           describing the expectations is required. A field
   *                           reference to a sampled distribution can be
   *                           provided using the Aggregate element."
   *   chiSquareIndependence  needs a contingency table built from field
   *                           references describing its rows.
   *   scalarProduct          "sum over i of Ci x ci/N" over an observed
   *                           distribution, incremented per record via
   *                           weightField.
   *   CUSUM                  a running sum across records, and requires an
   *                           Alternate element.
   *
   * All four are defined over a window of records ("windowSize... how much
   * history the model uses"), which cPMML has no notion of. What was computed
   * for "chiSquare" was (x - count(x))^2 / count(x): a category's *encoded
   * value* differenced against that category's *count*, which is not a
   * quantity. A load failure naming the reason beats a number that means
   * nothing.
   */
  //
  // zScore, which this used to accept (case-folded) as an alias, is not a
  // PMML token and is now invalid like any other.
  static constexpr EnumEntry<bool> TEST_STATISTIC_TABLE[] = {
      {"zValue", true},
      {"chiSquareIndependence", false, nullptr, EnumSupport::UNSUPPORTED},
      {"chiSquareDistribution", false, nullptr, EnumSupport::UNSUPPORTED},
      {"CUSUM", false, nullptr, EnumSupport::UNSUPPORTED},
      {"scalarProduct", false, nullptr, EnumSupport::UNSUPPORTED},
  };

  static void parse_test_statistic(const XmlNode& test_distributions) {
    read_enum(test_distributions, "testStatistic", TEST_STATISTIC_TABLE);
  }

  double compute(const Sample& sample) const {
    const double x = sample[test_field_idx].value.as_double();
    return (x - baseline.mean) / std::sqrt(baseline.variance);
  }
};

#endif
