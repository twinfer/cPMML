
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_REGRESSIONMODEL_H
#define CPMML_REGRESSIONMODEL_H

#include <algorithm>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/miningfunction.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"
#include "normalizationmethodbuilder.h"
#include "normalizationmethodtype.h"
#include "regressionscore.h"
#include "regressiontable.h"
#include "treemodel/treescore.h"

/**
 * @class RegressionModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/Regression.html">PMML RegressionModel</a>.
 *
 * Through this class are represented models such as Linear Regression Model,
 * Logistic Regression Model, etc.
 */
class RegressionModel : public InternalModel {
 public:
  NormalizationMethodType normalization_methodtype;
  std::function<double(const double&)> regression_normalization;
  std::function<std::vector<double>(const std::vector<double>&)> classification_normalization;
  std::vector<RegressionTable> regression_tables;
  std::vector<std::string> classes;

  RegressionModel() = default;

  RegressionModel(const XmlNode& node, const DataDictionary& data_dictionary,
                  const TransformationDictionary& transformationDictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformationDictionary, indexer),
        normalization_methodtype(node.get_attribute("normalizationMethod")),
        regression_normalization(SingleNormalizationMethodBuilder::build(normalization_methodtype)),
        classification_normalization(MultiNormalizationMethodBuilder::build(
            normalization_methodtype, target_field.optype.value == OpType::OpTypeValue::ORDINAL)),
        regression_tables(RegressionTable::to_regressiontables(node.get_childs("RegressionTable"), indexer)) {
    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION)
      for (const auto& regression_table : regression_tables) classes.push_back(regression_table.target_category);
    else
      classes.push_back(mining_schema.target.name);
  }

  // A NumericPredictor or PredictorTerm whose input is missing makes the whole
  // model yield no prediction, as the spec says ("the result evaluates to a
  // missing value", and the probability formulas "do not apply") and as JPMML
  // does, for regression and for classification under every
  // normalizationMethod: one table with a missing term voids every table's
  // probability. Scoring the term as 0 gave a confident wrong answer. The spec
  // says priorProbability in Targets then defines the prediction; Target fills
  // in those probabilities but does not yet pick the predicted class from them
  // (JPMML does), so such a model returns the priors with an empty prediction.
  // A missing CategoricalPredictor field is not
  // missing here: its indicators are simply 0. A model wanting a substitute
  // says so with missingValueReplacement, which MiningSchema applies before
  // this point.
  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    if (has_missing_input(sample)) return std::make_unique<InternalScore>();
    std::vector<double> scores;
    double regressed_value;
    switch (mining_function.value) {
      case MiningFunction::MiningFunctionType::REGRESSION:
        regressed_value = regression_normalization(regression_tables[0].score(sample));
        return std::make_unique<InternalScore>(regressed_value);
      case MiningFunction::MiningFunctionType::CLASSIFICATION:
        scores = classification_normalization(get_scores(sample));
        regressed_value = *std::max_element(scores.begin(), scores.end());
        return std::make_unique<RegressionScore>(get_class(scores), regressed_value, classes, scores);
      default:
        throw cpmml::ParsingException(mining_function.to_string() + "not available in RegressionModel");
    }
  }

  inline std::string predict_raw(const Sample& sample) const override {
    if (has_missing_input(sample)) return "";
    switch (mining_function.value) {
      case MiningFunction::MiningFunctionType::REGRESSION:
        return double_to_string(regression_normalization(regression_tables[0].score(sample)));
      case MiningFunction::MiningFunctionType::CLASSIFICATION:
        return get_class(classification_normalization(get_scores(sample)));
      default:
        throw cpmml::ParsingException(mining_function.to_string() + "not available in RegressionModel");
    }
  }

  inline bool has_missing_input(const Sample& sample) const {
    for (const auto& regression_table : regression_tables)
      if (regression_table.has_missing_input(sample)) return true;
    return false;
  }

  inline std::vector<double> get_scores(const Sample& sample) const {
    std::vector<double> scores;

    for (const auto& regressionTable : regression_tables) scores.push_back(regressionTable.score(sample));

    return scores;
  }

  inline std::string get_class(const std::vector<double>& scores) const {
    double max = std::numeric_limits<double>::lowest();
    size_t _class = 0;

    for (auto i = 0u; i < regression_tables.size(); i++)
      if (scores[i] > max) {
        max = scores[i];
        _class = i;
      }

    return regression_tables[_class].target_category;
  }
};

#endif
