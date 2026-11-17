
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_ENSEMBLEMODEL_H
#define CPMML_ENSEMBLEMODEL_H

#include <memory>

#include "associationmodel/associationmodel.h"
#include "baselinemodel/baselinemodel.h"
#include "clusteringmodel/clusteringmodel.h"
#include "core/internal_model.h"
#include "gaussianprocessmodel/gaussianprocessmodel.h"
#include "generalregressionmodel/generalregressionmodel.h"
#include "knnmodel/knnmodel.h"
#include "multiplemodelmethod.h"
#include "naivebayesmodel/naivebayesmodel.h"
#include "neuralnetwork/neuralnetworkmodel.h"
#include "regressionmodel/regressionmodel.h"
#include "rulesetmodel/rulesetmodel.h"
#include "scorecard/scorecardmodel.h"
#include "svmmodel/svmmodel.h"
#include "treemodel/treemodel.h"

/**
 * @class EnsembleModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/MultipleModels.html">PMML MiningModel</a>.
 *
 * Through this class are represented all ensemble models. For instance, the
 * Random Forest Model or the Gradient Boosted Trees model. See also
 * MultipleModelMethod.
 */
class EnsembleModel : public InternalModel {
 public:
  Predicate predicate;
  MultipleModelMethod multiplemodelmethod;
  std::vector<Segment> ensemble;
  std::function<std::unique_ptr<InternalScore>(const Sample&)> score_ensemble;

  EnsembleModel() = default;

  EnsembleModel(const XmlNode& node, const DataDictionary& data_dictionary,
                const TransformationDictionary& transformation_dictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        multiplemodelmethod(node.get_child("Segmentation").get_attribute("multipleModelMethod"),
                            InternalModel::mining_function,
                            MultipleModelMethod::missing_prediction_from(node.get_child("Segmentation")),
                            MultipleModelMethod::classes_of(InternalModel::target_field)) {
    PredicateBuilder predicate_builder(indexer);
    for (const auto& child : node.get_child("Segmentation").get_childs("Segment")) {
      std::shared_ptr<InternalModel> model = build_segment_model(
          child, data_dictionary, InternalModel::transformation_dictionary, predicate_builder, indexer);
      // What this MiningModel's scope has evaluated by the time a segment runs:
      // its whole derivedfields_dag (see InternalModel::scope_under).
      model->scope_under(InternalModel::derivedfields_dag);
      ensemble.push_back(Segment(child, predicate_builder, model));
      ensemble.back().has_outputs = declares_outputs(*model);
    }

    // The weighted median is defined for positive weights, and JPMML 1.7.7
    // throws on a negative Segment@weight under weightedSum and weightedMedian
    // whenever that segment predicts. cPMML refuses the document at load.
    const auto method = multiplemodelmethod.value;
    if (method == MultipleModelMethod::MultipleModelMethodType::WEIGHTED_SUM ||
        method == MultipleModelMethod::MultipleModelMethodType::WEIGHTED_MEDIAN)
      for (const auto& segment : ensemble)
        if (segment.weight < 0)
          throw cpmml::ParsingException("Segment@weight is negative under " + multiplemodelmethod.to_string());

    score_ensemble = std::bind(multiplemodelmethod.function, std::placeholders::_1, ensemble);
    base_sample = create_basesample(indexer);
  };

  // The method's result may carry segment OutputFields (see
  // MultipleModelMethod, "segment OutputFields"); this MiningModel's own
  // OutputFields, added by the caller, shadow those of the same name.
  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    std::unique_ptr<InternalScore> score = score_ensemble(sample);
    MultipleModelMethod::drop_shadowed(*score, output);
    return score;
  }

  // Whether a segment model declares an OutputField, itself or in one of its
  // own segments.
  static bool declares_outputs(const InternalModel& model) {
    if (!model.output.raw_outputfields.empty()) return true;
    if (const auto* nested = dynamic_cast<const EnsembleModel*>(&model))
      for (const auto& segment : nested->ensemble)
        if (segment.has_outputs) return true;
    return false;
  }

  inline std::string predict_raw(const Sample& sample) const override {
    std::unique_ptr<InternalScore> score(score_ensemble(sample));

    return score->score;
  }

  // For modelChain, prefer the outer model's first predictedValue OutputField.
  // If none exists, delegate to the last segment's output_name() so that the
  // regression/classification result of the chain is surfaced correctly.
  std::string output_name() const {
    if (multiplemodelmethod.value == MultipleModelMethod::MultipleModelMethodType::MODEL_CHAIN
        && !ensemble.empty()) {
      // Check outer Output for an explicit predictedValue field first.
      if (!output.empty) {
        for (const auto& of : output.raw_outputfields) {
          const auto t = of.expression_type.value;
          if (t == OutputExpressionType::OutputExpressionTypeValue::PREDICTED_VALUE ||
              t == OutputExpressionType::OutputExpressionTypeValue::PREDICTED_DISPLAY_VALUE)
            return of.name;
        }
      }
      // Fall back to last segment's output_name (covers empty outer Output and
      // outer Output with only entityId/transformedValue fields).
      const auto& last_model = ensemble.back().model;
      if (!last_model->output.empty)
        return last_model->output_name();
    }
    return InternalModel::output_name();
  }

  static std::unique_ptr<InternalModel> build_segment_model(const XmlNode& node, const DataDictionary& data_dictionary,
                                                            const TransformationDictionary& transformation_dictionary,
                                                            const PredicateBuilder& predicate_builder,
                                                            const std::shared_ptr<Indexer>& indexer) {
    // Support all PMML MODEL-ELEMENT types inside Segmentation/Segment.
    if (node.exists_child("MiningModel"))
      return std::make_unique<EnsembleModel>(node.get_child("MiningModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("TreeModel"))
      return std::make_unique<TreeModel>(node.get_child("TreeModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("RegressionModel"))
      return std::make_unique<RegressionModel>(node.get_child("RegressionModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("NeuralNetwork"))
      return std::make_unique<NeuralNetworkModel>(node.get_child("NeuralNetwork"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("SupportVectorMachineModel"))
      return std::make_unique<SupportVectorMachineModel>(node.get_child("SupportVectorMachineModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("NearestNeighborModel"))
      return std::make_unique<NearestNeighborModel>(node.get_child("NearestNeighborModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("GeneralRegressionModel"))
      return std::make_unique<GeneralRegressionModel>(node.get_child("GeneralRegressionModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("NaiveBayesModel"))
      return std::make_unique<NaiveBayesModel>(node.get_child("NaiveBayesModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("ClusteringModel"))
      return std::make_unique<ClusteringModel>(node.get_child("ClusteringModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("Scorecard"))
      return std::make_unique<ScorecardModel>(node.get_child("Scorecard"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("RuleSetModel"))
      return std::make_unique<RuleSetModel>(node.get_child("RuleSetModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("BaselineModel"))
      return std::make_unique<BaselineModel>(node.get_child("BaselineModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("GaussianProcessModel"))
      return std::make_unique<GaussianProcessModel>(node.get_child("GaussianProcessModel"), data_dictionary, transformation_dictionary, indexer);
    if (node.exists_child("AssociationModel"))
      return std::make_unique<AssociationModel>(node.get_child("AssociationModel"), data_dictionary, transformation_dictionary, indexer);

    throw cpmml::ParsingException("Unsupported model type in ensemble segment");
  }
};

#endif
