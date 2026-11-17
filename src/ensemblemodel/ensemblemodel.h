
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_ENSEMBLEMODEL_H
#define CPMML_ENSEMBLEMODEL_H

#include <memory>
#include <string>

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
  Segments ensemble;
  std::function<std::unique_ptr<InternalScore>(const Sample&)> score_ensemble;

  EnsembleModel() = default;

  EnsembleModel(const XmlNode& node, const DataDictionary& data_dictionary,
                const TransformationDictionary& transformation_dictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        multiplemodelmethod(node.get_child("Segmentation"), InternalModel::mining_function,
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

    ensemble.parallel = parallel_worth_it(node.get_child("Segmentation"));
    score_ensemble = std::bind(multiplemodelmethod.function, std::placeholders::_1, ensemble);
    base_sample = create_basesample(indexer);
  };

  // ---- MULTITHREADING ---------------------------------------------------------
  //
  // Whether an OpenMP build scores this ensemble's segments in parallel. A
  // parallel loop costs a fork and a join on every record -- about 10 us with
  // Homebrew libomp and 4 threads on the machine the benchmarks ran on -- so
  // it pays only when one record's segments take several times that. The work
  // is estimated once, at load, from the document (estimated_work), in units
  // of one tree-node test. On the benchmark models a unit costs 8-50 ns, and
  // the parallel loop was faster only on AuditRandomForest (estimated 2439,
  // 70 us sequential) and slower on every other ensemble (IrisRandomForest
  // 1037, 8 us; HousingGBT* 400, 12 us; HousingRF* 95-118, 4-9 us;
  // ensemble_*_dectree 22-32, 1.5 us).
  // PARALLEL_WORK sits between them, on the sequential side: a wrong
  // "parallel" costs the fork and join on every record, a wrong "sequential"
  // only the speed-up. See docs/design/typed-core.md, Q12.
  static bool parallel_worth_it(const XmlNode& segmentation) {
#ifdef MULTITHREADING
    return estimated_work(segmentation.xml()) >= PARALLEL_WORK;
#else
    (void)segmentation;
    return false;
#endif
  }

  // CPMML_PARALLEL_WORK overrides the threshold at build time; 0 scores every
  // ensemble in parallel, which is how the parallel loop is tested on every
  // fixture.
  static constexpr double SEGMENT_WORK = 1;  // a segment's own cost: its predicate
#ifdef CPMML_PARALLEL_WORK
  static constexpr double PARALLEL_WORK = CPMML_PARALLEL_WORK;
#else
  static constexpr double PARALLEL_WORK = 1500;
#endif

  // The estimated work of scoring one record with `model` (a model element)
  // or with every segment of `model` (a Segmentation).
  static double estimated_work(const pugi::xml_node& model) {
    const std::string name = model.name();
    if (name == "Segmentation") {
      double work = 0;
      for (const auto& segment : model.children("Segment")) {
        work += SEGMENT_WORK;
        for (const auto& child : segment.children())
          if (is_model_element(child.name())) work += estimated_work(child);
      }
      return work;
    }
    if (name == "MiningModel") return estimated_work(model.child("Segmentation"));
    if (name == "TreeModel") {
      // A record walks one path: the mean depth of the leaves.
      double depth_sum = 0, leaves = 0;
      leaf_depths(model.child("Node"), 1, depth_sum, leaves);
      return leaves > 0 ? depth_sum / leaves : 0;
    }
    // Every other model reads all of its parameters on every record: count them.
    double elements = 0;
    for (const auto& child : model.children()) elements += count_elements(child);
    return elements;
  }

  static bool is_model_element(const std::string& name) {
    return name == "MiningModel" || name == "TreeModel" || name == "RegressionModel" || name == "NeuralNetwork" ||
           name == "SupportVectorMachineModel" || name == "NearestNeighborModel" ||
           name == "GeneralRegressionModel" || name == "NaiveBayesModel" || name == "ClusteringModel" ||
           name == "Scorecard" || name == "RuleSetModel" || name == "BaselineModel" ||
           name == "GaussianProcessModel" || name == "AssociationModel";
  }

  static void leaf_depths(const pugi::xml_node& node, const double depth, double& depth_sum, double& leaves) {
    bool leaf = true;
    for (const auto& child : node.children("Node")) {
      leaf = false;
      leaf_depths(child, depth + 1, depth_sum, leaves);
    }
    if (leaf) {
      depth_sum += depth;
      leaves += 1;
    }
  }

  static double count_elements(const pugi::xml_node& node) {
    if (node.type() != pugi::node_element) return 0;
    double n = 1;
    for (const auto& child : node.children()) n += count_elements(child);
    return n;
  }

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
