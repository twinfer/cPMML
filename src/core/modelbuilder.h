
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MODELBUILDER_H
#define CPMML_MODELBUILDER_H

#include <memory>
#include <pugixml.hpp>
#include <string>

#include "anomalydetectionmodel/anomalydetectionevaluator.h"
#include "associationmodel/associationevaluator.h"
#include "baselinemodel/baselinemodel.h"
#include "bind/load_scope.h"
#include "clusteringmodel/clusteringmodel.h"
#include "datadictionary.h"
#include "ensemblemodel/ensemblemodel.h"
#include "gaussianprocessmodel/gaussianprocessmodel.h"
#include "generalregressionmodel/generalregressionmodel.h"
#include "header.h"
#include "internal_evaluator.h"
#include "knnmodel/knnmodel.h"
#include "model_evaluator.h"
#include "naivebayesmodel/naivebayesmodel.h"
#include "neuralnetwork/neuralnetworkmodel.h"
#include "regressionmodel/regressionmodel.h"
#include "rulesetmodel/rulesetmodel.h"
#include "scorecard/scorecardmodel.h"
#include "svmmodel/svmmodel.h"
#include "textmodel/textmodel.h"
#include "timeseriesmodel/timeseriesevaluator.h"
#include "treemodel/treemodel.h"
#include "xmlnode.h"

/**
 * @class ModelBuilder
 *
 * Factory class to create InternalEvaluator objects.
 */
class ModelBuilder {
 public:
  inline static std::unique_ptr<InternalEvaluator> build(const std::string& filename, const bool zipped) {
    std::vector<char> file_data = read_file(filename, zipped);
    pugi::xml_document document;
    pugi::xml_parse_result result = document.load_buffer(file_data.data(), file_data.size());
    if (!result) throw cpmml::ParsingException(std::string("XML parsing error: ") + result.description());

    XmlNode xmlNode(document.child("PMML"));
    // The binder decides names and scopes at load (docs/design/typed-core.md
    // P2.1, bind/load_scope.h): its errors of names and scopes fail the load,
    // and while the model is built every field reference is resolved through
    // it (scope_resolve).
    const auto scope = std::make_shared<const cpmml::bind::LoadScope>(
        document.child("PMML"), std::string_view(file_data.data(), file_data.size()));
    scope->check();
    const cpmml::bind::LoadScope::Active active_scope(*scope);
    // Parse-time indexer scope guard: evaluator constructors set
    // Value::active_indexer (via make_and_activate_indexer); this guard
    // restores the previous value on every exit path so the thread-local never
    // dangles past the model's lifetime.
    Value::ScopedIndexer parse_scope(Value::active_indexer);

    std::unique_ptr<InternalEvaluator> evaluator;
    for (const auto& entry : MODEL_TABLE)
      if (xmlNode.exists_child(entry.element)) {
        evaluator = entry.make(xmlNode, entry.element);
        break;
      }

    if (!evaluator) throw cpmml::ParsingException("unsupported model type");

    // Parsing is done: freeze the per-model string map so concurrent scoring
    // threads can read it without synchronization.
    if (evaluator->indexer) evaluator->indexer->freeze();
    evaluator->load_scope = scope;

    return evaluator;
  }

 private:
  using Factory = std::unique_ptr<InternalEvaluator> (*)(const XmlNode&, const char*);

  // Most models want the stock ModelEvaluator<T>; a few carry their own
  // evaluator because they score from something other than a flattened
  // string map. Both shapes go in one table.
  template <class ModelT>
  static std::unique_ptr<InternalEvaluator> wrap_model(const XmlNode& node, const char* element) {
    return std::make_unique<ModelEvaluator<ModelT>>(node, element);
  }

  template <class EvaluatorT>
  static std::unique_ptr<InternalEvaluator> wrap_evaluator(const XmlNode& node, const char*) {
    return std::make_unique<EvaluatorT>(node);
  }

  struct ModelKind {
    const char* element;
    Factory make;
  };

  // Order is the order of the if/else chain this replaces, and it matters: a
  // document may carry more than one MODEL-ELEMENT and the first match wins.
  static constexpr ModelKind MODEL_TABLE[] = {
      {"MiningModel", &wrap_model<EnsembleModel>},
      {"RegressionModel", &wrap_model<RegressionModel>},
      {"TreeModel", &wrap_model<TreeModel>},
      {"NeuralNetwork", &wrap_model<NeuralNetworkModel>},
      {"SupportVectorMachineModel", &wrap_model<SupportVectorMachineModel>},
      {"NearestNeighborModel", &wrap_model<NearestNeighborModel>},
      {"GeneralRegressionModel", &wrap_model<GeneralRegressionModel>},
      {"Scorecard", &wrap_model<ScorecardModel>},
      {"NaiveBayesModel", &wrap_model<NaiveBayesModel>},
      {"ClusteringModel", &wrap_model<ClusteringModel>},
      {"RuleSetModel", &wrap_model<RuleSetModel>},
      {"BaselineModel", &wrap_model<BaselineModel>},
      {"AnomalyDetectionModel", &wrap_evaluator<AnomalyDetectionEvaluator>},
      {"GaussianProcessModel", &wrap_model<GaussianProcessModel>},
      {"TextModel", &wrap_evaluator<TextEvaluator>},
      {"TimeSeriesModel", &wrap_evaluator<TimeSeriesEvaluator>},
      {"AssociationModel", &wrap_evaluator<AssociationEvaluator>},
  };
};

#endif
