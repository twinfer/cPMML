
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MODELBUILDER_H
#define CPMML_MODELBUILDER_H

#include <string>

#include <pugixml.hpp>

#include "datadictionary.h"
#include "ensemblemodel/ensembleevaluator.h"
#include "header.h"
#include "internal_evaluator.h"
#include "clusteringmodel/clusteringevaluator.h"
#include "rulesetmodel/rulesetevaluator.h"
#include "baselinemodel/baselineevaluator.h"
#include "anomalydetectionmodel/anomalydetectionevaluator.h"
#include "gaussianprocessmodel/gaussianprocessevaluator.h"
#include "textmodel/textmodel.h"
#include "generalregressionmodel/generalregressionevaluator.h"
#include "knnmodel/knnevaluator.h"
#include "naivebayesmodel/naivebayesevaluator.h"
#include "neuralnetwork/neuralnetworkevaluator.h"
#include "regressionmodel/regressionevaluator.h"
#include "scorecard/scorecardevaluator.h"
#include "svmmodel/svmevaluator.h"
#include "treemodel/treeevaluator.h"
#include "treemodel/treemodel.h"
#include "xmlnode.h"

/**
 * @class ModelBuilder
 *
 * Factory class to create InternalEvaluator objects.
 */
class ModelBuilder {
 public:
  inline static std::unique_ptr<InternalEvaluator> build(const std::string &filename, const bool zipped) {
    std::vector<char> file_data = read_file(filename, zipped);
    pugi::xml_document document;
    pugi::xml_parse_result result = document.load_buffer(file_data.data(), file_data.size());
    if (!result)
      throw cpmml::ParsingException(std::string("XML parsing error: ") + result.description());

    XmlNode xmlNode(document.child("PMML"));
    std::unique_ptr<InternalEvaluator> evaluator;
    if (xmlNode.exists_child("MiningModel"))
      evaluator = std::make_unique<EnsembleEvaluator>(xmlNode);
    else if (xmlNode.exists_child("RegressionModel"))
      evaluator = std::make_unique<RegressionEvaluator>(xmlNode);
    else if (xmlNode.exists_child("TreeModel"))
      evaluator = std::make_unique<TreeEvaluator>(xmlNode);
    else if (xmlNode.exists_child("NeuralNetwork"))
      evaluator = std::make_unique<NeuralNetworkEvaluator>(xmlNode);
    else if (xmlNode.exists_child("SupportVectorMachineModel"))
      evaluator = std::make_unique<SvmEvaluator>(xmlNode);
    else if (xmlNode.exists_child("NearestNeighborModel"))
      evaluator = std::make_unique<KnnEvaluator>(xmlNode);
    else if (xmlNode.exists_child("GeneralRegressionModel"))
      evaluator = std::make_unique<GeneralRegressionEvaluator>(xmlNode);
    else if (xmlNode.exists_child("Scorecard"))
      evaluator = std::make_unique<ScorecardEvaluator>(xmlNode);
    else if (xmlNode.exists_child("NaiveBayesModel"))
      evaluator = std::make_unique<NaiveBayesEvaluator>(xmlNode);
    else if (xmlNode.exists_child("ClusteringModel"))
      evaluator = std::make_unique<ClusteringEvaluator>(xmlNode);
    else if (xmlNode.exists_child("RuleSetModel"))
      evaluator = std::make_unique<RuleSetEvaluator>(xmlNode);
    else if (xmlNode.exists_child("BaselineModel"))
      evaluator = std::make_unique<BaselineEvaluator>(xmlNode);
    else if (xmlNode.exists_child("AnomalyDetectionModel"))
      evaluator = std::make_unique<AnomalyDetectionEvaluator>(xmlNode);
    else if (xmlNode.exists_child("GaussianProcessModel"))
      evaluator = std::make_unique<GaussianProcessEvaluator>(xmlNode);
    else if (xmlNode.exists_child("TextModel"))
      evaluator = std::make_unique<TextEvaluator>(xmlNode);
    else
      throw cpmml::ParsingException("unsupported model type");

    return evaluator;
  }
};

#endif
