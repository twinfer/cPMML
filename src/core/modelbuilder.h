
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
#include "neuralnetwork/neuralnetworkevaluator.h"
#include "regressionmodel/regressionevaluator.h"
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
    else
      throw cpmml::ParsingException("unsupported model type");

    return evaluator;
  }
};

#endif
