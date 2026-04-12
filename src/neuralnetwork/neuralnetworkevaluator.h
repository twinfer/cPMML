
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_NEURALNETWORKEVALUATOR_H
#define CPMML_NEURALNETWORKEVALUATOR_H

#include <string>

#include "core/datadictionary.h"
#include "core/header.h"
#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "neuralnetworkmodel.h"

/**
 * @class NeuralNetworkEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of NeuralNetworkModel.
 */
class NeuralNetworkEvaluator : public InternalEvaluator {
 public:
  explicit NeuralNetworkEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        nn(node.get_child("NeuralNetwork"), data_dictionary, transformation_dictionary, indexer) {}

  NeuralNetworkModel nn;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return nn.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return nn.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return nn.target_field.name; }
  inline std::string output_name() const override { return nn.output_name(); }
  inline std::string mining_function_name() const override { return nn.mining_function.to_string(); }
};

#endif
