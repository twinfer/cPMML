
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

  inline bool validate(const std::unordered_map<std::string, std::string>& sample) override {
    return nn.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string>& sample) const override {
    return nn.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string>& sample) const override {
    return nn.predict(sample);
  }

  inline std::string get_target_name() const override { return nn.target_field.name; }
  inline std::string output_name() const override { return nn.output_name(); }
};

#endif
