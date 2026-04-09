
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_NEURALNETWORKMODEL_H
#define CPMML_NEURALNETWORKMODEL_H

#include <Eigen/Dense>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/datadictionary.h"
#include "core/derivedfield.h"
#include "core/internal_model.h"
#include "core/miningfunction.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"
#include "neuralactivation.h"
#include "regressionmodel/regressionscore.h"

/**
 * @class NeuralNetworkModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/NeuralNetwork.html">PMML NeuralNetworkModel</a>.
 *
 * Forward pass is computed with Eigen matrix-vector operations.
 * Supports all 14 PMML activation functions, softmax/simplemax normalization,
 * classification (argmax over output neurons) and regression.
 */
class NeuralNetworkModel : public InternalModel {
 public:
  // --- Nested types ---

  struct NeuralInput {
    std::string id;
    DerivedField derived_field;
  };

  struct Layer {
    Eigen::MatrixXd weights;  // [n_neurons x n_inputs_to_layer]
    Eigen::VectorXd biases;   // [n_neurons]
    NeuralActivationType activation;
    double threshold;
    double width;
    std::string normalization;  // "none", "softmax", "simplemax"
    std::vector<std::string> output_ids;
  };

  struct NeuralOutput {
    std::string output_neuron;
    std::string target_category;
  };

  // --- Members ---

  NeuralActivationType default_activation;
  double default_threshold;
  double default_width;
  std::string model_normalization;

  std::vector<NeuralInput> neural_inputs;
  std::vector<Layer> layers;
  std::vector<NeuralOutput> neural_outputs;
  std::vector<std::string> classes;

  // Prebuilt: output neuron id → index in last layer output vector
  std::unordered_map<std::string, size_t> output_id_to_idx;

  // --- Constructors ---

  NeuralNetworkModel() = default;

  NeuralNetworkModel(const XmlNode& node, const DataDictionary& data_dictionary,
                     const TransformationDictionary& transformation_dictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        default_activation(parse_activation(node.get_attribute("activationFunction"))),
        default_threshold(node.exists_attribute("threshold") ? to_double(node.get_attribute("threshold")) : 0.0),
        default_width(node.exists_attribute("width") ? to_double(node.get_attribute("width")) : 1.0),
        model_normalization(normalize_method_str(node.get_attribute("normalizationMethod"))) {
    parse_neural_inputs(node.get_child("NeuralInputs"), indexer);
    parse_layers(node);
    parse_neural_outputs(node.get_child("NeuralOutputs"));

    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION)
      for (const auto& no : neural_outputs) classes.push_back(no.target_category);
    else
      classes.push_back(mining_schema.target.name);

    // Build output lookup map (avoid rebuilding per score call)
    if (!layers.empty()) {
      const auto& last_ids = layers.back().output_ids;
      for (size_t i = 0; i < last_ids.size(); i++) output_id_to_idx[last_ids[i]] = i;
    }
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    Eigen::VectorXd out = forward_pass(sample);
    std::vector<double> scores = collect_scores(out);

    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION) {
      size_t best = std::max_element(scores.begin(), scores.end()) - scores.begin();
      return std::make_unique<RegressionScore>(classes[best], scores[best], classes, scores);
    }

    return std::make_unique<RegressionScore>(std::to_string(scores[0]), scores[0], classes, scores);
  }

  inline std::string predict_raw(const Sample& sample) const override {
    Eigen::VectorXd out = forward_pass(sample);
    std::vector<double> scores = collect_scores(out);

    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION) {
      size_t best = std::max_element(scores.begin(), scores.end()) - scores.begin();
      return classes[best];
    }

    return std::to_string(scores[0]);
  }

 private:
  // --- Parse helpers ---

  static std::string normalize_method_str(const std::string& s) { return (s == "null") ? "none" : to_lower(s); }

  void parse_neural_inputs(const XmlNode& inputs_node, const std::shared_ptr<Indexer>& indexer) {
    for (const auto& ni : inputs_node.get_childs("NeuralInput")) {
      NeuralInput input;
      input.id = ni.get_attribute("id");
      input.derived_field = DerivedField(ni.get_child("DerivedField"), indexer);
      neural_inputs.push_back(std::move(input));
    }
  }

  void parse_layers(const XmlNode& model_node) {
    // prev_id_to_idx: maps each neuron id in the previous "level" to its
    // column index in the weight matrix of the next layer.
    std::unordered_map<std::string, size_t> prev_id_to_idx;
    for (size_t i = 0; i < neural_inputs.size(); i++) prev_id_to_idx[neural_inputs[i].id] = i;

    for (const auto& layer_node : model_node.get_childs("NeuralLayer")) {
      auto neuron_nodes = layer_node.get_childs("Neuron");
      const size_t n = neuron_nodes.size();
      const size_t prev_n = prev_id_to_idx.size();

      NeuralActivationType act = layer_node.exists_attribute("activationFunction")
                                     ? parse_activation(layer_node.get_attribute("activationFunction"))
                                     : default_activation;
      double thr = layer_node.exists_attribute("threshold") ? to_double(layer_node.get_attribute("threshold"))
                                                            : default_threshold;
      double wid = layer_node.exists_attribute("width") ? to_double(layer_node.get_attribute("width")) : default_width;
      std::string norm = normalize_method_str(layer_node.get_attribute("normalizationMethod"));

      Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n, prev_n);
      Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
      std::vector<std::string> output_ids;
      std::unordered_map<std::string, size_t> curr_id_to_idx;

      for (size_t i = 0; i < n; i++) {
        const auto& neuron = neuron_nodes[i];
        std::string nid = neuron.get_attribute("id");
        output_ids.push_back(nid);
        curr_id_to_idx[nid] = i;

        if (neuron.exists_attribute("bias")) b[i] = to_double(neuron.get_attribute("bias"));

        for (const auto& con : neuron.get_childs("Con")) {
          const std::string from = con.get_attribute("from");
          const auto it = prev_id_to_idx.find(from);
          if (it != prev_id_to_idx.end()) W(i, it->second) = to_double(con.get_attribute("weight"));
        }
      }

      layers.push_back({std::move(W), std::move(b), act, thr, wid, norm, std::move(output_ids)});
      prev_id_to_idx = std::move(curr_id_to_idx);
    }
  }

  void parse_neural_outputs(const XmlNode& outputs_node) {
    for (const auto& no : outputs_node.get_childs("NeuralOutput")) {
      NeuralOutput output;
      output.output_neuron = no.get_attribute("outputNeuron");

      XmlNode df = no.get_child("DerivedField");
      if (df.exists_child("NormDiscrete"))
        output.target_category = df.get_child("NormDiscrete").get_attribute("value");
      else
        output.target_category = mining_schema.target.name;

      neural_outputs.push_back(std::move(output));
    }
  }

  // --- Forward pass ---

  Eigen::VectorXd forward_pass(const Sample& sample) const {
    // Build a local mutable sample large enough to hold any new indices
    // that NeuralInput DerivedFields may have registered after base_sample.
    Sample local(indexer->size());
    for (size_t i = 0; i < sample.features.size(); i++) local[i] = sample.features[i];

    // Apply NeuralInput DerivedField transformations and read input vector
    Eigen::VectorXd x(static_cast<Eigen::Index>(neural_inputs.size()));
    for (size_t i = 0; i < neural_inputs.size(); i++) {
      neural_inputs[i].derived_field.prepare(local);
      x[static_cast<Eigen::Index>(i)] = local[neural_inputs[i].derived_field.index].value.value;
    }

    // Layer-by-layer forward pass
    for (const auto& layer : layers) {
      Eigen::VectorXd pre = layer.weights * x + layer.biases;

      // Element-wise activation
      for (Eigen::Index j = 0; j < pre.size(); j++)
        pre[j] = apply_activation(pre[j], layer.activation, layer.threshold, layer.width);

      // Layer normalization (softmax/simplemax)
      if (layer.normalization == "softmax") {
        const double max_val = pre.maxCoeff();
        Eigen::VectorXd shifted = (pre.array() - max_val).exp();
        pre = shifted / shifted.sum();
      } else if (layer.normalization == "simplemax") {
        const double s = pre.sum();
        if (s > 0.0) pre /= s;
      }

      x = std::move(pre);
    }

    return x;
  }

  std::vector<double> collect_scores(const Eigen::VectorXd& out) const {
    std::vector<double> scores;
    scores.reserve(neural_outputs.size());
    for (const auto& no : neural_outputs) {
      const auto it = output_id_to_idx.find(no.output_neuron);
      if (it != output_id_to_idx.end())
        scores.push_back(out[static_cast<Eigen::Index>(it->second)]);
      else
        scores.push_back(0.0);
    }
    return scores;
  }
};

#endif
