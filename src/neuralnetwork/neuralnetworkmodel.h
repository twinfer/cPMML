
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
    Eigen::MatrixXd weights;  // [n_neurons x n_inputs_to_layer]; RBF centers when radial_basis
    Eigen::VectorXd biases;   // [n_neurons]
    NeuralActivationType activation;
    double threshold;
    NeuralNormalization normalization = NeuralNormalization::NONE;
    bool radial_basis = false;
    Eigen::VectorXd widths;     // [n_neurons], RBF only
    Eigen::VectorXd altitudes;  // [n_neurons], RBF only
    Eigen::VectorXd fan_ins;    // [n_neurons], RBF only: number of Con elements
    std::vector<std::string> output_ids;
  };

  struct NeuralOutput {
    std::string output_neuron;
    std::string target_category;
    // Inverse of a NormContinuous on the NeuralOutput DerivedField: the network
    // works in normalized space, the prediction must come back in target scale.
    std::vector<double> norm;
    std::vector<double> orig;
  };

  // --- Members ---

  NeuralActivationType default_activation;
  double default_threshold;
  double default_width;
  double default_altitude;
  NeuralNormalization model_normalization = NeuralNormalization::NONE;

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
        default_activation(read_enum(node, "activationFunction", ACTIVATION_FUNCTION_TABLE)),
        default_threshold(node.get_double_attribute("threshold", 0.0)),
        default_width(node.get_double_attribute("width", 1.0)),
        default_altitude(node.get_double_attribute("altitude", 1.0)),
        model_normalization(read_enum(node, "normalizationMethod", NN_NORMALIZATION_TABLE, NeuralNormalization::NONE)) {
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

  // A NeuralInput whose value is missing makes the model yield no prediction.
  // The NeuralNetwork spec says nothing about missing inputs and the network
  // has no arithmetic for one; JPMML 1.7.7 returns no prediction, for
  // regression and classification alike, whether the NeuralInput is a plain
  // FieldRef, a NormContinuous or NormDiscrete without mapMissingTo, a
  // NormContinuous outlier under outliers="asMissingValues", or an input no
  // neuron is connected to. A NeuralInput that defines its own substitute
  // (mapMissingTo, or missingValueReplacement in MiningSchema, applied
  // earlier) is not missing and scores normally.
  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    Eigen::VectorXd x;
    if (!neural_input_values(sample, x)) return std::make_unique<InternalScore>();
    Eigen::VectorXd out = forward_pass(std::move(x));
    std::vector<double> scores = collect_scores(out);

    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION) {
      size_t best = std::max_element(scores.begin(), scores.end()) - scores.begin();
      return std::make_unique<RegressionScore>(classes[best], scores[best], classes, scores);
    }

    return std::make_unique<RegressionScore>(java::double_text(scores[0]), scores[0], classes, scores);
  }

  inline std::string predict_raw(const Sample& sample) const override {
    Eigen::VectorXd x;
    if (!neural_input_values(sample, x)) return "";
    Eigen::VectorXd out = forward_pass(std::move(x));
    std::vector<double> scores = collect_scores(out);

    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION) {
      size_t best = std::max_element(scores.begin(), scores.end()) - scores.begin();
      return classes[best];
    }

    return java::double_text(scores[0]);
  }

 private:
  // --- Parse helpers ---

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

      const NeuralActivationType act = read_enum(layer_node, "activationFunction", ACTIVATION_FUNCTION_TABLE,
                                                 default_activation);
      double thr = layer_node.get_double_attribute("threshold", default_threshold);
      double layer_width = layer_node.get_double_attribute("width", default_width);
      double layer_altitude = layer_node.get_double_attribute("altitude", default_altitude);
      // A layer without normalizationMethod inherits the model-level one.
      const NeuralNormalization norm =
          read_enum(layer_node, "normalizationMethod", NN_NORMALIZATION_TABLE, model_normalization);

      Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n, prev_n);
      Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
      Eigen::VectorXd widths = Eigen::VectorXd::Constant(n, layer_width);
      Eigen::VectorXd altitudes = Eigen::VectorXd::Constant(n, layer_altitude);
      Eigen::VectorXd fan_ins = Eigen::VectorXd::Zero(n);
      std::vector<std::string> output_ids;
      std::unordered_map<std::string, size_t> curr_id_to_idx;

      for (size_t i = 0; i < n; i++) {
        const auto& neuron = neuron_nodes[i];
        std::string nid = neuron.get_attribute("id");
        output_ids.push_back(nid);
        curr_id_to_idx[nid] = i;

        if (neuron.exists_attribute("bias")) b[i] = neuron.required_double_attribute("bias");
        widths[i] = neuron.get_double_attribute("width", layer_width);
        altitudes[i] = neuron.get_double_attribute("altitude", layer_altitude);

        for (const auto& con : neuron.get_childs("Con")) {
          const std::string from = con.get_attribute("from");
          const auto it = prev_id_to_idx.find(from);
          if (it != prev_id_to_idx.end()) W(i, it->second) = con.required_double_attribute("weight");
          fan_ins[i] += 1.0;
        }
      }

      const bool radial_basis = act == NeuralActivationType::RADIALBASIS;
      layers.push_back({std::move(W), std::move(b), act, thr, norm, radial_basis, std::move(widths),
                        std::move(altitudes), std::move(fan_ins), std::move(output_ids)});
      prev_id_to_idx = std::move(curr_id_to_idx);
    }
  }

  void parse_neural_outputs(const XmlNode& outputs_node) {
    for (const auto& no : outputs_node.get_childs("NeuralOutput")) {
      NeuralOutput output;
      output.output_neuron = no.get_attribute("outputNeuron");

      XmlNode df = no.get_child("DerivedField");
      if (df.exists_child("NormDiscrete")) {
        output.target_category = df.get_child("NormDiscrete").get_attribute("value");
      } else {
        output.target_category = mining_schema.target.name;
        if (df.exists_child("NormContinuous"))
          for (const auto& linear_norm : df.get_child("NormContinuous").get_childs("LinearNorm")) {
            output.orig.push_back(linear_norm.required_double_attribute("orig"));
            output.norm.push_back(linear_norm.required_double_attribute("norm"));
          }
      }

      neural_outputs.push_back(std::move(output));
    }
  }

  // --- Forward pass ---

  // Applies the NeuralInput DerivedFields and fills x with their values.
  // Returns false, leaving x incomplete, if any of them is missing: the one
  // place the no-prediction rule above is decided.
  bool neural_input_values(const Sample& sample, Eigen::VectorXd& x) const {
    // Build a local mutable sample large enough to hold any new indices
    // that NeuralInput DerivedFields may have registered after base_sample.
    Sample local(indexer->size());
    for (size_t i = 0; i < sample.features.size(); i++) local[i] = sample.features[i];

    x.resize(static_cast<Eigen::Index>(neural_inputs.size()));
    for (size_t i = 0; i < neural_inputs.size(); i++) {
      neural_inputs[i].derived_field.prepare(local);
      const Value& v = local[neural_inputs[i].derived_field.index].value;
      if (v.missing) return false;
      x[static_cast<Eigen::Index>(i)] = v.value;
    }
    return true;
  }

  Eigen::VectorXd forward_pass(Eigen::VectorXd x) const {
    // Layer-by-layer forward pass
    for (const auto& layer : layers) {
      Eigen::VectorXd pre(layer.weights.rows());

      if (layer.radial_basis) {
        // RBF neurons: the Con weights are the coordinates of the centre, not
        // a linear combination, and the neuron output is a Gaussian of the
        // distance to it, scaled by altitude^fanIn.
        for (Eigen::Index j = 0; j < layer.weights.rows(); j++) {
          const double squared_distance = (layer.weights.row(j).transpose() - x).squaredNorm();
          const double width = layer.widths[j];
          pre[j] = std::pow(layer.altitudes[j], layer.fan_ins[j]) *
                   std::exp(-squared_distance / (2.0 * width * width));
        }
      } else {
        pre = layer.weights * x + layer.biases;
        for (Eigen::Index j = 0; j < pre.size(); j++)
          pre[j] = apply_activation(pre[j], layer.activation, layer.threshold);
      }

      // Layer normalization (softmax/simplemax)
      if (layer.normalization == NeuralNormalization::SOFTMAX) {
        const double max_val = pre.maxCoeff();
        Eigen::VectorXd shifted = (pre.array() - max_val).exp();
        pre = shifted / shifted.sum();
      } else if (layer.normalization == NeuralNormalization::SIMPLEMAX) {
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
      if (it == output_id_to_idx.end()) {
        scores.push_back(0.0);
        continue;
      }

      scores.push_back(denormalize(no, out[static_cast<Eigen::Index>(it->second)]));
    }
    return scores;
  }

  // Piecewise-linear inverse of the NeuralOutput NormContinuous: maps a value
  // from network space back to target scale, extrapolating from the nearest
  // segment outside the mapped range (NormContinuous outliers="asIs").
  static double denormalize(const NeuralOutput& output, const double value) {
    if (output.norm.size() < 2) return value;

    size_t segment = 0;
    for (size_t i = 0; i + 1 < output.norm.size(); i++) {
      segment = i;
      if (value < output.norm[i + 1]) break;
    }

    const double span = output.norm[segment + 1] - output.norm[segment];
    if (span == 0.0) return output.orig[segment];

    return output.orig[segment] +
           (value - output.norm[segment]) / span * (output.orig[segment + 1] - output.orig[segment]);
  }
};

#endif
