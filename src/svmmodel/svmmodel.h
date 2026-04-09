
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SVMMODEL_H
#define CPMML_SVMMODEL_H

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/miningfunction.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"
#include "regressionmodel/regressionscore.h"
#include "svmkernel.h"

/**
 * @class SupportVectorMachineModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/SupportVectorMachine.html">PMML
 * SupportVectorMachineModel</a>.
 *
 * Supports:
 * - All four kernel types (linear, polynomial, RBF, sigmoid)
 * - Both SupportVectors and Coefficients representations
 * - Classification (OneAgainstAll, OneAgainstOne) and regression
 * - Sparse (REAL-SparseArray) and dense (REAL-Array) vector storage
 */
class SupportVectorMachineModel : public InternalModel {
 public:
  // One binary SVM: αᵢ coefficients + support vectors + bias
  struct SvmMachine {
    std::string target_category;
    std::string alternate_category;
    double bias = 0.0;       // absoluteValue from Coefficients element
    double threshold = 0.0;  // for OneAgainstOne voting

    // SupportVectors representation: pairs of (αᵢ, xᵢ)
    std::vector<double> alphas;
    std::vector<Eigen::VectorXd> support_vectors;

    // Coefficients representation (linear only): precomputed w vector
    bool is_weight_vector = false;
    Eigen::VectorXd weight_vector;

    inline double decision(const Eigen::VectorXd& x, const SvmKernel& kernel) const {
      if (is_weight_vector) return weight_vector.dot(x) + bias;
      double result = bias;
      for (size_t i = 0; i < alphas.size(); i++) result += alphas[i] * kernel.compute(x, support_vectors[i]);
      return result;
    }
  };

  // --- Members ---

  SvmKernel kernel;
  bool use_support_vectors;  // false → Coefficients representation
  bool is_ovo;               // false → OneAgainstAll
  double model_threshold;

  // Ordered field indices: position i in the feature vector maps to sample slot
  std::vector<size_t> field_indices;

  std::vector<SvmMachine> machines;
  std::vector<std::string> classes;

  // --- Constructors ---

  SupportVectorMachineModel() = default;

  SupportVectorMachineModel(const XmlNode& node, const DataDictionary& data_dictionary,
                            const TransformationDictionary& transformation_dictionary,
                            const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        kernel(SvmKernel::from_node(node)),
        use_support_vectors(to_lower(node.get_attribute("svmRepresentation")) != "coefficients"),
        is_ovo(to_lower(node.get_attribute("classificationMethod")) == "oneagainstone"),
        model_threshold(node.exists_attribute("threshold") ? to_double(node.get_attribute("threshold")) : 0.0) {
    parse_vector_fields(node, indexer);

    // VectorDictionary is only present in SupportVectors representation
    std::unordered_map<std::string, Eigen::VectorXd> vector_dict;
    if (use_support_vectors && node.exists_child("VectorDictionary"))
      vector_dict = parse_vector_dictionary(node.get_child("VectorDictionary"));

    for (const auto& svm_node : node.get_childs("SupportVectorMachine"))
      machines.push_back(parse_machine(svm_node, vector_dict));

    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION)
      for (const auto& m : machines) classes.push_back(m.target_category);
    else
      classes.push_back(mining_schema.target.name);
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    const Eigen::VectorXd x = sample_to_vector(sample);

    if (mining_function.value != MiningFunction::MiningFunctionType::CLASSIFICATION) {
      const double val = machines[0].decision(x, kernel);
      return std::make_unique<RegressionScore>(std::to_string(val), val, classes, std::vector<double>{val});
    }

    std::vector<double> scores;
    scores.reserve(machines.size());
    for (const auto& m : machines) scores.push_back(m.decision(x, kernel));

    std::string winner = is_ovo ? predict_ovo(x) : predict_ova(scores);
    const double best = *std::max_element(scores.begin(), scores.end());
    return std::make_unique<RegressionScore>(winner, best, classes, scores);
  }

  inline std::string predict_raw(const Sample& sample) const override {
    const Eigen::VectorXd x = sample_to_vector(sample);

    if (mining_function.value != MiningFunction::MiningFunctionType::CLASSIFICATION)
      return std::to_string(machines[0].decision(x, kernel));

    return is_ovo ? predict_ovo(x) : predict_ova_str(x);
  }

 private:
  // --- Feature vector construction ---

  void parse_vector_fields(const XmlNode& model_node, const std::shared_ptr<Indexer>& indexer) {
    if (!model_node.exists_child("VectorDictionary")) return;
    const XmlNode vf = model_node.get_child("VectorDictionary").get_child("VectorFields");
    for (const auto& fr : vf.get_childs("FieldRef"))
      field_indices.push_back(indexer->get_index(fr.get_attribute("field")));
  }

  Eigen::VectorXd sample_to_vector(const Sample& sample) const {
    Eigen::VectorXd x(static_cast<Eigen::Index>(field_indices.size()));
    for (size_t i = 0; i < field_indices.size(); i++)
      x[static_cast<Eigen::Index>(i)] = sample[field_indices[i]].value.value;
    return x;
  }

  // --- VectorDictionary parsing ---

  std::unordered_map<std::string, Eigen::VectorXd> parse_vector_dictionary(const XmlNode& dict_node) const {
    std::unordered_map<std::string, Eigen::VectorXd> result;
    const size_t n_fields = field_indices.size();

    for (const auto& vi : dict_node.get_childs("VectorInstance")) {
      const std::string id = vi.get_attribute("id");
      result[id] = parse_vector_instance(vi, n_fields);
    }
    return result;
  }

  // Parse a VectorInstance element — handles both REAL-SparseArray and REAL-Array
  static Eigen::VectorXd parse_vector_instance(const XmlNode& vi, size_t n) {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));

    if (vi.exists_child("REAL-SparseArray")) {
      const XmlNode sa = vi.get_child("REAL-SparseArray");
      const std::vector<std::string> indices = split(sa.get_child("Indices").value(), " ");
      const std::vector<std::string> entries = split(sa.get_child("REAL-Entries").value(), " ");
      for (size_t k = 0; k < indices.size() && k < entries.size(); k++) {
        // PMML sparse array indices are 1-based
        const size_t idx = static_cast<size_t>(std::stoul(indices[k])) - 1;
        if (idx < n) vec[static_cast<Eigen::Index>(idx)] = to_double(entries[k]);
      }
    } else if (vi.exists_child("REAL-Array")) {
      const std::vector<std::string> entries = split(vi.get_child("REAL-Array").value(), " ");
      for (size_t k = 0; k < entries.size() && k < n; k++) vec[static_cast<Eigen::Index>(k)] = to_double(entries[k]);
    }

    return vec;
  }

  // --- SupportVectorMachine parsing ---

  SvmMachine parse_machine(const XmlNode& node,
                           const std::unordered_map<std::string, Eigen::VectorXd>& vector_dict) const {
    SvmMachine m;
    m.target_category = node.get_attribute("targetCategory");
    m.alternate_category = node.get_attribute("alternateTargetCategory");
    m.threshold = node.exists_attribute("threshold") ? to_double(node.get_attribute("threshold")) : model_threshold;

    const XmlNode coeff_node = node.get_child("Coefficients");
    m.bias = coeff_node.exists_attribute("absoluteValue") ? to_double(coeff_node.get_attribute("absoluteValue")) : 0.0;

    if (use_support_vectors) {
      // Build (αᵢ, xᵢ) pairs from SupportVectors + Coefficients
      const std::vector<XmlNode> sv_refs = node.get_child("SupportVectors").get_childs("SupportVector");
      const std::vector<XmlNode> coeff_refs = coeff_node.get_childs("Coefficient");

      m.alphas.reserve(sv_refs.size());
      m.support_vectors.reserve(sv_refs.size());

      for (size_t i = 0; i < sv_refs.size() && i < coeff_refs.size(); i++) {
        const std::string vid = sv_refs[i].get_attribute("vectorId");
        const auto it = vector_dict.find(vid);
        if (it != vector_dict.end()) {
          m.alphas.push_back(to_double(coeff_refs[i].get_attribute("value")));
          m.support_vectors.push_back(it->second);
        }
      }
    } else {
      // Coefficients representation: one weight per feature
      m.is_weight_vector = true;
      const std::vector<XmlNode> coeff_refs = coeff_node.get_childs("Coefficient");
      m.weight_vector = Eigen::VectorXd(static_cast<Eigen::Index>(coeff_refs.size()));
      for (size_t i = 0; i < coeff_refs.size(); i++)
        m.weight_vector[static_cast<Eigen::Index>(i)] = to_double(coeff_refs[i].get_attribute("value"));
    }

    return m;
  }

  // --- Classification methods ---

  std::string predict_ova(const std::vector<double>& scores) const {
    size_t best = 0;
    for (size_t i = 1; i < scores.size(); i++)
      if (scores[i] > scores[best]) best = i;
    return machines[best].target_category;
  }

  std::string predict_ova_str(const Eigen::VectorXd& x) const {
    double best_score = -std::numeric_limits<double>::max();
    std::string best_class;
    for (const auto& m : machines) {
      const double score = m.decision(x, kernel);
      if (score > best_score) {
        best_score = score;
        best_class = m.target_category;
      }
    }
    return best_class;
  }

  std::string predict_ovo(const Eigen::VectorXd& x) const {
    std::unordered_map<std::string, int> votes;
    for (const auto& cls : classes) votes[cls] = 0;

    for (const auto& m : machines) {
      if (m.decision(x, kernel) > m.threshold)
        votes[m.target_category]++;
      else
        votes[m.alternate_category]++;
    }

    // Class with most votes; tie breaks to first in document order
    int max_votes = -1;
    std::string winner;
    for (const auto& cls : classes) {
      if (votes.at(cls) > max_votes) {
        max_votes = votes.at(cls);
        winner = cls;
      }
    }
    return winner;
  }
};

#endif
