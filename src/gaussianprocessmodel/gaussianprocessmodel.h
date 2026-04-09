
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_GAUSSIANPROCESSMODEL_H
#define CPMML_GAUSSIANPROCESSMODEL_H

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/internal_score.h"
#include "core/transformationdictionary.h"
#include "core/value.h"
#include "core/xmlnode.h"

/**
 * @class GaussianProcessModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/GaussianProcess.html">PMML GaussianProcessModel</a>.
 *
 * Supported kernels: RadialBasisKernel, ARDSquaredExponentialKernel,
 *                    AbsoluteExponentialKernel, GeneralizedExponentialKernel.
 *
 * At load time K is computed and Cholesky-factored so that:
 *   alpha = K^{-1} y_train
 *
 * At inference:
 *   mean = k(x*,X)^T alpha
 *   var  = K(x*,x*) - || L^{-1} k(x*,X) ||^2   (stored in num_outputs["variance"])
 */
class GaussianProcessModel : public InternalModel {
 public:
  enum class KernelType { RBF, ARD, ABS_EXP, GEN_EXP };

  // --- Members ---

  KernelType kernel_type = KernelType::RBF;
  double gamma = 1.0;         // amplitude / bandwidth
  double noise_var = 0.01;    // diagonal noise added to K
  double degree = 2.0;        // exponent (GeneralizedExponentialKernel)
  Eigen::VectorXd lambdas;    // per-dimension rates (ARD / AbsExp / GenExp)

  std::vector<std::string> feat_names;
  std::vector<size_t> feat_indices;
  size_t n_train = 0;
  size_t n_feat = 0;

  Eigen::MatrixXd X_train;   // n_train × n_feat
  Eigen::VectorXd y_train;   // n_train
  Eigen::VectorXd alpha;     // K^{-1} y_train (precomputed)
  Eigen::MatrixXd L;         // lower Cholesky factor of K (for variance)

  // --- Constructors ---

  GaussianProcessModel() = default;

  GaussianProcessModel(const XmlNode &node, const DataDictionary &data_dictionary,
                       const TransformationDictionary &transformation_dictionary,
                       const std::shared_ptr<Indexer> &indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer) {
    parse_kernel(node);
    parse_training(node, indexer);
    compute_alpha();
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample &sample) const override {
    Eigen::VectorXd k = compute_k_star(sample);
    double mean = k.dot(alpha);
    Eigen::VectorXd v = L.triangularView<Eigen::Lower>().solve(k);
    double var = std::max(kernel_self() - v.squaredNorm(), 0.0);
    auto sc = std::make_unique<InternalScore>(mean);
    sc->num_outputs["variance"] = var;
    return sc;
  }

  inline std::string predict_raw(const Sample &sample) const override {
    return std::to_string(compute_k_star(sample).dot(alpha));
  }

 private:
  // -----------------------------------------------------------------------
  // Kernel computation
  // -----------------------------------------------------------------------

  double kernel(const Eigen::VectorXd &a, const Eigen::VectorXd &b) const {
    switch (kernel_type) {
      case KernelType::RBF:
        return std::exp(-gamma * (a - b).squaredNorm());
      case KernelType::ARD: {
        double sum = 0.0;
        for (Eigen::Index i = 0; i < (Eigen::Index)n_feat; ++i) {
          double d = a[i] - b[i];
          sum += lambdas[i] * d * d;
        }
        return gamma * std::exp(-sum);
      }
      case KernelType::ABS_EXP: {
        // AbsoluteExponentialKernel: gamma * exp(-sum_i lambda_i * |x_i - z_i|)
        double sum = 0.0;
        for (Eigen::Index i = 0; i < (Eigen::Index)n_feat; ++i)
          sum += lambdas[i] * std::abs(a[i] - b[i]);
        return gamma * std::exp(-sum);
      }
      case KernelType::GEN_EXP: {
        // GeneralizedExponentialKernel: gamma * exp(-sum_i lambda_i * |x_i - z_i|^degree)
        double sum = 0.0;
        for (Eigen::Index i = 0; i < (Eigen::Index)n_feat; ++i)
          sum += lambdas[i] * std::pow(std::abs(a[i] - b[i]), degree);
        return gamma * std::exp(-sum);
      }
    }
    return 0.0;
  }

  double kernel_self() const {
    return (kernel_type == KernelType::RBF) ? 1.0 : gamma;
  }

  Eigen::VectorXd compute_k_star(const Sample &sample) const {
    Eigen::VectorXd x = extract(sample);
    Eigen::VectorXd k(n_train);
    for (size_t i = 0; i < n_train; ++i)
      k[static_cast<Eigen::Index>(i)] = kernel(x, X_train.row(static_cast<Eigen::Index>(i)));
    return k;
  }

  Eigen::VectorXd extract(const Sample &sample) const {
    Eigen::VectorXd x(static_cast<Eigen::Index>(n_feat));
    for (size_t i = 0; i < n_feat; ++i)
      x[static_cast<Eigen::Index>(i)] = sample[feat_indices[i]].value.value;
    return x;
  }

  // -----------------------------------------------------------------------
  // Parsing
  // -----------------------------------------------------------------------

  void parse_kernel(const XmlNode &node) {
    if (node.exists_child("RadialBasisKernel")) {
      kernel_type = KernelType::RBF;
      XmlNode k = node.get_child("RadialBasisKernel");
      gamma = k.get_double_attribute("gamma");
      noise_var = k.exists_attribute("noiseVariance")
                      ? k.get_double_attribute("noiseVariance")
                      : 0.01;
    } else if (node.exists_child("ARDSquaredExponentialKernel")) {
      kernel_type = KernelType::ARD;
      XmlNode k = node.get_child("ARDSquaredExponentialKernel");
      gamma = k.get_double_attribute("gamma");
      noise_var = k.exists_attribute("noiseVariance")
                      ? k.get_double_attribute("noiseVariance")
                      : 0.01;
      parse_lambdas(k);
    } else if (node.exists_child("AbsoluteExponentialKernel")) {
      kernel_type = KernelType::ABS_EXP;
      XmlNode k = node.get_child("AbsoluteExponentialKernel");
      gamma = k.get_double_attribute("gamma");
      noise_var = k.exists_attribute("noiseVariance")
                      ? k.get_double_attribute("noiseVariance")
                      : 0.01;
      parse_lambdas(k);
    } else if (node.exists_child("GeneralizedExponentialKernel")) {
      kernel_type = KernelType::GEN_EXP;
      XmlNode k = node.get_child("GeneralizedExponentialKernel");
      gamma = k.get_double_attribute("gamma");
      degree = k.exists_attribute("degree") ? k.get_double_attribute("degree") : 2.0;
      noise_var = k.exists_attribute("noiseVariance")
                      ? k.get_double_attribute("noiseVariance")
                      : 0.01;
      parse_lambdas(k);
    } else {
      throw cpmml::ParsingException("GaussianProcessModel: unsupported or missing kernel");
    }
  }

  void parse_lambdas(const XmlNode &k) {
    if (k.exists_child("Lambda")) {
      XmlNode arr = k.get_child("Lambda").get_child("Array");
      std::vector<std::string> parts = split(arr.value(), " ");
      lambdas.resize(static_cast<Eigen::Index>(parts.size()));
      for (size_t i = 0; i < parts.size(); ++i)
        lambdas[static_cast<Eigen::Index>(i)] = to_double(parts[i]);
    }
  }

  void parse_training(const XmlNode &node, const std::shared_ptr<Indexer> &indexer) {
    if (!node.exists_child("TrainingInstances"))
      throw cpmml::ParsingException("GaussianProcessModel: missing TrainingInstances");

    XmlNode ti = node.get_child("TrainingInstances");
    std::string target_name = target_field.name;

    // Build field → column map, separating active features from target
    std::vector<std::string> feat_cols;
    std::string target_col;
    for (const auto &f : ti.get_child("InstanceFields").get_childs("InstanceField")) {
      std::string field = f.get_attribute("field");
      std::string col = f.get_attribute("column");
      if (field == target_name) {
        target_col = col;
      } else {
        feat_names.push_back(field);
        feat_cols.push_back(col);
        feat_indices.push_back(indexer->get_index(field));
      }
    }
    n_feat = feat_names.size();

    // Count rows
    auto rows = ti.get_child("InlineTable").get_childs("row");
    n_train = rows.size();

    X_train.resize(static_cast<Eigen::Index>(n_train), static_cast<Eigen::Index>(n_feat));
    y_train.resize(static_cast<Eigen::Index>(n_train));

    for (size_t r = 0; r < n_train; ++r) {
      const XmlNode &row = rows[r];
      for (size_t c = 0; c < n_feat; ++c)
        X_train(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
            to_double(row.get_child(feat_cols[c]).value());
      y_train[static_cast<Eigen::Index>(r)] = to_double(row.get_child(target_col).value());
    }

    // For kernels with per-dimension lambdas: default to 1 if not yet parsed
    if (kernel_type != KernelType::RBF && lambdas.size() == 0) {
      lambdas = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(n_feat));
    }
  }

  void compute_alpha() {
    Eigen::MatrixXd K(static_cast<Eigen::Index>(n_train), static_cast<Eigen::Index>(n_train));
    for (size_t i = 0; i < n_train; ++i)
      for (size_t j = 0; j < n_train; ++j)
        K(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
            kernel(X_train.row(static_cast<Eigen::Index>(i)),
                   X_train.row(static_cast<Eigen::Index>(j)));

    K += noise_var * Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n_train),
                                                static_cast<Eigen::Index>(n_train));

    Eigen::LLT<Eigen::MatrixXd> llt(K);
    if (llt.info() != Eigen::Success)
      throw cpmml::ParsingException("GaussianProcessModel: kernel matrix not positive definite");

    L = llt.matrixL();
    alpha = llt.solve(y_train);
  }
};

#endif
