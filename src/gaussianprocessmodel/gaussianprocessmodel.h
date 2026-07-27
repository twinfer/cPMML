
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_GAUSSIANPROCESSMODEL_H
#define CPMML_GAUSSIANPROCESSMODEL_H

#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/traininginstances.h"
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
 * All four kernels share the shape
 *
 *   k(x,z) = gamma * exp(-1/2 * sum_i g(|x_i - z_i| / lambda_i))
 *
 * and differ only in g:
 *
 *   RadialBasisKernel             g(u) = u^2, one scalar lambda for every
 *                                 dimension (a `lambda` attribute, not a
 *                                 Lambda child like the other three)
 *   ARDSquaredExponentialKernel   g(u) = u^2, per-dimension lambda
 *   AbsoluteExponentialKernel     g(u) = u
 *   GeneralizedExponentialKernel  g(u) = u^degree
 *
 * so GeneralizedExponentialKernel with degree=2 is ARDSquaredExponentialKernel
 * and with degree=1 is AbsoluteExponentialKernel. Attribute defaults are
 * gamma=1, noiseVariance=1, lambda=1, degree=1.
 *
 * At load time K is computed and Cholesky-factored so that:
 *   alpha = (K + noiseVariance*I)^{-1} y_train
 *
 * At inference:
 *   mean = k(x*,X)^T alpha
 *   var  = k(x*,x*) - || L^{-1} k(x*,X) ||^2   (stored in num_outputs["variance"])
 */
class GaussianProcessModel : public InternalModel {
 public:
  enum class KernelType { RBF, ARD, ABS_EXP, GEN_EXP };

  // --- Members ---

  KernelType kernel_type = KernelType::RBF;
  double gamma = 1.0;          // amplitude
  double noise_var = 1.0;      // diagonal noise added to K
  double degree = 1.0;         // exponent (GeneralizedExponentialKernel)
  double scalar_lambda = 1.0;  // length scale (RadialBasisKernel)
  Eigen::VectorXd lambdas;     // per-dimension length scales, one per active field

  std::vector<std::string> feat_names;
  std::vector<size_t> feat_indices;
  size_t n_train = 0;
  size_t n_feat = 0;

  Eigen::MatrixXd X_train;  // n_train × n_feat
  Eigen::VectorXd y_train;  // n_train
  Eigen::VectorXd alpha;    // K^{-1} y_train (precomputed)
  Eigen::MatrixXd L;        // lower Cholesky factor of K (for variance)

  // --- Constructors ---

  GaussianProcessModel() = default;

  GaussianProcessModel(const XmlNode& node, const DataDictionary& data_dictionary,
                       const TransformationDictionary& transformation_dictionary,
                       const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer) {
    parse_kernel(node);
    parse_training(node, indexer);
    compute_alpha();
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    Eigen::VectorXd k = compute_k_star(sample);
    double mean = k.dot(alpha);
    Eigen::VectorXd v = L.triangularView<Eigen::Lower>().solve(k);
    double var = std::max(kernel_self() - v.squaredNorm(), 0.0);
    auto sc = std::make_unique<InternalScore>(mean);
    sc->num_outputs["variance"] = var;
    return sc;
  }

  inline std::string predict_raw(const Sample& sample) const override {
    return double_to_string(compute_k_star(sample).dot(alpha));
  }

 private:
  // -----------------------------------------------------------------------
  // Kernel computation
  // -----------------------------------------------------------------------

  double kernel(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const {
    double sum = 0.0;

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n_feat); ++i) {
      const double scaled = std::abs(a[i] - b[i]) / lambdas[i];

      switch (kernel_type) {
        case KernelType::RBF:
        case KernelType::ARD:
          sum += scaled * scaled;
          break;
        case KernelType::ABS_EXP:
          sum += scaled;
          break;
        case KernelType::GEN_EXP:
          sum += std::pow(scaled, degree);
          break;
      }
    }

    return gamma * std::exp(-0.5 * sum);
  }

  // k(x,x) collapses to gamma for every kernel: the scaled distance is zero.
  double kernel_self() const { return gamma; }

  Eigen::VectorXd compute_k_star(const Sample& sample) const {
    Eigen::VectorXd x = extract(sample);
    Eigen::VectorXd k(n_train);
    for (size_t i = 0; i < n_train; ++i)
      k[static_cast<Eigen::Index>(i)] = kernel(x, X_train.row(static_cast<Eigen::Index>(i)));
    return k;
  }

  Eigen::VectorXd extract(const Sample& sample) const {
    Eigen::VectorXd x(static_cast<Eigen::Index>(n_feat));
    for (size_t i = 0; i < n_feat; ++i) {
      const Value& v = sample[feat_indices[i]].value;
      if (v.missing) throw cpmml::MissingValueException("missing value for GaussianProcess input");
      x[static_cast<Eigen::Index>(i)] = v.value;
    }
    return x;
  }

  // -----------------------------------------------------------------------
  // Parsing
  // -----------------------------------------------------------------------

  void parse_kernel(const XmlNode& node) {
    std::string element;

    if (node.exists_child("RadialBasisKernel")) {
      kernel_type = KernelType::RBF;
      element = "RadialBasisKernel";
    } else if (node.exists_child("ARDSquaredExponentialKernel")) {
      kernel_type = KernelType::ARD;
      element = "ARDSquaredExponentialKernel";
    } else if (node.exists_child("AbsoluteExponentialKernel")) {
      kernel_type = KernelType::ABS_EXP;
      element = "AbsoluteExponentialKernel";
    } else if (node.exists_child("GeneralizedExponentialKernel")) {
      kernel_type = KernelType::GEN_EXP;
      element = "GeneralizedExponentialKernel";
    } else {
      throw cpmml::ParsingException("GaussianProcessModel: unsupported or missing kernel");
    }

    const XmlNode k = node.get_child(element);

    // Defaults are the ones the PMML 4.4 schema declares; every kernel carries
    // gamma and noiseVariance, lambda belongs to RadialBasisKernel alone and
    // degree to GeneralizedExponentialKernel alone.
    gamma = k.get_double_attribute("gamma", 1.0);
    noise_var = k.get_double_attribute("noiseVariance", 1.0);
    scalar_lambda = k.get_double_attribute("lambda", 1.0);
    degree = k.get_double_attribute("degree", 1.0);

    if (kernel_type != KernelType::RBF) parse_lambdas(k);
  }

  void parse_lambdas(const XmlNode& k) {
    if (k.exists_child("Lambda")) {
      XmlNode arr = k.get_child("Lambda").get_child("Array");
      std::vector<std::string> parts = parse_array(arr.value());
      lambdas.resize(static_cast<Eigen::Index>(parts.size()));
      for (size_t i = 0; i < parts.size(); ++i) lambdas[static_cast<Eigen::Index>(i)] = to_double(parts[i]);
    }
  }

  void parse_training(const XmlNode& node, const std::shared_ptr<Indexer>& indexer) {
    if (!node.exists_child("TrainingInstances"))
      throw cpmml::ParsingException("GaussianProcessModel: missing TrainingInstances");

    XmlNode ti = node.get_child("TrainingInstances");
    std::string target_name = target_field.name;

    // Build field → column map, separating active features from target
    std::vector<std::string> feat_cols;
    std::string target_col;
    for (const auto& f : ti.get_child("InstanceFields").get_childs("InstanceField")) {
      std::string field = f.get_attribute("field");
      std::string col = instance_field_column(f);
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
      const XmlNode& row = rows[r];
      for (size_t c = 0; c < n_feat; ++c)
        X_train(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
            to_double(row.get_child(feat_cols[c]).value());
      y_train[static_cast<Eigen::Index>(r)] = to_double(row.get_child(target_col).value());
    }

    resolve_lambdas();
  }

  /**
   * Give every active field a length scale, now that n_feat is known.
   *
   * RadialBasisKernel states a single one for all of them; the other three
   * carry a Lambda array, which the schema leaves optional and which defaults
   * to 1 per dimension.
   */
  void resolve_lambdas() {
    const auto n = static_cast<Eigen::Index>(n_feat);

    if (kernel_type == KernelType::RBF) {
      lambdas = Eigen::VectorXd::Constant(n, scalar_lambda);
      return;
    }

    if (lambdas.size() == 0) {
      lambdas = Eigen::VectorXd::Ones(n);
      return;
    }

    if (lambdas.size() != n)
      throw cpmml::ParsingException("GaussianProcessModel: Lambda has " + std::to_string(lambdas.size()) +
                                    " entries but the model has " + std::to_string(n_feat) + " active fields");
  }

  void compute_alpha() {
    Eigen::MatrixXd K(static_cast<Eigen::Index>(n_train), static_cast<Eigen::Index>(n_train));
    for (size_t i = 0; i < n_train; ++i)
      for (size_t j = 0; j < n_train; ++j)
        K(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
            kernel(X_train.row(static_cast<Eigen::Index>(i)), X_train.row(static_cast<Eigen::Index>(j)));

    K += noise_var * Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n_train), static_cast<Eigen::Index>(n_train));

    Eigen::LLT<Eigen::MatrixXd> llt(K);
    if (llt.info() != Eigen::Success)
      throw cpmml::ParsingException("GaussianProcessModel: kernel matrix not positive definite");

    L = llt.matrixL();
    alpha = llt.solve(y_train);
  }
};

#endif
