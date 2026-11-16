
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SVMKERNEL_H
#define CPMML_SVMKERNEL_H

#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <utility>

#include "core/xmlnode.h"
#include "utils/utils.h"

/**
 * @defgroup SvmKernel
 *
 * Kernel functions for PMML SupportVectorMachineModel.
 * See http://dmg.org/pmml/v4-4/SupportVectorMachine.html
 */

enum class SvmKernelType { LINEAR, POLYNOMIAL, RBF, SIGMOID };

struct SvmKernel {
  SvmKernelType type = SvmKernelType::LINEAR;
  double gamma = 1.0;
  double coef0 = 1.0;
  double degree = 1.0;

  SvmKernel() = default;

  /**
   * Reads the KERNEL-TYPE choice of a SupportVectorMachineModel. The element
   * names are the PMML 4.4 ones (LinearKernelType, PolynomialKernelType,
   * RadialBasisKernelType, SigmoidKernelType) and the attribute defaults are
   * the schema's: gamma = 1, coef0 = 1, degree = 1. A model with no kernel, or
   * with more than one, is rejected: scoring it with a guessed kernel would
   * return plausible numbers that mean nothing.
   */
  static SvmKernel from_node(const XmlNode& node) {
    static const std::pair<const char*, SvmKernelType> kinds[] = {
        {"LinearKernelType", SvmKernelType::LINEAR},
        {"PolynomialKernelType", SvmKernelType::POLYNOMIAL},
        {"RadialBasisKernelType", SvmKernelType::RBF},
        {"SigmoidKernelType", SvmKernelType::SIGMOID},
    };

    SvmKernel k;
    const char* found = nullptr;
    for (const auto& [element, type] : kinds) {
      if (!node.exists_child(element)) continue;
      if (found)
        throw cpmml::ParsingException(std::string("SupportVectorMachineModel: more than one kernel (") + found +
                                      ", " + element + ")");
      found = element;
      k.type = type;
    }
    if (!found)
      throw cpmml::ParsingException(
          "SupportVectorMachineModel: missing or unsupported kernel, expected one of LinearKernelType, "
          "PolynomialKernelType, RadialBasisKernelType, SigmoidKernelType");

    const XmlNode n = node.get_child(found);
    k.gamma = attribute_or(n, "gamma", 1.0);
    k.coef0 = attribute_or(n, "coef0", 1.0);
    k.degree = attribute_or(n, "degree", 1.0);
    return k;
  }

  inline double compute(const Eigen::VectorXd& x, const Eigen::VectorXd& sv) const {
    switch (type) {
      case SvmKernelType::LINEAR:
        return x.dot(sv);
      case SvmKernelType::POLYNOMIAL:
        return std::pow(gamma * x.dot(sv) + coef0, degree);
      case SvmKernelType::RBF:
        return std::exp(-gamma * (x - sv).squaredNorm());
      case SvmKernelType::SIGMOID:
        return std::tanh(gamma * x.dot(sv) + coef0);
    }
    return 0.0;
  }

 private:
  // Absent → the schema default; present but not a number → load error (to_double throws).
  static double attribute_or(const XmlNode& n, const char* name, double default_value) {
    return n.exists_attribute(name) ? to_double(n.get_attribute(name)) : default_value;
  }
};

#endif
