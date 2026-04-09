
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SVMKERNEL_H
#define CPMML_SVMKERNEL_H

#include <cmath>
#include <string>

#include <Eigen/Dense>

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
  SvmKernelType type = SvmKernelType::RBF;
  double gamma = 1.0;
  double coef0 = 0.0;
  double degree = 3.0;

  SvmKernel() = default;

  static SvmKernel from_node(const XmlNode &node) {
    SvmKernel k;
    if (node.exists_child("LinearKernel")) {
      k.type = SvmKernelType::LINEAR;
    } else if (node.exists_child("PolynomialKernel")) {
      k.type = SvmKernelType::POLYNOMIAL;
      const XmlNode n = node.get_child("PolynomialKernel");
      if (n.exists_attribute("gamma"))  k.gamma  = to_double(n.get_attribute("gamma"));
      if (n.exists_attribute("coef0"))  k.coef0  = to_double(n.get_attribute("coef0"));
      if (n.exists_attribute("degree")) k.degree = to_double(n.get_attribute("degree"));
    } else if (node.exists_child("RadialBasisKernel")) {
      k.type = SvmKernelType::RBF;
      const XmlNode n = node.get_child("RadialBasisKernel");
      if (n.exists_attribute("gamma")) k.gamma = to_double(n.get_attribute("gamma"));
    } else if (node.exists_child("SigmoidKernel")) {
      k.type = SvmKernelType::SIGMOID;
      const XmlNode n = node.get_child("SigmoidKernel");
      if (n.exists_attribute("gamma")) k.gamma = to_double(n.get_attribute("gamma"));
      if (n.exists_attribute("coef0")) k.coef0 = to_double(n.get_attribute("coef0"));
    }
    return k;
  }

  inline double compute(const Eigen::VectorXd &x, const Eigen::VectorXd &sv) const {
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
};

#endif
