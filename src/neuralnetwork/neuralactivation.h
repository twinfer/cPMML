
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_NEURALACTIVATION_H
#define CPMML_NEURALACTIVATION_H

#include <cmath>
#include <string>

#include "utils/utils.h"

/**
 * @defgroup NeuralActivation
 *
 * Activation functions for PMML NeuralNetwork models.
 * See http://dmg.org/pmml/v4-4/NeuralNetwork.html#xsdType_ACTIVATION-FUNCTION
 */

enum class NeuralActivationType {
  THRESHOLD,
  LOGISTIC,
  TANH,
  IDENTITY,
  EXPONENTIAL,
  RECIPROCAL,
  SQUARE,
  GAUSS,
  SINE,
  COSINE,
  ELLIOTT,
  ARCTAN,
  RECTIFIER,
  RADIALBASIS
};

inline NeuralActivationType parse_activation(const std::string& s) {
  const std::string lower = to_lower(s);
  if (lower == "threshold") return NeuralActivationType::THRESHOLD;
  if (lower == "logistic") return NeuralActivationType::LOGISTIC;
  if (lower == "tanh") return NeuralActivationType::TANH;
  if (lower == "exponential") return NeuralActivationType::EXPONENTIAL;
  if (lower == "reciprocal") return NeuralActivationType::RECIPROCAL;
  if (lower == "square") return NeuralActivationType::SQUARE;
  if (lower == "gauss") return NeuralActivationType::GAUSS;
  if (lower == "sine") return NeuralActivationType::SINE;
  if (lower == "cosine") return NeuralActivationType::COSINE;
  if (lower == "elliott") return NeuralActivationType::ELLIOTT;
  if (lower == "arctan") return NeuralActivationType::ARCTAN;
  if (lower == "rectifier") return NeuralActivationType::RECTIFIER;
  if (lower == "radialbasis") return NeuralActivationType::RADIALBASIS;
  return NeuralActivationType::IDENTITY;
}

inline double apply_activation(double x, NeuralActivationType type, double threshold = 0.0, double width = 1.0) {
  switch (type) {
    case NeuralActivationType::THRESHOLD:
      return x > threshold ? 1.0 : 0.0;
    case NeuralActivationType::LOGISTIC:
      return 1.0 / (1.0 + std::exp(-x));
    case NeuralActivationType::TANH:
      return std::tanh(x);
    case NeuralActivationType::IDENTITY:
      return x;
    case NeuralActivationType::EXPONENTIAL:
      return std::exp(x);
    case NeuralActivationType::RECIPROCAL:
      return x != 0.0 ? 1.0 / x : 0.0;
    case NeuralActivationType::SQUARE:
      return x * x;
    case NeuralActivationType::GAUSS:
      return std::exp(-(x * x));
    case NeuralActivationType::SINE:
      return std::sin(x);
    case NeuralActivationType::COSINE:
      return std::cos(x);
    case NeuralActivationType::ELLIOTT:
      return x / (1.0 + std::abs(x));
    case NeuralActivationType::ARCTAN:
      return std::atan(x);
    case NeuralActivationType::RECTIFIER:
      return std::max(0.0, x);
    case NeuralActivationType::RADIALBASIS:
      return std::exp(-(x * x) / (2.0 * width * width));
  }
  return x;
}

#endif
