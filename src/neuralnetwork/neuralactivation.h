
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_NEURALACTIVATION_H
#define CPMML_NEURALACTIVATION_H

#include <cmath>
#include <string>

#include "core/enum_parser.h"
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

// ACTIVATION-FUNCTION, spelled as the spec does: Gauss and Elliott are
// capitalised (JPMML 1.7.7 accepts "Gauss" and refuses "gauss"). Anything
// unrecognised used to become identity.
inline constexpr EnumEntry<NeuralActivationType> ACTIVATION_FUNCTION_TABLE[] = {
    {"threshold", NeuralActivationType::THRESHOLD},
    {"logistic", NeuralActivationType::LOGISTIC},
    {"tanh", NeuralActivationType::TANH},
    {"identity", NeuralActivationType::IDENTITY},
    {"exponential", NeuralActivationType::EXPONENTIAL},
    {"reciprocal", NeuralActivationType::RECIPROCAL},
    {"square", NeuralActivationType::SQUARE},
    {"Gauss", NeuralActivationType::GAUSS},
    {"sine", NeuralActivationType::SINE},
    {"cosine", NeuralActivationType::COSINE},
    {"Elliott", NeuralActivationType::ELLIOTT},
    {"arctan", NeuralActivationType::ARCTAN},
    {"rectifier", NeuralActivationType::RECTIFIER},
    {"radialBasis", NeuralActivationType::RADIALBASIS},
};

// NN-NORMALIZATION-METHOD.
enum class NeuralNormalization { NONE, SIMPLEMAX, SOFTMAX };

inline constexpr EnumEntry<NeuralNormalization> NN_NORMALIZATION_TABLE[] = {
    {"none", NeuralNormalization::NONE},
    {"simplemax", NeuralNormalization::SIMPLEMAX},
    {"softmax", NeuralNormalization::SOFTMAX},
};

/**
 * Element-wise activations, applied to a neuron's weighted sum.
 *
 * radialBasis is not one of them: an RBF neuron reads its Con weights as the
 * coordinates of a centre rather than as a linear combination, so it is
 * evaluated by the layer itself (see NeuralNetworkModel::forward_pass).
 */
inline double apply_activation(double x, NeuralActivationType type, double threshold = 0.0) {
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
      return x;  // handled by the layer
  }
  return x;
}

#endif
