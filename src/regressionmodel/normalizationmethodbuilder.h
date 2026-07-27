
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_NORMALIZATIONMETHODBUILDER_H
#define CPMML_NORMALIZATIONMETHODBUILDER_H

#include <functional>

#include "math/normalizationmethods.h"
#include "normalizationmethodtype.h"

/**
 * @class SingleNormalizationMethodBuilder
 *
 * Factory class building the normalization method for a RegressionModel
 * predicting a continuous variable.
 */
class SingleNormalizationMethodBuilder {
 public:
  static std::function<double(const double&)> build(const NormalizationMethodType& normalizationMethodType) {
    switch (normalizationMethodType.value) {
      case NormalizationMethodType::NormalizationMethodTypeValue::NONE:
        return single_none;
      case NormalizationMethodType::NormalizationMethodTypeValue::SOFTMAX:
        return single_softmax;
      case NormalizationMethodType::NormalizationMethodTypeValue::LOGIT:
        return single_logit;
      case NormalizationMethodType::NormalizationMethodTypeValue::PROBIT:
        return single_probit;
      case NormalizationMethodType::NormalizationMethodTypeValue::CLOGLOG:
        return single_cloglog;
      case NormalizationMethodType::NormalizationMethodTypeValue::EXP:
        return single_exp;
      case NormalizationMethodType::NormalizationMethodTypeValue::LOGLOG:
        return single_loglog;
      case NormalizationMethodType::NormalizationMethodTypeValue::CAUCHIT:
        return single_cauchit;
      default:
        throw cpmml::ParsingException("Incorrect normalization method");
    }
  }
};

/**
 * @class MultiNormalizationMethodBuilder
 *
 * Factory class building the normalization method for a RegressionModel
 * predicting a categorical variable.
 */
class MultiNormalizationMethodBuilder {
 public:
  // is_ordinal comes from the target's optype. It only changes the answer for
  // "none", which the spec allows for both kinds and defines differently for
  // each: categorical fills the last category with 1 - sum of the others, while
  // ordinal treats the values as cumulative and differences them. The link
  // methods are prohibited for multinomial categorical targets, so they are
  // unconditionally the cumulative form.
  static std::function<std::vector<double>(const std::vector<double>&)> build(
      const NormalizationMethodType& normalizationMethodType, bool is_ordinal = false) {
    switch (normalizationMethodType.value) {
      case NormalizationMethodType::NormalizationMethodTypeValue::NONE:
        return is_ordinal ? ordinal_none : categorical_none;
      case NormalizationMethodType::NormalizationMethodTypeValue::SIMPLEMAX:
        return categorical_simplemax;
      case NormalizationMethodType::NormalizationMethodTypeValue::SOFTMAX:
        return categorical_softmax;
      case NormalizationMethodType::NormalizationMethodTypeValue::LOGIT:
        return ordinal_logit;
      case NormalizationMethodType::NormalizationMethodTypeValue::PROBIT:
        return ordinal_probit;
      case NormalizationMethodType::NormalizationMethodTypeValue::CLOGLOG:
        return ordinal_cloglog;
      case NormalizationMethodType::NormalizationMethodTypeValue::LOGLOG:
        return ordinal_loglog;
      case NormalizationMethodType::NormalizationMethodTypeValue::CAUCHIT:
        return ordinal_cauchit;
      default:
        throw cpmml::ParsingException("Incorrect normalization method");
    }
  }
};

#endif
