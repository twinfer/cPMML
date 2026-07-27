
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_NORMALIZATIONMETHODTYPE_H
#define CPMML_NORMALIZATIONMETHODTYPE_H

#include <string>

#include "core/enum_parser.h"

/**
 * @class NormalizationMethodType
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Regression.html#xsdType_REGRESSIONNORMALIZATIONMETHOD">PMML
 * REGRESSIONNORMALIZATIONMETHOD</a>.
 *
 * For instance:
 *      - SIMPLEMAX
 *      - SOFTMAX
 *      - CAUCHIT
 *
 * See also NormalizationMethods.
 */
class NormalizationMethodType {
 public:
  enum class NormalizationMethodTypeValue { NONE, SIMPLEMAX, SOFTMAX, LOGIT, PROBIT, CLOGLOG, EXP, LOGLOG, CAUCHIT };

  static constexpr EnumEntry<NormalizationMethodTypeValue> TABLE[] = {
      {"none", NormalizationMethodTypeValue::NONE, "none"},
      {"simplemax", NormalizationMethodTypeValue::SIMPLEMAX, "simplemax"},
      {"softmax", NormalizationMethodTypeValue::SOFTMAX, "softmax"},
      {"logit", NormalizationMethodTypeValue::LOGIT, "logit"},
      {"probit", NormalizationMethodTypeValue::PROBIT, "probit"},
      {"cloglog", NormalizationMethodTypeValue::CLOGLOG, "cloglog"},
      {"exp", NormalizationMethodTypeValue::EXP, "exp"},
      {"loglog", NormalizationMethodTypeValue::LOGLOG, "loglog"},
      {"cauchit", NormalizationMethodTypeValue::CAUCHIT, "cauchit"},
  };

  NormalizationMethodTypeValue value = NormalizationMethodTypeValue::NONE;

  NormalizationMethodType() = default;

  explicit NormalizationMethodType(const std::string& value) : value(from_string(value)) {}

  inline bool operator==(const NormalizationMethodType& other) const { return value == other.value; };

  inline bool operator!=(const NormalizationMethodType& other) const { return value != other.value; };

  // normalizationMethod defaults to none. A misspelt one used to become NONE
  // too, which is not a degraded answer but a different model: the scores are
  // returned unsquashed and read as probabilities.
  static NormalizationMethodTypeValue from_string(const std::string& normalization_methodtype) {
    return enum_value(normalization_methodtype, TABLE, "normalizationMethod", NormalizationMethodTypeValue::NONE);
  }

  std::string to_string() const { return enum_display(value, TABLE, "none"); }
};

#endif
