
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MININGFUNCTION_H
#define CPMML_MININGFUNCTION_H

#include <string>

#include "enum_parser.h"

/**
 * @class MiningFunction
 *
 * Class representing the <a
 * href="http://dmg.org/pmml/v4-4/GeneralStructure.html#xsdType_MINING-FUNCTION">PMML
 * MINING-FUNCTION</a> a model performs.
 */
class MiningFunction {
 public:
  enum class MiningFunctionType {
    ASSOCIATION_RULES,
    //    SEQUENCES,
    CLASSIFICATION,
    REGRESSION,
    CLUSTERING,
    TIMESERIES,
    MIXED,
  };

  static constexpr EnumEntry<MiningFunctionType> TABLE[] = {
      {"associationrules", MiningFunctionType::ASSOCIATION_RULES, "ASSOCIATION_RULES"},
      {"classification", MiningFunctionType::CLASSIFICATION, "CLASSIFICATION"},
      {"regression", MiningFunctionType::REGRESSION, "REGRESSION"},
      {"clustering", MiningFunctionType::CLUSTERING, "CLUSTERING"},
      {"timeseries", MiningFunctionType::TIMESERIES, "TIMESERIES"},
      {"mixed", MiningFunctionType::MIXED, "MIXED"},
  };

  MiningFunctionType value = MiningFunctionType::CLASSIFICATION;

  MiningFunction() = default;

  explicit MiningFunction(const std::string& mining_function) : value(from_string(mining_function)) {}

  // functionName is required on every MODEL-ELEMENT.
  static MiningFunctionType from_string(const std::string& mining_function) {
    return enum_value(mining_function, TABLE, "mining function");
  }

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }
};

#endif
