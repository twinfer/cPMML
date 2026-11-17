
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
      {"associationRules", MiningFunctionType::ASSOCIATION_RULES, "ASSOCIATION_RULES"},
      {"sequences", MiningFunctionType::MIXED, nullptr, EnumSupport::UNSUPPORTED},
      {"classification", MiningFunctionType::CLASSIFICATION, "CLASSIFICATION"},
      {"regression", MiningFunctionType::REGRESSION, "REGRESSION"},
      {"clustering", MiningFunctionType::CLUSTERING, "CLUSTERING"},
      {"timeSeries", MiningFunctionType::TIMESERIES, "TIMESERIES"},
      {"mixed", MiningFunctionType::MIXED, "MIXED"},
  };

  MiningFunctionType value = MiningFunctionType::CLASSIFICATION;

  MiningFunction() = default;

  explicit MiningFunction(MiningFunctionType value) : value(value) {}

  // functionName is required on every model element.
  static MiningFunction read(const XmlNode& model_node) {
    return MiningFunction(read_enum(model_node, "functionName", TABLE));
  }

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }
};

#endif
