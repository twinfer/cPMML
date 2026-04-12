
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_REGRESSIONEVALUATOR_H
#define CPMML_REGRESSIONEVALUATOR_H

#include <sstream>
#include <string>

#include "core/datadictionary.h"
#include "core/header.h"
#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "regressionmodel.h"

class RegressionEvaluator : public InternalEvaluator {
 public:
  explicit RegressionEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        regression(node.get_child("RegressionModel"), data_dictionary, transformation_dictionary, indexer) {};

  RegressionModel regression;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return regression.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return regression.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return regression.target_field.name; }
  inline std::string output_name() const override { return regression.output_name(); }
  inline std::string mining_function_name() const override { return regression.mining_function.to_string(); }
};

#endif
