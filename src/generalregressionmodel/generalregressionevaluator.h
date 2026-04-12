
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_GENERALREGRESSIONEVALUATOR_H
#define CPMML_GENERALREGRESSIONEVALUATOR_H

#include <string>

#include "core/datadictionary.h"
#include "core/header.h"
#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "generalregressionmodel.h"

/**
 * @class GeneralRegressionEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of GeneralRegressionModel.
 */
class GeneralRegressionEvaluator : public InternalEvaluator {
 public:
  explicit GeneralRegressionEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        grm(node.get_child("GeneralRegressionModel"), data_dictionary, transformation_dictionary, indexer) {}

  GeneralRegressionModel grm;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return grm.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return grm.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return grm.target_field.name; }
  inline std::string output_name() const override { return grm.output_name(); }
  inline std::string mining_function_name() const override { return grm.mining_function.to_string(); }
};

#endif
