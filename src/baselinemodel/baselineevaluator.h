
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BASELINEEVALUATOR_H
#define CPMML_BASELINEEVALUATOR_H

#include "baselinemodel.h"
#include "core/internal_evaluator.h"
#include "core/xmlnode.h"

/**
 * @class BaselineEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of BaselineModel.
 */
class BaselineEvaluator : public InternalEvaluator {
 public:
  explicit BaselineEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        baseline(node.get_child("BaselineModel"), data_dictionary, transformation_dictionary, indexer) {}

  BaselineModel baseline;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return baseline.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return baseline.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return baseline.target_field.name; }
  inline std::string output_name() const override { return baseline.output_name(); }
  inline std::string mining_function_name() const override { return baseline.mining_function.to_string(); }
};

#endif
