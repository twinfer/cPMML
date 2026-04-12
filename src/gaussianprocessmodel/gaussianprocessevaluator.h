
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_GAUSSIANPROCESSEVALUATOR_H
#define CPMML_GAUSSIANPROCESSEVALUATOR_H

#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "gaussianprocessmodel.h"

/**
 * @class GaussianProcessEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of GaussianProcessModel.
 */
class GaussianProcessEvaluator : public InternalEvaluator {
 public:
  explicit GaussianProcessEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        gp(node.get_child("GaussianProcessModel"), data_dictionary, transformation_dictionary, indexer) {}

  GaussianProcessModel gp;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return gp.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return gp.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return gp.target_field.name; }
  inline std::string output_name() const override { return gp.output_name(); }
  inline std::string mining_function_name() const override { return gp.mining_function.to_string(); }
};

#endif
