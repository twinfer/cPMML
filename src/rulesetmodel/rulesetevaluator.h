
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_RULESETEVALUATOR_H
#define CPMML_RULESETEVALUATOR_H

#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "rulesetmodel.h"

/**
 * @class RuleSetEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of RuleSetModel.
 */
class RuleSetEvaluator : public InternalEvaluator {
 public:
  explicit RuleSetEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        ruleset(node.get_child("RuleSetModel"), data_dictionary, transformation_dictionary, indexer) {}

  RuleSetModel ruleset;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return ruleset.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return ruleset.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return ruleset.target_field.name; }
  inline std::string output_name() const override { return ruleset.output_name(); }
  inline std::string mining_function_name() const override { return ruleset.mining_function.to_string(); }
};

#endif
