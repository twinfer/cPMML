
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

  inline bool validate(const std::unordered_map<std::string, std::string>& sample) override {
    return ruleset.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string>& sample) const override {
    return ruleset.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string>& sample) const override {
    return ruleset.predict(sample);
  }

  inline std::string get_target_name() const override { return ruleset.target_field.name; }
  inline std::string output_name() const override { return ruleset.output_name(); }
};

#endif
