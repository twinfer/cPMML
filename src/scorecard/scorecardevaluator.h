
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SCORECARDEVALUATOR_H
#define CPMML_SCORECARDEVALUATOR_H

#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "scorecardmodel.h"

/**
 * @class ScorecardEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of ScorecardModel.
 */
class ScorecardEvaluator : public InternalEvaluator {
 public:
  explicit ScorecardEvaluator(const XmlNode& node)
      : InternalEvaluator(node), sc(node.get_child("Scorecard"), data_dictionary, transformation_dictionary, indexer) {}

  ScorecardModel sc;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return sc.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return sc.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return sc.target_field.name; }
  inline std::string output_name() const override { return sc.output_name(); }
  inline std::string mining_function_name() const override { return sc.mining_function.to_string(); }
};

#endif
