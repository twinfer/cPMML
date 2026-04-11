
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

  inline bool validate(const std::unordered_map<std::string, std::string>& sample) override {
    return sc.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string>& sample) const override {
    return sc.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string>& sample) const override {
    return sc.predict(sample);
  }

  inline std::string get_target_name() const override { return sc.target_field.name; }
  inline std::string output_name() const override { return sc.output_name(); }
};

#endif
