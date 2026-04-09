
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BASELINEEVALUATOR_H
#define CPMML_BASELINEEVALUATOR_H

#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "baselinemodel.h"

/**
 * @class BaselineEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of BaselineModel.
 */
class BaselineEvaluator : public InternalEvaluator {
 public:
  explicit BaselineEvaluator(const XmlNode &node)
      : InternalEvaluator(node),
        baseline(node.get_child("BaselineModel"), data_dictionary, transformation_dictionary, indexer) {}

  BaselineModel baseline;

  inline bool validate(const std::unordered_map<std::string, std::string> &sample) override {
    return baseline.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string> &sample) const override {
    return baseline.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string> &sample) const override {
    return baseline.predict(sample);
  }

  inline std::string get_target_name() const override { return baseline.target_field.name; }
};

#endif
