
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
  explicit GaussianProcessEvaluator(const XmlNode &node)
      : InternalEvaluator(node),
        gp(node.get_child("GaussianProcessModel"), data_dictionary, transformation_dictionary, indexer) {}

  GaussianProcessModel gp;

  inline bool validate(const std::unordered_map<std::string, std::string> &sample) override {
    return gp.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string> &sample) const override {
    return gp.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string> &sample) const override {
    return gp.predict(sample);
  }

  inline std::string get_target_name() const override { return gp.target_field.name; }
};

#endif
