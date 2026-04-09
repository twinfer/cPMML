
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
  explicit GeneralRegressionEvaluator(const XmlNode &node)
      : InternalEvaluator(node),
        grm(node.get_child("GeneralRegressionModel"), data_dictionary, transformation_dictionary, indexer) {}

  GeneralRegressionModel grm;

  inline bool validate(const std::unordered_map<std::string, std::string> &sample) override {
    return grm.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string> &sample) const override {
    return grm.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string> &sample) const override {
    return grm.predict(sample);
  }

  inline std::string get_target_name() const override { return grm.target_field.name; }
};

#endif
