
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SVMEVALUATOR_H
#define CPMML_SVMEVALUATOR_H

#include <string>

#include "core/datadictionary.h"
#include "core/header.h"
#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "svmmodel.h"

/**
 * @class SvmEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of
 * SupportVectorMachineModel.
 */
class SvmEvaluator : public InternalEvaluator {
 public:
  explicit SvmEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        svm(node.get_child("SupportVectorMachineModel"), data_dictionary, transformation_dictionary, indexer) {}

  SupportVectorMachineModel svm;

  inline bool validate(const std::unordered_map<std::string, std::string>& sample) override {
    return svm.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string>& sample) const override {
    return svm.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string>& sample) const override {
    return svm.predict(sample);
  }

  inline std::string get_target_name() const override { return svm.target_field.name; }
};

#endif
