
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

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return svm.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return svm.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return svm.target_field.name; }
  inline std::string output_name() const override { return svm.output_name(); }
  inline std::string mining_function_name() const override { return svm.mining_function.to_string(); }
};

#endif
