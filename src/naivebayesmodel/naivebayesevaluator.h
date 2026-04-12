
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_NAIVEBAYESEVALUATOR_H
#define CPMML_NAIVEBAYESEVALUATOR_H

#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "naivebayesmodel.h"

/**
 * @class NaiveBayesEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of NaiveBayesModel.
 */
class NaiveBayesEvaluator : public InternalEvaluator {
 public:
  explicit NaiveBayesEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        nb(node.get_child("NaiveBayesModel"), data_dictionary, transformation_dictionary, indexer) {}

  NaiveBayesModel nb;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return nb.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return nb.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return nb.target_field.name; }
  inline std::string output_name() const override { return nb.output_name(); }
  inline std::string mining_function_name() const override { return nb.mining_function.to_string(); }
};

#endif
