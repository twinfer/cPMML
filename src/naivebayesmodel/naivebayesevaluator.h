
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
  explicit NaiveBayesEvaluator(const XmlNode &node)
      : InternalEvaluator(node),
        nb(node.get_child("NaiveBayesModel"), data_dictionary, transformation_dictionary, indexer) {}

  NaiveBayesModel nb;

  inline bool validate(const std::unordered_map<std::string, std::string> &sample) override {
    return nb.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string> &sample) const override {
    return nb.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string> &sample) const override {
    return nb.predict(sample);
  }

  inline std::string get_target_name() const override { return nb.target_field.name; }
};

#endif
