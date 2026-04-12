
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_KNNEVALUATOR_H
#define CPMML_KNNEVALUATOR_H

#include <string>

#include "core/datadictionary.h"
#include "core/header.h"
#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "knnmodel.h"

/**
 * @class KnnEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of NearestNeighborModel.
 */
class KnnEvaluator : public InternalEvaluator {
 public:
  explicit KnnEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        knn(node.get_child("NearestNeighborModel"), data_dictionary, transformation_dictionary, indexer) {}

  NearestNeighborModel knn;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return knn.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return knn.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return knn.target_field.name; }
  inline std::string output_name() const override { return knn.output_name(); }
  inline std::string mining_function_name() const override { return knn.mining_function.to_string(); }
};

#endif
