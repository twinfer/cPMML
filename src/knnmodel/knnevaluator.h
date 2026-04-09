
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
  explicit KnnEvaluator(const XmlNode &node)
      : InternalEvaluator(node),
        knn(node.get_child("NearestNeighborModel"), data_dictionary, transformation_dictionary, indexer) {}

  NearestNeighborModel knn;

  inline bool validate(const std::unordered_map<std::string, std::string> &sample) override {
    return knn.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string> &sample) const override {
    return knn.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string> &sample) const override {
    return knn.predict(sample);
  }

  inline std::string get_target_name() const override { return knn.target_field.name; }
};

#endif
