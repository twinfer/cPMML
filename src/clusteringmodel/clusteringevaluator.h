
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_CLUSTERINGEVALUATOR_H
#define CPMML_CLUSTERINGEVALUATOR_H

#include "clusteringmodel.h"
#include "core/internal_evaluator.h"
#include "core/xmlnode.h"

/**
 * @class ClusteringEvaluator
 *
 * Implementation of InternalEvaluator, used as a wrapper of ClusteringModel.
 */
class ClusteringEvaluator : public InternalEvaluator {
 public:
  explicit ClusteringEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        clustering(node.get_child("ClusteringModel"), data_dictionary, transformation_dictionary, indexer) {}

  ClusteringModel clustering;

  inline bool validate(const std::unordered_map<std::string, std::string>& sample) override {
    return clustering.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string>& sample) const override {
    return clustering.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string>& sample) const override {
    return clustering.predict(sample);
  }

  inline std::string get_target_name() const override { return clustering.target_field.name; }
  inline std::string output_name() const override { return clustering.output_name(); }
};

#endif
