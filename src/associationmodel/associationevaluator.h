/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_ASSOCIATIONEVALUATOR_H
#define CPMML_ASSOCIATIONEVALUATOR_H

#include "associationmodel.h"
#include "core/internal_evaluator.h"
#include "core/xmlnode.h"

/**
 * @class AssociationEvaluator
 *
 * InternalEvaluator wrapper for AssociationModel.
 */
class AssociationEvaluator : public InternalEvaluator {
 public:
  AssociationModel model;

  explicit AssociationEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        model(node.get_child("AssociationModel"), data_dictionary, transformation_dictionary, indexer) {}

  inline bool validate(const std::unordered_map<std::string, std::string>& sample) override {
    return model.validate(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, std::string>& sample) const override {
    return model.score(sample);
  }

  inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, FieldValue>& sample) const override {
    return model.score(sample);
  }

  inline std::string predict(const std::unordered_map<std::string, std::string>& sample) const override {
    return model.predict(sample);
  }

  inline std::string get_target_name() const override { return model.target_field.name; }
  inline std::string output_name() const override { return model.output_name(); }
};

#endif  // CPMML_ASSOCIATIONEVALUATOR_H
