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

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    // Check if any value is a vector<string> — if so, use the collection-aware score path.
    bool has_collection = false;
    for (const auto& [k, v] : arguments) {
      if (std::holds_alternative<std::vector<std::string>>(v)) {
        has_collection = true;
        break;
      }
    }
    if (has_collection) {
      // Convert to InternalModel's FieldValue (variant<string, vector<string>>)
      std::unordered_map<std::string, AssociationModel::FieldValue> converted;
      for (const auto& [k, v] : arguments) {
        if (std::holds_alternative<std::string>(v))
          converted[k] = std::get<std::string>(v);
        else if (std::holds_alternative<std::vector<std::string>>(v))
          converted[k] = std::get<std::vector<std::string>>(v);
      }
      return model.score(converted);
    }
    return model.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return model.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return model.target_field.name; }
  inline std::string output_name() const override { return model.output_name(); }
  inline std::string mining_function_name() const override { return model.mining_function.to_string(); }
};

#endif  // CPMML_ASSOCIATIONEVALUATOR_H
