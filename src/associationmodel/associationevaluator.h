
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_ASSOCIATIONEVALUATOR_H
#define CPMML_ASSOCIATIONEVALUATOR_H

#include "associationmodel.h"
#include "core/model_evaluator.h"
#include "core/xmlnode.h"

/**
 * @class AssociationEvaluator
 *
 * InternalEvaluator wrapper for AssociationModel. Only evaluate() differs from
 * the stock ModelEvaluator: an itemset arrives as a vector<string>, which
 * flatten_input drops, so a collection-valued input takes a separate path.
 */
class AssociationEvaluator : public ModelEvaluator<AssociationModel> {
 public:
  explicit AssociationEvaluator(const XmlNode& node) : ModelEvaluator<AssociationModel>(node, "AssociationModel") {}

  std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    // Check if any value is a vector<string> — if so, use the collection-aware score path.
    bool has_collection = false;
    for (const auto& [k, v] : arguments) {
      if (std::holds_alternative<std::vector<std::string>>(v)) {
        has_collection = true;
        break;
      }
    }
    if (!has_collection) return model.score(flatten_input(arguments));

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
};

#endif  // CPMML_ASSOCIATIONEVALUATOR_H
