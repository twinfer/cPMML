
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_MODEL_EVALUATOR_H
#define CPMML_MODEL_EVALUATOR_H

#include <memory>
#include <string>
#include <vector>

#include "internal_evaluator.h"
#include "internal_score.h"
#include "xmlnode.h"

/**
 * @class ModelEvaluator
 *
 * The InternalEvaluator wrapper every single-model PMML document needs: parse
 * the MODEL-ELEMENT under the document root, then forward each of the six
 * virtuals to it.
 *
 * Thirteen hand-written classes used to say exactly this, differing only in the
 * model type, the element tag and the name of the member -- so a change to the
 * shared contract (a new virtual, a different flattening rule) had to be made
 * thirteen times or silently drift in twelve. ModelT is any InternalModel with
 * the usual score/validate/target_field/output_name(s)/mining_function surface;
 * models that genuinely score differently derive from this and override the one
 * method that differs (see AssociationEvaluator).
 */
template <class ModelT>
class ModelEvaluator : public InternalEvaluator {
 public:
  ModelT model;

  ModelEvaluator(const XmlNode& node, const std::string& model_element)
      : InternalEvaluator(node),
        model(node.get_child(model_element), data_dictionary, transformation_dictionary, indexer) {}

  std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return model.score(flatten_input(arguments));
  }

  bool validate(const Input& arguments) const override { return model.validate(flatten_input(arguments)); }

  std::string get_target_name() const override { return model.target_field.name; }
  std::string output_name() const override { return model.output_name(); }
  std::vector<std::string> output_names() const override { return model.output_names(); }
  std::string mining_function_name() const override { return model.mining_function.to_string(); }
};

#endif
