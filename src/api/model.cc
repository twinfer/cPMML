
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#include "cPMML.h"
#include "core/internal_evaluator.h"
#include "core/internal_score.h"
#include "core/modelbuilder.h"
#include "utils/csvreader.h"
#include "utils/utils.h"

namespace cpmml {
Model::Model(const std::string& model_filepath) : evaluator(ModelBuilder::build(model_filepath, false)) {}

Model::Model(const std::string& model_filepath, const bool zipped = false)
    : evaluator(ModelBuilder::build(model_filepath, zipped)) {}

Result Model::evaluate(const Input& arguments) const {
  return Result(evaluator->evaluate(arguments));
}

bool Model::validate(const Input& arguments) const {
  return evaluator->validate(arguments);
}

std::string Model::output_name() const { return evaluator->output_name(); }

std::string Model::mining_function() const { return evaluator->mining_function_name(); }
}  // namespace cpmml
