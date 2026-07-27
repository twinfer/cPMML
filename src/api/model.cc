
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

// Model is default-constructible and its default holds no evaluator. Unlike
// Result there is no sensible empty answer here — a model that was never loaded
// cannot score, validate, or name anything — so say so rather than dereference
// null. InvalidValueException is the existing type for "this object cannot
// answer that", and it inherits cpmml::Exception, so callers already catching
// the library's base type keep working.
static InternalEvaluator& require(const std::shared_ptr<InternalEvaluator>& evaluator, const char* operation) {
  if (!evaluator)
    throw cpmml::InvalidValueException(std::string("cannot ") + operation +
                                       ": Model was default-constructed and holds no PMML document");
  return *evaluator;
}

Result Model::evaluate(const Input& arguments) const {
  return Result(require(evaluator, "evaluate").evaluate(arguments));
}

bool Model::validate(const Input& arguments) const {
  return require(evaluator, "validate").validate(arguments);
}

std::string Model::output_name() const { return require(evaluator, "read output_name").output_name(); }

std::string Model::mining_function() const {
  return require(evaluator, "read mining_function").mining_function_name();
}
}  // namespace cpmml
