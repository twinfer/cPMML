
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_SEGMENT_H
#define CPMML_SEGMENT_H

#include <memory>

#include "core/internal_model.h"
#include "core/predicatebuilder.h"
#include "ensemblemodel.h"

/**
 * @class Segment
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/MultipleModels.html#xsdElement_Segmentation">PMML
 * Segment</a>.
 *
 * Inside of an ensemble of models, this is the wrapper for the single model
 * object. See also EnsembleModel.
 */
class Segment {
 public:
  std::string id;
  double weight = 1;
  Predicate predicate;
  std::shared_ptr<InternalModel> model;

  Segment() = default;

  Segment(const XmlNode& node, const PredicateBuilder& predicate_builder, const std::shared_ptr<InternalModel>& model)
      : id(node.get_attribute("id")),
        weight(node.get_double_attribute("weight", 1)),  // PMML default weight is 1
        predicate(predicate_builder.build_from_parent(node)),
        model(model) {}

  // A segment's result is its model's result after the model's own Targets,
  // exactly as when that model is scored on its own: min/max, rescaleFactor,
  // rescaleConstant and castInteger on a prediction, and priorProbability or
  // defaultValue when the model makes none. So a segment that makes no
  // prediction but has a prior or a default is not empty, and
  // missingPredictionTreatment never sees it. Classes stay raw values, not
  // displayValues, as in JPMML 1.7.7 (see Target::operator()).
  inline std::unique_ptr<InternalScore> score(const Sample& sample) const {
    std::unique_ptr<InternalScore> result = model->score_raw(sample);
    model->target(*result, false);
    return result;
  }

  inline std::string predict(const Sample& sample) const { return model->target(model->predict_raw(sample), false); }
};

#endif
