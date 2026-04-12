
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_OUTPUTFIELD_H
#define CPMML_OUTPUTFIELD_H

#include "core/xmlnode.h"
#include "expression/expression.h"
#include "outputexpressionbuilder.h"
#include "outputexpressiontype.h"

/**
 * @class OutputField
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Output.html#xsdElement_OutputField">PMML
 * OutputField</a>.
 *
 * It defines a which output features the model will produce from the raw
 * prediction.
 */
class OutputField {
 public:
  std::string name;
  OpType optype;
  //  std::string target_field; // needed in case of multiple outputs, not
  //  supported by choice
  OutputExpressionType expression_type;
  bool derived;
  DataType datatype;
  size_t index;
  std::shared_ptr<OutputExpression> expression;

  OutputField() : derived(false), index(std::numeric_limits<size_t>::max()) {}

  OutputField(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const std::string& model_target)
      : name(node.get_attribute("name")),
        optype(node.get_attribute("optype")),
        expression_type(node.get_attribute("feature")),
        derived(expression_type.value == OutputExpressionType::OutputExpressionTypeValue::TRANSFORMED_VALUE)
  //      target_field(node.get_attribute("targetField")),
  {
    if (!node.exists_attribute("dataType")) {
      // Probability features are always numeric; entityId is always string —
      // both regardless of the target field's datatype.
      const auto feat = OutputExpressionType(node.get_attribute("feature")).value;
      if (feat == OutputExpressionType::OutputExpressionTypeValue::PROBABILITY ||
          feat == OutputExpressionType::OutputExpressionTypeValue::RESIDUAL ||
          feat == OutputExpressionType::OutputExpressionTypeValue::SUPPORT ||
          feat == OutputExpressionType::OutputExpressionTypeValue::CONFIDENCE ||
          feat == OutputExpressionType::OutputExpressionTypeValue::LIFT) {
        datatype = DataType::DataTypeValue::DOUBLE;
      } else if (feat == OutputExpressionType::OutputExpressionTypeValue::ENTITY_ID ||
                 feat == OutputExpressionType::OutputExpressionTypeValue::RULE_VALUE ||
                 feat == OutputExpressionType::OutputExpressionTypeValue::ANTECEDENT ||
                 feat == OutputExpressionType::OutputExpressionTypeValue::CONSEQUENT ||
                 feat == OutputExpressionType::OutputExpressionTypeValue::RULE ||
                 feat == OutputExpressionType::OutputExpressionTypeValue::RULE_ID) {
        datatype = DataType::DataTypeValue::STRING;
      } else if (indexer->contains(model_target)) {
        datatype = indexer->get_type(model_target);
      } else {
        throw cpmml::ParsingException("Impossible to determine datatype for output: " + name);
      }
    } else
      datatype = node.get_attribute("dataType");

    index = indexer->get_or_set(name, datatype).first;
    expression = OutputExpressionBuilder::build(node, index, datatype, indexer, model_target);
  }

  inline void prepare(Sample& sample) const { sample.change_value_if_missing(index, expression->eval(sample)); }

  inline void add_output(const Sample& sample, InternalScore& score) const {
    switch (datatype.value) {
      case DataType::DataTypeValue::STRING: {
        std::string sv = expression->eval_str(sample, score);
        if (!sv.empty()) score.str_outputs[name] = sv;
        break;
      }
      default: {
        double v = expression->eval_double(sample, score);
        if (!is_double_min(v))
          score.num_outputs[name] = v;
        break;
      }
    }
  }

  inline static std::vector<OutputField> to_outputfields(const std::vector<XmlNode>& nodes,
                                                          std::shared_ptr<Indexer> indexer,
                                                          const std::string& model_target) {
    std::vector<OutputField> result;
    result.reserve(nodes.size());
    for (const auto& node : nodes)
      result.emplace_back(node, indexer, model_target);
    return result;
  }
};

#endif
