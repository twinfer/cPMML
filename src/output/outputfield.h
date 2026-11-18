
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
  // dataType="integer", which DataType carries as a double: a selectAll
  // collection writes such a value as an integer (see MultipleModelMethod).
  bool integer = false;
  size_t index;
  std::shared_ptr<OutputExpression> expression;

  OutputField() : derived(false), index(std::numeric_limits<size_t>::max()) {}

  OutputField(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const std::string& model_target)
      : name(node.required_attribute("name")),
        optype(OpType::read_optional(node)),
        expression_type(OutputExpressionType::read(node)),
        derived(expression_type.value == OutputExpressionType::OutputExpressionTypeValue::TRANSFORMED_VALUE)
  {
    if (!node.exists_attribute("dataType")) {
      // Probability features are always numeric; entityId is always string —
      // both regardless of the target field's datatype.
      const auto feat = expression_type.value;
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
                 feat == OutputExpressionType::OutputExpressionTypeValue::RULE_ID ||
                 feat == OutputExpressionType::OutputExpressionTypeValue::REASON_CODE) {
        datatype = DataType::DataTypeValue::STRING;
      } else if (indexer->contains(model_target)) {
        datatype = indexer->get_type(model_target);
      } else {
        throw cpmml::ParsingException("Impossible to determine datatype for output: " + name);
      }
    } else {
      datatype = DataType::read(node);
      integer = DataType::is_integer(node);
    }

    index = indexer->get_or_set(name, datatype).first;
    expression = OutputExpressionBuilder::build(node, index, datatype, indexer, model_target);
  }

  inline void prepare(Sample& sample) const { sample.change_value_if_missing(index, expression->eval(sample)); }

  inline Value add_output(const Sample& sample, InternalScore& score) const { return add_output(sample, score, score); }

  // Evaluates the field for the prediction `from` and records it in `into`'s
  // outputs. Returns the value, missing when the field has none, so that an
  // OutputField after it can read it (OutputDictionary::add_output).
  inline Value add_output(const Sample& sample, const InternalScore& from, InternalScore& into) const {
    switch (datatype.value) {
      case DataType::DataTypeValue::STRING: {
        std::string sv = expression->eval_str(sample, from);
        // A string OutputField over a number (a probability, a
        // transformedValue over a numeric expression): the number as JPMML
        // writes its type.
        if (sv.empty() && expression->value_kind) {
          const double v = expression->eval_double(sample, from);
          if (!is_double_min(v)) sv = java::text({v, *expression->value_kind});
        }
        if (sv.empty()) return Value();
        into.str_outputs[name] = sv;
        return Value(sv, datatype);
      }
      default: {
        const double v = expression->eval_double(sample, from);
        if (is_double_min(v)) return Value();
        into.num_outputs[name] = v;
        return Value(v, datatype);
      }
    }
  }

  inline bool is_display_value() const {
    return expression_type.value == OutputExpressionType::OutputExpressionTypeValue::PREDICTED_DISPLAY_VALUE;
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
