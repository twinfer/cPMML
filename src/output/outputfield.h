
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_OUTPUTFIELD_H
#define CPMML_OUTPUTFIELD_H

#include "bind/load_scope.h"
#include "core/xmlnode.h"
#include "expression/expression.h"
#include "fn/cast.h"
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
  // Its dataType; INTEGER for dataType="integer" (P2.4), which a selectAll
  // collection writes as an integer (MultipleModelMethod::collection_entry).
  DataType datatype;
  size_t index;
  std::shared_ptr<OutputExpression> expression;
  // §3.4 (design commit P4.7, B24): "predictedValue -> Cast(dataType,
  // prediction); probability, residual, ... double; transformedValue ->
  // Cast(dataType, expression)". Output.html: "The required attribute
  // dataType of an OutputField element specifies the data type for the
  // output column". The cast of a number or boolean field, from the static
  // type of what the feature gives (set_cast) to the declared dataType;
  // Op::None for an OutputField without dataType, whose type is inferred
  // from that value (nothing is declared, so nothing is cast), and for a
  // string field, whose Cast<string> is add_output's Java text (P2.4).
  cpmml::fn::DeclaredCast cast;
  // Where a number or boolean field reads its value from: eval_double (a
  // number), the expression's Value (a transformedValue over a string
  // expression: its text is parsed, §2.6), or eval_str (entityId: the id's
  // text is parsed).
  enum class Source : unsigned char { Number, Value, Text };
  Source source = Source::Number;

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
        // The target's type, as the engine carried it before P2.4: an
        // integer target was a double then, and stays one here, so that a
        // selectAll collection writes such a field as a double (7.0), as
        // before. (An untyped Discretize or MapValues in it finds no enclosing
        // dataType, D21, and bind::expression_type leaves it unknown: its
        // values are computed as doubles, as before.)
        datatype = indexer->get_type(model_target);
        if (datatype == DataType::DataTypeValue::INTEGER) datatype = DataType::DataTypeValue::DOUBLE;
      } else {
        throw cpmml::ParsingException("Impossible to determine datatype for output: " + name);
      }
    } else {
      datatype = DataType::read(node);
    }

    index = indexer->get_or_set(name, datatype).first;
    expression = OutputExpressionBuilder::build(node, index, datatype, indexer, model_target);
    if (node.exists_attribute("dataType") && datatype.value != DataType::DataTypeValue::STRING)
      set_cast(node, indexer, model_target);
  }

  // The static type of what the feature gives and the cast from it (§3.4):
  //   transformedValue  the expression's (bind::expression_type), the cast of
  //                     §2.6, a string's text read strictly (Source::Value);
  //   entityId          a string, the id: its text read strictly
  //                     (Source::Text); it used to give no value silently
  //                     when the id was not a number (the catch of
  //                     ParsingException in EntityId::eval_double);
  //   predictedValue    the target's type: the prediction's number, as
  //                     PredictedValue::eval_double gives it, checked to be a
  //                     value of the dataType (a string target's class is
  //                     the number of its text, as before);
  //   any other         a double (probability, residual, confidence, ...;
  //                     the number of the text of a predictedDisplayValue,
  //                     reasonCode, ruleValue, as before), checked the same.
  void set_cast(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const std::string& model_target) {
    using cpmml::bind::TypeId;
    const TypeId to = cpmml::bind::type_id(datatype);
    std::optional<TypeId> from = TypeId::Double;
    switch (expression_type.value) {
      case OutputExpressionType::OutputExpressionTypeValue::TRANSFORMED_VALUE:
        from = cpmml::bind::expression_type(node.get_child_bylist(expression_names));
        if (from == TypeId::String) source = Source::Value;
        break;
      case OutputExpressionType::OutputExpressionTypeValue::ENTITY_ID:
        // An association rule's id (algorithm=...) is AssociationOutput's,
        // a number of its own.
        if (!node.exists_attribute("algorithm")) {
          from = TypeId::String;
          source = Source::Text;
        }
        break;
      case OutputExpressionType::OutputExpressionTypeValue::PREDICTED_VALUE:
        if (indexer->contains(model_target)) {
          const TypeId target = cpmml::bind::type_id(indexer->get_type(model_target));
          if (target != TypeId::String) from = target;
        }
        break;
      default:
        break;
    }
    cast = cpmml::fn::DeclaredCast(from, to);
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
        return Value::string(sv);
      }
      default: {
        double v = 0;
        switch (source) {
          case Source::Number:
            v = expression->eval_double(sample, from);
            if (is_double_min(v)) return Value();
            // Q2 (decided (b)): a narrowing cast that fails is an invalid
            // result for the record; JPMML 1.7.7: TypeCheckException
            // "Expected integer value, got double value" ("... boolean
            // value, got integer value").
            if (cast.op != cpmml::fn::DeclaredCast::Op::None && !cast.apply_number(v)) raise_invalid(cast.failure(v));
            break;
          case Source::Value: {
            Value value = expression->eval(sample);
            if (value.is_missing()) return Value();
            if (!cast.apply(value)) raise_invalid(cast.failure(value));
            v = value.as_double();
            break;
          }
          case Source::Text: {
            const std::string id = expression->eval_str(sample, from);
            if (id.empty()) return Value();  // no entity: no value
            if (!cast.parse_text(id, v)) raise_invalid(cast.failure(std::string_view(id)));
            break;
          }
        }
        into.num_outputs[name] = v;
        return Value(v);
      }
    }
  }

  [[noreturn]] [[gnu::cold]] [[gnu::noinline]] void raise_invalid(const std::string& why) const {
    throw cpmml::InvalidValueException("OutputField \"" + name + "\": " + why);
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
