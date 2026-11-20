
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_DERIVEDFIELD_H
#define CPMML_DERIVEDFIELD_H

#include "bind/load_scope.h"
#include "expression/expression.h"
#include "expression/expressionbuilder.h"
#include "fn/cast.h"
#include "sample.h"
#include "xmlnode.h"

/**
 * @class DerivedField
 * Class representing a <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_DerivedField">PMML
 * DerivedField</a>.
 *
 * It defines a feature available to the model, which is obtained by
 * transforming other features. The associated transformation is represented by
 * the class Expression.
 *
 * @see Expression.
 */
class DerivedField {
 public:
  std::string name;
  OpType optype;
  DataType datatype;
  size_t index = std::numeric_limits<size_t>::max();
  std::shared_ptr<Expression> expression;
  // §3.4 "DerivedField dataType=X -> Cast<X>(expression)" (design commit
  // P4.7, B17, B24): the cast from the expression's static type (the
  // binder's, bind::expression_type) to the declared one, fixed at load;
  // fn::DeclaredCast::Op::None when there is nothing to do, so an identity
  // costs a byte compare per record.
  cpmml::fn::DeclaredCast cast;

  DerivedField() = default;

  explicit DerivedField(const XmlNode& node, const std::shared_ptr<Indexer>& indexer)
      // pmml-4-4.xsd: DerivedField@name is optional; it is unnamed only inside
      // a NeuralInput/NeuralOutput, and DerivedField::declared requires it
      // everywhere else.
      : name(node.get_attribute("name", "")),
        optype(OpType::read(node)),
        datatype(DataType::read(node)),
        index(indexer->get_or_set(name, datatype).first),
        expression(ExpressionBuilder::build(node.get_child_bylist(expression_names), index, datatype, indexer)),
        cast(cpmml::bind::expression_type(node.get_child_bylist(expression_names)), cpmml::bind::type_id(datatype)) {}

  // A DerivedField of a TransformationDictionary or LocalTransformations: its
  // name is how it is referenced, so an absent one is a load error.
  static DerivedField declared(const XmlNode& node, const std::shared_ptr<Indexer>& indexer) {
    node.required_attribute("name");
    return DerivedField(node, indexer);
  }

  inline void prepare(Sample& sample) const {
    Value v = expression->eval(sample);
    if (!v.is_missing()) {
      // Q2 (decided (b)): "a narrowing cast that fails at run time is an
      // invalid result for the record (InvalidValueException naming the
      // field)"; JPMML 1.7.7: TypeCheckException "Expected integer value,
      // got double value".
      // A float field's value is a Float (P6.2): the cast makes one from any
      // other number type (fn::DeclaredCast, Op::NumberToFloat); a float
      // expression whose value the engine holds as a double (a FieldRef to a
      // float target or OutputField slot) is rounded here, as §2.6 "double ->
      // float: IEEE round to nearest" (a no-op on a value that is a float).
      if (cast.op != cpmml::fn::DeclaredCast::Op::None && !cast.apply(v)) raise_invalid(v);
      if (datatype.value == DataType::DataTypeValue::FLOAT && !std::holds_alternative<cpmml::types::Float>(v.typed()))
        v.set_float(cpmml::types::narrow_to_float(v.as_double()));
    }
    sample.change_value(index, v);
  }

  [[noreturn]] [[gnu::cold]] [[gnu::noinline]] void raise_invalid(const Value& v) const {
    throw cpmml::InvalidValueException("DerivedField \"" + name + "\": " + cast.failure(v));
  }

  inline static std::unordered_map<std::string, DerivedField> to_derivedfields(
      const std::vector<XmlNode>& nodes, const std::shared_ptr<Indexer>& indexer) {
    std::unordered_map<std::string, DerivedField> result;
    for (const auto& node : nodes)
      result.insert(std::make_pair(node.required_attribute("name"), declared(node, indexer)));

    return result;
  }
};

#endif
