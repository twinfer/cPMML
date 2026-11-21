
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_DISCRETIZE_H
#define CPMML_DISCRETIZE_H

#include "bind/load_scope.h"
#include "expression.h"

/**
 * @class Discretize
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_Discretize">PMML
 * Discretize</a>.
 *
 * It performs the discretization of numerical input fields by mapping from
 * continuous to discrete values using intervals.
 *
 * Its values (binValue, mapMissingTo, defaultValue) are of its own dataType,
 * else of the nearest enclosing declared type (§3.4, D21: the type that
 * applies to the element's position, cpmml::bind::untyped_type); they were
 * read in the enclosing DerivedField's type even when the element declares
 * its own (B11). They are document constants, so where the Discretize gives
 * the field's value the enclosing declaration's cast (§2.6) is applied to
 * them at load; as an argument of an Apply they keep the Discretize's type
 * (cpmml::bind::element_constant, design commit P2.8b1: lessThan over an
 * untyped Discretize in a boolean DerivedField read binValue "2" as a
 * boolean and failed the load). The Interval margins are NUMBER
 * (xs:double), whatever the field's type.
 */
class Discretize : public Expression {
 public:
  XmlNode raw_node;
  bool exist_missingreplacement = false;
  Value mapmissing_to;
  std::string field_name;
  size_t index = std::numeric_limits<size_t>::max();
  DataType datatype;
  bool exist_defaultvalue = false;
  Value defaultValue;
  std::vector<Value> discretizebin_values;
  std::vector<Predicate> discretizebin_intervals;

  Discretize() = default;

  Discretize(const XmlNode& node, const size_t& output_index, const DataType& output_type,
             const std::shared_ptr<Indexer>& indexer)
      : Discretize(node, output_index, output_type, indexer,
                   cpmml::bind::declared_type_of(node).value_or(cpmml::bind::untyped_type(node, output_type))) {}

  Discretize(const XmlNode& node, const size_t& output_index, const DataType& output_type,
             const std::shared_ptr<Indexer>& indexer, const cpmml::bind::TypeId type)
      : Expression(output_index, output_type, indexer),
        raw_node(node),
        exist_missingreplacement(node.exists_attribute("mapMissingTo")),
        mapmissing_to(exist_missingreplacement
                          ? cpmml::bind::element_constant(node, node, "mapMissingTo",
                                                          node.required_attribute("mapMissingTo"), type, output_type,
                                                          cpmml::bind::slot_reads_values(node, type, output_type))
                          : Value()),
        field_name(node.required_attribute("field")),
        index(indexer->get_or_set(field_name,
                                  DataType::DataTypeValue::DOUBLE)
                  .first),  // discretize makes sense only on numeric fields
        // pmml-4-4.xsd: Interval@leftMargin and @rightMargin are NUMBER.
        datatype(DataType::DataTypeValue::DOUBLE),
        exist_defaultvalue(node.exists_attribute("defaultValue")),
        defaultValue(exist_defaultvalue ? cpmml::bind::element_constant(
                                              node, node, "defaultValue", node.required_attribute("defaultValue"), type,
                                              output_type, cpmml::bind::slot_reads_values(node, type, output_type))
                                        : Value()) {
    inputs.insert(field_name);
    const bool reads_slot = cpmml::bind::slot_reads_values(node, type, output_type);
    for (const auto& child : raw_node.get_childs("DiscretizeBin")) {
      discretizebin_values.push_back(cpmml::bind::element_constant(
          raw_node, child, "binValue", child.required_attribute("binValue"), type, output_type, reads_slot));
      discretizebin_intervals.push_back(bin_interval(child.get_child("Interval"), index));
    }
  }

  // A DiscretizeBin's Interval (design commit P4.7, B40). DataDictionary.html,
  // Interval: "The attributes leftMargin and rightMargin are optional but at
  // least one value must be defined. If a margin is missing, then ±∞ is
  // assumed." Transformations.html, Discretize, the example's bins
  // <Interval closure="closedOpen" leftMargin="0"/> ("right margin is
  // +infinity by default") and openOpen rightMargin="0" ("left margin is
  // -infinity by default"), which "maps values less than 0 to negative and
  // other values to positive ... CASE When "Profit" < 0 Then 'negative' When
  // "Profit" >= 0 Then 'positive' End": an absent margin bounds nothing, so
  // the side has no test, and INF (-INF) is in a bin open to the right
  // (left) whatever its closure. IntervalBuilder used DBL_MAX (DBL_LOWEST)
  // for an absent margin, which put INF in no bin (missing, or the
  // defaultValue; JPMML 1.7.7 gives the open last bin). A margin that is
  // present is compared as written, INF included (pmml-4-4.xsd: NUMBER).
  // pmml-4-4.xsd: Interval@leftMargin and @rightMargin are NUMBER.
  static Predicate bin_interval(const XmlNode& node, const size_t index) {
    using Op = PredicateOpType::PredicateOpTypeValue;
    const Closure closure(node);
    const bool left_closed =
        closure.value == Closure::ClosureValue::CLOSED_CLOSED || closure.value == Closure::ClosureValue::CLOSED_OPEN;
    const bool right_closed =
        closure.value == Closure::ClosureValue::CLOSED_CLOSED || closure.value == Closure::ClosureValue::OPEN_CLOSED;
    std::vector<Predicate> bounds;
    if (const auto left = node.optional_attribute("leftMargin"))
      bounds.push_back(Predicate(index, left_closed ? Op::GREATER_OR_EQUAL : Op::GREATER_THAN,
                                 Value::document(*left, DataType::DataTypeValue::DOUBLE)));
    if (const auto right = node.optional_attribute("rightMargin"))
      bounds.push_back(Predicate(index, right_closed ? Op::LESS_OR_EQUAL : Op::LESS_THAN,
                                 Value::document(*right, DataType::DataTypeValue::DOUBLE)));
    return Predicate(bounds, Op::AND);
  }

  inline Value eval(const Sample& sample) const override {
    const Value& input = sample[index].value;

    if (input.is_missing()) return mapmissing_to;
    // B38 (P6.8): the bins are intervals of numbers; a string field is no
    // number, an invalid record (as Value::as_number; the predicates
    // compared the string's code).
    if (input.is_string()) [[unlikely]]
      cpmml::raise_string_as_number("Discretize", field_name, input);

    for (auto i = 0u; i < discretizebin_intervals.size(); i++)
      if (discretizebin_intervals[i](input)) return discretizebin_values[i];

    return defaultValue;
  }
};

#endif
