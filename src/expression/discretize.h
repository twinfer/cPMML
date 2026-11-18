
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
 * else of the nearest enclosing declared type (§3.4, D21); they were read in
 * the enclosing DerivedField's type even when the element declares its own
 * (B11). They are document constants, so the enclosing declaration's cast
 * (§2.6) is applied to them at load (cpmml::bind::document_constant). The
 * Interval margins are NUMBER (xs:double), whatever the field's type.
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
                   cpmml::bind::declared_type_of(node).value_or(cpmml::bind::enclosing_type(node, output_type))) {}

  Discretize(const XmlNode& node, const size_t& output_index, const DataType& output_type,
             const std::shared_ptr<Indexer>& indexer, const cpmml::bind::TypeId type)
      : Expression(output_index, output_type, indexer),
        raw_node(node),
        exist_missingreplacement(node.exists_attribute("mapMissingTo")),
        mapmissing_to(exist_missingreplacement
                          ? cpmml::bind::document_constant(node, "mapMissingTo", node.required_attribute("mapMissingTo"),
                                                           type, output_type)
                          : Value()),
        field_name(node.required_attribute("field")),
        index(indexer->get_or_set(field_name,
                                  DataType::DataTypeValue::DOUBLE)
                  .first),  // discretize makes sense only on numeric fields
        // pmml-4-4.xsd: Interval@leftMargin and @rightMargin are NUMBER.
        datatype(DataType::DataTypeValue::DOUBLE),
        exist_defaultvalue(node.exists_attribute("defaultValue")),
        defaultValue(exist_defaultvalue
                         ? cpmml::bind::document_constant(node, "defaultValue", node.required_attribute("defaultValue"),
                                                          type, output_type)
                         : Value()) {
    inputs.insert(field_name);
    for (const auto& child : raw_node.get_childs("DiscretizeBin")) {
      discretizebin_values.push_back(cpmml::bind::document_constant(
          child, "binValue", child.required_attribute("binValue"), type, output_type));
      discretizebin_intervals.push_back(IntervalBuilder::build(child.get_child("Interval"), index, datatype));
    }
  }

  inline Value eval(const Sample& sample) const override {
    Value input = sample[index].value;

    if (input.missing) return mapmissing_to;

    for (auto i = 0u; i < discretizebin_intervals.size(); i++)
      if (discretizebin_intervals[i](input)) return discretizebin_values[i];

    return defaultValue;
  }
};

#endif
