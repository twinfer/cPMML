
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_FIELDREF_H
#define CPMML_FIELDREF_H

#include <string>

#include "bind/load_scope.h"
#include "core/value.h"
#include "expression.h"

/**
 * @class FieldRef
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_FieldRef">PMML
 * FieldRef</a>.
 *
 * It is used to simply reference (and rename) another DataField or
 * DerivedField.
 */
class FieldRef : public Expression {
 public:
  bool exist_missingreplacement = false;
  std::string field_name;
  size_t index = std::numeric_limits<size_t>::max();
  Value mapmissing_to;

  FieldRef() = default;

  FieldRef(const std::string& field_name, const std::shared_ptr<Indexer>& indexer, const size_t& output_index,
           const DataType& output_type)
      : Expression(output_index, output_type, indexer),
        exist_missingreplacement(false),
        field_name(field_name),
        index(indexer->get_or_set(field_name)) {}

  FieldRef(const XmlNode& node, const size_t& output_index, const DataType& output_type,
           const std::shared_ptr<Indexer>& indexer)
      : Expression(output_index, output_type, indexer),
        exist_missingreplacement(node.exists_attribute("mapMissingTo")),
        field_name(node.required_attribute("field")),
        index(indexer->get_or_set(field_name)),
        // §3.4: "mapMissingTo (xs:string) parsed as T(field) by the rule
        // above" (K1): a value of the referenced field, not of the enclosing
        // DerivedField's type (B11). Stored as cpmml::bind::slot_constant
        // says: a string field's "5" is the double 5 in a double DerivedField,
        // a double field's "5" is the string "5" that keeps its number 5 in a
        // string one (decided 2026-11-18, §9), and a nested FieldRef's is 5
        // with its text (JPMML 1.7.7 and the base build agree on each).
        mapmissing_to(exist_missingreplacement ? map_missing_constant(node, field_name, *indexer, output_type)
                                               : Value()) {
    inputs.insert(field_name);
  }

  static Value map_missing_constant(const XmlNode& node, const std::string& field, Indexer& indexer,
                                    const DataType& slot) {
    const std::string text = node.required_attribute("mapMissingTo");
    const auto type = cpmml::bind::k1_type(
        text, cpmml::bind::type_id(cpmml::bind::scope_resolve(indexer, node, field).datatype));
    return cpmml::bind::slot_constant(node, "mapMissingTo", text, type, slot);
  }

  inline Value eval(const Sample& sample) const override {
    const Value& value = sample[index].value;
    return value.is_missing() ? mapmissing_to : value;
  }
};

/**
 * @class SimpleFieldRef
 *
 * Class used to simply reference (and rename) another DataField or
 * DerivedField. It differs from FieldRef since it has local scope rather than
 * global scope.
 */
class SimpleFieldRef {
  // As fieldref but the derived field is supposed to have local visibility,
  // so it's not implementing Expression

 public:
  bool exist_missingreplacement;
  std::string field_name;
  size_t index;
  Value mapmissing_to;

  SimpleFieldRef() : exist_missingreplacement(false), index(std::numeric_limits<size_t>::max()) {}

  SimpleFieldRef(const XmlNode& node, const std::shared_ptr<Indexer>& indexer)
      : SimpleFieldRef(node, cpmml::bind::scope_resolve(*indexer, node, node.required_attribute("field"))) {}

  // scope.resolve (P2.1): the field's slot and declared type.
  SimpleFieldRef(const XmlNode& node, const cpmml::bind::ResolvedField& field)
      : exist_missingreplacement(node.exists_attribute("mapMissingTo")),
        field_name(node.required_attribute("field")),
        index(field.index),
        mapmissing_to(exist_missingreplacement
                          ? Value::document(node.required_attribute("mapMissingTo"), field.datatype)
                          : Value()) {}

  inline Value value(const Sample& sample) const {
    const Value& value = sample[index].value;
    return value.is_missing() ? mapmissing_to : value;
  }

  static std::vector<SimpleFieldRef> to_simplefields(const std::vector<XmlNode>& nodes,
                                                     const std::shared_ptr<Indexer>& indexer) {
    std::vector<SimpleFieldRef> result;
    for (const auto& node : nodes) result.push_back(SimpleFieldRef(node, indexer));

    return result;
  }
};

#endif
