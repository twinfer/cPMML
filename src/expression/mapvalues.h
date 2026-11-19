
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MAPVALUES_H
#define CPMML_MAPVALUES_H

#include <set>

#include "bind/load_scope.h"
#include "expression.h"
#include "treetable/treetable.h"

/**
 * @class MapValues
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_MapValues">PMML
 * MapValues</a>.
 *
 * Through the use of a table (implemented with TreeTable) it maps discrete
 * values to other discrete values.
 *
 * Document text (§3.4, design commit P2.2):
 *   - a key cell is a value of its FieldColumnPair's field, read by the K1
 *     rule in that field's type;
 *   - output cells, mapMissingTo and defaultValue are of the MapValues
 *     dataType, else of the nearest enclosing declared type (D21: the type
 *     that applies to the element's position, cpmml::bind::untyped_type);
 *     they were read in the enclosing DerivedField's type even when the
 *     element declares its own (B11). They are document constants, so where
 *     the MapValues gives the field's value the enclosing declaration's cast
 *     (§2.6) is applied to them at load; as an argument of an Apply they keep
 *     the MapValues type (cpmml::bind::element_constant, design commit
 *     P2.8b1: lessThan over a MapValues dataType="integer" in a boolean
 *     OutputField cast its cells to boolean and failed the load);
 *   - Transformations.html: "it is an error if the table entries used for
 *     matching are not unique" (B42: two rows whose keys are equal once typed
 *     were accepted, one of them silently winning): E-KEY-DUPLICATE;
 *   - a NaN key "can match nothing" (§2.7, D26): E-KEY-NAN. Both are load
 *     errors. JPMML 1.7.7 loads a NaN key (the row never matches); it
 *     refuses the duplicates, a 0 and -0 pair included ("Element InlineTable
 *     is not valid").
 */
class MapValues : public Expression {
 public:
  XmlNode raw_node;
  bool exist_missingreplacement = false;
  Value mapmissing_to;
  bool exist_defaultvalue = false;
  Value defaultValue;
  std::string out;
  std::unordered_map<std::string, size_t> index;
  std::unordered_map<std::string, DataType> datatype;
  std::vector<std::string> columns;
  TreeTable<Value, Value, Value::ValueHash> tree_table;
  std::unordered_map<std::string, std::string> fieldcolumn_pairs;
  std::vector<size_t> raw_columns;

  MapValues() = default;

  MapValues(const XmlNode& node, const size_t& output_index, const DataType& output_type,
            const std::shared_ptr<Indexer>& indexer)
      : MapValues(node, output_index, output_type, indexer,
                  cpmml::bind::declared_type_of(node).value_or(cpmml::bind::untyped_type(node, output_type))) {}

  MapValues(const XmlNode& node, const size_t& output_index, const DataType& output_type,
            const std::shared_ptr<Indexer>& indexer, const cpmml::bind::TypeId type)
      : Expression(output_index, output_type, indexer),
        raw_node(node),
        exist_missingreplacement(node.exists_attribute("mapMissingTo")),
        mapmissing_to(exist_missingreplacement
                          ? cpmml::bind::element_constant(node, node, "mapMissingTo",
                                                          node.required_attribute("mapMissingTo"), type, output_type,
                                                          cpmml::bind::slot_reads_values(node, type, output_type))
                          : Value()),
        exist_defaultvalue(node.exists_attribute("defaultValue")),
        defaultValue(exist_defaultvalue ? cpmml::bind::element_constant(
                                              node, node, "defaultValue", node.required_attribute("defaultValue"), type,
                                              output_type, cpmml::bind::slot_reads_values(node, type, output_type))
                                        : Value()),
        out(node.required_attribute("outputColumn")) {
    std::string tmp_column;
    std::string tmp_input;
    // scope.resolve (P2.1): a FieldColumnPair may name a field declared later
    // in its container (a forward reference, accepted with W-FORWARD-REF, Q4),
    // whose type the keys need now (B12).
    std::unordered_map<std::string, DataType> column_types;
    for (const auto& n : node.get_childs("FieldColumnPair")) {
      tmp_input = n.required_attribute("field");
      inputs.insert(tmp_input);
      tmp_column = n.required_attribute("column");
      columns.push_back(tmp_column);
      fieldcolumn_pairs[tmp_column] = tmp_input;
      const auto resolved = cpmml::bind::scope_resolve(*indexer, n, tmp_input);
      column_types[tmp_column] = resolved.datatype;
      raw_columns.push_back(resolved.index);
    }

    // The typed keys seen so far, for E-KEY-DUPLICATE: a key's engine value
    // (a number, or a string's code), a float column's narrowed to float, as
    // its typed value is (two texts of one float are one key).
    std::set<std::vector<double>> seen;
    const bool reads_slot = cpmml::bind::slot_reads_values(node, type, output_type);
    for (const auto& n : raw_node.get_child("InlineTable").get_childs("row")) {
      std::vector<Value> keys;
      std::vector<double> typed;
      for (const auto& column : columns) {
        // Keys must carry the same datatype as the field they are compared
        // against at lookup time: with an inferred type, a numeric-looking
        // category like "1" becomes the number 1 while the sample carries the
        // string-indexed value, and no lookup would ever match.
        const DataType& type = column_types.at(column);
        const std::string text = n.get_child(column).value();
        if (cpmml::bind::is_nan_key(text, type))
          throw cpmml::ParsingException(std::string(cpmml::bind::code::kKeyNaN) + ": MapValues row <" + column +
                                        ">NaN</" + column +
                                        ">: a NaN key matches no input (§2.7, D26: a NaN key is a load error)");
        keys.push_back(Value(text, type));
        typed.push_back(type == DataType::DataTypeValue::FLOAT ? static_cast<double>(static_cast<float>(keys.back().value))
                                                               : keys.back().value);
      }
      if (!seen.insert(typed).second) {
        std::string key;
        for (const auto& column : columns) key += " <" + column + ">" + n.get_child(column).value() + "</" + column + ">";
        throw cpmml::ParsingException(std::string(cpmml::bind::code::kKeyDuplicate) + ": MapValues row" + key +
                                      ": its key equals an earlier row's once typed (Transformations.html: \"it is "
                                      "an error if the table entries used for matching are not unique\", B42)");
      }
      tree_table.add(keys, cpmml::bind::element_constant(node, node, "output cell <" + out + ">",
                                                         n.get_child(out).value(), type, output_type, reads_slot));
    }
  }

  inline Value eval(const Sample& sample) const override {
    std::vector<Value> keys;
    Value tmp;
    bool missing_input = false;

    for (const auto& column : raw_columns) {
      tmp = sample[column].value;
      if (tmp.missing) missing_input = true;

      keys.push_back(tmp);
    }

    if (missing_input) {
      if (exist_missingreplacement) {
        return mapmissing_to;
      } else {
        return Value();
      }
    }

    try {
      return tree_table.get(keys);
    } catch (const std::out_of_range& e) {
      // Unmatched (non-missing) key: defaultValue or missing. mapMissingTo is
      // reserved for missing inputs and must not apply here.
      if (exist_defaultvalue) return defaultValue;
      return Value();
    }
  }
};

#endif
