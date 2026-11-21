
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MININGSCHEMA_H
#define CPMML_MININGSCHEMA_H

#include <unordered_map>
#include <unordered_set>

#include "datafield.h"
#include "miningfield.h"

/**
 * @class MiningSchema
 *
 * Class Representing  <a href="http://dmg.org/pmml/v4-4/MiningSchema.html">PMML
 * MiningSchema</a>.
 *
 * It is a collection of MiningFields. See MiningField.
 *
 * Notice that the link with the values defined in DataDictionary is performed
 * by deriving MiningFields from DataFields.
 *
 * Through the method validate, the presence of all fields needed by the model
 * is checked.
 */
class MiningSchema {
 public:
  std::unordered_map<std::string, MiningField> raw_miningfields;
  std::vector<MiningField> miningfields;
  MiningField target;
  size_t target_index = std::numeric_limits<size_t>::max();
  // MiningFields naming no DataField: in a Segment, the OutputFields of the
  // modelChain segments before it, or the DerivedFields of the enclosing
  // MiningModel. They carry no MiningField treatment, but they are in the
  // model's scope, so a LocalTransformations DerivedField may read them.
  std::unordered_set<std::string> other_fields;

  MiningSchema() = default;

  MiningSchema(const XmlNode& node, const DataDictionary& data_dictionary)
      : raw_miningfields(MiningField::to_miningfields(node.get_childs("MiningField"), data_dictionary)),
        miningfields(to_values<std::string, MiningField>(raw_miningfields)),
        target(get_target(miningfields)),
        target_index(target.index),
        other_fields(get_other_fields(node, data_dictionary)) {}

  inline static std::unordered_set<std::string> get_other_fields(const XmlNode& node,
                                                                const DataDictionary& data_dictionary) {
    std::unordered_set<std::string> result;
    for (const auto& child : node.get_childs("MiningField")) {
      const std::string name = child.required_attribute("name");
      if (data_dictionary.datafields.find(name) == data_dictionary.datafields.cend()) result.insert(name);
    }
    return result;
  }

  inline static MiningField get_target(const std::vector<MiningField>& miningfields) {
    for (const auto& miningfield : miningfields) {
      if (miningfield.field_usage_type == FieldUsageType::FieldUsageTypeValue::TARGET) return miningfield;
    }

    return MiningField();
  }

  inline const MiningField& operator[](const std::string& feature_name) const {
    return raw_miningfields.at(feature_name);
  }

  inline const MiningField& operator[](const size_t& feature_index) const { return miningfields[feature_index]; }

  inline bool contains(const std::string& feature_name) const {
    return raw_miningfields.find(feature_name) != raw_miningfields.cend();
  }

  // Whether the MiningSchema lists the field at all, DataField or not.
  inline bool references(const std::string& feature_name) const {
    return contains(feature_name) || other_fields.find(feature_name) != other_fields.cend();
  }

  inline const MiningField& at(const std::string& feature_name) const { return raw_miningfields.at(feature_name); }

  void prepare(Sample& sample, const std::unordered_map<std::string, std::string>& input) const {
#ifdef DEBUG
    std::cout << "BEFORE MINING SCHEMA PREPARATION: " << sample << std::endl;
#endif
    for (const auto& miningfield : miningfields) {
      if (miningfield.field_usage_type.value != FieldUsageType::FieldUsageTypeValue::TARGET &&
          miningfield.field_usage_type.value != FieldUsageType::FieldUsageTypeValue::GROUP) {
        const auto found = input.find(miningfield.name);
        if (found == input.cend()) {  // field is absent
          sample.change_value(miningfield.index, miningfield.handle_missing());
          continue;
        }
        // A categorical string input without its trailing blanks (P6.6).
        const std::string_view raw = miningfield.input_text(found->second);
        // A token the DataDictionary declared as a missing marker is missing
        // input, not data: it takes the same path as an absent field.
        if (raw.empty() || miningfield.is_missing_marker(raw)) {
          sample.change_value(miningfield.index, miningfield.handle_missing());
          continue;
        }
        // Text that does not read as the field's dataType is an invalid value.
        // It used to be caught here as a conversion failure and treated as
        // missing -- and "3abc" read as 3 and was not caught at all. The same
        // catch turned returnInvalid's exception into a missing value.
        Value& slot = sample[miningfield.index].value;
        switch (miningfield.parse_input(raw, slot)) {
          case MiningField::InputStatus::MALFORMED:
            slot = miningfield.handle_malformed(raw);
            continue;
          case MiningField::InputStatus::INVALID:
            slot = miningfield.handle_invalid(slot);
            continue;
          case MiningField::InputStatus::VALID:
            break;
        }
        // A value equal to a marker read in the field's type is missing too
        // (P6.4, B6: on a double field "-999.0" is the marker -999), before
        // validity: DataDictionary.html "any data that is outside the
        // intervals will be considered invalid (except for discrete Values
        // whose property attribute identify them as missing)".
        if (miningfield.is_missing_value(slot)) {
          sample.change_value(miningfield.index, miningfield.handle_missing());
          continue;
        }
        if (miningfield.hasInvalidTreatment || miningfield.hasOutlierTreatment) {
          Value tmp_value = sample.operator[](miningfield.index).value;
          if (miningfield.hasInvalidTreatment)
            if (miningfield.is_invalid(tmp_value))
              sample.change_value(miningfield.index, miningfield.handle_invalid(tmp_value));

          if (miningfield.hasOutlierTreatment)
            if (miningfield.is_outlier(tmp_value))
              sample.change_value(miningfield.index, miningfield.handle_outlier(tmp_value));
        }
      }
    }
#ifdef DEBUG
    std::cout << "AFTER MINING SCHEMA PREPARATION: " << sample << std::endl;
#endif
  }

  bool validate(const Sample& sample) const {
    for (const auto& mining_field : miningfields) {
      if (mining_field.field_usage_type.value == FieldUsageType::FieldUsageTypeValue::TARGET ||
          mining_field.field_usage_type.value == FieldUsageType::FieldUsageTypeValue::GROUP) continue;
      if (!mining_field.passes_validation(sample)) return false;
    }

    return true;
  }

  // The first MiningField whose value fails its DataField's valid values or
  // Intervals, or nullptr: the field the error of an invalid record names
  // (B45). Called only once validate() has failed, so the per-record path
  // stays validate()'s.
  const MiningField* first_invalid(const Sample& sample) const {
    for (const auto& mining_field : miningfields) {
      if (mining_field.field_usage_type.value == FieldUsageType::FieldUsageTypeValue::TARGET ||
          mining_field.field_usage_type.value == FieldUsageType::FieldUsageTypeValue::GROUP) continue;
      if (!mining_field.passes_validation(sample)) return &mining_field;
    }

    return nullptr;
  }
};

#endif
