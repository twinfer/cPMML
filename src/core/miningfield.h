
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MININGFIELD_H
#define CPMML_MININGFIELD_H

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "fieldusagetype.h"
#include "invalidvaluetreatmentmethod.h"
#include "missingvaluetreatmentmethod.h"
#include "optype.h"
#include "outliertreatmentmethod.h"
#include "xmlnode.h"

/**
 * @class MiningField
 *
 * Class representing  <a href="http://dmg.org/pmml/v4-4/MiningSchema.html">PMML
 * MiningField</a>.
 *
 * It defines a field used by the model, along with its usage type and the
 * behaviour in case of missing/invalid values. See also: DataDictionary,
 * InvalidValueTreatmentMethod, MissingValueTreatmentMethod,
 * OutlierTreatmentMethos, FieldUsageType.
 */
class MiningField : public DataField {
 public:
  bool empty = true;
  FieldUsageType field_usage_type;
  // No optype member here: it used to shadow DataField::optype and was set
  // unconditionally from the MiningField node, so a MiningField without the
  // optional attribute reported UNDEFINED and hid the DataDictionary's answer.
  // The spec has MiningField/@optype *override* the DataField's, not replace
  // it, so the ctor overwrites the inherited one only when the attribute is
  // there.
  double importance = double_min();
  bool hasOutlierTreatment = false;
  OutlierTreatmentMethod outliertreatmentmethod;
  bool hasLowValue = false;
  Value low_value;
  bool hasHighValue = false;
  Value high_value;
  bool hasMissingReplacement = false;
  Value missingvalue_replacement;
  MissingValueTreatmentMethod missingvalue_treatmentmethod;
  bool hasInvalidTreatment = false;
  InvalidValueTreatmentMethod invalidvalue_treatmentmethod;
  Value invalidvalue_replacement;  // asValue only

  MiningField() = default;

  MiningField(const std::string& name, const DataType& datatype) : DataField(name, datatype), empty(false) {}

  MiningField(XmlNode node, const DataField& data_field)
      : DataField(data_field),
        empty(false),
        field_usage_type(node.get_attribute("usageType")),
        importance(node.get_double_attribute("importance")),
        hasOutlierTreatment(node.exists_attribute("outliers")),
        outliertreatmentmethod(OutlierTreatmentMethod(node.get_attribute("outliers"))),
        hasLowValue(node.exists_attribute("lowValue")),
        low_value(hasOutlierTreatment && hasLowValue ? Value(node.get_attribute("lowValue"), datatype) : Value()),
        hasHighValue(node.exists_attribute("highValue")),
        high_value(hasOutlierTreatment && hasHighValue ? Value(node.get_attribute("highValue"), datatype) : Value()),
        hasMissingReplacement(node.exists_attribute("missingValueReplacement")),
        missingvalue_replacement(hasMissingReplacement ? Value(node.get_attribute("missingValueReplacement"), datatype)
                                                       : Value()),  //, data_dictionary.datafields[name].datatype)),
        // The attributes are spelled missingValueTreatment and
        // invalidValueTreatment; these read "...Method", a name no MiningField
        // carries, so both always saw an absent attribute and took the default.
        // Neither member is consulted while scoring, so the only thing that
        // changes is that a document spelling one of them wrongly is now
        // refused instead of silently ignored.
        missingvalue_treatmentmethod(node.get_attribute("missingValueTreatment")),
        hasInvalidTreatment(node.exists_attribute("invalidValueTreatment")),
        invalidvalue_treatmentmethod(hasInvalidTreatment
                                         ? InvalidValueTreatmentMethod(node.get_attribute("invalidValueTreatment"))
                                         : InvalidValueTreatmentMethod()) {
    if (node.exists_attribute("optype")) optype = OpType(node.get_attribute("optype"));
    // asValue without the value to use has nothing to replace with. JPMML
    // 1.7.7 loads such a document and fails on the first invalid input; the
    // fault is in the document, so it is reported when the document is read.
    if (invalidvalue_treatmentmethod.value == InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_VALUE) {
      if (!node.exists_attribute("invalidValueReplacement"))
        throw cpmml::ParsingException("MiningField " + name +
                                      ": invalidValueTreatment=\"asValue\" needs invalidValueReplacement");
      invalidvalue_replacement = Value(node.get_attribute("invalidValueReplacement"), datatype);
    }
  };

  static std::unordered_map<std::string, MiningField> to_miningfields(std::vector<XmlNode> nodes,
                                                                      const DataDictionary& data_dictionary) {
    std::unordered_map<std::string, MiningField> result;
    for (auto it = nodes.cbegin(); it != nodes.cend(); it++) {
      if (data_dictionary.datafields.find(it->get_attribute("name")) == data_dictionary.datafields.cend())
        continue;  // this field is an output

      MiningField mining_field(*it, data_dictionary[it->get_attribute("name")]);
      result.insert(std::make_pair(mining_field.name, mining_field));
    }

    return result;
  }

  inline Value handle_missing() const { return hasMissingReplacement ? missingvalue_replacement : Value(); }

  // True when the raw input token is one the DataDictionary declared as
  // standing for a missing value (<Value property="missing">).
  inline bool is_missing_marker(const std::string& raw) const {
    return !missing_markers.empty() && missing_markers.find(raw) != missing_markers.cend();
  }

  inline bool is_outlier(const Value& value) const {
    // if it doesn't have low or high value it must be treated "asIs"
    // since other treatment methods imply low and high value. If treated "asIs"
    // then it's not considered an outlier

    return hasOutlierTreatment && hasLowValue && hasHighValue && (value < low_value || value > high_value);
  }

  inline Value handle_outlier(const Value& value) const {
    switch (outliertreatmentmethod.value) {  // in this two cases it must have low or high value
      case OutlierTreatmentMethod::OutlierTreatmentMethodValue::AS_MISSING_VALUES:
        return handle_missing();
      case OutlierTreatmentMethod::OutlierTreatmentMethodValue::AS_EXTREME_VALUES:
        if (value < low_value) return low_value;
        return high_value;
      default:
        return value;  // "asIs" treatment
    }
  }

  inline bool is_invalid(const Value& value) const { return !constraints(value); }

  inline Value handle_invalid(const Value& value) const {
    switch (invalidvalue_treatmentmethod.value) {
      case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::RETURN_INVALID:
        throw cpmml::InvalidValueException("Invalid value for field: " + name);
      case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_MISSING:
        return handle_missing();
      case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_VALUE:
        return invalidvalue_replacement;
      case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_IS:
        return value;
    }
    return value;  // unreachable: the switch is exhaustive
  }

  // ---- reading an input value ------------------------------------------------
  //
  // An input value is text, read in the lexical space of the field's dataType
  // (lexical.h). Text that is not a value of that type is an *invalid* value,
  // not a missing one and never a silent 0 or false; it goes to
  // invalidValueTreatment, which applies whether or not the attribute is
  // written (its default is returnInvalid). As in JPMML 1.7.7:
  //   returnInvalid  the record is not scored (InvalidValueException);
  //   asMissing      the value is missing: missingValueReplacement, if any;
  //   asValue        the value is invalidValueReplacement;
  //   asIs           there is no value of the field's type to keep, so the
  //                  record is not scored either (JPMML: NumberFormatException).
  // A double or float that reads as NaN is well-formed but invalid (JPMML: the
  // field "cannot accept invalid value NaN"); asIs keeps the NaN.
  enum class InputStatus { VALID, INVALID, MALFORMED };

  // Writes into the caller's Value -- in MiningSchema::prepare, the sample's
  // own slot -- rather than building and moving a temporary per input.
  static inline void set_number(Value& out, const double number) {
    out.value = number;
    out.missing = false;
    out.svalue.clear();
  }

  inline InputStatus parse_input(const std::string& raw, Value& parsed) const {
    switch (datatype.value) {
      case DataType::DataTypeValue::STRING:
        parsed = Value(raw, datatype);
        return InputStatus::VALID;
      case DataType::DataTypeValue::BOOLEAN: {
        bool b = false;
        if (!lexical::parse_boolean(raw, b)) return InputStatus::MALFORMED;
        set_number(parsed, b ? 1.0 : 0.0);
        return InputStatus::VALID;
      }
      case DataType::DataTypeValue::INTEGER:
      case DataType::DataTypeValue::FLOAT:
      case DataType::DataTypeValue::DOUBLE: {
        double d = 0.0;
        const bool ok = integer ? lexical::parse_integer_as_double(raw, d) : lexical::parse_double(raw, d);
        if (!ok) return InputStatus::MALFORMED;
        set_number(parsed, d);
        return std::isnan(d) ? InputStatus::INVALID : InputStatus::VALID;
      }
    }
    return InputStatus::MALFORMED;  // unreachable: the switch is exhaustive
  }

  inline Value handle_malformed(const std::string& raw) const {
    switch (invalidvalue_treatmentmethod.value) {
      case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::RETURN_INVALID:
        throw cpmml::InvalidValueException("Invalid value for field " + name + ": \"" + raw + "\" is not a valid " +
                                           to_lower(integer ? "integer" : datatype.to_string()));
      case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_IS:
        throw cpmml::InvalidValueException("Invalid value for field " + name + ": \"" + raw + "\" is not a valid " +
                                           to_lower(integer ? "integer" : datatype.to_string()) +
                                           " and cannot be used asIs");
      default:
        return handle_invalid(Value());
    }
  }
};

#endif
