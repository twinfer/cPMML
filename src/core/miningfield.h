
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MININGFIELD_H
#define CPMML_MININGFIELD_H

#include <cmath>
#include <sstream>
#include <string>
#include <string_view>
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
        field_usage_type(FieldUsageType::read(node)),
        hasOutlierTreatment(node.exists_attribute("outliers")),
        outliertreatmentmethod(OutlierTreatmentMethod::read(node)),
        // pmml-4-4.xsd: lowValue and highValue are NUMBER (xs:double), not
        // values of the field (§3.4, L2): on an integer field a bound 2.5 is
        // the double 2.5, which the clamped value takes (D22, the slot
        // widened).
        hasLowValue(node.exists_attribute("lowValue")),
        low_value(hasOutlierTreatment && hasLowValue ? Value(node.required_double_attribute("lowValue")) : Value()),
        hasHighValue(node.exists_attribute("highValue")),
        high_value(hasOutlierTreatment && hasHighValue ? Value(node.required_double_attribute("highValue")) : Value()),
        // missingValueReplacement and invalidValueReplacement are xs:string
        // values of the field, read by the K1 rule (§3.4): Value(text,
        // datatype) is that rule in the engine's storage (Value::document),
        // so an integer field's "2.5" is the double 2.5 (rule 2; the design
        // widens the slot, W-SLOT-WIDENED, which a double slot already is).
        hasMissingReplacement(node.exists_attribute("missingValueReplacement")),
        missingvalue_replacement(hasMissingReplacement
                                     ? Value::document(node.required_attribute("missingValueReplacement"), datatype)
                                     : Value()),
        // missingValueTreatment is not consulted while scoring (the replacement
        // is missingValueReplacement), except that returnInvalid, which cPMML
        // does not implement, is refused at load.
        missingvalue_treatmentmethod(MissingValueTreatmentMethod::read(node)),
        hasInvalidTreatment(node.exists_attribute("invalidValueTreatment")),
        invalidvalue_treatmentmethod(InvalidValueTreatmentMethod::read(node)) {
    if (node.exists_attribute("optype")) optype = OpType::read(node);
    // asValue without the value to use has nothing to replace with. JPMML
    // 1.7.7 loads such a document and fails on the first invalid input; the
    // fault is in the document, so it is reported when the document is read.
    if (invalidvalue_treatmentmethod.value == InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_VALUE) {
      if (!node.exists_attribute("invalidValueReplacement"))
        throw cpmml::ParsingException("MiningField " + name +
                                      ": invalidValueTreatment=\"asValue\" needs invalidValueReplacement");
      invalidvalue_replacement = Value::document(node.required_attribute("invalidValueReplacement"), datatype);
    }
    // pmml-4-4.xsd: importance is an optional PROB-NUMBER without a default.
    // No scoring rule reads it, so it is checked and not kept (a member
    // would grow every MiningField, which scoring reads per record).
    (void)node.optional_double_attribute("importance");
  };

  static std::unordered_map<std::string, MiningField> to_miningfields(std::vector<XmlNode> nodes,
                                                                      const DataDictionary& data_dictionary) {
    std::unordered_map<std::string, MiningField> result;
    for (auto it = nodes.cbegin(); it != nodes.cend(); it++) {
      const std::string name = it->required_attribute("name");
      if (data_dictionary.datafields.find(name) == data_dictionary.datafields.cend())
        continue;  // this field is an output

      MiningField mining_field(*it, data_dictionary[name]);
      result.insert(std::make_pair(mining_field.name, mining_field));
    }

    return result;
  }

  inline Value handle_missing() const { return hasMissingReplacement ? missingvalue_replacement : Value(); }

  // The text of an input as the field reads it (P6.6). DataDictionary.html:
  // "Trailing blanks in categorical input values are not significant, and
  // categorical values in PMML must not have trailing blanks. Leading blanks
  // are significant." So for a field whose optype (the MiningField's, else
  // the DataField's) is categorical and whose dataType is string, the
  // trailing XML blanks (#x20, #x9, #xA, #xD; BuiltinFunctions.html
  // trimBlanks: "Blanks include tab and newline characters") are dropped
  // before anything reads the text: the missing markers, the field's Values
  // and every predicate, so "a " is the input "a", "NA " the marker "NA" and
  // "   " the same input as "", the empty string (P6.8). Leading blanks are
  // kept, and an ordinal or continuous field keeps its text as given (the
  // sentence names categorical values). Every other dataType reads its text
  // by the XSD lexical rules, whose whiteSpace collapse already drops blanks
  // at either end (lexical.h). JPMML 1.7.7 keeps trailing blanks (D49).
  // A view of `raw`, which must outlive it.
  inline std::string_view input_text(const std::string& raw) const {
    std::string_view text(raw);
    if (text.empty() || !lexical::is_xml_space(text.back()) || datatype.value != DataType::DataTypeValue::STRING ||
        optype.value != OpType::OpTypeValue::CATEGORICAL)
      return text;
    while (!text.empty() && lexical::is_xml_space(text.back())) text.remove_suffix(1);
    return text;
  }

  // True when the raw input token is one the DataDictionary declared as
  // standing for a missing value (<Value property="missing">).
  inline bool is_missing_marker(std::string_view raw) const {
    return !missing_markers.empty() && missing_markers.find(raw) != missing_markers.cend();
  }

  // True when `parsed`, an input read as a value of the field's type (not
  // missing, not a string), is one the DataDictionary declared as standing
  // for a missing value (P6.4, B6): numbers and booleans by the number they
  // are, in the promoted type (Functions.html: "implicit casts are allowed in
  // the direction of less restriction"; the integer -999 against an integer
  // field's marker "-999.0", the double -999, D22), so 0 = −0, and a NaN
  // marker matches any NaN input (XML Schema Part 2: Datatypes Second Edition
  // §3.2.5: "there is only one NaN and only one zero ... for schema purposes
  // NaN = NaN"; decided 2026-11-20 after P6.7, (7a), P6.8). The comparison
  // functions keep NaN ≠ NaN (D9); this is the Value list's membership.
  inline bool is_missing_value(const Value& parsed) const {
    for (const Value& marker : missing_values)
      if (parsed.as_double() == marker.as_double()) return true;
    return nan_missing && is_nan(parsed);
  }

  inline bool is_outlier(const Value& value) const {
    // if it doesn't have low or high value it must be treated "asIs"
    // since other treatment methods imply low and high value. If treated "asIs"
    // then it's not considered an outlier

    // lowValue, highValue: pmml-4-4.xsd NUMBER, compared as the doubles they
    // are (a number field's value widened to double: exact).
    return hasOutlierTreatment && hasLowValue && hasHighValue &&
           (value.as_double() < low_value.as_double() || value.as_double() > high_value.as_double());
  }

  inline Value handle_outlier(const Value& value) const {
    switch (outliertreatmentmethod.value) {  // in this two cases it must have low or high value
      case OutlierTreatmentMethod::OutlierTreatmentMethodValue::AS_MISSING_VALUES:
        return handle_missing();
      case OutlierTreatmentMethod::OutlierTreatmentMethodValue::AS_EXTREME_VALUES:
        if (value.as_double() < low_value.as_double()) return extreme(low_value);
        return extreme(high_value);
      default:
        return value;  // "asIs" treatment
    }
  }

  // The value an outlier takes under asExtremeValues: the bound, which
  // pmml-4-4.xsd types NUMBER (xs:double) and which is compared as one, as a
  // value of the field. MiningSchema.html: asExtremeValues "outlier values
  // are changed to a specific high or low value defined in MiningField"; the
  // field's value is of its dataType, and every double literal is a float
  // literal (XML Schema Part 2 §3.2.4.1, §3.2.5.1), so for a float field the
  // bound read as a float (§3.4 K1 rule 1): the double rounded to the nearest
  // float (P6.2; JPMML 1.7.7: a bound 0.1 gives 0.10000000149011612). An
  // integer field keeps the double bound (D22, the slot widened).
  inline Value extreme(const Value& bound) const {
    if (datatype.value != DataType::DataTypeValue::FLOAT) return bound;
    return Value(cpmml::types::Float{cpmml::types::narrow_to_float(bound.as_double())});
  }

  inline bool is_invalid(const Value& value) const { return !is_valid_value(value); }

  // Whether the record may be scored with the value in this field's slot
  // (P6.13, B64). MiningSchema.html: "asIs means to use the input without
  // modification", so under asIs a value outside the field's Values or
  // Intervals is the field's value and the record is scored with it: an
  // ordinal string outside the Values (`zzz` of low < mid < high) has no
  // place in the order, so an ordering of it is indeterminate (a missing
  // result, an UNKNOWN predicate; P6.3, D52), while equality and isIn compare
  // its text. Every other treatment has replaced or refused an invalid value
  // in prepare(), except returnInvalid by default (no attribute), which the
  // record's validation refuses here.
  inline bool passes_validation(const Sample& sample) const {
    return invalidvalue_treatmentmethod.value == InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_IS ||
           validate(sample);
  }

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
  // (lexical.h; DataDictionary.html: "The dataType reuses names and semantics
  // of atomic types in XML Schema"). Text that is not a value of that type is
  // an *invalid* value, not a missing one and never a silent 0 or false; it
  // goes to invalidValueTreatment, which applies whether or not the attribute
  // is written (its default is returnInvalid). As in JPMML 1.7.7:
  //   returnInvalid  the record is not scored (InvalidValueException);
  //   asMissing      the value is missing: missingValueReplacement, if any;
  //   asValue        the value is invalidValueReplacement;
  //   asIs           there is no value of the field's type to keep, so the
  //                  record is not scored either (JPMML: NumberFormatException).
  // A missing input is an absent one (DataDictionary.html: "Input value is
  // missing, for example, if a database column contains a null value") or a
  // declared marker; text, the empty text included, is read (P6.8, B37):
  //   string         any text is a value: XML Schema Part 2 §3.2.1, "The
  //                  value space of string is the set of finite-length
  //                  sequences of characters", the sequence of none included,
  //                  so "" is the empty string;
  //   other types    "" is in no lexical space (§3.2.2.1 boolean: "the
  //                  following legal literals {true, false, 1, 0}";
  //                  §3.3.13.1 integer: "a finite-length sequence of decimal
  //                  digits (#x30-#x39) with an optional leading sign";
  //                  §3.2.4.1 float and §3.2.5.1 double: "a mantissa
  //                  followed, optionally, by ... an exponent"), so it is
  //                  malformed, an invalid value.
  // A float or double "NaN" is a value (P6.8, B39): XML Schema Part 2 §3.2.5
  // double, "the value space of double also contains the following three
  // special values: positive and negative zero, positive and negative
  // infinity and not-a-number (NaN)" (§3.2.4 float alike), and §3.2.5.1 "The
  // special values positive and negative infinity and not-a-number have
  // lexical representations INF, -INF and NaN, respectively". It is valid
  // unless the field's Values or Intervals leave it out (it lies in no
  // Interval and equals no Value; DataDictionary.html: "If intervals are
  // present, any data that is outside the intervals will be considered
  // invalid"). JPMML 1.7.7 refuses it on every field ("cannot accept invalid
  // value NaN"): D55.
  enum class InputStatus { VALID, MALFORMED };

  inline InputStatus parse_input(std::string_view raw, Value& parsed) const {
    switch (datatype.value) {
      case DataType::DataTypeValue::STRING:
        parsed = Value::string(raw);
        return InputStatus::VALID;
      case DataType::DataTypeValue::BOOLEAN: {
        bool b = false;
        if (!lexical::parse_boolean(raw, b)) return InputStatus::MALFORMED;
        parsed = Value(cpmml::types::Boolean{b});
        return InputStatus::VALID;
      }
      case DataType::DataTypeValue::INTEGER: {
        double d = 0.0;
        if (!lexical::parse_integer_as_double(raw, d)) return InputStatus::MALFORMED;
        parsed = Value::integer_slot(d);
        return InputStatus::VALID;
      }
      case DataType::DataTypeValue::FLOAT: {
        // XML Schema Part 2 §3.2.4: "float is patterned after the IEEE
        // single-precision 32-bit floating point type"; a decimal
        // "maps to the normalized value in the value space of float that is
        // closest to d" (P6.2, B4: 0.1 is 0.10000000149011612, 3.4e39 is
        // INF, 1e-50 is 0).
        float f = 0;
        if (!cpmml::types::detail::read_float(raw, f)) return InputStatus::MALFORMED;
        parsed = Value(cpmml::types::Float{f});
        return InputStatus::VALID;
      }
      case DataType::DataTypeValue::DOUBLE: {
        double d = 0.0;
        if (!lexical::parse_double(raw, d)) return InputStatus::MALFORMED;
        parsed = Value(d);
        return InputStatus::VALID;
      }
    }
    return InputStatus::MALFORMED;  // unreachable: the switch is exhaustive
  }

  inline Value handle_malformed(std::string_view raw) const {
    switch (invalidvalue_treatmentmethod.value) {
      case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::RETURN_INVALID:
        throw cpmml::InvalidValueException("Invalid value for field " + name + ": \"" + std::string(raw) +
                                           "\" is not a valid " + to_lower(datatype.to_string()));
      case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_IS:
        throw cpmml::InvalidValueException("Invalid value for field " + name + ": \"" + std::string(raw) +
                                           "\" is not a valid " + to_lower(datatype.to_string()) +
                                           " and cannot be used asIs");
      default:
        return handle_invalid(Value());
    }
  }
};

#endif
