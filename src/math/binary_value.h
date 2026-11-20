/*******************************************************************************
 * The 0/1 a binary similarity measure counts for a value of a string field
 * (design P6.5, B3, D15).
 *
 * ClusteringModel.html, ComparisonMeasure: "For binary or categorical data,
 * let two individuals X and Y compare their values for each attribute, and
 * a11 = number of times where Xi=1 and Yi=1 ... a00 = number of times where
 * Xi=0 and Yi=0". The counts are over the values 1 and 0, so a value takes
 * part as the boolean it is. A string is cast to boolean (§2.6 "string ->
 * number, boolean: strict XSD reading of the text", D42): XML Schema Part 2
 * §3.2.2.1, "An instance of a datatype that is defined as boolean can have
 * the following legal literals {true, false, 1, 0}", after whitespace
 * collapse (§4.3.6: "For all atomic datatypes other than string ... the value
 * of whiteSpace is collapse"). So "true" and "1" count as 1,
 * "false" and "0" as 0 (D15: JPMML 1.7.7 takes only true/false, "Expected
 * false or true, got \"1\""). Any other string ("TRUE", "yes", "1.0", the
 * empty string) is no boolean, which the counts do not cover: NaN, which
 * binary_measure_similarity reads as undefined, no prediction, as for a
 * number other than 0 or 1 (JPMML throws "Expected false or true" for both).
 * Numbers and booleans keep their as_double(); only a string field's value
 * is read here.
 ******************************************************************************/

#ifndef CPMML_BINARY_VALUE_H
#define CPMML_BINARY_VALUE_H

#include <limits>
#include <string_view>

#include "core/datatype.h"
#include "core/value.h"
#include "fn/cast.h"
#include "utils/lexical.h"

namespace cpmml::binary_measure {

// The 0/1 of document text of a string field (a TrainingInstances cell, a
// token of a Cluster Array of type="string"), NaN when it is no boolean.
inline double of_text(std::string_view text) {
  bool b = false;
  if (!lexical::parse_boolean(text, b)) return std::numeric_limits<double>::quiet_NaN();
  return b ? 1.0 : 0.0;
}

// The 0/1 of a present value of a string field, by the declared string ->
// boolean cast (fn::DeclaredCast: a String's text read strictly; a
// numeric_text value its number; a text-less string no boolean), NaN when it
// has no boolean.
inline double of_string_value(const ::Value& value) {
  static const fn::DeclaredCast to_boolean(types::TypeId::String, types::TypeId::Boolean);
  ::Value cast = value;
  if (!to_boolean.apply(cast)) return std::numeric_limits<double>::quiet_NaN();
  return cast.as_double();
}

// Whether a field of `type` is read by of_string_value under a binary measure.
inline bool reads_as_string(const DataType& type) { return type == DataType::DataTypeValue::STRING; }

}  // namespace cpmml::binary_measure

#endif  // CPMML_BINARY_VALUE_H
