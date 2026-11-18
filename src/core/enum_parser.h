/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_ENUM_PARSER_H
#define CPMML_ENUM_PARSER_H

#include <cstddef>
#include <string>

#include "cPMML.h"
#include "core/xmlnode.h"

/**
 * The enum reader: every enumerated attribute (and every xs:choice of element
 * names) in a PMML document is read through one table and the functions below.
 *
 * Design note.
 *
 * - One table per enumeration: the spec spelling, the cPMML value, the text
 *   to_string() gives back, and whether cPMML implements the member. A member
 *   the spec defines but cPMML does not implement stays in the table marked
 *   UNSUPPORTED, so the error can tell "you wrote a valid PMML value cPMML
 *   cannot score" from "you wrote something that is not PMML".
 * - Matching is exact. PMML enumerations are XSD enumerations, which are
 *   case-sensitive; JPMML 1.7.7 agrees (dataType="Double", optype="Continuous",
 *   multipleModelMethod="Average", activationFunction="Logistic" and
 *   modelType="OrdinalMultinomial" are all "Required attribute ... is not
 *   defined"). The previous reader lower-cased first, so all of those loaded.
 * - There is no silent fallback. Absent takes the spec default where the spec
 *   has one, and is a load error where the attribute is required; a present
 *   value outside the table is always a load error. JPMML departs from the
 *   spec here: it reads an unknown value of an optional attribute as absent
 *   (normalizationMethod="bogus" scores as none), which is the bug class this
 *   reader exists to remove, so cPMML follows the spec.
 * - Errors are ParsingExceptions naming the attribute (Element@attribute
 *   where the element is known), the value given, and the allowed values:
 *     invalid value "Double" for DataField@dataType; allowed values are ...
 *     unsupported value "table" for KNNInput@compareFunction; cPMML supports ...
 * - A table is a constexpr array scanned linearly: enumerations have at most a
 *   few dozen members, and every lookup happens at load time, never while
 *   scoring.
 */

enum class EnumSupport { SUPPORTED, UNSUPPORTED };

template <class EnumT>
struct EnumEntry {
  const char* token;  // the spec spelling, matched exactly
  EnumT value;
  const char* display = nullptr;  // what to_string() gives back; nullptr means the token
  EnumSupport support = EnumSupport::SUPPORTED;
};

namespace enum_reader {

template <class EnumT, size_t N>
inline std::string list_tokens(const EnumEntry<EnumT> (&table)[N], bool supported_only) {
  std::string out;
  for (const auto& entry : table) {
    if (supported_only && entry.support != EnumSupport::SUPPORTED) continue;
    if (!out.empty()) out += ", ";
    out += entry.token;
  }
  return out.empty() ? "none" : out;
}

}  // namespace enum_reader

template <class EnumT, size_t N>
inline const EnumEntry<EnumT>* enum_find(const std::string& raw, const EnumEntry<EnumT> (&table)[N]) {
  for (const auto& entry : table)
    if (raw == entry.token) return &entry;

  return nullptr;
}

/**
 * The value `raw` names in `table`. `what` names what was read, as the error
 * should show it: "KNNInput@compareFunction", or "measure element of
 * ComparisonMeasure" for an element-name choice.
 */
template <class EnumT, size_t N>
inline EnumT enum_value(const std::string& raw, const EnumEntry<EnumT> (&table)[N], const std::string& what) {
  const EnumEntry<EnumT>* entry = enum_find(raw, table);
  if (!entry)
    throw cpmml::ParsingException("invalid value \"" + raw + "\" for " + what +
                                  "; allowed values are " + enum_reader::list_tokens(table, false));
  if (entry->support != EnumSupport::SUPPORTED)
    throw cpmml::ParsingException("unsupported value \"" + raw + "\" for " + what + "; cPMML supports " +
                                  enum_reader::list_tokens(table, true));
  return entry->value;
}

// Required attribute: absent is a load error.
template <class EnumT, size_t N>
inline EnumT read_enum(const XmlNode& node, const char* attribute, const EnumEntry<EnumT> (&table)[N]) {
  const std::string what = node.name() + "@" + attribute;
  if (!node.exists_attribute(attribute))
    throw cpmml::ParsingException("required attribute " + what + " is missing; allowed values are " +
                                  enum_reader::list_tokens(table, false));
  return enum_value(node.required_attribute(attribute), table, what);
}

// Optional attribute: absent takes the spec default, present must be in the table.
template <class EnumT, size_t N>
inline EnumT read_enum(const XmlNode& node, const char* attribute, const EnumEntry<EnumT> (&table)[N],
                       EnumT spec_default) {
  const std::optional<std::string> raw = node.optional_attribute(attribute);
  if (!raw) return spec_default;
  return enum_value(*raw, table, node.name() + "@" + attribute);
}

// An xs:choice of elements, read by the element's own name.
template <class EnumT, size_t N>
inline EnumT read_enum_element(const XmlNode& element, const EnumEntry<EnumT> (&table)[N], const std::string& what) {
  return enum_value(element.name(), table, what);
}

template <class EnumT, size_t N>
inline std::string enum_display(EnumT value, const EnumEntry<EnumT> (&table)[N], const char* fallback) {
  for (const auto& entry : table)
    if (entry.value == value && entry.support == EnumSupport::SUPPORTED)
      return entry.display ? entry.display : entry.token;

  return fallback;
}

#endif
