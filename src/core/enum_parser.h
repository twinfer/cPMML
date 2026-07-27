
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_ENUM_PARSER_H
#define CPMML_ENUM_PARSER_H

#include <cstddef>
#include <string>

#include "utils/utils.h"

/**
 * One name/value table and three functions over it, replacing a dozen
 * hand-rolled `from_string`/`to_string` pairs.
 *
 * A PMML enumeration has at most a few dozen members, so a linear scan of a
 * constexpr array beats hashing and needs no static initialisation at all.
 */
template <class EnumT>
struct EnumEntry {
  const char* token;    // lower-case PMML spelling; matched case-insensitively
  EnumT value;
  const char* display;  // what to_string() gives back for this value
};

/**
 * XmlNode::get_attribute reports an absent attribute as the literal "null" and
 * the defaulted overload reports it as the caller's default, which for these
 * enumerations is the empty string. Both mean "the document did not say", which
 * is not the same as "the document said something unrecognised": the first
 * takes the attribute's PMML default, the second is a fault. No PMML
 * enumeration has a member spelled "null", so this cannot swallow a real token.
 */
inline bool is_absent_attribute(const std::string& raw) { return raw.empty() || raw == "null"; }

template <class EnumT, size_t N>
inline const EnumEntry<EnumT>* enum_find(const std::string& raw, const EnumEntry<EnumT> (&table)[N]) {
  const std::string token = to_lower(raw);
  for (const auto& entry : table)
    if (token == entry.token) return &entry;

  return nullptr;
}

// For a required attribute: anything not in the table is a fault in the document.
template <class EnumT, size_t N>
inline EnumT enum_value(const std::string& raw, const EnumEntry<EnumT> (&table)[N], const char* what) {
  if (const EnumEntry<EnumT>* entry = enum_find(raw, table)) return entry->value;

  throw cpmml::ParsingException(std::string("unsupported ") + what + ": " + raw);
}

// For an optional attribute with a PMML default: absent takes the default, but
// a value that is present and unrecognised is still a fault. Silently treating
// the second case as the first is how a misspelt normalizationMethod became
// "no normalization" and shipped plausible-looking probabilities.
template <class EnumT, size_t N>
inline EnumT enum_value(const std::string& raw, const EnumEntry<EnumT> (&table)[N], const char* what,
                        EnumT when_absent) {
  if (is_absent_attribute(raw)) return when_absent;

  return enum_value(raw, table, what);
}

template <class EnumT, size_t N>
inline std::string enum_display(EnumT value, const EnumEntry<EnumT> (&table)[N], const char* fallback) {
  for (const auto& entry : table)
    if (entry.value == value) return entry.display;

  return fallback;
}

#endif
