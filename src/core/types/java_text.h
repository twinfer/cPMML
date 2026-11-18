/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TYPES_JAVA_TEXT_H
#define CPMML_TYPES_JAVA_TEXT_H

/**
 * Design note -- Java text of a typed value (§2.9, Q1, D17).
 *
 * The spec gives no text for a number cast to string. The design writes every
 * value the way JPMML 1.7.7 writes it, which is the way Java writes the
 * value's type (Q1, decided as recommended):
 *
 *   Integer  Long.toString         "7", "-3"
 *   Float    Float.toString        "0.1" for 0.1f, "1.0E7", "Infinity", "NaN"
 *   Double   Double.toString       "2.0", "1.0E-4", "-0.0"
 *   Boolean  "true" / "false"      (XSD §3.2.2.2 canonical representation)
 *   String   its text
 *
 * Not a second implementation: the digits and layout are the existing
 * writer's (src/utils/java_format.h, write_float/write_double/integer_text,
 * checked against Java by test/java_format.cc on some 12000 values).
 */

#include <charconv>
#include <string>

#include "core/types/cold.h"
#include "core/types/value.h"
#include "utils/java_format.h"

namespace cpmml::types {

inline void java_text(const Integer& x, std::string& out) {
  char buffer[java::MAX_TEXT];
  out.append(buffer, std::to_chars(buffer, buffer + sizeof(buffer), x.v).ptr);
}
inline void java_text(const Float& x, std::string& out) {
  char buffer[java::MAX_TEXT];
  out.append(buffer, java::write_float(buffer, x.v));
}
inline void java_text(const Double& x, std::string& out) {
  char buffer[java::MAX_TEXT];
  out.append(buffer, java::write_double(buffer, x.v));
}
inline void java_text(const Boolean& x, std::string& out) { out += x.v ? "true" : "false"; }
inline void java_text(const String& x, std::string& out) { out += x.text(); }

namespace cold {

// JavaText over a whole Value. Appends to `out` and returns true for a
// value; a sentinel has no text, nothing is appended and the answer is false
// (the caller decides what a missing or invalid value shows as).
struct JavaText {
  std::string* out;
  bool operator()(const Missing&) const { return false; }
  bool operator()(const Invalid&) const { return false; }
  template <PmmlParsedType T>
  bool operator()(const T& x) const {
    java_text(x, *out);
    return true;
  }
};
static_assert(ExhaustiveVisitor<JavaText> && NoCatchAll<JavaText>);

inline bool java_text(ColdPath, const Value& v, std::string& out) { return std::visit(JavaText{&out}, v); }

}  // namespace cold
}  // namespace cpmml::types

#endif  // CPMML_TYPES_JAVA_TEXT_H
