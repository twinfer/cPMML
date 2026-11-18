/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BIND_DIAGNOSTICS_H
#define CPMML_BIND_DIAGNOSTICS_H

/**
 * Design note -- what the binder reports (docs/design/typed-core.md §3.6, §3.7).
 *
 * A Diagnostic is a stable code, a message naming the element, the enclosing
 * model, and the document line (pugixml's offset_debug, counted in lines).
 * Code, message and line are enough to act on (review O4).
 *
 * Three severities:
 *
 *   Error        the document is wrong by a rule the spec states ("must",
 *                "is not valid PMML", "it is an error"), or the design makes
 *                it a load error (§2.7 string x number ordering, Q7 boolean
 *                mixed with numbers). P2 makes these fail the load.
 *   Warning      the document is accepted, as the design decided (Q3, Q4,
 *                Q6, D22 slot widening, Q2 narrowing casts), but a reader
 *                should know.
 *   Unsupported  the binder does not type this element or rule yet. Not a
 *                verdict on the document: it is the binder's own gap, listed
 *                so that nothing is skipped silently (the project rule).
 *
 * In report mode (P1.2) nothing fails: test/bind_report.cc writes every
 * diagnostic of every fixture to test/data/bind_baseline.txt. The codes are
 * the design's where it names one (E-SCOPE-UNRESOLVED, E-TYPE-ORDER-MIXED,
 * W-SCOPE-IMPLICIT, W-FORWARD-REF, W-ORDER-CATEGORICAL,
 * W-ORDINAL-STRING-NO-VALUES, W-SLOT-WIDENED); the others are listed below
 * with the rule each stands for.
 *
 * Nothing in the engine includes this header (P1.2).
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cpmml::bind {

enum class Severity : std::uint8_t { Error, Warning, Unsupported };

constexpr std::string_view severity_name(Severity s) {
  switch (s) {
    case Severity::Error:
      return "error";
    case Severity::Warning:
      return "warning";
    case Severity::Unsupported:
      return "unsupported";
  }
  return "?";
}

// The stable codes. Each names the rule it stands for; the binder's call
// site quotes the spec text.
namespace code {
// The document cannot be read at all (not XML, or no PMML element).
inline constexpr std::string_view kXml = "E-XML";
// FieldScope.html §3.2: a name that resolves nowhere.
inline constexpr std::string_view kScopeUnresolved = "E-SCOPE-UNRESOLVED";
// FieldScope.html: "For top-level Models, their MiningSchema can refer only
// to DataFields in the DataDictionary".
inline constexpr std::string_view kScopeTopLevel = "E-SCOPE-TOP-LEVEL";
// FieldScope.html naming rules ("must be unique", "cannot be re-used").
inline constexpr std::string_view kNameDuplicate = "E-NAME-DUPLICATE";
// FieldScope.html: "DerivedFields in the TransformationDictionary that have
// circular references ... are not valid PMML" (and any other cycle).
inline constexpr std::string_view kCycle = "E-CYCLE";
// An enumerated attribute with a value outside its XSD enumeration.
inline constexpr std::string_view kEnum = "E-ENUM";
// An attribute or child the XSD or the spec text requires is absent.
inline constexpr std::string_view kRequired = "E-REQUIRED";
// An attribute or child the spec says must not appear.
inline constexpr std::string_view kForbidden = "E-FORBIDDEN";
// An attribute of an XSD numeric type (NUMBER, INT-NUMBER, ...) whose text is
// not in that type's lexical space.
inline constexpr std::string_view kAttributeMalformed = "E-ATTRIBUTE-MALFORMED";
// Document text that must be a value of a type and is not (§3.4 rule 3, a
// typed Constant, a Discretize binValue, ...).
inline constexpr std::string_view kValueMalformed = "E-VALUE-MALFORMED";
// §2.7: ordering of a string against a number.
inline constexpr std::string_view kTypeOrderMixed = "E-TYPE-ORDER-MIXED";
// Functions.html / Q7: boolean mixed with another type.
inline constexpr std::string_view kTypeBooleanMixed = "E-TYPE-BOOLEAN-MIXED";
// An argument outside its parameter's category (§4.1): a string to a
// numeric parameter, a double to an integer parameter.
inline constexpr std::string_view kTypeParam = "E-TYPE-PARAM";
// Wrong number of arguments for a built-in function (§4.2 arity).
inline constexpr std::string_view kArity = "E-ARITY";
// Apply@function names no built-in and no DefineFunction.
inline constexpr std::string_view kFunctionUnknown = "E-FUNCTION-UNKNOWN";
// Transformations.html MapValues: "it is an error if the table entries used
// for matching are not unique" (B42).
inline constexpr std::string_view kKeyDuplicate = "E-KEY-DUPLICATE";
// §2.7 / D26: a NaN key can match nothing.
inline constexpr std::string_view kKeyNaN = "E-KEY-NAN";
// Transformations.html NormContinuous: "LinearNorm must be strictly sorted by
// ascending value of orig".
inline constexpr std::string_view kNormOrder = "E-NORM-ORDER";

// Q3: a name not in the model's scope, resolved in an enclosing one.
inline constexpr std::string_view kScopeImplicit = "W-SCOPE-IMPLICIT";
// Q4 / D12: a forward reference other than the allowed one.
inline constexpr std::string_view kForwardRef = "W-FORWARD-REF";
// Q6: ordering over a categorical operand.
inline constexpr std::string_view kOrderCategorical = "W-ORDER-CATEGORICAL";
// Q6: ordering over an ordinal string field without Values.
inline constexpr std::string_view kOrdinalNoValues = "W-ORDINAL-STRING-NO-VALUES";
// §3.4 / D22: a document value that is not an integer, written into an
// integer slot, widens the slot to double.
inline constexpr std::string_view kSlotWidened = "W-SLOT-WIDENED";
// §3.4 K1 rule 2 / D22: integer-field document text that is a double literal
// ("33.0") is read as a double and the comparison promotes.
inline constexpr std::string_view kTextWidened = "W-TEXT-WIDENED";
// §2.6 / Q2: a declared type more restrictive than its value's type; the
// cast can fail at runtime (B17, B24).
inline constexpr std::string_view kCastNarrowing = "W-CAST-NARROWING";
// §2.6: double into a float declaration rounds (never fails).
inline constexpr std::string_view kCastLossy = "W-CAST-LOSSY";
// A LocalTransformations DerivedField or OutputField named like a DataField
// outside the model's MiningSchema (FieldScope.html: "In general, field names
// in PMML should be unique"; no naming rule forbids it).
inline constexpr std::string_view kNameShadowsDataField = "W-NAME-SHADOWS-DATAFIELD";
// OutputField without the XSD-required dataType (§3.4: "inferred with
// warning when absent, as today").
inline constexpr std::string_view kOutputNoDataType = "W-OUTPUT-NO-DATATYPE";

// The binder does not type this element yet.
inline constexpr std::string_view kUnsupportedElement = "U-ELEMENT";
// A date/time dataType: mapped to double until P10 (Q8).
inline constexpr std::string_view kUnsupportedTemporal = "U-TEMPORAL";
// A built-in whose parameters need the temporal types (P10).
inline constexpr std::string_view kUnsupportedFunction = "U-FUNCTION";
// An Apply of a DefineFunction (§3.4: out of scope, refused at load today).
inline constexpr std::string_view kUnsupportedUserFunction = "U-USER-FUNCTION";
// OutputField@segmentId (B34, P8.15).
inline constexpr std::string_view kUnsupportedSegmentId = "U-SEGMENT-ID";
}  // namespace code

struct Diagnostic {
  Severity severity;
  std::string_view code;  // one of code::
  std::string message;    // names the element; does not repeat the model
  std::string model;      // the enclosing model, as a path ("" at document level)
  std::uint32_t line;     // 1-based; 0 when unknown
};

class Diagnostics {
 public:
  void add(Severity severity, std::string_view code, std::string message, std::string model, std::uint32_t line) {
    entries_.push_back(Diagnostic{severity, code, std::move(message), std::move(model), line});
  }

  const std::vector<Diagnostic>& all() const { return entries_; }

  std::size_t count(Severity severity) const {
    std::size_t n = 0;
    for (const auto& d : entries_) n += d.severity == severity;
    return n;
  }

  void clear() { entries_.clear(); }

 private:
  std::vector<Diagnostic> entries_;
};

}  // namespace cpmml::bind

#endif  // CPMML_BIND_DIAGNOSTICS_H
