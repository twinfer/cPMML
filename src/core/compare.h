/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_COMPARE_H
#define CPMML_COMPARE_H

/**
 * Design note -- the engine's comparisons (docs/design/typed-core.md §2.7,
 * P6.3: LegacyCompare is gone; B18, B44, D1, D2, D9).
 *
 * One ordering for every comparison of the engine's values: the built-ins
 * equal, notEqual, lessThan, lessOrEqual, greaterThan, greaterOrEqual, isIn,
 * isNotIn (fn/kernels.h) and SimplePredicate / SimpleSetPredicate
 * (core/predicate.h, core/fastpredicate.h). It is types::Comparison's
 * (core/types/compare.h) over the two values, dispatched by a switch on the
 * alternatives (no std::visit per record, §6.4); equal is `equivalent`,
 * lessOrEqual is `less or equivalent`, so equality and order cannot disagree
 * (L4). The rules, by operands:
 *
 *   number x number   in the promoted type, either order (D1). Functions.html:
 *                     "When the input parameters have multiple dataTypes, the
 *                     least restrictive dataType will be inherited by default
 *                     ... string, double, float, integer"; integer x float is
 *                     float, float x double is double (§2.5).
 *   string x number   equality only: the number's Java text (§2.6, Q1)
 *                     against the string, either order (D2: string is the
 *                     least restrictive type). Ordering one against the other
 *                     is a load error (E-TYPE-ORDER-MIXED, P2.7).
 *   string x string   equality by text (XML Schema Part 2 §3.2.1, P6.1);
 *                     order: UTF-8 code units (the binder warns, Q6).
 *   boolean x boolean false < true.
 *   an ordinal field  an ordering with an operand that is an ordinal field
 *   with Values       listing Values, of any type, follows their sequence,
 *                     for two operands of one kind (P6.3, P6.12;
 *                     DataDictionary.html: "For ordinal fields, if a sequence
 *                     of Value elements is given, the sequence defines the
 *                     ascending order on the respective field"): the integer
 *                     Values 3, 1, 2 order 3 < 1 < 2. Without Values the
 *                     rules above ("the ascending order of the
 *                     representation type").
 *   boolean x other   no common type: a load error (Functions.html: "boolean
 *                     dataTypes can not be mixed with other types", Q7).
 *   ±INF x same INF   indeterminate (Transformations.html: "the first
 *                     infinity is not equal to the second"; BuiltinFunctions
 *                     .html: "any attempt to either with itself is
 *                     indeterminate and will result in a missing value being
 *                     returned"; D9, B44).
 *   NaN x anything    indeterminate (D9, an extrapolation).
 *
 * "Indeterminate" is a missing result for a built-in and UNKNOWN for a
 * predicate. A missing operand never reaches here (the callers answer
 * missing / UNKNOWN first). "No common type" at evaluation is a binder bug
 * for a predicate (its value is read in the field's type) and an invalid
 * result for a built-in (§2.6: "Invalid(NoCast) otherwise").
 *
 * Static types. The engine's storage does not always hold a value in its
 * static type: a boolean result is the Double 1 or 0 (fn::adapter::boolean),
 * a number in a string slot may be a Double carrying its text. So the
 * built-ins read each operand as its static type first (operand(), the
 * binder's type, fixed at load): a string operand is its text (a String's,
 * or the text a number carries, else the number's Java text), a boolean one
 * true for 1 and false for 0, a float one the float. An unknown static type
 * (TypeId::Missing) reads the value as it is stored.
 *
 * Besides the comparisons, two readers of values of one type that are not
 * comparisons in the spec's sense (§2.7 "Hash containers"): same_value, the
 * identity of a field's value with a document value of its type (NormDiscrete,
 * SVM indicators, GRM factor levels, Association items), and ContainerKey, the
 * key of a container keyed by values of one type (MapValues, CategoricalPredictor,
 * DataField's Values): a number itself, a string its pool id (reps are
 * canonical, P6.1, so equal ids are equal texts).
 */

#include <cmath>
#include <compare>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/types/cast.h"
#include "core/types/compare.h"
#include "core/types/java_text.h"
#include "core/types/value.h"
#include "core/value.h"

namespace cpmml::compare {

using types::Purpose;
using types::Relation;
using types::Tri;
using types::TypeId;

inline const char* type_name(const TypeId t) {
  switch (t) {
    case TypeId::Missing:
      return "missing";
    case TypeId::Invalid:
      return "invalid";
    case TypeId::Integer:
      return "integer";
    case TypeId::Float:
      return "float";
    case TypeId::Double:
      return "double";
    case TypeId::Boolean:
      return "boolean";
    case TypeId::String:
      return "string";
  }
  return "?";
}

// ---------------------------------------------------------------- ordinal order
// The declared order of an ordinal field that lists Values, of any type
// (design items P6.3, P6.12; DataDictionary.html: "For ordinal fields, if a
// sequence of Value elements is given, the sequence defines the ascending
// order on the respective field"): the field's valid Values, read in its
// type at load (strings pooled). An ordinal field without Values has none
// ("If no Value elements are given, the order is always assumed to be the
// ascending order of the representation type"). A value is placed by
// types::same_member (a string by text, a number by value). A value outside
// the Values has no place in the order: a document value (a Constant
// operand, a SimplePredicate's value) outside them is a load error (D47,
// P6.9, P6.12); one computed per record (another field's value, an
// expression's, a DerivedField's) makes the comparison indeterminate
// (P6.3), an input kept outside a DataField's Values by
// invalidValueTreatment="asIs" included (P6.13, D52); any other input
// outside them is invalid and MiningSchema::validate fails the record first.
class OrdinalOrder {
 public:
  // An ordinal string field's Values (texts).
  explicit OrdinalOrder(const std::vector<std::string>& values) {
    members_.reserve(values.size());
    for (const auto& v : values) members_.push_back(::Value::string(v).typed());
  }
  // The Values read in the field's type (P6.12).
  explicit OrdinalOrder(std::vector<types::Value> members) : members_(std::move(members)) {
    for (const auto& m : members_)
      if (types::is_missing(m) || types::is_invalid(m))
        throw std::logic_error("compare::OrdinalOrder: a Value that is no value of the field");
  }
  // The rank of `x`, or -1.
  inline int rank(const types::Value& x) const {
    for (std::size_t i = 0; i < members_.size(); i++)
      if (types::same_member(x, members_[i])) return static_cast<int>(i);
    return -1;
  }
  inline std::size_t size() const { return members_.size(); }

  // a against b in the declared order: unordered when either has no place
  // in it, or when both are one infinity (D9: "infinity is not single
  // number, but a class of numbers"; the built-ins' "any attempt to either
  // with itself is indeterminate") or both NaN (a NaN Value is a member,
  // user decision 7a, but comparisons keep NaN != NaN, D51). A NaN against
  // another member takes the NaN Value's place in the sequence.
  inline std::partial_ordering order(const types::Value& a, const types::Value& b) const {
    const int ra = rank(a);
    const int rb = rank(b);
    if (ra < 0 || rb < 0) return std::partial_ordering::unordered;
    if (ra == rb && types::type_of(a) != types::TypeId::String && types::type_of(a) != types::TypeId::Boolean &&
        !std::isfinite(types::detail::member_double(a)))
      return std::partial_ordering::unordered;
    return ra <=> rb;
  }

 private:
  std::vector<types::Value> members_;
};

// ---------------------------------------------------------------- operands
// A comparison operand: the value as its static type (see the note). A
// String made here from a number's text points at `rep`, which points at
// `text`; both belong to the caller and outlive the comparison.
struct Scratch {
  std::string text[2];
  types::StringRep rep[2];
  std::string number;  // a number's Java text, for D2
};

inline Scratch& scratch() {
  static thread_local Scratch s;
  return s;
}

// The text a value has as a string (§2.6 number -> string: Java text, Q1): a
// String's own, the text a number carries, else its Java text.
inline void string_text(const ::Value& v, std::string& out) {
  out.clear();
  if (!v.text().empty() || std::holds_alternative<types::String>(v.typed())) {
    out.assign(v.text());
    return;
  }
  switch (types::type_of(v.typed())) {
    case TypeId::Integer:
      types::java_text(std::get<types::Integer>(v.typed()), out);
      return;
    case TypeId::Float:
      types::java_text(std::get<types::Float>(v.typed()), out);
      return;
    case TypeId::Double:
      types::java_text(std::get<types::Double>(v.typed()), out);
      return;
    case TypeId::Boolean:
      types::java_text(std::get<types::Boolean>(v.typed()), out);
      return;
    default:
      throw std::logic_error("compare::string_text: a missing value has no text");
  }
}

// `v` (not missing) read as `static_type` (TypeId::Missing: unknown, as
// stored); `slot` (0 or 1) picks the scratch a made String uses.
inline types::Value operand(const ::Value& v, const TypeId static_type, const int slot) {
  const types::Value& x = v.typed();
  const TypeId held = types::type_of(x);
  if (held == static_type || static_type == TypeId::Missing) return x;
  switch (static_type) {
    case TypeId::String: {
      Scratch& s = scratch();
      string_text(v, s.text[slot]);
      s.rep[slot] = types::StringRep{s.text[slot], types::StringRep::kUnpooled, 0};
      return types::String{&s.rep[slot]};
    }
    case TypeId::Boolean:
      // A boolean result is held as the Double 1 or 0 (fn::adapter::boolean).
      if (held == TypeId::Integer || held == TypeId::Float || held == TypeId::Double)
        return types::CastTo<types::Boolean>{nullptr, nullptr}(types::Double{v.as_double()});
      return x;
    case TypeId::Float:
      // A float operand is a float (P6.2 holds it as one; a guard for the rest).
      if (held == TypeId::Integer || held == TypeId::Double)
        return types::Float{types::narrow_to_float(v.as_double())};
      return x;
    case TypeId::Integer:
    case TypeId::Double:
      // An integer slot widened to double holds a Double (§3.4 N3): compared
      // as the number it holds. A Boolean in a number slot stays a Boolean:
      // no common type.
      return x;
    case TypeId::Missing:
    case TypeId::Invalid:
      break;
  }
  return x;
}

// ---------------------------------------------------------------- the ordering
namespace detail {
template <class A>
inline std::partial_ordering numeric_with(const A& a, const types::Value& b) {
  switch (types::type_of(b)) {
    case TypeId::Integer:
      return types::numeric_order(a, std::get<types::Integer>(b));
    case TypeId::Float:
      return types::numeric_order(a, std::get<types::Float>(b));
    default:
      return types::numeric_order(a, std::get<types::Double>(b));
  }
}

inline std::partial_ordering numeric(const types::Value& a, const types::Value& b) {
  switch (types::type_of(a)) {
    case TypeId::Integer:
      return numeric_with(std::get<types::Integer>(a), b);
    case TypeId::Float:
      return numeric_with(std::get<types::Float>(a), b);
    default:
      return numeric_with(std::get<types::Double>(a), b);
  }
}

// A string against a number's Java text (D2): equivalent or not.
inline std::partial_ordering text_against(const types::String& s, const types::Value& n) {
  std::string& t = scratch().number;
  t.clear();
  switch (types::type_of(n)) {
    case TypeId::Integer:
      types::java_text(std::get<types::Integer>(n), t);
      break;
    case TypeId::Float:
      types::java_text(std::get<types::Float>(n), t);
      break;
    default:
      types::java_text(std::get<types::Double>(n), t);
      break;
  }
  return s.text() == std::string_view(t) ? std::partial_ordering::equivalent : std::partial_ordering::unordered;
}

constexpr bool is_number(TypeId t) { return t == TypeId::Integer || t == TypeId::Float || t == TypeId::Double; }
}  // namespace detail

// a against b for `purpose`; nullopt: no common type. Neither is missing.
// `ordinal`: the declared order of an ordinal operand with Values, or null.
inline std::optional<std::partial_ordering> order(const types::Value& a, const types::Value& b, const Purpose purpose,
                                                  const OrdinalOrder* ordinal) {
  const TypeId ta = types::type_of(a);
  const TypeId tb = types::type_of(b);
  // An ordering with an ordinal field that lists Values: their order, for
  // operands of one kind (P6.12; two strings, two numbers, two booleans).
  // Equality is by value whatever the order.
  if (ordinal && purpose == Purpose::Ordering &&
      ((detail::is_number(ta) && detail::is_number(tb)) || (ta == tb && (ta == TypeId::String || ta == TypeId::Boolean))))
    [[unlikely]]
    return ordinal->order(a, b);
  if (detail::is_number(ta) && detail::is_number(tb)) return detail::numeric(a, b);
  if (ta == TypeId::String && tb == TypeId::String) {
    const auto& x = std::get<types::String>(a);
    const auto& y = std::get<types::String>(b);
    // Equality is by text whatever the order (equal or notEqual only ask
    // whether the ordering is `equivalent`).
    if (purpose == Purpose::Equality)
      return types::string_equal(x, y) ? std::partial_ordering::equivalent : std::partial_ordering::less;
    return types::compare(x, y);
  }
  if (ta == TypeId::Boolean && tb == TypeId::Boolean)
    return types::compare(std::get<types::Boolean>(a), std::get<types::Boolean>(b));
  if (purpose == Purpose::Equality) {
    // D2: the number as text, either order. "equivalent" or "not": the
    // relation only asks equal or notEqual of it.
    if (ta == TypeId::String && detail::is_number(tb)) {
      const auto o = detail::text_against(std::get<types::String>(a), b);
      return o == std::partial_ordering::equivalent ? o : std::partial_ordering::less;
    }
    if (detail::is_number(ta) && tb == TypeId::String) {
      const auto o = detail::text_against(std::get<types::String>(b), a);
      return o == std::partial_ordering::equivalent ? o : std::partial_ordering::greater;
    }
  }
  return std::nullopt;  // boolean x other; string x number ordering; a sentinel
}

// The relation, three-valued; nullopt: no common type.
inline std::optional<Tri> relate(const Relation r, const types::Value& a, const types::Value& b,
                                 const OrdinalOrder* ordinal) {
  const auto o = order(a, b, types::purpose_of(r), ordinal);
  if (!o) return std::nullopt;
  return types::relate(r, o);
}

// ---------------------------------------------------------------- not comparisons
// The identity of a field's value with a document value read in the field's
// type (NormDiscrete: "has value 1.0 if the value of input field f is v"; SVM
// indicators; GRM factor levels; Association items): two strings by text; a
// string and a number by the number's Java text (D2); numbers and booleans by
// the number they are (IEEE: INF is INF here, these are no built-in
// comparison). Neither is missing.
inline bool same_value(const ::Value& a, const ::Value& b) {
  const auto* sa = std::get_if<types::String>(&a.typed());
  const auto* sb = std::get_if<types::String>(&b.typed());
  if (sa && sb) return types::string_equal(*sa, *sb);
  if (sa && detail::is_number(types::type_of(b.typed())))
    return detail::text_against(*sa, b.typed()) == std::partial_ordering::equivalent;
  if (sb && detail::is_number(types::type_of(a.typed())))
    return detail::text_against(*sb, a.typed()) == std::partial_ordering::equivalent;
  if (sa || sb) return false;  // a string and a boolean
  return a.as_double() == b.as_double();
}

}  // namespace cpmml::compare

namespace cpmml {

// The key of a container keyed by values of one type (§2.7 "Hash
// containers"; the binder reads every key in the type of the field it is
// looked up with): a number or boolean the double it is, a string its pool
// id (reps are canonical, P6.1, so two strings have one id iff their texts
// are equal; a string the pool lacks has none, -1, and is never a key).
struct ContainerKey {
  static inline double of(const ::Value& v) {
    if (const auto* s = std::get_if<types::String>(&v.typed()))
      return s->rep->id == types::StringRep::kUnpooled ? -1.0 : static_cast<double>(s->rep->id);
    return v.as_double();
  }
  // A key: a document constant, whose string is pooled (document text is
  // interned at load). A string the pool lacks would have the key -1 of
  // every other string the pool lacks (B1): a programming error, never a key.
  static inline double of_constant(const ::Value& v) {
    if (const auto* s = std::get_if<types::String>(&v.typed()); s && s->rep->id == types::StringRep::kUnpooled)
      throw std::logic_error("ContainerKey::of_constant: the string \"" + std::string(v.text()) +
                             "\" is not in the document's pool");
    return of(v);
  }
};

}  // namespace cpmml

#endif  // CPMML_COMPARE_H
