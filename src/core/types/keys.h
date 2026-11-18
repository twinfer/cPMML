/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TYPES_KEYS_H
#define CPMML_TYPES_KEYS_H

/**
 * Design note -- keys of hash containers (§2.7 "Hash containers").
 *
 * The comparison (compare.h) is not reflexive -- INF against itself and NaN
 * are unordered -- so containers (MapValues tables, SimpleSetPredicate with
 * more than 150 members, NaiveBayes counts, distributions) use KeyHash and
 * KeyEqual: same alternative, same payload; -0.0 and 0.0 one key; pooled
 * strings by pointer, others by text. This is XSD identity (XML Schema Part
 * 2, §3.2.4/§3.2.5: the value space has "only one zero", and "for schema
 * purposes NaN = NaN"), so KeyEqual is an equivalence relation, as an
 * unordered container requires.
 *
 * Keys are of one type (the binder casts them at load), and the binder casts
 * the probe to that type before the lookup; a NaN key is a load error (D26).
 * The cast matters: against String keys an uncast numeric NaN or INF probe
 * would compare by Java text (D2, 'NaN' == NaN is true), which is not what
 * the table below gives. A probe that the comparison would call unordered
 * with some key of its own type is decided before the lookup
 * (decide_nonfinite), so the hashed path gives the answer the comparison
 * would give member by member:
 *
 *   ±INF probe  Indeterminate if the keys hold the same infinity (equal is
 *               missing for it, false for every other key), else NotFound
 *   NaN probe   Indeterminate if there is any key (equal is missing for
 *               every one), NotFound for no keys
 *
 * KeyHash and KeyEqual have typed overloads, for containers keyed by one
 * scalar type, and a Value overload. They are not behind ColdPath: a lookup
 * is per record by nature.
 */

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <variant>

#include "core/types/value.h"

namespace cpmml::types {

// The typed overloads only, so that a Value overload cannot be chosen for an
// alternative that lacks one (it would convert back to Value and recurse).
struct KeyHashOf {
  std::size_t operator()(const Missing&) const { return 0x6d; }
  std::size_t operator()(const Invalid& x) const { return 0x69 + static_cast<std::size_t>(x.reason); }
  std::size_t operator()(const Integer& x) const { return std::hash<std::int64_t>{}(x.v); }
  // Float and Double keys never meet in one container; both hash their value
  // as a double, with -0.0 as 0.0 and every NaN as one NaN.
  std::size_t operator()(const Float& x) const { return hash_double(x.v); }
  std::size_t operator()(const Double& x) const { return hash_double(x.v); }
  std::size_t operator()(const Boolean& x) const { return x.v ? 1 : 0; }
  std::size_t operator()(const String& x) const { return x.rep->hash; }

 private:
  // The bit pattern, after mapping -0.0 to 0.0 and every NaN to one NaN, so
  // the hash agrees with KeyEqual whatever std::hash<double> does.
  static std::size_t hash_double(double d) {
    if (d == 0) d = 0.0;
    if (std::isnan(d)) d = std::numeric_limits<double>::quiet_NaN();
    return std::hash<std::uint64_t>{}(std::bit_cast<std::uint64_t>(d));
  }
};
static_assert(ExhaustiveVisitor<KeyHashOf> && NoCatchAll<KeyHashOf>);

struct KeyHash : KeyHashOf {
  using KeyHashOf::operator();
  std::size_t operator()(const Value& v) const { return std::visit(static_cast<const KeyHashOf&>(*this), v); }
};

struct KeyEqualOf {
  bool operator()(const Missing&, const Missing&) const { return true; }
  bool operator()(const Invalid& a, const Invalid& b) const { return a.reason == b.reason; }
  bool operator()(const Integer& a, const Integer& b) const { return a.v == b.v; }
  bool operator()(const Float& a, const Float& b) const { return same_number(a.v, b.v); }
  bool operator()(const Double& a, const Double& b) const { return same_number(a.v, b.v); }
  bool operator()(const Boolean& a, const Boolean& b) const { return a.v == b.v; }
  bool operator()(const String& a, const String& b) const { return string_equal(a, b); }

 private:
  template <class F>
  static bool same_number(F a, F b) {
    return a == b || (std::isnan(a) && std::isnan(b));  // -0.0 == 0.0 by IEEE already
  }
};

struct KeyEqual : KeyEqualOf {
  using KeyEqualOf::operator();
  bool operator()(const Value& a, const Value& b) const {
    if (a.index() != b.index()) return false;
    return std::visit(Same{*this, b}, a);
  }

 private:
  struct Same {  // a's alternative against b's, which is the same one
    const KeyEqualOf& eq;
    const Value& b;
    template <Alternative T>
    bool operator()(const T& x) const {
      return eq(x, *std::get_if<T>(&b));
    }
  };
  static_assert(ExhaustiveVisitor<Same> && NoCatchAll<Same>);
};

// The infinities among a container's keys, recorded at load.
struct NonFiniteKeys {
  bool positive_infinity = false;
  bool negative_infinity = false;
  bool any_key = false;

  void note(const Value& key) {
    any_key = true;
    double d = 0;
    if (const auto* x = std::get_if<Double>(&key))
      d = x->v;
    else if (const auto* y = std::get_if<Float>(&key))
      d = y->v;
    else
      return;
    if (std::isinf(d)) (d > 0 ? positive_infinity : negative_infinity) = true;
  }
};

enum class Lookup : std::uint8_t { Found, NotFound, Indeterminate };

// The answer for a non-finite probe, decided before the lookup; nullopt for a
// finite probe (or a non-floating one), which is looked up as usual.
inline std::optional<Lookup> decide_nonfinite(const Value& probe, const NonFiniteKeys& keys) {
  double d = 0;
  if (const auto* x = std::get_if<Double>(&probe))
    d = x->v;
  else if (const auto* y = std::get_if<Float>(&probe))
    d = y->v;
  else
    return std::nullopt;
  if (std::isnan(d)) return keys.any_key ? Lookup::Indeterminate : Lookup::NotFound;
  if (std::isinf(d)) {
    const bool held = d > 0 ? keys.positive_infinity : keys.negative_infinity;
    return held ? Lookup::Indeterminate : Lookup::NotFound;
  }
  return std::nullopt;
}

}  // namespace cpmml::types

#endif  // CPMML_TYPES_KEYS_H
