/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BIND_SCOPE_H
#define CPMML_BIND_SCOPE_H

/**
 * Design note -- the scope tree (§3.2, §3.3). FieldScope.html is quoted for
 * each rule in binder.h, where they are applied; this header holds the data.
 *
 *   Document scope            DataDictionary DataFields; TransformationDictionary definitions
 *   └─ Model scope M          (a model element)
 *      ├─ MiningFields of M
 *      ├─ TD instantiations   (the TD fields M references, directly or transitively)
 *      ├─ LocalTransformations of M
 *      ├─ chain outputs       (modelChain: the OutputFields of the segments bound so far)
 *      ├─ Output of M
 *      └─ Segment k -> child model scope C_k (its MiningFields resolve in M)
 *
 * Every declaration is a Binding with a slot in the FrameLayout. A child
 * MiningField with no treatment attributes is an alias of the parent's slot
 * (§3.2 "Aliasing"). Bindings live in a deque so pointers stay valid.
 *
 * Nothing in the engine includes this header (P1.2).
 */

#include <cstdint>
#include <deque>
#include <optional>
#include <pugixml.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "bind/typing.h"

namespace cpmml::bind {

using SlotId = std::uint32_t;
using ScopeId = std::uint32_t;

struct FieldType {
  StaticType type;               // the slot's static type (widened per §3.4 when needed)
  std::optional<OpType> optype;  // absent for an OutputField without optype
  bool has_values = false;       // an ordinal order is declared by Value elements
};

enum class Origin : std::uint8_t { DataField, MiningField, DerivedField, OutputField, TargetPlaceholder };

// The container a declaration belongs to: forward references (§3.5) are
// judged within one container, in document order.
enum class Container : std::uint8_t {
  DataDictionary,
  TransformationDictionary,
  MiningSchema,
  LocalTransformations,
  Output,
};

constexpr std::string_view origin_name(Origin o) {
  switch (o) {
    case Origin::DataField:
      return "DataField";
    case Origin::MiningField:
      return "MiningField";
    case Origin::DerivedField:
      return "DerivedField";
    case Origin::OutputField:
      return "OutputField";
    case Origin::TargetPlaceholder:
      return "target";
  }
  return "?";
}

struct Binding {
  enum class State : std::uint8_t { Declared, Typing, Typed };

  std::string name;
  Origin origin;
  Container container;
  FieldType type;
  SlotId slot = 0;
  ScopeId scope = 0;
  pugi::xml_node declaration;
  State state = State::Declared;
  std::uint32_t index = 0;            // position in its container, document order
  const Binding* source = nullptr;    // a MiningField: the binding it reads in the enclosing scope
  bool target_usage = false;          // a MiningField with usageType target or predicted
  std::vector<const Binding*> reads;  // a DerivedField or OutputField: the fields its expression reads
};

struct Scope {
  enum class Kind : std::uint8_t { Document, Model };

  Kind kind = Kind::Model;
  ScopeId id = 0;
  Scope* enclosing = nullptr;  // the parent model's scope, or the document scope
  std::string label;           // "TreeModel \"t\"", "MiningModel > Segment \"3\" > TreeModel"
  pugi::xml_node element;
  bool chain = false;  // a MiningModel whose Segmentation is a modelChain

  // Document scope.
  std::unordered_map<std::string, Binding*> data_fields;
  std::unordered_map<std::string, Binding*> td_fields;
  // Model scope.
  std::unordered_map<std::string, Binding*> mining;
  std::unordered_map<std::string, Binding*> local;
  std::unordered_map<std::string, Binding*> chain_outputs;
  std::unordered_map<std::string, Binding*> outputs;
  std::unordered_set<const Binding*> td_instances;  // TD definitions instantiated here
  std::unordered_set<std::string> target_names;     // the fields its Targets describe

  // The model's target, for document values that are target values
  // (TargetValue@value, ScoreDistribution@value, Node@score, ...).
  const Binding* target = nullptr;
};

struct SlotInfo {  // FrameLayout entry (§3.3)
  std::string name;
  StaticType type;
  ScopeId scope;
  Origin origin;
  SlotId alias_of;  // == own id unless a pass-through
  bool lazy;        // a DerivedField: evaluated on first read (§5.3)
};

struct FrameLayout {
  std::vector<SlotInfo> slots;
};

struct ModelBinding {
  ScopeId scope;
  std::string element;                        // the model element's name
  std::vector<std::size_t> children;          // indices into Binder::models()
  std::vector<const Binding*> derived_order;  // its LocalTransformations, in dependency order (§3.5)
  std::vector<const Binding*> output_order;   // its OutputFields, in dependency order
};

}  // namespace cpmml::bind

#endif  // CPMML_BIND_SCOPE_H
