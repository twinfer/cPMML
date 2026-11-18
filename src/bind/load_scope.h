/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BIND_LOAD_SCOPE_H
#define CPMML_BIND_LOAD_SCOPE_H

/**
 * Design note -- the binder at load (docs/design/typed-core.md §3.1, design
 * commit P2.1).
 *
 * core/modelbuilder.h binds every document it loads (Binder::Mode::Load) and
 * keeps the result active on its thread while the model objects are built.
 * Wherever a model or expression constructor used to ask the Indexer for a
 * referenced field (indexer->get_index(name), indexer->get_type(name)) it
 * calls scope_resolve(indexer, node, name), which answers
 *
 *   - the binding the name refers to from that node, by FieldScope.html's
 *     rules (§3.2, Binder::lookup), so a reference to a field declared later
 *     in its container has that field's declared type (two-phase, §3.5): B12,
 *     which threw std::out_of_range ("unordered_map::at: key not found");
 *   - today's Indexer slot for the name (Indexer::get_or_set): slots are
 *     still one flat name space until P7;
 *   - the binding's static type as the engine's DataType (integer and the
 *     temporal types are carried as DOUBLE, as DataType::read does).
 *
 * A name that resolves nowhere is a load error naming the field and the model
 * (E-SCOPE-UNRESOLVED), where the Indexer threw std::out_of_range.
 *
 * Load errors (§3.6): after binding, check() throws one
 * cpmml::ParsingException listing the first 20 errors whose code is a rule of
 * names and scopes, the rules P2.1 makes authoritative: E-SCOPE-UNRESOLVED,
 * E-SCOPE-TOP-LEVEL, E-NAME-DUPLICATE, E-CYCLE. Every other binder error
 * stays report-only until the design commit that owns it (P2.2 document
 * text, P2.3 Constants, P2.7 ordering, ...). The clashes that documents
 * which load today contain are warnings (W-NAME-SHADOWS-MININGFIELD,
 * W-NAME-SHADOWS-TD, W-SCOPE-FLAT; binder.h).
 */

#include <cstddef>
#include <string>
#include <string_view>

#include "bind/binder.h"
#include "cPMML.h"
#include "core/datatype.h"
#include "core/indexer.h"
#include "core/xmlnode.h"

namespace cpmml::bind {

class LoadScope {
 public:
  LoadScope(pugi::xml_node pmml, std::string_view text) : binder_(pmml, text, Binder::Mode::Load) {}

  LoadScope(const LoadScope&) = delete;
  LoadScope& operator=(const LoadScope&) = delete;

  const Binder& binder() const { return binder_; }

  static bool is_load_error(const Diagnostic& d) {
    return d.severity == Severity::Error && (d.code == code::kScopeUnresolved || d.code == code::kScopeTopLevel ||
                                             d.code == code::kNameDuplicate || d.code == code::kCycle);
  }

  // §3.6: "The binder collects every error before failing and throws one
  // cpmml::ParsingException listing the first 20."
  void check() const {
    std::string message;
    std::size_t n = 0;
    for (const Diagnostic& d : binder_.diagnostics().all()) {
      if (!is_load_error(d)) continue;
      if (++n > 20) continue;
      message += "\n  line " + std::to_string(d.line) + ": " + std::string(d.code) + " " +
                 (d.model.empty() ? std::string() : d.model + ": ") + d.message;
    }
    if (n == 0) return;
    throw cpmml::ParsingException("the document has " + std::to_string(n) + " error" + (n == 1 ? "" : "s") +
                                  (n > 20 ? " (the first 20 are listed)" : "") + ":" + message);
  }

  // The scope being loaded on this thread (set by Active), or nullptr.
  static const LoadScope*& current() {
    thread_local const LoadScope* scope = nullptr;
    return scope;
  }

  // Makes `scope` current for the lifetime of the guard (model construction).
  class Active {
   public:
    explicit Active(const LoadScope& scope) : previous_(current()) { current() = &scope; }
    ~Active() { current() = previous_; }
    Active(const Active&) = delete;
    Active& operator=(const Active&) = delete;

   private:
    const LoadScope* previous_;
  };

 private:
  Binder binder_;
};

// The engine's DataType of a static type (core/datatype.h: integer and the
// temporal types are carried as DOUBLE).
inline DataType engine_type(TypeId t) {
  switch (t) {
    case TypeId::String:
      return DataType::DataTypeValue::STRING;
    case TypeId::Float:
      return DataType::DataTypeValue::FLOAT;
    case TypeId::Boolean:
      return DataType::DataTypeValue::BOOLEAN;
    default:
      return DataType::DataTypeValue::DOUBLE;
  }
}

struct ResolvedField {
  std::size_t index;  // today's Indexer slot
  DataType datatype;  // the declared type of the field the name refers to
};

// scope.resolve (§3.1) answering with today's Indexer slots (P2.1). `from` is
// the element holding the reference (SimplePredicate, NumericPredictor,
// FieldColumnPair, ...).
inline ResolvedField scope_resolve(Indexer& indexer, const XmlNode& from, const std::string& name) {
  const LoadScope* scope = LoadScope::current();
  if (!scope) throw std::logic_error("scope_resolve(\"" + name + "\") outside a model load");
  const Binding* b = scope->binder().lookup(from.xml(), name);
  if (!b) {
    const std::string model = scope->binder().scope_label(from.xml());
    throw cpmml::ParsingException(std::string(code::kScopeUnresolved) + ": " + from.name() + " in " +
                                  (model.empty() ? std::string("the TransformationDictionary") : model) + ": field \"" +
                                  name + "\" is not in scope");
  }
  if (!b->type.type)
    throw cpmml::ParsingException(from.name() + ": the type of field \"" + name +
                                  "\" is unknown (its declaration has no valid dataType)");
  return ResolvedField{indexer.get_or_set(name), engine_type(*b->type.type)};
}

}  // namespace cpmml::bind

#endif  // CPMML_BIND_LOAD_SCOPE_H
