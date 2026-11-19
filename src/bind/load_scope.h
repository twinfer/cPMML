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
 *   - the binding's static type as the engine's DataType (the temporal
 *     types are carried as DOUBLE, as DataType::read does).
 *
 * A name that resolves nowhere is a load error naming the field and the model
 * (E-SCOPE-UNRESOLVED), where the Indexer threw std::out_of_range.
 *
 * Load errors (§3.6): after binding, check() throws one
 * cpmml::ParsingException listing the first 20 errors whose code is a rule of
 * names and scopes, the rules P2.1 makes authoritative: E-SCOPE-UNRESOLVED,
 * E-SCOPE-TOP-LEVEL, E-NAME-DUPLICATE, E-CYCLE. Every other binder error
 * stays report-only until the design commit that owns it (Q7, ...); P2.3's
 * Constant errors are raised by the Constant constructor
 * (expression/constant.h). P2.2's document-text errors (E-VALUE-MALFORMED,
 * E-ATTRIBUTE-MALFORMED, E-KEY-NAN, E-KEY-DUPLICATE) are raised by the
 * engine's constructors as they read the text, through the helpers at the
 * end of this file: Load mode does not bind expression or model bodies, and
 * the constructors read every such element, inline DerivedFields included.
 * So is P2.7's E-TYPE-ORDER-MIXED (string x number ordering, §2.7), by the
 * Apply constructor over expression_type's argument types
 * (order_mixed_message).
 * The clashes that documents
 * which load today contain are warnings (W-NAME-SHADOWS-MININGFIELD,
 * W-NAME-SHADOWS-TD, W-SCOPE-FLAT; binder.h).
 */

#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bind/binder.h"
#include "cPMML.h"
#include "core/datatype.h"
#include "core/indexer.h"
#include "core/value.h"
#include "core/xmlnode.h"

namespace cpmml::bind {

class LoadScope {
 public:
  LoadScope(pugi::xml_node pmml, std::string_view text) : text_(text), binder_(pmml, text, Binder::Mode::Load) {}

  LoadScope(const LoadScope&) = delete;
  LoadScope& operator=(const LoadScope&) = delete;

  const Binder& binder() const { return binder_; }

  // The text content of `element`, an element of the engine's document with
  // no child node there, as the binder's reader sees it (P2.3,
  // expression/constant.h): pugixml's default flags, the engine's, drop a
  // text child that is only blanks, so <Constant> </Constant> and
  // <Constant/> look alike. The document text is parsed again, once per
  // load and only when an element asks, with parse_ws_pcdata_single (the
  // binder's flags, Binder(std::string_view)), so its encoding is converted
  // and its line ends normalised exactly as the engine's were (XML 1.0
  // §2.11: "translating both the two-character sequence #xD #xA and any #xD
  // that is not followed by #xA to a single #xA"), and `element` is found in
  // it by its position among the element children of each ancestor (the two
  // parses differ only in text nodes). Returns "" for an element with no
  // content, the blanks otherwise. Only while the model is built (Active).
  std::string lone_blanks(pugi::xml_node element) const {
    if (!blanks_doc_) {
      auto doc = std::make_unique<pugi::xml_document>();
      if (!doc->load_buffer(text_.data(), text_.size(), pugi::parse_default | pugi::parse_ws_pcdata_single))
        throw std::logic_error("LoadScope: the document the engine parsed does not parse again");
      blanks_doc_ = std::move(doc);
    }
    std::vector<std::size_t> path;  // element index under each ancestor, innermost first
    for (pugi::xml_node n = element; n.parent(); n = n.parent()) {
      std::size_t i = 0;
      for (pugi::xml_node s = n.previous_sibling(); s; s = s.previous_sibling())
        if (s.type() == pugi::node_element) i++;
      path.push_back(i);
    }
    pugi::xml_node m = *blanks_doc_;
    for (auto it = path.rbegin(); it != path.rend() && m; ++it) {
      std::size_t k = *it;
      pugi::xml_node c = m.first_child();
      for (; c; c = c.next_sibling())
        if (c.type() == pugi::node_element && k-- == 0) break;
      m = c;
    }
    if (!m || std::strcmp(m.name(), element.name()) != 0)
      throw std::logic_error(std::string("LoadScope: element ") + element.name() + " not found in the second parse");
    return m.child_value();
  }

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
    explicit Active(const LoadScope& scope) : scope_(scope), previous_(current()) { current() = &scope; }
    ~Active() {
      scope_.blanks_doc_.reset();  // the text it was parsed from is the loader's, gone after the build
      current() = previous_;
    }
    Active(const Active&) = delete;
    Active& operator=(const Active&) = delete;

   private:
    const LoadScope& scope_;
    const LoadScope* previous_;
  };

 private:
  std::string_view text_;  // the engine's document text, valid while the model is built
  Binder binder_;
  mutable std::unique_ptr<pugi::xml_document> blanks_doc_;  // lone_blanks, parsed on demand
};

// The engine's DataType of a static type (core/datatype.h: integer is its
// own value since P2.4; the temporal types are Double here, as
// parse_datatype gives them, until P10).
inline DataType engine_type(TypeId t) {
  switch (t) {
    case TypeId::String:
      return DataType::DataTypeValue::STRING;
    case TypeId::Float:
      return DataType::DataTypeValue::FLOAT;
    case TypeId::Boolean:
      return DataType::DataTypeValue::BOOLEAN;
    case TypeId::Integer:
      return DataType::DataTypeValue::INTEGER;
    case TypeId::Double:
      return DataType::DataTypeValue::DOUBLE;
    case TypeId::Missing:
    case TypeId::Invalid:
      break;
  }
  throw std::logic_error("engine_type: a sentinel is not a static type");
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

// ---------------------------------------------------------------- document text (P2.2)
//
// Design commit P2.2 (§3.4): every attribute is read with its XSD type from
// pmml-4-4.xsd, not with the enclosing field's type, and an xs:string
// attribute holding "a value of the field" is read by the K1 rule. Values are
// still the engine's Value (a double, or a string's code): only what the text
// is read as changes.
//
// The K1 rule itself (§3.4: "1. the text is a T literal -> a value of T;
// 2. else it is an xs:double literal -> a Double ...; 3. else (non-numeric
// text) -> load error") is what Value(text, DataType) computes in the old
// storage, where integer, float and double are all doubles: an integer
// field's "33.0" (rule 2) is the double 33 either way, and float text is not
// narrowed until P6 (B4). So a value of a field is Value(text, the field's
// type): what P2.2 changes is which type that is (the field's, from
// scope_resolve, not the enclosing DerivedField's).

// The engine type's TypeId (the temporal types are DOUBLE).
inline TypeId type_id(const DataType& t) {
  switch (t.value) {
    case DataType::DataTypeValue::STRING:
      return TypeId::String;
    case DataType::DataTypeValue::BOOLEAN:
      return TypeId::Boolean;
    case DataType::DataTypeValue::FLOAT:
      return TypeId::Float;
    case DataType::DataTypeValue::INTEGER:
      return TypeId::Integer;
    case DataType::DataTypeValue::DOUBLE:
      return TypeId::Double;
  }
  return TypeId::Double;
}

// The type an element's dataType attribute declares (integer kept apart from
// double; a temporal type is Double until P10), or nullopt when it has none.
inline StaticType declared_type_of(const XmlNode& node) {
  const auto v = node.optional_attribute("dataType");
  if (!v) return std::nullopt;
  const auto info = parse_datatype(*v);
  if (!info) throw cpmml::ParsingException(std::string(code::kEnum) + ": " + node.name() + "@dataType=\"" + *v +
                                           "\" is not a PMML DATATYPE");
  return info->type;
}

// D21 (§3.4): "The type is the nearest enclosing declared type (the
// DerivedField's or OutputField's dataType)". `fallback` is the engine type
// the expression was built with, for an OutputField whose dataType is
// inferred.
inline TypeId enclosing_type(const XmlNode& node, const DataType& fallback) {
  for (pugi::xml_node n = node.xml().parent(); n && n.type() == pugi::node_element; n = n.parent()) {
    const std::string_view name = n.name();
    if (name != "DerivedField" && name != "OutputField") continue;
    if (const StaticType t = declared_type_of(XmlNode(n))) return *t;
    break;
  }
  return type_id(fallback);
}

// The engine Value of a typed value. Float text is kept as its xs:double
// reading by the callers (B4, P6), so a Float here is only a cast's result.
inline ::Value engine_value(const types::Value& v) {
  if (const auto* i = std::get_if<types::Integer>(&v)) return ::Value(static_cast<double>(i->v));
  if (const auto* f = std::get_if<types::Float>(&v)) return ::Value(static_cast<double>(f->v));
  if (const auto* d = std::get_if<types::Double>(&v)) return ::Value(d->v);
  if (const auto* b = std::get_if<types::Boolean>(&v)) return ::Value(b->v ? 1.0 : 0.0);
  if (const auto* s = std::get_if<types::String>(&v))
    return ::Value(std::string(s->text()), DataType::DataTypeValue::STRING);
  throw std::logic_error("engine_value: a sentinel has no engine value");
}

// Document text read strictly as `type` (§3.4: Discretize binValue,
// mapMissingTo and defaultValue "parsed as that type"; MapValues "output
// cells, mapMissingTo, defaultValue as the MapValues type"; Apply
// "mapMissingTo, defaultValue (xs:string) parsed as the Apply's result
// type"), as an engine Value of `store`. When `store` is of another kind
// (string, boolean or number) than `type`, the value is cast to it (§2.6) at
// load: the old storage keeps one representation per declared slot, and the
// cast is the one the enclosing declaration applies to every value of the
// element (callers pass `store` = `type` where the element's other values are
// not constants). A malformed text, or a cast that has no value, is a load
// error naming `what`.
inline ::Value document_constant(const XmlNode& node, const std::string& what, const std::string& text, TypeId type,
                                 const DataType& store) {
  types::StringPool pool;
  types::ParseContext ctx{&pool, nullptr, nullptr};
  const types::Value v = types::cold::parse_as(ColdPathIssuer::make(), type, text, ctx);
  if (types::is_invalid(v))
    throw cpmml::ParsingException(std::string(code::kValueMalformed) + ": " + node.name() + " " + what + "=\"" + text +
                                  "\" is not a value of type " + std::string(type_name(type)) + " (§3.4)");
  const TypeId to = type_id(store);
  const auto kind = [](TypeId t) { return t == TypeId::String ? 0 : t == TypeId::Boolean ? 1 : 2; };
  // Same kind: the text as the engine reads it in `store` (a float's text
  // keeps its double reading, B4; an integer is a double).
  if (kind(type) == kind(to)) return ::Value(text, store);
  std::string scratch;
  const types::Value cast = types::cold::cast(ColdPathIssuer::make(), to, v, ctx, scratch);
  if (types::is_invalid(cast) || types::is_missing(cast))
    throw cpmml::ParsingException(std::string(code::kValueMalformed) + ": " + node.name() + " " + what + "=\"" + text +
                                  "\" of type " + std::string(type_name(type)) + " has no value of type " +
                                  std::string(type_name(to)) + " (§2.6)");
  ::Value out = engine_value(cast);
  // A number or boolean cast to a string slot keeps its number for numeric
  // readers (decided 2026-11-18, §9; Value::numeric_text).
  if (to == TypeId::String) out.numeric_text = true;
  return out;
}

// §3.4 K1, for text "parsed as T(field) by the rule above": "for a field of
// numeric type T: 1. the text is a T literal → a value of T; 2. else it is an
// xs:double literal → a Double ... 3. else (non-numeric text) → load error".
// "For boolean and string fields rule 2 does not exist". The type the text is
// read in; rule 3 is document_constant's E-VALUE-MALFORMED.
inline TypeId k1_type(const std::string& text, TypeId field) {
  if (field != TypeId::Integer && field != TypeId::Float) return field;
  types::StringPool pool;
  types::ParseContext ctx{&pool, nullptr, nullptr};
  if (!types::is_invalid(types::cold::parse_as(ColdPathIssuer::make(), field, text, ctx))) return field;
  return TypeId::Double;
}

// The DerivedField or OutputField whose value `node` gives (its slot): the
// element's parent, or the field above a chain of `if` Applies of which the
// element is a value argument (Functions.html, if: "Implements IF-THEN-ELSE
// logic", so the second or third argument is the Apply's value as it is).
// Null when the element's value is an argument of any other element, or the
// condition of an `if`.
inline pugi::xml_node slot_holder(const XmlNode& node) {
  pugi::xml_node n = node.xml();
  for (pugi::xml_node p = n.parent(); p && p.type() == pugi::node_element; n = p, p = p.parent()) {
    const std::string_view name = p.name();
    if (name == "DerivedField" || name == "OutputField") return p;
    if (name != "Apply" || std::string_view(p.attribute("function").value()) != "if") return {};
    pugi::xml_node condition = p.first_child();
    while (condition && condition.type() != pugi::node_element) condition = condition.next_sibling();
    if (condition == n) return {};
  }
  return {};
}

// Is `node` the expression of a DerivedField or OutputField, so that what it
// gives is that field's value (its slot), not an argument of another element?
inline bool is_slot_expression(const XmlNode& node) { return static_cast<bool>(slot_holder(node)); }

// The engine type a FieldRef's or an Apply's document constant (mapMissingTo,
// defaultValue, read in the element's own type `own`) is stored in. §3.4:
// "DerivedField dataType=X -> Cast<X>(expression)" (§2.6). Where the element
// gives the value of a DerivedField or OutputField (slot_holder) that
// declares a number or boolean type, the constant is cast to it at load:
// every reader of such a slot takes a number, so a string constant left
// uncast gives its string code (`concat` with mapMissingTo="5" in a double
// DerivedField gave 0 where JPMML 1.7.7 and the base build give 5). A nested
// element's constant goes to the Apply holding it, in the element's own type
// (§3.4). A string slot is slot_constant's case.
inline DataType constant_store(const XmlNode& node, TypeId own) {
  if (const pugi::xml_node holder = slot_holder(node))
    if (const StaticType declared = declared_type_of(XmlNode(holder)); declared && *declared != TypeId::String)
      return engine_type(*declared);
  return engine_type(own);
}

// Decided 2026-11-18 (typed-core.md §9, "after P2.1"): "A Constant of number
// or boolean type that lands in a string slot ... keeps its text as well as
// its number: string readers get the text (§3.4 Cast<string>, what base and
// JPMML give: "5", "true", "1"), numeric readers the number." The slot is a
// string (§3.4 "DerivedField dataType=X -> Cast<X>(expression)"), so the
// value is the text with its string code, as the base build stored it: every
// equality reader of the slot (a SimplePredicate or SimpleSetPredicate, a
// MapValues key, NormDiscrete, equal, notEqual, isIn) compares it as the
// string it is, and a string OutputField, concat and uppercase read the text.
// Value::numeric_text marks it, so that a numeric or boolean OutputField reads
// the number (`number`, the constant read in its own type) where the base
// build gave the string's code. Other numeric readers (an Apply argument, a
// predictor, a DerivedField of a number type over the slot) read the code as
// they do for every string value until P4.7 casts the reader.
inline ::Value string_slot_constant(const std::string& text, double number) {
  ::Value v(text, DataType::DataTypeValue::STRING);
  v.numeric_text = true;
  const double back = v.text_number();
  if (!(back == number || (std::isnan(back) && std::isnan(number))))
    throw std::logic_error("string_slot_constant: \"" + text + "\" does not read back as " + std::to_string(number));
  return v;
}

// A FieldRef's or an Apply's document constant `text` (mapMissingTo,
// defaultValue), read strictly in the element's own type `own` (§3.4: FieldRef
// "parsed as T(field)", Apply "parsed as the Apply's result type") and stored
// as constant_store says. `slot` is the engine type of the enclosing
// DerivedField or OutputField (the output_type every expression of it is
// built with). Where the slot is a string and the constant a number or
// boolean: the slot's own value (the element gives the field's value,
// slot_holder) is string_slot_constant; any other element's constant is an
// argument of the Apply holding it, so it stays in its own type (B11: "+"
// over FieldRef d mapMissingTo="5" adds 5) and carries its text for the
// string functions (concat gives "5z", as base and JPMML do).
inline ::Value slot_constant(const XmlNode& node, const std::string& what, const std::string& text, TypeId own,
                             const DataType& slot) {
  ::Value v = document_constant(node, what, text, own, constant_store(node, own));
  if (slot != DataType::DataTypeValue::STRING || own == TypeId::String) return v;
  if (is_slot_expression(node)) return string_slot_constant(text, v.value);
  if (v.svalue.empty()) v.svalue = text;
  return v;
}

// An attribute of XSD type NUMBER (xs:double): NormContinuous LinearNorm
// orig/norm, MiningField lowValue/highValue. Read as a double whatever the
// enclosing type (L1, B11).
inline ::Value number_attribute(const XmlNode& node, const std::string& name) {
  return ::Value(node.required_double_attribute(name), DataType::DataTypeValue::DOUBLE);
}

// NormContinuous@mapMissingTo and NormDiscrete@mapMissingTo: NUMBER (B11),
// a document constant of the element, stored in `slot` as slot_constant
// stores a FieldRef's. Its text in a string slot is the double's: §3.4 casts
// the number to the slot (Cast<string>), and D17 writes a number as Java text
// inside expressions, so mapMissingTo="9" is "9.0" there, as JPMML 1.7.7
// gives (the base build kept the attribute's text "9"; P2.2 before this rule
// gave no value).
inline ::Value number_constant(const XmlNode& node, const std::string& name, const DataType& slot) {
  ::Value v = number_attribute(node, name);
  if (slot != DataType::DataTypeValue::STRING) return v;
  const std::string text = java::double_text(v.value);
  if (is_slot_expression(node)) return string_slot_constant(text, v.value);
  v.svalue = text;
  return v;
}

// §2.7 / D26: "A NaN key is a load error (it can match nothing)". A key of a
// numeric field whose text is an xs:double NaN (rule 1 for float and double,
// rule 2 for integer).
inline bool is_nan_key(const std::string& text, const DataType& field_type) {
  if (field_type == DataType::DataTypeValue::STRING || field_type == DataType::DataTypeValue::BOOLEAN) return false;
  double d = 0;
  return lexical::parse_double(text, d) && std::isnan(d);
}

// ---------------------------------------------------------------- node types (P2.4)
//
// Design commit P2.4: the static type of an expression element, which the
// engine's load reads where it used to re-walk the document
// (BuiltInFunction::expression_kind, deleted): an Apply's argument and
// result types (expression/apply.h, core/builtinfunctions.h) and the type of
// a transformedValue's expression (output/transformedvalue.h). Load mode
// does not type expression bodies (Binder::resolve_expression_names), so the
// types are worked out on demand here by the binder's §3.4 rows, element by
// element as Binder::bind_expression gives them in report mode, with the
// binder's own scope lookup for FieldRef and its typing.h rules for Constant
// and Apply, and no diagnostics (report mode reports them):
//   Constant         its dataType (a temporal one is Double until P10); with
//                    none, the dataType of the built-in parameter it is
//                    passed to, when the pseudo-declaration gives one (S2,
//                    inherited_constant_type); else infer_constant
//                    (Transformations.html: "A Constant that consists solely
//                    of numbers will be treated as an integer, if a decimal
//                    point is present among the numbers the Constant will be
//                    treated as a double ... The presence of any non-numeric
//                    characters ... will result in the Constant being
//                    treated as a string");
//   FieldRef         T(field), the type of the binding the name refers to
//                    from the FieldRef (Binder::lookup, FieldScope.html), or
//                    of the undeclared LocalTransformations DerivedField or
//                    OutputField that shadows it and whose value the engine
//                    reads (Binder::value_type: a name already a MiningField
//                    or TransformationDictionary DerivedField, accepted with
//                    a warning),
//                    widened to the least restrictive with its mapMissingTo
//                    text's type when §3.4 rule 2 reads that text as a double
//                    (bind_fieldref);
//   NormContinuous, NormDiscrete, TextIndex
//                    double (§3.4);
//   Discretize, MapValues
//                    their dataType, else the nearest enclosing declared type
//                    (D21: the dataType of the DerivedField or OutputField
//                    the element is in; unknown in an OutputField without
//                    one, as the binder's ctx.enclosing is for a
//                    transformedValue or decision);
//   Apply            apply_result_type over its arguments (§4.1; `if` the
//                    common type of THEN and ELSE, D7, P2.6); unknown for a
//                    function that is not a built-in (the engine refuses it
//                    at load);
//   anything else    unknown (Aggregate, Lag: refused at load).
// Unknown (nullopt) is a type the binder does not give either: a boolean or
// string operand of arithmetic (Q7, E-TYPE-BOOLEAN-MIXED / E-TYPE-PARAM in
// report mode, not yet load errors), a FieldRef to an OutputField without a
// dataType whose type the binder cannot infer. Each caller says what it does
// with it.

// The type the dataType attribute of `node` names (temporal: Double, as
// parse_datatype gives it), or nullopt when it has none or it is not a
// DATATYPE (the element's own reader refuses that).
inline StaticType datatype_attribute(pugi::xml_node node) {
  const pugi::xml_attribute a = node.attribute("dataType");
  if (!a) return std::nullopt;
  const auto info = parse_datatype(a.value());
  return info ? StaticType(info->type) : std::nullopt;
}

// The position of `node` among the arguments of the Apply holding it (its
// element siblings after Extension*).
inline std::size_t argument_position(pugi::xml_node node) {
  std::size_t i = 0;
  for (pugi::xml_node s = node.previous_sibling(); s; s = s.previous_sibling())
    if (s.type() == pugi::node_element && std::strcmp(s.name(), "Extension") != 0) i++;
  return i;
}

inline StaticType expression_type(pugi::xml_node node) {
  const LoadScope* scope = LoadScope::current();
  if (!scope) throw std::logic_error("expression_type outside a model load");
  const std::string_view n = node.name();
  if (n == "Constant") {
    if (node.attribute("dataType")) return datatype_attribute(node);
    const pugi::xml_node parent = node.parent();
    if (std::strcmp(parent.name(), "Apply") == 0)
      if (const StaticType inherited =
              inherited_constant_type(parent.attribute("function").value(), argument_position(node)))
        return inherited;
    switch (infer_constant(node.text().get())) {
      case ConstantKind::Integer:
      case ConstantKind::IntegerOverflow:  // a load error of the Constant's own reader (D4)
        return TypeId::Integer;
      case ConstantKind::Double:
        return TypeId::Double;
      case ConstantKind::String:
        break;
    }
    return TypeId::String;
  }
  if (n == "FieldRef") {
    StaticType t = scope->binder().value_type(node, node.attribute("field").value());
    if (!t) return std::nullopt;
    if (const pugi::xml_attribute mm = node.attribute("mapMissingTo")) {
      types::StringPool pool;
      types::ParseContext ctx{&pool, nullptr, nullptr};
      if (read_document_text(*t, mm.value(), ctx).rule == TextRule::AsDouble)
        t = least_restrictive(t, TypeId::Double).type;
    }
    return t;
  }
  if (n == "NormContinuous" || n == "NormDiscrete" || n == "TextIndex") return TypeId::Double;
  if (n == "Discretize" || n == "MapValues") {
    if (node.attribute("dataType")) return datatype_attribute(node);
    for (pugi::xml_node p = node.parent(); p && p.type() == pugi::node_element; p = p.parent()) {
      const std::string_view pn = p.name();
      if (pn == "DerivedField" || pn == "OutputField") return datatype_attribute(p);
    }
    return std::nullopt;
  }
  if (n == "Apply") {
    const fn::Signature* sig = fn::find(node.attribute("function").value());
    if (!sig) return std::nullopt;
    std::vector<StaticType> args;
    for (pugi::xml_node c : node.children())
      if (c.type() == pugi::node_element && std::strcmp(c.name(), "Extension") != 0) args.push_back(expression_type(c));
    return apply_result_type(*sig, args).type;
  }
  return std::nullopt;
}

inline StaticType expression_type(const XmlNode& node) { return expression_type(node.xml()); }

// The load error of an Apply that orders a string against a number (§2.7,
// design commit P2.7; typing.h order_mixed), raised by the Apply constructor
// (core/builtinfunctions.h) with the argument types expression_type gives,
// in the binder's words for E-TYPE-ORDER-MIXED (Binder::check_comparable).
inline std::string order_mixed_message(const XmlNode& apply, const std::string& function, TypeId a, TypeId b) {
  const LoadScope* scope = LoadScope::current();
  const std::string model = scope ? scope->binder().scope_label(apply.xml()) : std::string();
  return std::string(code::kTypeOrderMixed) + ": Apply \"" + function + "\" in " +
         (model.empty() ? std::string("the TransformationDictionary") : model) + " orders " +
         std::string(type_name(a)) + " against " + std::string(type_name(b)) +
         " (§2.7; DataDictionary.html: \"Categorical field values can only be tested for equality\")";
}

}  // namespace cpmml::bind

#endif  // CPMML_BIND_LOAD_SCOPE_H
