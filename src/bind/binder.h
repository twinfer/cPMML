/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_BIND_BINDER_H
#define CPMML_BIND_BINDER_H

/**
 * Design note -- the binder in report mode (docs/design/typed-core.md §3,
 * design commit P1.2).
 *
 * One Binder per document. It parses the text with pugixml, as the engine
 * does, and produces without touching the engine:
 *
 *   - the scope tree (scope.h, §3.2): the document scope, one scope per model
 *     element, segments and the inner model of an AnomalyDetectionModel as
 *     child scopes, modelChain outputs in the MiningModel's scope;
 *   - a static type for every DataField, MiningField, DerivedField,
 *     OutputField and expression (§3.4), and a FrameLayout with a slot per
 *     declaration (aliases for pass-through MiningFields);
 *   - Diagnostics (§3.6): code, message naming the element, the enclosing
 *     model, line.
 *
 * Nothing here throws for a defect of the document: every defect is a
 * diagnostic, and binding goes on so that one run lists them all. An element
 * the binder does not type yet is an "unsupported" diagnostic, never a silent
 * skip: in a model body every element is either bound (a handler below), or
 * listed in kInertElements (by pmml-4-4.xsd it has no FIELD-NAME attribute,
 * no EXPRESSION or PREDICATE content, and no attribute holding a value of a
 * field, so there is nothing to bind), or reported as U-ELEMENT (one
 * diagnostic per model and element name, with the count).
 *
 * Rules, each quoted where it is applied:
 *
 *   names       FieldScope.html (resolution order §3.2, uniqueness, the
 *               MiningSchema gatekeeper, TD instantiation, modelChain)
 *   order       FieldScope.html "Forward referencing of fields is not
 *               allowed, with one exception" (§3.5; Q4: W-FORWARD-REF), cycles
 *   types       §3.4 per element, §2.5 promotion, §2.6 casts, §2.7 comparison
 *               (E-TYPE-ORDER-MIXED, Q6, Q7), the K1 document-text rule
 *   functions   fn/table.h (§4.2)
 *
 * Since P2.1 the engine binds every document at load through
 * bind/load_scope.h (Binder::Mode::Load).
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <pugixml.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bind/diagnostics.h"
#include "fn/table.h"
#include "bind/scope.h"
#include "bind/typing.h"
#include "core/types/cast.h"
#include "core/types/keys.h"
#include "core/types/parse.h"
#include "core/types/string_pool.h"
#include "core/types/value.h"
#include "utils/lexical.h"

namespace cpmml::bind {

namespace detail {

inline bool one_of(std::string_view s, std::initializer_list<std::string_view> list) {
  return std::find(list.begin(), list.end(), s) != list.end();
}

inline constexpr std::string_view kModelElements[] = {"AnomalyDetectionModel",
                                                      "AssociationModel",
                                                      "BayesianNetworkModel",
                                                      "BaselineModel",
                                                      "ClusteringModel",
                                                      "GaussianProcessModel",
                                                      "GeneralRegressionModel",
                                                      "MiningModel",
                                                      "NaiveBayesModel",
                                                      "NearestNeighborModel",
                                                      "NeuralNetwork",
                                                      "RegressionModel",
                                                      "RuleSetModel",
                                                      "SequenceModel",
                                                      "Scorecard",
                                                      "SupportVectorMachineModel",
                                                      "TextModel",
                                                      "TimeSeriesModel",
                                                      "TreeModel"};

inline bool is_model_element(std::string_view s) {
  // Every MODEL-ELEMENT name ends with "Model" but NeuralNetwork and
  // Scorecard: a cheap test first (the binder asks it of many elements).
  if (!s.ends_with("Model") && s != "NeuralNetwork" && s != "Scorecard") return false;
  return std::find(std::begin(kModelElements), std::end(kModelElements), s) != std::end(kModelElements);
}

inline bool is_predicate(std::string_view s) {
  return one_of(s, {"SimplePredicate", "CompoundPredicate", "SimpleSetPredicate", "True", "False"});
}

inline bool is_expression(std::string_view s) {
  return one_of(s, {"Constant", "FieldRef", "NormContinuous", "NormDiscrete", "Discretize", "MapValues", "TextIndex",
                    "Apply", "Aggregate", "Lag"});
}

// FIELD-NAME attributes that are references (pmml-4-4.xsd; declarations such
// as DerivedField@name are not listed). Applied to the elements of a model
// body that have no handler of their own, so every name is resolved.
struct FieldNameAttributes {
  std::string_view element;
  std::initializer_list<std::string_view> attributes;
};
inline const FieldNameAttributes kFieldNameAttributes[] = {
    {"VariableWeight", {"field"}},
    {"RegressionModel", {"targetFieldName"}},
    {"PredictiveModelQuality", {"targetField"}},
    {"UnivariateStats", {"field"}},
    {"Anova", {"target"}},
    {"PartitionFieldStats", {"field"}},
    {"VerificationField", {"field"}},
    {"GeneralRegressionModel",
     {"targetVariableName", "trialsVariable", "offsetVariable", "endTimeVariable", "startTimeVariable",
      "subjectIDVariable", "statusVariable", "baselineStrataVariable"}},
    {"TimeSeries", {"field"}},
    {"DynamicRegressor", {"field", "targetField"}},
    {"BlockIndicator", {"field"}},
};

// Elements of a model body with nothing to type (see the note above). Each
// was checked against pmml-4-4.xsd: no EXPRESSION or PREDICATE in its
// content, no attribute holding a value of a field, and no FIELD-NAME
// attribute other than those of kFieldNameAttributes, which are resolved.
inline constexpr std::string_view kInertElements[] = {
    // tree, rules, scorecard
    "Characteristics", "Characteristic", "RuleSelectionMethod",
    // regression, general regression
    "ParameterList", "Parameter", "FactorList", "CovariateList", "PPMatrix", "ParamMatrix",
    // neural network
    "NeuralInputs", "NeuralLayer", "Neuron", "Con", "NeuralOutputs",
    // support vector machine
    "VectorDictionary", "VectorInstance", "SupportVectors", "SupportVector", "Coefficients", "Coefficient",
    "LinearKernelType", "PolynomialKernelType", "RadialBasisKernelType", "SigmoidKernelType",
    // arrays and matrices (numbers)
    "Array", "REAL-SparseArray", "INT-SparseArray", "Indices", "REAL-Entries", "INT-Entries", "Matrix", "MatCell",
    // clustering, nearest neighbours
    "ComparisonMeasure", "euclidean", "squaredEuclidean", "chebychev", "cityBlock", "minkowski", "simpleMatching",
    "jaccard", "tanimoto", "binarySimilarity", "Cluster", "MissingValueWeights", "Comparisons", "KNNInputs",
    "InstanceFields",
    // naive Bayes
    "BayesInputs", "GaussianDistribution", "PoissonDistribution", "UniformDistribution", "AnyDistribution",
    // baseline
    "Baseline", "Alternate", "CountTable", "NormalizedCountTable",
    // gaussian process
    "RadialBasisKernel", "ARDSquaredExponentialKernel", "AbsoluteExponentialKernel", "GeneralizedExponentialKernel",
    "Lambda",
    // association
    "Itemset", "ItemRef", "AssociationRule",
    // anomaly detection
    "MeanClusterDistances",
    // text
    "TextDictionary", "TextCorpus", "TextDocument", "DocumentTermMatrix", "TextModelNormalization",
    "TextModelSimiliarity", "TextIndexNormalization",
    // time series (numbers, indices and times)
    "TimeValue", "Timestamp", "ExponentialSmoothing", "Level", "Trend_ExpoSmooth", "Seasonality_ExpoSmooth", "ARIMA",
    "NonseasonalComponent", "SeasonalComponent", "AR", "MA", "MACoefficients", "Residuals", "GARCH", "GARCHPart",
    "ARMAPart", "ResidualSquareCoefficients", "VarianceCoefficients", "PastVariances", "StateSpaceModel", "StateVector",
    "TransitionMatrix", "MeasurementMatrix", "PsiVector", "KalmanState", "FinalOmega", "FinalStateVector", "HVector",
    "MaximumLikelihoodStat", "Numerator", "Denominator", "RegressorValues", "TimeSeries", "DynamicRegressor",
    // producer content
    "Extension"};

inline bool is_inert(std::string_view s) {
  return std::find(std::begin(kInertElements), std::end(kInertElements), s) != std::end(kInertElements);
}

inline std::string quoted(std::string_view s) { return "\"" + std::string(s) + "\""; }

}  // namespace detail

// Arity of a built-in (§4.2), the function table's min_args/max_args: the
// BuiltinFunctions.html pseudo-declarations' ParameterFields ("+ ... <ParameterField
// name="a"/> <ParameterField name="b"/>", "The functions -, *, and / are
// defined in the same way and have two parameters"), "a variable number of
// <FieldRef/>" for the aggregates, "two or more parameters" (concat), "THEN
// part is required ... ELSE part is optional" (if). Shared by the binder's
// E-ARITY and the engine's load (design commit P2.8b2, fn/apply_node.h
// through load_scope.h arity_message).
inline bool arity_fits(const fn::Signature& sig, std::size_t n) {
  return n >= sig.min_args && (sig.max_args == fn::kVariadic || n <= sig.max_args);
}

// "has 1 argument; + takes 2 (BuiltinFunctions.html)".
inline std::string arity_mismatch(const fn::Signature& sig, std::size_t n) {
  const std::string expected =
      sig.max_args == fn::kVariadic
          ? std::to_string(sig.min_args) + " or more"
          : (sig.min_args == sig.max_args ? std::to_string(sig.min_args)
                                          : std::to_string(sig.min_args) + " to " + std::to_string(sig.max_args));
  return "has " + std::to_string(n) + " argument" + (n == 1 ? "" : "s") + "; " + std::string(sig.name) + " takes " +
         expected + " (BuiltinFunctions.html)";
}

class Binder {
 public:
  // Report: every element is bound (test/bind_report.cc, §3.7). Load: the
  // binder the engine runs on every document (P2.1, core/modelbuilder.h):
  // the scope tree, every declaration (two-phase, §3.5) and the bodies of the
  // DerivedFields and OutputFields, but not the model bodies (trees, tables,
  // predicates). A model body's names are resolved when the engine builds it,
  // through lookup() (bind/load_scope.h); binding all of them here as well
  // cost 25-56% of the load time of every benchmark model.
  enum class Mode : std::uint8_t { Report, Load };

  // Parses `text` and binds it. Every defect is in diagnostics().
  explicit Binder(std::string_view text) {
    for (std::size_t i = 0; i < text.size(); i++)
      if (text[i] == '\n') newlines_.push_back(i);
    newlines_ready_ = true;
    // pugixml's default drops whitespace-only PCDATA, which would make
    // <Constant dataType="double"> </Constant> read as empty content (a
    // missing value) when its xs:string content is " " (bind_constant). The
    // binder keeps a lone blanks-only text child; blanks between elements
    // (which have siblings) are still dropped. The engine's reader
    // (core/modelbuilder.h) keeps the default; P2 aligns it.
    const pugi::xml_parse_result parsed =
        doc_.load_buffer(text.data(), text.size(), pugi::parse_default | pugi::parse_ws_pcdata_single);
    if (!parsed) {
      add(Severity::Error, code::kXml, nullptr, pugi::xml_node(),
          std::string("the document is not well-formed XML: ") + parsed.description(),
          line_of_offset(static_cast<std::size_t>(parsed.offset)));
      return;
    }
    const pugi::xml_node pmml = doc_.child("PMML");
    if (!pmml) {
      add(Severity::Error, code::kXml, nullptr, pugi::xml_node(), "the document has no PMML element", 0);
      return;
    }
    bind_pmml(pmml);
  }

  // Binds `pmml`, a PMML element the engine already parsed from `text` (with
  // pugixml's default flags, so a blanks-only Constant reads as empty there;
  // P2.3 aligns the reader). `text` must outlive the Binder: lines are
  // counted in it only when a diagnostic needs one.
  Binder(pugi::xml_node pmml, std::string_view text, Mode mode) : mode_(mode), text_(text) {
    if (!pmml) {
      add(Severity::Error, code::kXml, nullptr, pugi::xml_node(), "the document has no PMML element", 0);
      return;
    }
    bind_pmml(pmml);
  }

  Binder(const Binder&) = delete;
  Binder& operator=(const Binder&) = delete;

  const Diagnostics& diagnostics() const { return diags_; }
  const FrameLayout& layout() const { return layout_; }
  const std::deque<Scope>& scopes() const { return scopes_; }
  const std::vector<ModelBinding>& models() const { return models_; }

  // §3.2 for the engine (P2.1): the binding `name` refers to from `from` (a
  // node of the bound document), by the rules of resolve() but without
  // diagnostics, or nullptr when it resolves nowhere. The context is read
  // from the ancestors of `from`: the nearest model element is the model
  // scope M, and an Output element on the way puts the reference inside M's
  // Output (step 1); a TransformationDictionary on the way, or no model
  // element at all, means the TD definitions' scope. A modelChain's outputs
  // are those of the segments before the one the reference is in, as
  // resolve() sees them while it binds the segments in order (FieldScope.html:
  // "All other fields in PMML must be defined before they are referenced").
  const Binding* lookup(pugi::xml_node from, std::string_view name) const { return lookup(from, name, nullptr); }

  // The static type of the value the engine reads for `name` from `from`
  // (load_scope.h expression_type): the type of lookup()'s binding, except
  // where, before it in lookup()'s order, the model has a LocalTransformations
  // DerivedField (or, from inside its Output, an OutputField) of that name
  // that the binder did not declare because the name is already a MiningField
  // or a TransformationDictionary DerivedField (shadow_warning). The engine
  // computes that field into the name's slot all the same, so its value is
  // what is read, and its dataType is its type, as the re-walk before P2.4
  // gave it (LocalTransformations, then Output, then the
  // TransformationDictionary, at the nearest enclosing model). nullopt: no
  // binding, or one of unknown type.
  //
  // A TransformationDictionary reference to a field whose slot a MiningField
  // widens (§3.4, N3, D22, W-SLOT-WIDENED) has the widened type, double
  // (design commit P4.7, As built). FieldScope.html: "When they are
  // instantiated inside a model element, DerivedFields from the
  // TransformationDictionary can only reference: MiningFields from the
  // MiningSchema of that model element", so what the reference reads is the
  // model's MiningField, whose slot holds the replacement 2.5. lookup()
  // resolves it to the DataField, as the TD bodies are bound once, in the
  // document scope; and the engine builds them once for every model of the
  // document (InternalEvaluator), so their types, casts and kernels are fixed
  // once: the type is the least restrictive over every MiningField of the
  // document that reads the field, double when one of them widens it.
  StaticType value_type(pugi::xml_node from, std::string_view name) const {
    std::optional<StaticType> shadow;
    const Binding* b = lookup(from, name, &shadow);
    if (shadow) return *shadow;
    if (!b) return StaticType{};
    return read_type(*b, in_transformation_dictionary(from));
  }

  // The type a reference to `b` reads: its own, or for a reference from the
  // TransformationDictionary to a field a MiningField widens, the widened
  // one (value_type above; bind_fieldref in report mode, so that the
  // binder's checks of a TD body see what the engine computes).
  StaticType read_type(const Binding& b, bool from_transformation_dictionary) const {
    if (from_transformation_dictionary && !widened_sources_.empty() && widened_sources_.count(&b))
      return least_restrictive(b.type.type, TypeId::Double).type;
    return b.type.type;
  }

 private:
  // lookup(); with `shadow`, an undeclared shadowing field found on the way
  // (value_type) is put there and the result is nullptr.
  const Binding* lookup(pugi::xml_node from, std::string_view name, std::optional<StaticType>* shadow) const {
    if (!document_) return nullptr;
    // The context of every element walked is remembered (a direct-mapped
    // cache), so a predicate deep in a tree stops at its parent Node, whose
    // own predicate was resolved before it.
    constexpr std::size_t kMaxVisited = 64;
    const void* visited[kMaxVisited];
    std::size_t depth = 0;
    std::size_t output_depth = kMaxVisited;  // the depth of the Output element walked, if any
    std::size_t segment_depth = kMaxVisited;  // the depth of the modelChain Segment walked, if any
    std::uint32_t segment = kNoSegment;       // its position: M's chain outputs before it are visible
    const Scope* model = nullptr;
    bool outer_in_output = false;  // the context found above the walk is inside an Output
    for (pugi::xml_node n = from; n && n.type() == pugi::node_element; n = n.parent(), depth++) {
      const void* id = n.internal_object();
      const ContextEntry& cached = context_cache_[context_slot(id)];
      if (cached.node == id) {
        model = cached.model;
        outer_in_output = cached.in_output;
        if (segment == kNoSegment) segment = cached.chain_segment;
        break;
      }
      const std::string_view nn = n.name();
      if (detail::is_model_element(nn)) {
        if (const auto it = element_scopes_.find(id); it != element_scopes_.end()) {
          model = it->second;
          break;
        }
      }
      if (depth < kMaxVisited) visited[depth] = id;
      if (nn == "Output" && output_depth == kMaxVisited) output_depth = depth;
      if (nn == "Segment" && segment == kNoSegment && !chain_segments_.empty())
        if (const auto it = chain_segments_.find(id); it != chain_segments_.end()) {
          segment = it->second;
          segment_depth = depth;
        }
      if (nn == "TransformationDictionary") {
        depth++;
        break;
      }
    }
    for (std::size_t i = 0; i < depth && i < kMaxVisited; i++)
      context_cache_[context_slot(visited[i])] =
          ContextEntry{visited[i], model, outer_in_output || (output_depth != kMaxVisited && i <= output_depth),
                       segment_depth == kMaxVisited || i <= segment_depth ? segment : kNoSegment};
    const bool in_output = outer_in_output || output_depth != kMaxVisited;
    const auto find = [name](const NameMap& map) -> const Binding* {
      if (map.empty()) return nullptr;
      const auto it = map.find(name);
      return it == map.end() ? nullptr : it->second;
    };
    if (!model) {  // TransformationDictionary definitions
      if (const Binding* b = find(document_->td_fields)) return b;
      return find(document_->data_fields);
    }
    const auto find_shadow = [name, shadow](const std::vector<std::pair<std::string, StaticType>>& shadows) {
      if (!shadow) return false;
      for (const auto& [n, t] : shadows)
        if (n == name) {
          *shadow = t;
          return true;
        }
      return false;
    };
    // 1. M's OutputFields (only from inside M's Output)
    if (in_output) {
      if (const Binding* b = find(model->outputs)) return b;
      if (find_shadow(model->output_shadows)) return nullptr;
    }
    // 2. M's LocalTransformations
    if (const Binding* b = find(model->local)) return b;
    if (find_shadow(model->local_shadows)) return nullptr;
    // 3. TransformationDictionary DerivedFields, unless M has a MiningField
    // of that name; 4. M's MiningFields
    const Binding* mining = find(model->mining);
    if (!mining)
      if (const Binding* b = find(document_->td_fields)) return b;
    if (mining) return mining;
    // modelChain outputs of the segments before the reference's
    if (const Binding* b = chain_output(*model, name, segment)) return b;
    // 5. Q3: an implicit pass-through from an enclosing scope (W-SCOPE-IMPLICIT)
    for (const Scope *inner = model, *s = model->enclosing; s; inner = s, s = s->enclosing) {
      if (s->kind == Scope::Kind::Document) {
        if (const Binding* b = find(s->data_fields)) return b;
        continue;
      }
      if (const Binding* b = find(s->local)) return b;
      if (find_shadow(s->local_shadows)) return nullptr;
      if (const Binding* b = find(s->mining)) return b;
      if (const Binding* b = chain_output(*s, name, inner->chain_position)) return b;
    }
    return nullptr;
  }

 public:

  // The label of the model scope `from` is in ("" outside any model), for
  // messages.
  std::string scope_label(pugi::xml_node from) const {
    for (pugi::xml_node n = from; n && n.type() == pugi::node_element; n = n.parent())
      if (const auto it = element_scopes_.find(n.internal_object()); it != element_scopes_.end())
        return label_of(*it->second);
    return {};
  }

  // The binding of `name` declared in the scope labelled `scope_label` ("" is
  // the document scope), or nullptr. For tests and the report.
  const Binding* find(std::string_view scope_label, std::string_view name) const {
    for (const Scope& s : scopes_) {
      if (s.label != scope_label) continue;
      for (const auto* map : {&s.data_fields, &s.td_fields, &s.mining, &s.local, &s.chain_outputs, &s.outputs}) {
        const auto it = map->find(std::string(name));
        if (it != map->end()) return it->second;
      }
    }
    return nullptr;
  }

 private:
  // ---------------------------------------------------------------- context
  struct Ctx {
    Scope* model = nullptr;          // nullptr: TransformationDictionary definitions
    bool in_output = false;          // inside the model's Output
    Binding* current = nullptr;      // the declaration whose expression is bound
    bool transformed_value = false;  // current is an OutputField with feature="transformedValue"
    StaticType enclosing;            // the nearest enclosing declared type (D21)
    // Where the diagnostics of this context are: an element (described only
    // when a diagnostic needs it: describing every MiningField of a forest
    // was most of the load-mode binder's time) or a text.
    pugi::xml_node where_node;
    std::string where_text;
    void set_where(pugi::xml_node n) {
      where_node = n;
      where_text.clear();
    }
    void set_where(std::string text) {
      where_node = pugi::xml_node();
      where_text = std::move(text);
    }
    std::string where() const { return where_node ? describe(where_node) : where_text; }
  };

  struct Typed {
    StaticType type;
    const Binding* field = nullptr;  // the expression is a FieldRef to this field
  };

  Mode mode_ = Mode::Report;
  pugi::xml_document doc_;
  std::string_view text_;  // Load mode: the engine's text, for line numbers
  mutable std::vector<std::size_t> newlines_;
  mutable bool newlines_ready_ = false;
  // The scope of every model element (lookup()).
  std::unordered_map<const void*, const Scope*> element_scopes_;
  // lookup()'s context of an element: its model scope (nullptr: the
  // TransformationDictionary definitions) and whether it is inside an Output.
  struct ContextEntry {
    const void* node = nullptr;
    const Scope* model = nullptr;
    bool in_output = false;
    std::uint32_t chain_segment = kNoSegment;  // inside Segment k of the model's modelChain
  };
  static constexpr std::uint32_t kNoSegment = UINT32_MAX;
  // The position of every Segment of a modelChain (lookup()).
  std::unordered_map<const void*, std::uint32_t> chain_segments_;
  // `chain`'s output `name` if it comes from a segment before position
  // `before` (kNoSegment: any segment, a reference outside the segments).
  const Binding* chain_output(const Scope& chain, std::string_view name, std::uint32_t before) const {
    if (chain.chain_outputs.empty()) return nullptr;
    const auto it = chain.chain_outputs.find(name);
    if (it == chain.chain_outputs.end()) return nullptr;
    return scopes_[it->second->scope].chain_position < before ? it->second : nullptr;
  }
  static constexpr std::size_t kContextCacheSize = 512;
  static std::size_t context_slot(const void* id) {
    return (reinterpret_cast<std::uintptr_t>(id) >> 5) & (kContextCacheSize - 1);
  }
  mutable std::array<ContextEntry, kContextCacheSize> context_cache_{};
  std::unique_ptr<types::StringPool> pool_ = std::make_unique<types::StringPool>();
  types::ParseContext parse_ctx_{pool_.get(), nullptr, nullptr};
  Diagnostics diags_;
  std::deque<Binding> bindings_;
  std::deque<Scope> scopes_;
  FrameLayout layout_;
  std::vector<ModelBinding> models_;
  Scope* document_ = nullptr;
  std::unordered_map<std::string, pugi::xml_node> user_functions_;
  // The document-level fields (a DataField, or a TransformationDictionary
  // DerivedField a sub-model's MiningField reads) whose slot a MiningField
  // widens, directly or through the MiningFields of enclosing models
  // (value_type).
  std::unordered_set<const Binding*> widened_sources_;

  // The pre-pass that fills widened_sources_ (bind_pmml), run after the
  // DataDictionary and the TransformationDictionary's names are declared and
  // before any body is typed: every MiningField of every model (nested and
  // segment models included) that widens an integer slot, by the rules
  // bind_mining_schema warns with (W-SLOT-WIDENED: a replacement that
  // read_document_text reads AsDouble, an asExtremeValues bound that is not
  // integral), noted at the document-level field its name resolves to. The
  // resolution follows resolve() from the parent model's scope (FieldScope.
  // html: a sub-model's MiningSchema "can refer to any fields defined within
  // the parent model's scope"): a parent's LocalTransformations DerivedField
  // or an earlier chain segment's OutputField is no document-level field (a TD
  // reference cannot read it), a TD DerivedField the parent has no
  // MiningField of is that field, a parent's MiningField is its own source,
  // and a name in no enclosing model is the DataField (Q3). A malformed
  // attribute widens nothing here; bind_mining_schema reports it.
  void note_widened_sources(pugi::xml_node pmml) {
    std::vector<pugi::xml_node> frames;
    for (pugi::xml_node child : pmml.children())
      if (child.type() == pugi::node_element && detail::is_model_element(child.name()))
        note_widened_in_model(child, frames);
  }

  void note_widened_in_model(pugi::xml_node model, std::vector<pugi::xml_node>& frames) {
    frames.push_back(model);
    for (pugi::xml_node mf : model.child("MiningSchema").children("MiningField")) {
      // Only these attributes widen; the source is looked for after them.
      if (!mf.attribute("missingValueReplacement") && !mf.attribute("invalidValueReplacement") &&
          !mf.attribute("outliers"))
        continue;
      const auto name = attr(mf, "name");
      if (!name) continue;
      const Binding* source = widened_source_of(*name, frames.size() - 1, frames);
      if (source && source->type.type && mining_field_widens(mf, *source->type.type)) widened_sources_.insert(source);
    }
    for (pugi::xml_node c : model.children()) {
      if (c.type() != pugi::node_element) continue;
      if (detail::is_model_element(c.name())) note_widened_in_model(c, frames);
      if (std::string_view(c.name()) == "Segmentation")
        for (pugi::xml_node seg : c.children("Segment"))
          for (pugi::xml_node m : seg.children())
            if (m.type() == pugi::node_element && detail::is_model_element(m.name())) note_widened_in_model(m, frames);
    }
    frames.pop_back();
  }

  // The document-level binding a MiningField `name` of frames[level] reads,
  // or nullptr when it reads a model's own field.
  const Binding* widened_source_of(std::string_view name, std::size_t level,
                                   const std::vector<pugi::xml_node>& frames) const {
    const auto declares = [&name](pugi::xml_node model, const char* container, const char* element) {
      for (pugi::xml_node f : model.child(container).children(element))
        if (std::string_view(f.attribute("name").value()) == name) return true;
      return false;
    };
    const auto data_field = [this, &name]() -> const Binding* {
      const auto it = document_->data_fields.find(name);
      return it == document_->data_fields.end() ? nullptr : it->second;
    };
    if (level == 0) return data_field();
    const pugi::xml_node parent = frames[level - 1];
    if (declares(parent, "LocalTransformations", "DerivedField")) return nullptr;
    const bool parent_mining = declares(parent, "MiningSchema", "MiningField");
    if (!parent_mining)
      if (const auto it = document_->td_fields.find(name); it != document_->td_fields.end()) return it->second;
    if (parent_mining) return widened_source_of(name, level - 1, frames);
    // modelChain: an earlier segment's OutputField.
    const pugi::xml_node segmentation = parent.child("Segmentation");
    if (segmentation && std::string_view(segmentation.attribute("multipleModelMethod").value()) == "modelChain")
      for (pugi::xml_node seg = frames[level].parent().previous_sibling("Segment"); seg;
           seg = seg.previous_sibling("Segment"))
        for (pugi::xml_node m : seg.children())
          if (m.type() == pugi::node_element && detail::is_model_element(m.name()) &&
              declares(m, "Output", "OutputField"))
            return nullptr;
    // Q3: an enclosing model's field, else the DataField.
    for (std::size_t j = level - 1; j-- > 0;) {
      if (declares(frames[j], "LocalTransformations", "DerivedField")) return nullptr;
      if (declares(frames[j], "MiningSchema", "MiningField")) return widened_source_of(name, j, frames);
    }
    return data_field();
  }

  // bind_mining_schema's W-SLOT-WIDENED rules, without diagnostics.
  bool mining_field_widens(pugi::xml_node mf, TypeId field_type) {
    for (const char* rep : {"missingValueReplacement", "invalidValueReplacement"})
      if (const auto text = attr(mf, rep))
        if (read_document_text(field_type, *text, parse_ctx_).rule == TextRule::AsDouble) return true;
    if (field_type != TypeId::Integer || std::string_view(mf.attribute("outliers").value()) != "asExtremeValues")
      return false;
    for (const char* bound : {"lowValue", "highValue"})
      if (const auto text = attr(mf, bound)) {
        double d = 0;
        if (lexical::parse_double(*text, d) && std::trunc(d) != d) return true;
      }
    return false;
  }

  static bool in_transformation_dictionary(pugi::xml_node from) {
    for (pugi::xml_node p = from; p && p.type() == pugi::node_element; p = p.parent())
      if (std::strcmp(p.name(), "TransformationDictionary") == 0) return true;
    return false;
  }
  // U-ELEMENT, one per (scope, element name): count and first line.
  struct Unsupported {
    std::string element;
    std::size_t count;
    std::uint32_t line;
    std::string reason;
  };
  std::map<ScopeId, std::vector<Unsupported>> unsupported_;

  // ---------------------------------------------------------------- reporting
  std::uint32_t line_of_offset(std::size_t offset) const {
    if (!newlines_ready_) {
      for (const char* p = text_.data();
           (p = static_cast<const char*>(std::memchr(p, '\n', text_.size() - (p - text_.data())))) != nullptr; p++)
        newlines_.push_back(static_cast<std::size_t>(p - text_.data()));
      newlines_ready_ = true;
    }
    return static_cast<std::uint32_t>(std::upper_bound(newlines_.begin(), newlines_.end(), offset) - newlines_.begin() +
                                      1);
  }
  std::uint32_t line_of(pugi::xml_node node) const {
    const std::ptrdiff_t offset = node ? node.offset_debug() : -1;
    return offset < 0 ? 0 : line_of_offset(static_cast<std::size_t>(offset));
  }

  void add(Severity severity, std::string_view code, const Scope* model, pugi::xml_node node, std::string message,
           std::uint32_t line = 0) {
    // Load mode counts lines for errors only: warnings are not shown until the
    // diagnostics API (Q9, P9), and counting needs a pass over the text.
    if (!line && (mode_ == Mode::Report || severity == Severity::Error)) line = line_of(node);
    diags_.add(severity, code, std::move(message), model ? label_of(*model) : std::string(), line);
  }
  void error(std::string_view code, const Ctx& ctx, pugi::xml_node node, const std::string& message) {
    add(Severity::Error, code, ctx.model, node, located(ctx, node, message));
  }
  void warning(std::string_view code, const Ctx& ctx, pugi::xml_node node, const std::string& message) {
    add(Severity::Warning, code, ctx.model, node, located(ctx, node, message));
  }
  void unsupported(std::string_view code, const Ctx& ctx, pugi::xml_node node, const std::string& message) {
    add(Severity::Unsupported, code, ctx.model, node, located(ctx, node, message));
  }

  // "<element> in <where>: <message>"
  static std::string located(const Ctx& ctx, pugi::xml_node node, const std::string& message) {
    std::string s = describe(node);
    const std::string where = ctx.where();
    if (!where.empty() && s != where) s += " in " + where;
    return s + ": " + message;
  }

  static std::string describe(pugi::xml_node node) {
    if (!node) return "document";
    const std::string_view n = node.name();
    std::string s(n);
    auto with = [&](const char* attribute) {
      const pugi::xml_attribute a = node.attribute(attribute);
      if (a) s += " " + detail::quoted(a.value());
    };
    if (detail::one_of(
            n, {"FieldRef", "NormContinuous", "NormDiscrete", "Discretize", "SimplePredicate", "SimpleSetPredicate",
                "Aggregate", "Lag", "ClusteringField", "KNNInput", "InstanceField", "Target", "FieldColumnPair"}))
      with("field");
    else if (n == "Apply")
      with("function");
    else if (detail::one_of(n, {"DataField", "DerivedField", "OutputField", "MiningField", "NumericPredictor",
                                "CategoricalPredictor", "Predictor", "DefineFunction"}))
      with("name");
    else if (n == "TextIndex")
      with("textField");
    else if (n == "MapValues")
      with("outputColumn");
    else if (detail::one_of(n, {"BayesInput", "BayesOutput"}))
      with("fieldName");
    else if (n == "PPCell")
      with("predictorName");
    else if (n == "Node" || n == "Segment")
      with("id");
    return s;
  }

  void note_unsupported(const Ctx& ctx, pugi::xml_node node, const std::string& reason) {
    const ScopeId sid = ctx.model ? ctx.model->id : document_->id;
    auto& list = unsupported_[sid];
    for (auto& u : list)
      if (u.element == node.name()) {
        u.count++;
        return;
      }
    list.push_back(Unsupported{node.name(), 1, mode_ == Mode::Report ? line_of(node) : 0, reason});
  }

  void flush_unsupported(Scope& scope) {
    const auto it = unsupported_.find(scope.id);
    if (it == unsupported_.end()) return;
    for (const auto& u : it->second)
      add(Severity::Unsupported, code::kUnsupportedElement, scope.kind == Scope::Kind::Model ? &scope : nullptr,
          pugi::xml_node(),
          u.element + ": not typed by the binder (" + u.reason + "); " + std::to_string(u.count) +
              (u.count == 1 ? " occurrence" : " occurrences"),
          u.line);
    unsupported_.erase(it);
  }

  // ---------------------------------------------------------------- attributes
  static std::optional<std::string_view> attr(pugi::xml_node node, const char* name) {
    const pugi::xml_attribute a = node.attribute(name);
    if (!a) return std::nullopt;
    return std::string_view(a.value());
  }

  // An enumerated attribute: E-ENUM for a value outside the enumeration.
  std::optional<std::string_view> enum_attr(const Ctx& ctx, pugi::xml_node node, const char* name,
                                            std::initializer_list<std::string_view> values, bool required = false) {
    const auto v = attr(node, name);
    if (!v) {
      if (required) error(code::kRequired, ctx, node, std::string("attribute ") + name + " is required (pmml-4-4.xsd)");
      return std::nullopt;
    }
    if (!detail::one_of(*v, values)) {
      error(code::kEnum, ctx, node,
            std::string("attribute ") + name + "=" + detail::quoted(*v) + " is not one of its XSD enumeration");
      return std::nullopt;
    }
    return v;
  }

  // An attribute of XSD type NUMBER (xs:double) or REAL-NUMBER/PROB-NUMBER.
  std::optional<double> number_attr(const Ctx& ctx, pugi::xml_node node, const char* name, bool required = false) {
    const auto v = attr(node, name);
    if (!v) {
      if (required) error(code::kRequired, ctx, node, std::string("attribute ") + name + " is required (pmml-4-4.xsd)");
      return std::nullopt;
    }
    double d = 0;
    if (!lexical::parse_double(*v, d)) {
      error(code::kAttributeMalformed, ctx, node,
            std::string("attribute ") + name + "=" + detail::quoted(*v) + " is not an xs:double (XSD type NUMBER)");
      return std::nullopt;
    }
    return d;
  }

  // An attribute of XSD type INT-NUMBER (xs:integer).
  std::optional<std::int64_t> int_attr(const Ctx& ctx, pugi::xml_node node, const char* name) {
    const auto v = attr(node, name);
    if (!v) return std::nullopt;
    std::int64_t i = 0;
    if (types::detail::read_integer(*v, i) != types::detail::IntegerText::Ok) {
      error(
          code::kAttributeMalformed, ctx, node,
          std::string("attribute ") + name + "=" + detail::quoted(*v) + " is not an xs:integer (XSD type INT-NUMBER)");
      return std::nullopt;
    }
    return i;
  }

  std::optional<bool> bool_attr(const Ctx& ctx, pugi::xml_node node, const char* name) {
    const auto v = attr(node, name);
    if (!v) return std::nullopt;
    bool b = false;
    if (!lexical::parse_boolean(*v, b)) {
      error(code::kAttributeMalformed, ctx, node,
            std::string("attribute ") + name + "=" + detail::quoted(*v) + " is not an xs:boolean");
      return std::nullopt;
    }
    return b;
  }

  // ---------------------------------------------------------------- declarations
  Binding& declare(std::string name, Origin origin, Container container, FieldType type, Scope& scope,
                   pugi::xml_node node, std::uint32_t index) {
    Binding& b = bindings_.emplace_back();
    b.name = std::move(name);
    b.origin = origin;
    b.container = container;
    b.type = type;
    b.scope = scope.id;
    b.declaration = node;
    b.index = index;
    b.slot = static_cast<SlotId>(layout_.slots.size());
    layout_.slots.push_back(SlotInfo{b.name, type.type, scope.id, origin, b.slot, origin == Origin::DerivedField});
    return b;
  }

  Scope& new_scope(Scope::Kind kind, Scope* enclosing, std::string label, pugi::xml_node element) {
    Scope& s = scopes_.emplace_back();
    s.kind = kind;
    s.id = static_cast<ScopeId>(scopes_.size() - 1);
    s.enclosing = enclosing;
    s.label = std::move(label);
    s.element = element;
    return s;
  }

  // The type a dataType attribute declares; E-ENUM, U-TEMPORAL as needed.
  StaticType declared_type(const Ctx& ctx, pugi::xml_node node, bool required) {
    const auto v = attr(node, "dataType");
    if (!v) {
      if (required) error(code::kRequired, ctx, node, "attribute dataType is required (pmml-4-4.xsd)");
      return std::nullopt;
    }
    const auto info = parse_datatype(*v);
    if (!info) {
      error(code::kEnum, ctx, node, "dataType=" + detail::quoted(*v) + " is not a PMML DATATYPE");
      return std::nullopt;
    }
    if (info->temporal)
      unsupported(code::kUnsupportedTemporal, ctx, node,
                  "dataType=" + detail::quoted(*v) + " is carried as double until P10 (Q8)");
    return info->type;
  }

  std::optional<OpType> declared_optype(const Ctx& ctx, pugi::xml_node node, bool required) {
    const auto v = enum_attr(ctx, node, "optype", {"categorical", "ordinal", "continuous"}, required);
    if (!v) return std::nullopt;
    return parse_optype(*v);
  }

  // ---------------------------------------------------------------- document values (K1)
  // A document value of a field of type `type` (§3.4). `what` names the
  // attribute or cell. Returns the value's type: the field's (rule 1), or
  // Double (rule 2), or nullopt when malformed or the type is unknown.
  StaticType document_value(const Ctx& ctx, pugi::xml_node node, StaticType type, std::string_view text,
                            const std::string& what, const Binding* field, types::Value* out = nullptr) {
    if (!type) return std::nullopt;
    const DocumentValue v = read_document_text(*type, text, parse_ctx_);
    if (out) *out = v.value;
    if (v.rule == TextRule::AsType) return type;
    const std::string of = field ? " of " + std::string(type_name(*type)) + " field " + detail::quoted(field->name)
                                 : " of type " + std::string(type_name(*type));
    switch (v.rule) {
      case TextRule::AsType:
        return type;
      case TextRule::AsDouble:
        // §3.4 rule 2 (D22): "else it is an xs:double literal -> a Double;
        // the comparison promotes the field's value to double".
        warning(code::kTextWidened, ctx, node,
                what + "=" + detail::quoted(text) + of + " is an xs:double literal, read as double (§3.4 rule 2, D22)");
        return TypeId::Double;
      case TextRule::Malformed:
        break;
    }
    // §3.4 rule 3: "else (non-numeric text) -> load error"; XSD boolean
    // lexical space {true, false, 1, 0}.
    error(code::kValueMalformed, ctx, node, what + "=" + detail::quoted(text) + of + " is not a value of that type");
    return std::nullopt;
  }

  // Text parsed strictly as `type` (Apply@mapMissingTo, Discretize binValue,
  // MapValues output cells, ...).
  bool strict_value(const Ctx& ctx, pugi::xml_node node, StaticType type, std::string_view text,
                    const std::string& what) {
    if (!type) return true;
    const types::Value v = types::cold::parse_as(ColdPathIssuer::make(), *type, text, parse_ctx_);
    if (!types::is_invalid(v)) return true;
    error(code::kValueMalformed, ctx, node,
          what + "=" + detail::quoted(text) + " is not a value of type " + std::string(type_name(*type)));
    return false;
  }

  // §2.6: a declaration's explicit cast of a value of type `from`.
  void check_cast(const Ctx& ctx, pugi::xml_node node, StaticType from, StaticType to, const std::string& what) {
    if (!from || !to) return;
    switch (classify_cast(*from, *to)) {
      case CastKind::Identity:
      case CastKind::Widening:
      case CastKind::Explicit:
        return;
      case CastKind::Lossy:
        warning(code::kCastLossy, ctx, node,
                "declares " + std::string(type_name(*to)) + " over " + what + " of type " +
                    std::string(type_name(*from)) + ": the value is rounded to float (§2.6)");
        return;
      case CastKind::Narrowing:
        // Functions.html: "a double expression can not be implicitly cast as
        // an integer"; the declaration casts explicitly and the cast can give
        // an invalid value (Q2: an invalid result for the record).
        warning(code::kCastNarrowing, ctx, node,
                "declares " + std::string(type_name(*to)) + " over " + what + " of type " +
                    std::string(type_name(*from)) + ": the cast can fail at runtime (§2.6, Q2)");
        return;
    }
  }

  // ---------------------------------------------------------------- document level
  void bind_pmml(pugi::xml_node pmml) {
    document_ = &new_scope(Scope::Kind::Document, nullptr, "", pmml);
    Ctx doc_ctx;
    if (const pugi::xml_node dd = pmml.child("DataDictionary"))
      bind_data_dictionary(dd);
    else
      error(code::kRequired, doc_ctx, pmml, "DataDictionary is required (pmml-4-4.xsd)");
    const pugi::xml_node td = pmml.child("TransformationDictionary");
    std::vector<Binding*> td_declared;
    if (td) declare_transformation_dictionary(td, td_declared);
    // Before the TransformationDictionary's bodies are typed (design commit
    // P4.7, As built): value_type gives a TD reference to a field that a
    // MiningField widens the widened type, so the set has to be complete
    // when the TD's expressions and casts are checked, in report mode as in
    // load mode (where the engine asks only after the binder has run).
    note_widened_sources(pmml);
    if (td) bind_transformation_dictionary_bodies(td_declared);
    for (pugi::xml_node child : pmml.children()) {
      if (child.type() != pugi::node_element) continue;
      const std::string_view n = child.name();
      if (detail::one_of(n, {"Header", "MiningBuildTask", "DataDictionary", "TransformationDictionary", "Extension"}))
        continue;  // Header and MiningBuildTask hold no fields; Extension is producer content
      if (detail::is_model_element(n)) {
        bind_model(child, *document_, nullptr, "");
        continue;
      }
      note_unsupported(doc_ctx, child, "not a PMML child the binder knows");
    }
    flush_unsupported(*document_);
  }

  void bind_data_dictionary(pugi::xml_node dd) {
    Ctx ctx;
    std::uint32_t index = 0;
    for (pugi::xml_node df : dd.children()) {
      if (df.type() != pugi::node_element) continue;
      const std::string_view n = df.name();
      if (n == "Extension") continue;
      if (n != "DataField") {
        note_unsupported(ctx, df, "DataDictionary content other than DataField");
        continue;
      }
      ctx.set_where(df);
      const auto name = attr(df, "name");
      if (!name) {
        error(code::kRequired, ctx, df, "attribute name is required (pmml-4-4.xsd)");
        continue;
      }
      FieldType type{declared_type(ctx, df, true), declared_optype(ctx, df, true), false};
      // DataDictionary.html: "The name of a DataField must be unique from
      // other names in the DataDictionary".
      if (document_->data_fields.count(std::string(*name))) {
        error(code::kNameDuplicate, ctx, df,
              "DataField " + detail::quoted(*name) +
                  " is declared twice (DataDictionary.html: \"The name of a DataField must be unique\")");
        continue;
      }
      Binding& b =
          declare(std::string(*name), Origin::DataField, Container::DataDictionary, type, *document_, df, index++);
      b.state = Binding::State::Typed;
      document_->data_fields.emplace(b.name, &b);
      // Load mode: a DataField's Values and Intervals are document text of the
      // field's type (§3.4), which the engine still reads itself until P2.2.
      if (mode_ == Mode::Load) continue;
      bind_values(ctx, df, b, /*allow_property=*/true);
      for (pugi::xml_node interval : df.children("Interval")) {
        // DataDictionary.html: "Intervals are not allowed for non-continuous
        // fields."
        if (type.optype && *type.optype != OpType::Continuous)
          error(code::kForbidden, ctx, interval,
                "an Interval on a " + std::string(optype_name(*type.optype)) +
                    " field (DataDictionary.html: \"Intervals are not allowed for non-continuous fields\")");
        bind_interval(ctx, interval);
      }
      for (pugi::xml_node child : df.children()) {
        if (child.type() != pugi::node_element) continue;
        if (!detail::one_of(child.name(), {"Value", "Interval", "Extension"}))
          note_unsupported(ctx, child, "DataField content other than Value and Interval");
      }
    }
  }

  // Value children of a DataField (valid/invalid/missing) or of a
  // DerivedField/OutputField (an ordinal order: no property).
  void bind_values(const Ctx& ctx, pugi::xml_node field, Binding& b, bool allow_property) {
    bool any_valid = false;
    for (pugi::xml_node v : field.children("Value")) {
      const auto property = enum_attr(ctx, v, "property", {"valid", "invalid", "missing"});
      if (!allow_property && attr(v, "property")) {
        // Transformations.html: "The attribute property must not be used for
        // Value elements within a DerivedField"; Output.html says the same
        // of an OutputField.
        error(code::kForbidden, ctx, v,
              "attribute property must not be used for Value elements within a " + std::string(field.name()));
      }
      const auto value = attr(v, "value");
      if (!value) {
        error(code::kRequired, ctx, v, "attribute value is required (pmml-4-4.xsd)");
        continue;
      }
      // §2.8: a missing marker is matched as raw text first, so it may be
      // anything; valid and invalid values are read by the §3.4 rule.
      if (property && *property == "missing") continue;
      if (!property || *property == "valid") any_valid = true;
      document_value(ctx, v, b.type.type, *value, "Value@value", &b);
    }
    b.type.has_values = any_valid;
  }

  void bind_interval(const Ctx& ctx, pugi::xml_node interval) {
    enum_attr(ctx, interval, "closure", {"openClosed", "openOpen", "closedOpen", "closedClosed"}, true);
    const auto left = number_attr(ctx, interval, "leftMargin");
    const auto right = number_attr(ctx, interval, "rightMargin");
    // DataDictionary.html: "The attributes leftMargin and rightMargin are
    // optional but at least one value must be defined."
    if (!attr(interval, "leftMargin") && !attr(interval, "rightMargin"))
      error(code::kRequired, ctx, interval,
            "neither leftMargin nor rightMargin is given (DataDictionary.html: \"at least one value must be "
            "defined\")");
    (void)left;
    (void)right;
  }

  void declare_transformation_dictionary(pugi::xml_node td, std::vector<Binding*>& declared) {
    Ctx ctx;
    std::uint32_t index = 0;
    // Phase 1 (§3.5): every name, with its declared type.
    for (pugi::xml_node child : td.children()) {
      if (child.type() != pugi::node_element) continue;
      const std::string_view n = child.name();
      if (n == "Extension") continue;
      if (n == "DefineFunction") {
        // §3.4: DefineFunction is out of scope, refused at load today.
        if (const auto name = attr(child, "name")) user_functions_.emplace(std::string(*name), child);
        note_unsupported(ctx, child, "user-defined functions are out of scope (§3.4)");
        continue;
      }
      if (n != "DerivedField") {
        note_unsupported(ctx, child, "TransformationDictionary content other than DerivedField");
        continue;
      }
      ctx.set_where(child);
      const auto name = attr(child, "name");
      if (!name) {
        // Transformations.html: "If a DerivedField is contained in
        // TransformationDictionary or LocalTransformations, then the name
        // attribute is required."
        error(code::kRequired, ctx, child, "attribute name is required in a TransformationDictionary");
        continue;
      }
      FieldType type{declared_type(ctx, child, true), declared_optype(ctx, child, true), false};
      // FieldScope.html: "The names of DataFields in the DataDictionary must
      // be unique from the names of any other DataFields and the names of
      // DerivedFields in the TransformationDictionary"; "The names of
      // DerivedFields in the TransformationDictionary must be unique across
      // the enitre PMML document."
      const std::string key(*name);
      if (document_->data_fields.count(key) || document_->td_fields.count(key)) {
        error(code::kNameDuplicate, ctx, child,
              "name " + detail::quoted(*name) + " is already a " +
                  (document_->data_fields.count(key) ? "DataField" : "TransformationDictionary DerivedField") +
                  " (FieldScope.html: TransformationDictionary names \"must be unique across the enitre PMML "
                  "document\")");
        continue;
      }
      Binding& b =
          declare(key, Origin::DerivedField, Container::TransformationDictionary, type, *document_, child, index++);
      document_->td_fields.emplace(key, &b);
      declared.push_back(&b);
    }
  }

  void bind_transformation_dictionary_bodies(const std::vector<Binding*>& declared) {
    // Phase 2: bodies, in the document scope.
    for (Binding* b : declared) {
      Ctx body;
      body.current = b;
      body.set_where(b->declaration);
      body.enclosing = b->type.type;
      bind_derived_body(body, *b);
    }
    check_cycles(declared, nullptr);
  }

  // ---------------------------------------------------------------- resolution (§3.2)
  // FieldScope.html: "Forward referencing of fields is not allowed, with one
  // exception. OutputField with feature="transformedValue" in a PMML model can
  // refer to a DerivedField defined in LocalTransformations of that model in
  // addition to the other fields in its scope that are defined before it."
  void check_forward(const Ctx& ctx, pugi::xml_node node, const Binding& target) {
    if (!ctx.current || &target == ctx.current) return;  // a self-reference is a cycle (check_cycles)
    const bool same_container = target.container == ctx.current->container && target.scope == ctx.current->scope;
    const bool later = same_container && target.index > ctx.current->index;
    // Output precedes LocalTransformations in every model's content model
    // (pmml-4-4.xsd), so an Output reference to LocalTransformations is
    // forward unless it is the one exception.
    const bool output_to_local = ctx.current->container == Container::Output &&
                                 target.container == Container::LocalTransformations &&
                                 target.scope == ctx.current->scope && !ctx.transformed_value;
    if (later || output_to_local)
      warning(code::kForwardRef, ctx, node,
              std::string(origin_name(target.origin)) + " " + detail::quoted(target.name) +
                  " is declared after the reference (FieldScope.html: \"Forward referencing of fields is not "
                  "allowed\"; accepted, Q4/D12)");
  }

  void record_read(const Ctx& ctx, const Binding* target) {
    if (ctx.current) ctx.current->reads.push_back(target);
  }

  // FieldScope.html: "A DerivedField from the TransformationDictionary is
  // instantiated in a model element when it is referenced in the model
  // element, either directly or transitively through other DerivedFields
  // ... When they are instantiated inside a model element, DerivedFields
  // from the TransformationDictionary can only reference: MiningFields from
  // the MiningSchema of that model element [and] Other DerivedFields which
  // appear earlier in the TransformationDictionary."
  void instantiate(const Ctx& ctx, pugi::xml_node node, Scope& model, const Binding& td) {
    if (!model.td_instances.insert(&td).second) return;
    for (const Binding* r : td.reads) {
      if (r->origin == Origin::DerivedField) {
        instantiate(ctx, node, model, *r);
      } else if (r->origin == Origin::DataField && !model.mining.count(r->name)) {
        // Q3: an implicit pass-through, with a warning (B14).
        add(Severity::Warning, code::kScopeImplicit, &model, node,
            located(ctx, node,
                    "TransformationDictionary DerivedField " + detail::quoted(td.name) + " reads DataField " +
                        detail::quoted(r->name) +
                        ", which is not a MiningField of this model (FieldScope.html: an instantiated TD field "
                        "\"can only reference: MiningFields from the MiningSchema of that model element\"; "
                        "accepted, Q3)"));
      }
    }
  }

  // The binding `name` refers to from an expression or a model-internal
  // reference in ctx, nearest first (§3.2), or nullptr after a diagnostic.
  const Binding* resolve(const Ctx& ctx, pugi::xml_node node, std::string_view name, bool report_unresolved = true) {
    const std::string_view key = name;
    if (!ctx.model) {  // TransformationDictionary definitions
      if (auto it = document_->td_fields.find(key); it != document_->td_fields.end()) {
        check_forward(ctx, node, *it->second);
        record_read(ctx, it->second);
        return it->second;
      }
      if (auto it = document_->data_fields.find(key); it != document_->data_fields.end()) {
        record_read(ctx, it->second);
        return it->second;
      }
      error(code::kScopeUnresolved, ctx, node,
            "field " + detail::quoted(name) +
                " is neither a DataField nor a TransformationDictionary DerivedField (FieldScope.html: TD fields "
                "\"can only reference\" MiningFields and earlier TD DerivedFields)");
      return nullptr;
    }
    Scope& m = *ctx.model;
    // 1. M's OutputFields (only from inside M's Output)
    if (ctx.in_output)
      if (auto it = m.outputs.find(key); it != m.outputs.end()) {
        check_forward(ctx, node, *it->second);
        record_read(ctx, it->second);
        return it->second;
      }
    // 2. M's LocalTransformations
    if (auto it = m.local.find(key); it != m.local.end()) {
      check_forward(ctx, node, *it->second);
      record_read(ctx, it->second);
      return it->second;
    }
    // 3. TransformationDictionary DerivedFields, instantiated in M -- unless
    // M has a MiningField of that name: a sub-model whose MiningSchema
    // passes the parent's instance down reads it through the gatekeeper.
    if (auto it = document_->td_fields.find(key); it != document_->td_fields.end() && !m.mining.count(key)) {
      instantiate(ctx, node, m, *it->second);
      return it->second;
    }
    // 4. M's MiningFields
    if (auto it = m.mining.find(key); it != m.mining.end()) return it->second;
    // modelChain: "all Outputs from the segment models become part of the
    // enclosing mining model scope"
    if (auto it = m.chain_outputs.find(key); it != m.chain_outputs.end()) return it->second;
    // 5. not found: Q3's lenience, an implicit pass-through resolved in an
    // enclosing scope.
    for (Scope* s = m.enclosing; s; s = s->enclosing) {
      const Binding* found = nullptr;
      if (s->kind == Scope::Kind::Document) {
        if (auto it = s->data_fields.find(key); it != s->data_fields.end()) found = it->second;
      } else {
        for (const auto* map : {&s->local, &s->mining, &s->chain_outputs})
          if (auto it = map->find(key); !found && it != map->end()) found = it->second;
      }
      if (found) {
        warning(code::kScopeImplicit, ctx, node,
                "field " + detail::quoted(name) + " is not in this model's scope; it is a " +
                    std::string(origin_name(found->origin)) +
                    (s->kind == Scope::Kind::Document ? std::string(" of the document") : " of " + label_of(*s)) +
                    " (FieldScope.html: \"Within a model element, all data must flow in through its "
                    "MiningSchema\"; accepted as a pass-through, Q3)");
        return found;
      }
    }
    if (report_unresolved)
      error(code::kScopeUnresolved, ctx, node, "field " + detail::quoted(name) + " is not in scope");
    return nullptr;
  }

  // ---------------------------------------------------------------- cycles (§3.5)
  // "DerivedFields in the TransformationDictionary that have circular
  // references ... are not valid PMML"; the same holds in any container, as
  // a cycle has no evaluation order. Returns the dependency order.
  std::vector<const Binding*> check_cycles(const std::vector<Binding*>& container, Scope* model) {
    if (container.empty()) return {};
    std::unordered_map<const Binding*, int> colour;  // 0 white, 1 grey, 2 black
    std::unordered_set<const Binding*> members(container.begin(), container.end());
    std::unordered_set<const Binding*> reported;
    std::vector<const Binding*> order;
    std::vector<const Binding*> stack;
    std::function<void(const Binding*)> visit = [&](const Binding* b) {
      colour[b] = 1;
      stack.push_back(b);
      for (const Binding* r : b->reads) {
        if (!members.count(r)) continue;
        if (colour[r] == 1) {
          if (reported.insert(r).second) {
            std::string path;
            const auto from = std::find(stack.begin(), stack.end(), r);
            for (auto it = from; it != stack.end(); ++it) path += detail::quoted((*it)->name) + " -> ";
            path += detail::quoted(r->name);
            add(Severity::Error, code::kCycle, model, r->declaration,
                describe(r->declaration) + ": circular reference " + path +
                    " (FieldScope.html: \"circular references ... are not valid PMML\")");
          }
          continue;
        }
        if (colour[r] == 0) visit(r);
      }
      stack.pop_back();
      colour[b] = 2;
      order.push_back(b);
    };
    for (const Binding* b : container)
      if (colour[b] == 0) visit(b);
    return order;
  }

  // ---------------------------------------------------------------- DerivedField
  void bind_derived_body(Ctx& ctx, Binding& b) {
    b.state = Binding::State::Typing;
    const pugi::xml_node expr = expression_child(ctx, b.declaration);
    if (expr && mode_ == Mode::Load) {
      resolve_expression_names(ctx, expr);
    } else if (expr) {
      const Typed t = bind_expression(ctx, expr, std::nullopt);
      // §3.4: "DerivedField dataType=X -> Cast<X>(expression); widening
      // implicit, narrowing per §2.6".
      check_cast(ctx, b.declaration, t.type, b.type.type, "an expression");
    }
    bind_values(ctx, b.declaration, b, /*allow_property=*/false);
    b.state = Binding::State::Typed;
  }

  // Load mode: the names an expression references, resolved in ctx as
  // bind_expression() resolves them (so scopes, forward references, reads for
  // E-CYCLE and TD instantiation are the same), without its typing, whose
  // diagnostics are report-only until P2.2-P2.7 and which a DerivedField does
  // not need (its dataType is declared). Typing every Apply was a third of
  // the binder's time on a model with LocalTransformations.
  void resolve_expression_names(Ctx& ctx, pugi::xml_node node) {
    const std::string_view n = node.name();
    if (n == "Constant") return;
    if (n == "Apply") {
      for (pugi::xml_node c : node.children())
        if (c.type() == pugi::node_element && std::string_view(c.name()) != "Extension")
          resolve_expression_names(ctx, c);
      return;
    }
    if (n == "FieldRef" || n == "NormContinuous" || n == "NormDiscrete" || n == "Discretize") {
      if (const auto f = attr(node, "field")) resolve(ctx, node, *f);
      return;
    }
    if (n == "MapValues") {
      for (pugi::xml_node pair : node.children("FieldColumnPair"))
        if (const auto f = attr(pair, "field")) resolve(ctx, pair, *f);
      return;
    }
    if (n == "TextIndex") {
      if (const auto f = attr(node, "textField")) resolve(ctx, node, *f);
      for (pugi::xml_node c : node.children())
        if (c.type() == pugi::node_element && detail::is_expression(c.name())) {
          resolve_expression_names(ctx, c);  // the term (the first EXPRESSION)
          break;
        }
      return;
    }
    if (n == "Aggregate" || n == "Lag") {
      if (const auto f = attr(node, "field")) resolve(ctx, node, *f);
      if (const auto g = attr(node, "groupField")) resolve(ctx, node, *g);
      for (pugi::xml_node bi : node.children("BlockIndicator"))
        if (const auto f = attr(bi, "field")) resolve(ctx, bi, *f);
    }
  }

  // The one EXPRESSION child of a DerivedField or OutputField (after
  // Extension*, before Value* / Decisions).
  pugi::xml_node expression_child(const Ctx& ctx, pugi::xml_node node, bool required = true) {
    pugi::xml_node found;
    for (pugi::xml_node c : node.children()) {
      if (c.type() != pugi::node_element) continue;
      if (detail::is_expression(c.name())) {
        if (!found) found = c;
        continue;
      }
      if (detail::one_of(c.name(), {"Extension", "Value", "Decisions"})) continue;
      Ctx inner = ctx;
      note_unsupported(inner, c, std::string("not an EXPRESSION in a ") + node.name());
    }
    if (!found && required) error(code::kRequired, ctx, node, "an EXPRESSION is required (Transformations.html)");
    return found;
  }

  // An inline DerivedField (NeuralInput, NeuralOutput, BayesInput): no name in
  // scope, but the same typing.
  void bind_inline_derived(const Ctx& outer, pugi::xml_node df) {
    Ctx ctx = outer;
    ctx.current = nullptr;
    ctx.set_where(describe(df) + " in " + describe(df.parent()));
    const StaticType declared = declared_type(ctx, df, true);
    declared_optype(ctx, df, true);
    ctx.enclosing = declared;
    if (const pugi::xml_node expr = expression_child(ctx, df)) {
      const Typed t = bind_expression(ctx, expr, std::nullopt);
      check_cast(ctx, df, t.type, declared, "an expression");
    }
  }

  // ---------------------------------------------------------------- expressions (§3.4)
  Typed bind_expression(Ctx& ctx, pugi::xml_node node, StaticType param_type) {
    const std::string_view n = node.name();
    if (n == "Constant") return bind_constant(ctx, node, param_type);
    if (n == "FieldRef") return bind_fieldref(ctx, node);
    if (n == "NormContinuous") return bind_normcontinuous(ctx, node);
    if (n == "NormDiscrete") return bind_normdiscrete(ctx, node);
    if (n == "Discretize") return bind_discretize(ctx, node);
    if (n == "MapValues") return bind_mapvalues(ctx, node);
    if (n == "TextIndex") return bind_textindex(ctx, node);
    if (n == "Apply") return bind_apply(ctx, node);
    if (n == "Aggregate" || n == "Lag") {
      // §3.4: "Aggregate, Lag -> load error "unsupported", as today". Their
      // field names are still resolved.
      if (const auto f = attr(node, "field")) resolve(ctx, node, *f);
      if (const auto g = attr(node, "groupField")) resolve(ctx, node, *g);
      for (pugi::xml_node bi : node.children("BlockIndicator"))
        if (const auto f = attr(bi, "field")) resolve(ctx, bi, *f);
      note_unsupported(ctx, node, "unsupported, as in the engine (§3.4)");
      return {};
    }
    note_unsupported(ctx, node, "not an EXPRESSION");
    return {};
  }

  // Transformations.html, Constant (quoted in typing.h).
  Typed bind_constant(Ctx& ctx, pugi::xml_node node, StaticType param_type) {
    const std::string_view text = node.text().get();
    const auto missing = bool_attr(ctx, node, "missing");
    StaticType type;
    bool typed = false;
    if (attr(node, "dataType")) {
      type = declared_type(ctx, node, false);
      typed = true;
      if (!type) return {};
    } else if (param_type) {
      // "If the ParameterField definition includes a dataType, the Constant
      // will inherit the dataType specified in the ParameterField" (S2).
      type = param_type;
      typed = true;
    }
    // "If the missing attribute is true, the constant will be taken as a
    // missing value, regardless of the content": its text is not read, so
    // it is not checked, typed or not (an untyped one keeps the type its
    // text infers, as the engine's Constant does not read it either).
    const bool missing_value = missing && *missing;
    if (missing_value && typed) return {type};
    if (typed) {
      // "If the content is empty ... unless dataType is something besides
      // "string", in which case, it will be taken as a missing value of the
      // specified type." Empty is the xs:string content (pmml-4-4.xsd:
      // Constant extends xs:string, whiteSpace "preserve"), so blanks-only
      // text is not empty: it is read as a value of the type, whose
      // whiteSpace facet collapses it to "", not in that type's lexical
      // space (E-VALUE-MALFORMED; JPMML 1.7.7 also refuses it). The same
      // facet lets a typed " 1" read as 1, as it does for an untyped one
      // only through the string rule (typing.h infer_constant).
      if (text.empty() && *type != TypeId::String) return {type};
      if (attr(node, "dataType") && parse_datatype(*attr(node, "dataType"))->temporal) return {type};
      const types::Value v = parse_constant(*type, text, parse_ctx_);  // typing.h, shared with the engine
      if (types::is_invalid(v))
        error(code::kValueMalformed, ctx, node,
              "text " + detail::quoted(text) + " is not a value of type " + std::string(type_name(*type)) +
                  (attr(node, "dataType") ? std::string() : std::string(" (inherited from the parameter, S2)")));
      return {type};
    }
    switch (infer_constant(text)) {
      case ConstantKind::Integer:
        return {TypeId::Integer};
      case ConstantKind::IntegerOverflow:
        // §3.4: "integer beyond int64 -> load error" (D4).
        if (!missing_value)
          error(code::kValueMalformed, ctx, node,
                "untyped integer " + detail::quoted(text) + " is beyond the 64-bit range (§3.4, D4)");
        return {TypeId::Integer};
      case ConstantKind::Double:
        return {TypeId::Double};
      case ConstantKind::String:
        break;
    }
    return {TypeId::String};
  }

  Typed bind_fieldref(Ctx& ctx, pugi::xml_node node) {
    const auto f = attr(node, "field");
    if (!f) {
      error(code::kRequired, ctx, node, "attribute field is required (pmml-4-4.xsd)");
      return {};
    }
    const Binding* b = resolve(ctx, node, *f);
    if (!b) return {};
    // ctx.model is null exactly in a TransformationDictionary body.
    Typed t{read_type(*b, ctx.model == nullptr), b};
    // §3.4: "mapMissingTo (xs:string) parsed as T(field) by the rule above".
    if (const auto mm = attr(node, "mapMissingTo")) {
      const StaticType vt = document_value(ctx, node, b->type.type, *mm, "mapMissingTo", b);
      if (vt && t.type && *vt != *t.type) t.type = least_restrictive(t.type, vt).type;
    }
    return t;
  }

  // A field that must be numeric (NormContinuous, Discretize).
  void require_numeric(Ctx& ctx, pugi::xml_node node, const Binding* b, const char* why) {
    if (!b || !b->type.type || is_numeric(*b->type.type)) return;
    error(code::kTypeParam, ctx, node,
          "field " + detail::quoted(b->name) + " is " + std::string(type_name(*b->type.type)) + "; " + why);
  }

  Typed bind_normcontinuous(Ctx& ctx, pugi::xml_node node) {
    const auto f = attr(node, "field");
    if (!f)
      error(code::kRequired, ctx, node, "attribute field is required (pmml-4-4.xsd)");
    else
      require_numeric(ctx, node, resolve(ctx, node, *f),
                      "NormContinuous normalizes a numeric field (Transformations.html: \"piecewise linear "
                      "interpolation\")");
    number_attr(ctx, node, "mapMissingTo");
    enum_attr(ctx, node, "outliers", {"asIs", "asMissingValues", "asExtremeValues"});
    std::size_t count = 0;
    std::optional<double> previous;
    for (pugi::xml_node ln : node.children("LinearNorm")) {
      count++;
      const auto orig = number_attr(ctx, ln, "orig", true);
      number_attr(ctx, ln, "norm", true);
      // Transformations.html: "LinearNorm must be strictly sorted by
      // ascending value of orig."
      if (orig && previous && !(*orig > *previous))
        error(code::kNormOrder, ctx, ln,
              "orig is not above the previous LinearNorm's (Transformations.html: \"LinearNorm must be strictly "
              "sorted by ascending value of orig\")");
      if (orig) previous = orig;
    }
    // "The sequence must contain at least two elements."
    if (count < 2)
      error(code::kRequired, ctx, node,
            "has " + std::to_string(count) +
                " LinearNorm (Transformations.html: \"The sequence must contain at least two elements\")");
    return {TypeId::Double};
  }

  Typed bind_normdiscrete(Ctx& ctx, pugi::xml_node node) {
    const auto f = attr(node, "field");
    const Binding* b = nullptr;
    if (!f)
      error(code::kRequired, ctx, node, "attribute field is required (pmml-4-4.xsd)");
    else
      b = resolve(ctx, node, *f);
    // "An element (f, v) defines that the unit has value 1.0 if the value of
    // input field f is v" (B10): v is a value of f, read by §3.4.
    if (const auto v = attr(node, "value")) {
      if (b) document_value(ctx, node, b->type.type, *v, "value", b);
    } else {
      error(code::kRequired, ctx, node, "attribute value is required (pmml-4-4.xsd)");
    }
    number_attr(ctx, node, "mapMissingTo");
    return {TypeId::Double};
  }

  Typed bind_discretize(Ctx& ctx, pugi::xml_node node) {
    const auto f = attr(node, "field");
    if (!f)
      error(code::kRequired, ctx, node, "attribute field is required (pmml-4-4.xsd)");
    else
      require_numeric(ctx, node, resolve(ctx, node, *f),
                      "Discretize maps numerical input (Transformations.html: \"Discretization of numerical input "
                      "fields\")");
    // §3.4 / D21: its dataType, else the nearest enclosing declared type, the
    // one that applies to its position (typing.h untyped_value_type).
    const StaticType type =
        attr(node, "dataType") ? declared_type(ctx, node, false) : untyped_value_type(node, ctx.enclosing);
    if (const auto mm = attr(node, "mapMissingTo")) strict_value(ctx, node, type, *mm, "mapMissingTo");
    if (const auto dv = attr(node, "defaultValue")) strict_value(ctx, node, type, *dv, "defaultValue");
    for (pugi::xml_node bin : node.children("DiscretizeBin")) {
      if (const auto bv = attr(bin, "binValue"))
        strict_value(ctx, bin, type, *bv, "binValue");
      else
        error(code::kRequired, ctx, bin, "attribute binValue is required (pmml-4-4.xsd)");
      const pugi::xml_node interval = bin.child("Interval");
      if (interval)
        bind_interval(ctx, interval);
      else
        error(code::kRequired, ctx, bin, "an Interval is required (pmml-4-4.xsd)");
    }
    return {type};
  }

  Typed bind_mapvalues(Ctx& ctx, pugi::xml_node node) {
    const StaticType type =
        attr(node, "dataType") ? declared_type(ctx, node, false) : untyped_value_type(node, ctx.enclosing);
    const auto output = attr(node, "outputColumn");
    if (!output) error(code::kRequired, ctx, node, "attribute outputColumn is required (pmml-4-4.xsd)");
    if (const auto mm = attr(node, "mapMissingTo")) strict_value(ctx, node, type, *mm, "mapMissingTo");
    if (const auto dv = attr(node, "defaultValue")) strict_value(ctx, node, type, *dv, "defaultValue");
    struct Column {
      std::string column;
      const Binding* field;
    };
    std::vector<Column> keys;
    for (pugi::xml_node pair : node.children("FieldColumnPair")) {
      const auto f = attr(pair, "field");
      const auto c = attr(pair, "column");
      if (!f || !c) {
        error(code::kRequired, ctx, pair, "attributes field and column are required (pmml-4-4.xsd)");
        continue;
      }
      keys.push_back(Column{std::string(*c), resolve(ctx, pair, *f)});
    }
    if (node.child("TableLocator")) note_unsupported(ctx, node.child("TableLocator"), "external tables");
    const pugi::xml_node table = node.child("InlineTable");
    if (!table || !output) return {type};
    // Key cells by §3.4 for each FieldColumnPair's field; output cells as the
    // MapValues type. Then "it is an error if the table entries used for
    // matching are not unique" (B42), and a NaN key is a load error (D26).
    struct Row {
      std::vector<types::Value> key;
      pugi::xml_node node;
    };
    std::vector<Row> rows;
    std::vector<StaticType> key_types(keys.size());
    for (std::size_t k = 0; k < keys.size(); k++)
      key_types[k] = keys[k].field ? keys[k].field->type.type : std::nullopt;
    bool complete = true;
    for (pugi::xml_node row : table.children("row")) {
      Row r{{}, row};
      bool usable = true;
      for (std::size_t k = 0; k < keys.size(); k++) {
        const pugi::xml_node cell = row.child(keys[k].column.c_str());
        if (!cell) {
          // The row has no entry to match on (Transformations.html: "the
          // table entries used for matching").
          error(code::kRequired, ctx, row, "the row has no cell <" + keys[k].column + "> to match on");
          usable = false;
          continue;
        }
        types::Value v;
        const StaticType vt = document_value(ctx, cell, keys[k].field ? keys[k].field->type.type : std::nullopt,
                                             cell.text().get(), "cell <" + keys[k].column + ">", keys[k].field, &v);
        if (!vt) {
          usable = false;
          complete = false;
          continue;
        }
        if (key_types[k] && *vt != *key_types[k]) key_types[k] = least_restrictive(key_types[k], vt).type;
        r.key.push_back(v);
      }
      if (const pugi::xml_node out = row.child(std::string(*output).c_str()))
        strict_value(ctx, out, type, out.text().get(), "cell <" + std::string(*output) + ">");
      if (usable && r.key.size() == keys.size()) rows.push_back(std::move(r));
    }
    if (!complete) return {type};
    // Keys are of one type per column (the binder casts them at load, §2.7).
    std::string scratch;
    for (Row& r : rows)
      for (std::size_t k = 0; k < keys.size(); k++)
        if (key_types[k])
          r.key[k] = types::cold::cast(ColdPathIssuer::make(), *key_types[k], r.key[k], parse_ctx_, scratch);
    struct KeyVecHash {
      std::size_t operator()(const std::vector<types::Value>& v) const {
        std::size_t h = 0;
        for (const auto& x : v) h = h * 1000003u ^ types::KeyHash{}(x);
        return h;
      }
    };
    struct KeyVecEqual {
      bool operator()(const std::vector<types::Value>& a, const std::vector<types::Value>& b) const {
        for (std::size_t i = 0; i < a.size(); i++)
          if (!types::KeyEqual{}(a[i], b[i])) return false;
        return true;
      }
    };
    std::unordered_map<std::vector<types::Value>, pugi::xml_node, KeyVecHash, KeyVecEqual> seen;
    for (const Row& r : rows) {
      bool nan = false;
      for (const auto& v : r.key) {
        const auto* d = std::get_if<types::Double>(&v);
        const auto* f = std::get_if<types::Float>(&v);
        if ((d && std::isnan(d->v)) || (f && std::isnan(f->v))) nan = true;
      }
      if (nan) {
        error(code::kKeyNaN, ctx, r.node, "a NaN key matches no input (§2.7, D26: a NaN key is a load error)");
        continue;
      }
      const auto [it, inserted] = seen.emplace(r.key, r.node);
      if (!inserted)
        error(code::kKeyDuplicate, ctx, r.node,
              "the row's key equals the key of the row at line " + std::to_string(line_of(it->second)) +
                  " once typed (Transformations.html: \"it is an error if the table entries used for matching are "
                  "not unique\", B42)");
    }
    return {type};
  }

  Typed bind_textindex(Ctx& ctx, pugi::xml_node node) {
    const auto f = attr(node, "textField");
    if (!f) {
      error(code::kRequired, ctx, node, "attribute textField is required (pmml-4-4.xsd)");
    } else if (const Binding* b = resolve(ctx, node, *f); b && b->type.type && *b->type.type != TypeId::String) {
      // §3.4: "textField must be string".
      error(code::kTypeParam, ctx, node,
            "textField " + detail::quoted(*f) + " is " + std::string(type_name(*b->type.type)) +
                "; TextIndex indexes text (§3.4)");
    }
    enum_attr(ctx, node, "localTermWeights",
              {"termFrequency", "binary", "logarithmic", "augmentedNormalizedTermFrequency"});
    enum_attr(ctx, node, "countHits", {"allHits", "bestHits"});
    bool_attr(ctx, node, "isCaseSensitive");
    int_attr(ctx, node, "maxLevenshteinDistance");
    pugi::xml_node term;
    for (pugi::xml_node c : node.children()) {
      if (c.type() != pugi::node_element) continue;
      if (detail::is_expression(c.name())) {
        if (!term) term = c;
      } else if (!detail::one_of(c.name(), {"Extension", "TextIndexNormalization"})) {
        note_unsupported(ctx, c, "not TextIndex content");
      }
    }
    if (term)
      bind_expression(ctx, term, TypeId::String);
    else
      error(code::kRequired, ctx, node, "the term EXPRESSION is required (Transformations.html)");
    return {TypeId::Double};
  }

  // ---------------------------------------------------------------- Apply (§4)
  // Q6: ordering over an operand whose field defines no order.
  void check_order_operand(Ctx& ctx, pugi::xml_node node, const Binding* field, const std::string& what) {
    if (!field || !field->type.optype) return;
    if (*field->type.optype == OpType::Categorical) {
      // DataDictionary.html: "Categorical field values can only be tested
      // for equality" (Q6: a warning).
      warning(code::kOrderCategorical, ctx, node,
              what + " orders categorical field " + detail::quoted(field->name) +
                  " (DataDictionary.html: \"Categorical field values can only be tested for equality\"; accepted, "
                  "Q6)");
    } else if (*field->type.optype == OpType::Ordinal && field->type.type == TypeId::String &&
               !field->type.has_values) {
      // DataDictionary.html: "Ordinal DataFields of type string must always
      // specify the order explicitly via respective Value elements".
      warning(code::kOrdinalNoValues, ctx, node,
              what + " orders ordinal string field " + detail::quoted(field->name) +
                  " that has no Value elements (DataDictionary.html: \"Ordinal DataFields of type string must "
                  "always specify the order explicitly\"; code-unit order, Q6)");
    }
  }

  // §2.7: may values of these types be compared for this purpose?
  void check_comparable(Ctx& ctx, pugi::xml_node node, bool ordering, const Typed& a, const Typed& b,
                        const std::string& what) {
    if (!a.type || !b.type) return;
    const TypeId x = *a.type, y = *b.type;
    if ((x == TypeId::Boolean) != (y == TypeId::Boolean)) {
      // Functions.html: "boolean dataTypes can not be mixed with other types
      // in a defined function without explicitly defining the expected
      // output dataType" (Q7: load error).
      error(code::kTypeBooleanMixed, ctx, node,
            what + " compares " + std::string(type_name(x)) + " with " + std::string(type_name(y)) +
                " (Functions.html: boolean \"can not be mixed with other types\", Q7)");
      return;
    }
    if (ordering && order_mixed(x, y)) {
      // §2.7: string x number ordering is a load error (P2.7; typing.h
      // order_mixed, raised at load by the Apply constructor).
      error(code::kTypeOrderMixed, ctx, node,
            what + " orders " + std::string(type_name(x)) + " against " + std::string(type_name(y)) +
                " (§2.7; DataDictionary.html: \"Categorical field values can only be tested for equality\")");
      return;
    }
    if (ordering) {
      check_order_operand(ctx, node, a.field, what);
      check_order_operand(ctx, node, b.field, what);
    }
  }

  Typed bind_apply(Ctx& ctx, pugi::xml_node node) {
    const auto function = attr(node, "function");
    // "children after Extension*" (B25).
    std::vector<pugi::xml_node> args;
    for (pugi::xml_node c : node.children()) {
      if (c.type() != pugi::node_element || std::string_view(c.name()) == "Extension") continue;
      args.push_back(c);
    }
    enum_attr(ctx, node, "invalidValueTreatment", {"returnInvalid", "asIs", "asMissing", "asValue"});
    if (attr(node, "invalidValueTreatment") && *attr(node, "invalidValueTreatment") == "asValue")
      // Functions.html describes returnInvalid, asMissing and asIs for an
      // Apply; asValue needs an invalidValueReplacement, which Apply has not
      // (§4.3: "asValue on an Apply stays a load error").
      error(code::kForbidden, ctx, node,
            "invalidValueTreatment=\"asValue\" needs a replacement value, which Apply does not have (§4.3)");
    if (!function) {
      error(code::kRequired, ctx, node, "attribute function is required (pmml-4-4.xsd)");
      for (auto a : args) bind_expression(ctx, a, std::nullopt);
      return {};
    }
    const fn::Signature* sig = fn::find(*function);
    if (!sig) {
      for (auto a : args) bind_expression(ctx, a, std::nullopt);
      if (auto it = user_functions_.find(std::string(*function)); it != user_functions_.end()) {
        unsupported(code::kUnsupportedUserFunction, ctx, node,
                    "applies DefineFunction " + detail::quoted(*function) + ", out of scope (§3.4)");
        return {attr(it->second, "dataType") ? declared_type(ctx, it->second, false) : std::nullopt};
      }
      const std::string_view hint = fn::hint_for(*function);
      // FieldScope.html: DerivedFields that "apply undefined functions are
      // not valid PMML".
      error(code::kFunctionUnknown, ctx, node,
            "function " + detail::quoted(*function) + " is not a PMML built-in and no DefineFunction defines it" +
                (hint.empty() ? std::string() : "; " + std::string(hint)));
      return {};
    }
    // Arity (§4.2).
    const std::size_t n = args.size();
    if (!arity_fits(*sig, n)) error(code::kArity, ctx, node, arity_mismatch(*sig, n));
    std::vector<Typed> types(n);
    bool temporal = false;
    for (std::size_t i = 0; i < n; i++) {
      const fn::Param p = sig->param(i);
      types[i] = bind_expression(ctx, args[i], parameter_type(*sig, i));  // S2
      check_param(ctx, node, *sig, i, p, args[i], types[i], temporal);
    }
    if (temporal)
      unsupported(code::kUnsupportedFunction, ctx, node,
                  std::string(sig->name) + " takes a date or time, carried as double until P10 (Q8)");
    // Comparisons (§2.7).
    const std::string_view name = sig->name;
    if (sig->params[0] == fn::Param::Comparable && n >= 2) {
      const std::string what = "Apply " + detail::quoted(name);
      const bool ordering = is_ordering(name);
      for (std::size_t i = 1; i < n; i++) check_comparable(ctx, node, ordering, types[0], types[i], what);
    }
    // Result (§4.1, typing.h apply_result_type, shared with the engine's load).
    std::vector<StaticType> arg_types(n);
    for (std::size_t i = 0; i < n; i++) arg_types[i] = types[i].type;
    const ApplyType applied = apply_result_type(*sig, arg_types);
    if (applied.boolean_mixed)
      error(code::kTypeBooleanMixed, ctx, node,
            "THEN is " + type_name(types[1].type) + " and ELSE is " + type_name(types[2].type) +
                ": no common type (Functions.html: boolean \"can not be mixed with other types\", Q7; D7)");
    const StaticType result = applied.type;
    // §3.4: "mapMissingTo, defaultValue (xs:string) parsed as the Apply's
    // result type".
    if (const auto mm = attr(node, "mapMissingTo")) strict_value(ctx, node, result, *mm, "mapMissingTo");
    if (const auto dv = attr(node, "defaultValue")) strict_value(ctx, node, result, *dv, "defaultValue");
    return {result};
  }

  void check_param(Ctx& ctx, pugi::xml_node apply, const fn::Signature& sig, std::size_t i, fn::Param p,
                   pugi::xml_node arg, const Typed& t, bool& temporal) {
    // Built only for a message (the binder runs at every load).
    const auto what = [&] {
      return "argument " + std::to_string(i + 1) + " of " + std::string(sig.name) + " is " + type_name(t.type);
    };
    if (p == fn::Param::Temporal) {
      temporal = true;
      return;
    }
    if (p == fn::Param::FieldRef) {
      // isValid/isNotValid: "ParameterField name=\"field\"" -- the validity
      // of a field value (§4.2 "field reference").
      if (std::string_view(arg.name()) != "FieldRef")
        error(code::kTypeParam, ctx, apply,
              "argument " + std::to_string(i + 1) + " of " + std::string(sig.name) +
                  " is not a FieldRef; it tests the validity of a field (BuiltinFunctions.html)");
      return;
    }
    if (!t.type || p == fn::Param::Any || p == fn::Param::Comparable) return;
    const TypeId x = *t.type;
    const bool boolean = x == TypeId::Boolean;
    switch (p) {
      case fn::Param::Numeric:
      case fn::Param::Double:
        if (boolean)
          error(code::kTypeBooleanMixed, ctx, apply,
                what() +
                    ", where a number is expected (Functions.html: boolean \"can not be mixed with other types\", "
                    "Q7)");
        else if (!is_numeric(x))
          // §2.5: "no arithmetic accepts" string.
          error(code::kTypeParam, ctx, apply, what() + ", where a number is expected (§2.5)");
        return;
      case fn::Param::Integer:
        if (boolean)
          error(code::kTypeBooleanMixed, ctx, apply,
                what() + ", where an integer is expected (Functions.html: boolean \"can not be mixed\", Q7)");
        else if (x != TypeId::Integer)
          // The pseudo-declaration declares dataType="integer"; Functions.html:
          // "a double expression can not be implicitly cast as an integer".
          error(code::kTypeParam, ctx, apply,
                what() +
                    ", where the parameter is declared integer (Functions.html: \"a double expression can not "
                    "be implicitly cast as an integer\")");
        return;
      case fn::Param::String:
        // Numbers cast to string toward less restriction ("string double
        // float integer"); boolean is outside the list.
        if (boolean)
          error(code::kTypeBooleanMixed, ctx, apply,
                what() + ", where a string is expected (Functions.html: boolean \"can not be mixed\", Q7)");
        return;
      case fn::Param::Boolean:
        // For if's condition the engine's load raises the same error
        // (load_scope.h if_condition_message, design item P4.3c, B53).
        if (!boolean)
          error(code::kTypeBooleanMixed, ctx, apply,
                what() +
                    ", where the parameter is declared boolean (Functions.html: boolean \"can not be mixed with "
                    "other types\", Q7)");
        return;
      default:
        return;
    }
  }

  // ---------------------------------------------------------------- predicates
  void bind_predicate(Ctx& ctx, pugi::xml_node node) {
    const std::string_view n = node.name();
    if (n == "True" || n == "False") return;
    if (n == "SimplePredicate") return bind_simple_predicate(ctx, node);
    if (n == "SimpleSetPredicate") return bind_set_predicate(ctx, node);
    if (n == "CompoundPredicate") {
      enum_attr(ctx, node, "booleanOperator", {"or", "and", "xor", "surrogate"}, true);
      std::size_t count = 0;
      for (pugi::xml_node c : node.children()) {
        if (c.type() != pugi::node_element || std::string_view(c.name()) == "Extension") continue;
        count++;
        bind_predicate(ctx, c);
      }
      // pmml-4-4.xsd: CompoundPredicate holds PREDICATE minOccurs="2".
      if (count < 2)
        error(code::kRequired, ctx, node,
              "has " + std::to_string(count) + " predicate" + (count == 1 ? "" : "s") +
                  "; at least two are required (pmml-4-4.xsd)");
      return;
    }
    note_unsupported(ctx, node, "not a PREDICATE");
  }

  // The PREDICATE child of a Node, Segment, rule or Scorecard Attribute.
  void bind_predicate_child(Ctx& ctx, pugi::xml_node owner) {
    for (pugi::xml_node c : owner.children())
      if (c.type() == pugi::node_element && detail::is_predicate(c.name())) return bind_predicate(ctx, c);
    error(code::kRequired, ctx, owner, "a PREDICATE is required (pmml-4-4.xsd)");
  }

  void bind_simple_predicate(Ctx& ctx, pugi::xml_node node) {
    const auto f = attr(node, "field");
    const Binding* b = nullptr;
    if (!f)
      error(code::kRequired, ctx, node, "attribute field is required (pmml-4-4.xsd)");
    else
      b = resolve(ctx, node, *f);
    const auto op = enum_attr(
        ctx, node, "operator",
        {"equal", "notEqual", "lessThan", "lessOrEqual", "greaterThan", "greaterOrEqual", "isMissing", "isNotMissing"},
        true);
    if (!op) return;
    const auto value = attr(node, "value");
    // TreeModel.html: "If the operator is isMissing or isNotMissing, the
    // attribute value must not appear. With all other operators, however,
    // the attribute value is required."
    if (*op == "isMissing" || *op == "isNotMissing") {
      if (value)
        error(code::kForbidden, ctx, node,
              "attribute value must not appear with operator " + detail::quoted(*op) + " (TreeModel.html)");
      return;
    }
    if (!value) {
      error(code::kRequired, ctx, node,
            "attribute value is required with operator " + detail::quoted(*op) + " (TreeModel.html)");
      return;
    }
    if (!b) return;
    // §3.4: "value read by the rule above"; the comparison then follows §2.7.
    document_value(ctx, node, b->type.type, *value, "value", b);
    if (detail::one_of(*op, {"lessThan", "lessOrEqual", "greaterThan", "greaterOrEqual"}))
      check_order_operand(ctx, node, b, "operator " + detail::quoted(*op));
  }

  void bind_set_predicate(Ctx& ctx, pugi::xml_node node) {
    const auto f = attr(node, "field");
    const Binding* b = nullptr;
    if (!f)
      error(code::kRequired, ctx, node, "attribute field is required (pmml-4-4.xsd)");
    else
      b = resolve(ctx, node, *f);
    enum_attr(ctx, node, "booleanOperator", {"isIn", "isNotIn"}, true);
    const pugi::xml_node array = node.child("Array");
    if (!array) {
      error(code::kRequired, ctx, node, "an Array is required (pmml-4-4.xsd)");
      return;
    }
    enum_attr(ctx, array, "type", {"int", "real", "string"}, true);
    if (!b || !b->type.type) return;
    // §3.4: "Array members read by the rule above"; a NaN member can match
    // nothing (§2.7, D26).
    std::size_t index = 0;
    for (const std::string& member : split_array(array.text().get())) {
      index++;
      types::Value v;
      const StaticType vt =
          document_value(ctx, array, b->type.type, member, "Array member " + std::to_string(index), b, &v);
      if (!vt) continue;
      const auto* d = std::get_if<types::Double>(&v);
      const auto* fl = std::get_if<types::Float>(&v);
      if ((d && std::isnan(d->v)) || (fl && std::isnan(fl->v)))
        error(code::kKeyNaN, ctx, array,
              "Array member " + std::to_string(index) + " is NaN, which matches no input (§2.7, D26)");
    }
  }

  // ---------------------------------------------------------------- models
  static std::string model_label(pugi::xml_node model, const std::string& prefix) {
    std::string s = prefix.empty() ? std::string() : prefix + " > ";
    s += model.name();
    if (const auto name = attr(model, "modelName")) s += " " + detail::quoted(*name);
    return s;
  }

  // "Segment \"3\"", or "Segment #k" (the k-th Segment) without an id.
  static std::string segment_label(pugi::xml_node segment) {
    if (const auto id = attr(segment, "id")) return "Segment " + detail::quoted(*id);
    std::size_t k = 0;
    for (pugi::xml_node s = segment; s; s = s.previous_sibling("Segment")) k++;
    return "Segment #" + std::to_string(k);
  }

  // A scope's label (Scope::label), made on first use in Load mode.
  const std::string& label_of(const Scope& s) const {
    if (!s.label.empty() || s.kind == Scope::Kind::Document) return s.label;
    std::string prefix = s.enclosing && s.enclosing->kind == Scope::Kind::Model ? label_of(*s.enclosing) : "";
    if (const pugi::xml_node parent = s.element.parent(); std::string_view(parent.name()) == "Segment")
      prefix += " > " + segment_label(parent);
    s.label = model_label(s.element, prefix);
    return s.label;
  }

  // Returns the index of the ModelBinding.
  std::size_t bind_model(pugi::xml_node node, Scope& parent, Scope* parent_model, const std::string& prefix) {
    Scope& m =
        new_scope(Scope::Kind::Model, &parent, mode_ == Mode::Report ? model_label(node, prefix) : std::string(), node);
    element_scopes_.emplace(node.internal_object(), &m);
    const std::size_t index = models_.size();
    models_.push_back(ModelBinding{m.id, node.name(), {}, {}, {}});
    Ctx ctx;
    ctx.model = &m;
    enum_attr(ctx, node, "functionName",
              {"associationRules", "sequences", "classification", "regression", "clustering", "timeSeries", "mixed"},
              true);

    // MiningSchema.html: "All data entering a model must pass through the
    // MiningSchema."
    const pugi::xml_node schema = node.child("MiningSchema");
    if (schema)
      bind_mining_schema(ctx, schema, m, parent_model);
    else
      error(code::kRequired, ctx, node, "a MiningSchema is required (pmml-4-4.xsd)");

    // Phase 1 of LocalTransformations and Output (§3.5).
    std::vector<Binding*> locals, outputs;
    const pugi::xml_node lt = node.child("LocalTransformations");
    if (lt) declare_locals(ctx, lt, m, locals);
    const pugi::xml_node output = node.child("Output");
    if (output) declare_outputs(ctx, output, m, outputs);

    // Phase 2 of LocalTransformations.
    for (Binding* b : locals) {
      Ctx body = ctx;
      body.current = b;
      body.set_where(b->declaration);
      body.enclosing = b->type.type;
      bind_derived_body(body, *b);
    }
    models_[index].derived_order = check_cycles(locals, &m);

    if (const pugi::xml_node targets = node.child("Targets")) bind_targets(ctx, targets, m);
    if (!m.target)
      for (const auto& [name, b] : m.mining)
        if (b->target_usage && (!m.target || b->index < m.target->index)) m.target = b;

    // The model element's own field references.
    resolve_field_attributes(ctx, node);

    // The body.
    for (pugi::xml_node child : node.children()) {
      if (child.type() != pugi::node_element) continue;
      const std::string_view n = child.name();
      if (detail::one_of(n, {"MiningSchema", "LocalTransformations", "Output", "Targets", "Extension"})) continue;
      if (detail::one_of(n, {"ModelStats", "ModelExplanation", "ModelVerification"})) {
        note_unsupported(ctx, child, "statistics and verification data, not used in scoring");
        continue;
      }
      if (n == "Segmentation") {
        bind_segmentation(ctx, child, m, index);
        continue;
      }
      if (detail::is_model_element(n)) {
        // AnomalyDetectionModel: "whose inner model is a child scope" (§5.2).
        const std::size_t c = bind_model(child, m, &m, mode_ == Mode::Report ? m.label : std::string());
        models_[index].children.push_back(c);
        continue;
      }
      if (mode_ == Mode::Load) continue;  // the engine resolves the body's names (lookup())
      visit_body(ctx, child);
    }

    // Phase 2 of Output.
    for (Binding* b : outputs) bind_output_body(ctx, *b, m);
    models_[index].output_order = check_cycles(outputs, &m);
    flush_unsupported(m);
    return index;
  }

  void bind_mining_schema(Ctx& ctx, pugi::xml_node schema, Scope& m, Scope* parent_model) {
    std::uint32_t index = 0;
    for (pugi::xml_node mf : schema.children()) {
      if (mf.type() != pugi::node_element) continue;
      if (std::string_view(mf.name()) == "Extension") continue;
      if (std::string_view(mf.name()) != "MiningField") {
        note_unsupported(ctx, mf, "MiningSchema content other than MiningField");
        continue;
      }
      Ctx fctx = ctx;
      fctx.set_where(mf);
      const auto name = attr(mf, "name");
      if (!name) {
        error(code::kRequired, fctx, mf, "attribute name is required (pmml-4-4.xsd)");
        continue;
      }
      const std::string_view key = *name;
      // FieldScope.html: "The names of MiningFields cannot be re-used within
      // the same model element."
      if (m.mining.count(key)) {
        error(code::kNameDuplicate, fctx, mf,
              "MiningField " + detail::quoted(*name) +
                  " appears twice (FieldScope.html: \"The names of MiningFields cannot be re-used within the same "
                  "model element\")");
        continue;
      }
      const Binding* source = resolve_mining_source(fctx, mf, key, parent_model);
      FieldType type = source ? source->type : FieldType{};
      // The attributes present, in one pass: every reader below is skipped for
      // an absent attribute, which it would only have looked for (a
      // MiningField has a dozen optional attributes and usually one or two).
      enum : unsigned {
        kOptype = 1,
        kInvalid = 2,
        kMissingTreatment = 4,
        kOutliers = 8,
        kImportance = 16,
        kMissingReplacement = 32,
        kInvalidReplacement = 64,
        kLow = 128,
        kHigh = 256,
        kOther = 512
      };
      unsigned present = 0;
      for (pugi::xml_attribute a = mf.first_attribute(); a; a = a.next_attribute()) {
        const std::string_view an = a.name();
        if (an == "name" || an == "usageType") continue;
        present |= an == "optype"                    ? kOptype
                   : an == "invalidValueTreatment"   ? kInvalid
                   : an == "missingValueTreatment"   ? kMissingTreatment
                   : an == "outliers"                ? kOutliers
                   : an == "importance"              ? kImportance
                   : an == "missingValueReplacement" ? kMissingReplacement
                   : an == "invalidValueReplacement" ? kInvalidReplacement
                   : an == "lowValue"                ? kLow
                   : an == "highValue"               ? kHigh
                                                     : kOther;
      }
      const bool plain = present == 0;  // no attribute but name and usageType
      if (plain) {
        // The MiningFields of a typical forest: no treatment and nothing else
        // to check -- the source's type, and an alias of its slot (§3.2
        // "Aliasing"). The general path below reads every attribute.
        const auto usage = enum_attr(
            fctx, mf, "usageType",
            {"active", "predicted", "target", "supplementary", "group", "order", "frequencyWeight", "analysisWeight"});
        const bool target_usage = usage && (*usage == "target" || *usage == "predicted");
        if (mode_ == Mode::Load && source && !target_usage) {
          // Load mode keeps no FrameLayout of its own (the engine's slots are
          // the Indexer's until P7): a pass-through is its source.
          m.mining.emplace(key, const_cast<Binding*>(source));
          index++;
          continue;
        }
        Binding& b = declare(std::string(key), Origin::MiningField, Container::MiningSchema, type, m, mf, index++);
        b.source = source;
        b.state = Binding::State::Typed;
        b.target_usage = usage && (*usage == "target" || *usage == "predicted");
        m.mining.emplace(key, &b);
        if (source) layout_.slots[b.slot].alias_of = source->slot;
        continue;
      }
      const auto usage = enum_attr(
          fctx, mf, "usageType",
          {"active", "predicted", "target", "supplementary", "group", "order", "frequencyWeight", "analysisWeight"});
      const auto optype = present & kOptype ? declared_optype(fctx, mf, false) : std::nullopt;
      if (optype) type.optype = optype;
      const auto invalid = present & kInvalid ? enum_attr(fctx, mf, "invalidValueTreatment",
                                                          {"returnInvalid", "asIs", "asMissing", "asValue"})
                                              : std::nullopt;
      if (present & kMissingTreatment)
        enum_attr(fctx, mf, "missingValueTreatment",
                  {"asIs", "asMean", "asMode", "asMedian", "asValue", "returnInvalid"});
      const auto outliers = present & kOutliers
                                ? enum_attr(fctx, mf, "outliers", {"asIs", "asMissingValues", "asExtremeValues"})
                                : std::nullopt;
      if (present & kImportance) number_attr(fctx, mf, "importance");
      // MiningSchema.html: "asValue specifies that an invalid input value
      // should be replaced with the value specified by attribute
      // invalidValueReplacement which must be present in this case, or the
      // PMML is invalid."
      if (invalid && *invalid == "asValue" && !(present & kInvalidReplacement))
        error(code::kRequired, fctx, mf,
              "invalidValueTreatment=\"asValue\" without invalidValueReplacement (MiningSchema.html: \"which must be "
              "present in this case, or the PMML is invalid\")");
      // §3.4: replacements are read by the K1 rule; a value that is not of the
      // field's type widens the slot (N3, W-SLOT-WIDENED).
      const StaticType field_type = type.type;
      for (const char* rep : {"missingValueReplacement", "invalidValueReplacement"}) {
        if (!(present & (kMissingReplacement | kInvalidReplacement))) break;
        const auto text = attr(mf, rep);
        if (!text || !field_type) continue;
        const DocumentValue v = read_document_text(*field_type, *text, parse_ctx_);
        if (v.rule == TextRule::Malformed) {
          error(code::kValueMalformed, fctx, mf,
                std::string(rep) + "=" + detail::quoted(*text) + " is not a value of " +
                    std::string(type_name(*field_type)) + " field " + detail::quoted(*name) + " (§3.4)");
        } else if (v.rule == TextRule::AsDouble) {
          type.type = least_restrictive(type.type, TypeId::Double).type;
          warning(code::kSlotWidened, fctx, mf,
                  std::string(rep) + "=" + detail::quoted(*text) + " is not an integer: the slot of integer field " +
                      detail::quoted(*name) + " is widened to double (§3.4, N3, D22)");
        }
      }
      const auto low = present & kLow ? number_attr(fctx, mf, "lowValue") : std::nullopt;
      const auto high = present & kHigh ? number_attr(fctx, mf, "highValue") : std::nullopt;
      if (outliers && *outliers == "asExtremeValues" && field_type == TypeId::Integer) {
        for (const auto& bound : {low, high})
          if (bound && std::trunc(*bound) != *bound) {
            type.type = TypeId::Double;
            warning(code::kSlotWidened, fctx, mf,
                    "an asExtremeValues bound is not an integer: the slot of integer field " + detail::quoted(*name) +
                        " is widened to double (§3.4, N3, D22)");
            break;
          }
      }
      const bool target_usage = usage && (*usage == "target" || *usage == "predicted");
      if (mode_ == Mode::Load && source && !target_usage && type.type == source->type.type &&
          type.optype == source->type.optype) {
        // Load mode: the field reads its source with the source's type (its
        // treatments are the engine's MiningField's): a pass-through, as above.
        // A MiningField whose optype differs from its source's is not one:
        // MiningSchema.html, optype: "overrides the corresponding value in
        // the DataField", and the model's ordinal comparisons read it
        // (bind::ordinal_values, P6.3).
        m.mining.emplace(key, const_cast<Binding*>(source));
        index++;
        continue;
      }
      Binding& b = declare(std::string(key), Origin::MiningField, Container::MiningSchema, type, m, mf, index++);
      b.source = source;
      b.state = Binding::State::Typed;
      b.target_usage = target_usage;
      m.mining.emplace(key, &b);
      // §3.2 "Aliasing": no treatment attributes -> an alias of the parent's
      // slot.
      const bool treats = (present & (kMissingReplacement | kInvalidReplacement | kLow | kHigh)) ||
                          (invalid && *invalid != "returnInvalid") || (outliers && *outliers != "asIs") || optype;
      if (source && !treats) layout_.slots[b.slot].alias_of = source->slot;
    }
  }

  // MiningSchema.html: "name: symbolic name of field, must refer to a field
  // in the scope of the parent of the MiningSchema's model element".
  const Binding* resolve_mining_source(Ctx& ctx, pugi::xml_node mf, std::string_view key, Scope* parent_model) {
    if (!parent_model) {
      // FieldScope.html: "For top-level Models, their MiningSchema can refer
      // only to DataFields in the DataDictionary".
      if (auto it = document_->data_fields.find(key); it != document_->data_fields.end()) return it->second;
      if (document_->td_fields.count(key))
        error(code::kScopeTopLevel, ctx, mf,
              "names TransformationDictionary DerivedField " + detail::quoted(key) +
                  " (FieldScope.html: \"For top-level Models, their MiningSchema can refer only to DataFields in "
                  "the DataDictionary\")");
      else
        error(code::kScopeUnresolved, ctx, mf, "names no DataField " + detail::quoted(key));
      return nullptr;
    }
    // FieldScope.html: "For sub-models, their MiningSchema can refer to any
    // fields defined within the parent model's scope, including The
    // MiningFields of the parent model, DerivedFields defined in the
    // LocalTransformations of the parent model ... OutputFields defined in
    // Segments that appear above/earlier in the parent MiningModel".
    Ctx parent_ctx;
    parent_ctx.model = parent_model;
    parent_ctx.where_node = ctx.where_node;
    parent_ctx.where_text = ctx.where_text;
    const Binding* b = resolve(parent_ctx, mf, key, /*report_unresolved=*/false);
    if (b) return b;
    // Not in the parent's scope. When the name is a LocalTransformations
    // DerivedField of another model (an earlier segment's:
    // fix_ensemble_localtrans_scope), the engine's flat name space reaches it
    // today and the document loads; P2.1 keeps it loading (W-SCOPE-FLAT)
    // until P7 gives each scope its slots (B14, B16). Anything else is
    // E-SCOPE-UNRESOLVED.
    const auto declared = std::find_if(bindings_.begin(), bindings_.end(), [&key](const Binding& d) {
      return d.name == key && d.container == Container::LocalTransformations;
    });
    if (declared != bindings_.end()) {
      warning(code::kScopeFlat, parent_ctx, mf,
              "field " + detail::quoted(key) + " is not in the parent model's scope; it is a " +
                  std::string(origin_name(declared->origin)) + " of " +
                  (scopes_[declared->scope].kind == Scope::Kind::Document ? std::string("the document")
                                                                          : label_of(scopes_[declared->scope])) +
                  " (MiningSchema.html: \"must refer to a field in the scope of the parent of the MiningSchema's model "
                  "element\"; accepted until P7, B14)");
      return &*declared;  // what the flat name space reads: its slot and type
    }
    error(code::kScopeUnresolved, parent_ctx, mf, "field " + detail::quoted(key) + " is not in scope");
    return nullptr;
  }

  void declare_locals(Ctx& ctx, pugi::xml_node lt, Scope& m, std::vector<Binding*>& out) {
    std::uint32_t index = 0;
    for (pugi::xml_node df : lt.children()) {
      if (df.type() != pugi::node_element || std::string_view(df.name()) == "Extension") continue;
      if (std::string_view(df.name()) != "DerivedField") {
        note_unsupported(ctx, df, "LocalTransformations content other than DerivedField");
        continue;
      }
      Ctx fctx = ctx;
      fctx.set_where(df);
      const auto name = attr(df, "name");
      if (!name) {
        error(code::kRequired, fctx, df, "attribute name is required in LocalTransformations (Transformations.html)");
        continue;
      }
      FieldType type{declared_type(fctx, df, true), declared_optype(fctx, df, true), false};
      const std::string key(*name);
      // FieldScope.html: "The names of DerivedFields in the
      // LocalTransformations must be unique from any other names in their
      // scope."
      if (const std::string clash = name_clash(m, key, /*outputs=*/false); !clash.empty()) {
        if (shadow_warning(fctx, df, m, key, "LocalTransformations DerivedField", /*td_too=*/true)) {
          m.local_shadows.emplace_back(key, type.type);
          continue;
        }
        error(code::kNameDuplicate, fctx, df,
              "name " + detail::quoted(*name) + " is already " + clash +
                  " (FieldScope.html: LocalTransformations names \"must be unique from any other names in their "
                  "scope\")");
        continue;
      }
      warn_shadowed_datafield(fctx, df, key, "LocalTransformations DerivedField");
      Binding& b = declare(key, Origin::DerivedField, Container::LocalTransformations, type, m, df, index++);
      m.local.emplace(key, &b);
      out.push_back(&b);
    }
  }

  // The two name clashes the engine scores and which are therefore warnings,
  // not E-NAME-DUPLICATE, now that the binder decides at load (P2.1). The
  // second declaration is not declared: lookups find the first, which is the
  // slot type today's Indexer keeps (first declaration wins).
  //   - a LocalTransformations DerivedField or OutputField named like a
  //     MiningField of the same model. FieldScope.html: "The names of
  //     DerivedFields in the LocalTransformations must be unique from any
  //     other names in their scope", "The names of OutputFields must be unique
  //     from any other names in their scope"; but real exporters write it
  //     (AuditTree, IrisTree, IrisMultinomReg) and JPMML scores those
  //     documents: decided 2026-11-18 (design §9).
  //   - a LocalTransformations DerivedField named like a TransformationDictionary
  //     DerivedField (fix_derivedfield_shadow, fix_ensemble_localtrans_shadow):
  //     the same FieldScope.html rule, kept as a warning because documents
  //     that load today must keep loading (P2.1; a decision for the user).
  bool shadow_warning(const Ctx& ctx, pugi::xml_node node, const Scope& m, const std::string& key,
                      std::string_view what, bool td_too) {
    if (m.mining.count(key)) {
      warning(code::kNameShadowsMiningField, ctx, node,
              std::string(what) + " name " + detail::quoted(key) +
                  " is already a MiningField of this model (FieldScope.html: names \"must be unique from any other "
                  "names in their scope\"; accepted, decided 2026-11-18)");
      return true;
    }
    if (td_too && !m.local.count(key) && document_->td_fields.count(key)) {
      warning(code::kNameShadowsTD, ctx, node,
              std::string(what) + " name " + detail::quoted(key) +
                  " is already a TransformationDictionary DerivedField (FieldScope.html: LocalTransformations names "
                  "\"must be unique from any other names in their scope\"; accepted, P2.1)");
      return true;
    }
    return false;
  }

  // What `key` already names in model scope m, or "".
  std::string name_clash(const Scope& m, const std::string& key, bool outputs) const {
    if (m.mining.count(key)) return "a MiningField of this model";
    if (m.local.count(key)) return "a LocalTransformations DerivedField of this model";
    if (document_->td_fields.count(key)) return "a TransformationDictionary DerivedField";
    if (m.chain_outputs.count(key)) return "an OutputField of an earlier segment of the model chain";
    if (outputs && m.outputs.count(key)) return "an OutputField of this model";
    return {};
  }

  // A LocalTransformations DerivedField or OutputField named like a DataField
  // that is not one of this model's MiningFields (the MiningField case is a
  // name_clash). FieldScope.html bounds a model's scope at its MiningSchema
  // ("all data must flow in through its MiningSchema"), and its naming rules
  // make DataField names unique only "from the names of any other DataFields
  // and the names of DerivedFields in the TransformationDictionary"; the
  // stricter "DerivedFields in the TransformationDictionary or
  // LocalTransformations together with DataFields in the DataDictionary must
  // have unique names" is quoted there as PMML 3.0 history. What remains is
  // "In general, field names in PMML should be unique": a warning.
  void warn_shadowed_datafield(const Ctx& ctx, pugi::xml_node node, const std::string& key, std::string_view what) {
    if (!document_->data_fields.count(key)) return;
    warning(code::kNameShadowsDataField, ctx, node,
            std::string(what) + " name " + detail::quoted(key) +
                " is also a DataField this model does not list in its MiningSchema (FieldScope.html: \"In "
                "general, field names in PMML should be unique\")");
  }

  static bool is_feature(std::string_view f) {
    return detail::one_of(f, {"predictedValue",
                              "predictedDisplayValue",
                              "transformedValue",
                              "decision",
                              "probability",
                              "affinity",
                              "residual",
                              "standardError",
                              "standardDeviation",
                              "clusterId",
                              "clusterAffinity",
                              "entityId",
                              "entityAffinity",
                              "warning",
                              "ruleValue",
                              "reasonCode",
                              "antecedent",
                              "consequent",
                              "rule",
                              "ruleId",
                              "confidence",
                              "support",
                              "lift",
                              "leverage",
                              "confidenceIntervalLower",
                              "confidenceIntervalUpper"});
  }

  static bool double_feature(std::string_view f) {
    return detail::one_of(f, {"probability", "affinity", "residual", "standardError", "standardDeviation",
                              "clusterAffinity", "entityAffinity", "confidence", "support", "lift", "leverage",
                              "confidenceIntervalLower", "confidenceIntervalUpper"});
  }

  void declare_outputs(Ctx& ctx, pugi::xml_node output, Scope& m, std::vector<Binding*>& out) {
    std::uint32_t index = 0;
    for (pugi::xml_node of : output.children()) {
      if (of.type() != pugi::node_element || std::string_view(of.name()) == "Extension") continue;
      if (std::string_view(of.name()) != "OutputField") {
        note_unsupported(ctx, of, "Output content other than OutputField");
        continue;
      }
      Ctx fctx = ctx;
      fctx.in_output = true;
      fctx.set_where(of);
      const auto name = attr(of, "name");
      if (!name) {
        error(code::kRequired, fctx, of, "attribute name is required (pmml-4-4.xsd)");
        continue;
      }
      const std::string key(*name);
      // FieldScope.html: "The names of OutputFields must be unique from any
      // other names in their scope, i.e. within a model or model segment and
      // across all model segments in a model chain."
      if (const std::string clash = name_clash(m, key, /*outputs=*/true); !clash.empty()) {
        if (shadow_warning(fctx, of, m, key, "OutputField", /*td_too=*/false)) {
          // Its dataType without diagnostics: the field is not declared, and
          // report mode never reported its attributes.
          const auto dt = attr(of, "dataType");
          const auto info = dt ? parse_datatype(*dt) : std::nullopt;
          m.output_shadows.emplace_back(key, info ? StaticType(info->type) : StaticType{});
          continue;
        }
        error(code::kNameDuplicate, fctx, of,
              "name " + detail::quoted(*name) + " is already " + clash +
                  " (FieldScope.html: OutputField names \"must be unique from any other names in their scope\")");
        continue;
      }
      warn_shadowed_datafield(fctx, of, key, "OutputField");
      StaticType type;
      if (attr(of, "dataType")) {
        type = declared_type(fctx, of, false);
      } else {
        // pmml-4-4.xsd: dataType use="required"; §3.4: "inferred with
        // warning when absent, as today".
        warning(code::kOutputNoDataType, fctx, of,
                "has no dataType (pmml-4-4.xsd: required); its type is inferred from its feature");
        const std::string_view feature = attr(of, "feature").value_or("predictedValue");
        if (double_feature(feature)) type = TypeId::Double;
        if (feature == "predictedDisplayValue") type = TypeId::String;
      }
      FieldType ft{type, declared_optype(fctx, of, false), false};
      Binding& b = declare(key, Origin::OutputField, Container::Output, ft, m, of, index++);
      m.outputs.emplace(key, &b);
      out.push_back(&b);
    }
  }

  void bind_output_body(Ctx& model_ctx, Binding& b, Scope& m) {
    const pugi::xml_node of = b.declaration;
    Ctx ctx = model_ctx;
    ctx.in_output = true;
    ctx.current = &b;
    ctx.set_where(of);
    ctx.enclosing = b.type.type;
    b.state = Binding::State::Typing;
    const auto feature_attr = attr(of, "feature");
    if (feature_attr && !is_feature(*feature_attr)) {
      error(code::kEnum, ctx, of, "feature=" + detail::quoted(*feature_attr) + " is not a RESULT-FEATURE");
      b.state = Binding::State::Typed;
      return;
    }
    const std::string_view feature = feature_attr.value_or("predictedValue");
    ctx.transformed_value = feature == "transformedValue";
    // Output.html: "If present, the attribute targetField must refer either
    // to a MiningField of usage type target or a field described in Targets
    // element."
    const Binding* target = m.target;
    if (const auto tf = attr(of, "targetField")) {
      const std::string key(*tf);
      const auto it = m.mining.find(key);
      const bool described = m.target_names.count(key) > 0;
      target = nullptr;
      if (it != m.mining.end() && (it->second->target_usage || described))
        target = it->second;
      else if (it != m.mining.end())
        error(code::kScopeUnresolved, ctx, of,
              "targetField " + detail::quoted(*tf) +
                  " is a MiningField that is not of usage type target (Output.html: \"must refer either to a "
                  "MiningField of usage type target or a field described in Targets element\")");
      else if (!described)
        error(code::kScopeUnresolved, ctx, of,
              "targetField " + detail::quoted(*tf) +
                  " is neither a target MiningField nor a field of the Targets element (Output.html)");
    }
    if (attr(of, "segmentId"))
      // Output.html segmentId (B34): not implemented until P8.15.
      unsupported(code::kUnsupportedSegmentId, ctx, of, "segmentId is not bound until P8.15 (B34)");
    bind_values(ctx, of, b, /*allow_property=*/false);
    if (feature == "transformedValue" || feature == "decision") {
      // Output.html: "For this result feature, OutputField must contain an
      // EXPRESSION, unless it is used to refer to a transformed value of a
      // segment model through the segmentID attribute."
      const pugi::xml_node expr = expression_child(ctx, of, !attr(of, "segmentId"));
      if (expr) {
        const Typed t = bind_expression(ctx, expr, std::nullopt);
        if (!attr(of, "dataType") && t.type) {
          b.type.type = t.type;
          layout_.slots[b.slot].type = t.type;
        }
        check_cast(ctx, of, t.type, b.type.type, "its expression");
      }
    } else {
      // Every other feature carries no EXPRESSION.
      for (pugi::xml_node c : of.children())
        if (c.type() == pugi::node_element && detail::is_expression(c.name()))
          error(code::kForbidden, ctx, c,
                "an EXPRESSION in an OutputField with feature " + detail::quoted(feature) +
                    " (Output.html: only transformedValue and decision contain one)");
      // §3.4: "predictedValue -> Cast(dataType, prediction); probability,
      // residual, ... double".
      if (feature == "predictedValue") {
        if (target) {
          check_cast(ctx, of, target->type.type, b.type.type, "the predicted value");
          if (!attr(of, "dataType")) {
            b.type.type = target->type.type;
            layout_.slots[b.slot].type = b.type.type;
          }
        }
      } else if (double_feature(feature)) {
        check_cast(ctx, of, TypeId::Double, b.type.type, "a " + std::string(feature));
      } else if (feature == "predictedDisplayValue") {
        check_cast(ctx, of, TypeId::String, b.type.type, "a display value");
      }
      // Output.html: "when used with the feature probability, the attribute
      // value indicates the category" -- a value of the target.
      if (feature == "probability" && target)
        if (const auto v = attr(of, "value")) document_value(ctx, of, target->type.type, *v, "value", target);
    }
    for (const char* a : {"rank"}) int_attr(ctx, of, a);
    bool_attr(ctx, of, "isFinalResult");
    b.state = Binding::State::Typed;
  }

  void bind_targets(Ctx& ctx, pugi::xml_node targets, Scope& m) {
    for (pugi::xml_node t : targets.children()) {
      if (t.type() != pugi::node_element || std::string_view(t.name()) == "Extension") continue;
      if (std::string_view(t.name()) != "Target") {
        note_unsupported(ctx, t, "Targets content other than Target");
        continue;
      }
      Ctx tctx = ctx;
      tctx.set_where(t);
      // Targets.html: Target@field names the target; absent, the model's
      // single target.
      const Binding* field = nullptr;
      if (const auto f = attr(t, "field")) {
        m.target_names.insert(std::string(*f));
        field = resolve(tctx, t, *f);
        if (field && !m.target) {
          const auto it = m.mining.find(std::string(*f));
          if (it != m.mining.end()) m.target = it->second;
        }
      } else {
        for (const auto& [name, b] : m.mining)
          if (b->target_usage && (!field || b->index < field->index)) field = b;
      }
      declared_optype(tctx, t, false);
      enum_attr(tctx, t, "castInteger", {"round", "ceiling", "floor"});
      for (const char* a : {"min", "max", "rescaleConstant", "rescaleFactor"}) number_attr(tctx, t, a);
      for (pugi::xml_node tv : t.children()) {
        if (tv.type() != pugi::node_element || std::string_view(tv.name()) == "Extension") continue;
        if (std::string_view(tv.name()) != "TargetValue") {
          note_unsupported(tctx, tv, "Target content other than TargetValue");
          continue;
        }
        // §3.4: "TargetValue@value read by the rule above; defaultValue
        // NUMBER".
        if (const auto v = attr(tv, "value"); v && field)
          document_value(tctx, tv, field->type.type, *v, "TargetValue@value", field);
        number_attr(tctx, tv, "priorProbability");
        number_attr(tctx, tv, "defaultValue");
        for (pugi::xml_node c : tv.children())
          if (c.type() == pugi::node_element && std::string_view(c.name()) != "Extension")
            note_unsupported(tctx, c, "TargetValue content");
      }
    }
  }

  void bind_segmentation(Ctx& ctx, pugi::xml_node seg, Scope& m, std::size_t model_index) {
    const auto method =
        enum_attr(ctx, seg, "multipleModelMethod",
                  {"majorityVote", "weightedMajorityVote", "average", "weightedAverage", "median", "weightedMedian",
                   "max", "sum", "weightedSum", "selectFirst", "selectAll", "modelChain"},
                  true);
    enum_attr(ctx, seg, "missingPredictionTreatment", {"returnMissing", "skipSegment", "continue"});
    number_attr(ctx, seg, "missingThreshold");
    m.chain = method && *method == "modelChain";
    std::size_t k = 0;
    for (pugi::xml_node s : seg.children()) {
      if (s.type() != pugi::node_element || std::string_view(s.name()) == "Extension") continue;
      if (std::string_view(s.name()) != "Segment") {
        note_unsupported(ctx, s, "Segmentation content other than Segment");
        continue;
      }
      k++;
      if (m.chain) chain_segments_.emplace(s.internal_object(), static_cast<std::uint32_t>(k));
      Ctx sctx = ctx;
      const auto id = attr(s, "id");
      if (mode_ == Mode::Report)
        sctx.set_where("Segment " + (id ? detail::quoted(*id) : "#" + std::to_string(k)));
      else
        sctx.set_where(s);
      number_attr(sctx, s, "weight");
      // Segment predicates are "evaluated in M" (§3.2), after the chain
      // outputs of the segments above. Load mode: the engine resolves them.
      if (mode_ == Mode::Report) {
        bind_predicate_child(sctx, s);
        for (pugi::xml_node vw : s.children("VariableWeight"))
          if (const auto f = attr(vw, "field")) resolve(sctx, vw, *f);
      }
      pugi::xml_node child;
      for (pugi::xml_node c : s.children())
        if (c.type() == pugi::node_element && detail::is_model_element(c.name())) child = c;
      if (!child) {
        error(code::kRequired, sctx, s, "a model element is required (pmml-4-4.xsd)");
        continue;
      }
      const std::size_t c =
          bind_model(child, m, &m, mode_ == Mode::Report ? m.label + " > " + sctx.where_text : std::string());
      models_[model_index].children.push_back(c);
      if (!m.chain) continue;
      scopes_[models_[c].scope].chain_position = static_cast<std::uint32_t>(k);
      // FieldScope.html: "In a model chain, all Outputs from the segment
      // models become part of the enclosing mining model scope"; "Outputs
      // cannot be duplicated across segment since they all belong to the
      // same scope".
      Scope& child_scope = scopes_[models_[c].scope];
      for (const Binding* ordered : models_[c].output_order) {
        const auto found = child_scope.outputs.find(ordered->name);
        if (found == child_scope.outputs.end()) continue;
        Binding* o = found->second;
        // "The names of OutputFields must be unique from any other names in
        // their scope": the chain's scope also holds the MiningModel's own
        // OutputFields, declared before its segments are bound.
        if (m.chain_outputs.count(o->name) || m.mining.count(o->name) || m.local.count(o->name) ||
            m.outputs.count(o->name)) {
          Ctx octx = sctx;
          if (!m.chain_outputs.count(o->name) && !m.local.count(o->name) && !m.outputs.count(o->name) &&
              shadow_warning(octx, o->declaration, m, o->name, "OutputField of a model chain segment", false))
            continue;
          error(code::kNameDuplicate, octx, o->declaration,
                "OutputField " + detail::quoted(o->name) +
                    " is already a name in the model chain's scope (FieldScope.html: \"Outputs cannot be duplicated "
                    "across segment since they all belong to the same scope\")");
          // Still a chain output unless an earlier segment's is (emplace
          // keeps the first): a later segment's MiningField naming it then
          // resolves, and the duplicate is the document's only error.
        }
        m.chain_outputs.emplace(o->name, o);
      }
    }
  }

  // ---------------------------------------------------------------- model bodies
  void resolve_field_attributes(Ctx& ctx, pugi::xml_node node) {
    const std::string_view n = node.name();
    for (const auto& e : detail::kFieldNameAttributes) {
      if (e.element != n) continue;
      for (std::string_view a : e.attributes)
        if (const auto v = attr(node, std::string(a).c_str())) resolve(ctx, node, *v);
    }
  }

  // A value of the model's target (Node@score, ScoreDistribution@value,
  // RegressionTable@targetCategory, ...): §3.4 "category values by the rule
  // above".
  void target_value(Ctx& ctx, pugi::xml_node node, const char* attribute) {
    const auto v = attr(node, attribute);
    if (!v || !ctx.model || !ctx.model->target) return;
    document_value(ctx, node, ctx.model->target->type.type, *v, attribute, ctx.model->target);
  }

  // A value of a field (CategoricalPredictor@value, PPCell@value, ...).
  void field_value(Ctx& ctx, pugi::xml_node node, const Binding* field, const char* attribute) {
    const auto v = attr(node, attribute);
    if (!v || !field) return;
    document_value(ctx, node, field->type.type, *v, attribute, field);
  }

  const Binding* resolve_attr(Ctx& ctx, pugi::xml_node node, const char* attribute, bool required = true) {
    const auto v = attr(node, attribute);
    if (!v) {
      if (required)
        error(code::kRequired, ctx, node, std::string("attribute ") + attribute + " is required (pmml-4-4.xsd)");
      return nullptr;
    }
    return resolve(ctx, node, *v);
  }

  void visit_children(Ctx& ctx, pugi::xml_node node) {
    for (pugi::xml_node c : node.children())
      if (c.type() == pugi::node_element) visit_body(ctx, c);
  }

  void visit_body(Ctx& ctx, pugi::xml_node node) {
    const std::string_view n = node.name();
    if (detail::is_predicate(n)) return bind_predicate(ctx, node);
    if (n == "Node") {
      target_value(ctx, node, "score");
      number_attr(ctx, node, "recordCount");
      bind_predicate_child(ctx, node);
      for (pugi::xml_node c : node.children())
        if (c.type() == pugi::node_element && !detail::is_predicate(c.name())) visit_body(ctx, c);
      return;
    }
    if (n == "ScoreDistribution") {
      if (!attr(node, "value")) error(code::kRequired, ctx, node, "attribute value is required (pmml-4-4.xsd)");
      target_value(ctx, node, "value");
      for (const char* a : {"recordCount", "confidence", "probability"}) number_attr(ctx, node, a);
      return visit_children(ctx, node);
    }
    if (n == "SimpleRule" || n == "CompoundRule") {
      target_value(ctx, node, "score");
      bind_predicate_child(ctx, node);
      for (pugi::xml_node c : node.children())
        if (c.type() == pugi::node_element && !detail::is_predicate(c.name())) visit_body(ctx, c);
      return;
    }
    if (n == "RuleSet") {
      target_value(ctx, node, "defaultScore");
      return visit_children(ctx, node);
    }
    if (n == "RegressionTable") {
      target_value(ctx, node, "targetCategory");
      number_attr(ctx, node, "intercept", true);
      return visit_children(ctx, node);
    }
    if (n == "NumericPredictor") {
      resolve_attr(ctx, node, "name");
      // B7: NumericPredictor@exponent is INT-NUMBER.
      int_attr(ctx, node, "exponent");
      number_attr(ctx, node, "coefficient", true);
      return visit_children(ctx, node);
    }
    if (n == "CategoricalPredictor") {
      const Binding* b = resolve_attr(ctx, node, "name");
      field_value(ctx, node, b, "value");
      number_attr(ctx, node, "coefficient", true);
      return visit_children(ctx, node);
    }
    if (n == "PredictorTerm") {
      number_attr(ctx, node, "coefficient", true);
      for (pugi::xml_node c : node.children()) {
        if (c.type() != pugi::node_element) continue;
        if (std::string_view(c.name()) == "FieldRef")
          bind_fieldref(ctx, c);
        else
          visit_body(ctx, c);
      }
      return;
    }
    if (n == "NeuralInput" || n == "NeuralOutput") {
      for (pugi::xml_node c : node.children()) {
        if (c.type() != pugi::node_element) continue;
        if (std::string_view(c.name()) == "DerivedField")
          bind_inline_derived(ctx, c);
        else
          visit_body(ctx, c);
      }
      return;
    }
    if (n == "BayesInput" || n == "BayesOutput") {
      const Binding* b = resolve_attr(ctx, node, "fieldName");
      StaticType input_type = b ? b->type.type : std::nullopt;
      for (pugi::xml_node c : node.children()) {
        if (c.type() != pugi::node_element) continue;
        const std::string_view cn = c.name();
        if (cn == "DerivedField") {
          bind_inline_derived(ctx, c);
          input_type = declared_type(ctx, c, false);
        } else if (cn == "PairCounts") {
          // PairCounts@value is a value of the input (after its DerivedField).
          if (const auto v = attr(c, "value")) document_value(ctx, c, input_type, *v, "PairCounts@value", b);
          visit_children(ctx, c);
        } else {
          visit_body(ctx, c);
        }
      }
      return;
    }
    if (n == "TargetValueCounts" || n == "TargetValueStats") return visit_children(ctx, node);
    if (n == "TargetValueCount" || n == "TargetValueStat") {
      target_value(ctx, node, "value");
      return visit_children(ctx, node);
    }
    if (n == "ClusteringField" || n == "KNNInput") {
      resolve_attr(ctx, node, "field");
      return visit_children(ctx, node);
    }
    if (n == "TrainingInstances") return bind_training_instances(ctx, node);
    if (n == "Predictor") {
      const Binding* b = resolve_attr(ctx, node, "name");
      for (pugi::xml_node c : node.children()) {
        if (c.type() != pugi::node_element) continue;
        if (std::string_view(c.name()) == "Categories") {
          for (pugi::xml_node cat : c.children("Category")) field_value(ctx, cat, b, "value");
        } else {
          visit_body(ctx, c);
        }
      }
      return;
    }
    if (n == "PPCell") {
      const Binding* b = resolve_attr(ctx, node, "predictorName");
      // A factor's PPCell@value is a level of the factor; a covariate's is an
      // exponent (GeneralRegression.html), a number.
      bool factor = false;
      if (const auto pn = attr(node, "predictorName"))
        for (pugi::xml_node p : ctx.model->element.child("FactorList").children("Predictor"))
          if (attr(p, "name") == pn) factor = true;
      if (factor)
        field_value(ctx, node, b, "value");
      else
        number_attr(ctx, node, "value", true);
      target_value(ctx, node, "targetCategory");
      return visit_children(ctx, node);
    }
    if (n == "PCell") {
      target_value(ctx, node, "targetCategory");
      number_attr(ctx, node, "beta", true);
      return visit_children(ctx, node);
    }
    if (n == "SupportVectorMachine") {
      target_value(ctx, node, "targetCategory");
      target_value(ctx, node, "alternateTargetCategory");
      return visit_children(ctx, node);
    }
    if (n == "VectorFields") {
      for (pugi::xml_node c : node.children()) {
        if (c.type() != pugi::node_element) continue;
        if (std::string_view(c.name()) == "FieldRef")
          bind_fieldref(ctx, c);
        else
          visit_body(ctx, c);
      }
      return;
    }
    if (n == "Attribute") {  // Scorecard
      bind_predicate_child(ctx, node);
      number_attr(ctx, node, "partialScore");
      for (pugi::xml_node c : node.children()) {
        if (c.type() != pugi::node_element || detail::is_predicate(c.name())) continue;
        if (std::string_view(c.name()) == "ComplexPartialScore") {
          if (const pugi::xml_node e = expression_child(ctx, c)) bind_expression(ctx, e, std::nullopt);
        } else {
          visit_body(ctx, c);
        }
      }
      return;
    }
    if (n == "Item") {  // AssociationModel
      // AssociationRules.html: "PMML 4.3 added optional attributes field and
      // category that help to link the items to the data"; Item@value is the
      // item as in the input data (it may be a field name), category the
      // field's value.
      const Binding* b = resolve_attr(ctx, node, "field", false);
      field_value(ctx, node, b, "category");
      number_attr(ctx, node, "weight");
      return visit_children(ctx, node);
    }
    if (n == "TestDistributions") {  // BaselineModel
      resolve_attr(ctx, node, "field");
      resolve_attr(ctx, node, "weightField", false);
      return visit_children(ctx, node);
    }
    if (n == "FieldValue" || n == "FieldValueCount") {
      const Binding* b = resolve_attr(ctx, node, "field");
      field_value(ctx, node, b, "value");
      return visit_children(ctx, node);
    }
    if (n == "DerivedField") return bind_inline_derived(ctx, node);
    if (detail::is_expression(n)) {
      bind_expression(ctx, node, std::nullopt);
      return;
    }
    if (detail::is_inert(n)) {
      resolve_field_attributes(ctx, node);
      if (n != "Extension") visit_children(ctx, node);
      return;
    }
    note_unsupported(ctx, node, "no binder rule for it yet");
  }

  void bind_training_instances(Ctx& ctx, pugi::xml_node node) {
    std::unordered_map<std::string, const Binding*> columns;
    for (pugi::xml_node c : node.children()) {
      if (c.type() != pugi::node_element) continue;
      const std::string_view cn = c.name();
      if (cn == "InstanceFields") {
        for (pugi::xml_node inf : c.children("InstanceField")) {
          const auto f = attr(inf, "field");
          if (!f) {
            error(code::kRequired, ctx, inf, "attribute field is required (pmml-4-4.xsd)");
            continue;
          }
          // NearestNeighborModel's instanceIdVariable may be a column of the
          // table only; it is not a field of the model.
          const Binding* b = nullptr;
          if (attr(ctx.model->element, "instanceIdVariable") != f) b = resolve(ctx, inf, *f);
          columns.emplace(std::string(attr(inf, "column").value_or(*f)), b);
        }
      } else if (cn == "InlineTable") {
        // A cell is a value of its column's field (§3.4).
        for (pugi::xml_node row : c.children("row"))
          for (pugi::xml_node cell : row.children()) {
            if (cell.type() != pugi::node_element) continue;
            const auto it = columns.find(cell.name());
            if (it == columns.end() || !it->second) continue;
            document_value(ctx, cell, it->second->type.type, cell.text().get(), "cell <" + it->first + ">", it->second);
          }
      } else if (cn == "Extension") {
        continue;
      } else {
        note_unsupported(ctx, c, cn == "TableLocator" ? "external tables" : "TrainingInstances content");
      }
    }
  }
};

}  // namespace cpmml::bind

#endif  // CPMML_BIND_BINDER_H
