
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MODEL_H
#define CPMML_MODEL_H

#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "dagbuilder.h"
#include "miningfunction.h"
#include "miningschema.h"
#include "output/outputdictionary.h"
#include "target.h"
#include "transformationdictionary.h"

/**
 * @class InternalModel
 *
 * Abstract class representing a <a
 * href="http://dmg.org/pmml/v4-4/GeneralStructure.html#xsdGroup_MODEL-ELEMENT">PMML
 * MODEL-ELEMENT</a>.
 *
 * Implementations of this class are the central element of cPMML and they are
 * the ones actually producing the scoring.
 *
 * It contains:
 * - MiningSchema
 * - LocalTransformations
 * - Target
 * - Output
 *
 * Through the method validate, the presence of all fields needed by the model
 * is checked. This includes DerivedFields obtained through
 * TransformationDictionary and also LocalTransformations.
 */
class InternalModel {
 public:
  std::shared_ptr<Indexer> indexer;
  MiningFunction mining_function;
  MiningSchema mining_schema;
  MiningField target_field;
  TransformationDictionary transformation_dictionary;
  bool has_local_transformations = false;
  Target target;
  OutputDictionary output;
  Sample base_sample;
  std::vector<std::string> derivedfields_dag;
  // The names this model's own LocalTransformations define.
  std::unordered_set<std::string> local_derivedfields;
  // The DerivedFields this model evaluates itself when it runs as a Segment,
  // in dependency order: its own LocalTransformations, and any field of its
  // derivedfields_dag that the enclosing MiningModel does not evaluate. Set by
  // scope_under(); empty for a top-level model, which evaluates its whole
  // derivedfields_dag in score().
  std::vector<std::string> segment_dag;

  // How the predicted value is written (utils/java_format.h), fixed at load by
  // the target field. A regression prediction is a number of the target's
  // dataType: Double.toString for double ("1.5"), Float.toString for float,
  // "7" for integer. A classification prediction of a non-string target is a
  // label written as JPMML writes that type: "1.0" for the class value="1" of
  // a double target, "true" for value="1" of a boolean one. String targets,
  // clustering and association results are text and stay as they are.
  enum class Shown : unsigned char { AS_IS, NUMBER, LABEL };
  Shown shown = Shown::AS_IS;
  java::Kind shown_kind = java::Kind::DOUBLE;

  InternalModel() = default;

  // There was a second constructor here taking no TransformationDictionary. It
  // left base_sample empty and derivedfields_dag unbuilt, so any model built
  // through it would have indexed out of bounds on the first score. Nothing
  // reached it — only TreeModel's and RegressionModel's own uncalled overloads
  // did, and those are gone too. Deleted rather than repaired: a second way to
  // half-construct a model is not worth keeping for a caller that never came.

  InternalModel(const XmlNode& node, const DataDictionary& data_dictionary,
                const TransformationDictionary& transformation_dictionary, const std::shared_ptr<Indexer>& indexer)
      : indexer(indexer),
        mining_function(MiningFunction::read(node)),
        mining_schema(MiningSchema(node.get_child("MiningSchema"), data_dictionary)),
        target_field(get_target(mining_function, mining_schema, indexer)),
        transformation_dictionary(transformation_dictionary),
        has_local_transformations(add_local_transformations(node, this->transformation_dictionary, indexer)),
        target(get_target(node, mining_schema, this->transformation_dictionary, mining_function)),
        output(get_output(node, indexer, target_field.name)),
        base_sample(create_basesample(indexer)),
        derivedfields_dag(DagBuilder::build(mining_schema, this->transformation_dictionary)),
        local_derivedfields(get_local_derivedfields(node)),
        is_scorable(check_scorable(node)) {
    set_shown();
  }

  void set_shown() {
    shown_kind = java_kind(target_field.datatype, target_field.integer);
    const bool text = target_field.datatype.value == DataType::DataTypeValue::STRING;
    // A double target is rewritten too: most models already compute a double
    // and write it with Double.toString, but a TreeModel's prediction is the
    // Node@score text, which is "10" or "1e7" in the document and 10.0 or
    // 1.0E7 to JPMML.
    if (mining_function.value == MiningFunction::MiningFunctionType::REGRESSION && !text)
      shown = Shown::NUMBER;
    else if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION && !text)
      shown = Shown::LABEL;
    else
      shown = Shown::AS_IS;
  }

  // The class `label` of a non-string target as JPMML writes it; `label`
  // itself when it is not a value of the target's type. A model names its
  // classes with a handful of texts, so the answer is remembered per thread
  // (it depends on the text and shown_kind only): rewriting "1" as "1.0" on
  // every record cost AuditBinaryReg 15% before.
  inline const std::string& label_text(const std::string& label) const {
    thread_local std::unordered_map<std::string, std::string> known[4];
    auto& cache = known[static_cast<size_t>(shown_kind)];
    const auto it = cache.find(label);
    if (it != cache.end()) return it->second;
    if (cache.size() > 4096) cache.clear();  // labels are few; this bounds a pathological model
    return cache.emplace(label, compute_label_text(label)).first->second;
  }

  inline std::string compute_label_text(const std::string& label) const {
    if (shown_kind == java::Kind::BOOLEAN) {
      bool parsed = false;
      return lexical::parse_boolean(label, parsed) ? java::boolean_text(parsed) : label;
    }
    if (shown_kind == java::Kind::INTEGER && is_long_text(label)) return label;  // already "7"
    double parsed = 0;
    return java::parse(label, parsed) ? java::text({parsed, shown_kind}) : label;
  }

  // "0", or an optional '-' and digits without a leading zero: Long.toString
  // of itself.
  static bool is_long_text(const std::string& text) {
    size_t i = !text.empty() && text[0] == '-' ? 1 : 0;
    if (i >= text.size() || (text[i] == '0' && (text.size() > i + 1 || i == 1))) return false;
    if (text.size() - i > 18) return false;
    for (; i < text.size(); i++)
      if (text[i] < '0' || text[i] > '9') return false;
    return true;
  }

  // The prediction `score`, after the Target, written as `shown` says. A
  // displayValue is text and is left alone: the score is a displayValue when
  // the Target set raw_score to something else.
  inline void present(InternalScore& score) const {
    if (shown == Shown::AS_IS || score.empty) return;
    if (shown == Shown::NUMBER) {
      if (is_double_min(score.double_score)) return;  // not a number: nothing to rewrite
      score.score = java::text({score.double_score, shown_kind});
      return;
    }
    if (score.raw_score.empty() || score.raw_score == score.score) {
      score.score = label_text(score.score);
      if (!score.raw_score.empty()) score.raw_score = score.score;
    } else {
      score.raw_score = label_text(score.raw_score);  // score is a displayValue
    }
    // Rename the classes of the distribution in place: the nodes are moved,
    // not reallocated. Two spellings of one class ("1", "1.0") add up.
    using Node = std::unordered_map<std::string, double>::node_type;
    auto& probabilities = score.probabilities;
    std::array<Node, 8> few;
    std::vector<Node> many;
    size_t renamed = 0;
    for (auto it = probabilities.begin(); it != probabilities.end();) {
      const auto next = std::next(it);
      const std::string& shown_label = label_text(it->first);
      if (shown_label != it->first) {
        Node node = probabilities.extract(it);
        node.key() = shown_label;
        if (renamed < few.size())
          few[renamed] = std::move(node);
        else
          many.push_back(std::move(node));
        renamed++;
      }
      it = next;
    }
    const auto put_back = [&probabilities](Node& node) {
      auto inserted = probabilities.insert(std::move(node));
      if (!inserted.inserted) inserted.position->second += inserted.node.mapped();
    };
    for (size_t i = 0; i < std::min(renamed, few.size()); i++) put_back(few[i]);
    for (auto& node : many) put_back(node);
  }

  // The Target, then the text of the prediction (present).
  inline void finish(InternalScore& score, const bool display = true) const {
    target(score, display);
    present(score);
  }

  // The same for a prediction carried as text (predict_raw). A label is
  // rewritten before the Target, which matches classes by value and may
  // rename it to a displayValue; a number after it, as the Target rescales
  // the number.
  inline std::string finish(const std::string& predicted, const bool display = true) const {
    switch (shown) {
      case Shown::LABEL: {
        const std::string shown_text = target(predicted.empty() ? predicted : label_text(predicted), display);
        // A prior (no prediction) is the document's text: rewrite it too
        // unless it is a displayValue.
        if (predicted.empty() && !shown_text.empty() && !target.is_display_value(shown_text, display))
          return label_text(shown_text);
        return shown_text;
      }
      case Shown::NUMBER: {
        const std::string number = target(predicted, display);
        double value = 0;
        return !number.empty() && java::parse(number, value) ? java::text({value, shown_kind}) : number;
      }
      case Shown::AS_IS:
        break;
    }
    return target(predicted, display);
  }

  std::string output_name() const {
    if (!output.empty) {
      for (const auto& of : output.raw_outputfields) {
        const auto t = of.expression_type.value;
        if (t == OutputExpressionType::OutputExpressionTypeValue::PREDICTED_VALUE ||
            t == OutputExpressionType::OutputExpressionTypeValue::PREDICTED_DISPLAY_VALUE)
          return of.name;
      }
      if (!output.raw_outputfields.empty()) return output.raw_outputfields.front().name;
    }
    return target_field.name;
  }

  // Every OutputField the document declares, in document order. Callers need
  // this to tell "the model produced no value for that field" apart from "that
  // field is not one of this model's outputs" -- num_outputs/str_outputs only
  // carry what was actually produced, so a silently missing output is otherwise
  // indistinguishable from a correct one.
  inline std::vector<std::string> output_names() const {
    std::vector<std::string> names;
    if (output.empty) return names;
    names.reserve(output.raw_outputfields.size());
    for (const auto& of : output.raw_outputfields) names.push_back(of.name);
    return names;
  }

  inline bool validate(const std::unordered_map<std::string, std::string>& sample) const {
    Value::ScopedIndexer _scope(indexer.get());
    Sample internal_sample = base_sample;
    mining_schema.prepare(internal_sample, sample);
    if (!transformation_dictionary.empty)
      for (const auto& derivedfield_name : derivedfields_dag)
        transformation_dictionary[derivedfield_name].prepare(internal_sample);

    return mining_schema.validate(internal_sample);
  }

  // ---- this model as a Segment ---------------------------------------------
  //
  // A segment model's LocalTransformations are evaluated as when the model is
  // scored on its own, and stay local to it, as in JPMML 1.7.7: a sibling
  // segment, a later modelChain segment, or the enclosing MiningModel's
  // OutputFields do not see them. The enclosing MiningModel has already
  // evaluated its own derivedfields_dag (in score(), or as a segment itself),
  // so a segment evaluates only what that leaves out: its own
  // LocalTransformations (which may shadow an outer field of the same name)
  // and an outer field whose inputs only this model's MiningSchema provides.
  // These used to be evaluated only for the first segment a modelChain ran,
  // and there on the shared sample, so every other segment read its own
  // LocalTransformations as missing (no prediction), and later modelChain
  // segments read the first one's.
  void scope_under(const std::vector<std::string>& parent_dag) {
    const std::unordered_set<std::string> evaluated(parent_dag.cbegin(), parent_dag.cend());
    segment_dag.clear();
    for (const auto& name : derivedfields_dag)
      if (local_derivedfields.count(name) || !evaluated.count(name)) segment_dag.push_back(name);
  }

  // Runs f on the sample as this segment sees it. A model with nothing to
  // evaluate reads the enclosing sample directly; otherwise on a private copy,
  // so that segments scored in parallel (MULTITHREADING) never share one.
  template <class F>
  inline auto in_segment_scope(const Sample& sample, F&& f) const {
    if (segment_dag.empty()) return f(sample);
    Sample local = sample;
    for (const auto& name : segment_dag) transformation_dictionary[name].prepare(local);
    return f(static_cast<const Sample&>(local));
  }

  // The same for a modelChain step, which writes its target and OutputFields
  // into the chain's sample: segment_dag is evaluated in place and the
  // previous values put back when the step ends, so only what the step writes
  // outlives it.
  class ChainScope {
   public:
    ChainScope(const InternalModel& model, Sample& sample) : model(model), sample(sample) {
      saved.reserve(model.segment_dag.size());
      for (const auto& name : model.segment_dag) {
        const DerivedField& field = model.transformation_dictionary[name];
        saved.emplace_back(field.index, sample[field.index].value);
        field.prepare(sample);
      }
    }
    ~ChainScope() {
      for (auto it = saved.rbegin(); it != saved.rend(); ++it)
        if (!model.writes(it->first)) sample.change_value(it->first, it->second);
    }
    ChainScope(const ChainScope&) = delete;
    ChainScope& operator=(const ChainScope&) = delete;

   private:
    const InternalModel& model;
    Sample& sample;
    std::vector<std::pair<size_t, Value>> saved;
  };

  // Whether a modelChain step writes the sample slot: its target or one of its
  // OutputFields.
  inline bool writes(const size_t index) const {
    if (index == indexer->get_index(target_field.name)) return true;
    for (const auto& of : output.dag)
      if (of.index == index) return true;
    return false;
  }

  // Model-chain steps. Each returns whether the model made a prediction, after
  // its own Targets: a step with no prediction of its own but a
  // priorProbability or defaultValue predicts that, as a standalone model does.
  // One that still has none leaves its target missing and writes no
  // OutputField, so the segments after it see missing inputs, rather than
  // converting "" to a number and throwing. The value written for the next
  // steps is the raw class, not its displayValue, as in JPMML 1.7.7.
  inline bool augment(Sample& sample) const {
    ChainScope scope(*this, sample);
    std::unique_ptr<InternalScore> score = score_raw(sample);
    finish(*score, false);
    if (score->empty) {
      sample.change_value(indexer->get_index(target_field.name), Value());
      return false;
    }
    // A number goes in as the number: its text may be Java's "Infinity",
    // which the document reader refuses.
    const bool numeric = target_field.datatype.value != DataType::DataTypeValue::STRING &&
                         target_field.datatype.value != DataType::DataTypeValue::BOOLEAN &&
                         !is_double_min(score->double_score);
    sample.change_value(indexer->get_index(target_field.name),
                        numeric ? Value(score->double_score, target_field.datatype)
                                : Value(score->score, target_field.datatype));
    _write_outputs_to_sample(sample, *score);
    return true;
  };

  // The final step: its score is the MiningModel's, by raw class like any
  // segment's, so that only the MiningModel's own Targets name it. This step's
  // displayValue showed through as the MiningModel's prediction and
  // predictedDisplayValue ("HIGH" where JPMML 1.7.7 gives "hi"), and kept the
  // MiningModel's own displayValue from applying.
  inline std::unique_ptr<InternalScore> augment_last(Sample& sample) const {
    ChainScope scope(*this, sample);
    std::unique_ptr<InternalScore> score = score_raw(sample);
    finish(*score, false);
    add_outputs(sample, *score);

    return score;
  };

  // This model's OutputFields, as a segment, for `score`, which names the raw
  // class (target(..., false)). A predictedDisplayValue OutputField reports the
  // class's displayValue from this model's Targets, as when the model is scored
  // on its own: JPMML 1.7.7 does so for selectFirst, selectAll and modelChain
  // segments, while the MiningModel's prediction stays the raw class.
  inline void add_outputs(const Sample& sample, InternalScore& score) const {
    const std::unique_ptr<InternalScore> shown = displayed(score);
    output.add_output(sample, score, shown.get());
  }

  // `score`, a raw-class prediction, named by its displayValue; nullptr when no
  // TargetValue renames it. What a predictedDisplayValue OutputField reads.
  inline std::unique_ptr<InternalScore> displayed(const InternalScore& score) const {
    if (score.empty) return nullptr;
    const TargetValue* shown = target.display_of(score.score);
    if (shown == nullptr) return nullptr;
    auto result = std::make_unique<InternalScore>(shown->display_value);
    result->raw_score = score.score;
    return result;
  }

  // Write each OutputField value into the sample so downstream chain segments
  // can read them.  Probability-type OutputFields require the InternalScore
  // (which has score.probabilities); PredictedValue-type OutputFields read
  // from the sample directly via eval().
  inline void _write_outputs_to_sample(Sample& sample, const InternalScore& score) const {
    const std::unique_ptr<InternalScore> shown = displayed(score);
    for (const auto& of : output.dag) {
      if (shown &&
          of.expression_type.value == OutputExpressionType::OutputExpressionTypeValue::PREDICTED_DISPLAY_VALUE) {
        // The displayValue, which the next steps read as the field's value
        // (JPMML 1.7.7), not the raw class in the target slot.
        if (of.datatype.value == DataType::DataTypeValue::STRING)
          sample.change_value_if_missing(of.index, Value(shown->score, of.datatype));
        else if (!is_double_min(shown->double_score))
          sample.change_value(of.index, Value(shown->double_score));
        continue;
      }
      if (of.datatype.value == DataType::DataTypeValue::STRING) {
        sample.change_value_if_missing(of.index, of.expression->eval(sample));
      } else {
        // Prefer the sample-derived value: for predictedValue-type outputs it
        // carries the post-Target (rescaled) prediction written by augment().
        // The score is post-Target too, so probabilities include the priors.
        const Value sv = of.expression->eval(sample);
        if (!sv.missing) {
          sample.change_value(of.index, sv);
        } else {
          const double v = of.expression->eval_double(sample, score);
          if (!is_double_min(v)) sample.change_value(of.index, Value(v));
        }
      }
    }
  };

  inline std::unique_ptr<InternalScore> score(const std::unordered_map<std::string, std::string>& sample) const {
    if (!is_scorable)
      throw cpmml::EvaluationException("The model is defined as non-scorable (isScorable=false)");
    Value::ScopedIndexer _scope(indexer.get());
    Sample internal_sample = base_sample;
    mining_schema.prepare(internal_sample, sample);
    if (!transformation_dictionary.empty)
      for (const auto& derivedfield_name : derivedfields_dag)
        transformation_dictionary[derivedfield_name].prepare(internal_sample);

    if (!mining_schema.validate(internal_sample)) invalid_input(mining_schema, internal_sample, sample);

    // Load actual target value if provided in the raw input (needed for
    // feature="residual" output computation).  MiningSchema::prepare skips the
    // target field, so we have to do this separately.
    if (!mining_schema.target.empty) {
      auto it = sample.find(mining_schema.target.name);
      // A blank cell is an absent actual value, not a value of "". Converting it
      // threw for any numeric target, so a dataset carrying an empty target
      // column -- an ordinary thing to score against -- failed outright.
      //
      // The actual value is read as an input of the target's dataType. Only
      // feature="residual" uses it, so as in JPMML 1.7.7 text that is not a
      // value of that type fails the record (or takes the target's
      // invalidValueTreatment) only when the model declares a residual; a
      // model that never reads the column does not reject it.
      if (it != sample.end() && !it->second.empty()) {
        const MiningField& actual_field = mining_schema.target;
        Value& actual = internal_sample[mining_schema.target_index].value;
        switch (actual_field.parse_input(it->second, actual)) {
          case MiningField::InputStatus::VALID:
            break;
          case MiningField::InputStatus::INVALID:
            actual = declares_residual() ? actual_field.handle_invalid(actual) : Value();
            break;
          case MiningField::InputStatus::MALFORMED:
            actual = declares_residual() ? actual_field.handle_malformed(it->second) : Value();
            break;
        }
      }
    }

    std::unique_ptr<InternalScore> score = score_raw(internal_sample);
    finish(*score);
    output.add_output(internal_sample, *score);

    return score;
  };

  using FieldValue = std::variant<std::string, std::vector<std::string>>;

  // Variant-aware score: default flattens to string map.
  // AssociationModel overrides to extract collection inputs.
  virtual inline std::unique_ptr<InternalScore> score(
      const std::unordered_map<std::string, FieldValue>& sample) const {
    std::unordered_map<std::string, std::string> flat;
    for (const auto& [k, v] : sample) {
      if (std::holds_alternative<std::string>(v))
        flat[k] = std::get<std::string>(v);
    }
    return score(flat);
  }

  virtual std::unique_ptr<InternalScore> score_raw(const Sample& sample) const = 0;

  inline std::string predict(const std::unordered_map<std::string, std::string>& sample) const {
    Value::ScopedIndexer _scope(indexer.get());
    Sample internal_sample = base_sample;
    mining_schema.prepare(internal_sample, sample);
    if (!transformation_dictionary.empty)
      for (const auto& derivedfield_name : derivedfields_dag)
        transformation_dictionary[derivedfield_name].prepare(internal_sample);

    if (!mining_schema.validate(internal_sample)) invalid_input(mining_schema, internal_sample, sample);

    return finish(predict_raw(internal_sample));
  };

  // B45: the error names the field and its input (it used to print the whole
  // record as "Sample: {...}didn't pass input validation"), as
  // MiningField::handle_invalid does for returnInvalid. Out of line and cold:
  // the field is looked for only after validate() has failed, so a valid
  // record costs what it did before.
  [[noreturn]] [[gnu::cold]] [[gnu::noinline]] static void invalid_input(
      const MiningSchema& mining_schema, const Sample& internal_sample,
      const std::unordered_map<std::string, std::string>& sample) {
    const MiningField* field = mining_schema.first_invalid(internal_sample);
    if (!field) throw std::logic_error("invalid_input: no MiningField fails validation");
    const auto it = sample.find(field->name);
    throw cpmml::InvalidValueException("Invalid value for field: " + field->name +
                                       (it == sample.end() ? std::string() : " (input \"" + it->second + "\")"));
  }

  static inline Target get_target(const XmlNode& node, const MiningSchema& mining_schema,
                                  const TransformationDictionary& transformation_dictionary,
                                  const MiningFunction& mining_function) {
    if (node.exists_child("Targets"))
      if (node.get_child("Targets").exists_child("Target"))
        return Target(node.get_child("Targets").get_child("Target"), mining_schema, transformation_dictionary,
                      mining_function);

    return Target();
  }

  static inline OutputDictionary get_output(const XmlNode& node, const std::shared_ptr<Indexer>& indexer,
                                            const std::string& target_field_name) {
    return node.exists_child("Output") ? OutputDictionary(node.get_child("Output"), indexer, target_field_name)
                                       : OutputDictionary();
  }

  bool is_scorable = true;

  inline bool declares_residual() const {
    for (const auto& of : output.raw_outputfields)
      if (of.expression_type.value == OutputExpressionType::OutputExpressionTypeValue::RESIDUAL) return true;
    return false;
  }

  static inline bool check_scorable(const XmlNode& node) {
    return node.get_bool_attribute("isScorable", true);
  }

  static inline bool add_local_transformations(const XmlNode& node, TransformationDictionary& transformation_dictionary,
                                               const std::shared_ptr<Indexer>& indexer) {
    if (node.exists_child("LocalTransformations")) {
      for (auto const& n : node.get_child("LocalTransformations").get_childs("DerivedField"))
        transformation_dictionary.add_derived_field(DerivedField::declared(n, indexer));
      return true;
    }

    return false;
  }

  static inline std::unordered_set<std::string> get_local_derivedfields(const XmlNode& node) {
    std::unordered_set<std::string> names;
    if (node.exists_child("LocalTransformations"))
      for (auto const& n : node.get_child("LocalTransformations").get_childs("DerivedField"))
        names.insert(n.required_attribute("name"));
    return names;
  }

  static Sample create_basesample(const std::shared_ptr<Indexer>& indexer) {
    Sample sample(indexer->size());
    for (const auto& index : *indexer) sample[index.second] = Feature(index.first, Value());

    return sample;
  }

  static MiningField get_target(const MiningFunction& mining_function, const MiningSchema& mining_schema,
                                const std::shared_ptr<Indexer>& indexer) {
    if (mining_schema.target.empty) {
      DataType datatype = (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION ||
                         mining_function.value == MiningFunction::MiningFunctionType::ASSOCIATION_RULES)
                            ? DataType::DataTypeValue::STRING
                            : DataType::DataTypeValue::DOUBLE;
      MiningField target_placeholder("_target", datatype);
      indexer->get_or_set(target_placeholder.name, target_placeholder.datatype);

      return target_placeholder;
    }

    return mining_schema.target;
  }

  virtual std::string predict_raw(const Sample& sample) const = 0;

  InternalModel(const InternalModel&) = default;

  InternalModel(InternalModel&&) = default;

  InternalModel& operator=(const InternalModel&) = default;

  InternalModel& operator=(InternalModel&&) = default;

  virtual ~InternalModel() = default;
};

#endif
