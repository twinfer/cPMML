
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MULTIPLEMODELMETHOD_H
#define CPMML_MULTIPLEMODELMETHOD_H

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef MULTITHREADING
#include <omp.h>
#endif

#include "core/enum_parser.h"
#include "core/internal_score.h"
#include "core/target.h"
#include "segment.h"
#include "utils/utils.h"

/**
 * @class MultipleModelMethod
 *
 * Class reresenting <a
 * href="http://dmg.org/pmml/v4-4/MultipleModels.html#xsdType_MULTIPLE-MODEL-METHOD">PMML
 * MULTIPLE-MODEL-METHOD</a>.
 *
 * For instance:
 * - majorityVote
 * - weightedAverage, weightedSum, weightedMedian
 * - modelChain
 */
class MultipleModelMethod {
 public:
  enum class MultipleModelMethodType {
    MAJORITY_VOTE,
    WEIGHTED_MAJORITY_VOTE,
    AVERAGE,
    WEIGHTED_AVERAGE,
    MEDIAN,
    WEIGHTED_MEDIAN,
    MAX,
    SUM,
    WEIGHTED_SUM,
    SELECT_FIRST,
    SELECT_ALL,
    MODEL_CHAIN
  };

  // Segmentation/@missingPredictionTreatment: what a selected segment that
  // yields no prediction does to the ensemble.
  enum class MissingPredictionTreatment { RETURN_MISSING, SKIP_SEGMENT, CONTINUE };

  struct MissingPrediction {
    MissingPredictionTreatment treatment = MissingPredictionTreatment::CONTINUE;  // PMML default
    double threshold = 1;                                                         // missingThreshold
  };

  // The categories of the ensemble's target: the <Value property="valid">
  // entries of its DataField. A category no segment voted for (or gave a
  // probability to) still has a probability, 0, as in JPMML 1.7.7 and as the
  // spec's "proportion of votes" implies.
  struct Classes {
    std::vector<std::string> values;
    Category category;  // for a numeric target "1" and "1.0" are one category
  };

  using Function = std::function<std::unique_ptr<InternalScore>(const Sample&, const Segments&)>;

  MultipleModelMethodType value = MultipleModelMethodType::MAJORITY_VOTE;
  MissingPrediction missing_prediction;
  Function function;

  MultipleModelMethod() = default;

  // Segmentation@multipleModelMethod is required.
  MultipleModelMethod(const XmlNode& segmentation, const MiningFunction& mining_function,
                      const MissingPrediction& missing_prediction, const Classes& classes)
      : value(read_enum(segmentation, "multipleModelMethod", TABLE)),
        missing_prediction(missing_prediction),
        function(to_function(value, mining_function, missing_prediction, classes)) {}

  // The target field's declared categories, in document order.
  static Classes classes_of(const DataField& target_field) {
    Classes classes;
    classes.values = target_field.valid_values;
    classes.category = Category(target_field.datatype);
    return classes;
  }

  static constexpr EnumEntry<MultipleModelMethodType> TABLE[] = {
      {"majorityVote", MultipleModelMethodType::MAJORITY_VOTE},
      {"weightedMajorityVote", MultipleModelMethodType::WEIGHTED_MAJORITY_VOTE},
      {"average", MultipleModelMethodType::AVERAGE},
      {"weightedAverage", MultipleModelMethodType::WEIGHTED_AVERAGE},
      {"median", MultipleModelMethodType::MEDIAN},
      {"weightedMedian", MultipleModelMethodType::WEIGHTED_MEDIAN},
      {"max", MultipleModelMethodType::MAX},
      {"sum", MultipleModelMethodType::SUM},
      {"weightedSum", MultipleModelMethodType::WEIGHTED_SUM},
      {"selectFirst", MultipleModelMethodType::SELECT_FIRST},
      {"selectAll", MultipleModelMethodType::SELECT_ALL},
      {"modelChain", MultipleModelMethodType::MODEL_CHAIN},
  };

  static constexpr EnumEntry<MissingPredictionTreatment> MISSING_PREDICTION_TABLE[] = {
      {"returnMissing", MissingPredictionTreatment::RETURN_MISSING},
      {"skipSegment", MissingPredictionTreatment::SKIP_SEGMENT},
      {"continue", MissingPredictionTreatment::CONTINUE},
  };

  // Segmentation/@missingPredictionTreatment (default continue) and
  // @missingThreshold (default 1).
  static MissingPrediction missing_prediction_from(const XmlNode& segmentation) {
    MissingPrediction result;
    result.treatment =
        read_enum(segmentation, "missingPredictionTreatment", MISSING_PREDICTION_TABLE,
                  MissingPredictionTreatment::CONTINUE);
    result.threshold = segmentation.get_double_attribute("missingThreshold", 1);
    return result;
  }

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }

  static Function to_function(MultipleModelMethodType method, const MiningFunction& mining_function,
                              const MissingPrediction& mp, const Classes& classes) {
    // The spec says skipSegment "should not be used with modelChain", and JPMML
    // 1.7.7 refuses to score it there and with selectAll. cPMML keeps scoring
    // those documents as it did before the attribute was read: such a segment
    // is treated as under continue (select_all, model_chain).
    switch (method) {
      case MultipleModelMethodType::MAJORITY_VOTE:
        return [mp, classes](const Sample& s, const Segments& e) { return vote(s, e, mp, false, classes); };
      case MultipleModelMethodType::WEIGHTED_MAJORITY_VOTE:
        return [mp, classes](const Sample& s, const Segments& e) { return vote(s, e, mp, true, classes); };
      case MultipleModelMethodType::AVERAGE:
        switch (mining_function.value) {
          case MiningFunction::MiningFunctionType::CLASSIFICATION:
            return [mp, classes](const Sample& s, const Segments& e) {
              return classification_average(s, e, mp, false, classes);
            };
          case MiningFunction::MiningFunctionType::REGRESSION:
            return [mp](const Sample& s, const Segments& e) { return regression_average(s, e, mp, false); };
          default:
            throw cpmml::ParsingException("average not supported for this mining function");
        }
      case MultipleModelMethodType::WEIGHTED_AVERAGE:
        switch (mining_function.value) {
          case MiningFunction::MiningFunctionType::CLASSIFICATION:
            return [mp, classes](const Sample& s, const Segments& e) {
              return classification_average(s, e, mp, true, classes);
            };
          case MiningFunction::MiningFunctionType::REGRESSION:
            return [mp](const Sample& s, const Segments& e) { return regression_average(s, e, mp, true); };
          default:
            throw cpmml::ParsingException("weightedAverage not supported for this mining function");
        }
      case MultipleModelMethodType::MEDIAN:
        return [mp](const Sample& s, const Segments& e) { return median(s, e, mp); };
      case MultipleModelMethodType::WEIGHTED_MEDIAN:
        // The spec also allows weightedMedian for classification, on the
        // segments' probabilities; JPMML 1.7.7 refuses to load it there, as for
        // clustering and every other function ("Segmentation@multipleModelMethod
        // =weightedMedian is not valid"). cPMML implements it for regression only.
        switch (mining_function.value) {
          case MiningFunction::MiningFunctionType::REGRESSION:
            return [mp](const Sample& s, const Segments& e) { return weighted_median(s, e, mp); };
          default:
            throw cpmml::ParsingException("weightedMedian not supported for this mining function");
        }
      case MultipleModelMethodType::MAX:
        // The spec lists max among the methods for classification, not for
        // regression ("For regression models only average, weightedAverage,
        // median, weightedMedian, sum, weightedSum, modelChain, selectFirst, or
        // selectAll are applicable"), and JPMML 1.7.7 refuses to score it there.
        switch (mining_function.value) {
          case MiningFunction::MiningFunctionType::CLASSIFICATION:
            return [mp, classes](const Sample& s, const Segments& e) {
              return classification_max(s, e, mp, classes);
            };
          default:
            throw cpmml::ParsingException("max not supported for this mining function");
        }
      case MultipleModelMethodType::SUM:
        return [mp](const Sample& s, const Segments& e) { return sum(s, e, mp); };
      case MultipleModelMethodType::WEIGHTED_SUM:
        // "For classification models all the combination methods, except for sum
        // and weightedSum, can be used", and clustering lists neither. JPMML
        // 1.7.7 refuses weightedSum for every function but regression.
        switch (mining_function.value) {
          case MiningFunction::MiningFunctionType::REGRESSION:
            return [mp](const Sample& s, const Segments& e) { return weighted_sum(s, e, mp); };
          default:
            throw cpmml::ParsingException("weightedSum not supported for this mining function");
        }
      case MultipleModelMethodType::SELECT_FIRST:
        return [mp](const Sample& s, const Segments& e) { return select_first(s, e, mp); };
      case MultipleModelMethodType::SELECT_ALL:
        return [mp](const Sample& s, const Segments& e) { return select_all(s, e, mp); };
      case MultipleModelMethodType::MODEL_CHAIN:
        return [mp](const Sample& s, const Segments& e) { return model_chain(s, e, mp); };
    }

    throw cpmml::ParsingException("multipleModelMethod " + enum_display(method, TABLE, "?") + " not supported");
  }

  // Deterministic winner selection: highest probability wins, ties broken by
  // name so results do not depend on unordered_map iteration order.
  inline static std::string argmax(const std::unordered_map<std::string, double>& probabilities) {
    std::string best;
    double best_prob = -1;
    for (const auto& [category, prob] : probabilities)
      if (category != "" && (prob > best_prob || (prob == best_prob && category < best))) {
        best_prob = prob;
        best = category;
      }
    return best;
  }

  // Gives every declared category that has no probability yet a probability of
  // 0. Called after the winner is chosen, so it never changes the prediction.
  inline static void add_unpredicted(std::unordered_map<std::string, double>& probabilities, const Classes& classes) {
    for (const auto& category : classes.values)
      if (classes.category.find(probabilities, category) == probabilities.end()) probabilities.emplace(category, 0.0);
  }

  // Adds amount to the entry of `category`, however a segment spelled it: two
  // segments predicting "4" and "4.0" of a numeric target vote for one class.
  // The first spelling seen names it.
  inline static void accumulate(std::unordered_map<std::string, double>& probabilities, const std::string& category,
                                const double amount, const Classes& classes) {
    const auto it = classes.category.find(probabilities, category);
    if (it != probabilities.end())
      it->second += amount;
    else
      probabilities.emplace(category, amount);
  }

  inline static void normalize(std::unordered_map<std::string, double>& probabilities, const double total) {
    if (total <= 0) return;
    for (auto& [category, prob] : probabilities) prob /= total;
  }

  // ---- the aggregation boundary --------------------------------------------
  //
  // A sub-model says "no prediction" with an empty InternalScore from
  // score_raw, or "" from predict_raw. Every aggregating method goes through
  // the two steps below, which apply Segmentation/@missingPredictionTreatment
  // as JPMML 1.7.7 does:
  //
  // - Only selected segments (predicate true) take part. No selected segment
  //   means no prediction.
  // - returnMissing: one selected segment without a prediction voids the result.
  // - skipSegment: such a segment is dropped. If the weight of the dropped
  //   segments over the weight of the selected ones exceeds missingThreshold,
  //   there is no prediction. JPMML weighs by Segment@weight for every method,
  //   not only the weighted ones, and so does this.
  // - continue (the default): for majorityVote and weightedMajorityVote the
  //   segment votes for "no prediction", which wins when it has strictly more
  //   (weighted) votes than every class, or when the fraction above exceeds
  //   missingThreshold. For every other aggregating method one missing
  //   prediction voids the result.
  //
  // selectFirst, selectAll and modelChain do not aggregate and apply the
  // treatment in their own loops.

  static bool is_missing(const std::string& prediction) { return prediction.empty(); }
  static bool is_missing(const std::unique_ptr<InternalScore>& score) { return score->empty; }

  template <class R>
  struct Selected {
    const Segment* segment;
    R result;
  };

  // Every selected segment's result, in document order.
  //
  // A MULTITHREADING build scores the segments of an ensemble that
  // EnsembleModel marked parallel (Segments::parallel) in an OpenMP loop, and
  // every other ensemble -- and every ensemble nested inside a parallel one --
  // in the same plain loop as a sequential build, with no OpenMP region and no
  // per-segment allocation. Both loops give the same result: each segment is
  // scored on its own, the results are collected in document order, and a
  // failure is the one of the first segment, in document order, that fails.
  template <class R, class Eval>
  static std::vector<Selected<R>> evaluate_selected(const Sample& sample, const Segments& ensemble, Eval eval) {
#ifdef MULTITHREADING
    if (ensemble.parallel && !omp_in_parallel()) return evaluate_parallel<R>(sample, ensemble, eval);
#endif
    std::vector<Selected<R>> selected;
    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) selected.push_back({&segment, eval(segment, sample)});
    return selected;
  }

#ifdef MULTITHREADING
  template <class R, class Eval>
  static std::vector<Selected<R>> evaluate_parallel(const Sample& sample, const Segments& ensemble, Eval eval) {
    const long n = static_cast<long>(ensemble.size());
    std::vector<std::optional<R>> slots(ensemble.size());  // one allocation for the whole ensemble
    std::exception_ptr error = nullptr;
    long error_at = n;
#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for (long i = 0; i < n; i++) {
      try {
        // A worker thread scores in a StringScope of its own (the strings it
        // makes are its own, in its own arena); on the calling thread it
        // shares the record's.
        const cpmml::StringScope strings_scope(cpmml::StringScope::Scoring{}, *ensemble[i].model->strings);
        if (ensemble[i].predicate(sample)) slots[i].emplace(eval(ensemble[i], sample));
      } catch (...) {
        // Exceptions may not escape an OMP region (std::terminate). Keep the
        // one of the first failing segment, which is the one the sequential
        // loop would have thrown, and rethrow it after the barrier.
#pragma omp critical(cpmml_segment_error)
        if (i < error_at) {
          error_at = i;
          error = std::current_exception();
        }
      }
    }
    if (error) std::rethrow_exception(error);

    std::vector<Selected<R>> selected;
    selected.reserve(ensemble.size());
    for (long i = 0; i < n; i++)
      if (slots[i]) selected.push_back({&ensemble[i], std::move(*slots[i])});
    return selected;
  }
#endif

  template <class R>
  struct Usable {
    bool missing = false;              // the ensemble yields no prediction
    std::vector<Selected<R>> results;  // the results to aggregate, none of them missing
    double missing_votes = 0;          // continue + voting: the (weighted) votes for "no prediction"
  };

  template <class R>
  static Usable<R> apply_missing_prediction(std::vector<Selected<R>> selected, const MissingPrediction& mp,
                                            const bool voting, const bool weighted) {
    Usable<R> usable;
    if (selected.empty()) {
      usable.missing = true;
      return usable;
    }

    double selected_weight = 0;
    for (const auto& s : selected) selected_weight += s.segment->weight;

    double missing_weight = 0;
    for (auto& s : selected) {
      if (!is_missing(s.result)) {
        usable.results.push_back(std::move(s));
        continue;
      }
      if (mp.treatment == MissingPredictionTreatment::RETURN_MISSING ||
          (mp.treatment == MissingPredictionTreatment::CONTINUE && !voting)) {
        usable.missing = true;
        return usable;
      }
      missing_weight += s.segment->weight;
      if (missing_weight / selected_weight > mp.threshold) {
        usable.missing = true;
        return usable;
      }
      if (mp.treatment == MissingPredictionTreatment::CONTINUE)
        usable.missing_votes += weighted ? s.segment->weight : 1;
    }
    return usable;
  }

  static std::string predict(const Segment& segment, const Sample& sample) { return segment.predict(sample); }
  static std::unique_ptr<InternalScore> score(const Segment& segment, const Sample& sample) {
    return segment.score(sample);
  }

  // ---- segment OutputFields ---------------------------------------------------
  //
  // When a method's result is one segment's result -- selectFirst's selected
  // segment, modelChain's last executed segment -- that segment model's whole
  // result is returned next to the MiningModel's own OutputFields: the
  // segment's OutputFields and, when it is itself a MiningModel, whatever its
  // result carries, recursively. No other segment's OutputFields are returned.
  // selectAll returns every such field as a collection with one entry per
  // selected segment, in document order. The aggregating methods (sum,
  // average, majorityVote, ...) return no segment field. A segment field
  // named like one of the MiningModel's own OutputFields is not returned: the
  // MiningModel's own one stands, with its value or none (drop_shadowed).
  //
  // This is what JPMML 1.7.7 does, and what the spec allows: an engine "need
  // only return the output fields specified for the topmost MiningModel", and
  // of OutputFields duplicated across segments the one returned "comes from
  // the Segment selected by" the method.
  //
  // A segment without a prediction still returns its transformedValue fields
  // (missing or not); a field of another feature only when it has a value.
  //
  // selectAll needs every selected segment to return the same field names.
  // JPMML throws "Field sets ... do not match" when they differ (for instance
  // when one selected segment has a prediction and another none); cPMML keeps
  // the prediction and returns no segment field, which the spec allows.

  // One field of a segment's result, with the text a selectAll collection
  // writes for it. A missing value is written null, as JPMML does.
  struct ResultField {
    std::string name;
    bool missing = true;
    std::string text;
  };

  // The score of a MiningModel whose result is, or collects, segment results:
  // `fields` are the segment fields it returns (also in num_outputs and
  // str_outputs), with their collection text, so that an enclosing selectAll
  // can collect them.
  struct EnsembleScore : InternalScore {
    using InternalScore::InternalScore;
    explicit EnsembleScore(InternalScore&& score) : InternalScore(std::move(score)) {}
    std::vector<ResultField> fields;
  };

  // A numeric OutputField value as JPMML writes it in a collection, by the
  // field's dataType (java_kind): 7 for an integer, true/false for a boolean,
  // Java's float and double notation otherwise. The date and time types are
  // carried as doubles (see DataType) and are written as such; so is the value
  // of an OutputField without a dataType attribute, whatever its target's type.
  static std::string collection_entry(const double value, const OutputField& field) {
    return java::text({value, java_kind(field.datatype)});
  }

  // The fields a segment model's result returns: its own OutputFields, then
  // those its score carries from its own segments (a MiningModel), less any
  // its own OutputFields shadow.
  static std::vector<ResultField> fields_of(const InternalModel& model, const InternalScore& score) {
    std::vector<ResultField> fields;
    std::unordered_set<std::string> own;
    for (const auto& of : model.output.raw_outputfields) {
      if (!own.insert(of.name).second) continue;
      ResultField field{of.name};
      const auto s = score.str_outputs.find(of.name);
      const auto n = score.num_outputs.find(of.name);
      if (s != score.str_outputs.end() && !s->second.empty()) {
        field.missing = false;
        field.text = s->second;
      } else if (n != score.num_outputs.end()) {
        field.missing = false;
        field.text = collection_entry(n->second, of);
      }
      if (!field.missing || !score.empty ||
          of.expression_type.value == OutputExpressionType::OutputExpressionTypeValue::TRANSFORMED_VALUE)
        fields.push_back(std::move(field));
    }
    if (const auto* carried = dynamic_cast<const EnsembleScore*>(&score))
      for (const auto& field : carried->fields)
        if (!own.count(field.name)) fields.push_back(field);
    return fields;
  }

  // `score`, a segment's result, as the result of the MiningModel: it keeps
  // the segment's fields (see fields_of).
  static std::unique_ptr<InternalScore> carrying(const Segment& segment, std::unique_ptr<InternalScore> score) {
    if (!segment.has_outputs) return score;
    std::vector<ResultField> fields = fields_of(*segment.model, *score);
    auto result = std::make_unique<EnsembleScore>(std::move(*score));
    result->fields = std::move(fields);
    return result;
  }

  // The MiningModel's own OutputFields shadow the segment fields of the same
  // name, even when they have no value: called on the method's result before
  // the MiningModel adds its own.
  static void drop_shadowed(InternalScore& score, const OutputDictionary& output) {
    if (output.empty || (score.num_outputs.empty() && score.str_outputs.empty())) return;
    for (const auto& of : output.raw_outputfields) {
      score.num_outputs.erase(of.name);
      score.str_outputs.erase(of.name);
    }
    if (auto* carried = dynamic_cast<EnsembleScore*>(&score)) {
      auto& fields = carried->fields;
      fields.erase(std::remove_if(fields.begin(), fields.end(),
                                  [&output](const ResultField& field) { return output.contains(field.name); }),
                   fields.end());
    }
  }

  // selectAll: when every selected segment returns the same field names, each
  // field as a collection, "[entry, entry, ...]", one entry per selected
  // segment in document order. Otherwise none (see above).
  static void add_collections(EnsembleScore& score, const std::vector<std::vector<ResultField>>& selected) {
    if (selected.empty() || selected.front().empty()) return;
    const auto names_of = [](const std::vector<ResultField>& fields) {
      std::unordered_set<std::string> names;
      for (const auto& field : fields) names.insert(field.name);
      return names;
    };
    const auto names = names_of(selected.front());
    for (const auto& fields : selected)
      if (names_of(fields) != names) return;

    for (const auto& name : selected.front()) {
      std::string text = "[";
      for (size_t i = 0; i < selected.size(); i++) {
        const auto it = std::find_if(selected[i].begin(), selected[i].end(),
                                     [&name](const ResultField& field) { return field.name == name.name; });
        if (i > 0) text += ", ";
        text += it->missing ? "null" : it->text;
      }
      text += "]";
      score.str_outputs[name.name] = text;
      score.fields.push_back({name.name, false, text});
    }
  }

  // ---- methods --------------------------------------------------------------

  inline static std::unique_ptr<InternalScore> vote(const Sample& sample, const Segments& ensemble,
                                                    const MissingPrediction& mp, const bool weighted,
                                                    const Classes& classes) {
    auto usable =
        apply_missing_prediction(evaluate_selected<std::string>(sample, ensemble, predict), mp, true, weighted);
    // Every selected segment skipped or voting for no prediction: JPMML throws
    // "Empty classification"; no prediction is the answer that does not throw.
    if (usable.missing || usable.results.empty()) return std::make_unique<InternalScore>();

    std::unordered_map<std::string, double> probabilities;  // zero initialized
    double total = 0;
    for (const auto& r : usable.results) {
      const double w = weighted ? r.segment->weight : 1.0;
      accumulate(probabilities, r.result, w, classes);
      total += w;
    }

    double most = 0;
    for (const auto& [category, votes] : probabilities) most = std::max(most, votes);
    if (usable.missing_votes > most) return std::make_unique<InternalScore>();

    normalize(probabilities, total);
    const std::string winner = argmax(probabilities);
    add_unpredicted(probabilities, classes);
    return std::make_unique<InternalScore>(winner, probabilities);
  }

  inline static std::unique_ptr<InternalScore> classification_average(const Sample& sample,
                                                                      const Segments& ensemble,
                                                                      const MissingPrediction& mp,
                                                                      const bool weighted,
                                                                      const Classes& classes) {
    auto usable = apply_missing_prediction(evaluate_selected<std::unique_ptr<InternalScore>>(sample, ensemble, score),
                                           mp, false, weighted);
    if (usable.missing || usable.results.empty()) return std::make_unique<InternalScore>();

    std::unordered_map<std::string, double> probabilities;
    double total = 0;
    for (const auto& r : usable.results) {
      const double w = weighted ? r.segment->weight : 1.0;
      for (const auto& probability : r.result->probabilities)
        accumulate(probabilities, probability.first, probability.second * w, classes);
      total += w;
    }

    normalize(probabilities, total);
    const std::string winner = argmax(probabilities);
    add_unpredicted(probabilities, classes);
    return std::make_unique<InternalScore>(winner, probabilities);
  }

  inline static std::unique_ptr<InternalScore> regression_average(const Sample& sample,
                                                                  const Segments& ensemble,
                                                                  const MissingPrediction& mp, const bool weighted) {
    auto usable =
        apply_missing_prediction(evaluate_selected<std::string>(sample, ensemble, predict), mp, false, weighted);
    if (usable.missing || usable.results.empty()) return std::make_unique<InternalScore>();

    double score = 0;
    double total = 0;
    for (const auto& r : usable.results) {
      const double w = weighted ? r.segment->weight : 1.0;
      score += weighted ? from_java_text(r.result) * w : from_java_text(r.result);
      total += w;
    }

    if (total == 0) return std::make_unique<InternalScore>();
    return std::make_unique<InternalScore>(score / total);
  }

  static std::vector<double> values_of(const std::vector<Selected<std::string>>& results) {
    std::vector<double> values;
    values.reserve(results.size());
    for (const auto& r : results) values.push_back(from_java_text(r.result));
    return values;
  }

  inline static std::unique_ptr<InternalScore> median(const Sample& sample, const Segments& ensemble,
                                                      const MissingPrediction& mp) {
    auto usable = apply_missing_prediction(evaluate_selected<std::string>(sample, ensemble, predict), mp, false, false);
    if (usable.missing || usable.results.empty()) return std::make_unique<InternalScore>();

    std::vector<double> values = values_of(usable.results);
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double med = values[mid];
    if (values.size() % 2 == 0) {
      double lo = *std::max_element(values.begin(), values.begin() + mid);
      med = (lo + med) / 2.0;
    }
    return std::make_unique<InternalScore>(med);
  }

  // weightedMedian, as the spec defines it: for values x1 < ... < xn with
  // positive weights adding up to 1, the xk such that the weights before it
  // and the weights after it each add up to at most 1/2; when two values
  // qualify, their mean.
  //
  // Computed as JPMML 1.7.7 does -- the usable segments' values sorted, ties
  // in document order, the first one at which the running weight reaches half
  // the total weight -- except where JPMML departs from the spec: when the
  // running weight is half the total there (up to rounding), the next value with a
  // positive weight qualifies too, and the result is the mean of the two
  // (JPMML returns the lower one). A segment of weight 0 adds nothing and is
  // never the result. When the usable segments' weights add up to 0 there is
  // no weighted median and no prediction (JPMML returns the smallest value).
  //
  // A segment without a prediction is handled as by median, missingThreshold
  // weighing by Segment@weight; every segment skipped is no prediction (JPMML
  // throws "Undefined result"). A negative Segment@weight is refused at load
  // (EnsembleModel).
  inline static std::unique_ptr<InternalScore> weighted_median(const Sample& sample,
                                                               const Segments& ensemble,
                                                               const MissingPrediction& mp) {
    auto usable = apply_missing_prediction(evaluate_selected<std::unique_ptr<InternalScore>>(sample, ensemble, score),
                                           mp, false, true);
    if (usable.missing || usable.results.empty()) return std::make_unique<InternalScore>();

    struct Entry {
      double value;
      double weight;
    };
    std::vector<Entry> entries;
    entries.reserve(usable.results.size());
    double total = 0;
    for (const auto& r : usable.results) {
      entries.push_back({r.result->double_score, r.segment->weight});
      total += r.segment->weight;
    }
    if (!(total > 0)) return std::make_unique<InternalScore>();

    std::stable_sort(entries.begin(), entries.end(),
                     [](const Entry& lhs, const Entry& rhs) { return lhs.value < rhs.value; });
    // Whether the running weight is exactly half the total is decided up to
    // rounding: the total adds the weights in document order and the running
    // weight in value order, so decimal weights such as .1, .2, .3 can miss
    // half by an ulp in either direction and flip between the mean and a
    // single value with the order of the inputs.
    const double half = 0.5 * total;
    const double tolerance = 1e-9 * total;
    double running = 0;
    for (size_t k = 0; k < entries.size(); k++) {
      running += entries[k].weight;
      if (running < half - tolerance) continue;
      if (running <= half + tolerance)
        for (size_t next = k + 1; next < entries.size(); next++)
          if (entries[next].weight > 0)
            return std::make_unique<InternalScore>((entries[k].value + entries[next].value) / 2.0);
      return std::make_unique<InternalScore>(entries[k].value);
    }
    return std::make_unique<InternalScore>();  // unreachable: running ends at total
  }

  // The place of `category` among the target's declared categories; past them
  // all when it is not declared.
  static size_t rank(const std::string& category, const Classes& classes) {
    for (size_t i = 0; i < classes.values.size(); i++)
      if (classes.category.same(classes.values[i], category)) return i;
    return classes.values.size();
  }

  // max for classification. The spec: max is "applied to the predicted
  // probabilities of target categories in each of the models used for the
  // case, then the winning category is selected based on the highest combined
  // probability", and its probabilities: "consider the model(s) that have
  // contributed the chosen probability for the winning category. Return their
  // average probabilities". As in JPMML 1.7.7:
  //
  // - The winner is the category of the highest probability any usable
  //   segment gives. A tie between categories goes to the one declared first
  //   in the target's DataField; an undeclared one comes after the declared
  //   ones, and among them by name.
  // - The probabilities are the plain average, Segment@weight ignored, of the
  //   segments that give the winner that probability.
  // - A segment without a prediction is handled as by average: continue and
  //   returnMissing void the result, skipSegment drops it (missingThreshold
  //   weighs by Segment@weight). No usable segment is no prediction (JPMML
  //   throws "Empty classification").
  //
  // JPMML departs in one case cPMML cannot follow: when a segment chosen above
  // gives a second category the same, highest probability, JPMML predicts
  // whichever of the two that segment's model lists first (its
  // ScoreDistribution order); cPMML keeps the winner chosen above. Without
  // declared categories JPMML breaks every tie in that model order, cPMML by
  // name.
  inline static std::unique_ptr<InternalScore> classification_max(const Sample& sample,
                                                                  const Segments& ensemble,
                                                                  const MissingPrediction& mp, const Classes& classes) {
    auto usable = apply_missing_prediction(evaluate_selected<std::unique_ptr<InternalScore>>(sample, ensemble, score),
                                           mp, false, false);
    if (usable.missing || usable.results.empty()) return std::make_unique<InternalScore>();

    std::string winner;
    double highest = -1;
    size_t winner_rank = 0;
    for (const auto& r : usable.results)
      for (const auto& [category, prob] : r.result->probabilities) {
        if (category.empty() || prob < highest) continue;
        const size_t category_rank = rank(category, classes);
        if (prob > highest || category_rank < winner_rank ||
            (category_rank == winner_rank && category < winner && !classes.category.same(category, winner))) {
          highest = prob;
          winner = category;
          winner_rank = category_rank;
        }
      }
    if (winner.empty()) return std::make_unique<InternalScore>();  // no segment gave a probability

    std::unordered_map<std::string, double> probabilities;
    double contributors = 0;
    for (const auto& r : usable.results) {
      const auto& given = r.result->probabilities;
      const auto it = classes.category.find(given, winner);
      if (it == given.end() || it->second != highest) continue;
      for (const auto& probability : given) accumulate(probabilities, probability.first, probability.second, classes);
      contributors++;
    }

    normalize(probabilities, contributors);
    const std::string predicted = classes.category.find(probabilities, winner)->first;
    add_unpredicted(probabilities, classes);
    return std::make_unique<InternalScore>(predicted, probabilities);
  }

  // A sum over no selected segment is no prediction. A sum whose selected
  // segments were all skipped (skipSegment) is 0, as in JPMML.
  inline static std::unique_ptr<InternalScore> sum(const Sample& sample, const Segments& ensemble,
                                                   const MissingPrediction& mp) {
    auto usable = apply_missing_prediction(evaluate_selected<std::unique_ptr<InternalScore>>(sample, ensemble, score),
                                           mp, false, false);
    if (usable.missing) return std::make_unique<InternalScore>();

    double score = 0;
    for (const auto& r : usable.results) score += r.result->double_score;
    return std::make_unique<InternalScore>(score);
  }

  // weightedSum: the sum of each usable segment's prediction times its
  // Segment@weight, in document order, as JPMML 1.7.7 adds them. A segment
  // without a prediction is handled as by sum, missingThreshold weighing by
  // Segment@weight; every segment skipped gives 0, as in JPMML. A negative
  // Segment@weight is refused at load (EnsembleModel).
  inline static std::unique_ptr<InternalScore> weighted_sum(const Sample& sample, const Segments& ensemble,
                                                            const MissingPrediction& mp) {
    auto usable = apply_missing_prediction(evaluate_selected<std::unique_ptr<InternalScore>>(sample, ensemble, score),
                                           mp, false, true);
    if (usable.missing) return std::make_unique<InternalScore>();

    double score = 0;
    for (const auto& r : usable.results) score += r.segment->weight * r.result->double_score;
    return std::make_unique<InternalScore>(score);
  }

  // The first selected segment's result stands, even when it has none; only
  // skipSegment moves on to the next selected segment. Its OutputFields come
  // with it, also when it has no prediction (see "segment OutputFields").
  inline static std::unique_ptr<InternalScore> select_first(const Sample& sample, const Segments& ensemble,
                                                            const MissingPrediction& mp) {
    for (const auto& segment : ensemble) {
      if (!segment.predicate(sample)) continue;
      std::unique_ptr<InternalScore> score = segment.result(sample);
      if (!score->empty) return carrying(segment, std::move(score));
      if (mp.treatment == MissingPredictionTreatment::SKIP_SEGMENT) continue;
      if (!segment.has_outputs) break;
      // No prediction, so a fresh empty score: only the fields carry over.
      auto none = std::make_unique<EnsembleScore>();
      none->fields = fields_of(*segment.model, *score);
      none->num_outputs = std::move(score->num_outputs);
      none->str_outputs = std::move(score->str_outputs);
      return none;
    }
    return std::make_unique<InternalScore>();
  }

  // selectAll: collect every active segment's prediction and format as PMML
  // Collection string "[pred1, pred2, ...]", each as the segment writes it
  // (java_format.h). A segment without a prediction is written null, as JPMML
  // writes a missing entry (it was left empty, "[, ]"), or under returnMissing
  // voids the result. skipSegment, which JPMML refuses here, acts as continue. The
  // segments' OutputFields are returned as collections (add_collections); a
  // segment is scored for them, on top of its prediction, only when it has any.
  inline static std::unique_ptr<InternalScore> select_all(const Sample& sample, const Segments& ensemble,
                                                          const MissingPrediction& mp) {
    std::string result = "[";
    bool first = true;
    bool with_fields = false;
    std::vector<std::vector<ResultField>> fields;
    for (const auto& segment : ensemble) {
      if (segment.predicate(sample)) {
        const std::string prediction = segment.predict(sample);
        if (prediction.empty() && mp.treatment == MissingPredictionTreatment::RETURN_MISSING)
          return std::make_unique<InternalScore>();
        if (!first) result += ", ";
        result += prediction.empty() ? "null" : prediction;
        first = false;
        if (segment.has_outputs) {
          with_fields = true;
          fields.push_back(fields_of(*segment.model, *segment.result(sample)));
        } else {
          fields.emplace_back();
        }
      }
    }
    result += "]";
    if (!with_fields) return std::make_unique<InternalScore>(result);
    auto score = std::make_unique<EnsembleScore>(result);
    add_collections(*score, fields);
    return score;
  }

  // modelChain: "Segments whose Predicates evaluate to TRUE are executed in the
  // order they appear", and the result is that of the last one executed. A
  // predicate may read the OutputFields of the segments before it, so segment
  // i's predicate is evaluated only after every earlier selected segment has
  // run and written its target and OutputFields into the sample. An OutputField
  // of a segment that did not run stays missing, as the spec says. (JPMML 1.7.7
  // throws "Field ... is not defined" there instead.)
  //
  // Whether a segment is the last one executed is known only once every
  // predicate after it has been evaluated, which needs its outputs. So each
  // selected segment runs as an intermediate step (augment), and if it then
  // turns out to be the last, it is run again as the final step (augment_last)
  // on the sample as it was before its intermediate run. That keeps the final
  // step's input exactly what it was before modelChain ran in document order.
  // The segment last in the document is final whenever it is selected and runs
  // only once.
  //
  // Each step evaluates its own LocalTransformations, which are gone again when
  // the step ends: the next segments read its target and OutputFields only.
  //
  // The result carries the last executed segment's OutputFields (see "segment
  // OutputFields"), not those of the segments before it.
  //
  // A segment without a prediction leaves its target and OutputFields missing
  // for the segments after it (continue), or voids the chain (returnMissing).
  // skipSegment, which JPMML refuses here, acts as continue. JPMML also throws
  // when such a segment declares an OutputField; the spec's "processing should
  // continue normally" is followed instead.
  inline static std::unique_ptr<InternalScore> model_chain(const Sample& sample, const Segments& ensemble,
                                                           const MissingPrediction& mp) {
    Sample tmp_sample = sample;
    Sample before_last;                // tmp_sample as it was before `last` ran
    const Segment* last = nullptr;     // the last segment executed so far

    for (size_t i = 0; i < ensemble.size(); i++) {
      const Segment& segment = ensemble[i];
      if (!segment.predicate(tmp_sample)) continue;
      if (i + 1 == ensemble.size()) return carrying(segment, segment.model->augment_last(tmp_sample));

      before_last = tmp_sample;
      const bool predicted = segment.model->augment(tmp_sample);
      if (!predicted && mp.treatment == MissingPredictionTreatment::RETURN_MISSING)
        return std::make_unique<InternalScore>();
      last = &segment;
    }

    if (!last) return std::make_unique<InternalScore>();  // no segment selected
    tmp_sample = std::move(before_last);
    return carrying(*last, last->model->augment_last(tmp_sample));
  }
};

#endif
