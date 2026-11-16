
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MULTIPLEMODELMETHOD_H
#define CPMML_MULTIPLEMODELMETHOD_H

#include <algorithm>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef MULTITHREADING
#include <omp.h>
#endif

#include "core/enum_parser.h"
#include "core/internal_score.h"
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
 * - weightedAverage
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
    MAX,
    SUM,
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

  using Function = std::function<std::unique_ptr<InternalScore>(const Sample&, const std::vector<Segment>&)>;

  MultipleModelMethodType value = MultipleModelMethodType::MAJORITY_VOTE;
  MissingPrediction missing_prediction;
  Function function;

  MultipleModelMethod() = default;

  MultipleModelMethod(const std::string& multiplemodelmethod, const MiningFunction& mining_function)
      : MultipleModelMethod(multiplemodelmethod, mining_function, MissingPrediction{}) {}

  MultipleModelMethod(const std::string& multiplemodelmethod, const MiningFunction& mining_function,
                      const MissingPrediction& missing_prediction)
      : value(from_string(multiplemodelmethod)),
        missing_prediction(missing_prediction),
        function(to_function(multiplemodelmethod, mining_function, missing_prediction)) {}

  static constexpr EnumEntry<MultipleModelMethodType> TABLE[] = {
      {"majorityvote", MultipleModelMethodType::MAJORITY_VOTE, "majorityVote"},
      {"weightedmajorityvote", MultipleModelMethodType::WEIGHTED_MAJORITY_VOTE, "weightedMajorityVote"},
      {"average", MultipleModelMethodType::AVERAGE, "average"},
      {"weightedaverage", MultipleModelMethodType::WEIGHTED_AVERAGE, "weightedAverage"},
      {"median", MultipleModelMethodType::MEDIAN, "median"},
      {"max", MultipleModelMethodType::MAX, "max"},
      {"sum", MultipleModelMethodType::SUM, "sum"},
      {"selectfirst", MultipleModelMethodType::SELECT_FIRST, "selectFirst"},
      {"selectall", MultipleModelMethodType::SELECT_ALL, "selectAll"},
      {"modelchain", MultipleModelMethodType::MODEL_CHAIN, "modelChain"},
  };

  static constexpr EnumEntry<MissingPredictionTreatment> MISSING_PREDICTION_TABLE[] = {
      {"returnmissing", MissingPredictionTreatment::RETURN_MISSING, "returnMissing"},
      {"skipsegment", MissingPredictionTreatment::SKIP_SEGMENT, "skipSegment"},
      {"continue", MissingPredictionTreatment::CONTINUE, "continue"},
  };

  // Segmentation/@multipleModelMethod is required.
  static MultipleModelMethodType from_string(const std::string& multiplemodelmethod) {
    return enum_value(multiplemodelmethod, TABLE, "multipleModelMethod");
  }

  // Segmentation/@missingPredictionTreatment (default continue) and
  // @missingThreshold (default 1).
  static MissingPrediction missing_prediction_from(const XmlNode& segmentation) {
    MissingPrediction result;
    result.treatment =
        enum_value(segmentation.get_attribute("missingPredictionTreatment", ""), MISSING_PREDICTION_TABLE,
                   "Segmentation@missingPredictionTreatment", MissingPredictionTreatment::CONTINUE);
    result.threshold = segmentation.get_double_attribute("missingThreshold", 1);
    return result;
  }

  std::string to_string() const { return enum_display(value, TABLE, "UNDEFINED"); }

  static Function to_function(const std::string& multiplemodelmethod, const MiningFunction& mining_function,
                              const MissingPrediction& mp) {
    // The spec says skipSegment "should not be used with modelChain", and JPMML
    // 1.7.7 refuses to score it there and with selectAll. cPMML keeps scoring
    // those documents as it did before the attribute was read: such a segment
    // is treated as under continue (select_all, model_chain).
    switch (from_string(multiplemodelmethod)) {
      case MultipleModelMethodType::MAJORITY_VOTE:
        return [mp](const Sample& s, const std::vector<Segment>& e) { return vote(s, e, mp, false); };
      case MultipleModelMethodType::WEIGHTED_MAJORITY_VOTE:
        return [mp](const Sample& s, const std::vector<Segment>& e) { return vote(s, e, mp, true); };
      case MultipleModelMethodType::AVERAGE:
        switch (mining_function.value) {
          case MiningFunction::MiningFunctionType::CLASSIFICATION:
            return [mp](const Sample& s, const std::vector<Segment>& e) {
              return classification_average(s, e, mp, false);
            };
          case MiningFunction::MiningFunctionType::REGRESSION:
            return [mp](const Sample& s, const std::vector<Segment>& e) { return regression_average(s, e, mp, false); };
          default:
            throw cpmml::ParsingException("average not supported for this mining function");
        }
      case MultipleModelMethodType::WEIGHTED_AVERAGE:
        switch (mining_function.value) {
          case MiningFunction::MiningFunctionType::CLASSIFICATION:
            return
                [mp](const Sample& s, const std::vector<Segment>& e) { return classification_average(s, e, mp, true); };
          case MiningFunction::MiningFunctionType::REGRESSION:
            return [mp](const Sample& s, const std::vector<Segment>& e) { return regression_average(s, e, mp, true); };
          default:
            throw cpmml::ParsingException("weightedAverage not supported for this mining function");
        }
      case MultipleModelMethodType::MEDIAN:
        return [mp](const Sample& s, const std::vector<Segment>& e) { return median(s, e, mp); };
      case MultipleModelMethodType::MAX:
        return [mp](const Sample& s, const std::vector<Segment>& e) { return max(s, e, mp); };
      case MultipleModelMethodType::SUM:
        return [mp](const Sample& s, const std::vector<Segment>& e) { return sum(s, e, mp); };
      case MultipleModelMethodType::SELECT_FIRST:
        return [mp](const Sample& s, const std::vector<Segment>& e) { return select_first(s, e, mp); };
      case MultipleModelMethodType::SELECT_ALL:
        return [mp](const Sample& s, const std::vector<Segment>& e) { return select_all(s, e, mp); };
      case MultipleModelMethodType::MODEL_CHAIN:
        return [mp](const Sample& s, const std::vector<Segment>& e) { return model_chain(s, e, mp); };
    }

    throw cpmml::ParsingException(multiplemodelmethod + " not supported");
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
  template <class R, class Eval>
  static std::vector<Selected<R>> evaluate_selected(const Sample& sample, const std::vector<Segment>& ensemble,
                                                    Eval eval) {
    std::vector<Selected<R>> selected;
#ifndef MULTITHREADING
    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) selected.push_back({&segment, eval(segment, sample)});
#else
    std::vector<std::unique_ptr<R>> slots(ensemble.size());
    std::exception_ptr worker_error = nullptr;
#pragma omp parallel for if (ensemble.size() > 25) default(shared) num_threads(NUM_THREADS)
    for (auto i = 0u; i < ensemble.size(); i++) {
      try {
        // OMP worker threads have their own thread-local active_indexer;
        // without this scope string values would be interned in the wrong map.
        Value::ScopedIndexer _scope(ensemble[i].model->indexer.get());
        if (ensemble[i].predicate(sample)) slots[i] = std::make_unique<R>(eval(ensemble[i], sample));
      } catch (...) {
        // Exceptions may not escape an OMP region (std::terminate); capture
        // the first one and rethrow after the barrier.
#pragma omp critical
        if (!worker_error) worker_error = std::current_exception();
      }
    }
    if (worker_error) std::rethrow_exception(worker_error);

    for (auto i = 0u; i < ensemble.size(); i++)
      if (slots[i]) selected.push_back({&ensemble[i], std::move(*slots[i])});
#endif
    return selected;
  }

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

  // ---- methods --------------------------------------------------------------

  inline static std::unique_ptr<InternalScore> vote(const Sample& sample, const std::vector<Segment>& ensemble,
                                                    const MissingPrediction& mp, const bool weighted) {
    auto usable =
        apply_missing_prediction(evaluate_selected<std::string>(sample, ensemble, predict), mp, true, weighted);
    // Every selected segment skipped or voting for no prediction: JPMML throws
    // "Empty classification"; no prediction is the answer that does not throw.
    if (usable.missing || usable.results.empty()) return std::make_unique<InternalScore>();

    std::unordered_map<std::string, double> probabilities;  // zero initialized
    double total = 0;
    for (const auto& r : usable.results) {
      const double w = weighted ? r.segment->weight : 1.0;
      probabilities[r.result] += w;
      total += w;
    }

    double most = 0;
    for (const auto& [category, votes] : probabilities) most = std::max(most, votes);
    if (usable.missing_votes > most) return std::make_unique<InternalScore>();

    normalize(probabilities, total);
    return std::make_unique<InternalScore>(argmax(probabilities), probabilities);
  }

  inline static std::unique_ptr<InternalScore> classification_average(const Sample& sample,
                                                                      const std::vector<Segment>& ensemble,
                                                                      const MissingPrediction& mp,
                                                                      const bool weighted) {
    auto usable = apply_missing_prediction(evaluate_selected<std::unique_ptr<InternalScore>>(sample, ensemble, score),
                                           mp, false, weighted);
    if (usable.missing || usable.results.empty()) return std::make_unique<InternalScore>();

    std::unordered_map<std::string, double> probabilities;
    double total = 0;
    for (const auto& r : usable.results) {
      const double w = weighted ? r.segment->weight : 1.0;
      for (const auto& probability : r.result->probabilities)
        probabilities[probability.first] += probability.second * w;
      total += w;
    }

    normalize(probabilities, total);
    return std::make_unique<InternalScore>(argmax(probabilities), probabilities);
  }

  inline static std::unique_ptr<InternalScore> regression_average(const Sample& sample,
                                                                  const std::vector<Segment>& ensemble,
                                                                  const MissingPrediction& mp, const bool weighted) {
    auto usable =
        apply_missing_prediction(evaluate_selected<std::string>(sample, ensemble, predict), mp, false, weighted);
    if (usable.missing || usable.results.empty()) return std::make_unique<InternalScore>();

    double score = 0;
    double total = 0;
    for (const auto& r : usable.results) {
      const double w = weighted ? r.segment->weight : 1.0;
      score += weighted ? to_double(r.result) * w : to_double(r.result);
      total += w;
    }

    if (total == 0) return std::make_unique<InternalScore>();
    return std::make_unique<InternalScore>(score / total);
  }

  static std::vector<double> values_of(const std::vector<Selected<std::string>>& results) {
    std::vector<double> values;
    values.reserve(results.size());
    for (const auto& r : results) values.push_back(to_double(r.result));
    return values;
  }

  inline static std::unique_ptr<InternalScore> median(const Sample& sample, const std::vector<Segment>& ensemble,
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

  inline static std::unique_ptr<InternalScore> max(const Sample& sample, const std::vector<Segment>& ensemble,
                                                   const MissingPrediction& mp) {
    auto usable = apply_missing_prediction(evaluate_selected<std::string>(sample, ensemble, predict), mp, false, false);
    if (usable.missing || usable.results.empty()) return std::make_unique<InternalScore>();

    const std::vector<double> values = values_of(usable.results);
    return std::make_unique<InternalScore>(*std::max_element(values.begin(), values.end()));
  }

  // A sum over no selected segment is no prediction. A sum whose selected
  // segments were all skipped (skipSegment) is 0, as in JPMML.
  inline static std::unique_ptr<InternalScore> sum(const Sample& sample, const std::vector<Segment>& ensemble,
                                                   const MissingPrediction& mp) {
    auto usable = apply_missing_prediction(evaluate_selected<std::unique_ptr<InternalScore>>(sample, ensemble, score),
                                           mp, false, false);
    if (usable.missing) return std::make_unique<InternalScore>();

    double score = 0;
    for (const auto& r : usable.results) score += r.result->double_score;
    return std::make_unique<InternalScore>(score);
  }

  // The first selected segment's result stands, even when it has none; only
  // skipSegment moves on to the next selected segment.
  inline static std::unique_ptr<InternalScore> select_first(const Sample& sample, const std::vector<Segment>& ensemble,
                                                            const MissingPrediction& mp) {
    for (const auto& segment : ensemble) {
      if (segment.predicate(sample)) {
        auto score = segment.score(sample);
        if (!score->empty) return score;
        if (mp.treatment != MissingPredictionTreatment::SKIP_SEGMENT) break;
      }
    }
    return std::make_unique<InternalScore>();
  }

  // selectAll: collect every active segment's prediction and format as PMML
  // Collection string "[pred1, pred2, ...]". A segment without a prediction
  // leaves an empty entry (JPMML writes null), or under returnMissing voids the
  // result. skipSegment, which JPMML refuses here, acts as continue.
  inline static std::unique_ptr<InternalScore> select_all(const Sample& sample, const std::vector<Segment>& ensemble,
                                                          const MissingPrediction& mp) {
    std::string result = "[";
    bool first = true;
    for (const auto& segment : ensemble) {
      if (segment.predicate(sample)) {
        const std::string prediction = segment.predict(sample);
        if (prediction.empty() && mp.treatment == MissingPredictionTreatment::RETURN_MISSING)
          return std::make_unique<InternalScore>();
        if (!first) result += ", ";
        result += prediction;
        first = false;
      }
    }
    result += "]";
    return std::make_unique<InternalScore>(result);
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
  // A segment without a prediction leaves its target and OutputFields missing
  // for the segments after it (continue), or voids the chain (returnMissing).
  // skipSegment, which JPMML refuses here, acts as continue. JPMML also throws
  // when such a segment declares an OutputField; the spec's "processing should
  // continue normally" is followed instead.
  inline static std::unique_ptr<InternalScore> model_chain(const Sample& sample, const std::vector<Segment>& ensemble,
                                                           const MissingPrediction& mp) {
    Sample tmp_sample = sample;
    Sample before_last;                // tmp_sample as it was before `last` ran
    const Segment* last = nullptr;     // the last segment executed so far

    for (size_t i = 0; i < ensemble.size(); i++) {
      const Segment& segment = ensemble[i];
      if (!segment.predicate(tmp_sample)) continue;
      if (i + 1 == ensemble.size()) return segment.model->augment_last(tmp_sample);

      before_last = tmp_sample;
      const bool predicted = last ? segment.model->augment(tmp_sample) : segment.model->augment_first(tmp_sample);
      if (!predicted && mp.treatment == MissingPredictionTreatment::RETURN_MISSING)
        return std::make_unique<InternalScore>();
      last = &segment;
    }

    if (!last) return std::make_unique<InternalScore>();  // no segment selected
    tmp_sample = std::move(before_last);
    return last->model->augment_last(tmp_sample);
  }
};

#endif
