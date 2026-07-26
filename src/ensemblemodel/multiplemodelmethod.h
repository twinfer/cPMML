
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_MULTIPLEMODELMETHOD_H
#define CPMML_MULTIPLEMODELMETHOD_H

#include <string>
#include <unordered_map>
#include <vector>

#ifdef MULTITHREADING
#include <omp.h>
#endif

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

  MultipleModelMethodType value;
  std::function<std::unique_ptr<InternalScore>(const Sample&, const std::vector<Segment>&)> function;

  MultipleModelMethod() = default;

  explicit MultipleModelMethod(const std::string& multiplemodelmethod, const MiningFunction& mining_function)
      : value(from_string(multiplemodelmethod)), function(to_function(multiplemodelmethod, mining_function)) {}

  static MultipleModelMethodType from_string(const std::string& multiplemodelmethod) {
    static const std::unordered_map<std::string, MultipleModelMethodType> multiplemodelmethod_converter = {
        {"majorityvote", MultipleModelMethodType::MAJORITY_VOTE},
        {"weightedmajorityvote", MultipleModelMethodType::WEIGHTED_MAJORITY_VOTE},
        {"average", MultipleModelMethodType::AVERAGE},
        {"weightedaverage", MultipleModelMethodType::WEIGHTED_AVERAGE},
        {"median", MultipleModelMethodType::MEDIAN},
        {"max", MultipleModelMethodType::MAX},
        {"sum", MultipleModelMethodType::SUM},
        {"selectfirst", MultipleModelMethodType::SELECT_FIRST},
        {"selectall", MultipleModelMethodType::SELECT_ALL},
        {"modelchain", MultipleModelMethodType::MODEL_CHAIN}};

    try {
      return multiplemodelmethod_converter.at(::to_lower(multiplemodelmethod));
    } catch (const std::out_of_range& e) {
      throw cpmml::ParsingException(multiplemodelmethod + " not supported");
    }
  }

  static std::function<std::unique_ptr<InternalScore>(const Sample&, const std::vector<Segment>&)> to_function(
      const std::string& multiplemodelmethod, const MiningFunction& mining_function) {
    switch (from_string(multiplemodelmethod)) {
      case MultipleModelMethodType::MAJORITY_VOTE:
        return majority_vote;
      case MultipleModelMethodType::WEIGHTED_MAJORITY_VOTE:
        return weighted_majority_vote;
      case MultipleModelMethodType::AVERAGE:
        switch (mining_function.value) {
          case MiningFunction::MiningFunctionType::CLASSIFICATION:
            return classification_average;
          case MiningFunction::MiningFunctionType::REGRESSION:
            return regression_average;
          default:
            throw cpmml::ParsingException("average not supported for this mining function");
        }
      case MultipleModelMethodType::WEIGHTED_AVERAGE:
        switch (mining_function.value) {
          case MiningFunction::MiningFunctionType::CLASSIFICATION:
            return classification_weighted_average;
          case MiningFunction::MiningFunctionType::REGRESSION:
            return regression_weighted_average;
          default:
            throw cpmml::ParsingException("weightedAverage not supported for this mining function");
        }
      case MultipleModelMethodType::MEDIAN:
        return median;
      case MultipleModelMethodType::MAX:
        return max;
      case MultipleModelMethodType::SUM:
        return sum;
      case MultipleModelMethodType::SELECT_FIRST:
        return select_first;
      case MultipleModelMethodType::SELECT_ALL:
        return select_all;
      case MultipleModelMethodType::MODEL_CHAIN:
        return model_chain;
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

#ifndef MULTITHREADING

  static std::unique_ptr<InternalScore> majority_vote(const Sample& sample, const std::vector<Segment>& ensemble) {
    std::unordered_map<std::string, double> probabilities;  // zero initialized

    double votes = 0;
    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) {
        const std::string prediction = segment.predict(sample);
        if (prediction.empty()) continue;  // null prediction casts no vote
        probabilities[prediction] += 1.0;
        votes++;
      }

    normalize(probabilities, votes);
    return std::make_unique<InternalScore>(argmax(probabilities), probabilities);
  }

#else
  inline static std::unique_ptr<InternalScore> majority_vote(const Sample& sample,
                                                             const std::vector<Segment>& ensemble) {
    std::unordered_map<std::string, double> probabilities;  // zero initialized
    std::unordered_map<std::string, double> tmp_probabilities[NUM_THREADS];

    std::exception_ptr worker_error = nullptr;
#pragma omp parallel for if (ensemble.size() > 25) default(shared) num_threads(NUM_THREADS)
    for (auto i = 0u; i < ensemble.size(); i++) {
      try {
        // OMP worker threads have their own thread-local active_indexer;
        // without this scope string values would be interned in the wrong map.
        Value::ScopedIndexer _scope(ensemble[i].model->indexer.get());
        if (ensemble[i].predicate(sample)) {
          const std::string prediction = ensemble[i].predict(sample);
          if (!prediction.empty()) tmp_probabilities[omp_get_thread_num()][prediction] += 1.0;
        }
      } catch (...) {
        // Exceptions may not escape an OMP region (std::terminate); capture
        // the first one and rethrow after the barrier.
#pragma omp critical
        if (!worker_error) worker_error = std::current_exception();
      }
    }
    if (worker_error) std::rethrow_exception(worker_error);

    double votes = 0;
    for (auto i = 0u; i < NUM_THREADS; i++)
      for (const auto& pair : tmp_probabilities[i]) {
        probabilities[pair.first] += pair.second;
        votes += pair.second;
      }

    normalize(probabilities, votes);
    return std::make_unique<InternalScore>(argmax(probabilities), probabilities);
  }

#endif

  inline static std::unique_ptr<InternalScore> weighted_majority_vote(const Sample& sample,
                                                                      const std::vector<Segment>& ensemble) {
    std::unordered_map<std::string, double> probabilities;  // zero initialized

    double total_weight = 0;
    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) {
        const std::string prediction = segment.predict(sample);
        if (prediction.empty()) continue;
        probabilities[prediction] += segment.weight;
        total_weight += segment.weight;
      }

    normalize(probabilities, total_weight);
    return std::make_unique<InternalScore>(argmax(probabilities), probabilities);
  }

  inline static std::unique_ptr<InternalScore> classification_average(const Sample& sample,
                                                                      const std::vector<Segment>& ensemble) {
    std::unordered_map<std::string, double> probabilities;

    double count = 0;
    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) {
        std::unique_ptr<InternalScore> tmp_score(segment.score(sample));
        for (const auto& probability : tmp_score->probabilities) probabilities[probability.first] += probability.second;
        count++;
      }

    normalize(probabilities, count);
    return std::make_unique<InternalScore>(argmax(probabilities), probabilities);
  }

#ifndef MULTITHREADING
  static std::unique_ptr<InternalScore> regression_average(const Sample& sample, const std::vector<Segment>& ensemble) {
    double score = 0;
    double count = 0;

    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) {
        count++;
        score += to_double(segment.predict(sample));
      }

    if (count == 0) return std::make_unique<InternalScore>();
    score /= count;

    return std::make_unique<InternalScore>(score);
  }
#else
  inline static std::unique_ptr<InternalScore> regression_average(const Sample& sample,
                                                                  const std::vector<Segment>& ensemble) {
    double score = 0;
    double scores[NUM_THREADS];
    double count = 0;

    for (auto i = 0u; i < NUM_THREADS; i++) scores[i] = 0;

    std::exception_ptr worker_error = nullptr;
#pragma omp parallel for if (ensemble.size() > 25) default(shared) num_threads(NUM_THREADS)
    for (auto i = 0u; i < ensemble.size(); i++) {
      try {
        Value::ScopedIndexer _scope(ensemble[i].model->indexer.get());
        if (ensemble[i].predicate(sample)) {
#pragma omp atomic
          count++;
          scores[omp_get_thread_num()] += to_double(ensemble[i].predict(sample));
        }
      } catch (...) {
#pragma omp critical
        if (!worker_error) worker_error = std::current_exception();
      }
    }
    if (worker_error) std::rethrow_exception(worker_error);

    for (auto i = 0u; i < NUM_THREADS; i++) score += scores[i];

    if (count == 0) return std::make_unique<InternalScore>();
    score /= count;

    return std::make_unique<InternalScore>(score);
  }
#endif

  inline static std::unique_ptr<InternalScore> classification_weighted_average(const Sample& sample,
                                                                               const std::vector<Segment>& ensemble) {
    std::unordered_map<std::string, double> probabilities;

    double total_weight = 0;
    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) {
        std::unique_ptr<InternalScore> tmp_score(segment.score(sample));
        for (const auto& probability : tmp_score->probabilities)
          probabilities[probability.first] += probability.second * segment.weight;
        total_weight += segment.weight;
      }

    normalize(probabilities, total_weight);
    return std::make_unique<InternalScore>(argmax(probabilities), probabilities);
  }

  inline static std::unique_ptr<InternalScore> regression_weighted_average(const Sample& sample,
                                                                           const std::vector<Segment>& ensemble) {
    double score = 0;
    double total_weight = 0;

    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) {
        score += to_double(segment.predict(sample)) * segment.weight;
        total_weight += segment.weight;
      }

    if (total_weight == 0) return std::make_unique<InternalScore>();
    return std::make_unique<InternalScore>(score / total_weight);
  }

  inline static std::unique_ptr<InternalScore> median(const Sample& sample, const std::vector<Segment>& ensemble) {
    std::vector<double> values;
    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) values.push_back(to_double(segment.predict(sample)));

    if (values.empty()) return std::make_unique<InternalScore>();

    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double med = values[mid];
    if (values.size() % 2 == 0) {
      double lo = *std::max_element(values.begin(), values.begin() + mid);
      med = (lo + med) / 2.0;
    }
    return std::make_unique<InternalScore>(med);
  }

  inline static std::unique_ptr<InternalScore> max(const Sample& sample, const std::vector<Segment>& ensemble) {
    std::vector<double> values;
    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) values.push_back(to_double(segment.predict(sample)));

    if (values.empty()) return std::make_unique<InternalScore>();
    return std::make_unique<InternalScore>(*std::max_element(values.begin(), values.end()));
  }

#ifndef MULTITHREADING

  inline static std::unique_ptr<InternalScore> sum(const Sample& sample, const std::vector<Segment>& ensemble) {
    double score = 0;

    for (const auto& segment : ensemble)
      if (segment.predicate(sample)) score += std::unique_ptr<InternalScore>(segment.score(sample))->double_score;

    return std::make_unique<InternalScore>(score);
  }

#else
  inline static std::unique_ptr<InternalScore> sum(const Sample& sample, const std::vector<Segment>& ensemble) {
    double score = 0;
    double scores[NUM_THREADS];

    for (auto i = 0u; i < NUM_THREADS; i++) scores[i] = 0;

    std::exception_ptr worker_error = nullptr;
#pragma omp parallel for if (ensemble.size() > 25) default(shared) num_threads(NUM_THREADS)
    for (auto i = 0u; i < ensemble.size(); i++) {
      try {
        Value::ScopedIndexer _scope(ensemble[i].model->indexer.get());
        if (ensemble[i].predicate(sample))
          scores[omp_get_thread_num()] += std::unique_ptr<InternalScore>(ensemble[i].score(sample))->double_score;
      } catch (...) {
#pragma omp critical
        if (!worker_error) worker_error = std::current_exception();
      }
    }
    if (worker_error) std::rethrow_exception(worker_error);

    for (auto i = 0u; i < NUM_THREADS; i++) score += scores[i];

    return std::make_unique<InternalScore>(score);
  }
#endif

  inline static std::unique_ptr<InternalScore> select_first(const Sample& sample, const std::vector<Segment>& ensemble) {
    for (const auto& segment : ensemble) {
      if (segment.predicate(sample)) {
        auto score = segment.score(sample);
        if (!score->empty) return score;
      }
    }
    return std::make_unique<InternalScore>();
  }

  // selectAll: collect every active segment's prediction and format as PMML
  // Collection string "[pred1, pred2, ...]".
  inline static std::unique_ptr<InternalScore> select_all(const Sample& sample, const std::vector<Segment>& ensemble) {
    std::string result = "[";
    bool first = true;
    for (const auto& segment : ensemble) {
      if (segment.predicate(sample)) {
        if (!first) result += ", ";
        result += segment.predict(sample);
        first = false;
      }
    }
    result += "]";
    return std::make_unique<InternalScore>(result);
  }

  // modelChain: segments whose predicate is false are skipped entirely; the
  // last *matching* segment supplies the chain result. Since predicates are
  // evaluated against the evolving sample, each matching segment's execution
  // is deferred until a later match proves it is not the last one.
  inline static std::unique_ptr<InternalScore> model_chain(const Sample& sample, const std::vector<Segment>& ensemble) {
    Sample tmp_sample = sample;
    bool first = true;
    const Segment* pending = nullptr;  // last matching segment, not yet executed

    for (const auto& segment : ensemble) {
      if (!segment.predicate(tmp_sample)) continue;
      if (pending) {
        if (first) {
          pending->model->augment_first(tmp_sample);
          first = false;
        } else {
          pending->model->augment(tmp_sample);
        }
      }
      pending = &segment;
    }

    if (!pending) return std::make_unique<InternalScore>();  // no segment matched
    return pending->model->augment_last(tmp_sample);
  }
};

#endif
