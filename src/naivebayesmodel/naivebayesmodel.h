
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_NAIVEBAYESMODEL_H
#define CPMML_NAIVEBAYESMODEL_H

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "bind/load_scope.h"
#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"
#include "regressionmodel/regressionscore.h"

/**
 * @class NaiveBayesModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/NaiveBayes.html">PMML NaiveBayesModel</a>.
 *
 * Scoring: log-space posterior = log(prior) + Σ log(likelihood)
 * Supports discrete inputs (count tables) and continuous inputs
 * (GaussianDistribution). Zero counts are replaced with `threshold`.
 */
class NaiveBayesModel : public InternalModel {
 public:
  // One discrete input value's likelihood per class
  struct DiscreteInput {
    size_t field_index;
    // input_value (as double via Value encoding) → class → count
    std::unordered_map<double, std::unordered_map<std::string, double>> counts;
    // Σ counts over every input value, per class, aligned to `classes`. It is a
    // property of the model, but scoring recomputed it by walking the whole
    // counts map once per class per record.
    std::vector<double> class_totals;
  };

  // One continuous input (Gaussian per class)
  struct GaussianInput {
    size_t field_index;
    // class → (mean, variance)
    std::unordered_map<std::string, std::pair<double, double>> params;
  };

  // --- Members ---

  double threshold;
  std::vector<std::string> classes;
  std::unordered_map<std::string, double> log_priors;  // log(count/total)
  std::vector<DiscreteInput> discrete_inputs;
  std::vector<GaussianInput> gaussian_inputs;

  // --- Constructors ---

  NaiveBayesModel() = default;

  NaiveBayesModel(const XmlNode& node, const DataDictionary& data_dictionary,
                  const TransformationDictionary& transformation_dictionary, const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        threshold(node.get_double_attribute("threshold", 1e-6)) {
    parse_output(node);
    parse_inputs(node, data_dictionary, indexer);
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    auto log_posts = compute_log_posteriors(sample);
    std::vector<double> probs = to_probs(log_posts);
    size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
    return std::make_unique<RegressionScore>(classes[best], probs[best], classes, probs);
  }

  inline std::string predict_raw(const Sample& sample) const override {
    auto log_posts = compute_log_posteriors(sample);
    size_t best = 0;
    for (size_t i = 1; i < classes.size(); i++)
      if (log_posts[i] > log_posts[best]) best = i;
    return classes[best];
  }

 private:
  void parse_output(const XmlNode& node) {
    const XmlNode bo = node.get_child("BayesOutput");
    double total = 0.0;
    std::vector<std::pair<std::string, double>> counts;
    for (const auto& tvc : bo.get_child("TargetValueCounts").get_childs("TargetValueCount")) {
      const std::string cls = tvc.required_attribute("value");
      const double cnt = tvc.required_double_attribute("count");
      classes.push_back(cls);
      counts.emplace_back(cls, cnt);
      total += cnt;
    }
    for (const auto& [cls, cnt] : counts) log_priors[cls] = std::log(cnt / total);
  }

  void parse_inputs(const XmlNode& node, const DataDictionary& data_dictionary,
                    const std::shared_ptr<Indexer>& indexer) {
    for (const auto& bi : node.get_child("BayesInputs").get_childs("BayesInput")) {
      const std::string field = bi.required_attribute("fieldName");
      const size_t idx = cpmml::bind::scope_resolve(*indexer, bi, field).index;  // scope.resolve (P2.1)

      // Discrete: PairCounts → PairValueCounts → TargetValueCounts
      if (bi.exists_child("PairCounts")) {
        DiscreteInput di;
        di.field_index = idx;
        for (const auto& pvc : bi.get_childs("PairCounts")) {
          const Value input_val(pvc.required_attribute("value"), data_dictionary.at(field).datatype);
          for (const auto& tvc : pvc.get_child("TargetValueCounts").get_childs("TargetValueCount"))
            di.counts[input_val.as_double()][tvc.required_attribute("value")] = tvc.required_double_attribute("count");
        }
        di.class_totals.assign(classes.size(), 0.0);
        for (const auto& [v, cls_map] : di.counts)
          for (size_t i = 0; i < classes.size(); i++) {
            const auto jt = cls_map.find(classes[i]);
            if (jt != cls_map.end()) di.class_totals[i] += jt->second;
          }

        discrete_inputs.push_back(std::move(di));
      }

      // Continuous: TargetValueStats → TargetValueStat → GaussianDistribution
      if (bi.exists_child("TargetValueStats")) {
        GaussianInput gi;
        gi.field_index = idx;
        for (const auto& tvs : bi.get_child("TargetValueStats").get_childs("TargetValueStat")) {
          const std::string cls = tvs.required_attribute("value");
          if (tvs.exists_child("GaussianDistribution")) {
            const XmlNode gd = tvs.get_child("GaussianDistribution");
            gi.params[cls] = {gd.required_double_attribute("mean"), gd.required_double_attribute("variance")};
          }
        }
        gaussian_inputs.push_back(std::move(gi));
      }
    }
  }

  std::vector<double> compute_log_posteriors(const Sample& sample) const {
    std::vector<double> log_posts;
    log_posts.reserve(classes.size());
    for (const auto& cls : classes) log_posts.push_back(log_priors.at(cls));

    // Discrete likelihoods. PMML defines threshold as the *default
    // probability* substituted when the count for (value, class) is zero —
    // not a pseudo-count added to the totals.
    for (const auto& di : discrete_inputs) {
      const Value& xv = sample[di.field_index].value;
      if (xv.is_missing()) continue;  // missing input: skip the feature entirely
      const double x = xv.as_double();
      const auto it = di.counts.find(x);
      for (size_t i = 0; i < classes.size(); i++) {
        double cnt = 0.0;
        const double class_total = di.class_totals[i];
        if (it != di.counts.end()) {
          const auto jt = it->second.find(classes[i]);
          if (jt != it->second.end()) cnt = jt->second;
        }
        const double p = (cnt > 0 && class_total > 0) ? cnt / class_total : threshold;
        log_posts[i] += std::log(p);
      }
    }

    // Gaussian likelihoods
    for (const auto& gi : gaussian_inputs) {
      if (sample[gi.field_index].value.is_missing()) continue;
      const double x = sample[gi.field_index].value.as_double();
      for (size_t i = 0; i < classes.size(); i++) {
        const auto it = gi.params.find(classes[i]);
        if (it == gi.params.end()) continue;
        const double mean = it->second.first;
        const double var = it->second.second > 0 ? it->second.second : threshold;
        // Gaussian pdf, clipped at threshold to avoid log(0)
        const double pdf = (1.0 / std::sqrt(2.0 * M_PI * var)) * std::exp(-(x - mean) * (x - mean) / (2.0 * var));
        log_posts[i] += std::log(std::max(pdf, threshold));
      }
    }

    return log_posts;
  }

  // Convert log-posteriors to probabilities via softmax
  static std::vector<double> to_probs(const std::vector<double>& log_posts) {
    const double max_lp = *std::max_element(log_posts.begin(), log_posts.end());
    std::vector<double> exps;
    double sum = 0.0;
    for (double lp : log_posts) {
      exps.push_back(std::exp(lp - max_lp));
      sum += exps.back();
    }
    for (auto& e : exps) e /= sum;
    return exps;
  }
};

#endif
