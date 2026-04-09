
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_GENERALREGRESSIONMODEL_H
#define CPMML_GENERALREGRESSIONMODEL_H

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/miningfunction.h"
#include "core/optype.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"
#include "math/misc.h"
#include "regressionmodel/regressionscore.h"

/**
 * @class GeneralRegressionModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/GeneralRegression.html">PMML
 * GeneralRegressionModel</a>.
 *
 * Supports modelTypes: regression, generalLinear, multinomialLogistic,
 * ordinalMultinomial, generalizedLinear.
 *
 * Prediction formula:
 *   η_c = Σ_j β_{j,c} · Π_{PPCell for j} f(predictor)
 * where f is x^exponent (continuous) or indicator (categorical).
 * The inverse link function maps η → predicted value.
 */
class GeneralRegressionModel : public InternalModel {
 public:
  // Encodes one PPMatrix cell: which predictor contributes to which parameter
  struct PPCell {
    std::string predictor_name;
    size_t predictor_index;
    std::string value;        // exponent (continuous) or category level (factor)
    bool is_factor;           // false → continuous covariate
  };

  // One β coefficient from ParamMatrix
  struct PCell {
    std::string parameter_name;
    std::string target_category;
    double beta;
  };

  // --- Members ---

  std::string model_type;    // to_lower of modelType attribute
  std::string link_function; // to_lower of linkFunction attribute
  double link_parameter;     // for power/oddspower/negbin links

  // Factors (categorical predictors) — set of field names
  std::unordered_set<std::string> factors;

  // Ordered parameter names (from ParameterList)
  std::vector<std::string> parameters;

  // parameterName → list of PPCells (empty list = intercept term)
  std::unordered_map<std::string, std::vector<PPCell>> pp_matrix;

  // (parameterName, targetCategory) → β; "" targetCategory for regression
  std::unordered_map<std::string, std::unordered_map<std::string, double>> param_matrix;

  // Ordered target categories (for classification)
  std::vector<std::string> classes;
  std::string reference_category;

  // Offset
  bool has_offset_value = false;
  double offset_value = 0.0;
  bool has_offset_variable = false;
  size_t offset_variable_index = 0;

  // --- Constructors ---

  GeneralRegressionModel() = default;

  GeneralRegressionModel(const XmlNode &node, const DataDictionary &data_dictionary,
                          const TransformationDictionary &transformation_dictionary,
                          const std::shared_ptr<Indexer> &indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        model_type(to_lower(node.get_attribute("modelType"))),
        link_function(to_lower(node.get_attribute("linkFunction") == "null"
                                   ? "identity"
                                   : node.get_attribute("linkFunction"))),
        link_parameter(node.exists_attribute("linkParameter")
                           ? to_double(node.get_attribute("linkParameter"))
                           : 1.0),
        reference_category(node.get_attribute("targetReferenceCategory")) {
    // Offset
    if (node.exists_attribute("offsetValue")) {
      has_offset_value = true;
      offset_value = to_double(node.get_attribute("offsetValue"));
    }
    if (node.exists_attribute("offsetVariable")) {
      has_offset_variable = true;
      offset_variable_index = indexer->get_index(node.get_attribute("offsetVariable"));
    }

    parse_factors(node);
    parse_parameters(node);
    parse_pp_matrix(node, data_dictionary, indexer);
    parse_param_matrix(node);
  }

  // --- Scoring ---

  inline std::unique_ptr<InternalScore> score_raw(const Sample &sample) const override {
    if (model_type == "multinomiallogistic") {
      auto [winner, scores] = predict_multinomial(sample);
      return std::make_unique<RegressionScore>(winner, 1.0, classes, scores);
    }
    if (model_type == "ordinalmultinomial") {
      auto [winner, scores] = predict_ordinal(sample);
      return std::make_unique<RegressionScore>(winner, 1.0, classes, scores);
    }
    // regression / generallinear / generalizedlinear
    const double val = predict_continuous(sample);
    return std::make_unique<RegressionScore>(std::to_string(val), val, classes, std::vector<double>{val});
  }

  inline std::string predict_raw(const Sample &sample) const override {
    if (model_type == "multinomiallogistic") {
      auto [winner, scores] = predict_multinomial(sample);
      return winner;
    }
    if (model_type == "ordinalmultinomial") {
      auto [winner, scores] = predict_ordinal(sample);
      return winner;
    }
    return std::to_string(predict_continuous(sample));
  }

 private:
  // --- Parsing ---

  void parse_factors(const XmlNode &node) {
    if (!node.exists_child("FactorList")) return;
    for (const auto &fr : node.get_child("FactorList").get_childs("Predictor"))
      factors.insert(fr.get_attribute("name"));
  }

  void parse_parameters(const XmlNode &node) {
    for (const auto &p : node.get_child("ParameterList").get_childs("Parameter"))
      parameters.push_back(p.get_attribute("name"));
  }

  void parse_pp_matrix(const XmlNode &node, const DataDictionary &data_dictionary,
                        const std::shared_ptr<Indexer> &indexer) {
    for (const auto &param_name : parameters) pp_matrix[param_name] = {};

    for (const auto &cell : node.get_child("PPMatrix").get_childs("PPCell")) {
      const std::string pred  = cell.get_attribute("predictorName");
      const std::string param = cell.get_attribute("parameterName");
      const std::string val   = cell.get_attribute("value");

      bool is_factor = factors.count(pred) > 0;
      // Fallback: check DataDictionary optype
      if (!is_factor && data_dictionary.datafields.count(pred) > 0)
        is_factor = (data_dictionary.datafields.at(pred).optype.value != OpType::OpTypeValue::CONTINUOUS);

      PPCell pc;
      pc.predictor_name  = pred;
      pc.predictor_index = indexer->get_index(pred);
      pc.value           = val;
      pc.is_factor       = is_factor;

      pp_matrix[param].push_back(std::move(pc));
    }
  }

  void parse_param_matrix(const XmlNode &node) {
    for (const auto &cell : node.get_child("ParamMatrix").get_childs("PCell")) {
      const std::string param    = cell.get_attribute("parameterName");
      const std::string cat      = cell.get_attribute("targetCategory");
      const double      beta_val = to_double(cell.get_attribute("beta"));
      param_matrix[param][cat]   = beta_val;

      // Collect ordered classes
      if (!cat.empty() && cat != "null" && cat != reference_category) {
        if (std::find(classes.begin(), classes.end(), cat) == classes.end())
          classes.push_back(cat);
      }
    }

    // For regression, use target field name as the single class
    if (classes.empty())
      classes.push_back(mining_schema.target.name);

    // Add reference category at the end (for probability output completeness)
    if (!reference_category.empty() && reference_category != "null") {
      if (std::find(classes.begin(), classes.end(), reference_category) == classes.end())
        classes.push_back(reference_category);
    }
  }

  // --- Design value computation ---

  // Compute the design value for one parameter given the sample
  double design_value(const std::string &param_name, const Sample &sample) const {
    const auto &cells = pp_matrix.at(param_name);
    if (cells.empty()) return 1.0;  // intercept

    double result = 1.0;
    for (const auto &cell : cells) {
      const double x = sample[cell.predictor_index].value.value;
      if (cell.is_factor) {
        // Categorical indicator: 1.0 if sample value matches the level
        // The sample value is stored as a double index in Value::string_converter
        const double level_val = Value(cell.value, DataType(DataType::DataTypeValue::STRING)).value;
        if (x != level_val) return 0.0;  // short-circuit
      } else {
        // Continuous covariate: x^exponent
        const double exp_val = to_double(cell.value);
        result *= std::pow(x, exp_val);
      }
    }
    return result;
  }

  // Compute offset for this sample
  double get_offset(const Sample &sample) const {
    double off = 0.0;
    if (has_offset_value)    off += offset_value;
    if (has_offset_variable) off += sample[offset_variable_index].value.value;
    return off;
  }

  // Compute linear predictor η for a given targetCategory (or "" for regression)
  double linear_predictor(const std::string &target_cat, const Sample &sample) const {
    double eta = get_offset(sample);
    for (const auto &param_name : parameters) {
      const double dv = design_value(param_name, sample);
      if (dv == 0.0) continue;
      const auto &betas = param_matrix.at(param_name);
      const auto it = betas.find(target_cat);
      if (it != betas.end()) eta += it->second * dv;
    }
    return eta;
  }

  // --- Inverse link functions ---

  double apply_inverse_link(double eta) const {
    if (link_function == "identity")    return eta;
    if (link_function == "log")         return std::exp(eta);
    if (link_function == "logit")       return 1.0 / (1.0 + std::exp(-eta));
    if (link_function == "probit")      return probit(eta);
    if (link_function == "cloglog")     return 1.0 - std::exp(-std::exp(eta));
    if (link_function == "loglog")      return std::exp(-std::exp(-eta));
    if (link_function == "cauchit")     return 0.5 + std::atan(eta) / M_PI;
    if (link_function == "power")       return std::pow(eta, 1.0 / link_parameter);
    if (link_function == "oddspower") {
      if (link_parameter == 0.0) return 1.0 / (1.0 + std::exp(-eta));
      return 1.0 / (1.0 + std::pow(1.0 + link_parameter * eta, -1.0 / link_parameter));
    }
    if (link_function == "negbin") {
      return 1.0 / (link_parameter * (std::exp(-eta) - 1.0));
    }
    return eta;
  }

  // --- Model-type prediction methods ---

  double predict_continuous(const Sample &sample) const {
    // For regression / generalLinear / generalizedLinear:
    // single linear predictor → inverse link
    const std::string cat = (param_matrix.begin()->second.begin()->first);
    // Use empty targetCategory for single-output models
    double eta = linear_predictor("", sample);
    if (eta == get_offset(sample)) {  // fallback: try first available category
      for (const auto &p : parameters) {
        for (const auto &[tc, b] : param_matrix.at(p)) {
          eta = linear_predictor(tc, sample);
          return apply_inverse_link(eta);
        }
      }
    }
    return apply_inverse_link(eta);
  }

  std::pair<std::string, std::vector<double>> predict_multinomial(const Sample &sample) const {
    // Compute η for each non-reference class; reference gets η=0
    std::vector<double> etas;
    etas.reserve(classes.size());

    for (const auto &cls : classes) {
      if (cls == reference_category) { etas.push_back(0.0); continue; }
      etas.push_back(linear_predictor(cls, sample));
    }

    // Softmax
    const double max_eta = *std::max_element(etas.begin(), etas.end());
    std::vector<double> exps;
    double sum_exp = 0.0;
    for (double e : etas) { exps.push_back(std::exp(e - max_eta)); sum_exp += exps.back(); }
    std::vector<double> probs;
    probs.reserve(classes.size());
    for (double e : exps) probs.push_back(e / sum_exp);

    size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
    return {classes[best], probs};
  }

  std::pair<std::string, std::vector<double>> predict_ordinal(const Sample &sample) const {
    // Each class c has its own intercept (cumulative threshold)
    // P(Y <= c) = F(η_c) where F is the cumulative link
    // classes are ordered; last class probability fills to 1
    std::vector<double> cum_probs;
    for (size_t i = 0; i < classes.size() - 1; i++)
      cum_probs.push_back(apply_inverse_link(linear_predictor(classes[i], sample)));
    cum_probs.push_back(1.0);

    std::vector<double> probs;
    probs.push_back(cum_probs[0]);
    for (size_t i = 1; i < classes.size(); i++)
      probs.push_back(cum_probs[i] - cum_probs[i - 1]);

    size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
    return {classes[best], probs};
  }
};

#endif
