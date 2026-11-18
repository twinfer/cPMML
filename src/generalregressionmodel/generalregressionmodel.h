
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

#include "bind/load_scope.h"
#include "core/datadictionary.h"
#include "core/enum_parser.h"
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
    std::string value;  // exponent (continuous) or category level (factor)
    bool is_factor;     // false → continuous covariate
    Value level;        // a factor's level, read once at load with the predictor's dataType
    double exponent = 1;  // a covariate's exponent, read once at load (D28: xs:integer)
  };

  // One β coefficient from ParamMatrix
  struct PCell {
    std::string parameter_name;
    std::string target_category;
    double beta;
  };

  // The inverse link F. One enumeration serves both attributes: cumulativeLink
  // (CUMULATIVE-LINK-FUNCTION) and linkFunction (LINK-FUNCTION) share five
  // members with identical formulas, and each attribute is parsed against its
  // own table so neither accepts the other's extra members.
  enum class Link { IDENTITY, LOG, LOGC, LOGIT, PROBIT, CLOGLOG, LOGLOG, CAUCHIT, POWER, ODDSPOWER, NEGBIN };

  enum class ModelType { REGRESSION, GENERAL_LINEAR, MULTINOMIAL_LOGISTIC, ORDINAL_MULTINOMIAL, GENERALIZED_LINEAR };

  // modelType is required. CoxRegression used to fall through to the
  // continuous branch and score as a linear model.
  static constexpr EnumEntry<ModelType> MODEL_TYPE_TABLE[] = {
      {"regression", ModelType::REGRESSION},
      {"generalLinear", ModelType::GENERAL_LINEAR},
      {"multinomialLogistic", ModelType::MULTINOMIAL_LOGISTIC},
      {"ordinalMultinomial", ModelType::ORDINAL_MULTINOMIAL},
      {"generalizedLinear", ModelType::GENERALIZED_LINEAR},
      {"CoxRegression", ModelType::REGRESSION, nullptr, EnumSupport::UNSUPPORTED},
  };

  // --- Members ---

  ModelType model_type;
  Link link;               // cumulativeLink for ordinalMultinomial, else linkFunction
  double link_parameter;   // for power/oddspower/negbin links

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

  // Category key used by continuous prediction: "" when PCells omit
  // targetCategory (the standard encoding), else the first category seen.
  std::string continuous_category;

  // Offset
  bool has_offset_value = false;
  double offset_value = 0.0;
  bool has_offset_variable = false;
  size_t offset_variable_index = 0;

  // --- Constructors ---

  GeneralRegressionModel() = default;

  GeneralRegressionModel(const XmlNode& node, const DataDictionary& data_dictionary,
                         const TransformationDictionary& transformation_dictionary,
                         const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        model_type(read_enum(node, "modelType", MODEL_TYPE_TABLE)),
        link(parse_link(node, model_type)),
        link_parameter(node.get_double_attribute("linkParameter", 1.0)),
        // pmml-4-4.xsd: targetReferenceCategory is optional without a default;
        // absent is the empty text (and the category is then recovered from the
        // DataDictionary). A category may be the text "null".
        reference_category(node.get_attribute("targetReferenceCategory", "")) {
    // Offset
    if (node.exists_attribute("offsetValue")) {
      has_offset_value = true;
      offset_value = node.required_double_attribute("offsetValue");
    }
    if (node.exists_attribute("offsetVariable")) {
      has_offset_variable = true;
      offset_variable_index =
          cpmml::bind::scope_resolve(*indexer, node, node.required_attribute("offsetVariable")).index;
    }

    parse_factors(node);
    parse_parameters(node);
    parse_pp_matrix(node, data_dictionary, indexer);
    parse_param_matrix(node, data_dictionary);
  }

  // --- Scoring ---

  // A missing predictor yields no prediction, as in JPMML and as a TreeModel's
  // nullPrediction does. Throwing made one incomplete record an error for the
  // caller; scoring it as 0 (the pre-bf17bd7 behaviour) gave a confident wrong
  // answer. A model wanting a substitute says so with missingValueReplacement,
  // which MiningSchema applies before this point.
  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    if (has_missing_input(sample)) return std::make_unique<InternalScore>();
    if (model_type == ModelType::MULTINOMIAL_LOGISTIC) {
      auto [winner, scores] = predict_multinomial(sample);
      return std::make_unique<RegressionScore>(winner, 1.0, classes, scores);
    }
    if (model_type == ModelType::ORDINAL_MULTINOMIAL) {
      auto [winner, scores] = predict_ordinal(sample);
      return std::make_unique<RegressionScore>(winner, 1.0, classes, scores);
    }
    // regression / generallinear / generalizedlinear
    const double val = predict_continuous(sample);
    return std::make_unique<InternalScore>(val);
  }

  inline std::string predict_raw(const Sample& sample) const override {
    if (has_missing_input(sample)) return "";
    if (model_type == ModelType::MULTINOMIAL_LOGISTIC) {
      auto [winner, scores] = predict_multinomial(sample);
      return winner;
    }
    if (model_type == ModelType::ORDINAL_MULTINOMIAL) {
      auto [winner, scores] = predict_ordinal(sample);
      return winner;
    }
    return java::double_text(predict_continuous(sample));
  }

 private:
  bool has_missing_input(const Sample& sample) const {
    if (has_offset_variable && sample[offset_variable_index].value.missing) return true;
    for (const auto& [param, cells] : pp_matrix)
      for (const auto& cell : cells)
        if (sample[cell.predictor_index].value.missing) return true;
    return false;
  }

  // --- Parsing ---

  static constexpr EnumEntry<Link> CUMULATIVE_LINK_TABLE[] = {
      {"logit", Link::LOGIT},     {"probit", Link::PROBIT}, {"cloglog", Link::CLOGLOG},
      {"loglog", Link::LOGLOG},   {"cauchit", Link::CAUCHIT},
  };

  // LINK-FUNCTION's ten members. cauchit used to be accepted here too; it is
  // a CUMULATIVE-LINK-FUNCTION member only, and JPMML 1.7.7 refuses it as a
  // linkFunction ("Required attribute GeneralRegressionModel@linkFunction is
  // not defined").
  static constexpr EnumEntry<Link> LINK_FUNCTION_TABLE[] = {
      {"identity", Link::IDENTITY}, {"log", Link::LOG},         {"logc", Link::LOGC},
      {"logit", Link::LOGIT},       {"probit", Link::PROBIT},   {"cloglog", Link::CLOGLOG},
      {"loglog", Link::LOGLOG},     {"power", Link::POWER},     {"oddspower", Link::ODDSPOWER},
      {"negbin", Link::NEGBIN},
  };

  /**
   * The spec gives each model type its own link attribute: cumulativeLink "when
   * ordinalMultinomial model type is specified", linkFunction "when
   * generalizedLinear model type is specified". This read linkFunction for
   * every type, so an ordinal model -- which carries cumulativeLink and no
   * linkFunction -- fell through to identity and returned raw differences of
   * linear predictors as probabilities (-4, 2, 3).
   *
   * ordinalMultinomial: cumulativeLink is required. The schema marks it
   * optional with no default, and an ordinal model has no meaning without F;
   * JPMML 1.7.7 rejects such a document ("Required attribute
   * GeneralRegressionModel@cumulativeLink is not defined") whether or not
   * linkFunction is present, and when both are present scores with
   * cumulativeLink alone. linkFunction is ignored here for the same reason.
   *
   * Other types: linkFunction as before, absent meaning identity (which is the
   * only correct reading for regression/generalLinear, and what cPMML has
   * always done for generalizedLinear). multinomialLogistic reads neither.
   *
   * In both cases a value that is present and unrecognised is a load error;
   * it used to become identity.
   */
  static Link parse_link(const XmlNode& node, ModelType model_type) {
    if (model_type == ModelType::ORDINAL_MULTINOMIAL) {
      if (!node.exists_attribute("cumulativeLink"))
        throw cpmml::ParsingException(
            "GeneralRegressionModel: ordinalMultinomial requires cumulativeLink (linkFunction does not apply to it)");
      return read_enum(node, "cumulativeLink", CUMULATIVE_LINK_TABLE);
    }
    // linkFunction belongs to generalizedLinear alone, the same way cumulativeLink
    // belongs to ordinalMultinomial. regression and generalLinear are identity
    // by definition, and producers do write linkFunction="identity" on them, so
    // it is ignored there -- as JPMML does -- though it must still be a
    // LINK-FUNCTION. Applying a present one there scored linkFunction="log" on
    // a regression as exp(eta).
    const Link link = read_enum(node, "linkFunction", LINK_FUNCTION_TABLE, Link::IDENTITY);
    if (model_type != ModelType::GENERALIZED_LINEAR) return Link::IDENTITY;  // incl. multinomialLogistic: softmax
    return link;
  }

  void parse_factors(const XmlNode& node) {
    if (!node.exists_child("FactorList")) return;
    for (const auto& fr : node.get_child("FactorList").get_childs("Predictor"))
      factors.insert(fr.required_attribute("name"));
  }

  void parse_parameters(const XmlNode& node) {
    for (const auto& p : node.get_child("ParameterList").get_childs("Parameter"))
      parameters.push_back(p.required_attribute("name"));
  }

  void parse_pp_matrix(const XmlNode& node, const DataDictionary& data_dictionary,
                       const std::shared_ptr<Indexer>& indexer) {
    for (const auto& param_name : parameters) pp_matrix[param_name] = {};

    for (const auto& cell : node.get_child("PPMatrix").get_childs("PPCell")) {
      const std::string pred = cell.required_attribute("predictorName");
      const std::string param = cell.required_attribute("parameterName");
      const std::string val = cell.required_attribute("value");

      bool is_factor = factors.count(pred) > 0;
      // Fallback: check DataDictionary optype
      if (!is_factor && data_dictionary.datafields.count(pred) > 0)
        is_factor = (data_dictionary.datafields.at(pred).optype.value != OpType::OpTypeValue::CONTINUOUS);

      PPCell pc;
      pc.predictor_name = pred;
      // scope.resolve (P2.1): the predictor's slot and declared type.
      const auto field = cpmml::bind::scope_resolve(*indexer, cell, pred);
      pc.predictor_index = field.index;
      pc.value = val;
      pc.is_factor = is_factor;
      // Read at load, while the model's Indexer still interns: a string level
      // gets its own code, which no string the model never names can share.
      // A factor's PPCell@value (xs:string) is a level of the factor, read by
      // the K1 rule (§3.4) in its declared type, which Value(text, datatype)
      // is in the engine's storage (bind/load_scope.h): an integer factor's
      // "1.0" is the double 1 (rule 2, D22) and matches the input 1, and
      // "1.5" matches no integer. A covariate's is its exponent
      // (GeneralRegression.html: "the cell value is set to the exponent"),
      // which D28 (§8.5, decided 2026-11-18) reads as INT-NUMBER, xs:integer,
      // as NumericPredictor@exponent: "2.0", "2e0", "2.5" are load errors,
      // "+2", "02", " 2 " are 2. It is read here once; it used to be parsed
      // as a double on every record that reached the cell, and malformed
      // text failed each of them.
      if (is_factor) {
        pc.level = Value(val, field.datatype);
      } else {
        long exponent = 0;
        if (!lexical::parse_long(val, exponent))
          throw cpmml::ParsingException(std::string(cpmml::bind::code::kAttributeMalformed) + ": PPCell@value=\"" +
                                        val + "\" of covariate " + pred +
                                        " is not an integer (its exponent; D28: INT-NUMBER, xs:integer)");
        pc.exponent = static_cast<double>(exponent);
      }

      pp_matrix[param].push_back(std::move(pc));
    }
  }

  void parse_param_matrix(const XmlNode& node, const DataDictionary& data_dictionary) {
    bool has_shared_cells = false;
    std::string first_category;
    for (const auto& cell : node.get_child("ParamMatrix").get_childs("PCell")) {
      const std::string param = cell.required_attribute("parameterName");
      // A PCell without targetCategory is a shared beta: it applies to every
      // target category (ordinal slope terms, common covariates). Key it "".
      const std::string cat = cell.get_attribute("targetCategory", "");
      const double beta_val = cell.required_double_attribute("beta");
      param_matrix[param][cat] = beta_val;

      if (cat.empty()) {
        has_shared_cells = true;
      } else {
        if (first_category.empty()) first_category = cat;
        // Collect ordered classes
        if (cat != reference_category && std::find(classes.begin(), classes.end(), cat) == classes.end())
          classes.push_back(cat);
      }
    }
    // Continuous prediction reads shared ("") betas when any exist; otherwise
    // some producers attach a targetCategory even to regression PCells — use
    // the first one in document order (deterministic).
    continuous_category = has_shared_cells ? "" : first_category;

    // For regression, use target field name as the single class
    if (classes.empty()) classes.push_back(mining_schema.target.name);

    // Add reference category at the end (for probability output completeness)
    if (!reference_category.empty()) {
      if (std::find(classes.begin(), classes.end(), reference_category) == classes.end())
        classes.push_back(reference_category);
      return;
    }

    recover_reference_category(data_dictionary);
  }

  /**
   * The reference category is the one whose betas are all zero, so a
   * well-formed document writes no PCell for it. When targetReferenceCategory
   * names it that is enough; when the attribute is absent -- and it is optional
   * -- the category exists only in the target DataField's Value list, and this
   * built `classes` from ParamMatrix alone.
   *
   * The result was a softmax over K-1 classes with no baseline at zero: every
   * probability inflated, the missing category unable to win however strong the
   * evidence, and the distribution one entry short of the target's own domain.
   *
   * Recovering it needs the DataDictionary, which is the only place that says
   * what the target's categories are.
   */
  void recover_reference_category(const DataDictionary& data_dictionary) {
    if (model_type != ModelType::MULTINOMIAL_LOGISTIC && model_type != ModelType::ORDINAL_MULTINOMIAL) return;

    const auto target = data_dictionary.datafields.find(mining_schema.target.name);
    if (target == data_dictionary.datafields.cend()) return;

    // Appended, not merged in document order: an explicit
    // targetReferenceCategory goes on the end too, so both paths agree on where
    // the baseline sits in the distribution.
    for (const auto& declared : target->second.valid_values)
      if (std::find(classes.begin(), classes.end(), declared) == classes.end()) {
        classes.push_back(declared);
        if (reference_category.empty()) reference_category = declared;
      }
  }

  // --- Design value computation ---

  // Compute the design value for one parameter given the sample
  double design_value(const std::string& param_name, const Sample& sample) const {
    const auto& cells = pp_matrix.at(param_name);
    if (cells.empty()) return 1.0;  // intercept

    double result = 1.0;
    for (const auto& cell : cells) {
      const Value& xv = sample[cell.predictor_index].value;
      if (xv.missing) throw cpmml::MissingValueException("missing value for GeneralRegression predictor");
      const double x = xv.value;
      if (cell.is_factor) {
        // "set x_i to 1 if the case being scored matches this row, 0 if it
        // does not" (GeneralRegression, scoring). The level used to be read
        // here, as a string whatever the predictor's dataType: an integer
        // factor never matched, and since the Indexer is frozen once the model
        // is loaded, a level first read here got the code of every unseen
        // string, so h="zz" matched level "a".
        if (xv != cell.level) return 0.0;  // short-circuit
      } else {
        // Continuous covariate: x^exponent
        result *= std::pow(x, cell.exponent);
      }
    }
    return result;
  }

  // Compute offset for this sample
  double get_offset(const Sample& sample) const {
    double off = 0.0;
    if (has_offset_value) off += offset_value;
    if (has_offset_variable) {
      const Value& v = sample[offset_variable_index].value;
      if (v.missing) throw cpmml::MissingValueException("missing value for offset variable");
      off += v.value;
    }
    return off;
  }

  // Compute linear predictor η for a given targetCategory (or "" for regression)
  double linear_predictor(const std::string& target_cat, const Sample& sample) const {
    double eta = get_offset(sample);
    for (const auto& param_name : parameters) {
      const auto pit = param_matrix.find(param_name);
      if (pit == param_matrix.end()) continue;  // parameter with no PCell → implicit β = 0
      const double dv = design_value(param_name, sample);
      if (dv == 0.0) continue;
      const auto& betas = pit->second;
      const auto it = betas.find(target_cat);
      if (it != betas.end()) eta += it->second * dv;
      // Shared betas (PCells without targetCategory) apply to every category.
      if (!target_cat.empty()) {
        const auto shared = betas.find("");
        if (shared != betas.end()) eta += shared->second * dv;
      }
    }
    return eta;
  }

  // --- Inverse link functions ---

  // Formulas as the spec states them for both the ordinal and the
  // generalizedLinear sections. cloglog and loglog are mirror images, not
  // inverses: cloglog = 1 - exp(-exp(y)), loglog = exp(-exp(-y)).
  double apply_inverse_link(double eta) const {
    switch (link) {
      case Link::IDENTITY: return eta;
      case Link::LOG: return std::exp(eta);
      case Link::LOGC: return 1.0 - std::exp(eta);
      case Link::LOGIT: return 1.0 / (1.0 + std::exp(-eta));
      case Link::PROBIT: return probit(eta);
      case Link::CLOGLOG: return 1.0 - std::exp(-std::exp(eta));
      case Link::LOGLOG: return std::exp(-std::exp(-eta));
      case Link::CAUCHIT: return 0.5 + std::atan(eta) / M_PI;
      case Link::POWER:
        // y^(1/d) for d != 0, exp(y) for d = 0 (pow(y, inf) before).
        if (link_parameter == 0.0) return std::exp(eta);
        return std::pow(eta, 1.0 / link_parameter);
      case Link::ODDSPOWER:
        if (link_parameter == 0.0) return 1.0 / (1.0 + std::exp(-eta));
        return 1.0 / (1.0 + std::pow(1.0 + link_parameter * eta, -1.0 / link_parameter));
      case Link::NEGBIN: return 1.0 / (link_parameter * (std::exp(-eta) - 1.0));
    }
    return eta;  // unreachable: every Link is handled above
  }

  // --- Model-type prediction methods ---

  double predict_continuous(const Sample& sample) const {
    // For regression / generalLinear / generalizedLinear:
    // single linear predictor → inverse link. continuous_category is "" for
    // the standard no-targetCategory encoding, else the first document-order
    // category (some producers tag regression PCells with a category).
    return apply_inverse_link(linear_predictor(continuous_category, sample));
  }

  std::pair<std::string, std::vector<double>> predict_multinomial(const Sample& sample) const {
    // Compute η for each non-reference class; reference gets η=0
    std::vector<double> etas;
    etas.reserve(classes.size());

    for (const auto& cls : classes) {
      if (cls == reference_category) {
        etas.push_back(0.0);
        continue;
      }
      etas.push_back(linear_predictor(cls, sample));
    }

    // Softmax
    const double max_eta = *std::max_element(etas.begin(), etas.end());
    std::vector<double> exps;
    double sum_exp = 0.0;
    for (double e : etas) {
      exps.push_back(std::exp(e - max_eta));
      sum_exp += exps.back();
    }
    std::vector<double> probs;
    probs.reserve(classes.size());
    for (double e : exps) probs.push_back(e / sum_exp);

    size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
    return {classes[best], probs};
  }

  std::pair<std::string, std::vector<double>> predict_ordinal(const Sample& sample) const {
    // Each class c has its own intercept (cumulative threshold)
    // P(Y <= c) = F(η_c) where F is the cumulative link
    // classes are ordered; last class probability fills to 1
    std::vector<double> cum_probs;
    for (size_t i = 0; i < classes.size() - 1; i++)
      cum_probs.push_back(apply_inverse_link(linear_predictor(classes[i], sample)));
    cum_probs.push_back(1.0);

    std::vector<double> probs;
    probs.push_back(cum_probs[0]);
    for (size_t i = 1; i < classes.size(); i++) probs.push_back(cum_probs[i] - cum_probs[i - 1]);

    size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
    return {classes[best], probs};
  }
};

#endif
