
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_EVALUATOR_H
#define CPMML_EVALUATOR_H

#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "datadictionary.h"
#include "header.h"
#include "indexer.h"
#include "internal_score.h"
#include "transformationdictionary.h"

/**
 * @class InternalEvaluator
 *
 * Class encapsulating the high level elements of the <a
 * href="http://dmg.org/pmml/v4-4/GeneralStructure.html">PMML document</a>.:
 *
 * It contains:
 *  - Header
 *  - DataDictionary
 *  - TransformationDictionary
 *  - MODEL-ELEMENT
 */
class InternalEvaluator {
 public:
  std::shared_ptr<Indexer> indexer;
  std::string name;
  std::string version;
  Header header;
  DataDictionary data_dictionary;
  bool hasPreprocessing = false;
  TransformationDictionary transformation_dictionary;

  InternalEvaluator() = default;

  // Helper: creates an Indexer and activates it as the thread-local string context.
  static std::shared_ptr<Indexer> make_and_activate_indexer() {
    auto idx = std::make_shared<Indexer>();
    Value::active_indexer = idx.get();
    return idx;
  }

  InternalEvaluator(const XmlNode& node)
      : indexer(make_and_activate_indexer()),
        name(node.name()),
        version(node.get_attribute("version")),
        header(node.get_child("Header")),
        data_dictionary(node.get_child("DataDictionary"), indexer),
        hasPreprocessing(node.exists_child("TransformationDictionary")),
        transformation_dictionary(hasPreprocessing
                                      ? TransformationDictionary(node.get_child("TransformationDictionary"), indexer)
                                      : TransformationDictionary()) {};

  virtual inline bool validate(const std::unordered_map<std::string, std::string>& sample) { return false; }

  using FieldValue = std::variant<std::string, std::vector<std::string>>;

  virtual std::unique_ptr<InternalScore> score(const std::unordered_map<std::string, std::string>& sample) const = 0;

  virtual std::unique_ptr<InternalScore> score(const std::unordered_map<std::string, FieldValue>& sample) const {
    // Default: extract string values and delegate to the string-only overload.
    // AssociationEvaluator overrides this to pass collections through.
    std::unordered_map<std::string, std::string> flat;
    for (const auto& [k, v] : sample) {
      if (std::holds_alternative<std::string>(v))
        flat[k] = std::get<std::string>(v);
      else
        flat[k] = "";  // non-association models ignore collection fields
    }
    return score(flat);
  }

  // Simple score, due to the type of value returned is 2/300 ns faster
  virtual std::string predict(const std::unordered_map<std::string, std::string>& sample) const = 0;

  virtual inline std::string get_target_name() const { return ""; }

  virtual inline std::string output_name() const { return get_target_name(); }

  virtual std::vector<double> forecast(int /*horizon*/) const {
    throw cpmml::ParsingException("forecast() is only available for TimeSeriesModel");
  }

  virtual std::vector<double> forecast(
      int /*horizon*/, const std::unordered_map<std::string, std::vector<double>>& /*regressors*/) const {
    throw cpmml::ParsingException("forecast() is only available for TimeSeriesModel");
  }

  virtual std::vector<std::pair<double, double>> forecast_with_variance(int /*horizon*/) const {
    throw cpmml::ParsingException("forecast_with_variance() is only available for TimeSeriesModel");
  }

  virtual std::vector<std::pair<double, double>> forecast_with_variance(
      int /*horizon*/, const std::unordered_map<std::string, std::vector<double>>& /*regressors*/) const {
    throw cpmml::ParsingException("forecast_with_variance() is only available for TimeSeriesModel");
  }

  InternalEvaluator(const InternalEvaluator&) = default;

  InternalEvaluator(InternalEvaluator&&) = default;

  InternalEvaluator& operator=(const InternalEvaluator&) = default;

  InternalEvaluator& operator=(InternalEvaluator&&) = default;

  virtual ~InternalEvaluator() = default;
};

#endif
