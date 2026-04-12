
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
  using FieldValue = std::variant<std::string, int, std::vector<std::string>, std::vector<double>>;
  using Input = std::unordered_map<std::string, FieldValue>;

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

  // Single evaluation entry point — all subclasses must implement.
  virtual std::unique_ptr<InternalScore> evaluate(const Input& arguments) const = 0;

  virtual inline bool validate(const Input& arguments) const { return false; }

  virtual inline std::string get_target_name() const { return ""; }

  virtual inline std::string output_name() const { return get_target_name(); }

  virtual inline std::string mining_function_name() const { return ""; }

  InternalEvaluator(const InternalEvaluator&) = default;

  InternalEvaluator(InternalEvaluator&&) = default;

  InternalEvaluator& operator=(const InternalEvaluator&) = default;

  InternalEvaluator& operator=(InternalEvaluator&&) = default;

  virtual ~InternalEvaluator() = default;

 protected:
  // Helper: flatten Input → unordered_map<string, string> for standard models.
  static std::unordered_map<std::string, std::string> flatten_input(const Input& args) {
    std::unordered_map<std::string, std::string> flat;
    for (const auto& [k, v] : args) {
      if (std::holds_alternative<std::string>(v))
        flat[k] = std::get<std::string>(v);
      else if (std::holds_alternative<int>(v))
        flat[k] = std::to_string(std::get<int>(v));
    }
    return flat;
  }
};

#endif
