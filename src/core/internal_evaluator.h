
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
#include "value.h"

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
namespace cpmml {
class InternalEvaluator {
 public:
  using FieldValue = std::variant<std::string, int, std::vector<std::string>, std::vector<double>>;
  using Input = std::unordered_map<std::string, FieldValue>;

  std::shared_ptr<Indexer> indexer;
  // The document's strings (§2.3, design P5): its StringPool, which replaced
  // the Indexer's string map. ModelBuilder::build makes it and loads the
  // document in a StringScope of it.
  std::shared_ptr<DocumentStrings> strings;
  std::string name;
  std::string version;
  Header header;
  DataDictionary data_dictionary;
  bool hasPreprocessing = false;
  TransformationDictionary transformation_dictionary;
  // The binder that resolved the model's names at load (core/modelbuilder.h,
  // typed-core P2.1), kept with the model and not read after the load: freed
  // after the load, its memory was reused by every record's temporaries and
  // cost the small regression models 2-6% per record. P7 makes its scopes
  // the engine's.
  std::shared_ptr<const void> load_scope;

  InternalEvaluator() = default;

  InternalEvaluator(const XmlNode& node)
      : indexer(std::make_shared<Indexer>()),
        strings(StringScope::loading()),
        name(node.name()),
        version(node.required_attribute("version")),
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

  virtual inline std::vector<std::string> output_names() const { return {}; }

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
}  // namespace cpmml

// See internal_score.h: declared as cpmml::InternalEvaluator in the public
// header, spelled unqualified everywhere inside.
using cpmml::InternalEvaluator;


#endif
