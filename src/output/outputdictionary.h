
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_OUTPUT_H
#define CPMML_OUTPUT_H

#include <unordered_set>

#include "core/derivedfield.h"
#include "core/xmlnode.h"
#include "outputfield.h"

/**
 * @class OutputDictionary
 * Class representing <a href="http://dmg.org/pmml/v4-4/Output.html">PMML
 * Output</a>.
 *
 * It is a collection of OutputFields. See OutputField.
 *
 * Also in this case a DAG is built to keep track of dependencies between
 * OutputFields. However this DAG is unrelated to the one built by DagBuilder
 * since DerivedFields and OuputFields have different scope: the former deals
 * with preprocessing of fields while the second deals with postprocessing of
 * fields.
 */
class OutputDictionary {
 public:
  bool empty;
  std::vector<OutputField> raw_outputfields;  // XML-document order
  std::vector<OutputField> dag;               // topological order for eval

  OutputDictionary() : empty(true) {}

  OutputDictionary(const XmlNode& node, const std::shared_ptr<Indexer>& indexer, const std::string& model_target)
      : empty(false),
        raw_outputfields(OutputField::to_outputfields(node.get_childs("OutputField"), indexer, model_target)),
        dag(build_dag(raw_outputfields)) {}

  inline bool contains(const std::string& field_name) const {
    for (const auto& of : raw_outputfields)
      if (of.name == field_name) return true;
    return false;
  }

  inline const OutputField& operator[](const std::string& feature_name) const {
    for (const auto& of : raw_outputfields)
      if (of.name == feature_name) return of;
    throw std::out_of_range("OutputField not found: " + feature_name);
  }

  static std::vector<OutputField> build_dag(const std::vector<OutputField>& raw_outputfields) {
    std::vector<OutputField> dag;
    std::unordered_set<std::string> visited;
    for (const auto& output_field : raw_outputfields)
      build_dagR(output_field, dag, raw_outputfields, visited);
    return dag;
  }

  static void build_dagR(const OutputField& output_field, std::vector<OutputField>& dag,
                         const std::vector<OutputField>& raw_outputfields,
                         std::unordered_set<std::string>& visited) {
    if (visited.count(output_field.name)) return;
    visited.insert(output_field.name);

    for (const auto& input : output_field.expression->inputs) {
      for (const auto& candidate : raw_outputfields) {
        if (candidate.name == input) {
          if (candidate.derived) build_dagR(candidate, dag, raw_outputfields, visited);
          if (!visited.count(candidate.name)) {
            visited.insert(candidate.name);
            dag.push_back(candidate);
          }
          break;
        }
      }
    }

    dag.push_back(output_field);
  }

  inline void prepare(Sample& sample) const {
    for (const auto& outputfield : dag) outputfield.prepare(sample);
  }

  inline void add_output(Sample& sample, InternalScore& score) const {
    for (const auto& outputfield : dag) outputfield.add_output(sample, score);
  }
};

#endif
