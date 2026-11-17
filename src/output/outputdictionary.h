
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
        dag(build_dag(raw_outputfields)),
        reads_outputs(any_reads_outputs(raw_outputfields)) {}

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

  // Every OutputField for the prediction `score`, in dependency order. A
  // transformedValue may read another OutputField of the model (PMML 4.4.1
  // Output, Example 1: NormContinuous field="RawResult", a predictedValue
  // OutputField; JPMML 1.7.7 evaluates it): each value is put into a copy of
  // the sample as it is computed, so a FieldRef to it reads the post-Target
  // value the field reports. They used to be read from the sample, where
  // nothing had written them, so such a field had no value.
  //
  // `shown`, when given, is the prediction named by its displayValue: what
  // the predictedDisplayValue fields of a segment report (InternalModel::
  // add_outputs), and what a field reading them sees.
  inline void add_output(const Sample& sample, InternalScore& score, const InternalScore* shown = nullptr) const {
    if (!reads_outputs) {
      for (const auto& outputfield : dag)
        outputfield.add_output(sample, shown && outputfield.is_display_value() ? *shown : score, score);
      return;
    }
    Sample local = sample;
    for (const auto& outputfield : dag) {
      const Value value =
          outputfield.add_output(local, shown && outputfield.is_display_value() ? *shown : score, score);
      local.change_value(outputfield.index, value);
    }
  }

 private:
  // Whether an OutputField reads another one: only then does add_output
  // need a copy of the sample to put the values in.
  bool reads_outputs = false;

  static bool any_reads_outputs(const std::vector<OutputField>& fields) {
    for (const auto& field : fields)
      for (const auto& other : fields)
        if (field.expression->inputs.count(other.name)) return true;
    return false;
  }
};

#endif
