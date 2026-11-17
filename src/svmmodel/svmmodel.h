
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_SVMMODEL_H
#define CPMML_SVMMODEL_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/datadictionary.h"
#include "core/internal_model.h"
#include "core/miningfunction.h"
#include "core/transformationdictionary.h"
#include "core/xmlnode.h"
#include "regressionmodel/regressionscore.h"
#include "svmkernel.h"

/**
 * @class SupportVectorMachineModel
 *
 * Implementation of InternalModel representing a <a
 * href="http://dmg.org/pmml/v4-4/SupportVectorMachine.html">PMML
 * SupportVectorMachineModel</a>.
 *
 * Supports:
 * - All four kernel types (linear, polynomial, RBF, sigmoid)
 * - Both SupportVectors and Coefficients representations
 * - Classification (OneAgainstAll, OneAgainstOne) and regression
 * - Sparse (REAL-SparseArray) and dense (Array type="real") vector storage
 * - VectorFields made of FieldRef and CategoricalPredictor entries
 *
 * Classification follows PMML 4.4. With maxWins="false" (the default) the LOWER
 * value wins: a binary or one-against-one machine votes for targetCategory when
 * its value is below the threshold and for alternateTargetCategory otherwise,
 * and one-against-all predicts the targetCategory of the machine with the
 * smallest value. maxWins="true" turns both comparisons around.
 */
class SupportVectorMachineModel : public InternalModel {
 public:
  // One binary SVM: αᵢ coefficients + support vectors + bias
  struct SvmMachine {
    std::string target_category;
    std::string alternate_category;
    double bias = 0.0;       // absoluteValue from Coefficients element
    double threshold = 0.0;  // binary / OneAgainstOne discrimination boundary

    // SupportVectors representation: pairs of (αᵢ, xᵢ)
    std::vector<double> alphas;
    std::vector<Eigen::VectorXd> support_vectors;

    // Coefficients representation (linear only): precomputed w vector
    bool is_weight_vector = false;
    Eigen::VectorXd weight_vector;

    inline double decision(const Eigen::VectorXd& x, const SvmKernel& kernel) const {
      if (is_weight_vector) return weight_vector.dot(x) + bias;
      double result = bias;
      for (size_t i = 0; i < alphas.size(); i++) result += alphas[i] * kernel.compute(x, support_vectors[i]);
      return result;
    }
  };

  // --- Members ---

  SvmKernel kernel;
  bool use_support_vectors;  // false → Coefficients representation
  bool is_ovo;               // false → OneAgainstAll
  bool max_wins;             // false (spec default) → the lower value wins
  double model_threshold;

  // One entry of VectorFields. A FieldRef contributes the field's value; a
  // CategoricalPredictor contributes 1 when the field equals `category` and 0
  // otherwise (its coefficient is ignored in this model, per the spec).
  struct VectorField {
    size_t index;
    bool is_indicator = false;
    Value category;
  };

  // Ordered entries: position i in the feature vector is built from fields[i]
  std::vector<VectorField> vector_fields;

  std::vector<SvmMachine> machines;
  std::vector<std::string> classes;

  // --- Constructors ---

  // SVM-REPRESENTATION (true: SupportVectors, the default) and
  // SVM-CLASSIFICATION-METHOD (true: OneAgainstOne; default OneAgainstAll).
  // Both used to be lower-cased and anything unrecognised taken as the
  // default.
  static constexpr EnumEntry<bool> SVM_REPRESENTATION_TABLE[] = {
      {"SupportVectors", true},
      {"Coefficients", false},
  };
  static constexpr EnumEntry<bool> CLASSIFICATION_METHOD_TABLE[] = {
      {"OneAgainstAll", false},
      {"OneAgainstOne", true},
  };

  SupportVectorMachineModel() = default;

  SupportVectorMachineModel(const XmlNode& node, const DataDictionary& data_dictionary,
                            const TransformationDictionary& transformation_dictionary,
                            const std::shared_ptr<Indexer>& indexer)
      : InternalModel(node, data_dictionary, transformation_dictionary, indexer),
        kernel(SvmKernel::from_node(node)),
        use_support_vectors(read_enum(node, "svmRepresentation", SVM_REPRESENTATION_TABLE, true)),
        is_ovo(read_enum(node, "classificationMethod", CLASSIFICATION_METHOD_TABLE, false)),
        max_wins(node.get_bool_attribute("maxWins", false)),
        model_threshold(node.get_double_attribute("threshold", 0.0)) {
    // The Coefficients form is the hyperplane w.x + b, which only exists for the
    // linear kernel; scoring it with any other declared kernel would ignore that kernel.
    if (!use_support_vectors && kernel.type != SvmKernelType::LINEAR)
      throw cpmml::ParsingException(
          "SupportVectorMachineModel: svmRepresentation=\"Coefficients\" requires LinearKernelType");
    parse_vector_fields(node, indexer);

    // VectorDictionary is only present in SupportVectors representation
    std::unordered_map<std::string, Eigen::VectorXd> vector_dict;
    if (use_support_vectors && node.exists_child("VectorDictionary"))
      vector_dict = parse_vector_dictionary(node.get_child("VectorDictionary"));

    for (const auto& svm_node : node.get_childs("SupportVectorMachine"))
      machines.push_back(parse_machine(svm_node, vector_dict));
    if (machines.empty()) throw cpmml::ParsingException("SupportVectorMachineModel: no SupportVectorMachine");

    // alternateBinaryTargetCategory is not a PMML 4.4 attribute (JPMML reads it
    // from its own schema); here it only supplies a missing alternateTargetCategory.
    // Must be resolved before collecting the class list below.
    if (node.exists_attribute("alternateBinaryTargetCategory"))
      for (auto& m : machines)
        if (m.alternate_category == "null" || m.alternate_category.empty())
          m.alternate_category = node.get_attribute("alternateBinaryTargetCategory");

    // Binary model without alternateTargetCategory: take the other value the
    // target's DataDictionary declares. Must run before the class list below.
    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION && machines.size() == 1 &&
        machines[0].alternate_category == "null") {
      auto it = data_dictionary.datafields.find(mining_schema.target.name);
      if (it != data_dictionary.datafields.end()) {
        const Predicate& c = it->second.constraints;
        for (const auto& pred : c.is_compound_predicate ? c.predicates : std::vector<Predicate>{c}) {
          if (pred.is_set_predicate && pred.predicatetype.value == PredicateOpType::PredicateOpTypeValue::IS_IN) {
            for (const auto& v : pred.values) {
              std::string lbl = v.svalue;
              if (!lbl.empty() && lbl != machines[0].target_category) {
                machines[0].alternate_category = lbl;
                break;
              }
            }
            break;
          }
        }
      }
    }

    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION) {
      // Collect the distinct classes in document order. alternateTargetCategory
      // must be included: in one-against-one the last class typically appears
      // ONLY as an alternate (machines (A,B),(A,C),(B,C) → C is never a
      // target_category) and would otherwise be unpredictable.
      auto add_class = [this](const std::string& cls) {
        if (cls.empty() || cls == "null") return;
        if (std::find(classes.begin(), classes.end(), cls) == classes.end()) classes.push_back(cls);
      };
      for (const auto& m : machines) {
        add_class(m.target_category);
        add_class(m.alternate_category);
      }
    } else {
      classes.push_back(mining_schema.target.name);
    }

    // Every machine of a classifier names its targetCategory, and a binary or
    // one-against-one machine its alternateTargetCategory too: without them the
    // model would "predict" the string "null".
    const auto absent = [](const std::string& v) { return v.empty() || v == "null"; };
    if (mining_function.value == MiningFunction::MiningFunctionType::CLASSIFICATION) {
      for (const auto& m : machines) {
        if (absent(m.target_category))
          throw cpmml::ParsingException("SupportVectorMachineModel: SupportVectorMachine has no targetCategory");
        if (votes_by_threshold() && absent(m.alternate_category))
          throw cpmml::ParsingException("SupportVectorMachineModel: SupportVectorMachine for targetCategory '" +
                                        m.target_category + "' has no alternateTargetCategory");
      }
    } else if (machines.size() != 1) {
      throw cpmml::ParsingException("SupportVectorMachineModel: regression needs exactly one SupportVectorMachine, got " +
                                    std::to_string(machines.size()));
    }
  }

  // --- Scoring ---

  // A FieldRef in VectorFields whose value is missing makes the whole model
  // yield no prediction: score_raw returns an empty score and predict_raw "".
  // The SupportVectorMachineModel page says nothing about missing inputs;
  // JPMML 1.7.7 throws MissingFieldValueException, and cPMML gives no
  // prediction instead, as RegressionModel does for a missing NumericPredictor
  // ("the result evaluates to a missing value"). A missing CategoricalPredictor
  // field is not missing here: the spec takes CategoricalPredictor from the
  // Regression model, where "if the input value is missing then
  // variable_name(v) yields 0 for any v", so every indicator of that field is
  // 0 (JPMML throws there too). missingValueReplacement still applies first,
  // in MiningSchema.
  inline bool has_missing_input(const Sample& sample) const {
    for (const auto& f : vector_fields)
      if (!f.is_indicator && sample[f.index].value.missing) return true;
    return false;
  }

  inline std::unique_ptr<InternalScore> score_raw(const Sample& sample) const override {
    if (has_missing_input(sample)) return std::make_unique<InternalScore>();
    const Eigen::VectorXd x = sample_to_vector(sample);

    if (mining_function.value != MiningFunction::MiningFunctionType::CLASSIFICATION) {
      const double val = machines[0].decision(x, kernel);
      return std::make_unique<RegressionScore>(java::double_text(val), val, classes, std::vector<double>{val});
    }

    std::vector<double> values;
    values.reserve(machines.size());
    for (const auto& m : machines) values.push_back(m.decision(x, kernel));

    // Per-class figures, aligned with `classes` (RegressionScore indexes the two
    // vectors together): the machine value for one-against-all, the share of
    // votes for binary and one-against-one.
    std::vector<double> per_class(classes.size(), 0.0);
    std::string winner;
    double winner_value = 0.0;
    if (votes_by_threshold()) {
      std::vector<size_t> vote_order;
      const std::vector<int> votes = count_votes(values, vote_order);
      winner = classes[most_voted(votes, vote_order)];
      for (size_t c = 0; c < classes.size(); c++) per_class[c] = double(votes[c]) / double(machines.size());
      winner_value = per_class[class_index(winner)];
    } else {
      const size_t best = best_machine(values);
      winner = machines[best].target_category;
      winner_value = values[best];
      for (size_t i = 0; i < machines.size(); i++) per_class[class_index(machines[i].target_category)] = values[i];
    }
    return std::make_unique<RegressionScore>(winner, winner_value, classes, per_class);
  }

  inline std::string predict_raw(const Sample& sample) const override {
    if (has_missing_input(sample)) return "";
    const Eigen::VectorXd x = sample_to_vector(sample);

    if (mining_function.value != MiningFunction::MiningFunctionType::CLASSIFICATION)
      return java::double_text(machines[0].decision(x, kernel));

    std::vector<double> values;
    values.reserve(machines.size());
    for (const auto& m : machines) values.push_back(m.decision(x, kernel));

    if (votes_by_threshold()) {
      std::vector<size_t> vote_order;
      const std::vector<int> votes = count_votes(values, vote_order);
      return classes[most_voted(votes, vote_order)];
    }
    return machines[best_machine(values)].target_category;
  }

 private:
  // --- Feature vector construction ---

  // VectorFields is a sequence of FieldRef and CategoricalPredictor in any
  // order; the order is the order of the vector entries.
  void parse_vector_fields(const XmlNode& model_node, const std::shared_ptr<Indexer>& indexer) {
    if (!model_node.exists_child("VectorDictionary")) return;
    const XmlNode vf = model_node.get_child("VectorDictionary").get_child("VectorFields");
    for (const auto& child : vf.get_childs()) {
      const std::string kind = child.name();
      if (kind == "FieldRef") {
        vector_fields.push_back({indexer->get_index(child.get_attribute("field")), false, Value()});
      } else if (kind == "CategoricalPredictor") {
        const std::string field = child.get_attribute("name");
        vector_fields.push_back(
            {indexer->get_index(field), true, Value(child.get_attribute("value"), indexer->get_type(field))});
      } else if (kind != "Extension" && !kind.empty()) {
        throw cpmml::ParsingException("SupportVectorMachineModel: unexpected element " + kind + " in VectorFields");
      }
    }
  }

  // Callers check has_missing_input first; the throw is a backstop. A missing
  // CategoricalPredictor field gives 0 for each of its indicators.
  Eigen::VectorXd sample_to_vector(const Sample& sample) const {
    Eigen::VectorXd x(static_cast<Eigen::Index>(vector_fields.size()));
    for (size_t i = 0; i < vector_fields.size(); i++) {
      const VectorField& f = vector_fields[i];
      const Value& v = sample[f.index].value;
      if (f.is_indicator) {
        x[static_cast<Eigen::Index>(i)] = !v.missing && v == f.category ? 1.0 : 0.0;
        continue;
      }
      if (v.missing) throw cpmml::MissingValueException("missing value for SVM vector field");
      x[static_cast<Eigen::Index>(i)] = v.value;
    }
    return x;
  }

  // --- VectorDictionary parsing ---

  std::unordered_map<std::string, Eigen::VectorXd> parse_vector_dictionary(const XmlNode& dict_node) const {
    std::unordered_map<std::string, Eigen::VectorXd> result;
    const size_t n_fields = vector_fields.size();

    for (const auto& vi : dict_node.get_childs("VectorInstance")) {
      const std::string id = vi.get_attribute("id");
      result[id] = parse_vector_instance(vi, n_fields);
    }
    return result;
  }

  // Parse a VectorInstance element. PMML 4.4 allows exactly one of
  // REAL-SparseArray or the REAL-ARRAY group, i.e. <Array type="real">.
  static Eigen::VectorXd parse_vector_instance(const XmlNode& vi, size_t n) {
    const std::string id = vi.get_attribute("id");
    const auto fail = [&id](const std::string& what) {
      return cpmml::ParsingException("SupportVectorMachineModel: VectorInstance '" + id + "': " + what);
    };

    if (vi.exists_child("REAL-SparseArray")) {
      const XmlNode sa = vi.get_child("REAL-SparseArray");
      // Entries not listed take defaultValue (schema default 0).
      const double fill = sa.get_double_attribute("defaultValue", 0.0);
      Eigen::VectorXd vec = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n), fill);
      if (sa.exists_attribute("n") && sa.required_long_attribute("n") != static_cast<long>(n))
        throw fail("REAL-SparseArray n=" + sa.get_attribute("n") + " but VectorFields has " + std::to_string(n));

      const std::vector<std::string> raw_indices =
          sa.exists_child("Indices") ? parse_array(sa.get_child("Indices").value()) : std::vector<std::string>{};
      const std::vector<std::string> entries =
          sa.exists_child("REAL-Entries") ? parse_array(sa.get_child("REAL-Entries").value()) : std::vector<std::string>{};
      if (raw_indices.size() != entries.size())
        throw fail(std::to_string(raw_indices.size()) + " Indices but " + std::to_string(entries.size()) +
                   " REAL-Entries");
      for (size_t k = 0; k < raw_indices.size(); k++) {
        // PMML sparse array indices are 1-based
        const double pos = to_double(raw_indices[k]);
        if (pos < 1 || pos > double(n) || pos != std::floor(pos)) throw fail("index " + raw_indices[k] + " out of range");
        vec[static_cast<Eigen::Index>(pos) - 1] = to_double(entries[k]);
      }
      return vec;
    }

    if (vi.exists_child("Array")) {
      const XmlNode arr = vi.get_child("Array");
      // REAL-ARRAY is an Array of type "real"; its optional n must match its entries.
      if (arr.get_attribute("type") != "real") throw fail("Array type must be \"real\", got \"" + arr.get_attribute("type") + "\"");
      const std::vector<std::string> entries = parse_array(arr.value());
      if (arr.exists_attribute("n") && arr.required_long_attribute("n") != static_cast<long>(entries.size()))
        throw fail("Array n=" + arr.get_attribute("n") + " but it has " + std::to_string(entries.size()) + " entries");
      if (entries.size() != n)
        throw fail("Array has " + std::to_string(entries.size()) + " entries but VectorFields has " +
                   std::to_string(n));
      Eigen::VectorXd vec(static_cast<Eigen::Index>(n));
      for (size_t k = 0; k < n; k++) vec[static_cast<Eigen::Index>(k)] = to_double(entries[k]);
      return vec;
    }

    throw fail("expected REAL-SparseArray or Array");
  }

  // --- SupportVectorMachine parsing ---

  SvmMachine parse_machine(const XmlNode& node,
                           const std::unordered_map<std::string, Eigen::VectorXd>& vector_dict) const {
    SvmMachine m;
    m.target_category = node.get_attribute("targetCategory");
    m.alternate_category = node.get_attribute("alternateTargetCategory");
    m.threshold = node.get_double_attribute("threshold", model_threshold);

    const XmlNode coeff_node = node.get_child("Coefficients");
    m.bias = coeff_node.get_double_attribute("absoluteValue", 0.0);

    if (use_support_vectors) {
      // Build (αᵢ, xᵢ) pairs from SupportVectors + Coefficients
      const std::vector<XmlNode> sv_refs = node.get_child("SupportVectors").get_childs("SupportVector");
      const std::vector<XmlNode> coeff_refs = coeff_node.get_childs("Coefficient");

      m.alphas.reserve(sv_refs.size());
      m.support_vectors.reserve(sv_refs.size());

      if (sv_refs.size() != coeff_refs.size())
        throw cpmml::ParsingException("SupportVectorMachineModel: " + std::to_string(sv_refs.size()) +
                                      " SupportVector but " + std::to_string(coeff_refs.size()) + " Coefficient");

      for (size_t i = 0; i < sv_refs.size(); i++) {
        const std::string vid = sv_refs[i].get_attribute("vectorId");
        const auto it = vector_dict.find(vid);
        if (it == vector_dict.end())
          throw cpmml::ParsingException("SupportVectorMachineModel: SupportVector references unknown vectorId '" +
                                        vid + "'");
        m.alphas.push_back(coeff_refs[i].get_double_attribute("value", 0.0));
        m.support_vectors.push_back(it->second);
      }
    } else {
      // Coefficients representation: one weight per feature
      m.is_weight_vector = true;
      const std::vector<XmlNode> coeff_refs = coeff_node.get_childs("Coefficient");
      m.weight_vector = Eigen::VectorXd(static_cast<Eigen::Index>(coeff_refs.size()));
      if (coeff_refs.size() != vector_fields.size())
        throw cpmml::ParsingException("SupportVectorMachineModel: " + std::to_string(coeff_refs.size()) +
                                      " Coefficient but " + std::to_string(vector_fields.size()) + " VectorFields");
      for (size_t i = 0; i < coeff_refs.size(); i++)
        m.weight_vector[static_cast<Eigen::Index>(i)] = coeff_refs[i].get_double_attribute("value", 0.0);
    }

    return m;
  }

  // --- Classification methods ---

  // Binary (a single machine) and one-against-one both decide by comparing each
  // machine's value with its threshold; one-against-all compares the machines'
  // values with each other.
  bool votes_by_threshold() const { return is_ovo || machines.size() == 1; }

  // PMML 4.4: "If maxWins is true and the prediction is larger than the
  // threshold or maxWins is false and the prediction is smaller than the
  // threshold, the class label is the targetCategory attribute, otherwise, it is
  // the alternateTargetCategory attribute." Equality goes to the alternate.
  bool votes_for_target(const SvmMachine& m, double value) const {
    return max_wins ? value > m.threshold : value < m.threshold;
  }

  size_t class_index(const std::string& cls) const {
    return static_cast<size_t>(std::find(classes.begin(), classes.end(), cls) - classes.begin());
  }

  // Votes per class (aligned with `classes`). `order` receives the class
  // indices in the order in which each class got its first vote.
  std::vector<int> count_votes(const std::vector<double>& values, std::vector<size_t>& order) const {
    std::vector<int> votes(classes.size(), 0);
    order.clear();
    for (size_t i = 0; i < machines.size(); i++) {
      const SvmMachine& m = machines[i];
      const size_t c = class_index(votes_for_target(m, values[i]) ? m.target_category : m.alternate_category);
      if (votes[c]++ == 0) order.push_back(c);
    }
    return votes;
  }

  // "In case of a tie, the predicted class label is the first category with
  // maximal number of votes". The spec does not say first in which order;
  // JPMML 1.7.7 takes the tied class that received a vote first (machines in
  // document order), and so does this.
  static size_t most_voted(const std::vector<int>& votes, const std::vector<size_t>& order) {
    size_t best = order.front();
    for (const size_t c : order)
      if (votes[c] > votes[best]) best = c;
    return best;
  }

  // One-against-all: "Depending on the model attribute maxWins, the SVM with
  // the largest or the smallest value determines the predicted class label."
  // Ties go to the first machine.
  size_t best_machine(const std::vector<double>& values) const {
    size_t best = 0;
    for (size_t i = 1; i < values.size(); i++)
      if (max_wins ? values[i] > values[best] : values[i] < values[best]) best = i;
    return best;
  }
};

#endif
