
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TEXTMODEL_H
#define CPMML_TEXTMODEL_H

#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/enum_parser.h"
#include "core/fieldusagetype.h"
#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "utils/utils.h"

/**
 * @class TextEvaluator
 *
 * Implementation of InternalEvaluator for <a
 * href="http://dmg.org/pmml/v4-4/Text.html">PMML TextModel</a>.
 *
 * Scoring: tokenise the input text field → compute weighted TF vector →
 * optionally normalise → cosine or Euclidean similarity against each
 * training document in the DocumentTermMatrix → return the class label
 * of the most similar document.
 *
 * Supported TextModelNormalization options:
 *   localTermWeights : termFrequency (default), binary, logarithmic
 *   documentNormalization : none (default), cosine (L2 normalise rows)
 *
 * TextModelSimilarity: cosineSimilarity (default), euclideanDistance
 */
class TextEvaluator : public InternalEvaluator {
 public:
  std::string text_field;
  std::string target_field_name;

  std::vector<std::string> vocabulary;
  std::unordered_map<std::string, size_t> vocab_index;
  Eigen::MatrixXd doc_matrix;  // n_docs × n_terms
  std::vector<std::string> class_labels;

  enum class LocalWeights { TERM_FREQUENCY, BINARY, LOGARITHMIC };

  bool use_cosine = true;                                    // TextModelSimiliarity
  bool normalize_docs = false;                               // documentNormalization (default "none")
  LocalWeights local_weights = LocalWeights::TERM_FREQUENCY;  // localTermWeights

  // TextModelNormalization (4.4.1). augmentedNormalizedTermFrequency used to
  // fall through to raw counts; it is a valid value this model does not
  // implement. globalTermWeights is not read (see the known gap in the
  // commit introducing this table).
  static constexpr EnumEntry<LocalWeights> LOCAL_TERM_WEIGHTS_TABLE[] = {
      {"termFrequency", LocalWeights::TERM_FREQUENCY},
      {"binary", LocalWeights::BINARY},
      {"logarithmic", LocalWeights::LOGARITHMIC},
      {"augmentedNormalizedTermFrequency", LocalWeights::TERM_FREQUENCY, nullptr, EnumSupport::UNSUPPORTED},
  };
  static constexpr EnumEntry<bool> DOCUMENT_NORMALIZATION_TABLE[] = {
      {"none", false},
      {"cosine", true},
  };
  // TextModelSimiliarity@similarityType (true: cosine).
  static constexpr EnumEntry<bool> SIMILARITY_TYPE_TABLE[] = {
      {"euclidean", false},
      {"cosine", true},
  };
  // Matrix@kind (GeneralStructure). The cells are read either as one Array per
  // row or as MatCells; diagonal and symmetric store only part of the matrix,
  // which this reader does not expand.
  static constexpr EnumEntry<bool> MATRIX_KIND_TABLE[] = {
      {"any", true},
      {"diagonal", false, nullptr, EnumSupport::UNSUPPORTED},
      {"symmetric", false, nullptr, EnumSupport::UNSUPPORTED},
  };

  explicit TextEvaluator(const XmlNode& pmml_root) : InternalEvaluator(pmml_root) {
    XmlNode model = pmml_root.get_child("TextModel");

    // Identify text (first active) and target (predicted) field
    for (const auto& mf : model.get_child("MiningSchema").get_childs("MiningField")) {
      const FieldUsageType usage = FieldUsageType::read(mf);
      std::string fname = mf.get_attribute("name");
      if (usage == FieldUsageType::FieldUsageTypeValue::ACTIVE && text_field.empty())
        text_field = fname;
      else if (usage == FieldUsageType::FieldUsageTypeValue::TARGET)
        target_field_name = fname;
    }

    parse_normalization(model);  // must come before parse_matrix
    parse_vocabulary(model);
    parse_matrix(model);
    parse_labels(model);
    parse_similarity(model);
  }

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return std::make_unique<InternalScore>(classify(flatten_input(arguments)));
  }

  inline bool validate(const Input& arguments) const override {
    auto flat = flatten_input(arguments);
    return flat.count(text_field) > 0;
  }

  inline std::string get_target_name() const override { return target_field_name; }
  inline std::string output_name() const override { return target_field_name; }
  inline std::string mining_function_name() const override { return "CLASSIFICATION"; }

 private:
  // -----------------------------------------------------------------------
  // Parsing
  // -----------------------------------------------------------------------

  void parse_normalization(const XmlNode& model) {
    if (!model.exists_child("TextModelNormalization")) return;
    XmlNode tn = model.get_child("TextModelNormalization");
    normalize_docs = read_enum(tn, "documentNormalization", DOCUMENT_NORMALIZATION_TABLE, false);
    local_weights = read_enum(tn, "localTermWeights", LOCAL_TERM_WEIGHTS_TABLE, LocalWeights::TERM_FREQUENCY);
  }

  void parse_vocabulary(const XmlNode& model) {
    XmlNode arr = model.get_child("TextDictionary").get_child("Array");
    std::vector<std::string> terms = parse_array(arr.value());
    for (size_t i = 0; i < terms.size(); ++i) {
      vocabulary.push_back(terms[i]);
      vocab_index[terms[i]] = i;
    }
  }

  void parse_matrix(const XmlNode& model) {
    XmlNode mat = model.get_child("DocumentTermMatrix").get_child("Matrix");
    int n_rows = static_cast<int>(mat.required_long_attribute("nbRows"));
    int n_cols = static_cast<int>(mat.required_long_attribute("nbCols"));
    read_enum(mat, "kind", MATRIX_KIND_TABLE, true);

    doc_matrix = Eigen::MatrixXd::Zero(n_rows, n_cols);

    // kind="sparse" used to select the MatCell form; it is not a Matrix kind.
    // The form is the content: MatCells or Arrays.
    if (mat.exists_child("MatCell")) {
      for (const auto& cell : mat.get_childs("MatCell")) {
        int r = static_cast<int>(cell.required_long_attribute("row")) - 1;  // 1-based → 0-based
        int c = static_cast<int>(cell.required_long_attribute("col")) - 1;
        doc_matrix(r, c) = to_double(cell.value());
      }
    } else {
      // Dense: one Array element per row
      int r = 0;
      for (const auto& row_arr : mat.get_childs("Array")) {
        std::vector<std::string> vals = parse_array(row_arr.value());
        for (int c = 0; c < static_cast<int>(vals.size()) && c < n_cols; ++c)
          doc_matrix(r, c) = to_double(vals[static_cast<size_t>(c)]);
        ++r;
      }
    }

    // documentNormalization="cosine" → L2-normalise each stored document row
    if (normalize_docs) {
      for (int i = 0; i < doc_matrix.rows(); ++i) {
        double norm = doc_matrix.row(i).norm();
        if (norm > 1e-10) doc_matrix.row(i) /= norm;
      }
    }
  }

  void parse_labels(const XmlNode& model) {
    if (!model.exists_child("ClassLabels")) return;
    XmlNode arr = model.get_child("ClassLabels").get_child("Array");
    class_labels = parse_array(arr.value());
  }

  // TextModelSimiliarity -- the spec's spelling of the element -- with
  // similarityType euclidean or cosine. This read an element spelled
  // TextModelSimilarity with values cosineSimilarity / euclideanDistance (or a
  // <euclideanDistance/> child), none of which is PMML, and took cosine for
  // anything else. The attribute has no default in the schema; absent keeps
  // cosine, as before.
  void parse_similarity(const XmlNode& model) {
    if (model.exists_child("TextModelSimilarity"))
      throw cpmml::ParsingException(
          "TextModel: invalid element TextModelSimilarity; the PMML element is TextModelSimiliarity");
    if (!model.exists_child("TextModelSimiliarity")) return;
    use_cosine = read_enum(model.get_child("TextModelSimiliarity"), "similarityType", SIMILARITY_TYPE_TABLE, true);
  }

  // -----------------------------------------------------------------------
  // Inference
  // -----------------------------------------------------------------------

  std::string classify(const std::unordered_map<std::string, std::string>& sample) const {
    if (!sample.count(text_field) || class_labels.empty()) return "";
    Eigen::VectorXd q = vectorize(sample.at(text_field));  // always L2-normalised

    int best = 0;
    double best_sim = -std::numeric_limits<double>::max();

    for (int i = 0; i < doc_matrix.rows(); ++i) {
      Eigen::VectorXd row = doc_matrix.row(i);
      double sim;
      if (use_cosine) {
        // q is unit-length; if doc is not, divide by its norm
        if (normalize_docs) {
          sim = q.dot(row);  // both unit → dot = cosine
        } else {
          double dr = row.norm();
          sim = (dr < 1e-10) ? 0.0 : q.dot(row) / dr;
        }
      } else {
        sim = -(q - row).norm();  // negated distance: max = closest
      }
      if (sim > best_sim) {
        best_sim = sim;
        best = i;
      }
    }

    return (best < static_cast<int>(class_labels.size())) ? class_labels[static_cast<size_t>(best)] : "";
  }

  // Tokenise text and apply localTermWeights
  Eigen::VectorXd vectorize(const std::string& text) const {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(vocabulary.size()));
    std::string lower_text = to_lower(text);
    std::istringstream iss(lower_text);
    std::string token;
    while (iss >> token) {
      token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) { return !std::isalpha(c); }),
                  token.end());
      auto it = vocab_index.find(token);
      if (it != vocab_index.end()) v[static_cast<Eigen::Index>(it->second)] += 1.0;
    }

    // Apply localTermWeights
    if (local_weights == LocalWeights::BINARY) {
      for (Eigen::Index i = 0; i < v.size(); ++i) v[i] = (v[i] > 0) ? 1.0 : 0.0;
    } else if (local_weights == LocalWeights::LOGARITHMIC) {
      for (Eigen::Index i = 0; i < v.size(); ++i) v[i] = (v[i] > 0) ? 1.0 + std::log(v[i]) : 0.0;
    }
    // termFrequency (default): keep raw counts

    // Always L2-normalise the query vector
    double norm = v.norm();
    if (norm > 1e-10) v /= norm;
    return v;
  }
};

#endif
