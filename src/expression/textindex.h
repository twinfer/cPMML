
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TEXTINDEX_H
#define CPMML_TEXTINDEX_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string>

#include "core/value.h"
#include "expression.h"

/**
 * @class TextIndex
 *
 * Implementation of <a
 * href="http://dmg.org/pmml/v4-4/Text.html#xsdElement_TextIndex">PMML TextIndex</a>.
 *
 * A TextIndex DerivedField computes how many times a search term occurs in a
 * text field, optionally applying a local term-weight transformation:
 *   - termFrequency (default): raw count
 *   - binary: 1 if count > 0, else 0
 *   - logarithmic: log(1 + count)
 *
 * Supported attributes:
 *   textField              : name of the input text field
 *   localTermWeights       : termFrequency | binary | logarithmic
 *   isCaseSensitive        : false (default)
 *   maxLevenshteinDistance : 0 (only exact match supported)
 *   countHits              : allHits (default)
 *   tokenize               : true (default)
 *
 * The child expression must evaluate to the search term string (typically a
 * Constant element).
 */
class TextIndex : public Expression {
 public:
  size_t field_index = std::numeric_limits<size_t>::max();
  std::string local_weights = "termfrequency";
  bool case_sensitive = false;

  // The search term extracted from the child Constant at parse time
  std::string term;

  TextIndex() = default;

  TextIndex(const XmlNode& node, const size_t& output_index, const DataType& output_type,
            const std::shared_ptr<Indexer>& indexer)
      : Expression(output_index, output_type, indexer) {
    std::string field_name = node.get_attribute("textField");
    field_index = indexer->get_or_set(field_name);
    inputs.insert(field_name);

    if (node.exists_attribute("localTermWeights")) local_weights = to_lower(node.get_attribute("localTermWeights"));

    case_sensitive =
        node.exists_attribute("isCaseSensitive") && to_lower(node.get_attribute("isCaseSensitive")) == "true";

    // The child expression holds the search term. We support a Constant child;
    // evaluate it at construction time to extract the term string.
    if (node.exists_child("Constant")) {
      term = node.get_child("Constant").value();
    } else {
      // Fallback: look for any child element and use its text content
      for (const auto& child : node.get_childs("Constant"))
        if (!child.value().empty()) {
          term = child.value();
          break;
        }
    }

    if (!case_sensitive) term = to_lower(term);
  }

  inline Value eval(const Sample& sample) const override {
    // Recover the original string for the text field via reverse lookup
    const Value& fv = sample[field_index].value;
    if (fv.missing) return Value();

    std::string text = fv.svalue;
    if (!case_sensitive) text = to_lower(text);

    // Tokenize on whitespace and count exact-match occurrences
    int count = 0;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
      // Strip non-alpha punctuation (same as TextModel)
      token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) { return !std::isalpha(c); }),
                  token.end());
      if (!token.empty() && token == term) ++count;
    }

    double result;
    if (local_weights == "binary")
      result = count > 0 ? 1.0 : 0.0;
    else if (local_weights == "logarithmic")
      result = count > 0 ? std::log(1.0 + static_cast<double>(count)) : 0.0;
    else  // termFrequency
      result = static_cast<double>(count);

    return Value(result);
  }
};

#endif
