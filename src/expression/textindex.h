
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TEXTINDEX_H
#define CPMML_TEXTINDEX_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/value.h"
#include "expression.h"
#include "fieldref.h"

/**
 * @class TextIndex
 *
 * Implementation of <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_TextIndex">PMML
 * TextIndex</a>.
 *
 * Counts how often a search term occurs in a text field and applies a local
 * term-weight transformation.
 *
 * Tokenization follows the spec: the text is split on `wordSeparatorCharacterRE`,
 * whose default is `\s+`. There is no character-class filtering anywhere in the
 * definition -- this used to strip every non-alphabetic character out of each
 * token while leaving the search term untouched, so a term containing a digit
 * ("covid19") or punctuation could never match anything, and a term of more
 * than one word could not match either because only single tokens were compared.
 *
 * localTermWeights, verbatim from the spec, with freq the number of hits and
 * X the indicator function:
 *   termFrequency                    x = freq
 *   binary                           x = X(freq)
 *   logarithmic                      x = log(1 + freq)      -- base 10
 *   augmentedNormalizedTermFrequency x = 0.5 * (X(freq) + freq / max_k(freq_k))
 *
 * Only maxLevenshteinDistance="0" is implemented; a document asking for
 * approximate matching is rejected rather than silently scored as exact.
 * countHits then has nothing to choose between, since allHits and bestHits
 * differ only in how non-zero edit distances are counted.
 */
class TextIndex : public Expression {
 public:
  enum class LocalTermWeights { TERM_FREQUENCY, BINARY, LOGARITHMIC, AUGMENTED_NORMALIZED };

  size_t field_index = std::numeric_limits<size_t>::max();
  LocalTermWeights local_weights = LocalTermWeights::TERM_FREQUENCY;
  bool case_sensitive = false;
  bool tokenize = true;

  // Non-default wordSeparatorCharacterRE only; the default `\s+` takes the
  // allocation-free path below.
  bool custom_separator = false;
  std::regex separator;

  // A Constant term is fixed at parse time; a FieldRef one has to be read from
  // the sample on every call. The FieldRef case used to leave `term` empty and
  // therefore counted zero, always.
  std::vector<std::string> constant_term;
  std::shared_ptr<FieldRef> term_field;

  TextIndex() = default;

  TextIndex(const XmlNode& node, const size_t& output_index, const DataType& output_type,
            const std::shared_ptr<Indexer>& indexer)
      : Expression(output_index, output_type, indexer) {
    const std::string field_name = node.required_attribute("textField");
    field_index = indexer->get_or_set(field_name);
    inputs.insert(field_name);

    // Case-sensitive, as in JPMML; "TermFrequency" used to be accepted.
    static constexpr EnumEntry<LocalTermWeights> LOCAL_TERM_WEIGHTS_TABLE[] = {
        {"termFrequency", LocalTermWeights::TERM_FREQUENCY},
        {"binary", LocalTermWeights::BINARY},
        {"logarithmic", LocalTermWeights::LOGARITHMIC},
        {"augmentedNormalizedTermFrequency", LocalTermWeights::AUGMENTED_NORMALIZED},
    };
    local_weights = read_enum(node, "localTermWeights", LOCAL_TERM_WEIGHTS_TABLE, LocalTermWeights::TERM_FREQUENCY);
    // countHits is validated though, with maxLevenshteinDistance 0, both
    // values count the same hits.
    static constexpr EnumEntry<bool> COUNT_HITS_TABLE[] = {{"allHits", true}, {"bestHits", false}};
    read_enum(node, "countHits", COUNT_HITS_TABLE, true);

    case_sensitive = node.get_bool_attribute("isCaseSensitive", false);

    tokenize = node.get_bool_attribute("tokenize", tokenize);

    if (node.exists_attribute("maxLevenshteinDistance") &&
        node.required_double_attribute("maxLevenshteinDistance") != 0.0)
      throw cpmml::ParsingException("TextIndex: only maxLevenshteinDistance=\"0\" is supported, got " +
                                    node.required_attribute("maxLevenshteinDistance"));

    if (node.exists_attribute("wordSeparatorCharacterRE")) {
      const std::string re = node.required_attribute("wordSeparatorCharacterRE");
      if (re != "\\s+") {
        custom_separator = true;
        try {
          separator = std::regex(re, std::regex::ECMAScript);
        } catch (const std::regex_error& e) {
          throw cpmml::ParsingException("TextIndex: invalid wordSeparatorCharacterRE: " + re);
        }
      }
    }

    parse_term(node, output_index, output_type, indexer);
  }

  inline Value eval(const Sample& sample) const override {
    const Value& fv = sample[field_index].value;
    if (fv.missing) return Value();

    const std::vector<std::string> term = current_term(sample);
    if (term.empty()) return Value(0.0);

    const std::vector<std::string> text = tokenized(fold(fv.svalue));

    const size_t freq = count(text, term);

    double result = 0.0;
    switch (local_weights) {
      case LocalTermWeights::TERM_FREQUENCY:
        result = static_cast<double>(freq);
        break;
      case LocalTermWeights::BINARY:
        result = freq > 0 ? 1.0 : 0.0;
        break;
      case LocalTermWeights::LOGARITHMIC:
        // "take the logarithm (base 10) of 1 + the number of times the term
        // occurs" -- this was a natural log, off by a factor of ln(10).
        result = std::log10(1.0 + static_cast<double>(freq));
        break;
      case LocalTermWeights::AUGMENTED_NORMALIZED: {
        const size_t max_freq = max_frequency(text, term.size());
        const double normalized = max_freq > 0 ? static_cast<double>(freq) / static_cast<double>(max_freq) : 0.0;
        result = 0.5 * ((freq > 0 ? 1.0 : 0.0) + normalized);
        break;
      }
    }

    return Value(result);
  }

 private:
  void parse_term(const XmlNode& node, const size_t& output_index, const DataType& output_type,
                  const std::shared_ptr<Indexer>& indexer) {
    // "a single EXPRESSION element nested within containing the term value to
    // look for, which will usually be a simple Constant."
    if (node.exists_child("Constant")) {
      constant_term = tokenized(fold(node.get_child("Constant").value()));
      return;
    }
    if (node.exists_child("FieldRef")) {
      term_field = std::make_shared<FieldRef>(node.get_child("FieldRef"), output_index, output_type, indexer);
      inputs.insert(term_field->field_name);
      return;
    }

    for (const auto& child : node.get_childs())
      if (child.name() != "Extension")
        throw cpmml::ParsingException("TextIndex: unsupported term expression <" + child.name() +
                                      ">, only Constant and FieldRef are implemented");

    throw cpmml::ParsingException("TextIndex: no term expression");
  }

  inline std::vector<std::string> current_term(const Sample& sample) const {
    if (!term_field) return constant_term;
    const Value v = term_field->eval(sample);
    return v.missing ? std::vector<std::string>() : tokenized(fold(v.svalue));
  }

  inline std::string fold(std::string s) const { return case_sensitive ? s : to_lower(s); }

  std::vector<std::string> tokenized(const std::string& s) const {
    // tokenize="false" makes the field a single token matched as a whole.
    if (!tokenize) return s.empty() ? std::vector<std::string>() : std::vector<std::string>{s};

    std::vector<std::string> tokens;
    if (custom_separator) {
      for (std::sregex_token_iterator it(s.cbegin(), s.cend(), separator, -1), end; it != end; ++it)
        if (it->length() > 0) tokens.push_back(*it);
      return tokens;
    }

    std::string token;
    for (char c : s) {
      if (std::isspace(static_cast<unsigned char>(c))) {
        if (!token.empty()) tokens.push_back(std::move(token));
        token.clear();
      } else {
        token += c;
      }
    }
    if (!token.empty()) tokens.push_back(std::move(token));
    return tokens;
  }

  // Occurrences of the term's token sequence in the text's. A term of several
  // words is ordinary -- "credit card" is one term -- so this is a subsequence
  // count, not a token equality count.
  static size_t count(const std::vector<std::string>& text, const std::vector<std::string>& term) {
    if (term.empty() || text.size() < term.size()) return 0;

    size_t hits = 0;
    for (size_t i = 0; i + term.size() <= text.size(); i++)
      if (std::equal(term.cbegin(), term.cend(), text.cbegin() + static_cast<std::ptrdiff_t>(i))) hits++;

    return hits;
  }

  // max_k(freq_k): the highest frequency of any term in the document. Windows
  // of the search term's own length, so a multi-word term is measured against
  // multi-word candidates rather than against single words.
  static size_t max_frequency(const std::vector<std::string>& text, size_t width) {
    if (width == 0 || text.size() < width) return 0;

    std::unordered_map<std::string, size_t> counts;
    size_t best = 0;
    for (size_t i = 0; i + width <= text.size(); i++) {
      std::string key;
      for (size_t j = 0; j < width; j++) {
        key += text[i + j];
        key += '\x1f';  // unit separator: keeps "ab c" and "a bc" distinct
      }
      best = std::max(best, ++counts[key]);
    }

    return best;
  }
};

#endif
