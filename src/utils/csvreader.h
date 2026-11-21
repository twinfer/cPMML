
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CSVREADER_H
#define CSVREADER_H

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils.h"

/**
 * @class CSVReader
 *
 * Class implementing a simple csv file reader.
 *
 * Though it is used for benchmarking and testing of cPMML, it is not part of
 * its core or of its API, since it's best-effort and not directly involved in
 * model scoring.
 */
class CSVReader {
 public:
  explicit CSVReader(const std::string& filename) : file(filename) {
    for (const auto& field : split_row(read_line())) this->header.push_back(field);
  }

  std::unordered_map<std::string, std::string> read() {
    std::unordered_map<std::string, std::string> result;
    const std::string row = read_line();
    if (row.empty()) return result;

    std::vector<bool> quoted;
    const std::vector<std::string> fields = split_row(row, &quoted);
    quoted_empty_.clear();
    for (size_t i = 0; i < header.size() && i < fields.size(); i++) {
      result[header[i]] = fields[i];
      if (quoted[i] && fields[i].empty()) quoted_empty_.insert(header[i]);
    }

    return result;
  }

  // The columns of the row read last whose cell was written `""`, a quoted
  // empty field, which read() gives as "" as it does an empty cell. The test
  // driver reads the first as the empty string and the second as no value
  // (test/pmml_tester.cc, design P6.8).
  bool quoted_empty(const std::string& column) const { return quoted_empty_.count(column) > 0; }

  std::string read_raw() { return read_line(); }

 private:
  // std::istream::getline into a fixed buffer sets failbit on an over-long
  // line, so a single row past the buffer did not merely truncate itself -- it
  // wedged the stream and every subsequent read() returned nothing, silently
  // ending the fixture early. std::getline grows the string instead.
  std::string read_line() {
    std::string row;
    if (!std::getline(file, row)) return std::string();
    if (!row.empty() && row.back() == '\r') row.pop_back();
    return row;
  }

  // RFC 4180 quoting: a field may be quoted, and inside a quoted field a
  // literal double quote is written as two of them. The old reader treated the
  // first inner quote as the terminator, so a value like "say ""hi""" came back
  // as `say ` with the rest of the row shifted into the wrong columns.
  static std::vector<std::string> split_row(const std::string& row, std::vector<bool>* quoted = nullptr) {
    std::vector<std::string> fields;
    size_t pos = 0;

    while (true) {
      std::string field;
      if (quoted) quoted->push_back(pos < row.size() && row[pos] == '"');
      if (pos < row.size() && row[pos] == '"') {
        ++pos;
        while (pos < row.size()) {
          if (row[pos] == '"') {
            if (pos + 1 < row.size() && row[pos + 1] == '"') {  // "" is one literal quote
              field += '"';
              pos += 2;
              continue;
            }
            ++pos;  // closing quote
            break;
          }
          field += row[pos++];
        }
      } else {
        const size_t comma = row.find(',', pos);
        const size_t end = (comma == std::string::npos) ? row.size() : comma;
        field = row.substr(pos, end - pos);
        pos = end;
      }

      fields.push_back(remove_all(field, '\r'));
      if (pos >= row.size()) break;
      if (row[pos] == ',') ++pos;
    }

    return fields;
  }

  std::ifstream file;
  std::vector<std::string> header;
  std::unordered_set<std::string> quoted_empty_;
};

#endif
