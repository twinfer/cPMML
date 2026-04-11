
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CSVREADER_H
#define CSVREADER_H

#include <fstream>
#include <sstream>
#include <vector>

#include "utils.h"

#define MAX_LINE_LENGTH 8192

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
    this->file.getline(line, MAX_LINE_LENGTH);
    std::stringstream line_stream(line);
    std::string field;
    while (getline(line_stream, field, ',')) this->header.push_back(remove_all(remove_all(field, '\r'), '"'));
  }

  std::unordered_map<std::string, std::string> read() {
    std::unordered_map<std::string, std::string> result;
    line[0] = 0;
    this->file.getline(line, MAX_LINE_LENGTH);
    if (line[0] == 0) return result;

    std::string row(line);
    // Remove trailing CR if present
    if (!row.empty() && row.back() == '\r') row.pop_back();

    std::vector<std::string> fields;
    size_t pos = 0;
    while (pos <= row.size()) {
      std::string field;
      if (pos < row.size() && row[pos] == '"') {
        // Quoted field: read until closing quote
        ++pos;
        while (pos < row.size() && row[pos] != '"') field += row[pos++];
        if (pos < row.size()) ++pos;  // skip closing quote
        // skip comma after closing quote
        if (pos < row.size() && row[pos] == ',') ++pos;
      } else {
        size_t comma = row.find(',', pos);
        if (comma == std::string::npos) {
          field = row.substr(pos);
          pos = row.size() + 1;
        } else {
          field = row.substr(pos, comma - pos);
          pos = comma + 1;
        }
      }
      fields.push_back(field);
      if (fields.size() >= header.size()) break;
    }

    for (size_t i = 0; i < header.size() && i < fields.size(); i++)
      result[header[i]] = fields[i];

    return result;
  }

  std::string read_raw() {
    this->file.getline(line, MAX_LINE_LENGTH);
    return line;
  }

 private:
  std::ifstream file;
  std::vector<std::string> header;
  char line[MAX_LINE_LENGTH];
};

#endif
