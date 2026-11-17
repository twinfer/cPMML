/*******************************************************************************
 * fuzz_runner -- the cPMML side of the differential fuzzer
 * (docs/design/typed-core.md §8.2): scores every case of a corpus in one
 * process, as tools/fuzz/JpmmlOracle.java does for JPMML.
 *
 * Usage: fuzz_runner.exe MANIFEST
 *
 * MANIFEST has one case per line, tab-separated:
 *   case-id  model.pmml  inputs.csv  output.csv
 * inputs.csv holds the case's input columns (RFC 4180; an unquoted N/A is a
 * missing value and is left out of the cpmml::Input, any other cell -- the
 * empty one included -- is passed as that text). output.csv gets, per input
 * row, the columns pmml_tester --dump writes: the target field, the
 * OutputFields of the model element, then the OutputFields declared in
 * segments; N/A for no value; a numeric OutputField through cPMML's Java
 * formatter by its dataType. A row whose evaluation throws has every cell
 * "ERROR: <what>"; a document that does not load gives a one-column file
 * "LOAD-ERROR" with the exception in its only row.
 *
 * Not part of ctest (P11 adds the quick fuzz run under the label fuzz).
 *******************************************************************************/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <pugixml.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cPMML.h"
#include "utils/java_format.h"

namespace {

const char* const MISSING = "N/A";

struct Cell {
  std::string text;
  bool missing = false;
};

std::vector<Cell> split_row(const std::string& line) {
  std::vector<Cell> row;
  std::string current;
  bool quoted = false, was_quoted = false;
  for (size_t i = 0; i < line.size(); i++) {
    const char c = line[i];
    if (quoted) {
      if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
        current += '"';
        i++;
      } else if (c == '"') {
        quoted = false;
      } else {
        current += c;
      }
    } else if (c == '"') {
      quoted = was_quoted = true;
    } else if (c == ',') {
      row.push_back({current, !was_quoted && current == MISSING});
      current.clear();
      was_quoted = false;
    } else {
      current += c;
    }
  }
  row.push_back({current, !was_quoted && current == MISSING});
  return row;
}

std::string csv_cell(const std::string& text) {
  if (text.find_first_of(",\"\n") == std::string::npos) return text;
  std::string quoted = "\"";
  for (const char c : text) quoted += c == '"' ? std::string("\"\"") : std::string(1, c);
  return quoted + "\"";
}

std::string one_line(std::string text) {
  std::replace(text.begin(), text.end(), '\n', ' ');
  std::replace(text.begin(), text.end(), '\r', ' ');
  return text;
}

void write_row(std::ostream& out, const std::vector<std::string>& cells) {
  for (size_t i = 0; i < cells.size(); i++) out << (i ? "," : "") << csv_cell(cells[i]);
  out << "\n";
}

java::Kind kind_of_type(const std::string& datatype) {
  if (datatype == "integer") return java::Kind::INTEGER;
  if (datatype == "float") return java::Kind::FLOAT;
  if (datatype == "boolean") return java::Kind::BOOLEAN;
  return java::Kind::DOUBLE;
}

pugi::xml_node model_element(const pugi::xml_node& pmml) {
  for (const auto& child : pmml.children())
    if (child.attribute("functionName")) return child;
  return {};
}

std::string target_name(const pugi::xml_node& top) {
  for (const auto& field : top.child("MiningSchema").children("MiningField")) {
    const std::string usage = field.attribute("usageType").value();
    if (usage == "target" || usage == "predicted") return field.attribute("name").value();
  }
  return "";
}

// The columns and the Java kind of each numeric OutputField, as pmml_tester's
// run_dump decides them.
void layout(const pugi::xml_node& pmml, std::vector<std::string>& columns,
            std::unordered_map<std::string, java::Kind>& kinds) {
  const pugi::xml_node top = model_element(pmml);
  std::unordered_map<std::string, std::string> types;
  std::unordered_set<std::string> data_fields;
  for (const auto& field : pmml.child("DataDictionary").children("DataField")) {
    types[field.attribute("name").value()] = field.attribute("dataType").value();
    data_fields.insert(field.attribute("name").value());
  }
  const std::string target = target_name(top);
  columns.assign(1, target);
  for (const auto& node : pmml.select_nodes("//Output/OutputField")) {
    const pugi::xml_node of = node.node();
    const std::string name = of.attribute("name").value();
    const std::string feature = of.attribute("feature") ? of.attribute("feature").value() : "predictedValue";
    if (of.attribute("dataType"))
      kinds[name] = kind_of_type(of.attribute("dataType").value());
    else if (feature == "predictedValue" || feature == "predictedDisplayValue" || feature == "transformedValue")
      kinds[name] = kind_of_type(types[target]);
    else
      kinds[name] = java::Kind::DOUBLE;
    if (of.parent().parent() == top && std::find(columns.begin(), columns.end(), name) == columns.end())
      columns.push_back(name);
  }
  for (const auto& node : pmml.select_nodes("//Segment//Output/OutputField")) {
    const std::string name = node.node().attribute("name").value();
    if (!data_fields.count(name) && std::find(columns.begin(), columns.end(), name) == columns.end())
      columns.push_back(name);
  }
}

void run_case(const std::string& model_path, const std::string& inputs_path, const std::string& output_path) {
  std::ofstream out(output_path);
  std::vector<std::string> columns;
  std::unordered_map<std::string, java::Kind> kinds;
  cpmml::Model model;
  try {
    pugi::xml_document xml;
    if (!xml.load_file(model_path.c_str())) throw std::runtime_error("cannot parse " + model_path);
    layout(xml.child("PMML"), columns, kinds);
    model = cpmml::Model(model_path, false);
  } catch (const std::exception& e) {
    write_row(out, {"LOAD-ERROR"});
    write_row(out, {"LOAD-ERROR: " + one_line(e.what())});
    return;
  }
  const std::string target = columns.front();
  write_row(out, columns);

  std::ifstream in(inputs_path);
  std::string line;
  std::getline(in, line);
  std::vector<std::string> header;
  for (const auto& cell : split_row(line)) header.push_back(cell.text);
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto row = split_row(line);
    cpmml::Input input;
    for (size_t i = 0; i < header.size() && i < row.size(); i++)
      if (!row[i].missing) input[header[i]] = row[i].text;
    std::vector<std::string> cells;
    try {
      const cpmml::Result result = model.evaluate(input);
      const auto sout = result.str_outputs();
      const auto nout = result.num_outputs();
      for (const auto& column : columns) {
        if (column == target) {
          cells.push_back(result.empty() || result.as_string().empty() ? MISSING : result.as_string());
          continue;
        }
        const auto s = sout.find(column);
        const auto n = nout.find(column);
        if (s != sout.end() && !s->second.empty())
          cells.push_back(s->second);
        else if (n != nout.end())
          cells.push_back(java::text({n->second, kinds.count(column) ? kinds[column] : java::Kind::DOUBLE}));
        else
          cells.push_back(MISSING);
      }
    } catch (const std::exception& e) {
      cells.assign(columns.size(), "ERROR: " + one_line(e.what()));
    }
    write_row(out, cells);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: fuzz_runner.exe MANIFEST\n";
    return 2;
  }
  std::ifstream manifest(argv[1]);
  std::string line;
  size_t cases = 0;
  while (std::getline(manifest, line)) {
    if (line.empty()) continue;
    std::vector<std::string> f;
    std::stringstream ss(line);
    for (std::string part; std::getline(ss, part, '\t');) f.push_back(part);
    if (f.size() != 4) {
      std::cerr << "bad manifest line: " << line << "\n";
      return 2;
    }
    run_case(f[1], f[2], f[3]);
    cases++;
  }
  std::cerr << "fuzz_runner: " << cases << " cases\n";
  return 0;
}
