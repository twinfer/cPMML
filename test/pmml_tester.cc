
/*******************************************************************************
 * Unified test harness for all cPMML model types.
 *
 * Usage: pmml_tester.exe <model.zip> <fixture.csv> [--locale <name>]
 *
 * Mode is determined by model.mining_function():
 *   "TIMESERIES"         →  forecast mode
 *   "ASSOCIATION_RULES"  →  transactional or scoring mode (auto-detected)
 *   otherwise            →  scoring mode
 *
 * --locale installs a host locale before scoring: results must be identical
 * under a comma-decimal locale. Exits 77 (ctest SKIP) if it is unavailable.
 *
 * pmml_tester.exe <model.zip> <inputs.csv> --dump <out.csv> writes the
 * predicted value and every OutputField as text instead of checking a fixture
 * (see run_dump; used by tools/cross-validate-text.py and
 * tools/fixture-dump-diff.sh), or for a time-series model the forecast (and
 * variance) of every step of the fixture (run_dump_forecast).
 *******************************************************************************/

#include <algorithm>
#include <clocale>
#include <fstream>
#include <cmath>
#include <iostream>
#include <limits>
#include <locale>
#include <pugixml.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cPMML.h"
#include "utils/csvreader.h"
#include "utils/utils.h"

static const double REL_TOL = 0.001;
static const double ABS_TOL = 1e-9;

static bool within_tolerance(double actual, double expected) {
  // NaN is an expectation, not an error. A model type for which PMML defines no
  // predictive variance reports NaN rather than inventing a number, and the
  // fixture asserts exactly that.
  if (std::isnan(expected)) return std::isnan(actual);
  // An infinite expectation has no relative error: inf - inf is NaN.
  if (std::isinf(expected)) return actual == expected;
  double err = std::abs(actual - expected);
  double rel = (std::abs(expected) > ABS_TOL) ? (err / std::abs(expected)) : err;
  return rel <= REL_TOL;
}

// An expected value as the oracle printed it. ::to_double reads the XSD
// lexical form only, which is right for PMML text and inputs but not for the
// fixtures: they carry what Java ("Infinity", "-Infinity") or Python ("nan",
// "inf") printed. Throws ParsingException for text that is not a number, which
// the callers catch to mean "compare as a string".
static double expected_double(const std::string& text) {
  if (text == "nan" || text == "NaN") return std::numeric_limits<double>::quiet_NaN();
  if (text == "inf" || text == "Infinity") return std::numeric_limits<double>::infinity();
  if (text == "-inf" || text == "-Infinity") return -std::numeric_limits<double>::infinity();
  return to_double(text);
}

// Convert CSV row (string map) to cpmml::Input
static cpmml::Input to_input(const std::unordered_map<std::string, std::string>& row) {
  cpmml::Input input;
  for (const auto& [k, v] : row) input[k] = v;
  return input;
}

// ---- scoring mode -----------------------------------------------------------

// Checks every column of `row` other than the predicted one against the
// OutputFields of `result`, on scored rows and on no-prediction rows alike.
//   - empty cell: not asserted (input columns, and hand-written fixtures that
//     leave an output unspecified);
//   - "N/A" (the oracle's missing value): the output must be absent, or an
//     empty string;
//   - anything else: the output must be present and match (numeric outputs
//     within tolerance, string outputs exactly).
static bool check_other_outputs(const cpmml::Result& result,
                                const std::unordered_map<std::string, std::string>& row,
                                const std::string& out_col,
                                const std::unordered_set<std::string>& declared_outputs) {
  const auto nout = result.num_outputs();
  const auto sout = result.str_outputs();
  for (const auto& [col, expected_val] : row) {
    if (col == out_col || expected_val.empty()) continue;

    auto nit = nout.find(col);
    auto sit = sout.find(col);

    if (expected_val == "N/A") {
      if (nit != nout.end()) {
        std::cerr << "output '" << col << "': predicted: " << nit->second << "  expected: no value"
                  << "  sample: " << to_string(row) << std::endl;
        return false;
      }
      if (sit != sout.end() && !sit->second.empty()) {
        std::cerr << "output '" << col << "': predicted: " << sit->second << "  expected: no value"
                  << "  sample: " << to_string(row) << std::endl;
        return false;
      }
      continue;
    }

    // A declared OutputField that produced nothing is a failure, not a
    // column to skip. Without this an evaluator that silently stopped
    // emitting an output passed every fixture that asserted it.
    if (declared_outputs.count(col) && nit == nout.end() && sit == sout.end()) {
      std::cerr << "output '" << col << "': model declares it but produced no value"
                << "  expected: " << expected_val << "  sample: " << to_string(row) << std::endl;
      return false;
    }

    if (nit != nout.end()) {
      try {
        if (!within_tolerance(nit->second, expected_double(expected_val))) {
          std::cerr << "output '" << col << "': predicted: " << nit->second << "  expected: " << expected_val
                    << "  sample: " << to_string(row) << std::endl;
          return false;
        }
      } catch (const cpmml::ParsingException&) {
        // A boolean OutputField is held as 1 or 0 and written true/false.
        if ((expected_val == "true" || expected_val == "false") && (nit->second != 0) != (expected_val == "true")) {
          std::cerr << "output '" << col << "': predicted: " << nit->second << "  expected: " << expected_val
                    << "  sample: " << to_string(row) << std::endl;
          return false;
        }
      }
      continue;
    }

    if (sit != sout.end() && sit->second != expected_val) {
      std::cerr << "output '" << col << "': predicted: " << sit->second << "  expected: " << expected_val
                << "  sample: " << to_string(row) << std::endl;
      return false;
    }
  }
  return true;
}

// The OutputFields declared inside a Segment, at any depth, other than those
// named like a DataField (a fixture column of that name is an input). A
// MiningModel may return them (selectFirst, selectAll, modelChain), and
// model.output_names() lists only the top-level ones: without these a fixture
// column asserting a segment field passed whenever the field was missing.
static std::unordered_set<std::string> segment_output_names(const std::string& model_zip) {
  std::unordered_set<std::string> names;
  const std::vector<char> document = read_file(model_zip, true);
  pugi::xml_document xml;
  if (!xml.load_string(document.data())) throw std::runtime_error("cannot parse " + model_zip);
  const pugi::xml_node pmml = xml.child("PMML");
  std::unordered_set<std::string> data_fields;
  for (const auto& field : pmml.child("DataDictionary").children("DataField"))
    data_fields.insert(field.attribute("name").value());
  for (const auto& node : pmml.select_nodes("//Segment//Output/OutputField")) {
    const std::string name = node.node().attribute("name").value();
    if (!data_fields.count(name)) names.insert(name);
  }
  return names;
}

// Whether the model element of the document predicts a number:
// functionName="regression", which an AnomalyDetectionModel has too (its
// mining_function() says CLASSIFICATION). Its predicted value is compared
// within tolerance; any other prediction is text.
static bool predicts_number(const std::string& model_zip) {
  const std::vector<char> document = read_file(model_zip, true);
  pugi::xml_document xml;
  if (!xml.load_string(document.data())) throw std::runtime_error("cannot parse " + model_zip);
  for (const auto& child : xml.child("PMML").children())
    if (child.attribute("functionName")) return std::string(child.attribute("functionName").value()) == "regression";
  return false;
}

static int run_score(cpmml::Model& model, CSVReader& reader, std::unordered_map<std::string, std::string> row,
                     const std::string& model_zip) {
  const std::string out_col = model.output_name();
  const bool numeric_prediction = predicts_number(model_zip);
  const auto names = model.output_names();
  std::unordered_set<std::string> declared_outputs(names.cbegin(), names.cend());
  for (const auto& name : segment_output_names(model_zip)) declared_outputs.insert(name);

  if (!row.count(out_col)) {
    std::cerr << "output column '" << out_col << "' not found in fixture CSV" << std::endl;
    return -1;
  }

  while (!row.empty()) {
    const std::string& expected = row.at(out_col);

    // The oracle emits "N/A" (and hand-written fixtures leave the cell blank)
    // when the model yields no prediction: require the same of cPMML rather
    // than skipping the row.
    // The other output columns of such a row are still checked: a model that
    // gives no prediction can still emit (wrong) probabilities.
    if (expected.empty() || expected == "N/A") {
      const cpmml::Result result = model.evaluate(to_input(row));
      const std::string predicted = result.as_string();
      if (!predicted.empty()) {
        std::cerr << "predicted: " << predicted << "  expected: no prediction"
                  << "  sample: " << to_string(row) << std::endl;
        return -1;
      }
      if (!check_other_outputs(result, row, out_col, declared_outputs)) return -1;
      row = reader.read();
      continue;
    }

    cpmml::Result result = model.evaluate(to_input(row));

    // The predicted value is compared as a number, within tolerance, only
    // where it is one: the prediction of a regression model, or an output
    // column that is a numeric OutputField (a probability named as the
    // output). Everything else -- a class label, a cluster id, a
    // selectAll collection, a string OutputField -- is text, and the text
    // must be the oracle's exactly: "1.0" for the class value="1" of a double
    // target, as JPMML writes it, is not "1".
    bool ok = (result.as_string() == expected);
    const auto nout = result.num_outputs();
    const auto nit = nout.find(out_col);
    if (!ok && nit != nout.end()) {
      try {
        ok = within_tolerance(nit->second, expected_double(expected));
      } catch (const cpmml::ParsingException&) {}
    } else if (!ok && numeric_prediction && !is_double_min(result.as_double())) {
      try {
        ok = within_tolerance(result.as_double(), expected_double(expected));
      } catch (const cpmml::ParsingException&) {}
    }
    if (!ok) {
      const auto sout = result.str_outputs();
      auto sit = sout.find(out_col);
      if (sit != sout.end())
        ok = (sit->second == expected);
    }

    if (!ok) {
      std::cerr << "predicted: " << result.as_string() << "  expected: " << expected << "  sample: " << to_string(row)
                << std::endl;
      return -1;
    }

    // Verify every other oracle column the model claims to produce
    // (probability/transformedValue/... OutputFields keyed by name).
    if (!check_other_outputs(result, row, out_col, declared_outputs)) return -1;
    row = reader.read();
  }
  return 0;
}

// ---- transactional scoring mode ---------------------------------------------

static int run_transactional(cpmml::Model& model, CSVReader& reader,
                              std::unordered_map<std::string, std::string> row) {
  const std::string out_col = model.output_name();

  if (!row.count(out_col)) {
    std::cerr << "output column '" << out_col << "' not found in fixture CSV" << std::endl;
    return -1;
  }

  // Detect item column.  Prefer a column literally named "item" (standard
  // JPMML fixture name).  Fallback: score with a real item to discover all
  // output columns, then pick the first non-transaction/non-output column.
  std::string item_col;
  if (row.count("item"))
    item_col = "item";

  if (item_col.empty()) {
    // Discover output columns via a trial score with the first row's value
    // in every candidate column.  Empty-basket scoring may not produce all
    // output fields, so we try each non-transaction column as item.
    std::unordered_set<std::string> output_cols;
    output_cols.insert(out_col);
    for (const auto& [k, v] : row) {
      if (k == "transaction" || k == out_col) continue;
      cpmml::Input trial;
      trial[k] = std::vector<std::string>{v};
      try {
        cpmml::Result r = model.evaluate(trial);
        for (const auto& [ok, ov] : r.str_outputs()) output_cols.insert(ok);
        for (const auto& [ok, ov] : r.num_outputs()) output_cols.insert(ok);
      } catch (...) {}
    }
    for (const auto& [k, v] : row) {
      if (k != "transaction" && !output_cols.count(k)) {
        item_col = k;
        break;
      }
    }
  }
  if (item_col.empty()) {
    std::cerr << "could not detect item column in transactional CSV" << std::endl;
    return -1;
  }

  // Group rows by transaction ID
  struct Transaction {
    std::vector<std::string> items;
    std::string expected;
  };
  std::vector<Transaction> transactions;
  std::unordered_map<std::string, size_t> tx_index;

  while (!row.empty()) {
    const std::string& tx_id = row.at("transaction");
    const std::string& expected = row.at(out_col);

    auto it = tx_index.find(tx_id);
    if (it == tx_index.end()) {
      tx_index[tx_id] = transactions.size();
      transactions.push_back({{row.at(item_col)}, expected});
    } else {
      transactions[it->second].items.push_back(row.at(item_col));
    }
    row = reader.read();
  }

  // Score each transaction
  for (const auto& tx : transactions) {
    if (tx.expected.empty()) continue;

    cpmml::Input input;
    input[item_col] = tx.items;

    cpmml::Result result = model.evaluate(input);

    bool ok = (result.as_string() == tx.expected);

    if (!ok) {
      const auto sout = result.str_outputs();
      auto sit = sout.find(out_col);
      if (sit != sout.end())
        ok = (sit->second == tx.expected);
    }
    if (!ok) {
      const auto nout = result.num_outputs();
      auto nit = nout.find(out_col);
      if (nit != nout.end()) {
        try {
          ok = within_tolerance(nit->second, expected_double(tx.expected));
        } catch (const cpmml::ParsingException&) {}
      }
    }

    if (!ok) {
      std::string items_str;
      for (const auto& i : tx.items) {
        if (!items_str.empty()) items_str += ",";
        items_str += i;
      }
      std::cerr << "predicted: " << result.as_string() << "  expected: " << tx.expected
                << "  items: [" << items_str << "]" << std::endl;
      return -1;
    }
  }
  return 0;
}

// ---- forecast mode ----------------------------------------------------------

static int run_forecast(cpmml::Model& model, CSVReader& reader, std::unordered_map<std::string, std::string> row) {
  std::vector<double> expected, expected_variance;
  std::unordered_map<std::string, std::vector<double>> regressors;

  // An optional "variance" column turns on predictive-variance checking; any
  // other column beside "forecast" is a regressor.
  const bool with_variance = row.count("variance") > 0;
  std::vector<std::string> reg_cols;
  for (const auto& kv : row)
    if (kv.first != "forecast" && kv.first != "variance") reg_cols.push_back(kv.first);

  while (!row.empty()) {
    expected.push_back(expected_double(row.at("forecast")));
    if (with_variance) expected_variance.push_back(expected_double(row.at("variance")));
    for (const auto& col : reg_cols) regressors[col].push_back(to_double(row.at(col)));
    row = reader.read();
  }

  if (expected.empty()) {
    std::cerr << "no forecast rows in CSV" << std::endl;
    return -1;
  }

  int horizon = static_cast<int>(expected.size());

  // Build evaluate() input
  cpmml::Input args;
  args["horizon"] = horizon;
  if (with_variance) args["_variance"] = std::string("true");
  for (const auto& [col, values] : regressors)
    args[col] = values;

  cpmml::Result result = model.evaluate(args);
  auto actual = result.series();

  if (with_variance) {
    const auto pairs = result.series_with_variance();
    if (pairs.size() != expected_variance.size()) {
      std::cerr << "variance size mismatch: got " << pairs.size() << "  expected " << expected_variance.size()
                << std::endl;
      return -1;
    }
    for (size_t i = 0; i < pairs.size(); i++) {
      if (!within_tolerance(pairs[i].second, expected_variance[i])) {
        std::cerr << "variance h=" << (i + 1) << "  predicted=" << pairs[i].second
                  << "  expected=" << expected_variance[i] << std::endl;
        return -1;
      }
    }
  }

  if (actual.size() != expected.size()) {
    std::cerr << "size mismatch: got " << actual.size() << "  expected " << expected.size() << std::endl;
    return -1;
  }

  for (size_t i = 0; i < expected.size(); i++) {
    if (!within_tolerance(actual[i], expected[i])) {
      std::cerr << "h=" << (i + 1) << "  predicted=" << actual[i] << "  expected=" << expected[i] << std::endl;
      return -1;
    }
  }
  return 0;
}

// ---- entry point ------------------------------------------------------------

// ---- dump mode --------------------------------------------------------------
//
// pmml_tester.exe model.zip inputs.csv --dump out.csv scores every row of
// inputs.csv (input columns only, "N/A" or an empty cell for a missing value)
// and writes, per row, the predicted value and every OutputField as text, in
// the layout of JPMML's EvaluationExample output: the target field's column,
// then the OutputFields, "N/A" for no value. tools/cross-validate-text.py
// compares that with JPMML's own output, text for text.
//
// A string result is written as cPMML gives it; a numeric OutputField
// (num_outputs) through cPMML's formatter as its dataType says: the
// OutputField's own dataType, else double for probability-like features and
// the target's dataType for the others.

static std::string csv_cell(const std::string& text) {
  if (text.find_first_of(",\"\n") == std::string::npos) return text;
  std::string quoted = "\"";
  for (const char c : text) quoted += c == '"' ? std::string("\"\"") : std::string(1, c);
  return quoted + "\"";
}

static java::Kind kind_of_type(const std::string& datatype) {
  if (datatype == "integer") return java::Kind::INTEGER;
  if (datatype == "float") return java::Kind::FLOAT;
  if (datatype == "boolean") return java::Kind::BOOLEAN;
  return java::Kind::DOUBLE;
}

// A time-series model is dumped the way run_forecast scores it: the forecast
// fixture itself is the input (one row per step, regressor columns beside
// "forecast", an optional "variance" column asking for the predictive
// variance), and the dump is a "forecast" column (and "variance") with one
// row per step, as doubles. Used by tools/fixture-dump-diff.sh; JPMML has no
// counterpart.
static int run_dump_forecast(cpmml::Model& model, const std::string& inputs, const std::string& out_path) {
  CSVReader reader(inputs);
  int horizon = 0;
  bool with_variance = false;
  std::unordered_map<std::string, std::vector<double>> regressors;
  for (auto row = reader.read(); !row.empty(); row = reader.read()) {
    horizon++;
    for (const auto& [k, v] : row) {
      if (k == "variance") with_variance = true;
      else if (k != "forecast") regressors[k].push_back(to_double(v));
    }
  }
  std::ofstream out(out_path);
  out << "forecast" << (with_variance ? ",variance" : "") << "\n";
  try {
    cpmml::Input args;
    args["horizon"] = horizon;
    if (with_variance) args["_variance"] = std::string("true");
    for (const auto& [col, values] : regressors) args[col] = values;
    const cpmml::Result result = model.evaluate(args);
    if (with_variance) {
      for (const auto& [f, v] : result.series_with_variance())
        out << java::text({f, java::Kind::DOUBLE}) << "," << java::text({v, java::Kind::DOUBLE}) << "\n";
    } else {
      for (const double f : result.series()) out << java::text({f, java::Kind::DOUBLE}) << "\n";
    }
  } catch (const std::exception& e) {
    out << csv_cell(std::string("ERROR: ") + e.what()) << "\n";
  }
  return 0;
}

static int run_dump(cpmml::Model& model, const std::string& model_zip, const std::string& inputs,
                    const std::string& out_path) {
  if (model.mining_function() == "TIMESERIES") return run_dump_forecast(model, inputs, out_path);
  const std::vector<char> document = read_file(model_zip, true);
  pugi::xml_document xml;
  if (!xml.load_string(document.data())) throw std::runtime_error("cannot parse " + model_zip);
  const pugi::xml_node pmml = xml.child("PMML");
  pugi::xml_node top;
  for (const auto& child : pmml.children())
    if (child.attribute("functionName")) {
      top = child;
      break;
    }
  std::unordered_map<std::string, std::string> types;
  for (const auto& field : pmml.child("DataDictionary").children("DataField"))
    types[field.attribute("name").value()] = field.attribute("dataType").value();
  std::string target;
  for (const auto& field : top.child("MiningSchema").children("MiningField")) {
    const std::string usage = field.attribute("usageType").value();
    if (usage == "target" || usage == "predicted") {
      target = field.attribute("name").value();
      break;
    }
  }
  // No target field (clustering, say): JPMML names the predicted column "".

  std::vector<std::string> columns{target};
  std::unordered_map<std::string, java::Kind> kinds;
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
  for (const auto& name : segment_output_names(model_zip))
    if (std::find(columns.begin(), columns.end(), name) == columns.end()) columns.push_back(name);

  std::ofstream out(out_path);
  for (size_t i = 0; i < columns.size(); i++) out << (i ? "," : "") << csv_cell(columns[i]);
  out << "\n";

  CSVReader reader(inputs);
  for (auto row = reader.read(); !row.empty(); row = reader.read()) {
    cpmml::Input input;
    for (const auto& [k, v] : row)
      if (!v.empty() && v != "N/A") input[k] = v;
    std::vector<std::string> cells;
    try {
      const cpmml::Result result = model.evaluate(input);
      const auto sout = result.str_outputs();
      const auto nout = result.num_outputs();
      for (const auto& column : columns) {
        if (column == target) {
          cells.push_back(result.empty() || result.as_string().empty() ? "N/A" : result.as_string());
          continue;
        }
        const auto s = sout.find(column);
        const auto n = nout.find(column);
        if (s != sout.end() && !s->second.empty())
          cells.push_back(s->second);
        else if (n != nout.end())
          cells.push_back(java::text({n->second, kinds.count(column) ? kinds[column] : java::Kind::DOUBLE}));
        else
          cells.push_back("N/A");
      }
    } catch (const std::exception& e) {
      cells.assign(columns.size(), std::string("ERROR: ") + e.what());
    }
    for (size_t i = 0; i < cells.size(); i++) out << (i ? "," : "") << csv_cell(cells[i]);
    out << "\n";
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: pmml_tester.exe model.zip fixture.csv [--locale <name> | --dump <out.csv>]\n";
    return -1;
  }

  if (argc >= 5 && std::string(argv[3]) == "--locale") {
    if (std::setlocale(LC_ALL, argv[4]) == nullptr) {
      std::cerr << "locale '" << argv[4] << "' unavailable on this host\n";
      return 77;  // ctest SKIP_RETURN_CODE
    }
    try {
      std::locale::global(std::locale(argv[4]));
    } catch (const std::exception&) {
      std::cerr << "locale '" << argv[4] << "' unavailable to std::locale\n";
      return 77;
    }
    std::cout << "running under locale " << argv[4] << "\n";
  }

  try {
    cpmml::Model model(argv[1], true);
    if (argc >= 5 && std::string(argv[3]) == "--dump") return run_dump(model, argv[1], argv[2], argv[4]);
    CSVReader reader(argv[2]);

    auto first = reader.read();
    if (first.empty()) {
      std::cerr << "empty CSV: " << argv[2] << "\n";
      return -1;
    }

    std::string mf = model.mining_function();

    if (mf == "TIMESERIES")
      return run_forecast(model, reader, std::move(first));
    else if (mf == "ASSOCIATION_RULES" && first.count("transaction"))
      return run_transactional(model, reader, std::move(first));
    else
      return run_score(model, reader, std::move(first), argv[1]);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }
}
