
/*******************************************************************************
 * Unified test harness for all cPMML model types.
 *
 * Usage: pmml_tester.exe <model.zip> <fixture.csv>
 *
 * Mode is determined by model.mining_function():
 *   "TIMESERIES"         →  forecast mode
 *   "ASSOCIATION_RULES"  →  transactional or scoring mode (auto-detected)
 *   otherwise            →  scoring mode
 *******************************************************************************/

#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cPMML.h"
#include "utils/csvreader.h"
#include "utils/utils.h"

static const double REL_TOL = 0.001;
static const double ABS_TOL = 1e-9;

static bool within_tolerance(double actual, double expected) {
  double err = std::abs(actual - expected);
  double rel = (std::abs(expected) > ABS_TOL) ? (err / std::abs(expected)) : err;
  return rel <= REL_TOL;
}

// Convert CSV row (string map) to cpmml::Input
static cpmml::Input to_input(const std::unordered_map<std::string, std::string>& row) {
  cpmml::Input input;
  for (const auto& [k, v] : row) input[k] = v;
  return input;
}

// ---- scoring mode -----------------------------------------------------------

static int run_score(cpmml::Model& model, CSVReader& reader, std::unordered_map<std::string, std::string> row) {
  const std::string out_col = model.output_name();

  if (!row.count(out_col)) {
    std::cerr << "output column '" << out_col << "' not found in fixture CSV" << std::endl;
    return -1;
  }

  while (!row.empty()) {
    const std::string& expected = row.at(out_col);

    // Skip rows where the oracle expected value is absent (null prediction)
    if (expected.empty()) {
      row = reader.read();
      continue;
    }

    cpmml::Result result = model.evaluate(to_input(row));

    bool ok = (result.as_string() == expected);
    if (!ok && !is_double_min(result.as_double())) {
      try {
        ok = within_tolerance(result.as_double(), to_double(expected));
      } catch (const cpmml::ParsingException&) {}
    }

    // Also check named output fields (handles probability/confidence output columns)
    if (!ok) {
      const auto nout = result.num_outputs();
      auto nit = nout.find(out_col);
      if (nit != nout.end()) {
        try {
          ok = within_tolerance(nit->second, to_double(expected));
        } catch (const cpmml::ParsingException&) {}
      }
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
          ok = within_tolerance(nit->second, to_double(tx.expected));
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
  std::vector<double> expected;
  std::unordered_map<std::string, std::vector<double>> regressors;

  // Any column other than "forecast" is treated as a regressor
  std::vector<std::string> reg_cols;
  for (const auto& kv : row)
    if (kv.first != "forecast") reg_cols.push_back(kv.first);

  while (!row.empty()) {
    expected.push_back(to_double(row.at("forecast")));
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
  for (const auto& [col, values] : regressors)
    args[col] = values;

  cpmml::Result result = model.evaluate(args);
  auto actual = result.series();

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

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: pmml_tester.exe model.zip fixture.csv\n";
    return -1;
  }

  try {
    cpmml::Model model(argv[1], true);
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
      return run_score(model, reader, std::move(first));
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }
}
