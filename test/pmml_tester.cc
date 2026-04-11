
/*******************************************************************************
 * Unified test harness for all cPMML model types.
 *
 * Usage: pmml_tester.exe <model.zip> <fixture.csv>
 *
 * CSV mode is auto-detected from column headers:
 *   "forecast" column present    →  forecast mode: model.forecast(n[, regressors])
 *   otherwise                    →  scoring mode: model.score() per row, primary
 *                                   output column matched via model.output_name()
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

// ---- scoring mode -----------------------------------------------------------

static int run_score(cpmml::Model& model, CSVReader& reader, std::unordered_map<std::string, std::string> row) {
  const std::string out_col = model.output_name();
  while (!row.empty()) {
    cpmml::Prediction pred = model.score(row);
    const std::string& expected = row.at(out_col);

    bool ok = (pred.as_string() == expected) ||
              (pred.as_double() != double_min() && within_tolerance(pred.as_double(), to_double(expected)));

    if (!ok) {
      std::cerr << "predicted: " << pred.as_string() << "  expected: " << expected << "  sample: " << to_string(row)
                << std::endl;
      return -1;
    }
    row = reader.read();
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
  std::vector<double> actual = reg_cols.empty() ? model.forecast(horizon) : model.forecast(horizon, regressors);

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

  cpmml::Model model(argv[1], true);
  CSVReader reader(argv[2]);

  auto first = reader.read();
  if (first.empty()) {
    std::cerr << "empty CSV: " << argv[2] << "\n";
    return -1;
  }

  if (first.count("forecast"))
    return run_forecast(model, reader, std::move(first));
  else
    return run_score(model, reader, std::move(first));
}
