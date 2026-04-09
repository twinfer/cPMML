
/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

// Usage: timeseries_tester.exe <model.zip> <expected_forecasts.csv>
//
// CSV format (single column, header "forecast"):
//   forecast
//   105.0
//   110.0
//   ...
//
// Calls model.forecast(n_rows) and checks each value against expected
// with 0.1% relative tolerance (or 1e-9 absolute tolerance for near-zero).

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "cPMML.h"
#include "utils/csvreader.h"
#include "utils/utils.h"

static const double REL_TOL = 0.001;
static const double ABS_TOL = 1e-9;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: timeseries_tester.exe model.zip forecast.csv\n";
    return -1;
  }

  cpmml::Model model(argv[1], true);

  // Read expected forecast values from CSV (column "forecast")
  CSVReader reader(argv[2]);
  std::vector<double> expected;
  std::unordered_map<std::string, std::string> row;
  while ((row = reader.read()).size() > 0)
    expected.push_back(to_double(row.at("forecast")));

  if (expected.empty()) {
    std::cerr << "no forecast rows in " << argv[2] << "\n";
    return -1;
  }

  std::vector<double> actual = model.forecast(static_cast<int>(expected.size()));

  if (actual.size() != expected.size()) {
    std::cerr << "size mismatch: got " << actual.size()
              << " expected " << expected.size() << "\n";
    return -1;
  }

  for (size_t i = 0; i < expected.size(); ++i) {
    double exp = expected[i], act = actual[i];
    double err = std::abs(act - exp);
    double rel = (std::abs(exp) > ABS_TOL) ? (err / std::abs(exp)) : err;
    if (rel > REL_TOL) {
      std::cerr << "h=" << (i + 1)
                << "  predicted=" << act
                << "  expected=" << exp
                << "  rel_err=" << rel << "\n";
      return -1;
    }
  }

  return 0;
}
