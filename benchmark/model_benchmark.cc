
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

// model_benchmark.exe [--zip] [--read-all] [--repeat N] <model> <dataset.csv>
//
// Loads <model> once and scores every row of <dataset.csv>, N times over
// (default 1), timing each Model::evaluate call. Prints one line:
//
//   Loading time: <ms> ms  Evaluate time: <us> us  Evaluate TPS: <n>
//   Records: <calls timed>  Errors: <calls that threw>
//
// "Evaluate time" is the mean over the calls that return; a call that throws a
// cpmml::Exception is counted under Errors and not timed, as before these
// options existed, so the numbers stay comparable with the baseline of
// docs/design/typed-core.md §6.1 (AuditRandomForest has 101 such rows of 2000).
//
//   --zip       the model file is a zip archive holding one PMML document (the
//               test fixtures, test/data/model/<name>.zip)
//   --read-all  also call as_string, distribution, num_outputs and str_outputs
//               on each result inside the timed region, so that work deferred
//               from evaluate to the accessors is still counted
//   --repeat N  score the whole dataset N times (the fixtures have 6-118 rows)
//
// Every row is read and converted to a cpmml::Input before the clock starts;
// an empty CSV cell is passed as an empty string, as pmml_tester does. The
// tools/bench-ab.sh gate parses this line; keep its format.

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cPMML.h"
#include "utils/csvreader.h"
#include "utils/utils.h"

namespace {

[[noreturn]] void usage(const std::string& error) {
  std::cerr << "model_benchmark.exe: " << error << "\n"
            << "usage: model_benchmark.exe [--zip] [--read-all] [--repeat N] <model> <dataset.csv>" << std::endl;
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  bool zipped = false;
  bool read_all = false;
  long repeat = 1;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "--zip") {
      zipped = true;
    } else if (arg == "--read-all") {
      read_all = true;
    } else if (arg == "--repeat") {
      if (i + 1 >= argc) usage("--repeat needs a value");
      const std::string value = argv[++i];
      char* end = nullptr;
      errno = 0;
      repeat = std::strtol(value.c_str(), &end, 10);
      if (value.empty() || *end != '\0' || errno == ERANGE || repeat < 1)
        usage("--repeat needs a positive integer that fits a long, got '" + value + "'");
    } else if (arg.rfind("--", 0) == 0) {
      usage("unknown option " + arg);
    } else {
      positional.push_back(arg);
    }
  }
  if (positional.size() != 2) usage("expected a model and a dataset");

  auto start = std::chrono::steady_clock::now();
  cpmml::Model model(positional[0], zipped);
  auto end = std::chrono::steady_clock::now();
  const double elapsed_load = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::vector<cpmml::Input> inputs;
  CSVReader reader(positional[1]);
  for (auto sample = reader.read(); !sample.empty(); sample = reader.read()) {
    cpmml::Input input;
    for (const auto& [k, v] : sample) input[k] = v;
    inputs.push_back(std::move(input));
  }
  if (inputs.empty()) usage("no rows in " + positional[1]);

  // The accessors' results are folded into a volatile so the optimiser cannot
  // drop the calls.
  volatile std::size_t sink = 0;
  double elapsed_eval = 0;
  long n_eval = 0;
  long n_errors = 0;
  for (long r = 0; r < repeat; r++) {
    for (const auto& input : inputs) {
      try {
        start = std::chrono::steady_clock::now();
        const cpmml::Result result = model.evaluate(input);
        if (read_all) {
          sink = sink + result.as_string().size() + result.distribution().size() + result.num_outputs().size() +
                 result.str_outputs().size();
        }
        end = std::chrono::steady_clock::now();
        elapsed_eval += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        n_eval++;
      } catch (const cpmml::Exception&) {
        n_errors++;
      }
    }
  }

  if (n_eval == 0) {
    std::cerr << "model_benchmark.exe: every call threw" << std::endl;
    return 1;
  }
  std::cout << "\tLoading time: " << std::setprecision(3) << std::fixed << elapsed_load / 1000 / 1000 << " ms";
  std::cout << "\tEvaluate time: " << std::setprecision(4) << std::fixed << elapsed_eval / n_eval / 1000 << " us";
  std::cout << "\tEvaluate TPS: " << format_int((int)(1e9 / (elapsed_eval / n_eval)));
  std::cout << "\tRecords: " << n_eval << "\tErrors: " << n_errors << std::endl;

  return 0;
}
