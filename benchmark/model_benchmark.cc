
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#include <chrono>
#include <iostream>
#include <unordered_map>

#include "cPMML.h"
#include "utils/csvreader.h"
#include "utils/utils.h"

int main(int argc, char** argv) {
  auto start = std::chrono::steady_clock::now();
  cpmml::Model model(argv[1]);
  auto end = std::chrono::steady_clock::now();
  double elapsed_load = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  CSVReader reader(argv[2]);
  double n_eval = 0;
  std::unordered_map<std::string, std::string> sample;
  double elapsed_eval = 0;
  while ((sample = reader.read()).size() > 0) {
    try {
      cpmml::Input input;
      for (const auto& [k, v] : sample) input[k] = v;

      start = std::chrono::steady_clock::now();
      model.evaluate(input);
      end = std::chrono::steady_clock::now();
      elapsed_eval += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      n_eval++;
    } catch (const cpmml::Exception& exception) {
    }
  }

  std::cout << "\tLoading time: " << std::setprecision(2) << std::fixed << elapsed_load / 1000 / 1000 << " ms";
  std::cout << "\tEvaluate time: " << std::setprecision(2) << std::fixed << elapsed_eval / n_eval / 1000 << " us";
  std::cout << "\tEvaluate TPS: " << format_int((int)1e9 / (elapsed_eval / n_eval)) << std::endl;

  return 0;
}
