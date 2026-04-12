
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#include <iostream>
#include <unordered_map>

#include "cPMML.h"

int main() {
  std::string model_filepath = "../benchmark/data/model/IrisTree.xml";

  cpmml::Model model(model_filepath);
  cpmml::Input sample = {
      {"sepal_length", "6.6"},
      {"sepal_width", "2.9"},
      {"petal_length", "4.6"},
      {"petal_width", "1.3"}
  };

  for (const auto &probability : model.evaluate(sample).distribution())
    std::cout << probability.first << ": " << probability.second << std::endl;

  return 0;
}
