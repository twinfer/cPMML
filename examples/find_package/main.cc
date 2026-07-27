#include <iostream>

#include "cPMML.h"

int main() {
  std::cout << "cPMML " << cpmml::version << "\n";

  // A default-constructed Model holds no document, so this must throw -- and
  // the catch must match, which it only does if the exception's typeinfo is
  // exported from the shared library.
  try {
    cpmml::Model model;
    model.output_name();
  } catch (const cpmml::InvalidValueException& exception) {
    std::cout << "caught by type across the library boundary: " << exception.what() << "\n";
    return 0;
  }

  std::cerr << "expected cpmml::InvalidValueException\n";
  return 1;
}
