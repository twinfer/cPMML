
/*******************************************************************************
 * Guards that the fixture suite cannot express.
 *
 * Every other test here loads a model and compares scores, which only exercises
 * paths a well-formed document reaches. These are the opposite: default-
 * constructed API objects, empty containers, and a document that is malformed in
 * a way the parser used to accept and then crash on.
 *
 * Each assertion below fails by segfault, hang or stack overflow on the code it
 * was written against, not by returning a wrong value.
 *******************************************************************************/

#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "cPMML.h"
#include "utils/utils.h"

static int failures = 0;

#define CHECK(cond)                                                                  \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << #cond << "\n";  \
      ++failures;                                                                    \
    }                                                                                \
  } while (0)

// A default-constructed Result holds no score. Every accessor has to answer for
// that, not just empty(): declaring a Result before a branch assigns it is the
// first thing a caller writes.
static void default_result_is_answerable() {
  cpmml::Result result;

  CHECK(result.empty());
  CHECK(result.as_string().empty());
  CHECK(is_double_min(result.as_double()));
  CHECK(result.distribution().empty());
  CHECK(result.num_outputs().empty());
  CHECK(result.str_outputs().empty());
  CHECK(result.series().empty());
  CHECK(result.series_with_variance().empty());
}

// A default-constructed Model has nothing to score with, and unlike Result there
// is no sensible empty answer -- so it must throw rather than dereference null.
static void default_model_throws() {
  cpmml::Model model;

  const auto throws = [](auto&& call) {
    try {
      call();
    } catch (const cpmml::InvalidValueException&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  };

  CHECK(throws([&] { return model.evaluate(cpmml::Input{}); }));
  CHECK(throws([&] { return model.validate(cpmml::Input{}); }));
  CHECK(throws([&] { return model.output_name(); }));
  CHECK(throws([&] { return model.mining_function(); }));
}

// mkstring opened by advancing a lookahead iterator past cbegin(), which has
// nowhere to go when the container is empty.
static void mkstring_handles_empty() {
  CHECK(mkstring(std::vector<int>{}) == "[]");
  CHECK(mkstring(std::set<std::string>{}) == "[]");
  CHECK(mkstring(std::vector<std::pair<std::string, std::string>>{}) == "[]");

  // Non-empty output must not have changed shape while fixing that.
  CHECK(mkstring(std::vector<int>{1}) == "[1]");
  CHECK(mkstring(std::vector<int>{1, 2, 3}) == "[1,2,3]");
  CHECK(mkstring(std::vector<int>{1, 2}, ";") == "[1;2]");
}

// A DerivedField that reaches itself through its own inputs. DagBuilder marked a
// field visited only after resolving it, so this recursed until the stack ran
// out -- at load time, on a document the parser otherwise accepts.
static const char* CYCLIC_PMML = R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">
  <DataDictionary numberOfFields="1">
    <DataField name="x" optype="continuous" dataType="double"/>
  </DataDictionary>
  <TransformationDictionary>
    <DerivedField name="a" optype="continuous" dataType="double">
      <Apply function="+"><FieldRef field="b"/><Constant>1</Constant></Apply>
    </DerivedField>
    <DerivedField name="b" optype="continuous" dataType="double">
      <Apply function="+"><FieldRef field="a"/><Constant>1</Constant></Apply>
    </DerivedField>
  </TransformationDictionary>
  <RegressionModel functionName="regression">
    <MiningSchema>
      <MiningField name="x" usageType="active"/>
      <MiningField name="a" usageType="active"/>
    </MiningSchema>
    <RegressionTable intercept="0.0">
      <NumericPredictor name="x" coefficient="1.0"/>
    </RegressionTable>
  </RegressionModel>
</PMML>
)";

static void cyclic_derived_fields_are_rejected() {
  const std::string path = "cyclic_derivedfield.pmml";
  {
    FILE* out = std::fopen(path.c_str(), "w");
    if (out == nullptr) {
      std::cerr << "FAIL could not write " << path << "\n";
      ++failures;
      return;
    }
    std::fputs(CYCLIC_PMML, out);
    std::fclose(out);
  }

  bool rejected = false;
  try {
    cpmml::Model model(path, false);
  } catch (const cpmml::Exception&) {
    rejected = true;  // any library exception beats recursing off the stack
  }
  CHECK(rejected);

  std::remove(path.c_str());
}

int main() {
  default_result_is_answerable();
  default_model_throws();
  mkstring_handles_empty();
  cyclic_derived_fields_are_rejected();

  if (failures > 0) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "api_guards: all checks passed\n";
  return 0;
}
