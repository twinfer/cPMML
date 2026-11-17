/*******************************************************************************
 * The enum reader (src/core/enum_parser.h): every enumerated attribute is read
 * through one table, exactly (PMML enumerations are case-sensitive), with no
 * fallback for a value outside it.
 *
 *   1. Spellings the old reader case-folded. JPMML 1.7.7, probed with
 *      pmml-evaluator-example-executable-1.7.7.jar, refuses each of them
 *      ("Required attribute DataField@dataType is not defined", and so on).
 *   2. Values outside the enumeration of an optional attribute. JPMML reads
 *      them as absent and takes the default (normalizationMethod="bogus"
 *      scores as none); the spec makes them invalid, and so does cPMML.
 *   3. Values the spec defines and cPMML does not implement: a load error
 *      that says "unsupported", not "invalid".
 *   4. The error names the attribute, the value and the allowed values.
 *   5. Conforming spellings that the old reader did not have, or got wrong.
 *
 * Uses the public API only, so it builds against an older library too. Exit
 * status is the number of failed checks.
 *******************************************************************************/

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "cPMML.h"

static int failures = 0;

#define CHECK(cond)                                                                  \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << #cond << "\n";  \
      ++failures;                                                                    \
    }                                                                                \
  } while (0)

static int file_counter = 0;

// The load error a document gives, or "" when it loads.
static std::string load_error(const std::string& document) {
  const std::string path = "enum_reader_" + std::to_string(file_counter++) + ".pmml";
  {
    std::ofstream out(path);
    out << document;
  }
  std::string error;
  try {
    cpmml::Model model(path, false);
  } catch (const cpmml::ParsingException& e) {
    error = e.what();
    if (error.empty()) error = "(empty message)";
  } catch (const std::exception& e) {
    error = std::string("unexpected exception: ") + e.what();
  }
  std::remove(path.c_str());
  return error;
}

static bool loads(const std::string& document) { return load_error(document).empty(); }

static bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

static std::string replaced(std::string text, const std::string& from, const std::string& to) {
  const size_t at = text.find(from);
  if (at == std::string::npos) {
    std::cerr << "FAIL template has no \"" << from << "\"\n";
    ++failures;
    return text;
  }
  return text.replace(at, from.size(), to);
}

static const std::string REGRESSION = R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="x" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="double"/>
  </DataDictionary>
  <RegressionModel functionName="regression" normalizationMethod="none">
    <MiningSchema>
      <MiningField name="x" invalidValueTreatment="returnInvalid" missingValueTreatment="asIs" usageType="active"/>
      <MiningField name="y" usageType="target"/>
    </MiningSchema>
    <Output>
      <OutputField name="out" feature="predictedValue"/>
    </Output>
    <RegressionTable intercept="1">
      <NumericPredictor name="x" coefficient="2"/>
    </RegressionTable>
  </RegressionModel>
</PMML>)";

static const std::string CLUSTERING = R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary><DataField name="x" optype="continuous" dataType="double"/></DataDictionary>
  <ClusteringModel functionName="clustering" modelClass="centerBased" numberOfClusters="2">
    <MiningSchema><MiningField name="x"/></MiningSchema>
    <ComparisonMeasure kind="distance" compareFunction="absDiff"><euclidean/></ComparisonMeasure>
    <ClusteringField field="x" compareFunction="absDiff"/>
    <Cluster id="a"><Array n="1" type="real">0</Array></Cluster>
    <Cluster id="b"><Array n="1" type="real">10</Array></Cluster>
  </ClusteringModel>
</PMML>)";

static const std::string ENSEMBLE = R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="x" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="double"/>
  </DataDictionary>
  <MiningModel functionName="regression">
    <MiningSchema><MiningField name="x"/><MiningField name="y" usageType="target"/></MiningSchema>
    <Segmentation multipleModelMethod="average" missingPredictionTreatment="continue">
      <Segment id="1"><True/>
        <TreeModel functionName="regression" missingValueStrategy="none" noTrueChildStrategy="returnNullPrediction">
          <MiningSchema><MiningField name="x"/><MiningField name="y" usageType="target"/></MiningSchema>
          <Node score="1"><True/>
            <Node score="2"><SimplePredicate field="x" operator="lessThan" value="5"/></Node>
            <Node score="3"><CompoundPredicate booleanOperator="or">
              <SimplePredicate field="x" operator="greaterOrEqual" value="5"/>
              <SimplePredicate field="x" operator="isMissing"/>
            </CompoundPredicate></Node>
          </Node>
        </TreeModel>
      </Segment>
    </Segmentation>
  </MiningModel>
</PMML>)";

static const std::string GRM = R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="x" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="double"/>
  </DataDictionary>
  <GeneralRegressionModel functionName="regression" modelType="generalizedLinear" linkFunction="log">
    <MiningSchema><MiningField name="x"/><MiningField name="y" usageType="target"/></MiningSchema>
    <ParameterList><Parameter name="p0"/><Parameter name="p1"/></ParameterList>
    <CovariateList><Predictor name="x"/></CovariateList>
    <PPMatrix><PPCell value="1" predictorName="x" parameterName="p1"/></PPMatrix>
    <ParamMatrix><PCell parameterName="p0" beta="0.5"/><PCell parameterName="p1" beta="0.1"/></ParamMatrix>
  </GeneralRegressionModel>
</PMML>)";

// ---- 0. the templates load as written --------------------------------------

static void templates_load() {
  CHECK(load_error(REGRESSION) == "");
  CHECK(load_error(CLUSTERING) == "");
  CHECK(load_error(ENSEMBLE) == "");
  CHECK(load_error(GRM) == "");
}

// ---- 1. case-folded spellings: JPMML refuses each --------------------------

static void case_sensitive() {
  // JPMML: Required attribute DataField@dataType is not defined.
  CHECK(!loads(replaced(REGRESSION, R"(dataType="double"/>)", R"(dataType="Double"/>)")));
  // JPMML: Required attribute DataField@optype is not defined.
  CHECK(!loads(replaced(REGRESSION, R"(optype="continuous")", R"(optype="Continuous")")));
  // JPMML: Required attribute RegressionModel@functionName is not defined.
  CHECK(!loads(replaced(REGRESSION, R"(functionName="regression")", R"(functionName="Regression")")));
  // JPMML: Required attribute Segmentation@multipleModelMethod is not defined.
  CHECK(!loads(replaced(ENSEMBLE, R"(multipleModelMethod="average")", R"(multipleModelMethod="Average")")));
  // JPMML: Required element ComparisonMeasure/<Measure> is not defined.
  CHECK(!loads(replaced(CLUSTERING, "<euclidean/>", "<Euclidean/>")));
  // JPMML: Required attribute GeneralRegressionModel@modelType is not defined.
  CHECK(!loads(replaced(GRM, R"(modelType="generalizedLinear")", R"(modelType="GeneralizedLinear")")));
  // JPMML: Required attribute GeneralRegressionModel@linkFunction is not defined.
  CHECK(!loads(replaced(GRM, R"(linkFunction="log")", R"(linkFunction="Log")")));
  // Operators of the predicates; the old reader lower-cased these too.
  CHECK(!loads(replaced(ENSEMBLE, R"(operator="lessThan")", R"(operator="lessthan")")));
  CHECK(!loads(replaced(ENSEMBLE, R"(booleanOperator="or")", R"(booleanOperator="OR")")));
  CHECK(!loads(replaced(REGRESSION, R"(invalidValueTreatment="returnInvalid")", R"(invalidValueTreatment="asis")")));
  CHECK(!loads(replaced(REGRESSION, R"(missingValueTreatment="asIs")", R"(missingValueTreatment="AsIs")")));
}

// ---- 2. unknown values of optional attributes: JPMML takes the default -----

static void no_fallback() {
  // JPMML 1.7.7 scores each of these as if the attribute were absent.
  CHECK(!loads(replaced(REGRESSION, R"(normalizationMethod="none")", R"(normalizationMethod="bogus")")));
  CHECK(!loads(replaced(REGRESSION, R"(normalizationMethod="none")", R"(normalizationMethod="SoftMax")")));
  CHECK(!loads(replaced(REGRESSION, R"(feature="predictedValue")", R"(feature="bogus")")));
  CHECK(!loads(replaced(REGRESSION, R"(feature="predictedValue")", R"(feature="PredictedValue")")));
  // Not PMML spellings (the spec has frequencyWeight); the old table had them.
  CHECK(!loads(replaced(REGRESSION, R"(usageType="active")", R"(usageType="frequency_weight")")));
  CHECK(!loads(replaced(CLUSTERING, R"(kind="distance")", R"(kind="Distance")")));
  CHECK(!loads(replaced(CLUSTERING, R"(<ComparisonMeasure kind="distance" compareFunction="absDiff">)",
                        R"(<ComparisonMeasure kind="distance" compareFunction="bogus">)")));
  CHECK(!loads(replaced(CLUSTERING, R"(<ClusteringField field="x" compareFunction="absDiff"/>)",
                        R"(<ClusteringField field="x" compareFunction="AbsDiff"/>)")));
  CHECK(!loads(replaced(ENSEMBLE, R"(missingPredictionTreatment="continue")",
                        R"(missingPredictionTreatment="ReturnMissing")")));
  CHECK(!loads(replaced(ENSEMBLE, R"(missingValueStrategy="none")", R"(missingValueStrategy="DefaultChild")")));
  // A missing ComparisonMeasure used to mean euclidean.
  std::string no_measure = CLUSTERING;
  const size_t from = no_measure.find("<ComparisonMeasure");
  no_measure.erase(from, no_measure.find("</ComparisonMeasure>") + 20 - from);
  CHECK(!loads(no_measure));
}

// ---- 3. valid in PMML, not implemented in cPMML -----------------------------

static void unsupported() {
  // JPMML: Attribute with value ClusteringField@compareFunction=table is not supported.
  std::string error = load_error(replaced(CLUSTERING, R"(<ClusteringField field="x" compareFunction="absDiff"/>)",
                                          R"(<ClusteringField field="x" compareFunction="table"/>)"));
  CHECK(contains(error, "unsupported value \"table\" for ClusteringField@compareFunction"));
  CHECK(contains(error, "absDiff, gaussSim, delta, equal"));
  // JPMML: Attribute with value OutputField@feature=standardError is not supported.
  // feature="standardError" used to answer with the predicted value.
  error = load_error(replaced(REGRESSION, R"(feature="predictedValue")", R"(feature="standardError")"));
  CHECK(contains(error, "unsupported value \"standardError\" for OutputField@feature"));
  CHECK(contains(load_error(replaced(REGRESSION, R"(feature="predictedValue")", R"(feature="affinity")")),
                 "unsupported"));
  // CoxRegression used to score as a linear model.
  CHECK(contains(load_error(replaced(GRM, R"(modelType="generalizedLinear")", R"(modelType="CoxRegression")")),
                 "unsupported value \"CoxRegression\" for GeneralRegressionModel@modelType"));
  CHECK(contains(load_error(replaced(REGRESSION, R"(missingValueTreatment="asIs")",
                                     R"(missingValueTreatment="returnInvalid")")),
                 "unsupported value \"returnInvalid\" for MiningField@missingValueTreatment"));
  CHECK(contains(load_error(replaced(ENSEMBLE, R"(missingValueStrategy="none")",
                                     R"(missingValueStrategy="weightedConfidence")")),
                 "unsupported value \"weightedConfidence\" for TreeModel@missingValueStrategy"));
  CHECK(contains(load_error(replaced(REGRESSION, R"(functionName="regression")", R"(functionName="sequences")")),
                 "unsupported value \"sequences\" for RegressionModel@functionName"));
}

// ---- 4. the message ---------------------------------------------------------

static void message() {
  const std::string error =
      load_error(replaced(ENSEMBLE, R"(multipleModelMethod="average")", R"(multipleModelMethod="Average")"));
  CHECK(contains(error, "invalid value \"Average\" for Segmentation@multipleModelMethod"));
  CHECK(contains(error, "allowed values are majorityVote, weightedMajorityVote, average, weightedAverage, median, "
                        "weightedMedian, max, sum, weightedSum, selectFirst, selectAll, modelChain"));

  const std::string measure = load_error(replaced(CLUSTERING, "<euclidean/>", "<Euclidean/>"));
  CHECK(contains(measure, "invalid value \"Euclidean\" for measure element in ComparisonMeasure"));

  std::string required = REGRESSION;
  required = replaced(required, R"(<DataField name="x" optype="continuous" dataType="double"/>)",
                      R"(<DataField name="x" optype="continuous"/>)");
  CHECK(contains(load_error(required), "required attribute DataField@dataType is missing"));

  CHECK(contains(load_error(replaced(ENSEMBLE, R"(booleanOperator="or")", R"(booleanOperator="isIn")")),
                 "invalid value \"isIn\" for CompoundPredicate@booleanOperator; allowed values are or, and, xor, "
                 "surrogate"));
}

// ---- 5. conforming spellings ------------------------------------------------

static void conforming() {
  CHECK(loads(replaced(REGRESSION, R"(usageType="active")", R"(usageType="frequencyWeight")")));
  CHECK(loads(replaced(REGRESSION, R"(dataType="double"/>)", R"(dataType="dateTime"/>)")));
  CHECK(loads(replaced(CLUSTERING, "<euclidean/>", "<squaredEuclidean/>")));
  CHECK(loads(replaced(CLUSTERING, R"(kind="distance")", R"(kind="similarity")")));
  CHECK(loads(replaced(ENSEMBLE, R"(missingPredictionTreatment="continue")",
                       R"(missingPredictionTreatment="returnMissing")")));
}

int main() {
  templates_load();
  case_sensitive();
  no_fallback();
  unsupported();
  message();
  conforming();

  if (failures) std::cerr << failures << " check(s) failed\n";
  return failures;
}
