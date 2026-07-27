
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
#include "utils/csvreader.h"
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

// A BaselineModel test statistic that cPMML cannot compute must be refused at
// load. Every one of these needs an observed distribution aggregated over a
// window of records; cPMML scores one record at a time, and what it used to do
// with "chiSquare" -- difference a category's encoded value against that
// category's count -- is not a quantity at all. Anything it failed to recognise,
// CUSUM and scalarProduct included, silently became a logProb the spec does not
// define.
static const char* BASELINE_PMML_TEMPLATE = R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">
  <DataDictionary numberOfFields="2">
    <DataField name="v" optype="continuous" dataType="double"/>
    <DataField name="s" optype="continuous" dataType="double"/>
  </DataDictionary>
  <BaselineModel functionName="regression">
    <MiningSchema>
      <MiningField name="v" usageType="active"/>
      <MiningField name="s" usageType="predicted"/>
    </MiningSchema>
    <TestDistributions field="v" testStatistic="@STAT@">
      <Baseline>@BASELINE@</Baseline>
    </TestDistributions>
  </BaselineModel>
</PMML>
)";

static bool baseline_loads(const std::string& statistic, const std::string& baseline) {
  std::string pmml = BASELINE_PMML_TEMPLATE;
  pmml.replace(pmml.find("@STAT@"), 6, statistic);
  pmml.replace(pmml.find("@BASELINE@"), 10, baseline);

  const std::string path = "baseline_guard.pmml";
  FILE* out = std::fopen(path.c_str(), "w");
  if (out == nullptr) {
    std::cerr << "FAIL could not write " << path << "\n";
    ++failures;
    return false;
  }
  std::fputs(pmml.c_str(), out);
  std::fclose(out);

  bool loaded = false;
  try {
    cpmml::Model model(path, false);
    loaded = true;
  } catch (const cpmml::Exception&) {
  }

  std::remove(path.c_str());
  return loaded;
}

static void unsupported_baseline_statistics_are_rejected() {
  const std::string gaussian = R"(<GaussianDistribution mean="35.0" variance="100.0"/>)";
  const std::string counts =
      R"(<CountTable><FieldValue field="v" value="1" count="10"/></CountTable>)";

  CHECK(baseline_loads("zValue", gaussian));
  CHECK(baseline_loads("zScore", gaussian));  // tolerated alias

  CHECK(!baseline_loads("chiSquareIndependence", counts));
  CHECK(!baseline_loads("chiSquareDistribution", counts));
  CHECK(!baseline_loads("chiSquare", counts));
  CHECK(!baseline_loads("CUSUM", gaussian));
  CHECK(!baseline_loads("scalarProduct", counts));
  CHECK(!baseline_loads("notAStatistic", gaussian));

  // zValue needs a mean and a variance, which a count table does not supply,
  // and sqrt(0) would make the division that follows 1/0.
  CHECK(!baseline_loads("zValue", counts));
  CHECK(!baseline_loads("zValue", R"(<GaussianDistribution mean="1.0" variance="0.0"/>)"));
}

// getline into a fixed 8192-byte buffer sets failbit on a longer line, so one
// over-long row did not merely truncate itself: it wedged the stream and every
// row after it read as end-of-file, ending a fixture early and silently.
static void csvreader_reads_past_a_long_row() {
  const std::string path = "long_row.csv";
  {
    FILE* out = std::fopen(path.c_str(), "w");
    if (out == nullptr) {
      std::cerr << "FAIL could not write " << path << "\n";
      ++failures;
      return;
    }
    std::fputs("a,b\n", out);
    std::fputs((std::string(20000, 'x') + ",1\n").c_str(), out);
    std::fputs("short,2\n", out);
    std::fputs("\"say \"\"hi\"\"\",3\n", out);  // "" is one literal quote
    std::fclose(out);
  }

  CSVReader reader(path);
  const auto first = reader.read();
  const auto second = reader.read();
  const auto third = reader.read();

  CHECK(first.count("a") && first.at("a").size() == 20000);
  CHECK(first.count("b") && first.at("b") == "1");
  CHECK(second.count("a") && second.at("a") == "short");  // empty before the fix
  CHECK(third.count("a") && third.at("a") == "say \"hi\"");
  CHECK(third.count("b") && third.at("b") == "3");

  std::remove(path.c_str());
}

// MeanClusterDistances is "required when the algorithm type is clusterMeanDist"
// and its array length must match the cluster count -- an off-by-one there
// silently divides by the wrong cluster's mean, which is the same class of
// error as not dividing at all.
static const char* ANOMALY_PMML_TEMPLATE = R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4">
  <DataDictionary numberOfFields="2">
    <DataField name="a" optype="continuous" dataType="double"/>
    <DataField name="s" optype="continuous" dataType="double"/>
  </DataDictionary>
  <AnomalyDetectionModel functionName="regression" algorithmType="@ALGO@">
    <MiningSchema>
      <MiningField name="a" usageType="active"/>
    </MiningSchema>
    <ClusteringModel functionName="clustering" modelClass="centerBased" numberOfClusters="2">
      <MiningSchema>
        <MiningField name="a" usageType="active"/>
        <MiningField name="s" usageType="predicted"/>
      </MiningSchema>
      <ComparisonMeasure kind="distance"><euclidean/></ComparisonMeasure>
      <ClusteringField field="a"/>
      <Cluster name="1"><Array n="1" type="real">0.0</Array></Cluster>
      <Cluster name="2"><Array n="1" type="real">1.0</Array></Cluster>
    </ClusteringModel>@MCD@
  </AnomalyDetectionModel>
</PMML>
)";

static bool anomaly_loads(const std::string& algo, const std::string& mcd) {
  std::string pmml = ANOMALY_PMML_TEMPLATE;
  pmml.replace(pmml.find("@ALGO@"), 6, algo);
  pmml.replace(pmml.find("@MCD@"), 5, mcd);

  const std::string path = "anomaly_guard.pmml";
  FILE* out = std::fopen(path.c_str(), "w");
  if (out == nullptr) {
    std::cerr << "FAIL could not write " << path << "\n";
    ++failures;
    return false;
  }
  std::fputs(pmml.c_str(), out);
  std::fclose(out);

  bool loaded = false;
  try {
    cpmml::Model model(path, false);
    loaded = true;
  } catch (const cpmml::Exception&) {
  }

  std::remove(path.c_str());
  return loaded;
}

static void mean_cluster_distances_is_validated() {
  const std::string two = R"(
    <MeanClusterDistances><Array n="2" type="real">0.5 0.25</Array></MeanClusterDistances>)";
  const std::string three = R"(
    <MeanClusterDistances><Array n="3" type="real">0.5 0.25 0.1</Array></MeanClusterDistances>)";
  const std::string negative = R"(
    <MeanClusterDistances><Array n="2" type="real">0.5 -0.25</Array></MeanClusterDistances>)";

  CHECK(anomaly_loads("clusterMeanDist", two));

  CHECK(!anomaly_loads("clusterMeanDist", ""));        // required for this algorithm
  CHECK(!anomaly_loads("clusterMeanDist", three));     // one entry per cluster
  CHECK(!anomaly_loads("clusterMeanDist", negative));  // "an array of non-negative real values"

  // Not "iforest" used to mean "score the inner model raw", so clusterMeanDist,
  // an unknown value and an absent attribute were one branch.
  CHECK(!anomaly_loads("other", ""));
  CHECK(!anomaly_loads("notAnAlgorithm", ""));
  CHECK(!anomaly_loads("ocsvm", two));  // MeanClusterDistances belongs to clusterMeanDist alone
}

int main() {
  default_result_is_answerable();
  default_model_throws();
  mkstring_handles_empty();
  cyclic_derived_fields_are_rejected();
  unsupported_baseline_statistics_are_rejected();
  mean_cluster_distances_is_validated();
  csvreader_reads_past_a_long_row();

  if (failures > 0) {
    std::cerr << failures << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "api_guards: all checks passed\n";
  return 0;
}
