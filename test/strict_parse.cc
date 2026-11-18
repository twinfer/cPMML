/*******************************************************************************
 * The strict lexical reader (src/utils/lexical.h) and what is built on it.
 *
 *   1. The reader itself: XSD double/integer/boolean, whole text, XML
 *      whitespace only, locale-independent.
 *   2. XmlNode's typed getters: a malformed attribute is a load error, an
 *      absent one takes the default or fails when required.
 *   3. Documents that loaded before and must not: p-parameter="3abc" and kin.
 *   4. Input values that do not read as their field's dataType, under the
 *      invalidValueTreatments that end the record (returnInvalid, and asIs with
 *      nothing to keep) -- the cases a fixture CSV cannot express. Expected
 *      behaviour is JPMML 1.7.7's, probed per row with
 *      pmml-evaluator-example-executable-1.7.7.jar: it throws
 *      "Field "d" cannot accept invalid value "5abc"" (returnInvalid) and
 *      NumberFormatException (asIs). Where JPMML departs from the XSD lexical
 *      space the spec defines PMML values over, the spec is followed and the
 *      case says so.
 *
 * Exit status is the number of failed checks.
 *******************************************************************************/

#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <locale>
#include <pugixml.hpp>
#include <string>

#include "cPMML.h"
#include "core/xmlnode.h"
#include "utils/lexical.h"
#include "utils/utils.h"

static int failures = 0;

#define CHECK(cond)                                                                  \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << #cond << "\n";  \
      ++failures;                                                                    \
    }                                                                                \
  } while (0)

template <class E>
static bool throws(const std::function<void()>& call) {
  try {
    call();
  } catch (const E&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

static bool reads_double(const std::string& text, double expected) {
  double parsed = 0.0;
  if (!lexical::parse_double(text, parsed)) return false;
  if (std::isnan(expected)) return std::isnan(parsed);
  return parsed == expected && std::signbit(parsed) == std::signbit(expected);
}

static bool rejects_double(const std::string& text) {
  double parsed = 0.0;
  return !lexical::parse_double(text, parsed);
}

// ---- 1. the reader ----------------------------------------------------------

static void xsd_double() {
  const double inf = std::numeric_limits<double>::infinity();

  CHECK(reads_double("5", 5));
  CHECK(reads_double("+5", 5));
  CHECK(reads_double("-5", -5));
  CHECK(reads_double("5.", 5));
  CHECK(reads_double(".5", 0.5));
  CHECK(reads_double("-.5e1", -5));
  CHECK(reads_double("1e2", 100));
  CHECK(reads_double("1E+2", 100));
  CHECK(reads_double("1e-2", 0.01));
  CHECK(reads_double("007.50", 7.5));
  CHECK(reads_double("-0", -0.0));
  CHECK(reads_double("INF", inf));
  CHECK(reads_double("-INF", -inf));
  CHECK(reads_double("NaN", std::numeric_limits<double>::quiet_NaN()));

  // XML whitespace around the value is insignificant; inside it is not.
  CHECK(reads_double(" 5 ", 5));
  CHECK(reads_double("\t\r\n5\n", 5));
  CHECK(rejects_double("5 5"));
  CHECK(rejects_double("- 5"));
  CHECK(rejects_double("\v5"));  // not XML whitespace

  // Out of the double range is still a double: it rounds (XSD 1.1, Java).
  CHECK(reads_double("1.5e400", inf));
  CHECK(reads_double("-1.5e400", -inf));
  CHECK(reads_double("1e-400", 0.0));
  CHECK(reads_double("-1e-400", -0.0));
  CHECK(reads_double("1e99999999999999999999", inf));
  CHECK(reads_double("0.0000001e-99999999999999999999", 0.0));
  CHECK(reads_double("0e99999", 0.0));
  CHECK(reads_double("1e-310", 1e-310));  // subnormal, not underflow

  // The text the old reader accepted by stopping early, or through stod/strtod.
  CHECK(rejects_double("3abc"));
  CHECK(rejects_double("5abc"));
  CHECK(rejects_double("1e"));
  CHECK(rejects_double("1e+"));
  CHECK(rejects_double("."));
  CHECK(rejects_double("+"));
  CHECK(rejects_double(""));
  CHECK(rejects_double("   "));
  CHECK(rejects_double("1,5"));
  CHECK(rejects_double("1_0"));
  CHECK(rejects_double("0x10"));
  CHECK(rejects_double("++5"));
  CHECK(rejects_double("+-5"));
  // XSD 1.0 has no "+INF" (JPMML rejects it too); the spellings below are
  // Java's or C's, which JPMML accepts for an input value through
  // Double.parseDouble but not in an attribute. The XSD lexical space wins.
  CHECK(rejects_double("+INF"));
  CHECK(rejects_double("inf"));
  CHECK(rejects_double("Infinity"));
  CHECK(rejects_double("nan"));
  CHECK(rejects_double("1d"));
  CHECK(rejects_double("1f"));
}

static void xsd_integer() {
  long parsed = 0;
  CHECK(lexical::parse_long("42", parsed) && parsed == 42);
  CHECK(lexical::parse_long("+42", parsed) && parsed == 42);
  CHECK(lexical::parse_long("-42", parsed) && parsed == -42);
  CHECK(lexical::parse_long(" 7\n", parsed) && parsed == 7);
  CHECK(!lexical::parse_long("3.0", parsed));
  CHECK(!lexical::parse_long("3abc", parsed));
  CHECK(!lexical::parse_long("1e2", parsed));
  CHECK(!lexical::parse_long("", parsed));
  CHECK(!lexical::parse_long("-", parsed));
  CHECK(!lexical::parse_long("+-1", parsed));
  CHECK(!lexical::parse_long("99999999999999999999999", parsed));  // well-formed, but not a long

  double as_double = 0.0;
  CHECK(lexical::parse_integer_as_double("-3", as_double) && as_double == -3);
  CHECK(lexical::parse_integer_as_double("99999999999999999999999", as_double) && as_double == 1e23);
  CHECK(!lexical::parse_integer_as_double("3.5", as_double));
  CHECK(!lexical::parse_integer_as_double("3.0", as_double));
  CHECK(!lexical::parse_integer_as_double("INF", as_double));
  CHECK(!lexical::parse_integer_as_double("NaN", as_double));
}

static void xsd_boolean() {
  bool parsed = false;
  CHECK(lexical::parse_boolean("true", parsed) && parsed);
  CHECK(lexical::parse_boolean("1", parsed) && parsed);
  CHECK(lexical::parse_boolean("false", parsed) && !parsed);
  CHECK(lexical::parse_boolean("0", parsed) && !parsed);
  CHECK(lexical::parse_boolean(" true ", parsed) && parsed);
  CHECK(!lexical::parse_boolean("TRUE", parsed));
  CHECK(!lexical::parse_boolean("True", parsed));
  CHECK(!lexical::parse_boolean("yes", parsed));
  CHECK(!lexical::parse_boolean("2", parsed));
  CHECK(!lexical::parse_boolean("1.0", parsed));
  CHECK(!lexical::parse_boolean("", parsed));
}

// A host that installs a comma-decimal locale must not change what is read.
static void locale_independent() {
  const char* names[] = {"de_DE.UTF-8", "de_DE", "fr_FR.UTF-8"};
  const char* installed = nullptr;
  for (const char* name : names)
    if (std::setlocale(LC_ALL, name)) {
      installed = name;
      break;
    }
  if (!installed) {
    std::cout << "no comma-decimal locale on this host; locale check skipped\n";
    return;
  }
  CHECK(reads_double("1.5", 1.5));
  CHECK(rejects_double("1,5"));
  CHECK(to_double("2.25") == 2.25);
  std::setlocale(LC_ALL, "C");
}

// ---- 2. XmlNode typed getters -----------------------------------------------

static void typed_getters() {
  pugi::xml_document doc;
  doc.load_string(R"(<E d=" 2.5 " bad="3abc" l="7" lbad="7.0" b="1" bbad="yes" bcase="TRUE" s="null" e=""/>)");
  const XmlNode node(doc.child("E"));

  CHECK(node.required_double_attribute("d") == 2.5);
  CHECK(node.get_double_attribute("d", 9) == 2.5);
  CHECK(node.get_double_attribute("absent", 9) == 9);
  CHECK(node.optional_double_attribute("d") == 2.5);
  CHECK(!node.optional_double_attribute("absent"));
  CHECK(throws<cpmml::ParsingException>([&] { node.required_double_attribute("absent"); }));
  CHECK(throws<cpmml::ParsingException>([&] { node.required_double_attribute("bad"); }));
  CHECK(throws<cpmml::ParsingException>([&] { node.get_double_attribute("bad", 9); }));
  CHECK(throws<cpmml::ParsingException>([&] { node.optional_double_attribute("bad"); }));

  CHECK(node.required_long_attribute("l") == 7);
  CHECK(node.get_long_attribute("absent", 3) == 3);
  CHECK(node.optional_long_attribute("l") == 7L);
  CHECK(!node.optional_long_attribute("absent"));
  CHECK(throws<cpmml::ParsingException>([&] { node.optional_long_attribute("lbad"); }));
  CHECK(throws<cpmml::ParsingException>([&] { node.required_long_attribute("lbad"); }));
  CHECK(throws<cpmml::ParsingException>([&] { node.get_long_attribute("lbad", 3); }));

  CHECK(node.required_bool_attribute("b"));
  CHECK(node.get_bool_attribute("absent", true));
  CHECK(!node.get_bool_attribute("absent", false));
  CHECK(throws<cpmml::ParsingException>([&] { node.get_bool_attribute("bbad", false); }));
  CHECK(throws<cpmml::ParsingException>([&] { node.get_bool_attribute("bcase", false); }));
  CHECK(throws<cpmml::ParsingException>([&] { node.required_bool_attribute("absent"); }));

  // Text attributes (CODE_REVIEW.md R1): the text "null" is a value like any
  // other, and absence is told apart from it and from the empty text.
  CHECK(node.required_attribute("s") == "null");
  CHECK(node.optional_attribute("s") == std::string("null"));
  CHECK(node.optional_attribute("e") == std::string(""));
  CHECK(!node.optional_attribute("absent"));
  CHECK(node.get_attribute("absent", "dflt") == "dflt");
  CHECK(node.get_attribute("s", "dflt") == "null");
  CHECK(throws<cpmml::ParsingException>([&] { node.required_attribute("absent"); }));
}

// ---- 3. documents that must not load ----------------------------------------

static int file_counter = 0;

static std::string write_document(const std::string& body) {
  const std::string path = "strict_parse_" + std::to_string(file_counter++) + ".pmml";
  std::ofstream out(path);
  out << body;
  return path;
}

static bool loads(const std::string& document) {
  const std::string path = write_document(document);
  bool ok = true;
  try {
    cpmml::Model model(path, false);
  } catch (const cpmml::ParsingException&) {
    ok = false;
  }
  std::remove(path.c_str());
  return ok;
}

static std::string regression(const std::string& mining_field_attrs, const std::string& coefficient = "2",
                              const std::string& model_attrs = "") {
  return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="d" optype="continuous" dataType="double"/>
    <DataField name="i" optype="continuous" dataType="integer"/>
    <DataField name="b" optype="categorical" dataType="boolean"/>
    <DataField name="y" optype="continuous" dataType="double"/>
  </DataDictionary>
  <RegressionModel functionName="regression")" +
         model_attrs + R"(>
    <MiningSchema>
      <MiningField name="d")" +
         mining_field_attrs + R"(/>
      <MiningField name="i")" +
         mining_field_attrs + R"(/>
      <MiningField name="b")" +
         mining_field_attrs + R"(/>
      <MiningField name="y" usageType="target"/>
    </MiningSchema>
    <RegressionTable intercept="0">
      <NumericPredictor name="d" coefficient=")" +
         coefficient + R"("/>
      <NumericPredictor name="i" coefficient="10"/>
    </RegressionTable>
  </RegressionModel>
</PMML>)";
}

static std::string minkowski(const std::string& p) {
  return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary><DataField name="x" optype="continuous" dataType="double"/></DataDictionary>
  <ClusteringModel functionName="clustering" modelClass="centerBased" numberOfClusters="2">
    <MiningSchema><MiningField name="x"/></MiningSchema>
    <ComparisonMeasure kind="distance"><minkowski p-parameter=")" +
         p + R"("/></ComparisonMeasure>
    <ClusteringField field="x"/>
    <Cluster id="a"><Array n="1" type="real">0</Array></Cluster>
    <Cluster id="b"><Array n="1" type="real">10</Array></Cluster>
  </ClusteringModel>
</PMML>)";
}

static void malformed_documents() {
  // Sanity: the well-formed versions load.
  CHECK(loads(regression("")));
  CHECK(loads(minkowski("3")));
  CHECK(loads(minkowski(" 3 ")));

  // JPMML 1.7.7: "Required attribute minkowski@p-parameter is not defined" --
  // it reads a malformed value as no value. The old reader took "3abc" as 3.
  CHECK(!loads(minkowski("3abc")));
  CHECK(!loads(minkowski("abc")));
  // JPMML 1.7.7: "Required attribute NumericPredictor@c..." for each of these.
  CHECK(!loads(regression("", "3abc")));
  CHECK(!loads(regression("", "abc")));
  CHECK(!loads(regression("", "1d")));
  CHECK(!loads(regression("", "inf")));
  CHECK(!loads(regression("", "0x2")));
  CHECK(loads(regression("", "INF")));
  CHECK(loads(regression("", "NaN")));
  CHECK(loads(regression("", "+2")));
  // A required attribute that is absent: NumericPredictor/@coefficient.
  std::string no_coefficient = regression("");
  no_coefficient.erase(no_coefficient.find(R"( coefficient="2")"), std::string(R"( coefficient="2")").size());
  CHECK(!loads(no_coefficient));
  // isScorable is an XSD boolean. JPMML 1.7.7 scores isScorable="yes" and
  // "TRUE" (JAXB's lenient boolean); the spec's lexical space does not have
  // them, so they are load errors here.
  CHECK(loads(regression("", "2", R"( isScorable="1")")));
  CHECK(!loads(regression("", "2", R"( isScorable="yes")")));
  CHECK(!loads(regression("", "2", R"( isScorable="TRUE")")));
  // asValue with nothing to replace with; JPMML loads it and fails on the
  // first invalid input instead. A malformed replacement likewise.
  CHECK(!loads(regression(R"( invalidValueTreatment="asValue")")));
  CHECK(!loads(regression(R"( invalidValueTreatment="asValue" invalidValueReplacement="x7")")));
  // Element text goes through the same reader: an Array member "1x".
  std::string bad_array = minkowski("3");
  bad_array.replace(bad_array.find(">10<"), 4, ">10x<");
  CHECK(!loads(bad_array));
}

static cpmml::Model load(const std::string& document);

// An absent attribute that a reader needs is a load error, not the text "null"
// (CODE_REVIEW.md R1). The base build loaded the RegressionTable and
// DerivedField documents, reading that text; it already refused the
// SimplePredicate without value.
static void absent_attributes() {
  const std::string head = R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="x" optype="continuous" dataType="double"/>
    <DataField name="y" optype="categorical" dataType="string"><Value value="a"/><Value value="b"/></DataField>
  </DataDictionary>)";
  const std::string schema = R"(<MiningSchema><MiningField name="x"/><MiningField name="y" usageType="target"/></MiningSchema>)";

  // Regression.html: for a categorical target "each one must have the
  // attribute targetCategory defined with a unique value".
  const auto classification = [&](const std::string& second_table) {
    return head + R"(<RegressionModel functionName="classification" normalizationMethod="softmax">)" + schema +
           R"(<RegressionTable intercept="0" targetCategory="a"><NumericPredictor name="x" coefficient="1"/></RegressionTable>)" +
           second_table + "</RegressionModel></PMML>";
  };
  CHECK(loads(classification(R"(<RegressionTable intercept="0" targetCategory="b"/>)")));
  CHECK(!loads(classification(R"(<RegressionTable intercept="0"/>)")));

  // Predicates.html: value is required by every operator but isMissing and
  // isNotMissing.
  const auto tree = [&](const std::string& predicate) {
    return head + R"(<TreeModel functionName="classification">)" + schema +
           R"(<Node score="a"><True/><Node score="b">)" + predicate + "</Node></Node></TreeModel></PMML>";
  };
  CHECK(loads(tree(R"(<SimplePredicate field="x" operator="lessThan" value="1"/>)")));
  CHECK(loads(tree(R"(<SimplePredicate field="x" operator="isMissing"/>)")));
  CHECK(!loads(tree(R"(<SimplePredicate field="x" operator="lessThan"/>)")));

  // A TransformationDictionary DerivedField is referenced by its name.
  const auto derived = [&](const std::string& name_attr) {
    return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary><DataField name="x" optype="continuous" dataType="double"/><DataField name="y" optype="continuous" dataType="double"/></DataDictionary>
  <TransformationDictionary><DerivedField)" +
           name_attr + R"( optype="continuous" dataType="double"><FieldRef field="x"/></DerivedField></TransformationDictionary>
  <RegressionModel functionName="regression"><MiningSchema><MiningField name="x"/><MiningField name="y" usageType="target"/></MiningSchema>
    <RegressionTable intercept="0"><NumericPredictor name="x" coefficient="1"/></RegressionTable></RegressionModel>
</PMML>)";
  };
  CHECK(loads(derived(R"( name="d")")));
  CHECK(!loads(derived("")));

  // TimeSeriesModel.html: a known point needs value, and "In addition, time
  // or index must be used." The history is ordered by index, or by time when
  // no point has an index: the same series given by time, out of document
  // order, forecasts the same as by index.
  const auto arima = [](const std::string& points) {
    return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary><DataField name="ts" optype="continuous" dataType="double"/></DataDictionary>
  <TimeSeriesModel functionName="timeSeries" bestFit="ARIMA"><MiningSchema><MiningField name="ts" usageType="target"/></MiningSchema>
    <TimeSeries usage="original">)" +
           points + R"(</TimeSeries>
    <ARIMA constantTerm="0" predictionMethod="conditionalLeastSquares"><NonseasonalComponent p="2" d="0" q="0">
      <AR><Array n="2" type="real">0.5 0.25</Array></AR></NonseasonalComponent></ARIMA>
  </TimeSeriesModel></PMML>)";
  };
  const auto forecast = [&](const std::string& points) {
    cpmml::Input args;
    args["horizon"] = 2;
    return load(arima(points)).evaluate(args).series();
  };
  // y[t+1] = 0.5 y[t] + 0.25 y[t-1], history 4, 8: 0.5*8 + 0.25*4 = 5, then
  // 0.5*5 + 0.25*8 = 4.5.
  const std::vector<double> expected{5.0, 4.5};
  CHECK(forecast(R"(<TimeValue index="1" value="4"/><TimeValue index="2" value="8"/>)") == expected);
  CHECK(forecast(R"(<TimeValue index="2" value="8"/><TimeValue index="1" value="4"/>)") == expected);
  CHECK(forecast(R"(<TimeValue time="20.5" value="8"/><TimeValue time="3" value="4"/>)") == expected);
  CHECK(!loads(arima(R"(<TimeValue index="1" value="4"/><TimeValue value="8"/>)")));
  CHECK(!loads(arima(R"(<TimeValue index="1" value="4"/><TimeValue time="2" value="8"/>)")));
}

// ---- 4. invalid input values that end the record ----------------------------

static cpmml::Model load(const std::string& document) {
  const std::string path = write_document(document);
  cpmml::Model model(path, false);
  std::remove(path.c_str());
  return model;
}

static cpmml::Input row(const std::string& d, const std::string& i = "1", const std::string& b = "true") {
  cpmml::Input input;
  input["d"] = d;
  input["i"] = i;
  input["b"] = b;
  return input;
}

static void invalid_inputs() {
  // Absent attribute: the default treatment is returnInvalid.
  for (const std::string& attrs : {std::string(""), std::string(R"( invalidValueTreatment="returnInvalid")")}) {
    cpmml::Model model = load(regression(attrs));
    CHECK(model.evaluate(row("5")).as_double() == 20);
    CHECK(model.evaluate(row(" 5 ")).as_double() == 20);
    CHECK(model.evaluate(row("5", "+3")).as_double() == 40);
    CHECK(model.evaluate(row("INF")).as_double() == std::numeric_limits<double>::infinity());
    // JPMML: Field "d" cannot accept invalid value "5abc" / "abc" / "+INF" /
    // "1e" / "0x10" / NaN; Field "i" cannot accept invalid value "3.5" /
    // "INF"; Field "b" cannot accept invalid value "yes" / "2".
    for (const auto& input : {row("5abc"), row("abc"), row("+INF"), row("1e"), row("0x10"), row("NaN"),
                              row("5", "3.5"), row("5", "3abc"), row("5", "INF"), row("5", "1", "yes"),
                              row("5", "1", "2"), row("5", "1", "-1")})
      CHECK(throws<cpmml::InvalidValueException>([&] { model.evaluate(input); }));
    // Spec departures, each the XSD lexical space against JPMML's Java
    // parsing: JPMML scores these (i=3.0 as 3, i=1e2 as 100, b=TRUE as true,
    // d=1d as 1, d=inf as Infinity) and rejects b=" true ".
    for (const auto& input :
         {row("5", "3.0"), row("5", "1e2"), row("5", "1", "TRUE"), row("1d"), row("inf"), row("Infinity")})
      CHECK(throws<cpmml::InvalidValueException>([&] { model.evaluate(input); }));
    CHECK(model.evaluate(row("5", "1", " true ")).as_double() == 20);
    // validate() goes through the same reader.
    CHECK(throws<cpmml::InvalidValueException>([&] { model.validate(row("5abc")); }));
  }

  // asIs keeps a well-formed value -- NaN included -- but text that is not a
  // number has no value to keep. JPMML: NumberFormatException: For input
  // string: "5abc".
  {
    cpmml::Model model = load(regression(R"( invalidValueTreatment="asIs")"));
    CHECK(std::isnan(model.evaluate(row("NaN")).as_double()));
    for (const auto& input : {row("5abc"), row("abc"), row("+INF"), row("5", "3.5"), row("5", "1", "yes")})
      CHECK(throws<cpmml::InvalidValueException>([&] { model.evaluate(input); }));
  }
}

// The writer whose text the reader takes back: predictions travel between
// evaluators as Java text (java_format.h: ensemble combination, Targets, KNN),
// read with from_java_text, so every double -- non-finite included -- must
// come out in a form that reader accepts. The input reader stays XSD-strict:
// Java's "Infinity" is not an input value.
static void writer_round_trip() {
  const double inf = std::numeric_limits<double>::infinity();
  CHECK(java::double_text(inf) == "Infinity");
  CHECK(java::double_text(-inf) == "-Infinity");
  CHECK(java::double_text(std::numeric_limits<double>::quiet_NaN()) == "NaN");
  CHECK(java::double_text(-std::numeric_limits<double>::quiet_NaN()) == "NaN");
  CHECK(java::double_text(2.5) == "2.5");
  for (const double value : {inf, -inf, 0.0, -1.25, 1e300, 0.1, std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::denorm_min()})
    CHECK(from_java_text(java::double_text(value)) == value);
  CHECK(std::isnan(from_java_text(java::double_text(std::numeric_limits<double>::quiet_NaN()))));
  CHECK(throws<cpmml::ParsingException>([] { ::to_double("Infinity"); }));
}

int main() {
  xsd_double();
  xsd_integer();
  xsd_boolean();
  locale_independent();
  typed_getters();
  malformed_documents();
  absent_attributes();
  invalid_inputs();
  writer_round_trip();

  if (failures) std::cerr << failures << " check(s) failed\n";
  return failures;
}
