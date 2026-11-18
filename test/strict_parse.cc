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
 *   5. Names and scopes, decided by the binder at load (typed-core P2.1,
 *      bind/load_scope.h): FieldScope.html's errors fail the load and name
 *      the field; the clashes real documents contain still load; an invalid
 *      input names its field (B45).
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
#include <tuple>
#include <utility>
#include <vector>

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

// The ParsingException a document's load throws, "" when it loads, or
// "OTHER: <what>" for any other exception (the base build threw
// std::out_of_range for an unresolved name).
static std::string load_error(const std::string& document) {
  const std::string path = write_document(document);
  std::string error;
  try {
    cpmml::Model model(path, false);
  } catch (const cpmml::ParsingException& e) {
    error = e.what();
    if (error.empty()) error = "?";
  } catch (const std::exception& e) {
    error = std::string("OTHER: ") + e.what();
  }
  std::remove(path.c_str());
  return error;
}

static bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

static std::string scoped(const std::string& dictionary, const std::string& td, const std::string& mining,
                          const std::string& output, const std::string& lt, const std::string& table) {
  return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="x" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="double"/>)" +
         dictionary + "\n  </DataDictionary>" + td + R"(
  <RegressionModel modelName="m" functionName="regression">
    <MiningSchema><MiningField name="x"/><MiningField name="y" usageType="target"/>)" +
         mining + "</MiningSchema>" + output + lt + "<RegressionTable intercept=\"1\">" + table +
         "</RegressionTable>\n  </RegressionModel>\n</PMML>";
}

static std::string lt_fields(const std::string& fields) {
  return "<LocalTransformations>" + fields + "</LocalTransformations>";
}

static std::string derived(const std::string& name, const std::string& expression) {
  return R"(<DerivedField name=")" + name + R"(" optype="continuous" dataType="double">)" + expression +
         "</DerivedField>";
}

static std::string field_ref(const std::string& name) { return R"(<FieldRef field=")" + name + R"("/>)"; }

static void names_and_scopes() {
  // Sanity: the plain document loads.
  CHECK(load_error(scoped("", "", "", "", "", R"(<NumericPredictor name="x" coefficient="2"/>)")).empty());

  // FieldScope.html: "DerivedFields in the TransformationDictionary that have
  // circular references ... are not valid PMML" (here in LocalTransformations,
  // which has no evaluation order either). The base build loaded it.
  const std::string cycle =
      load_error(scoped("", "", "", "", lt_fields(derived("a", field_ref("b")) + derived("b", field_ref("a"))), ""));
  CHECK(contains(cycle, "E-CYCLE") && contains(cycle, "circular reference"));
  // A name that resolves nowhere (§3.2 step 5): in a transformation (the base
  // build loaded it and the field was always missing) and in a model body (the
  // base build threw std::out_of_range "unordered_map::at: key not found").
  const std::string in_lt = load_error(scoped("", "", "", "", lt_fields(derived("a", field_ref("nope"))), ""));
  CHECK(contains(in_lt, "E-SCOPE-UNRESOLVED") && contains(in_lt, "\"nope\""));
  const std::string in_body =
      load_error(scoped("", "", "", "", "", R"(<NumericPredictor name="nope" coefficient="2"/>)"));
  CHECK(contains(in_body, "E-SCOPE-UNRESOLVED") && contains(in_body, "\"nope\"") &&
        contains(in_body, "RegressionModel \"m\""));
  // DataDictionary.html: "The name of a DataField must be unique".
  CHECK(
      contains(load_error(scoped(R"(<DataField name="x" optype="continuous" dataType="double"/>)", "", "", "", "", "")),
               "E-NAME-DUPLICATE"));
  // FieldScope.html: "For top-level Models, their MiningSchema can refer only
  // to DataFields in the DataDictionary".
  CHECK(contains(
      load_error(scoped("", "<TransformationDictionary>" + derived("t", field_ref("x")) + "</TransformationDictionary>",
                        R"(<MiningField name="t"/>)", "", "", "")),
      "E-SCOPE-TOP-LEVEL"));

  // Accepted with a warning: a forward reference (Q4, B12), a name equal to
  // a MiningField of the same model (decided 2026-11-18), a LocalTransformations
  // field named like a TransformationDictionary field (fix_derivedfield_shadow).
  CHECK(load_error(scoped("", "", "", "", lt_fields(derived("a", field_ref("b")) + derived("b", field_ref("x"))),
                          R"(<NumericPredictor name="a" coefficient="2"/>)"))
            .empty());
  CHECK(
      load_error(
          scoped(
              "", "", "",
              R"(<Output><OutputField name="y" optype="continuous" dataType="double" feature="predictedValue"/></Output>)",
              "", ""))
          .empty());
  CHECK(
      load_error(scoped("", "<TransformationDictionary>" + derived("t", field_ref("x")) + "</TransformationDictionary>",
                        "", "", lt_fields(derived("t", field_ref("y"))), ""))
          .empty());

  // Segment OutputFields across siblings. Two segments of `method`, the first
  // with OutputField `segment_output`; the second reads `o` in its MiningSchema
  // (chain) or its Segment predicate (otherwise); the MiningModel has OutputField
  // `parent_output`.
  const auto segments = [](const std::string& method, const std::string& parent_output,
                           const std::string& segment_output) {
    const bool chain = method == "modelChain";
    return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary><DataField name="i" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="double"/></DataDictionary>
  <MiningModel functionName="regression">
    <MiningSchema><MiningField name="i"/><MiningField name="y" usageType="target"/></MiningSchema>
    <Output><OutputField name=")" +
           parent_output + R"(" optype="continuous" dataType="double" feature="predictedValue"/></Output>
    <Segmentation multipleModelMethod=")" +
           method + R"(">
      <Segment id="1"><True/><RegressionModel functionName="regression">
        <MiningSchema><MiningField name="i"/></MiningSchema>
        <Output><OutputField name=")" +
           segment_output + R"(" optype="continuous" dataType="double" feature="predictedValue"/></Output>
        <RegressionTable intercept="1"><NumericPredictor name="i" coefficient="1"/></RegressionTable>
      </RegressionModel></Segment>
      <Segment id="2">)" +
           (chain ? std::string("<True/>")
                  : std::string(R"(<SimplePredicate field="o" operator="greaterThan" value="2"/>)")) +
           R"(<RegressionModel functionName="regression">
        <MiningSchema><MiningField name=")" +
           (chain ? "o" : "i") + R"("/></MiningSchema>
        <RegressionTable intercept="0"><NumericPredictor name=")" +
           (chain ? "o" : "i") + R"(" coefficient="2"/></RegressionTable>
      </RegressionModel></Segment>
    </Segmentation>
  </MiningModel>
</PMML>)";
  };
  // FieldScope.html: "In a model chain, all Outputs from the segment models
  // become part of the enclosing mining model scope."
  CHECK(load_error(segments("modelChain", "p", "o")).empty());
  // "Note that for model chaining, Outputs cannot be duplicated across segment
  // since they all belong to the same scope": one error, and the later
  // segment's MiningField "o" still resolves (no follow-on E-SCOPE-UNRESOLVED).
  const std::string chain_duplicate = load_error(segments("modelChain", "o", "o"));
  CHECK(contains(chain_duplicate, "has 1 error:") && contains(chain_duplicate, "E-NAME-DUPLICATE") &&
        !contains(chain_duplicate, "E-SCOPE-UNRESOLVED"));
  // "Outputs are generally local to a model, except when the model is used in
  // a model chain": outside a chain an earlier sibling's OutputField is not in
  // the parent's scope. JPMML 1.7.7 and the base build score this document
  // (JPMML evaluates the predicate lazily, after segment 1); a departure.
  CHECK(contains(load_error(segments("selectFirst", "p", "o")), "E-SCOPE-UNRESOLVED"));
  // Within a model chain, only the Outputs of the segments before a reference
  // are defined: FieldScope.html "All other fields in PMML must be defined
  // before they are referenced" (Q4/D12 accept forward references within one
  // container, not to a later segment's Output). Segment 1 reads segment 2's
  // OutputField "q" in its predicate and in its model body (an implicit
  // pass-through, Q3). JPMML 1.7.7: 'Field "q" is not defined'; the base build
  // threw std::out_of_range; a first P2.1 build loaded both and read q missing.
  const auto chain_forward = [](bool in_body) {
    return std::string(R"(<?xml version="1.0"?><PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary><DataField name="x" optype="continuous" dataType="double"/>
    <DataField name="s" optype="categorical" dataType="string"/>
    <DataField name="y" optype="continuous" dataType="double"/></DataDictionary>
  <MiningModel functionName="regression">
    <MiningSchema><MiningField name="x"/><MiningField name="s"/><MiningField name="y" usageType="target"/></MiningSchema>
    <Segmentation multipleModelMethod="modelChain">
      <Segment id="1">)") +
           (in_body ? "<True/>" : R"(<SimplePredicate field="q" operator="greaterThan" value="1"/>)") +
           R"(<TreeModel functionName="regression" noTrueChildStrategy="returnLastPrediction">
        <MiningSchema><MiningField name="x"/><MiningField name="s"/></MiningSchema>
        <Output><OutputField name="p1" optype="continuous" dataType="double" feature="predictedValue"/></Output>
        <Node score="7"><True/>)" +
           (in_body ? R"(<Node score="8"><SimplePredicate field="q" operator="greaterThan" value="1"/></Node>)" : "") +
           R"(</Node>
      </TreeModel></Segment>
      <Segment id="2"><True/><TreeModel functionName="regression">
        <MiningSchema><MiningField name="x"/><MiningField name="s"/></MiningSchema>
        <Output><OutputField name="q" optype="continuous" dataType="double" feature="predictedValue"/></Output>
        <Node score="3"><True/></Node>
      </TreeModel></Segment>
    </Segmentation>
  </MiningModel>
</PMML>)";
  };
  for (const bool in_body : {false, true}) {
    const std::string forward = load_error(chain_forward(in_body));
    CHECK(contains(forward, "E-SCOPE-UNRESOLVED") && contains(forward, "\"q\""));
  }
  // The backward reference loads: segment 2's predicate reads segment 1's p1.
  std::string backward = chain_forward(true);
  const auto swap = [&backward](const std::string& from, const std::string& to) {
    const std::size_t at = backward.find(from);
    if (at != std::string::npos) backward.replace(at, from.size(), to);
    return at != std::string::npos;
  };
  CHECK(swap(R"(<Node score="8"><SimplePredicate field="q" operator="greaterThan" value="1"/></Node>)", "") &&
        swap(R"(<Segment id="2"><True/>)",
             R"(<Segment id="2"><SimplePredicate field="p1" operator="greaterThan" value="1"/>)"));
  CHECK(load_error(backward).empty());
  // A Segment's predicate is evaluated before its model: the segment's own
  // OutputField is not defined there (JPMML 1.7.7: 'Field "q" is not defined').
  std::string own = backward;
  own.replace(own.find(R"(field="p1")"), 10, R"(field="q")");
  const std::string own_error = load_error(own);
  CHECK(contains(own_error, "E-SCOPE-UNRESOLVED") && contains(own_error, "\"q\""));

  // B45: the record is invalid when any MiningField fails its valid values
  // (D25), and the error names that field, read by the model or not.
  cpmml::Model model =
      load(scoped(R"(<DataField name="z" optype="categorical" dataType="string"><Value value="a"/></DataField>)", "",
                  R"(<MiningField name="z"/>)", "", "", ""));
  cpmml::Input input;
  input["x"] = "1";
  input["z"] = "b";
  std::string message;
  try {
    model.evaluate(input);
  } catch (const cpmml::InvalidValueException& e) {
    message = e.what();
  }
  CHECK(contains(message, "Invalid value for field: z") && contains(message, "\"b\""));
}

// Document text (design commit P2.2, typed-core.md §3.4): every attribute is
// read with its XSD type, a value of a field by the K1 rule in the field's
// type, and a MapValues table's keys must be unique and not NaN.
static std::string typed_doc(const std::string& td, const std::string& table = "") {
  return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="s" optype="categorical" dataType="string"/>
    <DataField name="i" optype="continuous" dataType="integer"/>
    <DataField name="f" optype="continuous" dataType="float"/>
    <DataField name="d" optype="continuous" dataType="double"/>
    <DataField name="b" optype="categorical" dataType="boolean"/>
    <DataField name="y" optype="continuous" dataType="double"/>
  </DataDictionary>
  <TransformationDictionary>)" +
         td + R"(</TransformationDictionary>
  <RegressionModel functionName="regression">
    <MiningSchema><MiningField name="s"/><MiningField name="i"/><MiningField name="f"/><MiningField name="d"/>
      <MiningField name="b"/><MiningField name="y" usageType="target"/></MiningSchema>
    <RegressionTable intercept="0">)" +
         table + R"(</RegressionTable>
  </RegressionModel>
</PMML>)";
}

// A DerivedField x of `type` over a MapValues keyed by `field` with `rows`
// (key, output) and the MapValues attributes `attrs`.
static std::string map_values(const std::string& type, const std::string& field,
                              const std::vector<std::pair<std::string, std::string>>& rows,
                              const std::string& attrs = "") {
  std::string table;
  for (const auto& [k, v] : rows) table += "<row><k>" + k + "</k><v>" + v + "</v></row>";
  return R"(<DerivedField name="x" optype="continuous" dataType=")" + type + R"("><MapValues outputColumn="v")" +
         attrs + R"(><FieldColumnPair field=")" + field + R"(" column="k"/><InlineTable>)" + table +
         "</InlineTable></MapValues></DerivedField>";
}

static std::string set_tree(const std::string& field, const std::string& members) {
  return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="s" optype="categorical" dataType="string"/>
    <DataField name="d" optype="continuous" dataType="double"/>
    <DataField name="y" optype="categorical" dataType="string"/>
  </DataDictionary>
  <TreeModel functionName="classification">
    <MiningSchema><MiningField name="s"/><MiningField name="d"/><MiningField name="y" usageType="target"/></MiningSchema>
    <Node score="a"><True/><Node score="b"><SimpleSetPredicate field=")" +
         field + R"(" booleanOperator="isIn"><Array type="string">)" + members +
         R"(</Array></SimpleSetPredicate></Node></Node>
  </TreeModel>
</PMML>)";
}

static std::string grm_covariate(const std::string& value) {
  return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML version="4.4" xmlns="http://www.dmg.org/PMML-4_4"><Header/>
  <DataDictionary><DataField name="x1" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="double"/></DataDictionary>
  <GeneralRegressionModel modelType="regression" functionName="regression" targetVariableName="y">
    <MiningSchema><MiningField name="x1"/><MiningField name="y" usageType="target"/></MiningSchema>
    <ParameterList><Parameter name="p0"/><Parameter name="p1"/></ParameterList>
    <CovariateList><Predictor name="x1"/></CovariateList>
    <PPMatrix><PPCell value=")" +
         value + R"(" predictorName="x1" parameterName="p1"/></PPMatrix>
    <ParamMatrix><PCell parameterName="p0" beta="0.2"/><PCell parameterName="p1" beta="-0.5"/></ParamMatrix>
  </GeneralRegressionModel>
</PMML>)";
}

static void document_text() {
  // §2.7 / D26: "A NaN key is a load error (it can match nothing)". The
  // fuzzer case behind fix_mapvalues_nan_key (B35), as it was registered:
  // JPMML 1.7.7 loads it and the row never matches.
  const std::string nan_key = R"PMML(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/><DataDictionary><DataField name="s" optype="categorical" dataType="string"></DataField><DataField name="i" optype="continuous" dataType="integer"></DataField><DataField name="f" optype="continuous" dataType="float"></DataField><DataField name="d" optype="continuous" dataType="double"></DataField><DataField name="b" optype="categorical" dataType="boolean"></DataField><DataField name="y" optype="continuous" dataType="double"/></DataDictionary><TransformationDictionary><DerivedField name="x" optype="continuous" dataType="integer"><MapValues outputColumn="v"><FieldColumnPair field="d" column="k"/><InlineTable><row><k>NaN</k><v>3</v></row><row><k>-1.25</k><v>007</v></row><row><k>0.1</k><v>1</v></row></InlineTable></MapValues></DerivedField></TransformationDictionary><RegressionModel functionName="regression"><MiningSchema><MiningField name="s"/><MiningField name="i"/><MiningField name="f"/><MiningField name="d"/><MiningField name="b"/><MiningField name="y" usageType="target"/></MiningSchema><Output><OutputField name="o_s" optype="categorical" dataType="string" feature="transformedValue"><FieldRef field="x"/></OutputField><OutputField name="o_t" optype="continuous" dataType="integer" feature="transformedValue"><FieldRef field="x"/></OutputField></Output><RegressionTable intercept="0"><NumericPredictor name="x" coefficient="1"/></RegressionTable></RegressionModel></PMML>)PMML";
  CHECK(contains(load_error(nan_key), "E-KEY-NAN"));
  // A NaN key over an integer field is the xs:double NaN (§3.4 rule 2).
  CHECK(contains(load_error(typed_doc(map_values("double", "i", {{"NaN", "1"}, {"3", "2"}}))), "E-KEY-NAN"));
  CHECK(load_error(typed_doc(map_values("double", "s", {{"NaN", "1"}, {"3", "2"}}))).empty());  // a string
  // A SimpleSetPredicate member too, whatever the number of members (the
  // linear path must give the hashed path's answers, P1.2).
  CHECK(contains(load_error(set_tree("d", "1 NaN 3")), "E-KEY-NAN"));
  CHECK(load_error(set_tree("d", "1 2 3")).empty());
  CHECK(load_error(set_tree("s", "1 NaN 3")).empty());

  // B42, Transformations.html (MapValues): "it is an error if the table
  // entries used for matching are not unique". Keys are equal once typed:
  // an integer 3 and 3.0, a double 3.5 and 3.50, a float 0.1 and its double
  // spelling, 0 and -0 (§2.7 KeyEqual), boolean true and 1, the same
  // string. The base build loaded each, one row silently winning; JPMML
  // refuses each ("Element InlineTable is not valid").
  for (const auto& [field, a, b] : std::vector<std::tuple<std::string, std::string, std::string>>{
           {"i", "3", "3.0"},
           {"d", "3.5", "3.50"},
           {"f", "0.1", "0.10000000149011612"},
           {"d", "0", "-0"},
           {"b", "true", "1"},
           {"s", "a", "a"}}) {
    const std::string error = load_error(typed_doc(map_values("double", field, {{a, "1"}, {b, "2"}})));
    CHECK(contains(error, "E-KEY-DUPLICATE"));
  }
  CHECK(load_error(typed_doc(map_values("double", "s", {{"1", "1"}, {"1.0", "2"}}))).empty());  // two strings
  CHECK(load_error(typed_doc(map_values("double", "d", {{"0.1", "1"}, {"0.10000000149011612", "2"}}))).empty());

  // MapValues and Discretize values are of their own dataType, else of the
  // nearest enclosing declared type (§3.4, D21), parsed strictly as it: an
  // integer's "1.5" is malformed (the base build read it in the engine's
  // double).
  CHECK(contains(load_error(typed_doc(map_values("double", "s", {{"a", "1.5"}}, R"( dataType="integer")"))),
                 "E-VALUE-MALFORMED"));
  CHECK(contains(load_error(typed_doc(map_values("integer", "s", {{"a", "1.5"}}))), "E-VALUE-MALFORMED"));
  CHECK(load_error(typed_doc(map_values("integer", "s", {{"a", "007"}}))).empty());
  CHECK(contains(load_error(typed_doc(map_values("double", "s", {{"a", "1"}}, R"( dataType="double" defaultValue="x")"))),
                 "E-VALUE-MALFORMED"));
  const std::string discretize =
      R"(<DerivedField name="x" optype="continuous" dataType="integer"><Discretize field="d"><DiscretizeBin binValue=")";
  const std::string bin = R"("><Interval closure="closedOpen" leftMargin="0" rightMargin="3"/></DiscretizeBin></Discretize></DerivedField>)";
  CHECK(contains(load_error(typed_doc(discretize + "1.50" + bin)), "E-VALUE-MALFORMED"));
  CHECK(load_error(typed_doc(discretize + "2" + bin)).empty());
  // A string MapValues in a double DerivedField is cast at load (§2.6); text
  // with no double value has no value to give.
  CHECK(load_error(typed_doc(map_values("double", "s", {{"a", "1.50"}}, R"( dataType="string")"))).empty());
  CHECK(contains(load_error(typed_doc(map_values("double", "s", {{"a", "abc"}}, R"( dataType="string")"))),
                 "E-VALUE-MALFORMED"));

  // Apply mapMissingTo/defaultValue are of the Apply's result type (§3.4):
  // "true" for equal in a double DerivedField (the base build refused it:
  // "true" is not a valid double), not "abc" for +. They are then cast to the
  // enclosing DerivedField's type at load (§2.6): concat's "5" is the double
  // 5, its "abc" has no double value.
  const auto apply = [](const std::string& type, const std::string& function, const std::string& attrs) {
    return R"(<DerivedField name="x" optype="continuous" dataType=")" + type + R"("><Apply function=")" + function +
           R"(" )" + attrs + R"(><FieldRef field="i"/><Constant>3</Constant></Apply></DerivedField>)";
  };
  CHECK(load_error(typed_doc(apply("double", "equal", R"(mapMissingTo="true")"))).empty());
  CHECK(load_error(typed_doc(apply("double", "concat", R"(mapMissingTo="5")"))).empty());
  CHECK(contains(load_error(typed_doc(apply("double", "concat", R"(mapMissingTo="abc")"))), "E-VALUE-MALFORMED"));
  CHECK(load_error(typed_doc(apply("string", "concat", R"(mapMissingTo="abc")"))).empty());
  // FieldRef@mapMissingTo is "parsed as T(field) by the rule above" (§3.4,
  // K1), then cast to the enclosing DerivedField's type: an integer field's
  // "2.5" is a double (rule 2); a double field's "abc" is malformed; a string
  // field's "5" is the double 5 in a double DerivedField, its "abc" has no
  // double value.
  const auto fieldref = [](const std::string& type, const std::string& field, const std::string& text) {
    return R"(<DerivedField name="x" optype="continuous" dataType=")" + type + R"("><FieldRef field=")" + field +
           R"(" mapMissingTo=")" + text + R"("/></DerivedField>)";
  };
  CHECK(load_error(typed_doc(fieldref("double", "i", "2.5"))).empty());
  CHECK(contains(load_error(typed_doc(fieldref("string", "d", "abc"))), "E-VALUE-MALFORMED"));
  CHECK(load_error(typed_doc(fieldref("double", "s", "5"))).empty());
  CHECK(contains(load_error(typed_doc(fieldref("double", "s", "abc"))), "E-VALUE-MALFORMED"));
  CHECK(contains(load_error(typed_doc(apply("string", "+", R"(mapMissingTo="abc")"))), "E-VALUE-MALFORMED"));
  CHECK(contains(load_error(typed_doc(apply("string", "+", R"(defaultValue="abc")"))), "E-VALUE-MALFORMED"));
  // NormContinuous and NormDiscrete mapMissingTo and LinearNorm orig/norm are
  // NUMBER (xs:double) in a string DerivedField too (B11): the base build read
  // them as strings, so these loaded and interpolated string codes.
  const std::string nc = R"(<DerivedField name="x" optype="categorical" dataType="string"><NormContinuous field="d")";
  CHECK(load_error(typed_doc(nc + R"(><LinearNorm orig="0" norm="0"/><LinearNorm orig="1" norm="1"/></NormContinuous></DerivedField>)"))
            .empty());
  CHECK(!load_error(typed_doc(nc + R"(><LinearNorm orig="a" norm="0"/><LinearNorm orig="b" norm="1"/></NormContinuous></DerivedField>)"))
             .empty());
  CHECK(!load_error(typed_doc(nc + R"( mapMissingTo="m"><LinearNorm orig="0" norm="0"/><LinearNorm orig="1" norm="1"/></NormContinuous></DerivedField>)"))
             .empty());
  CHECK(!load_error(typed_doc(R"(<DerivedField name="x" optype="categorical" dataType="string"><NormDiscrete field="s" value="a" mapMissingTo="m"/></DerivedField>)"))
             .empty());

  // B7, pmml-4-4.xsd: <xs:attribute name="exponent" type="INT-NUMBER"
  // default="1"/>; INT-NUMBER is xs:integer. The base build read "2.5" as a
  // double (4^2.5 = 32); JPMML 1.7.7 loads it and scores x^1.
  const auto exponent = [](const std::string& e) {
    return typed_doc("", R"(<NumericPredictor name="i" coefficient="1" exponent=")" + e + R"("/>)");
  };
  // D28 (decided 2026-11-18): "2.0", "2e0", "2.5" are not in the xs:integer
  // lexical space ("a finite-length sequence of decimal digits (#x30-#x39)
  // with an optional leading sign", XML Schema Part 2 §3.3.13.1) and are load
  // errors; "+2", "02" and " 2 " (whiteSpace collapse) are the integer 2
  // (fixture fix_numericpredictor_exponent_text scores them).
  CHECK(!load_error(exponent("2.5")).empty());
  CHECK(!load_error(exponent("2.0")).empty());
  CHECK(!load_error(exponent("2e0")).empty());
  CHECK(load_error(exponent("2")).empty());
  CHECK(load_error(exponent("-1")).empty());
  CHECK(load_error(exponent("+2")).empty());
  CHECK(load_error(exponent("02")).empty());
  CHECK(load_error(exponent(" 2 ")).empty());
  // D29 (decided 2026-11-18): NormDiscrete@value is read in its field's type
  // by K1 (§3.4): on an integer field "3.5" is the double 3.5 (rule 2) and
  // loads (JPMML 1.7.7 refuses it, NumberFormatException; fixture
  // fix_normdiscrete_integer_decimal scores it 0); non-numeric text on a
  // numeric field is rule 3, a load error.
  const auto norm_discrete = [](const std::string& field, const std::string& value) {
    return typed_doc(R"(<DerivedField name="x" optype="continuous" dataType="double"><NormDiscrete field=")" + field +
                     R"(" value=")" + value + R"("/></DerivedField>)");
  };
  CHECK(load_error(norm_discrete("i", "3.5")).empty());
  CHECK(load_error(norm_discrete("i", "3.0")).empty());
  CHECK(!load_error(norm_discrete("i", "abc")).empty());
  CHECK(!load_error(norm_discrete("d", "abc")).empty());
  CHECK(!load_error(norm_discrete("b", "yes")).empty());
  CHECK(load_error(norm_discrete("s", "abc")).empty());
  // A GeneralRegression covariate's PPCell@value is its exponent
  // (GeneralRegression.html: "the cell value is set to the exponent"), and
  // D28 (§8.5) reads it as INT-NUMBER, xs:integer, as NumericPredictor's:
  // malformed or non-integer text is a load error (the base build failed
  // every record for "two" and scored x^2.5 for "2.5").
  CHECK(load_error(grm_covariate("2")).empty());
  CHECK(load_error(grm_covariate("+2")).empty());
  CHECK(load_error(grm_covariate("02")).empty());
  CHECK(load_error(grm_covariate(" 2 ")).empty());
  CHECK(contains(load_error(grm_covariate("two")), "E-ATTRIBUTE-MALFORMED"));
  CHECK(contains(load_error(grm_covariate("2.5")), "E-ATTRIBUTE-MALFORMED"));
  CHECK(contains(load_error(grm_covariate("2.0")), "E-ATTRIBUTE-MALFORMED"));
  CHECK(contains(load_error(grm_covariate("2e0")), "E-ATTRIBUTE-MALFORMED"));
  // D29 is fixture fix_normdiscrete_integer_decimal (value="3.5" on an
  // integer field loads and gives 0) and the norm_discrete checks above.
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
  names_and_scopes();
  document_text();
  writer_round_trip();

  if (failures) std::cerr << failures << " check(s) failed\n";
  return failures;
}
