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
 *   6. Constant typing (typed-core P2.3): typed text read strictly, untyped
 *      text inferred or inherited from the built-in's parameter (S2).
 *   7. Surrogate predicates (typed-core P2.8b3): an error from an operand is
 *      not UNKNOWN; it ends the record as it would outside the surrogate.
 *   8. The condition of `if` is boolean (typed-core P4.3c, B53): a number or
 *      string condition fails the load.
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
#include "core/datadictionary.h"  // first: predicate.h does not compile as the first of its cycle
#include "core/fastpredicate.h"
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
  // D30: XSD 1.1 §3.3.5.2, "the literals 'INF', '+INF', '-INF', and 'NaN'"
  // (XSD 1.0 had no +INF; JPMML 1.7.7 refuses it: NumberFormatException).
  // Case-sensitive like the others, and only INF takes a '+'.
  CHECK(reads_double("+INF", inf));
  CHECK(reads_double(" +INF\n", inf));
  CHECK(rejects_double("+inf"));
  CHECK(rejects_double("+Infinity"));
  CHECK(rejects_double("+NaN"));
  CHECK(rejects_double("-NaN"));
  CHECK(rejects_double("++INF"));
  CHECK(rejects_double("+-INF"));
  CHECK(rejects_double("+ INF"));

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
  // The spellings below are Java's or C's, which JPMML accepts for an input
  // value through Double.parseDouble but not in an attribute. The XSD
  // lexical space wins.
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
  CHECK(loads(regression("", "+INF")));  // D30 (JPMML: NumberFormatException)
  CHECK(!loads(regression("", "+inf")));
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
    // D30: an input reads the same XSD 1.1 double grammar as the document,
    // so "+INF" is +INF (JPMML 1.7.7: Field "d" cannot accept invalid value
    // "+INF"); "+inf" is not in it, and an integer field takes no INF.
    CHECK(model.evaluate(row("+INF")).as_double() == std::numeric_limits<double>::infinity());
    // JPMML: Field "d" cannot accept invalid value "5abc" / "abc" / "1e" /
    // "0x10" / NaN; Field "i" cannot accept invalid value "3.5" / "INF";
    // Field "b" cannot accept invalid value "yes" / "2".
    for (const auto& input :
         {row("5abc"), row("abc"), row("+inf"), row("1e"), row("0x10"), row("NaN"), row("5", "3.5"), row("5", "3abc"),
          row("5", "INF"), row("5", "+INF"), row("5", "1", "yes"), row("5", "1", "2"), row("5", "1", "-1")})
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
    CHECK(model.evaluate(row("+INF")).as_double() == std::numeric_limits<double>::infinity());  // D30
    for (const auto& input : {row("5abc"), row("abc"), row("+inf"), row("5", "3.5"), row("5", "1", "yes")})
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

// Constant typing (typed-core P2.3, §3.4; expression/constant.h): the load
// errors a fixture cannot hold. Each document is one TransformationDictionary
// DerivedField over the Constant (an OutputField would catch the error until
// P2.5, B13). JPMML 1.7.7 was probed on each and is noted where it departs.
static std::string constant_field(const std::string& type, const std::string& expression) {
  return typed_doc(R"(<DerivedField name="x" optype="continuous" dataType=")" + type + R"(">)" + expression +
                   "</DerivedField>");
}

static void constants() {
  const auto error = [](const std::string& type, const std::string& expression) {
    return load_error(constant_field(type, expression));
  };
  const auto malformed = [&](const std::string& type, const std::string& expression) {
    return contains(error(type, expression), "E-VALUE-MALFORMED");
  };
  // dataType given ("The dataType of Constant can be optionally specified"):
  // the text is read in that type, XSD-strict. The
  // base build read integer text as a double (1.5 loaded). JPMML 1.7.7
  // refuses the first four too (NumberFormatException) but scores
  // dataType="boolean" TRUE as true (Java's parseBoolean) and
  // dataType="integer" 1e3 as 1000: departures, the spec deciding (XML Schema
  // Part 2 §3.2.2.1, boolean: "{true, false, 1, 0}"; §3.3.13.1, integer: "a
  // finite-length sequence of decimal digits (#x30-#x39) with an optional
  // leading sign").
  CHECK(malformed("double", R"(<Constant dataType="integer">1.5</Constant>)"));
  CHECK(malformed("double", R"(<Constant dataType="integer">INF</Constant>)"));  // only float, double (S3)
  CHECK(malformed("double", R"(<Constant dataType="double">Infinity</Constant>)"));
  CHECK(malformed("double", R"(<Constant dataType="double">1d</Constant>)"));
  CHECK(malformed("boolean", R"(<Constant dataType="boolean">TRUE</Constant>)"));
  CHECK(malformed("double", R"(<Constant dataType="integer">1e3</Constant>)"));
  CHECK(contains(error("double", R"(<Constant dataType="real">1</Constant>)"), "E-ENUM"));
  // S3: NaN, INF, -INF "are case insensitive and only meaningful as floating
  // point values"; the base build refused inf.
  CHECK(error("double", R"(<Constant dataType="double">inF</Constant>)").empty());
  CHECK(error("double", R"(<Constant dataType="float">-iNf</Constant>)").empty());
  CHECK(error("double", R"(<Constant dataType="double">nan</Constant>)").empty());
  // D30: +INF is INF's XSD 1.1 spelling (the Constant text links
  // xmlschema11-2 #float), case-insensitive as INF is. JPMML 1.7.7 refuses
  // each (NumberFormatException: For input string: "+INF"). +NaN is not a
  // literal, and an integer takes no INF.
  CHECK(error("double", R"(<Constant dataType="double">+INF</Constant>)").empty());
  CHECK(error("double", R"(<Constant dataType="double">+inf</Constant>)").empty());
  CHECK(error("double", R"(<Constant dataType="float">+Inf</Constant>)").empty());
  CHECK(malformed("double", R"(<Constant dataType="double">+NaN</Constant>)"));
  CHECK(malformed("double", R"(<Constant dataType="double">+Infinity</Constant>)"));
  CHECK(malformed("double", R"(<Constant dataType="integer">+INF</Constant>)"));
  // Empty content of a typed number is a missing value, blanks-only content
  // is not empty (xs:string, whiteSpace preserve) and collapses to "", which
  // no number reads (P1.2; JPMML: "empty String"). The engine's pugixml flags
  // drop a blanks-only text, so this is read from the document text.
  CHECK(error("double", R"(<Constant dataType="double"></Constant>)").empty());
  CHECK(error("double", R"(<Constant dataType="double"/>)").empty());
  CHECK(error("double", R"(<Constant dataType="double"><!-- none --></Constant>)").empty());
  CHECK(malformed("double", R"(<Constant dataType="double"> </Constant>)"));
  CHECK(malformed("double", "<Constant dataType=\"double\">\n\t</Constant>"));
  CHECK(malformed("double", R"(<Constant dataType="double"> <!-- none --> </Constant>)"));
  CHECK(malformed("double", R"(<Constant dataType="double" note="a>b"> </Constant>)"));  // '>' in an attribute
  CHECK(error("double", R"(<Constant dataType="double"> 1 </Constant>)").empty());
  // "If the missing attribute is true, the constant will be taken as a
  // missing value, regardless of the content."
  CHECK(error("double", R"(<Constant dataType="integer" missing="true">abc</Constant>)").empty());
  CHECK(error("double", R"(<Constant missing="true">99999999999999999999</Constant>)").empty());
  CHECK(contains(error("double", R"(<Constant dataType="double" missing="yes">1</Constant>)"), "missing"));
  // No dataType: an integer beyond int64 is a load error (§3.4, D4; JPMML
  // reads it as a string and refuses "+": "No PMML data type for the
  // intersection of PMML data types double and string").
  CHECK(malformed("double",
                  R"(<Apply function="+"><FieldRef field="d"/><Constant>99999999999999999999</Constant></Apply>)"));
  CHECK(error("double", R"(<Apply function="+"><FieldRef field="d"/><Constant>9223372036854775807</Constant></Apply>)")
            .empty());
  // S2: an untyped Constant argument inherits the parameter's dataType, so it
  // must be a value of it. substring's startPos is declared integer
  // (JPMML 1.7.7 loads 1.0 and scores it as 1: a departure, the spec's
  // "inherit the dataType" decides); the condition of if and the argument of
  // not are boolean (JPMML: IllegalArgumentException "yes" at the first
  // record).
  CHECK(malformed(
      "string",
      R"(<Apply function="substring"><FieldRef field="s"/><Constant>1.0</Constant><Constant>2</Constant></Apply>)"));
  CHECK(
      error("string",
            R"(<Apply function="substring"><FieldRef field="s"/><Constant>1</Constant><Constant>2</Constant></Apply>)")
          .empty());
  CHECK(malformed(
      "integer",
      R"(<Apply function="if"><Constant>yes</Constant><Constant>1</Constant><Constant>2</Constant></Apply>)"));
  CHECK(malformed("boolean", R"(<Apply function="not"><Constant>2</Constant></Apply>)"));
  // The blanks of a Constant are its text as XML hands it on, whatever the
  // document's encoding: XML 1.0 §2.11, "translating both the two-character
  // sequence #xD #xA and any #xD that is not followed by #xA to a single
  // #xA", so stringLength of CR LF is 1 (JPMML 1.7.7 agrees); a document
  // pugixml converts from ISO-8859-1 reads them as a UTF-8 one does (the
  // first P2.3 draft failed to load it once a non-ASCII byte preceded the
  // Constant).
  const auto length_of = [](const std::string& document) {
    const std::string path = write_document(document);
    double length = -1;
    try {
      cpmml::Model model(path, false);
      length = model.evaluate(row("1")).as_double();
    } catch (const std::exception&) {
    }
    std::remove(path.c_str());
    return length;
  };
  const auto blanks_doc = [](const std::string& blanks) {
    return typed_doc(R"(<DerivedField name="x" optype="continuous" dataType="integer">)"
                     R"(<Apply function="stringLength"><Constant>)" +
                         blanks + "</Constant></Apply></DerivedField>",
                     R"(<NumericPredictor name="x" coefficient="1"/>)");
  };
  const auto latin1 = [](std::string document) {
    document.replace(document.find("UTF-8"), 5, "ISO-8859-1");
    document.replace(document.find("<Header/>"), 9, "<Header description=\"caf\xE9\"/>");
    return document;
  };
  // The value of a Constant as the model's prediction (x times 1, d = 1).
  const auto value_of = [&](const std::string& expression) {
    return length_of(typed_doc(R"(<DerivedField name="x" optype="continuous" dataType="double">)" + expression +
                                   "</DerivedField>",
                               R"(<NumericPredictor name="x" coefficient="1"/>)"));
  };
  const double inf = std::numeric_limits<double>::infinity();
  CHECK(value_of(R"(<Constant dataType="double">+iNF</Constant>)") == inf);
  CHECK(value_of(R"(<Constant dataType="float">+INF</Constant>)") == inf);
  // An untyped +INF is a string (D10, D30): "non-numeric characters ... will
  // result in the Constant being treated as a string" (JPMML agrees).
  CHECK(value_of(R"(<Apply function="stringLength"><Constant>+INF</Constant></Apply>)") == 4);
  // D31: an untyped 1E400 is scientific notation, so a double ("A scientific
  // notation expression like 3.14e+5 or like 3.14E-5 should also be treated
  // as a double"), and beyond the double range it rounds to Infinity (XSD
  // 1.1). JPMML 1.7.7 reads it as a string: "No PMML data type for the
  // intersection of PMML data types double and string".
  CHECK(value_of(R"(<Apply function="+"><FieldRef field="d"/><Constant>1E400</Constant></Apply>)") == inf);
  CHECK(value_of(R"(<Apply function="+"><FieldRef field="d"/><Constant>-1e400</Constant></Apply>)") == -inf);
  CHECK(length_of(blanks_doc("\r\n")) == 1);
  CHECK(length_of(blanks_doc("\r")) == 1);
  CHECK(length_of(blanks_doc(" \r\n ")) == 3);
  CHECK(length_of(latin1(blanks_doc("  "))) == 2);
  CHECK(length_of(latin1(blanks_doc(""))) == 0);
  CHECK(contains(load_error(latin1(constant_field("double", R"(<Constant dataType="double"> </Constant>)"))),
                 "E-VALUE-MALFORMED"));
  CHECK(load_error(latin1(constant_field("double", R"(<Constant dataType="double"/>)"))).empty());
  // Parameters with no declared dataType inherit nothing: if's values and
  // concat's arguments are inferred, so "yes" is a string there.
  CHECK(error("string", R"(<Apply function="if"><Constant>true</Constant><Constant>yes</Constant></Apply>)").empty());
  CHECK(
      error("string", R"(<Apply function="concat"><Constant>yes</Constant><Constant>1.0</Constant></Apply>)").empty());
}

// A RegressionModel over typed_doc's fields whose Output holds the predicted
// value and `fields`.
static std::string output_doc(const std::string& fields, const std::string& td = "") {
  return R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="s" optype="categorical" dataType="string"/>
    <DataField name="i" optype="continuous" dataType="integer"/>
    <DataField name="d" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="double"/>
  </DataDictionary>
  <TransformationDictionary>)" +
         td + R"(</TransformationDictionary>
  <RegressionModel functionName="regression">
    <MiningSchema><MiningField name="s"/><MiningField name="i"/><MiningField name="d"/>
      <MiningField name="y" usageType="target"/></MiningSchema>
    <Output>
      <OutputField name="p" optype="continuous" dataType="double" feature="predictedValue"/>
      )" + fields +
         R"(
    </Output>
    <RegressionTable intercept="0"><NumericPredictor name="d" coefficient="1"/></RegressionTable>
  </RegressionModel>
</PMML>)";
}

// An OutputField o of `type` with feature transformedValue over `expression`.
static std::string transformed(const std::string& type, const std::string& expression,
                               const std::string& attrs = "") {
  return R"(<OutputField name="o" optype="continuous" dataType=")" + type + R"(" feature="transformedValue")" +
         attrs + ">" + expression + "</OutputField>";
}

// Design commit P2.5 (no silent fallbacks; §3.6: "Nothing is caught and
// turned into a silent default"). B13: an OutputField whose expression fails
// to load fails the load, as the same expression does in a DerivedField; the
// base build caught the ParsingException and left the field empty on every
// record. B25: an Apply's Extension children are not arguments (pmml-4-4.xsd
// Apply: Extension*, then EXPRESSION*); the base build refused them.
static void no_silent_fallbacks() {
  const auto output_error = [](const std::string& type, const std::string& expression) {
    return load_error(output_doc(transformed(type, expression)));
  };
  // Control: a valid transformedValue loads.
  CHECK(output_error("double", R"(<Apply function="+"><FieldRef field="d"/><Constant>1</Constant></Apply>)").empty());
  // B13. Each loads on the base build, the field empty. JPMML 1.7.7 refuses
  // each at its first evaluation (lazy): 'Function "noSuchFunction" is not
  // defined', NumberFormatException 'For input string: "abc"',
  // NumberFormatException, TypeCheckException (Aggregate).
  CHECK(contains(output_error("double", R"(<Apply function="noSuchFunction"><FieldRef field="d"/></Apply>)"),
                 "noSuchFunction"));
  // §3.4 NormDiscrete: "non-numeric text on a numeric field -> load error"
  // (P2.2's error, which the catch hid inside an OutputField).
  CHECK(contains(output_error("double", R"(<NormDiscrete field="d" value="abc"/>)"), "abc"));
  CHECK(contains(output_error("double", R"(<Constant dataType="integer">1.5</Constant>)"), "E-VALUE-MALFORMED"));
  CHECK(contains(output_error("double", R"(<Aggregate field="d" function="sum"/>)"), "Aggregate"));
  CHECK(contains(output_error("integer", R"(<MapValues outputColumn="o" dataType="integer">
        <FieldColumnPair field="i" column="k"/>
        <InlineTable><row><k>3</k><o>1</o></row><row><k>3.0</k><o>2</o></row></InlineTable></MapValues>)"),
                 "E-KEY-DUPLICATE"));
  // An inner expression of the OutputField's Apply fails the same way.
  CHECK(!output_error("double", R"(<Apply function="+"><FieldRef field="d"/>
        <Apply function="noSuchFunction"><FieldRef field="d"/></Apply></Apply>)")
             .empty());
  // Output.html: "For this result feature, OutputField must contain an
  // EXPRESSION, unless it is used to refer to a transformed value of a
  // segment model through the segmentID attribute." The base build loaded
  // both, the first empty on every record and the second a copy of the field
  // its `value` names (not defined by the spec: "The attribute value is used
  // in conjunction with result features referring to specific values").
  // JPMML 1.7.7: "Required element OutputField/<Expression> is not defined".
  CHECK(contains(load_error(output_doc(transformed("double", ""))), "must contain an EXPRESSION"));
  CHECK(contains(load_error(output_doc(transformed("double", "", R"( value="d")"))), "must contain an EXPRESSION"));
  CHECK(contains(load_error(output_doc(transformed("double", R"(<Extension name="e"/>)"))),
                 "must contain an EXPRESSION"));
  // segmentId is not supported (B34, P8.15).
  CHECK(contains(load_error(output_doc(transformed("double", "", R"( segmentId="1")"))), "segmentId"));

  // B25. JPMML 1.7.7 loads and scores each (fixture fix_apply_extension).
  const std::string ext = R"(<Extension extender="X" name="n" value="v"><X-Thing a="1"/></Extension>)";
  CHECK(load_error(output_doc("", R"(<DerivedField name="x" optype="continuous" dataType="double"><Apply function="+">)" +
                                      ext + R"(<FieldRef field="d"/><Constant>1</Constant></Apply></DerivedField>)"))
            .empty());
  CHECK(output_error("double", R"(<Apply function="+">)" + ext + ext +
                                   R"(<FieldRef field="d"/><Constant>1</Constant></Apply>)")
            .empty());
  // The Extension is not counted: an untyped Constant after it is at position
  // 0, if's boolean condition (S2), so TRUE there is not a boolean and true
  // is.
  CHECK(contains(output_error("double", R"(<Apply function="if">)" + ext +
                                            R"(<Constant>TRUE</Constant><Constant>1</Constant></Apply>)"),
                 "E-VALUE-MALFORMED"));
  CHECK(output_error("double", R"(<Apply function="if">)" + ext +
                                   R"(<Constant>true</Constant><Constant>1</Constant></Apply>)")
            .empty());
  // Out of the XSD's order (an Extension between arguments) it is skipped as
  // well, as the argument count and the S2 positions already skip it; JPMML
  // 1.7.7 scores it so.
  CHECK(output_error("double", R"(<Apply function="+"><FieldRef field="d"/>)" + ext +
                                   R"(<Constant>1</Constant></Apply>)")
            .empty());
}

// Design commit P2.7 (§2.7, review S5): ordering a string against a number,
// in either order, is a load error, E-TYPE-ORDER-MIXED. DataDictionary.html:
// "Categorical field values can only be tested for equality, while ordinal
// field values in addition have an order defined"; an ordinal string is
// ordered by its Value sequence, an order of strings, and the least
// restrictive type of a string and a number is string (Functions.html:
// "string double float integer"), whose text order is no order of numbers.
// §2.7 makes no carve-out for a number whose text is a declared Value.
// Each document loads on the base build (af0485b), which compares the
// string's code with the number. JPMML 1.7.7 refuses most at evaluation
// (lazily): "Categorical value cannot be used in comparison operations" for a
// categorical string field on the left, NumberFormatException for a number on
// the left or a non-numeric string Constant on the right of a number field,
// 'Values "10" and "4" cannot be ordered' for an ordinal string field against
// a number that is not one of its Values. Departures, the spec deciding: it
// orders a continuous string field against a number as text, an ordinal
// string field against a number that is one of its Values by the Value
// sequence, and a number field against a numeric-text string Constant as
// numbers.
static void ordering() {
  const auto output_error = [](const std::string& expression, const std::string& td = "") {
    return load_error(output_doc(transformed("boolean", expression), td));
  };
  const std::string mixed = "E-TYPE-ORDER-MIXED";
  // Every ordering built-in (typing.h is_ordering), either order. An untyped Constant "solely
  // of numbers" is an integer (Transformations.html).
  for (const std::string f : {"lessThan", "lessOrEqual", "greaterThan", "greaterOrEqual"}) {
    CHECK(contains(output_error("<Apply function=\"" + f + "\"><FieldRef field=\"s\"/><Constant>10</Constant></Apply>"),
                   mixed));
    CHECK(contains(output_error("<Apply function=\"" + f + "\"><Constant>10</Constant><FieldRef field=\"s\"/></Apply>"),
                   mixed));
  }
  // The message names the function and the types, as the binder's does.
  CHECK(contains(output_error(R"(<Apply function="lessThan"><FieldRef field="s"/><FieldRef field="i"/></Apply>)"),
                 "E-TYPE-ORDER-MIXED: Apply \"lessThan\" in "));
  CHECK(contains(output_error(R"(<Apply function="lessThan"><FieldRef field="s"/><FieldRef field="i"/></Apply>)"),
                 "orders string against integer"));
  CHECK(contains(output_error(R"(<Apply function="greaterThan"><FieldRef field="d"/><FieldRef field="s"/></Apply>)"),
                 "orders double against string"));
  // A string Constant, typed or inferred, against a number field.
  CHECK(contains(output_error(R"(<Apply function="lessOrEqual"><FieldRef field="d"/><Constant dataType="string">abc</Constant></Apply>)"),
                 mixed));
  CHECK(contains(output_error(R"(<Apply function="lessOrEqual"><FieldRef field="i"/><Constant>abc</Constant></Apply>)"),
                 mixed));
  CHECK(contains(output_error(R"(<Apply function="lessThan"><FieldRef field="s"/><Constant dataType="float">2.5</Constant></Apply>)"),
                 "orders string against float"));
  // A numeric-text string Constant on the right of a number field (JPMML
  // 1.7.7 orders it as a number: i = 3, 20, 1 give true, false, true).
  CHECK(contains(output_error(R"(<Apply function="lessThan"><FieldRef field="i"/><Constant dataType="string">10</Constant></Apply>)"),
                 "orders integer against string"));
  // A number-typed expression (a numeric Apply) and a string-typed one.
  CHECK(contains(output_error(R"(<Apply function="lessThan"><Apply function="uppercase"><FieldRef field="s"/></Apply>
        <Apply function="+"><FieldRef field="i"/><Constant>1</Constant></Apply></Apply>)"),
                 mixed));
  // A continuous string DerivedField: the optype gives no order to text
  // against a number (JPMML 1.7.7 orders the texts).
  const std::string sc =
      R"(<DerivedField name="sc" optype="continuous" dataType="string"><FieldRef field="s"/></DerivedField>)";
  CHECK(contains(output_error(R"(<Apply function="greaterThan"><FieldRef field="sc"/><FieldRef field="d"/></Apply>)", sc),
                 mixed));
  // An ordinal string field with Values against a number whose text is one
  // of them (JPMML 1.7.7 orders it by the Value sequence: so = 1, 10, 2 give
  // true, false, false); §2.7 has no carve-out.
  CHECK(contains(load_error(R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="so" optype="ordinal" dataType="string"><Value value="1"/><Value value="2"/><Value value="10"/></DataField>
    <DataField name="d" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="double"/>
  </DataDictionary>
  <TransformationDictionary>
    <DerivedField name="t" optype="categorical" dataType="boolean"><Apply function="lessThan"><FieldRef field="so"/><Constant>2</Constant></Apply></DerivedField>
  </TransformationDictionary>
  <RegressionModel functionName="regression">
    <MiningSchema><MiningField name="so"/><MiningField name="d"/><MiningField name="y" usageType="target"/></MiningSchema>
    <Output>
      <OutputField name="p" optype="continuous" dataType="double" feature="predictedValue"/>
      <OutputField name="o" optype="categorical" dataType="boolean" feature="transformedValue"><FieldRef field="t"/></OutputField>
    </Output>
    <RegressionTable intercept="0"><NumericPredictor name="d" coefficient="1"/></RegressionTable>
  </RegressionModel>
</PMML>)"),
                 "orders string against integer"));
  // In a DerivedField, nested in if, and in an OutputField's inner Apply.
  CHECK(contains(load_error(output_doc("", R"(<DerivedField name="t" optype="categorical" dataType="boolean">
        <Apply function="greaterOrEqual"><FieldRef field="i"/><FieldRef field="s"/></Apply></DerivedField>)")),
                 mixed));
  CHECK(contains(output_error(R"(<Apply function="if"><Apply function="lessThan"><FieldRef field="s"/><Constant>3</Constant></Apply>
        <Constant dataType="boolean">true</Constant><Constant dataType="boolean">false</Constant></Apply>)"),
                 mixed));
  // Controls that load. Equality of a string and a number (§2.7 D2: the
  // number as text), either order.
  CHECK(output_error(R"(<Apply function="equal"><FieldRef field="s"/><Constant>10</Constant></Apply>)").empty());
  CHECK(output_error(R"(<Apply function="notEqual"><Constant>3</Constant><FieldRef field="s"/></Apply>)").empty());
  // string x string ordering (Q6: a warning in report mode, D19).
  CHECK(output_error(R"(<Apply function="lessThan"><FieldRef field="s"/><Constant dataType="string">10</Constant></Apply>)")
            .empty());
  CHECK(output_error(R"(<Apply function="lessThan"><FieldRef field="sc"/><FieldRef field="s"/></Apply>)", sc).empty());
  // number x number in the promoted type (§2.5, D1).
  CHECK(output_error(R"(<Apply function="lessThan"><FieldRef field="i"/><FieldRef field="d"/></Apply>)").empty());
  CHECK(output_error(R"(<Apply function="greaterOrEqual"><Constant>10</Constant><FieldRef field="d"/></Apply>)").empty());
}

// Design commit P2.8b2 (§4.2): a built-in Apply with the wrong number of
// arguments is a load error, E-ARITY, naming the function. The arities are
// BuiltinFunctions.html's pseudo-declarations, written out here rather than
// read from the function table, so that a wrong row fails: "+" has
// <ParameterField name="a"/> and <ParameterField name="b"/>, "The functions
// -, *, and / are defined in the same way and have two parameters"; min, max,
// sum, avg, median and product take "a variable number of <FieldRef/>"
// (one or more, the table's reading); log10 "<ParameterField name="x"/>", "ln
// ..., sqrt ..., abs ..., exp ... are defined in the same way"; pow(x,y),
// threshold(x,y), modulo(x,y); floor "<ParameterField name="x"/>", ceil and
// round alike; isMissing and isNotMissing "<ParameterField name="input"/>";
// lessThan "<ParameterField name="x"/> <ParameterField name="y"/>" (equal,
// notEqual, lessOrEqual, greaterThan, greaterOrEqual "Result is dependent on
// applying either function to two input parameters"); not "<ParameterField
// name="x" dataType="boolean"/>"; isIn "<ParameterField name="x"/> The list
// takes a variable number of fields"; if "<ParameterField name="x"/>
// <ParameterField name="A"/> THEN part is required <ParameterField
// name="B"/> ELSE part is optional"; uppercase, lowercase, stringLength,
// trimBlanks one "input"; substring input, startPos, length; concat "two or
// more parameters"; replace input, pattern, replacement; matches input,
// pattern; normalCDF x, mu, sigma, "normalPDF(x, M, S)"; stdNormalCDF x,
// "stdNormalPDF(x)", erf "defined similar to stdNormalCDF"; expm1, ln1p, rint
// and the trigonometric functions x; hypot x, y. Each document loads on the
// base build (dd367e7), which checked the count on every record: every
// record failed ("evaluating apply function"), or was missing under
// invalidValueTreatment="asMissing", and concat with one argument and if with
// four scored. JPMML 1.7.7 refuses each at its first evaluation,
// InvalidArgumentListException 'Function "+" expects 2 values, got 1 values',
// except min with no argument, which it scores as missing (departure: the
// table's one or more).
static void arity() {
  struct Arity {
    const char* name;
    int min, max;  // max -1: variadic
  };
  static const Arity kSpec[] = {
      {"+", 2, 2},
      {"-", 2, 2},
      {"*", 2, 2},
      {"/", 2, 2},
      {"min", 1, -1},
      {"max", 1, -1},
      {"sum", 1, -1},
      {"avg", 1, -1},
      {"median", 1, -1},
      {"product", 1, -1},
      {"log10", 1, 1},
      {"ln", 1, 1},
      {"sqrt", 1, 1},
      {"abs", 1, 1},
      {"exp", 1, 1},
      {"pow", 2, 2},
      {"threshold", 2, 2},
      {"floor", 1, 1},
      {"ceil", 1, 1},
      {"round", 1, 1},
      {"modulo", 2, 2},
      {"isMissing", 1, 1},
      {"isNotMissing", 1, 1},
      {"equal", 2, 2},
      {"notEqual", 2, 2},
      {"lessThan", 2, 2},
      {"lessOrEqual", 2, 2},
      {"greaterThan", 2, 2},
      {"greaterOrEqual", 2, 2},
      {"not", 1, 1},
      {"and", 2, -1},
      {"or", 2, -1},
      {"isIn", 2, -1},
      {"isNotIn", 2, -1},
      {"if", 2, 3},
      {"uppercase", 1, 1},
      {"lowercase", 1, 1},
      {"stringLength", 1, 1},
      {"substring", 3, 3},
      {"trimBlanks", 1, 1},
      {"concat", 2, -1},
      {"replace", 3, 3},
      {"matches", 2, 2},
      {"normalCDF", 3, 3},
      {"normalPDF", 3, 3},
      {"stdNormalCDF", 1, 1},
      {"stdNormalPDF", 1, 1},
      {"erf", 1, 1},
      {"expm1", 1, 1},
      {"hypot", 2, 2},
      {"ln1p", 1, 1},
      {"rint", 1, 1},
      {"sin", 1, 1},
      {"asin", 1, 1},
      {"sinh", 1, 1},
      {"cos", 1, 1},
      {"acos", 1, 1},
      {"cosh", 1, 1},
      {"tan", 1, 1},
      {"atan", 1, 1},
      {"tanh", 1, 1},
      {"formatNumber", 2, 2},  // design item P4.9: input, pattern
  };
  const auto apply = [](const std::string& function, int n, const std::string& attrs = "") {
    std::string e = "<Apply function=\"" + function + "\"" + attrs + ">";
    for (int k = 0; k < n; k++) e += R"(<FieldRef field="d"/>)";
    return e + "</Apply>";
  };
  const auto output_error = [](const std::string& expression) {
    return load_error(output_doc(transformed("double", expression)));
  };
  const auto expected = [](const Arity& a) {
    return a.max < 0
               ? std::to_string(a.min) + " or more"
               : (a.min == a.max ? std::to_string(a.min) : std::to_string(a.min) + " to " + std::to_string(a.max));
  };
  for (const Arity& a : kSpec) {
    // One fewer than the least, and one more than the most (fixed arity).
    std::vector<int> wrong = {a.min - 1};
    if (a.max >= 0) wrong.push_back(a.max + 1);
    for (const int n : wrong) {
      const std::string error = output_error(apply(a.name, n));
      const std::string want = std::string("E-ARITY: Apply \"") + a.name + "\" in ";
      const std::string count = "has " + std::to_string(n) + " argument" + (n == 1 ? "" : "s") + "; " + a.name +
                                " takes " + expected(a) + " (BuiltinFunctions.html)";
      CHECK(contains(error, want) && contains(error, count));
      if (!(contains(error, want) && contains(error, count)))
        std::cerr << "  arity " << a.name << " with " << n << ": \"" << error << "\"\n";
    }
  }
  const std::string arity = "E-ARITY";
  // No argument at all, and three for a two-parameter function.
  CHECK(contains(output_error(apply("+", 0)), "has 0 arguments; + takes 2"));
  CHECK(contains(output_error(apply("+", 3)), "has 3 arguments; + takes 2"));
  // An Extension is not an argument (pmml-4-4.xsd Apply: Extension*, then
  // EXPRESSION*; B25): + with an Extension and one FieldRef has 1.
  const std::string ext = R"(<Extension extender="X" name="n" value="v"/>)";
  CHECK(contains(output_error(R"(<Apply function="+">)" + ext + R"(<FieldRef field="d"/></Apply>)"),
                 "has 1 argument; + takes 2"));
  CHECK(contains(output_error(R"(<Apply function="+">)" + ext + ext + R"(</Apply>)"), "has 0 arguments"));
  // Neither mapMissingTo nor invalidValueTreatment makes it a function.
  CHECK(contains(output_error(apply("+", 1, R"( mapMissingTo="7")")), arity));
  CHECK(contains(output_error(apply("+", 1, R"( invalidValueTreatment="asMissing")")), arity));
  // In a TransformationDictionary DerivedField, a LocalTransformations one,
  // nested in a valid Apply, and as if's condition.
  CHECK(contains(load_error(output_doc("", R"(<DerivedField name="t" optype="continuous" dataType="double">)" +
                                               apply("+", 1) + "</DerivedField>")),
                 "E-ARITY: Apply \"+\" in the TransformationDictionary has 1 argument"));
  CHECK(contains(load_error(R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="d" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="double"/>
  </DataDictionary>
  <RegressionModel functionName="regression">
    <MiningSchema><MiningField name="d"/><MiningField name="y" usageType="target"/></MiningSchema>
    <LocalTransformations><DerivedField name="t" optype="continuous" dataType="double">)" +
                            apply("abs", 2) + R"(</DerivedField></LocalTransformations>
    <RegressionTable intercept="0"><NumericPredictor name="t" coefficient="1"/></RegressionTable>
  </RegressionModel>
</PMML>)"),
                 "E-ARITY: Apply \"abs\" in RegressionModel"));
  CHECK(contains(output_error(R"(<Apply function="+"><FieldRef field="d"/>)" + apply("sqrt", 2) + "</Apply>"),
                 "Apply \"sqrt\""));
  CHECK(contains(output_error(R"(<Apply function="if">)" + apply("isMissing", 0) +
                              R"(<FieldRef field="d"/><FieldRef field="d"/></Apply>)"),
                 "Apply \"isMissing\""));
  // The count comes before the argument types: three arguments to lessThan,
  // one a string against numbers (E-TYPE-ORDER-MIXED at two).
  CHECK(contains(
      output_error(
          R"(<Apply function="lessThan"><FieldRef field="s"/><Constant>1</Constant><Constant>2</Constant></Apply>)"),
      "E-ARITY: Apply \"lessThan\""));
  // Controls that load: the least and the most, with Extensions around them.
  CHECK(output_error(apply("+", 2)).empty());
  CHECK(
      output_error(R"(<Apply function="+">)" + ext + R"(<FieldRef field="d"/><FieldRef field="d"/></Apply>)").empty());
  CHECK(output_error(apply("sum", 1)).empty());
  CHECK(output_error(apply("max", 5)).empty());
  CHECK(output_error(apply("isIn", 2)).empty());
  CHECK(
      output_error(R"(<Apply function="if"><Constant dataType="boolean">true</Constant><FieldRef field="d"/></Apply>)")
          .empty());
  CHECK(output_error(R"(<Apply function="if"><Constant dataType="boolean">true</Constant><FieldRef field="d"/>
        <FieldRef field="d"/></Apply>)")
            .empty());
  CHECK(load_error(output_doc(transformed("string", apply("concat", 2)))).empty());
  CHECK(load_error(output_doc(transformed("string", apply("concat", 4)))).empty());
}

// Design item P4.3c (B53): the condition of `if` is a boolean parameter.
// BuiltinFunctions.html: "<DefineFunction name="if"> <ParameterField name="x"
// dataType="boolean"/> ..."; Functions.html: "boolean dataTypes can not be
// mixed with other types", and "If the dataType for the ParameterField is
// more restrictive than the expression dataType, the PMML document is not
// valid" (Q7). A condition whose static type is a number or a string fails
// the load with E-TYPE-BOOLEAN-MIXED; the base build read a number other
// than 0 as true (integer 2, double 0.5) and a string by its code. An
// untyped condition is typed first (S2: an untyped Constant inherits
// boolean; D21: an untyped MapValues or Discretize is boolean when every
// value is boolean text) and loads. JPMML 1.7.7 refuses each number or
// string at evaluation, but accepts integer 0/1 values (dataType="integer"
// 1, a MapValues dataType="integer" of 1/0) and the string "true": a
// departure, the spec is followed.
static void if_condition() {
  const std::string td =
      R"(<DerivedField name="c" optype="categorical" dataType="boolean"><Apply function="greaterThan">)"
      R"(<FieldRef field="d"/><Constant>0</Constant></Apply></DerivedField>)"
      R"(<DerivedField name="di" optype="continuous" dataType="integer"><FieldRef field="i"/></DerivedField>)";
  const auto apply_if = [](const std::string& condition) {
    return R"(<Apply function="if">)" + condition +
           R"(<Constant dataType="string">T</Constant><Constant dataType="string">F</Constant></Apply>)";
  };
  const auto output_error = [&](const std::string& condition) {
    return load_error(output_doc(transformed("string", apply_if(condition)), td));
  };
  const auto mapvalues = [](const std::string& dt, const std::string& a, const std::string& b) {
    return R"(<MapValues outputColumn="out")" + dt +
           R"(><FieldColumnPair field="s" column="in"/><InlineTable><row><in>a</in><out>)" + a +
           "</out></row><row><in>b</in><out>" + b + "</out></row></InlineTable></MapValues>";
  };
  const auto discretize = [](const std::string& dt) {
    return R"(<Discretize field="d")" + dt +
           R"(><DiscretizeBin binValue="1"><Interval closure="openOpen" leftMargin="0"/></DiscretizeBin>)"
           R"(<DiscretizeBin binValue="0"><Interval closure="openClosed" rightMargin="0"/></DiscretizeBin></Discretize>)";
  };
  const std::string code = "E-TYPE-BOOLEAN-MIXED: Apply \"if\" in RegressionModel: argument 1 of if is ";
  const std::string declared = ", where the parameter is declared boolean";
  const auto refused = [&](const std::string& condition, const std::string& type) {
    const std::string error = output_error(condition);
    const bool ok = contains(error, code + type + declared);
    CHECK(ok);
    if (!ok) std::cerr << "  if condition " << condition << ": \"" << error << "\"\n";
  };
  // Constants of a declared number or string type, 0 and 1 included.
  refused(R"(<Constant dataType="integer">2</Constant>)", "integer");
  refused(R"(<Constant dataType="integer">1</Constant>)", "integer");
  refused(R"(<Constant dataType="integer">0</Constant>)", "integer");
  refused(R"(<Constant dataType="double">0.5</Constant>)", "double");
  refused(R"(<Constant dataType="float">-1</Constant>)", "float");
  refused(R"(<Constant dataType="string">true</Constant>)", "string");
  // Fields, a DerivedField, an Apply and the other expressions.
  refused(R"(<FieldRef field="i"/>)", "integer");
  refused(R"(<FieldRef field="d"/>)", "double");
  refused(R"(<FieldRef field="s"/>)", "string");
  refused(R"(<FieldRef field="di"/>)", "integer");
  refused(R"(<Apply function="+"><FieldRef field="i"/><Constant>0</Constant></Apply>)", "integer");
  refused(
      R"(<NormContinuous field="d"><LinearNorm orig="0" norm="0"/><LinearNorm orig="1" norm="1"/></NormContinuous>)",
      "double");
  refused(mapvalues(R"( dataType="integer")", "1", "0"), "integer");
  refused(discretize(R"( dataType="double")"), "double");
  // D21: an untyped MapValues with a value that is not boolean text takes the
  // enclosing OutputField's string.
  refused(mapvalues("", "2", "0"), "string");
  // An `if` of integer branches as the condition of another.
  refused(R"(<Apply function="if"><FieldRef field="c"/><Constant>1</Constant><Constant>0</Constant></Apply>)",
          "integer");
  // Untyped "true" and "false" as THEN and ELSE are strings (Transformations.html:
  // "The presence of any non-numeric characters ... will result in the Constant
  // being treated as a string"; if's A and B declare no dataType), so the inner
  // if is a string. The base build read its text by code: false for "true".
  refused(R"(<Apply function="if"><FieldRef field="c"/><Constant>true</Constant><Constant>false</Constant></Apply>)",
          "string");
  // An Extension before the condition is not the condition (B25).
  refused(R"(<Extension extender="X" name="n" value="v"/><Constant dataType="integer">2</Constant>)", "integer");
  // In a TransformationDictionary DerivedField.
  CHECK(contains(load_error(output_doc("", R"(<DerivedField name="t" optype="categorical" dataType="string">)" +
                                               apply_if(R"(<FieldRef field="d"/>)") + "</DerivedField>")),
                 "E-TYPE-BOOLEAN-MIXED: Apply \"if\" in the TransformationDictionary: argument 1 of if is double"));
  // Controls: boolean conditions load and take THEN for true, ELSE for false.
  const auto scored = [&](const std::string& condition, const std::string& s, const std::string& d) {
    cpmml::Input input;
    input["s"] = s;
    input["i"] = "1";
    input["d"] = d;
    try {
      const cpmml::Model model = load(output_doc(transformed("string", apply_if(condition)), td));
      return model.evaluate(input).str_outputs()["o"];
    } catch (const std::exception& e) {
      return std::string("ERROR: ") + e.what();
    }
  };
  CHECK(scored(R"(<FieldRef field="c"/>)", "a", "1") == "T");
  CHECK(scored(R"(<FieldRef field="c"/>)", "a", "-1") == "F");
  CHECK(scored(R"(<Apply function="lessThan"><FieldRef field="d"/><Constant>0</Constant></Apply>)", "a", "-1") == "T");
  CHECK(scored(R"(<Constant dataType="boolean">0</Constant>)", "a", "1") == "F");
  // S2: an untyped Constant inherits boolean ("1", "0", "true").
  CHECK(scored("<Constant>1</Constant>", "a", "1") == "T");
  CHECK(scored("<Constant>0</Constant>", "a", "1") == "F");
  CHECK(scored("<Constant>true</Constant>", "a", "1") == "T");
  // D21: untyped MapValues and Discretize whose values are boolean text.
  CHECK(scored(mapvalues("", "1", "0"), "a", "1") == "T");
  CHECK(scored(mapvalues("", "1", "0"), "b", "1") == "F");
  CHECK(scored(mapvalues("", "true", "false"), "b", "1") == "F");
  CHECK(scored(mapvalues(R"( dataType="boolean")", "1", "0"), "a", "1") == "T");
  CHECK(scored(discretize(""), "a", "0.5") == "T");
  CHECK(scored(discretize(""), "a", "-0.5") == "F");
  CHECK(scored(R"(<Apply function="if"><FieldRef field="c"/><Constant dataType="boolean">true</Constant>)"
               R"(<Constant dataType="boolean">false</Constant></Apply>)",
               "a", "-1") == "F");
  // An untyped Constant that is not boolean text was a load error already
  // (E-VALUE-MALFORMED, S2), and a number branch is not a condition.
  CHECK(contains(output_error("<Constant>2</Constant>"), "E-VALUE-MALFORMED"));
  CHECK(load_error(output_doc(transformed("double", R"(<Apply function="if"><FieldRef field="c"/><FieldRef field="i"/>)"
                                                    R"(<FieldRef field="d"/></Apply>)"),
                              td))
            .empty());
}

// Design item P4.8 (the rest of B53): not, and and or take boolean arguments.
// BuiltinFunctions.html, not: "<DefineFunction name="not" ...>
// <ParameterField name="x" dataType="boolean"/>"; and, or: "Evaluate the
// results of two or more boolean values"; Functions.html: "boolean dataTypes
// can not be mixed with other types", and "If the dataType for the
// ParameterField is more restrictive than the expression dataType, the PMML
// document is not valid" (Q7). An argument whose static type is a number or a
// string fails the load with E-TYPE-BOOLEAN-MIXED, as if's condition does
// (P4.3c, D36); the base build scored not(FieldRef integer) (false for 2, true
// for 0) and refused and/or as unsupported. S2 differs between them: not's
// ParameterField declares boolean, so an untyped Constant 1 inherits it; and
// and or have no ParameterField (bind/typing.h parameter_type), so an untyped
// "1" is an integer and "true" a string, both refused. JPMML 1.7.7 refuses
// each number or string at evaluation (TypeCheckException "Expected boolean
// value, got integer value") but accepts the values 1/0 and the strings
// "true"/"false": a departure (D36's), the spec is followed.
static void boolean_arguments() {
  const std::string td =
      R"(<DerivedField name="c" optype="categorical" dataType="boolean"><Apply function="greaterThan">)"
      R"(<FieldRef field="d"/><Constant>0</Constant></Apply></DerivedField>)"
      R"(<DerivedField name="di" optype="continuous" dataType="integer"><FieldRef field="i"/></DerivedField>)";
  const std::string c = R"(<FieldRef field="c"/>)";
  const auto call = [](const std::string& function, const std::string& args) {
    return R"(<Apply function=")" + function + R"(">)" + args + "</Apply>";
  };
  const auto output_error = [&](const std::string& expression) {
    return load_error(output_doc(transformed("boolean", expression), td));
  };
  const auto refused = [&](const std::string& expression, const std::string& function, int position,
                           const std::string& type) {
    const std::string error = output_error(expression);
    const std::string want = "E-TYPE-BOOLEAN-MIXED: Apply \"" + function + "\" in RegressionModel: argument " +
                             std::to_string(position) + " of " + function + " is " + type +
                             ", where the parameter is declared boolean";
    const bool ok = contains(error, want);
    CHECK(ok);
    if (!ok) std::cerr << "  " << expression << ": \"" << error << "\"\n";
  };
  const std::string mapvalues_int =
      R"(<MapValues outputColumn="out" dataType="integer"><FieldColumnPair field="s" column="in"/>)"
      R"(<InlineTable><row><in>a</in><out>1</out></row><row><in>b</in><out>0</out></row></InlineTable></MapValues>)";
  // not: fields, typed Constants (0 and 1 included), an Apply, a MapValues.
  refused(call("not", R"(<FieldRef field="i"/>)"), "not", 1, "integer");
  refused(call("not", R"(<FieldRef field="d"/>)"), "not", 1, "double");
  refused(call("not", R"(<FieldRef field="s"/>)"), "not", 1, "string");
  refused(call("not", R"(<FieldRef field="di"/>)"), "not", 1, "integer");
  refused(call("not", R"(<Constant dataType="integer">1</Constant>)"), "not", 1, "integer");
  refused(call("not", R"(<Constant dataType="integer">0</Constant>)"), "not", 1, "integer");
  refused(call("not", R"(<Constant dataType="string">true</Constant>)"), "not", 1, "string");
  refused(call("not", R"(<Apply function="+"><FieldRef field="i"/><Constant>0</Constant></Apply>)"), "not", 1,
          "integer");
  refused(call("not", mapvalues_int), "not", 1, "integer");
  refused(call("not", R"(<Extension extender="X" name="n" value="v"/><FieldRef field="i"/>)"), "not", 1, "integer");
  // and, or: every argument is checked, at its position after Extension*.
  refused(call("and", c + R"(<FieldRef field="i"/>)"), "and", 2, "integer");
  refused(call("and", R"(<FieldRef field="d"/>)" + c), "and", 1, "double");
  refused(call("or", c + c + R"(<FieldRef field="s"/>)"), "or", 3, "string");
  refused(call("or", c + R"(<Extension extender="X" name="n" value="v"/><Constant dataType="integer">1</Constant>)"),
          "or", 2, "integer");
  refused(call("and", c + R"(<Constant dataType="string">true</Constant>)"), "and", 2, "string");
  refused(call("or", c + mapvalues_int), "or", 2, "integer");
  refused(call("and", c + R"(<Apply function="if">)" + c + "<Constant>1</Constant><Constant>0</Constant></Apply>"),
          "and", 2, "integer");
  // S2: and and or declare no ParameterField, so untyped Constants are inferred.
  refused(call("and", c + "<Constant>1</Constant>"), "and", 2, "integer");
  refused(call("or", "<Constant>true</Constant>" + c), "or", 1, "string");
  // In a TransformationDictionary DerivedField.
  CHECK(contains(load_error(output_doc("", td + R"(<DerivedField name="t" optype="categorical" dataType="boolean">)" +
                                               call("or", c + R"(<FieldRef field="d"/>)") + "</DerivedField>")),
                 "E-TYPE-BOOLEAN-MIXED: Apply \"or\" in the TransformationDictionary: argument 2 of or is double"));
  // Controls: boolean arguments load and score; the result is read through
  // if(result, "true", "false"), a string OutputField (a missing result, a
  // missing condition, gives no "o").
  const auto scored = [&](const std::string& expression, const std::string& s, const std::string& d) {
    cpmml::Input input;
    input["s"] = s;
    input["i"] = "1";
    if (!d.empty()) input["d"] = d;
    try {
      const cpmml::Model model = load(output_doc(
          transformed("string", R"(<Apply function="if">)" + expression +
                                    R"(<Constant dataType="string">true</Constant><Constant dataType="string">false</Constant></Apply>)"),
          td));
      const auto out = model.evaluate(input).str_outputs();
      const auto it = out.find("o");
      return it == out.end() ? std::string("<missing>") : it->second;
    } catch (const std::exception& e) {
      return std::string("ERROR: ") + e.what();
    }
  };
  const std::string t = R"(<Constant dataType="boolean">true</Constant>)";
  const std::string f = R"(<Constant dataType="boolean">false</Constant>)";
  CHECK(scored(call("not", c), "a", "1") == "false");
  CHECK(scored(call("not", c), "a", "-1") == "true");
  // S2: not's parameter is declared boolean, so an untyped 1 or false is one.
  CHECK(scored(call("not", "<Constant>1</Constant>"), "a", "1") == "false");
  CHECK(scored(call("not", "<Constant>false</Constant>"), "a", "1") == "true");
  CHECK(scored(call("and", c + t), "a", "1") == "true");
  CHECK(scored(call("and", c + t), "a", "-1") == "false");
  CHECK(scored(call("and", t + t + t + t + c), "a", "1") == "true");  // five: ArgBuffer on the heap
  CHECK(scored(call("and", t + t + f + t + c), "a", "1") == "false");
  CHECK(scored(call("or", c + f), "a", "1") == "true");
  CHECK(scored(call("or", c + f), "a", "-1") == "false");
  CHECK(scored(call("or", f + f + f + f + c), "a", "1") == "true");
  CHECK(scored(call("or", c + R"(<Apply function="isMissing"><FieldRef field="s"/></Apply>)"), "a", "-1") == "false");
  CHECK(scored(call("not", call("or", c + call("and", c + f))), "a", "-1") == "true");
  // A missing input gives missing ("any missing inputs to a built-in function
  // will result in a missing value being returned"), with no three-valued
  // logic: or(missing, true) and and(missing, false) are missing, and
  // mapMissingTo is returned instead.
  CHECK(scored(call("or", c + t), "a", "") == "<missing>");
  CHECK(scored(call("and", c + f), "a", "") == "<missing>");
  CHECK(scored(R"(<Apply function="and" mapMissingTo="true">)" + c + f + "</Apply>", "a", "") == "true");
  CHECK(scored(R"(<Apply function="or" defaultValue="false">)" + c + t + "</Apply>", "a", "") == "false");
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

// Design item P2.8b3: the compiled surrogate (core/fastpredicate.h) caught
// every cpmml::Exception an operand threw and went on to the next operand, so
// an error behaved as UNKNOWN. TreeModel.html: "The cascaded predicates are
// applied when the primary predicate evaluates to UNKNOWN", "An expression
// surrogate(a,b) is equivalent to if not unknown(a) then a else b", and "A
// SimplePredicate <Field> <Operator> <Value> evaluates to UNKNOWN if the value
// of Field is missing"; an error is none of these. JPMML 1.7.7, probed with a
// TreeModel whose Node predicate is surrogate(d < 5, y < 5), d = ln(x), y = 1:
// x = -1 throws 'Function "ln" returned invalid value' and a missing x
// 'Function "ln" cannot accept missing value', inside the surrogate exactly as
// with the plain predicate d < 5, never taking y < 5. No operand cPMML
// compiles throws per record today (a missing input is UNKNOWN, everything
// else is a load error), so the operands that throw here are written for the
// test.
static void surrogate_errors() {
  using R = PredicateResult;
  using Fn = std::function<R(const Sample&)>;
  using Op = PredicateOpType::PredicateOpTypeValue;
  // The compound operators of BoundPredicates and Predicate
  // (cpmml::predicate::combine), over operands written for the test.
  const auto compound = [](Op op, const Sample& s, const std::vector<Fn>& operands) {
    return cpmml::predicate::combine(op, operands.size(), [&](std::size_t i) { return operands[i](s); });
  };
  const auto surrogate = [&](const Sample& s, const std::vector<Fn>& operands) {
    return compound(Op::SURROGATE, s, operands);
  };
  const auto _and = [&](const Sample& s, const std::vector<Fn>& operands) { return compound(Op::AND, s, operands); };
  const auto _or = [&](const Sample& s, const std::vector<Fn>& operands) { return compound(Op::OR, s, operands); };
  const Sample sample(1);
  int evaluated = 0;
  const Fn unknown = [&](const Sample&) {
    ++evaluated;
    return R::UNKNOWN;
  };
  const Fn yes = [&](const Sample&) {
    ++evaluated;
    return R::TRUE_;
  };
  const Fn no = [&](const Sample&) {
    ++evaluated;
    return R::FALSE_;
  };
  const Fn missing_error = [&](const Sample&) -> R {
    ++evaluated;
    throw cpmml::MissingValueException("operand error");
  };
  const Fn invalid_error = [&](const Sample&) -> R {
    ++evaluated;
    throw cpmml::InvalidValueException("operand invalid");
  };
  const auto message = [&](const std::vector<Fn>& operands) -> std::string {
    try {
      surrogate(sample, operands);
    } catch (const cpmml::Exception& e) {
      return e.what();
    }
    return "";
  };

  // An error from the primary ends the evaluation, with the operand's own
  // exception and message, and no surrogate is tried.
  evaluated = 0;
  CHECK(throws<cpmml::MissingValueException>([&] { surrogate(sample, {missing_error, yes}); }));
  CHECK(evaluated == 1);
  CHECK(message({missing_error, yes}) == "operand error");
  CHECK(throws<cpmml::InvalidValueException>([&] { surrogate(sample, {invalid_error, no}); }));
  CHECK(message({invalid_error, no}) == "operand invalid");
  // The same error as the operand gives alone.
  CHECK(throws<cpmml::MissingValueException>([&] { missing_error(sample); }));
  // An error from a surrogate reached after an UNKNOWN primary, and from the
  // last one.
  evaluated = 0;
  CHECK(throws<cpmml::InvalidValueException>([&] { surrogate(sample, {unknown, invalid_error, yes}); }));
  CHECK(evaluated == 2);
  CHECK(throws<cpmml::MissingValueException>([&] { surrogate(sample, {unknown, unknown, missing_error}); }));
  // Through the enclosing compound predicates.
  const Fn inner = [&](const Sample& s) { return surrogate(s, {missing_error, yes}); };
  CHECK(throws<cpmml::MissingValueException>([&] { _and(sample, {yes, inner}); }));
  CHECK(throws<cpmml::MissingValueException>([&] { _or(sample, {no, inner}); }));
  CHECK(throws<cpmml::MissingValueException>([&] { surrogate(sample, {unknown, inner, yes}); }));

  // Controls: a result that is not UNKNOWN wins and the operands after it are
  // not evaluated, an error among them included.
  evaluated = 0;
  CHECK(surrogate(sample, {yes, missing_error}) == R::TRUE_);
  CHECK(surrogate(sample, {no, missing_error}) == R::FALSE_);
  CHECK(evaluated == 2);
  CHECK(surrogate(sample, {unknown, yes, missing_error}) == R::TRUE_);
  CHECK(surrogate(sample, {unknown, no}) == R::FALSE_);
  CHECK(surrogate(sample, {unknown, unknown}) == R::UNKNOWN);
  // Compiled from Predicate: a missing input is UNKNOWN, not an error.
  Sample two(2);
  two.change_value(1, Value(3.0));
  const Predicate lt0 = Predicate::simple(0, PredicateOpType::PredicateOpTypeValue::LESS_THAN, Value(5.0));
  const Predicate lt1 = Predicate::simple(1, PredicateOpType::PredicateOpTypeValue::LESS_THAN, Value(5.0));
  const Predicate gt1 = Predicate::simple(1, PredicateOpType::PredicateOpTypeValue::GREATER_THAN, Value(5.0));
  BoundPredicates bound;
  const std::uint32_t fast = bound.add(Predicate({lt0, gt1}, PredicateOpType::PredicateOpTypeValue::SURROGATE));
  const std::uint32_t lt0_lt1 = bound.add(Predicate({lt0, lt1}, PredicateOpType::PredicateOpTypeValue::SURROGATE));
  const std::uint32_t lt0_lt0 = bound.add(Predicate({lt0, lt0}, PredicateOpType::PredicateOpTypeValue::SURROGATE));
  CHECK(bound.evaluate(fast, two) == R::FALSE_);
  CHECK(bound.evaluate(lt0_lt1, two) == R::TRUE_);
  CHECK(bound.evaluate(lt0_lt0, two) == R::UNKNOWN);
  two.change_value(0, Value(1.0));
  CHECK(bound.evaluate(fast, two) == R::TRUE_);
}

// Design items P5, P6.3: a tree's BoundPredicates answer exactly what the
// Predicate they are compiled from answers (Predicate::evaluate_typed), for
// every operator and every kind of operand and value: a pooled string, one
// the pool lacks, an edit of a pooled string (the string of its new text since
// P6.1: pooled ABC, or unseen ABCD), numbers (a number equal to a string's
// pool code included), ±INF and NaN, a boolean and a boolean result held as
// the Double 1 or 0, missing; a pair of no common type (a string ordered
// against a number, a boolean against a number) throws on both paths (the
// binder reads a predicate's value in its field's type, so it is a bug). Then
// the typed rules themselves (§2.7, P6.3): a number against a string by its
// Java text (D2), never by the string's code (B18); integer x float in float
// (D1); ±INF against the same infinity and NaN UNKNOWN (D9, B44); an ordinal
// string's declared order.
static void bound_predicates() {
  using Op = PredicateOpType::PredicateOpTypeValue;
  using R = PredicateResult;
  auto document = std::make_shared<cpmml::DocumentStrings>();
  const cpmml::StringScope load(cpmml::StringScope::Loading{}, document);
  const Value abc = Value::string("abc");
  const Value xyz = Value::string("xyz");
  const Value pooled_up = Value::string("ABC");
  const Value text25 = Value::string("2.5");
  const Value low = Value::string("low");
  const Value mid = Value::string("mid");
  const Value high = Value::string("high");
  const auto order = std::make_shared<const cpmml::compare::OrdinalOrder>(std::vector<std::string>{"low", "mid", "high"});
  document->pool.freeze();
  const cpmml::StringScope score(cpmml::StringScope::Scoring{}, *document);
  Value edited = abc;
  edited.uppercase();
  Value edited_unseen = Value::string("abcd");
  edited_unseen.uppercase();
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<Value> operands = {abc,
                                       xyz,
                                       Value::string("never named"),
                                       Value::string("another"),
                                       edited,
                                       edited_unseen,
                                       Value(cpmml::legacy::code_of(abc)),
                                       Value(-1.0),
                                       Value(2.5),
                                       Value(cpmml::types::Integer{3}),
                                       Value(cpmml::types::Float{2.5f}),
                                       Value(inf),
                                       Value(-inf),
                                       Value(nan),
                                       Value(cpmml::types::Boolean{true}),
                                       Value(1.0),
                                       Value()};
  const std::vector<Value> constants = {abc,      xyz,        text25, Value(cpmml::legacy::code_of(xyz)), Value(-1.0),
                                        Value(2.5), Value(inf), Value(nan), Value(cpmml::types::Integer{3}),
                                        Value(cpmml::types::Float{2.5f}), Value(cpmml::types::Boolean{true})};
  // The answer, or 3 for a throw (no common type).
  const auto answer = [](const auto& f) -> int {
    try {
      return static_cast<int>(f());
    } catch (const std::logic_error&) {
      return 3;
    }
  };
  int compared = 0, thrown = 0;
  for (const Op op : {Op::EQUAL, Op::NOT_EQUAL, Op::LESS_THAN, Op::LESS_OR_EQUAL, Op::GREATER_THAN,
                      Op::GREATER_OR_EQUAL}) {
    for (const Value& c : constants) {
      BoundPredicates bound;
      const Predicate tree = Predicate::simple(0, op, c);
      const std::uint32_t i = bound.add(tree);
      for (const Value& x : operands) {
        Sample sample(1);
        sample.change_value(0, x);
        const int a = answer([&] { return bound.evaluate(i, sample); });
        CHECK(a == answer([&] { return tree.evaluate(sample); }));
        thrown += a == 3;
        compared++;
      }
    }
  }
  CHECK(compared == 6 * 11 * 17 && thrown > 0);
  // The expected answers of the rules themselves.
  BoundPredicates bound;
  const std::uint32_t eq_abc = bound.add(Predicate::simple(0, Op::EQUAL, abc));
  const std::uint32_t ne_abc = bound.add(Predicate::simple(0, Op::NOT_EQUAL, abc));
  const std::uint32_t eq_up = bound.add(Predicate::simple(0, Op::EQUAL, pooled_up));
  const std::uint32_t eq_t25 = bound.add(Predicate::simple(0, Op::EQUAL, text25));
  const std::uint32_t eq_inf = bound.add(Predicate::simple(0, Op::EQUAL, Value(inf)));
  const std::uint32_t le_inf = bound.add(Predicate::simple(0, Op::LESS_OR_EQUAL, Value(inf)));
  const std::uint32_t lt_inf = bound.add(Predicate::simple(0, Op::LESS_THAN, Value(inf)));
  const std::uint32_t gt_1 = bound.add(Predicate::simple(0, Op::GREATER_THAN, Value(1.0)));
  const std::uint32_t ne_1 = bound.add(Predicate::simple(0, Op::NOT_EQUAL, Value(1.0)));
  const std::uint32_t eq_f = bound.add(Predicate::simple(0, Op::EQUAL, Value(cpmml::types::Float{16777216.0f})));
  const std::uint32_t eq_true = bound.add(Predicate::simple(0, Op::EQUAL, Value(cpmml::types::Boolean{true})));
  const std::uint32_t lt_mid = bound.add(Predicate::simple(0, Op::LESS_THAN, mid, order));
  const std::uint32_t ge_mid = bound.add(Predicate::simple(0, Op::GREATER_OR_EQUAL, mid, order));
  const std::uint32_t lt_mid_units = bound.add(Predicate::simple(0, Op::LESS_THAN, mid));
  Sample s(1);
  const auto at = [&](const Value& v, std::uint32_t i) {
    s.change_value(0, v);
    return bound.evaluate(i, s);
  };
  CHECK(at(edited, eq_abc) == R::FALSE_);  // P6.1, B2: uppercase(abc) is not abc
  CHECK(at(edited, ne_abc) == R::TRUE_);
  CHECK(at(edited, eq_up) == R::TRUE_);  // P6.1, B2: it is ABC
  CHECK(at(edited_unseen, eq_abc) == R::FALSE_ && at(edited_unseen, eq_up) == R::FALSE_);
  CHECK(at(Value::string("never named"), eq_abc) == R::FALSE_);  // a text the pool lacks equals no value
  CHECK(at(Value::string("never named"), ne_abc) == R::TRUE_);
  // B18 (P6.3): the number of abc's code is not abc; a number is its Java text (D2).
  CHECK(at(Value(cpmml::legacy::code_of(abc)), eq_abc) == R::FALSE_);
  CHECK(at(Value(cpmml::legacy::code_of(abc)), ne_abc) == R::TRUE_);
  CHECK(at(Value(2.5), eq_t25) == R::TRUE_ && at(Value(cpmml::types::Float{2.5f}), eq_t25) == R::TRUE_);
  CHECK(at(Value(cpmml::types::Integer{2}), eq_t25) == R::FALSE_);
  // D9, B44: INF against INF and NaN against anything are UNKNOWN.
  CHECK(at(Value(inf), eq_inf) == R::UNKNOWN && at(Value(inf), le_inf) == R::UNKNOWN &&
        at(Value(inf), lt_inf) == R::UNKNOWN);
  CHECK(at(Value(-inf), eq_inf) == R::FALSE_ && at(Value(-inf), lt_inf) == R::TRUE_ && at(Value(5.0), le_inf) == R::TRUE_);
  CHECK(at(Value(inf), gt_1) == R::TRUE_ && at(Value(-inf), gt_1) == R::FALSE_);
  CHECK(at(Value(nan), gt_1) == R::UNKNOWN && at(Value(nan), ne_1) == R::UNKNOWN && at(Value(nan), eq_inf) == R::UNKNOWN);
  CHECK(at(Value(cpmml::types::Float{std::numeric_limits<float>::infinity()}), eq_inf) == R::UNKNOWN);
  // D1: integer x float compares in float (16777217 is the float 16777216).
  CHECK(at(Value(cpmml::types::Integer{16777217}), eq_f) == R::TRUE_);
  CHECK(at(Value(cpmml::types::Integer{16777218}), eq_f) == R::FALSE_);
  // Two integers compare exactly, beyond 2^53 too (not in double).
  {
    BoundPredicates b3;
    const std::uint32_t eq_big = b3.add(Predicate::simple(0, Op::EQUAL, Value(cpmml::types::Integer{9007199254740992})));
    const std::uint32_t lt_big = b3.add(Predicate::simple(0, Op::LESS_THAN, Value(cpmml::types::Integer{9007199254740993})));
    Sample t(1);
    t.change_value(0, Value(cpmml::types::Integer{9007199254740993}));
    CHECK(b3.evaluate(eq_big, t) == R::FALSE_ && b3.evaluate(lt_big, t) == R::FALSE_);
    t.change_value(0, Value(cpmml::types::Integer{9007199254740992}));
    CHECK(b3.evaluate(eq_big, t) == R::TRUE_ && b3.evaluate(lt_big, t) == R::TRUE_);
  }
  // A boolean, and a boolean result held as the Double 1 or 0.
  CHECK(at(Value(cpmml::types::Boolean{true}), eq_true) == R::TRUE_ && at(Value(1.0), eq_true) == R::TRUE_ &&
        at(Value(0.0), eq_true) == R::FALSE_);
  CHECK(throws<std::logic_error>([&] { at(Value(2.0), eq_true); }));  // no boolean: no common type
  // §2.7: an ordinal string's declared order; without one, code units.
  CHECK(at(low, lt_mid) == R::TRUE_ && at(high, lt_mid) == R::FALSE_ && at(mid, lt_mid) == R::FALSE_);
  CHECK(at(high, ge_mid) == R::TRUE_ && at(low, ge_mid) == R::FALSE_);
  CHECK(at(Value::string("never named"), lt_mid) == R::UNKNOWN);  // outside the Values: no place in the order
  CHECK(at(high, lt_mid_units) == R::TRUE_ && at(xyz, lt_mid_units) == R::FALSE_);
  // L4: two strings the pool lacks are not equal and not ordered equal.
  {
    BoundPredicates b2;
    const Value a1 = Value::string("aaa unseen");
    const std::uint32_t le = b2.add(Predicate::simple(0, Op::LESS_OR_EQUAL, abc));
    const std::uint32_t ge = b2.add(Predicate::simple(0, Op::GREATER_OR_EQUAL, abc));
    Sample t(1);
    t.change_value(0, a1);
    CHECK(b2.evaluate(le, t) == R::TRUE_ && b2.evaluate(ge, t) == R::FALSE_);
  }
  CHECK(at(Value(), eq_abc) == R::UNKNOWN && at(Value(), ne_abc) == R::UNKNOWN && at(Value(), eq_inf) == R::UNKNOWN);
  // The untyped Predicate (an Interval's margins) stays IEEE: INF <= INF.
  CHECK(Predicate(0, Op::LESS_OR_EQUAL, Value(inf)).evaluate(Value(inf)) == R::TRUE_);
  CHECK(throws<std::logic_error>([&] { BoundPredicates().add(Predicate(0, Op::EQUAL, Value(1.0))); }));
  // Sets, both containers (SET_THRESHOLD), against the tree.
  for (const bool hashed : {false, true}) {
    for (const Op op : {Op::IS_IN, Op::IS_NOT_IN}) {
      for (const std::vector<Value>& members :
           {std::vector<Value>{abc, text25}, std::vector<Value>{Value(2.5), Value(inf)},
            std::vector<Value>{Value(cpmml::types::Float{2.5f})}}) {
        BoundPredicates sets;
        const Predicate tree = Predicate::simple_set(0, op, members, hashed);
        const std::uint32_t i = sets.add(tree);
        for (const Value& x : operands) {
          if (std::holds_alternative<cpmml::types::Boolean>(x.typed())) continue;  // no common type
          Sample sample(1);
          sample.change_value(0, x);
          CHECK(answer([&] { return sets.evaluate(i, sample); }) == answer([&] { return tree.evaluate(sample); }));
        }
      }
      // The rules: a number against strings by its text (D2), INF against a
      // set holding INF and NaN UNKNOWN (D9), INF against one without FALSE.
      BoundPredicates sets;
      const std::uint32_t strings = sets.add(Predicate::simple_set(0, op, {abc, text25}, hashed));
      const std::uint32_t numbers = sets.add(Predicate::simple_set(0, op, {Value(2.5), Value(inf)}, hashed));
      const std::uint32_t finite = sets.add(Predicate::simple_set(0, op, {Value(2.5)}, hashed));
      const R yes = op == Op::IS_IN ? R::TRUE_ : R::FALSE_;
      const R no = op == Op::IS_IN ? R::FALSE_ : R::TRUE_;
      Sample t(1);
      const auto in = [&](const Value& v, std::uint32_t i) {
        t.change_value(0, v);
        return sets.evaluate(i, t);
      };
      CHECK(in(Value(2.5), strings) == yes && in(Value(cpmml::legacy::code_of(abc)), strings) == no);
      CHECK(in(Value::string("2.5"), numbers) == yes && in(Value::string("2.50"), numbers) == no);
      CHECK(in(Value(inf), numbers) == R::UNKNOWN && in(Value(nan), numbers) == R::UNKNOWN);
      CHECK(in(Value(inf), finite) == no && in(Value(nan), finite) == R::UNKNOWN && in(Value(-inf), numbers) == no);
      CHECK(in(Value(2.5), numbers) == yes && in(Value(), numbers) == R::UNKNOWN);
      // Two integers compare exactly, beyond 2^53 too, where one double key
      // holds 2^53 and 2^53 + 1: a big member, or a big probe.
      const Value big0(cpmml::types::Integer{9007199254740992});
      const Value big1(cpmml::types::Integer{9007199254740993});
      const std::uint32_t with_big1 = sets.add(Predicate::simple_set(0, op, {big1, Value(cpmml::types::Integer{5})}, hashed));
      const std::uint32_t with_big0 = sets.add(Predicate::simple_set(0, op, {big0, Value(cpmml::types::Integer{5})}, hashed));
      CHECK(in(big0, with_big1) == no && in(big1, with_big1) == yes && in(Value(cpmml::types::Integer{5}), with_big1) == yes);
      CHECK(in(big1, with_big0) == no && in(big0, with_big0) == yes);
      CHECK(in(Value(cpmml::types::Integer{-9007199254740993}), with_big0) == no);
      CHECK(in(Value(cpmml::types::Integer{5}), with_big0) == yes && in(Value(cpmml::types::Integer{6}), with_big0) == no);
    }
  }
}

// P6.3 (§2.7, Q7): a comparison of a boolean with a value of another type has
// no common type, a load error in the binder's words (E-TYPE-BOOLEAN-MIXED).
// Functions.html: "The various date, time and dateTime as well as boolean
// dataTypes can not be mixed with other types in a defined function without
// explicitly defining the expected output dataType". These are the three
// cells fix_typed_equal_mixed had (e_b_ct, e_b_c1, e_s_b) and their
// neighbours; each document loads on base 49847ca, which compared the
// boolean's 1 or 0 with the other value's double (a string's pool code).
// JPMML 1.7.7 converts the second operand to the first's type (equal(b,
// "true") and equal(b, 1) true for b = true, equal(s, b) by text).
static void boolean_mixed() {
  const auto doc = [](const std::string& expression) {
    return std::string(R"(<?xml version="1.0" encoding="UTF-8"?>
<PMML xmlns="http://www.dmg.org/PMML-4_4" version="4.4"><Header/>
  <DataDictionary>
    <DataField name="b" optype="categorical" dataType="boolean"/>
    <DataField name="s" optype="categorical" dataType="string"/>
    <DataField name="i" optype="continuous" dataType="integer"/>
    <DataField name="d" optype="continuous" dataType="double"/>
    <DataField name="y" optype="continuous" dataType="double"/>
  </DataDictionary>
  <RegressionModel functionName="regression">
    <MiningSchema><MiningField name="b"/><MiningField name="s"/><MiningField name="i"/><MiningField name="d"/>
      <MiningField name="y" usageType="target"/></MiningSchema>
    <Output>
      <OutputField name="o" dataType="string" feature="transformedValue">)") +
           expression + R"(</OutputField>
    </Output>
    <RegressionTable intercept="1"/>
  </RegressionModel>
</PMML>)";
  };
  const std::string code = "E-TYPE-BOOLEAN-MIXED: Apply \"";
  CHECK(contains(load_error(doc(R"(<Apply function="equal"><FieldRef field="b"/><Constant dataType="string">true</Constant></Apply>)")),
                 code + "equal\" in RegressionModel compares boolean with string"));
  CHECK(contains(load_error(doc(R"(<Apply function="equal"><FieldRef field="b"/><Constant>1</Constant></Apply>)")),
                 "compares boolean with integer"));
  CHECK(contains(load_error(doc(R"(<Apply function="equal"><FieldRef field="s"/><FieldRef field="b"/></Apply>)")),
                 "compares string with boolean"));
  CHECK(contains(load_error(doc(R"(<Apply function="notEqual"><FieldRef field="d"/><FieldRef field="b"/></Apply>)")),
                 "compares double with boolean"));
  CHECK(contains(load_error(doc(R"(<Apply function="lessThan"><FieldRef field="b"/><FieldRef field="i"/></Apply>)")),
                 code + "lessThan\""));
  CHECK(contains(load_error(doc(R"(<Apply function="isIn"><FieldRef field="b"/><Constant dataType="boolean">true</Constant><Constant>0</Constant></Apply>)")),
                 code + "isIn\" in RegressionModel compares boolean with integer"));
  CHECK(contains(load_error(doc(R"(<Apply function="isNotIn"><FieldRef field="i"/><Constant>2</Constant><FieldRef field="b"/></Apply>)")),
                 "compares integer with boolean"));
  // Controls: boolean against boolean, and the other types among themselves.
  CHECK(load_error(doc(R"(<Apply function="equal"><FieldRef field="b"/><Constant dataType="boolean">true</Constant></Apply>)")).empty());
  CHECK(load_error(doc(R"(<Apply function="lessThan"><FieldRef field="b"/><FieldRef field="b"/></Apply>)")).empty());
  CHECK(load_error(doc(R"(<Apply function="isIn"><FieldRef field="b"/><Constant dataType="boolean">true</Constant><Constant dataType="boolean">0</Constant></Apply>)")).empty());
  // An untyped Constant true is a string (Transformations.html, Constant: a
  // Constant is a number only when it "consists solely of numbers"), so it
  // mixes too (JPMML 1.7.7 reads it as the boolean).
  CHECK(contains(load_error(doc(R"(<Apply function="equal"><FieldRef field="b"/><Constant>true</Constant></Apply>)")),
                 "compares boolean with string"));
  CHECK(load_error(doc(R"(<Apply function="equal"><FieldRef field="s"/><FieldRef field="i"/></Apply>)")).empty());
  CHECK(load_error(doc(R"(<Apply function="equal"><FieldRef field="d"/><Constant dataType="string">3</Constant></Apply>)")).empty());
}

int main() {
  bound_predicates();
  boolean_mixed();
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
  constants();
  no_silent_fallbacks();
  ordering();
  arity();
  if_condition();
  boolean_arguments();
  surrogate_errors();
  writer_round_trip();

  if (failures) std::cerr << failures << " check(s) failed\n";
  return failures;
}
