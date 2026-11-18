/*******************************************************************************
 * The binder in report mode (src/bind/, docs/design/typed-core.md §3.7,
 * design commit P1.2).
 *
 * Two parts:
 *
 * 1. Rule cases: small documents, each with the multiset of diagnostic codes
 *    the spec rule it cites makes the binder report (expected values come from
 *    the spec and the design's decisions, never from cPMML). A case fails when
 *    the codes differ in any way, so a rule that is broken, or one that fires
 *    where it should not, fails.
 *
 * 2. The baseline: every fixture under test/data/model, every benchmark model
 *    (benchmark/data/model) and every staged fixture (test/data/pending/model)
 *    is bound, and its diagnostics are compared with
 *    test/data/bind_baseline.txt. Within a document, identical diagnostics
 *    (same code and message) are one line with their count; lines are sorted
 *    by document, line, code and message:
 *
 *      <document>:<line> TAB <code> TAB <count> TAB <model> TAB <message>
 *
 *    This is the evidence the strictness decisions of P2 are made with (Q3,
 *    Q4, Q6, Q7, P2.7, K1/D22, B12, B17/B24, B42, D26).
 *
 * Usage: bind_report.exe <source-dir> [--update]
 *   --update  rewrite test/data/bind_baseline.txt from this run
 * Exit 0 when every rule case holds and the baseline matches.
 *******************************************************************************/

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// miniz.h defines static zlib-compatible wrappers this file does not use.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <miniz.h>
#pragma GCC diagnostic pop

#include "bind/binder.h"

namespace fs = std::filesystem;
using cpmml::bind::Binder;
using cpmml::bind::Diagnostic;

namespace {

int failures = 0;
int checks = 0;

// ---------------------------------------------------------------- rule cases
std::string pmml(const std::string& dictionary, const std::string& rest) {
  return "<?xml version=\"1.0\"?>\n<PMML xmlns=\"http://www.dmg.org/PMML-4_4\" version=\"4.4\">\n<Header/>\n"
         "<DataDictionary>\n" +
         dictionary + "</DataDictionary>\n" + rest + "</PMML>\n";
}

std::string field(const std::string& name, const std::string& type, const std::string& optype = "continuous",
                  const std::string& content = "") {
  return "<DataField name=\"" + name + "\" optype=\"" + optype + "\" dataType=\"" + type + "\">" + content +
         "</DataField>\n";
}

// A RegressionModel over `fields` (MiningFields), with optional content.
std::string regression(const std::string& mining, const std::string& content = "", const std::string& name = "m") {
  return "<RegressionModel modelName=\"" + name + "\" functionName=\"regression\">\n<MiningSchema>" + mining +
         "</MiningSchema>\n" + content + "<RegressionTable intercept=\"0\"/>\n</RegressionModel>\n";
}

std::string mf(const std::string& name, const std::string& extra = "") {
  return "<MiningField name=\"" + name + "\"" + (extra.empty() ? "" : " " + extra) + "/>";
}

std::string derived(const std::string& name, const std::string& type, const std::string& expr,
                    const std::string& optype = "continuous") {
  return "<DerivedField name=\"" + name + "\" optype=\"" + optype + "\" dataType=\"" + type + "\">" + expr +
         "</DerivedField>\n";
}

std::string lt(const std::string& fields) { return "<LocalTransformations>\n" + fields + "</LocalTransformations>\n"; }
std::string td(const std::string& fields) {
  return "<TransformationDictionary>\n" + fields + "</TransformationDictionary>\n";
}
std::string ref(const std::string& f) { return "<FieldRef field=\"" + f + "\"/>"; }
std::string call(const std::string& fn, const std::string& args, const std::string& extra = "") {
  return "<Apply function=\"" + fn + "\"" + (extra.empty() ? "" : " " + extra) + ">" + args + "</Apply>";
}
std::string constant(const std::string& text, const std::string& type = "") {
  return "<Constant" + (type.empty() ? "" : " dataType=\"" + type + "\"") + ">" + text + "</Constant>";
}

// A TreeModel with one Node holding `predicate`; its target is the string
// field c (kTree).
std::string tree(const std::string& mining, const std::string& predicate, const std::string& extra = "") {
  return "<TreeModel modelName=\"t\" functionName=\"classification\">\n<MiningSchema>" + mining +
         "<MiningField name=\"c\" usageType=\"target\"/>" + "</MiningSchema>\n" + extra +
         "<Node score=\"a\"><True/><Node score=\"a\">" + predicate + "</Node></Node>\n</TreeModel>\n";
}

std::vector<std::string> codes_of(const Binder& b) {
  std::vector<std::string> codes;
  for (const Diagnostic& d : b.diagnostics().all()) codes.emplace_back(d.code);
  std::sort(codes.begin(), codes.end());
  return codes;
}

std::string dump(const Binder& b) {
  std::string s;
  for (const Diagnostic& d : b.diagnostics().all())
    s += "      " + std::to_string(d.line) + " " + std::string(d.code) + " [" + d.model + "] " + d.message + "\n";
  return s;
}

void expect(const std::string& name, const std::string& rule, const std::string& doc,
            std::vector<std::string> expected) {
  ++checks;
  const Binder b(doc);
  std::sort(expected.begin(), expected.end());
  const auto got = codes_of(b);
  if (got == expected) return;
  ++failures;
  std::string want, have;
  for (const auto& c : expected) want += c + " ";
  for (const auto& c : got) have += c + " ";
  std::cerr << "FAILED rule case " << name << "\n    rule: " << rule << "\n    expected: " << want
            << "\n    got:      " << have << "\n"
            << dump(b);
}

void check(bool cond, const std::string& what, const std::string& rule) {
  ++checks;
  if (cond) return;
  ++failures;
  std::cerr << "FAILED " << what << "\n    rule: " << rule << "\n";
}

const std::string kNum = field("x", "double") + field("i", "integer") + field("s", "string", "categorical") +
                         field("b", "boolean", "categorical") + field("y", "double") +
                         field("c", "string", "categorical");
const std::string kMining = mf("x") + mf("i") + mf("s") + mf("b") + mf("y", "usageType=\"target\"");
const std::string kTree = mf("x") + mf("i") + mf("s") + mf("b");

void rule_cases() {
  // ---- names and scopes (FieldScope.html)
  expect("td-unresolved", "FieldScope.html: TD fields \"can only reference\" MiningFields and earlier TD fields",
         pmml(kNum, td(derived("d", "double", ref("nope"))) + regression(kMining)), {"E-SCOPE-UNRESOLVED"});
  expect("datafield-duplicate", "DataDictionary.html: \"The name of a DataField must be unique\"",
         pmml(kNum + field("x", "double"), regression(kMining)), {"E-NAME-DUPLICATE"});
  expect("td-duplicates-datafield", "FieldScope.html: DataField names unique from TD DerivedField names",
         pmml(kNum, td(derived("x", "double", ref("y"))) + regression(kMining)), {"E-NAME-DUPLICATE"});
  expect("miningfield-duplicate", "FieldScope.html: \"The names of MiningFields cannot be re-used\"",
         pmml(kNum, regression(kMining + mf("x"))), {"E-NAME-DUPLICATE"});
  expect("local-duplicates-miningfield", "FieldScope.html: LocalTransformations names \"must be unique\"",
         pmml(kNum, regression(kMining, lt(derived("x", "double", ref("y"))))), {"E-NAME-DUPLICATE"});
  expect("output-duplicates-local", "FieldScope.html: OutputField names \"must be unique\"",
         pmml(kNum, regression(kMining,
                               "<Output><OutputField name=\"d\" dataType=\"double\" "
                               "feature=\"predictedValue\"/></Output>" +
                                   lt(derived("d", "double", ref("x"))))),
         {"E-NAME-DUPLICATE"});
  expect("top-level-mining-td", "FieldScope.html: top-level MiningSchema \"can refer only to DataFields\"",
         pmml(kNum, td(derived("d", "double", ref("x"))) + regression(kMining + mf("d"))), {"E-SCOPE-TOP-LEVEL"});
  expect("mining-unknown", "MiningSchema.html: name \"must refer to a field in the scope of the parent\"",
         pmml(kNum, regression(kMining + mf("zz"))), {"E-SCOPE-UNRESOLVED"});
  expect("implicit-datafield", "FieldScope.html: \"all data must flow in through its MiningSchema\" (Q3)",
         pmml(kNum + field("z", "double"), regression(kMining, lt(derived("d", "double", ref("z"))))),
         {"W-SCOPE-IMPLICIT"});
  expect("td-instantiation", "FieldScope.html: instantiated TD fields reference \"MiningFields from the MiningSchema\"",
         pmml(kNum + field("z", "double"),
              td(derived("d", "double", ref("z"))) + regression(kMining, lt(derived("e", "double", ref("d"))))),
         {"W-SCOPE-IMPLICIT"});
  expect("td-not-instantiated", "FieldScope.html: TD fields \"not referenced ... are not instantiated\"",
         pmml(kNum + field("z", "double"), td(derived("d", "double", ref("z"))) + regression(kMining)), {});
  const std::string seg_model = [](const std::string& method, const std::string& seg2_mining) {
    return "<MiningModel functionName=\"regression\"><MiningSchema>" + kMining +
           "</MiningSchema><Segmentation multipleModelMethod=\"" + method + "\"><Segment id=\"1\"><True/>" +
           regression(kMining,
                      "<Output><OutputField name=\"o1\" dataType=\"double\" feature=\"predictedValue\"/></Output>" +
                          lt(derived("loc", "double", ref("x"))),
                      "s1") +
           "</Segment><Segment id=\"2\"><True/>" +
           regression(kMining + seg2_mining, lt(derived("loc", "double", ref("x"))), "s2") +
           "</Segment></Segmentation></MiningModel>\n";
  }("modelChain", mf("o1"));
  expect("chain-output-in-scope",
         "FieldScope.html: \"In a model chain, all Outputs from the segment models become part of the enclosing "
         "mining model scope\"; sibling LocalTransformations names may be duplicated",
         pmml(kNum, seg_model), {});
  expect("select-first-output-not-in-scope", "FieldScope.html: Outputs \"are generally local to a model\"",
         pmml(kNum, "<MiningModel functionName=\"regression\"><MiningSchema>" + kMining +
                        "</MiningSchema><Segmentation multipleModelMethod=\"selectFirst\"><Segment id=\"1\"><True/>" +
                        regression(kMining,
                                   "<Output><OutputField name=\"o1\" dataType=\"double\" feature=\"predictedValue\"/>"
                                   "</Output>",
                                   "s1") +
                        "</Segment><Segment id=\"2\"><True/>" + regression(kMining + mf("o1"), "", "s2") +
                        "</Segment></Segmentation></MiningModel>\n"),
         {"E-SCOPE-UNRESOLVED"});
  expect("chain-output-read-by-parent-output", "FieldScope.html modelChain (B15)",
         pmml(kNum, "<MiningModel functionName=\"regression\"><MiningSchema>" + kMining +
                        "</MiningSchema><Output><OutputField name=\"top\" dataType=\"double\" "
                        "feature=\"transformedValue\">" +
                        ref("o1") +
                        "</OutputField></Output><Segmentation multipleModelMethod=\"modelChain\">"
                        "<Segment id=\"1\"><True/>" +
                        regression(kMining,
                                   "<Output><OutputField name=\"o1\" dataType=\"double\" feature=\"predictedValue\"/>"
                                   "</Output>",
                                   "s1") +
                        "</Segment></Segmentation></MiningModel>\n"),
         {});
  expect("chain-output-duplicate", "FieldScope.html: \"Outputs cannot be duplicated across segment\" in a chain",
         pmml(kNum, "<MiningModel functionName=\"regression\"><MiningSchema>" + kMining +
                        "</MiningSchema><Segmentation multipleModelMethod=\"modelChain\"><Segment id=\"1\"><True/>" +
                        regression(kMining,
                                   "<Output><OutputField name=\"o\" dataType=\"double\" feature=\"predictedValue\"/>"
                                   "</Output>",
                                   "s1") +
                        "</Segment><Segment id=\"2\"><True/>" +
                        regression(kMining,
                                   "<Output><OutputField name=\"o\" dataType=\"double\" feature=\"predictedValue\"/>"
                                   "</Output>",
                                   "s2") +
                        "</Segment></Segmentation></MiningModel>\n"),
         {"E-NAME-DUPLICATE"});

  // ---- order (§3.5)
  expect("forward-local",
         "FieldScope.html: \"Forward referencing of fields is not allowed\" (Q4); B12: the later "
         "field's type is known (\"abc\" is not an integer)",
         pmml(kNum, regression(kMining, lt(derived("a", "double", "<NormDiscrete field=\"b2\" value=\"abc\"/>") +
                                           derived("b2", "integer", ref("i"))))),
         {"W-FORWARD-REF", "E-VALUE-MALFORMED"});
  expect("forward-td", "FieldScope.html: TD fields reference \"DerivedFields which appear earlier\"",
         pmml(kNum, td(derived("a", "double", ref("b2")) + derived("b2", "double", ref("x"))) +
                        regression(kMining, lt(derived("cc", "double", ref("a"))))),
         {"W-FORWARD-REF"});
  expect("cycle", "FieldScope.html: \"circular references ... are not valid PMML\"",
         pmml(kNum, regression(kMining, lt(derived("a", "double", ref("b2")) + derived("b2", "double", ref("a"))))),
         {"W-FORWARD-REF", "E-CYCLE"});
  expect("output-transformed-reads-local", "FieldScope.html: the one exception (transformedValue -> LT)",
         pmml(kNum, regression(kMining,
                               "<Output><OutputField name=\"o\" dataType=\"double\" "
                               "feature=\"transformedValue\">" +
                                   ref("d") + "</OutputField></Output>" + lt(derived("d", "double", ref("x"))))),
         {});
  expect("output-forward", "FieldScope.html: other OutputFields \"defined before it\" (D12)",
         pmml(kNum, regression(kMining,
                               "<Output><OutputField name=\"o\" dataType=\"double\" "
                               "feature=\"transformedValue\">" +
                                   ref("p") +
                                   "</OutputField><OutputField name=\"p\" dataType=\"double\" "
                                   "feature=\"predictedValue\"/></Output>")),
         {"W-FORWARD-REF"});
  expect(
      "output-decision-reads-local", "FieldScope.html: the exception names transformedValue only",
      pmml(kNum, regression(kMining, "<Output><OutputField name=\"o\" dataType=\"double\" feature=\"decision\">" +
                                         ref("d") + "</OutputField></Output>" + lt(derived("d", "double", ref("x"))))),
      {"W-FORWARD-REF"});

  // ---- Constant (Transformations.html)
  const auto in_model = [](const std::string& lts) { return pmml(kNum, regression(kMining, lt(lts))); };
  expect("constant-untyped-string", "B8: \"anything else string\"",
         in_model(derived("u", "string", call("uppercase", constant("abc")), "categorical")), {});
  expect("constant-1e10-double", "\"A scientific notation expression like 3.14e+5 ... treated as a double\" (B9)",
         in_model(derived("u", "double", call("+", constant("1e10") + ref("x")))), {});
  expect("constant-nan-string",
         "D10: untyped NaN \"the literal strings will be signified\"; §2.5 no arithmetic on string",
         in_model(derived("u", "double", call("+", constant("NaN") + ref("x")))), {"E-TYPE-PARAM"});
  expect("constant-inf-case", "\"These three constants are case insensitive\" (dataType double)",
         in_model(derived("u", "double", call("+", constant("inf", "double") + ref("x")))), {});
  expect("constant-typed-malformed", "a typed Constant's text is a value of its dataType",
         in_model(derived("u", "double", call("+", constant("2.5", "integer") + ref("x")))), {"E-VALUE-MALFORMED"});
  expect("constant-overflow", "§3.4: \"integer beyond int64 -> load error\" (D4)",
         in_model(derived("u", "double", call("+", constant("99999999999999999999") + ref("x")))),
         {"E-VALUE-MALFORMED"});
  expect("constant-inherits-boolean", "\"the Constant will inherit the dataType specified in the ParameterField\" (S2)",
         in_model(derived("u", "string", call("if", constant("true") + constant("a") + constant("b")), "categorical") +
                  derived("v", "boolean", call("not", constant("1")), "categorical")),
         {});
  expect("constant-inherits-malformed", "S2: the inherited type reads the text",
         in_model(derived("v", "boolean", call("not", constant("yes")), "categorical")), {"E-VALUE-MALFORMED"});

  // ---- functions (§4.2)
  expect("arity", "BuiltinFunctions.html: + \"have two parameters\"",
         in_model(derived("u", "double", call("+", ref("x")))), {"E-ARITY"});
  expect("unknown-function", "FieldScope.html: fields that \"apply undefined functions are not valid PMML\"",
         in_model(derived("u", "double", call("log", ref("x")))), {"E-FUNCTION-UNKNOWN"});
  expect("order-string-number", "§2.7: string x number ordering is a load error (P2.7)",
         in_model(derived("u", "boolean", call("lessThan", ref("s") + constant("1")), "categorical")),
         {"E-TYPE-ORDER-MIXED"});
  expect("equal-string-number", "§2.7 D2: string x number equality compares text",
         in_model(derived("u", "boolean", call("equal", ref("s") + constant("1")), "categorical")), {});
  expect("boolean-mixed", "Functions.html: boolean \"can not be mixed with other types\" (Q7)",
         in_model(derived("u", "boolean", call("equal", ref("b") + constant("1")), "categorical") +
                  derived("v", "double", call("+", ref("b") + constant("1")))),
         {"E-TYPE-BOOLEAN-MIXED", "E-TYPE-BOOLEAN-MIXED"});
  expect("integer-param", "Functions.html: \"a double expression can not be implicitly cast as an integer\"",
         in_model(derived("u", "string", call("substring", ref("s") + ref("x") + constant("2")), "categorical")),
         {"E-TYPE-PARAM"});
  expect("narrowing-derived", "§2.6, Q2: an integer DerivedField over a double (B17)",
         in_model(derived("u", "integer", call("*", ref("x") + constant("1.0")))), {"W-CAST-NARROWING"});
  expect("if-static-type", "D7: if has the common type of THEN and ELSE (B19)",
         in_model(derived("u", "integer",
                          call("if", call("greaterThan", ref("x") + constant("0")) + ref("i") + constant("1.5")))),
         {"W-CAST-NARROWING"});
  expect("lossy-float", "§2.6: double -> float rounds",
         in_model(derived("u", "float", call("*", ref("x") + constant("2")))), {"W-CAST-LOSSY"});
  expect("new-functions-typed", "§4.2 rows: and, or, formatNumber, isValid",
         in_model(derived("u", "boolean",
                          call("and", call("lessThan", ref("x") + constant("1")) + call("isValid", ref("x"))),
                          "categorical") +
                  derived("v", "string", call("formatNumber", ref("x") + constant("%.2f")), "categorical")),
         {});
  expect("isvalid-fieldref", "BuiltinFunctions.html isValid: ParameterField name=\"field\"",
         in_model(derived("u", "boolean", call("isValid", constant("1")), "categorical")), {"E-TYPE-PARAM"});
  expect("round-is-integer", "BuiltinFunctions.html: \"Functions floor, ceil, and round return an integer\"",
         in_model(derived("u", "integer", call("round", ref("x")))), {});
  expect("apply-mapmissingto", "§3.4: Apply mapMissingTo \"parsed as the Apply's result type\"",
         in_model(derived("u", "integer", call("round", ref("x"), "mapMissingTo=\"x\""))), {"E-VALUE-MALFORMED"});
  expect("apply-extension", "B25: \"children after Extension*\"",
         in_model(derived("u", "double", call("+", "<Extension name=\"e\"/>" + ref("x") + ref("x")))), {});

  // ---- predicates and document text (§3.4 K1)
  expect("k1-double-text", "§3.4 rule 2: integer field, \"33.0\" read as double (D22)",
         pmml(kNum, tree(kTree, "<SimplePredicate field=\"i\" operator=\"lessOrEqual\" value=\"33.0\"/>")),
         {"W-TEXT-WIDENED"});
  expect("k1-malformed", "§3.4 rule 3: non-numeric text on a numeric field",
         pmml(kNum, tree(kTree, "<SimplePredicate field=\"i\" operator=\"equal\" value=\"abc\"/>")),
         {"E-VALUE-MALFORMED"});
  expect("k1-boolean", "XSD boolean lexical space {true, false, 1, 0} (D11)",
         pmml(kNum, tree(kTree, "<SimplePredicate field=\"b\" operator=\"equal\" value=\"TRUE\"/>")),
         {"E-VALUE-MALFORMED"});
  expect("predicate-ismissing-value", "TreeModel.html: with isMissing \"the attribute value must not appear\"",
         pmml(kNum, tree(kTree, "<SimplePredicate field=\"x\" operator=\"isMissing\" value=\"1\"/>")), {"E-FORBIDDEN"});
  expect("predicate-value-required", "TreeModel.html: \"With all other operators ... the attribute value is required\"",
         pmml(kNum, tree(kTree, "<SimplePredicate field=\"x\" operator=\"lessThan\"/>")), {"E-REQUIRED"});
  expect("order-categorical", "DataDictionary.html: \"Categorical field values can only be tested for equality\" (Q6)",
         pmml(kNum, tree(kTree, "<SimplePredicate field=\"s\" operator=\"lessThan\" value=\"a\"/>")),
         {"W-ORDER-CATEGORICAL"});
  expect("ordinal-string-no-values", "DataDictionary.html: ordinal strings \"must always specify the order\" (Q6)",
         pmml(kNum + field("o", "string", "ordinal"),
              tree(kTree + mf("o"), "<SimplePredicate field=\"o\" operator=\"lessThan\" value=\"a\"/>")),
         {"W-ORDINAL-STRING-NO-VALUES"});
  expect("ordinal-string-values", "DataDictionary.html: \"the sequence defines the ascending order\"",
         pmml(kNum + field("o", "string", "ordinal", "<Value value=\"a\"/><Value value=\"b\"/>"),
              tree(kTree + mf("o"), "<SimplePredicate field=\"o\" operator=\"lessThan\" value=\"b\"/>")),
         {});
  expect("set-nan", "§2.7, D26: a NaN key matches nothing",
         pmml(kNum, tree(kTree,
                         "<SimpleSetPredicate field=\"x\" booleanOperator=\"isIn\"><Array type=\"real\">1 "
                         "NaN</Array></SimpleSetPredicate>")),
         {"E-KEY-NAN"});
  expect("compound-arity", "pmml-4-4.xsd: CompoundPredicate PREDICATE minOccurs=\"2\"",
         pmml(kNum, tree(kTree, "<CompoundPredicate booleanOperator=\"and\"><True/></CompoundPredicate>")),
         {"E-REQUIRED"});

  // ---- MapValues, Discretize, Norm*
  const auto mapvalues = [](const std::string& f, const std::string& rows, const std::string& type = "") {
    return "<MapValues outputColumn=\"out\"" + (type.empty() ? "" : " dataType=\"" + type + "\"") +
           "><FieldColumnPair field=\"" + f + "\" column=\"in\"/><InlineTable>" + rows + "</InlineTable></MapValues>";
  };
  const auto row = [](const std::string& in, const std::string& out) {
    return "<row><in>" + in + "</in><out>" + out + "</out></row>";
  };
  expect("mapvalues-duplicate-typed",
         "Transformations.html: \"it is an error if the table entries ... are not unique\" (B42)",
         in_model(derived("u", "string", mapvalues("i", row("1", "a") + row("1.0", "b")), "categorical")),
         {"W-TEXT-WIDENED", "E-KEY-DUPLICATE"});
  expect("mapvalues-duplicate-string", "Transformations.html: table entries unique (B42)",
         in_model(derived("u", "string", mapvalues("s", row("a", "a") + row("a", "b")), "categorical")),
         {"E-KEY-DUPLICATE"});
  expect("mapvalues-nan", "§2.7, D26: a NaN key is a load error",
         in_model(derived("u", "string", mapvalues("x", row("NaN", "a")), "categorical")), {"E-KEY-NAN"});
  expect("mapvalues-own-type", "§3.4: output cells \"as the MapValues type\" (B11)",
         in_model(derived("u", "string", mapvalues("s", row("a", "1.50")), "categorical") +
                  derived("v", "double", mapvalues("s", row("a", "abc"), "double"))),
         {"E-VALUE-MALFORMED"});
  expect("discretize-binvalue", "§3.4: binValue parsed as the Discretize type",
         in_model(derived("u", "double",
                          "<Discretize field=\"x\"><DiscretizeBin binValue=\"low\"><Interval "
                          "closure=\"openOpen\" rightMargin=\"1\"/></DiscretizeBin></Discretize>")),
         {"E-VALUE-MALFORMED"});
  expect("discretize-string-field", "Transformations.html: \"Discretization of numerical input fields\"",
         in_model(derived("u", "string",
                          "<Discretize field=\"s\"><DiscretizeBin binValue=\"low\"><Interval "
                          "closure=\"openOpen\" rightMargin=\"1\"/></DiscretizeBin></Discretize>",
                          "categorical")),
         {"E-TYPE-PARAM"});
  expect("normcontinuous-order",
         "Transformations.html: \"LinearNorm must be strictly sorted by ascending value of orig\"",
         in_model(derived("u", "double",
                          "<NormContinuous field=\"x\"><LinearNorm orig=\"1\" norm=\"0\"/><LinearNorm "
                          "orig=\"1\" norm=\"1\"/></NormContinuous>")),
         {"E-NORM-ORDER"});
  expect("normcontinuous-two", "Transformations.html: \"The sequence must contain at least two elements\"",
         in_model(derived("u", "double",
                          "<NormContinuous field=\"x\"><LinearNorm orig=\"1\" norm=\"0\"/></NormContinuous>")),
         {"E-REQUIRED"});
  expect("number-attribute", "pmml-4-4.xsd: LinearNorm@orig is NUMBER (xs:double)",
         in_model(derived("u", "double",
                          "<NormContinuous field=\"x\"><LinearNorm orig=\"1x\" norm=\"0\"/><LinearNorm "
                          "orig=\"2\" norm=\"1\"/></NormContinuous>")),
         {"E-ATTRIBUTE-MALFORMED"});
  expect("normdiscrete-integer", "Transformations.html NormDiscrete: v is a value of f (B10)",
         in_model(derived("u", "double", "<NormDiscrete field=\"i\" value=\"3\"/>")), {});

  // ---- MiningField
  expect("replacement-widens", "§3.4 / N3: a non-integer replacement widens the integer slot (D22)",
         pmml(kNum, regression(mf("x") + mf("i", "missingValueReplacement=\"2.5\"") + mf("y", "usageType=\"target\""))),
         {"W-SLOT-WIDENED"});
  expect("replacement-malformed", "§3.4 rule 3",
         pmml(kNum, regression(mf("x") + mf("i", "missingValueReplacement=\"abc\"") + mf("y", "usageType=\"target\""))),
         {"E-VALUE-MALFORMED"});
  expect("asvalue-needs-replacement", "MiningSchema.html: invalidValueReplacement \"must be present in this case\"",
         pmml(kNum, regression(mf("x", "invalidValueTreatment=\"asValue\"") + mf("y", "usageType=\"target\""))),
         {"E-REQUIRED"});

  // ---- enumerations, XSD types
  expect("enum-optype", "pmml-4-4.xsd OPTYPE", pmml(field("x", "double", "nominal"), regression(mf("x"))), {"E-ENUM"});
  expect("enum-datatype", "pmml-4-4.xsd DATATYPE", pmml(field("x", "int"), regression(mf("x"))), {"E-ENUM"});
  expect("enum-operator", "TreeModel.html SimplePredicate@operator",
         pmml(kNum, tree(kTree, "<SimplePredicate field=\"x\" operator=\"lt\" value=\"1\"/>")), {"E-ENUM"});
  expect("exponent-int-number", "pmml-4-4.xsd NumericPredictor@exponent INT-NUMBER (B7)",
         pmml(kNum, "<RegressionModel functionName=\"regression\"><MiningSchema>" + kMining +
                        "</MiningSchema><RegressionTable intercept=\"0\"><NumericPredictor name=\"x\" exponent=\"2.5\" "
                        "coefficient=\"1\"/></RegressionTable></RegressionModel>"),
         {"E-ATTRIBUTE-MALFORMED"});
  expect("interval-categorical", "DataDictionary.html: \"Intervals are not allowed for non-continuous fields\"",
         pmml(field("x", "double", "categorical", "<Interval closure=\"closedClosed\" leftMargin=\"0\"/>"),
              regression(mf("x"))),
         {"E-FORBIDDEN"});
  expect("interval-margins", "DataDictionary.html: \"at least one value must be defined\"",
         pmml(field("x", "double", "continuous", "<Interval closure=\"closedClosed\"/>"), regression(mf("x"))),
         {"E-REQUIRED"});

  // ---- Output and Targets
  const auto output = [](const std::string& fields) { return "<Output>" + fields + "</Output>"; };
  expect("output-predicted-narrowing", "§3.4 / Q2: an integer OutputField over a double prediction (B24)",
         pmml(kNum,
              regression(kMining, output("<OutputField name=\"o\" dataType=\"integer\" feature=\"predictedValue\"/>"))),
         {"W-CAST-NARROWING"});
  expect(
      "output-probability-boolean", "§3.4: probability is double; boolean narrows (B24)",
      pmml(kNum, regression(kMining, output("<OutputField name=\"o\" dataType=\"boolean\" feature=\"probability\"/>"))),
      {"W-CAST-NARROWING"});
  expect("output-no-datatype", "pmml-4-4.xsd: OutputField@dataType use=\"required\"",
         pmml(kNum, regression(kMining, output("<OutputField name=\"o\" feature=\"predictedValue\"/>"))),
         {"W-OUTPUT-NO-DATATYPE"});
  expect("output-transformed-needs-expression", "Output.html: \"OutputField must contain an EXPRESSION\"",
         pmml(kNum, regression(kMining,
                               output("<OutputField name=\"o\" dataType=\"double\" feature=\"transformedValue\"/>"))),
         {"E-REQUIRED"});
  expect("targetvalue-integer", "§3.4: TargetValue@value read by the rule (D22)",
         pmml(field("x", "double") + field("t", "integer", "categorical"),
              "<TreeModel functionName=\"classification\"><MiningSchema>" + mf("x") + mf("t", "usageType=\"target\"") +
                  "</MiningSchema><Targets><Target field=\"t\"><TargetValue value=\"4.0\"/></Target></Targets>"
                  "<Node score=\"4\"><True/></Node></TreeModel>"),
         {"W-TEXT-WIDENED"});
  expect("node-score-target", "§3.4: category values by the rule (Node@score is a value of the target)",
         pmml(field("x", "double") + field("t", "integer", "categorical"),
              "<TreeModel functionName=\"classification\"><MiningSchema>" + mf("x") + mf("t", "usageType=\"target\"") +
                  "</MiningSchema><Node score=\"yes\"><True/></Node></TreeModel>"),
         {"E-VALUE-MALFORMED"});

  // ---- unsupported, never silent
  expect("temporal", "§2.1 / Q8: date types are double until P10",
         pmml(kNum + field("dt", "date"), regression(kMining)), {"U-TEMPORAL"});
  expect("aggregate", "§3.4: \"Aggregate, Lag -> load error unsupported\"",
         in_model(derived("u", "double", "<Aggregate field=\"x\" function=\"sum\"/>")), {"U-ELEMENT"});
  expect("define-function", "§3.4: DefineFunction out of scope",
         pmml(kNum,
              "<TransformationDictionary><DefineFunction name=\"f\" optype=\"continuous\" dataType=\"double\">"
              "<ParameterField name=\"p\"/>" +
                  ref("p") + "</DefineFunction></TransformationDictionary>" +
                  regression(kMining, lt(derived("u", "double", call("f", ref("x")))))),
         {"U-ELEMENT", "U-USER-FUNCTION"});
  expect("unknown-body-element", "the project rule: no silent skips",
         pmml(kNum, tree(kTree, "<True/>", "<ModelStats/>")), {"U-ELEMENT"});
  expect("not-xml", "the document must be XML", "<PMML><DataDictionary>", {"E-XML"});
  expect("clean", "a document with nothing to report", in_model(derived("u", "double", ref("x"))), {});

  // ---- types and layout (§3.1, §3.3)
  {
    const Binder b(pmml(kNum, regression(kMining, "<Output><OutputField name=\"o\" feature=\"transformedValue\">" +
                                                      call("+", ref("i") + constant("1")) + "</OutputField></Output>" +
                                                      lt(derived("a", "float", ref("later")) +
                                                         derived("later", "integer", ref("i"))))));
    const auto* a = b.find("RegressionModel \"m\"", "a");
    const auto* later = b.find("RegressionModel \"m\"", "later");
    const auto* o = b.find("RegressionModel \"m\"", "o");
    check(a && a->type.type == cpmml::types::TypeId::Float, "DerivedField type is its dataType",
          "§3.4: DerivedField dataType=X -> X");
    check(later && later->type.type == cpmml::types::TypeId::Integer && a && !a->reads.empty() &&
              a->reads.front() == later,
          "forward reference resolved to the later declaration with its type", "§3.5 two-phase binding (L3)");
    check(o && o->type.type == cpmml::types::TypeId::Integer, "OutputField without dataType typed by its expression",
          "§3.4 OutputField: \"inferred ... when absent\"; + of integers is integer (§2.5)");
    const auto* mx = b.find("RegressionModel \"m\"", "x");
    check(mx && b.layout().slots[mx->slot].alias_of == mx->source->slot,
          "a MiningField without treatment is an alias of its DataField's slot", "§3.2 Aliasing");
  }
  {
    const Binder b(
        pmml(kNum, regression(mf("x") + mf("i", "missingValueReplacement=\"2.5\"") + mf("y", "usageType=\"target\""))));
    const auto* mi = b.find("RegressionModel \"m\"", "i");
    check(mi && mi->type.type == cpmml::types::TypeId::Double, "slot widened to double", "§3.4 / N3 slot widening");
    check(mi && b.layout().slots[mi->slot].alias_of == mi->slot, "a treated MiningField has its own slot",
          "§3.2 Aliasing: a replacement is a step");
  }
  {
    // Line numbers: the diagnostic names the element's line.
    const std::string doc =
        pmml(kNum, tree(kTree, "\n<SimplePredicate field=\"i\" operator=\"equal\" value=\"abc\"/>"));
    const Binder b(doc);
    std::uint32_t want = 1;
    const std::size_t at = doc.find("<SimplePredicate");
    for (std::size_t k = 0; k < at; k++) want += doc[k] == '\n';
    check(b.diagnostics().all().size() == 1 && b.diagnostics().all()[0].line == want &&
              b.diagnostics().all()[0].model == "TreeModel \"t\"",
          "a diagnostic carries the element's line and the enclosing model", "§3.6: code, message, line");
  }
}

// ---------------------------------------------------------------- the baseline
bool read_document(const fs::path& path, std::string& out) {
  if (path.extension() == ".zip") {
    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof(zip));
    if (!mz_zip_reader_init_file(&zip, path.string().c_str(), 0)) return false;
    std::size_t size = 0;
    void* p = mz_zip_reader_extract_to_heap(&zip, 0, &size, 0);
    mz_zip_reader_end(&zip);
    if (!p) return false;
    out.assign(static_cast<const char*>(p), size);
    mz_free(p);
    return true;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream s;
  s << in.rdbuf();
  out = s.str();
  return true;
}

std::vector<std::string> baseline_lines(const fs::path& root, std::map<std::string, std::size_t>& per_code,
                                        std::size_t& documents) {
  std::vector<fs::path> files;
  for (const char* dir : {"test/data/model", "benchmark/data/model", "test/data/pending/model"}) {
    const fs::path d = root / dir;
    if (!fs::exists(d)) continue;
    for (const auto& e : fs::directory_iterator(d)) {
      const auto ext = e.path().extension();
      if (ext == ".zip" || ext == ".xml" || ext == ".pmml") files.push_back(e.path());
    }
  }
  std::sort(files.begin(), files.end());
  std::vector<std::string> lines;
  for (const fs::path& f : files) {
    const std::string rel = fs::relative(f, root).generic_string();
    std::string text;
    documents++;
    if (!read_document(f, text)) {
      lines.push_back(rel + ":0\tE-READ\t1\t\tthe file cannot be read");
      continue;
    }
    const Binder b(text);
    // Identical diagnostics within a document: one line, with the count.
    struct Group {
      std::uint32_t line;
      std::string code, message, model;
      std::size_t count;
      std::size_t models;
      std::string last_model;
    };
    std::map<std::pair<std::string, std::string>, Group> groups;
    for (const Diagnostic& d : b.diagnostics().all()) {
      per_code[std::string(d.code)]++;
      auto key = std::make_pair(std::string(d.code), d.message);
      auto it = groups.find(key);
      if (it == groups.end()) {
        groups.emplace(key, Group{d.line, std::string(d.code), d.message, d.model, 1, 1, d.model});
      } else {
        it->second.count++;
        if (d.model != it->second.last_model) it->second.models++;
        it->second.last_model = d.model;
        it->second.line = std::min(it->second.line, d.line);
      }
    }
    std::vector<Group> sorted;
    for (auto& [k, g] : groups) sorted.push_back(g);
    std::sort(sorted.begin(), sorted.end(), [](const Group& a, const Group& b) {
      return std::tie(a.line, a.code, a.message) < std::tie(b.line, b.code, b.message);
    });
    for (const Group& g : sorted) {
      std::string model = g.model;
      if (g.models > 1) model += " (+" + std::to_string(g.models - 1) + " more)";
      lines.push_back(rel + ":" + std::to_string(g.line) + "\t" + g.code + "\t" + std::to_string(g.count) + "\t" +
                      model + "\t" + g.message);
    }
  }
  return lines;
}

int run_baseline(const fs::path& root, bool update) {
  std::map<std::string, std::size_t> per_code;
  std::size_t documents = 0;
  const std::vector<std::string> lines = baseline_lines(root, per_code, documents);
  std::vector<std::string> out;
  out.push_back("# Binder report-mode baseline (test/bind_report.cc, docs/design/typed-core.md §3.7).");
  out.push_back("# Regenerate with: bind_report.exe <source-dir> --update");
  out.push_back("# documents: " + std::to_string(documents));
  for (const auto& [code, n] : per_code) out.push_back("# " + code + ": " + std::to_string(n));
  out.insert(out.end(), lines.begin(), lines.end());

  const fs::path baseline = root / "test/data/bind_baseline.txt";
  if (update) {
    std::ofstream f(baseline, std::ios::binary);
    for (const auto& l : out) f << l << "\n";
    std::cout << "wrote " << baseline.string() << " (" << lines.size() << " lines, " << documents << " documents)\n";
    return 0;
  }
  std::ifstream f(baseline, std::ios::binary);
  if (!f) {
    std::cerr << "FAILED: no baseline " << baseline.string() << " (run with --update)\n";
    return 1;
  }
  std::vector<std::string> expected;
  for (std::string l; std::getline(f, l);) expected.push_back(l);
  if (expected == out) {
    std::cout << "baseline matches: " << documents << " documents, " << lines.size() << " lines\n";
    return 0;
  }
  std::vector<std::string> a = expected, b = out;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  std::vector<std::string> gone, added;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(gone));
  std::set_difference(b.begin(), b.end(), a.begin(), a.end(), std::back_inserter(added));
  std::cerr << "FAILED: the diagnostics differ from " << baseline.string() << " (" << gone.size() << " lines gone, "
            << added.size() << " new)\n";
  std::size_t shown = 0;
  for (const auto& l : gone)
    if (shown++ < 40) std::cerr << "  - " << l << "\n";
  shown = 0;
  for (const auto& l : added)
    if (shown++ < 40) std::cerr << "  + " << l << "\n";
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: bind_report.exe <source-dir> [--update]\n";
    return 2;
  }
  const fs::path root = argv[1];
  const bool update = argc > 2 && std::string(argv[2]) == "--update";
  rule_cases();
  std::cout << checks - failures << "/" << checks << " rule checks passed\n";
  const int baseline = run_baseline(root, update);
  return (failures || baseline) ? 1 : 0;
}
