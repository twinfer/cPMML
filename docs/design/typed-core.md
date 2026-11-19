# Typed core: values, binding and one evaluation pipeline

Status: design approved 2026-11-17 (all open questions decided as recommended, §9), not implemented; revision 2 after an independent review (the
"Revision log", §10, maps every review item to its resolution). Written
against `master` at `b4efd7d` (341/341 tests green). No source file is changed
by this document.

Authority: the PMML 4.4.1 specification (<https://dmg.org/pmml/v4-4-1/>,
including `pmml-4-4.xsd`). JPMML-Evaluator 1.7.7
(`tools/pmml-evaluator-example-executable-1.7.7.jar`) is an oracle only: where
the spec is silent we say what JPMML does and whether we follow it. Every rule
below that the design encodes is quoted from the spec page it comes from;
where the design departs from the spec, the departure is listed in §8.5 with
its reason.

Artefacts referred to below, under `docs/design/` (the `design/` prefix below means `docs/design/typed-core/`):

| path | what |
|---|---|
| `design/sketch/typed_value.h`, `design/sketch/typed_value_test.cc` | the type model of §2 as compilable C++20; behaviour asserts and a micro-benchmark |
| `design/sketch/neg_missing_case.cc`, `design/sketch/neg_uncategorised.cc` | must-not-compile tests (a visitor missing an alternative; an alternative in no category) |
| `design/sketch/build.sh`, `design/sketch/build.log` | builds with Apple clang 21, g++-16 and em++ 7.0 using the repo's flags, with and without the P10 temporal alternatives |
| `design/probe/` | the JPMML-vs-cPMML probes quoted in §1.3 and §3.4 |
| `design/k1/scan.py` | scans every fixture and benchmark model for document text outside its field's lexical space (§3.4) |
| `design/bench_baseline.txt`, `design/bench_housing.txt` | raw sequential benchmark runs behind §6.1 |
| `design/aab.txt`, `design/aab_stats.py`, `design/aab_stats.txt` | interleaved A/A noise study and OpenMP comparison behind §6.4 |
| `design/spec/`, `design/dl/` | the spec pages and XSD the quotes come from |
| `design-review/review.md`, `design-review/p/`, `design-review/tree.cc` | the review this revision answers, its probes and micro-benchmark |

---

## 1. Problem statement

### 1.1 Four root causes

**RC1 — a value has no type.** `Value` (`src/core/value.h:37-41`) is
`double value; bool missing; std::string svalue;`, 40 bytes. A string is its
code in the model's `Indexer`, reached through a `thread_local`
`Value::active_indexer` (`value.h:45`); after load the map is frozen and every
string the document never names gets the code `-1` (`indexer.h:83-93`). An
integer is a double (`datatype.h:30-48` maps `integer` to `DOUBLE`, with an
`integer` flag bolted onto `DataField` and `OutputField`); a float is a double
that may or may not have been narrowed; dates are doubles. Equality is
`value == other.value` whatever the two values are (`value.h:80`). String
functions edit `svalue` and keep the argument's code (`value.h:99-102`).

**RC2 — types and names are resolved lazily, at runtime, by several
mechanisms that disagree.** Expression constructors receive the enclosing
DerivedField's `output_type` and use it for their own attributes even when the
element declares its own type (`apply.h:49,51`, `normcontinuous.h:41-49`,
`discretize.h:42-52`, `mapvalues.h:45-47,70`, `normdiscrete.h:44`,
`fieldref.h:47`). The type of an expression is recomputed separately by
re-walking the XML (`BuiltInFunction::expression_kind`/`field_kind`,
`builtinfunctions.h:928-990`). Predicates ask the `Indexer` for a field's type
at load (`predicatebuilder.h`), which throws `std::out_of_range` for a field
declared later. All models share one flat name space (`Indexer::get_or_set`),
so two sibling segments with a field `z` share a slot. `DagBuilder` silently
drops a DerivedField whose input is not in the MiningSchema
(`dagbuilder.h:61,73`). `TransformedValue` swallows load errors
(`transformedvalue.h:39`).

**RC3 — missing and invalid semantics are per function.** `Apply::eval`
(`apply.h:95-136`) has two flags (`handles_missing`, `ignores_missing`), a
`try/catch (const std::exception&)` that turns any exception into "invalid",
and its own ordering of mapMissingTo/defaultValue; `if` treats a missing
condition as false (`builtinfunctions.h:608`); divisions return IEEE results.

**RC4 — each model assembles the per-record pipeline itself.**
`InternalModel::score` (`internal_model.h:428`), `predict` (`:494`),
`augment`/`augment_last` (`:339,:364`, modelChain), `Segment::score/result/
predict` (`segment.h`), `AnomalyDetectionEvaluator::evaluate` (which re-applies
a MiningSchema on *text* and writes the result back as text) and the 14
string-typed `predict_raw` overrides (plus the base virtual) each run some
subset of MiningSchema → transformations → model → Targets → Outputs. A
prediction travels between segments as text and is re-parsed
(`InternalScore::value_of`), classes are renamed after the fact (`present()`,
`label_text` cache), and whether a displayValue replaces the class depends on
a `display` flag threaded through `Target::operator()`.

### 1.2 Bugs mapped to root causes

Every row is a "Known gaps" item from the last 60 commits, a defect found by
the verifiers of the two failed patches or by the review, or a probe in §1.3.
"Phase" is the migration phase (§7) that closes it; each closing is its own
commit.

| id | symptom | where recorded | RC | phase |
|---|---|---|---|---|
| B1 | two strings the model never names are equal (`-1 == -1`); `equal(s, s2)` true for `abc`/`yy` | 315a815 (fixed for GRM only), probe `unseen_eq` | 1 | P6 |
| B2 | `uppercase/lowercase/substring/trimBlanks` keep the argument's code: `equal(uppercase(s),"ABC")` false for `abc` | typed_values verify2, probe `up_eq` | 1 | P6 |
| B3 | binary similarity measures score string codes as 0/1 | 55fd0bf, fix_binsim_* | 1 | P6 |
| B4 | float values are not floats: inputs and document text not narrowed; float compared in double outside built-ins; `3.4e39` stays finite | 6914ecd, 59513a2, 90847c1 | 1 | P6 |
| B5 | `double` OutputField over a float prediction not narrowed first; float OutputField read by another keeps double precision | ceb6cec, d23902a | 1 | P6, P8 |
| B6 | numeric missing marker compared as text: input `-999.0` is not the marker `-999` | 90847c1 | 1 | P6 |
| B7 | integer carried as double: GRM level `1.5` loads, `1.0` matches; `NumericPredictor@exponent` (XSD `INT-NUMBER`) read as double, `2.5` gives 32 | 315a815, 6914ecd | 1/2 | P2 |
| B8 | an untyped `Constant` loses its text: `uppercase(<Constant>abc</Constant>)` is missing (JPMML `ABC`); the spec's own `concat(month,"-",year)` | concat verify, probe `const_up` | 1/2 | P2 |
| B9 | untyped `Constant` inference: `1e10` becomes a string (int-range rule), `NaN`/`INF` become numbers | `Value::infer_value`, concat verify2 | 2 | P2 |
| B10 | `NormDiscrete` value is always read as a string, so it never matches an integer or double field (`NormDiscrete(i, "3")` gives 0.0; JPMML 1.0) | probe `nd_i` | 2 | P2 |
| B11 | NormContinuous points and every Norm* `mapMissingTo` are read with the enclosing DerivedField's dataType (NormContinuous in a string DerivedField interpolates string codes); Discretize and MapValues ignore their own `dataType` attribute | typed_values verify2 (q3, q4) | 2 | P2 |
| B12 | a forward reference works only while nobody asks the referenced field's type; asking (NormDiscrete, predicates) throws `std::out_of_range` | typed_values verify2 (p2) | 2 | P2 |
| B13 | an OutputField whose expression fails to load is silently empty forever | concat verify2 | 2 | P2 |
| B14 | a DerivedField whose input is not in the MiningSchema is silently dropped; a segment referencing an outer LocalTransformations/TransformationDictionary field predicts missing | 59513a2, `dagbuilder.h` | 2 | P7 |
| B15 | OutputFields of an outer MiningModel that read a modelChain segment's OutputFields give N/A | typed_values verify2 (p5) | 2 | P7 |
| B16 | one flat name space: sibling segments' LocalTransformations fields of the same name share a slot | 59513a2 (typing part fixed) | 2 | P7 |
| B17 | an integer DerivedField over a double keeps the fraction (`dint` = 3.5) | 59513a2, probe `dint` | 2 | P4 |
| B18 | mixed-type comparison compares codes or depends on argument order: `equal(s,"3" vs integer 3)` false | probe `eq_s_i`, `eq_i_s` | 1/2 | P6 |
| B19 | `if(c, integer, double)` takes the type of the branch taken | 59513a2 | 2 | P2 |
| B20 | `if` with a missing condition returns the ELSE branch | b4efd7d, probe `if_miss` | 3 | P4 |
| B21 | `0/0` gives NaN (spec: missing); integer `1/0` gives INF; `floor/ceil/round/modulo` of INF are not invalid | 59513a2, probes `div_0_0`, `idiv_1_0`, `floor_inf` | 3 | P4 |
| B22 | Apply `invalidValueTreatment="asIs"` returns missing after an error (spec: same as returnInvalid); `asMissing` prefers mapMissingTo to defaultValue | 59513a2 | 3 | P4 |
| B23 | `concat` drops number arguments: `concat(d,"|")` is `|` | d23902a, probe `concat_num` | 1/3 | P4 |
| B24 | integer `2` into a boolean OutputField gives `true`; a non-integral value in an integer OutputField of a selectAll collection is written as a double; a regression with an `integer` target returns a non-integral double | b4efd7d, bedf08f, review N5 (`design-review/p/b.pmml`: JPMML TypeCheckException) | 1 | P4 |
| B25 | an `Apply` with an `Extension` child fails to load | c9e8731 | 2 | P2 |
| B26 | AnomalyDetection's outer MiningSchema is applied on text and written back (`-999` → `-999.0`); an inner MiningField with no attributes re-checks the value against its Interval; an inner `asIs` is refused by the SVM's own `validate` | 90847c1 | 4 | P7, P8 |
| B27 | the displayValue leaks as the prediction (top-level `as_string`, modelChain's last step, `predict()` uses `display=true`) | 3547cd9, 01c1ecc, b4efd7d | 4 | P8 (+Q5) |
| B28 | classes are text: `"1"` vs `"1.0"` reconciled after the fact (`present()`, `label_text`, `Category::same`) and probability keys renamed in place | 3547cd9, ceb6cec | 1/4 | P8 |
| B29 | ScoreDistribution order lost in an `unordered_map`: ties break differently from JPMML | c57731f, 01c1ecc | 4 | P8 |
| B30 | a class with a prediction but no distribution: only the first TargetValue keeps its prior | 3547cd9 | 4 | P8 |
| B31 | a one-sided outlier bound (`lowValue` only) is ignored | 90847c1 | 4 | P7 |
| B32 | date, time, dateTime and the `*Since` types are read as XSD doubles (`2020-01-01` is invalid) | 6914ecd, c9e8731 | 1 | P10 |
| B33 | `pmml_tester` checks only the `output_name()` column, never the target column on its own: 38 fixtures whose `output_name()` is an OutputField had an unchecked target column. Fixed in 4298b39; a model with a residual OutputField keeps the actual value in that column, an input (Output.html), and it is not checked | ceb6cec, 4298b39 | test | P0 (done) |
| B34 | OutputField `@segmentId` is not supported ("output fields from other segments may always be included by specifying the segmentId attribute", MultipleModels.html) | review N6 | 4 | P8 |
| B35 | a MapValues InlineTable with a `NaN` key over a float or double field crashes the load (`TreeTableNode::add` dereferenced the null slot a second, never-matching lookup inserted) | fuzzer v0 `s20261118_c000152`, 16 crash cases | 3 | fixed in 7146160 (`fix_mapvalues_nan_key`); the §2.7 rule (NaN key is a load error, D26) in P2.2 |
| B36 | `and`, `or`, `isValid`, `isNotValid`, `formatNumber` are refused at load | fuzzer v0 | — | P4.8, P7b.5, P4.9 |
| B37 | an empty input cell `""` is treated as missing; for a string field it is the empty string, for any other type it is not in the lexical space, so invalid | fuzzer v0 (JPMML: empty string, or invalid for a number) | 1 | P6.8 |
| B38 | a numeric field computed over a string input uses the string's internal code: `"3"` gives `-1.0` | fuzzer v0 | 1 | P6.8 |
| B39 | a `NaN` input to a float or double field is rejected as invalid; the XSD float and double value spaces contain NaN | fuzzer v0 | 1 | P6.8 |
| B40 | an `INF` input to Discretize gives missing instead of the open last bin (`rightMargin` absent means +∞) | fuzzer v0 | 3 | P4.4 |
| B41 | built-ins outside their domain keep scoring: `substring` with `startPos` < 1, `sqrt` of a negative, `ln`/`log10` of a non-positive, `modulo` by 0 (spec: invalid, then invalidValueTreatment) | fuzzer v0 | 3 | P4.2 |
| B42 | two MapValues rows whose keys are equal once typed are accepted; one of them silently wins | fuzzer v0 | 3 | P2.2 |
| B43 | MiningField `invalidValueTreatment="asIs"` against an Interval throws; the spec says the value is used as is | fuzzer v0 checks | 4 | P7b.5 |
| B44 | `equal(INF, INF)` is true; BuiltinFunctions.html: comparing ±INF with itself "is indeterminate and will result in a missing value" (JPMML is also true, so the fuzzer cannot see it: hand-written fixture) | review of D9 | 1 | P6.3 |
| B45 | the validation error for an invalid value in a field the model does not read does not name the field | fuzzer v0 | — | P2.1 (diagnostics) |
| B46 | weightedMajorityVote probabilities differ from JPMML in the last digit on large ensembles (summation order) | fuzzer v0 checks | 4 | P8.14 |

### 1.3 Probes behind the table

Each probe is one OutputField (`dataType="string"`) in a RegressionModel over
fields `s` string, `f` float, `i` integer, `d` double, `b` boolean, scored one
field per document because JPMML aborts a whole file on the first error
(`design/probe/run.py`). Rows: `(s,f,i,d,b) = (3,0.1,3,3.5,true)
(3.0,0.1,3,1.5,1) (10,0.1,4,2,false) (abc,0.1,4,2.0,missing)`.

| probe | JPMML 1.7.7 | cPMML `b4efd7d` | this design |
|---|---|---|---|
| `equal(s, integer 3)` | true false false false | false false false false | true false false false |
| `equal(integer 3, s)` | NumberFormatException on `3.0` | false ×4 | true false false false (symmetric) |
| `equal(f, double 0.1)` | true ×4 (double narrowed to float) | false ×4 | false ×4 (compared in double) |
| `equal(double 0.1, f)` | false ×4 | false ×4 | false ×4 |
| `lessOrEqual(f, double 0.1)` | true ×4 | false ×4 | false ×4 (consistent with `equal`) |
| `lessThan(s, integer 10)` | "Categorical value cannot be used in comparison operations" | true ×4 | load error (§2.7) |
| `+(s, integer 1)` | "No PMML data type for the intersection of string and integer" | 0.0 ×4 | load error |
| `equal(i, double 3.5)` | TypeCheckException | false ×4 | false ×4 (compared in double) |
| `/(double 0, double 0)` | N/A | NaN | missing |
| `/(integer 1, integer 0)` (asMissing) | N/A | Infinity | invalid → missing under asMissing (D20) |
| `floor(double INF)` (asMissing) | ArithmeticException: Integer overflow | Infinity | invalid → missing |
| `if(b, "T", "E")`, b missing | MissingArgumentException | E | missing |
| `if(b, ...)` with `mapMissingTo="M"` | T T E M | T T E E | T T E M |
| DerivedField integer `= d * 1.0`, d=3.5 | TypeCheckException | 3.5 | invalid record (Q2) |
| `NormDiscrete(i, "3")` | 1.0 0.0 | 0.0 0.0 | 1.0 0.0 |
| `equal(uppercase(s), "ABC")`, s=`abc` | true | false | true |
| `equal(s, s2)` for unseen `abc`/`yy` | false | true | false |
| `uppercase(<Constant>abc</Constant>)` | ABC | N/A | ABC |
| `concat(d, "|")` | `3.5|` `2.0|` | `|` `|` | `3.5|` `2.0|` (Q1) |
| NormDiscrete on a field declared later | 1.0 0.0 | 1.0 0.0 | 1.0 0.0 (two-phase binding) |
| integer `x`, `SimplePredicate x lessOrEqual "2.5"` (review `tree_int.pmml`) | a a b for x = 1 2 3 | a a b | a a b (§3.4 numeric document text) |
| Discretize without `dataType` in a double / a string DerivedField, `binValue="1.50"` | `1.5` / `1.50` | `1.5` / `1.50` | same (§3.4) |

### 1.4 What the two failed attempts teach

`followups5/type_value_tag` and `followups6/typed_values` both added a kind tag
to the existing `Value` and converted at the edges, and both bundled the
change of representation with a dozen semantic fixes. Their verifiers
(journals `wf_939b3d1b-5d4`, `wf_6cb240fd-0a2`) rejected them for these
reasons:

| lesson | what broke | design answer |
|---|---|---|
| L1 | the repair narrowed all float document text, so NormContinuous breakpoints inside a float DerivedField became floats (0.49999997 where 0.5); in a string DerivedField they became string codes | an attribute's type is its **XSD type** (`LinearNorm@orig` is `NUMBER` = `xs:double`), decided by the binder per attribute (§3.4), never by the enclosing field |
| L2 | Interval margins on a float field read as float; JPMML and the XSD say `NUMBER` | same: `Interval@leftMargin` is `NUMBER` → `Double`; a `Float` input is widened to double to compare (§2.7) |
| L3 | NormDiscrete asked the referenced field's type at load and crashed on a forward reference | two-phase binding: all declarations of a container are typed before any body is resolved (§3.5) |
| L4 | `equal` and `lessOrEqual` disagreed on floats | one comparison visitor returns `std::partial_ordering`; every comparison and predicate derives from it (§2.7) |
| L5 | a FLOAT-tagged value could still hold an unnarrowed double, so two floats compared unequal | `Float` holds a C++ `float` (§2.1) |
| L6 | `Value::sum` seeded with `double_min()`; typed results were "whatever the function returned" | promotion is one table (§2.5); result types come from one function table at load (§4) |
| L7 | +5.3% on IrisTree from a 1-byte tag and per-compare branching | `Value` shrinks to 16 bytes, trivially copyable; hot comparisons bound per type at load, no `std::visit` per record (§6) |
| L8 | representation change and semantic fixes in one diff; regressions found one site at a time | each representation change is behaviour-preserving and proven so by a dump diff of every fixture; each semantic fix is its own commit with its fixtures (§7) |

---

## 2. The type model

### 2.1 The variant

```cpp
namespace cpmml::types {

enum class TypeId : std::uint8_t { Missing, Invalid, Integer, Float, Double, Boolean, String };  // == variant index

struct Missing {};
enum class InvalidReason : std::uint8_t { Malformed, NotIntegral, Overflow, DivisionByZero, Domain, NoCast };
struct Invalid  { InvalidReason reason; };

struct Integer  { std::int64_t v; };          // xs:integer in 64 bits; beyond is Invalid(Overflow)
struct Float    { float v; };                 // xs:float: always an IEEE single
struct Double   { double v; };                // xs:double
struct Boolean  { bool v; };                  // xs:boolean
struct String   { const StringRep* rep; };    // xs:string; text is identity (§2.3)

using Value = std::variant<Missing, Invalid, Integer, Float, Double, Boolean, String>;
}
```

Measured (sketch, all three compilers): `sizeof(Value) == 16`, `alignof == 8`,
trivially copyable and trivially destructible. Today's `Value` and `Feature`
are 40 bytes each (measured on `b4efd7d`, `-O3`, arm64).

Spec basis. DataDictionary.html: "The dataType reuses names and semantics of
atomic types in XML Schema. The representation of the values is defined in XML
Schema Part 2: Datatypes."

The PMML date/time types (`date`, `time`, `dateTime`, `timeSeconds`,
`dateDaysSince[Y]`, `dateTimeSecondsSince[Y]`) become alternatives only in P10
(Q8). The sketch builds both ways (`-DPMML_WITH_TEMPORAL=1` gives 19
alternatives, still 16 bytes). Because visitors have no catch-all (§2.2),
adding them in P10 turns every visitor into a compiler-guided checklist; until
then the binder maps those dataTypes to `Double` exactly as today
(`datatype.h`), so no temporal value exists.

`Missing` and `Invalid` are alternatives so that a function can return them
without an exception. A frame slot never holds `Invalid`, except as the
recorded failure of a lazily evaluated DerivedField, which is raised when the
slot is read (§5.3). An input kept "asIs" while invalid keeps its own typed
value plus a validity flag in the frame (§2.8, §5.2).

### 2.2 Concepts and exhaustiveness

Every alternative satisfies `PmmlType`; the parsed ones also `PmmlParsedType`;
the numbers also `PmmlNumeric`. Operations are free functions found by ADL so
the concepts can require them (`design/sketch/typed_value.h`):

```cpp
template <class T>
concept PmmlType = requires(const T& a, const T& b) {
  { T::type_id } -> std::convertible_to<TypeId>;
  { compare(a, b) } -> std::same_as<std::partial_ordering>;  // within one type
} && index_matches<T>();                                       // variant index == TypeId

template <class T>
concept PmmlParsedType = PmmlType<T> &&
    requires(std::string_view text, ParseContext& ctx, const T& a, std::string& out) {
      { parse(std::type_identity<T>{}, text, ctx) } -> std::same_as<std::optional<T>>;  // strict XSD (lexical.h)
      { java_text(a, out) } -> std::same_as<void>;                                     // java_format.h
      { T::name } -> std::convertible_to<std::string_view>;                            // the dataType spelling
    };

template <class T>
concept PmmlNumeric = PmmlParsedType<T> && requires(const T& a) {
  { a.as_double() } -> std::same_as<double>;
  { T::rank } -> std::convertible_to<int>;   // Functions.html precedence: integer 0, float 1, double 2
};

template <class T> concept Sentinel = std::same_as<T, Missing> || std::same_as<T, Invalid>;
template <class T> concept Textual  = std::same_as<T, String>;
template <class T> concept Logical  = std::same_as<T, Boolean>;
template <class T> concept Temporal = PmmlType<T> && !PmmlNumeric<T> && !Textual<T> && !Logical<T> && !Sentinel<T>;  // empty until P10

static_assert(all_categorised(static_cast<Value*>(nullptr)));   // every alternative in exactly one category
```

Visitors are overload sets constrained by these categories with **no
unconstrained catch-all**. Two must-not-compile tests prove it on all three
compilers (`design/sketch/build.log`):

| test | clang 21 | g++-16 | em++ 7.0 |
|---|---|---|---|
| a visitor without a `String` overload | `static assertion failed ... is_invocable_v<Incomplete, const pmml::String &>` | `no type named 'type' in std::invoke_result<Incomplete, const pmml::String&>` | `no matching function for call to '__invoke'` |
| a new alternative in no category | `static assertion failed ... all_categorised` | `static assertion failed` | `static assertion failed ... all_categorised` |

In the library these become CMake `try_compile` checks expected to fail
(`test/types_negative/`).

### 2.3 Strings: text is identity, interning is a cache

```cpp
struct StringRep { std::string_view text; std::uint32_t id; std::size_t hash; };  // id == kUnpooled for runtime strings

class StringPool {                 // ONE per document, shared by every model and segment; frozen after load
 public:
  const StringRep* intern(std::string_view);          // load only
  const StringRep* find(std::string_view) const;      // scoring: read-only, thread-safe
};
class RecordArena {                // per evaluation, heap-allocated on first use (§5.1)
 public:
  const StringRep* make(std::string_view);            // uppercase(), concat(), casts, unseen inputs
};

inline std::partial_ordering compare(String a, String b) {         // ordering
  if (a.rep == b.rep) return std::partial_ordering::equivalent;
  return a.text().compare(b.text()) <=> 0;
}
inline bool string_equal(String a, String b) {                     // equality
  if (a.rep == b.rep) return true;
  if (a.rep->id != StringRep::kUnpooled && b.rep->id != StringRep::kUnpooled) return false;  // both pooled, distinct
  return a.text() == b.text();
}
```

Equality is equality of text (XSD `string`), so B1 and B2 cannot happen: an
unseen input gets an arena rep of its own text, and `uppercase` makes a new
rep. Because there is one pool per document, two pooled reps are equal exactly
when they are the same pointer, so the common case (a pooled input against a
pooled predicate constant, in any segment) never reads text. The hash is
`std::hash<std::string_view>` of the text, computed once per rep, consistent
across pool and arena. `StringRep` is 32 bytes native, 16 on wasm32.

DataDictionary.html: "Trailing blanks in categorical input values are not
significant, and categorical values in PMML must not have trailing blanks.
Leading blanks are significant." The MiningSchema step drops trailing XML
blanks of a categorical string input before lookup (its own commit, F6 in
P6).

### 2.4 The visitor set

| visitor | signature | used by |
|---|---|---|
| `Comparison` | `optional<partial_ordering>(const Value&, const Value&)` | built-in comparisons, isIn, SimplePredicate (generic path), Target class lookup, ensemble votes |
| `CastTo<T>` | `Value(const Value&)` | Cast nodes (§2.6), Constant typing, MiningField replacement values |
| `Arithmetic<Op>` | `Value(const Value&, const Value&)` | `+ - * / pow modulo threshold hypot`, aggregates fold |
| `Unary<Op>` | `Value(const Value&)` | `exp ln sqrt abs floor ceil round rint …` |
| `JavaText` | `void(const Value&, std::string&)` | API boundary, number→string casts (Q1) |
| `KeyHash`, `KeyEqual` | reflexive hashing/equality for containers (§2.7, "hash containers") | MapValues tables, SimpleSetPredicate > 150 members, NaiveBayes counts, distributions |
| `ParseAs<T>` | `Value(std::string_view, ParseContext&)` | inputs, document text |

`std::visit` is the generic path. The binder instantiates monomorphic
operations wherever the static types are known — after binding, everywhere on
the scoring path — so generic visitors run only in cold code. That rule is
enforced by the API shape, not by grep: generic entry points live in
`cpmml::types::cold` and take a `ColdPath` tag argument that only load code,
the API layer and tests construct (§6.4).

### 2.5 Numeric promotion: one table

Functions.html: "functions are assumed to inherit the dataType of the current
input parameters unless otherwise specified ... When the input parameters have
multiple dataTypes, the least restrictive dataType will be inherited by
default ... The inheritance precedence for mixed type input parameters is as
follows: string, double, float, integer" and "if an integer and a double
parameter are used with the "+" function, by default the output dataType will
be a double."

```cpp
template <PmmlNumeric A, PmmlNumeric B>
using Promoted = std::conditional_t<(A::rank >= B::rank), A, B>;

template <PmmlNumeric To, PmmlNumeric From>
  requires(To::rank >= From::rank)      // "implicit casts are allowed in the direction of less restriction"
constexpr To widen(From x);
```

| | integer | float | double |
|---|---|---|---|
| **integer** | integer | float | double |
| **float** | float | float | double |
| **double** | double | double | double |

`string` is above `double` in the list but no arithmetic accepts it: the
binder refuses `+(string, …)` at load (JPMML: "No PMML data type for the
intersection of PMML data types string and integer"). Functions.html: "The
various date, time and dateTime as well as boolean dataTypes can not be mixed
with other types in a defined function without explicitly defining the
expected output dataType", so boolean (and, from P10, temporal) operands are
outside the table (Q7).

Integer arithmetic is exact in `int64_t` and overflow is `Invalid(Overflow)`
(`__builtin_add_overflow`/`_mul_overflow`, available in clang, gcc and
emscripten). The XSD integer is unbounded; JPMML uses 32-bit `int` (Q10, D4).

### 2.6 Cast rules

The binder inserts **implicit** casts only toward less restriction, in
argument positions (Functions.html: "implicit casts are allowed in the
direction of less restriction, according to the precedence list above"). It
inserts **explicit** Cast nodes where a declaration fixes a type: a
DerivedField's or OutputField's `dataType`, a Constant's `dataType`, a
MapValues/Discretize `dataType`, a MiningField replacement, the target field
(§5.1). "spec" means the rule is quoted, "XSD" that it follows from the value
spaces, "XPath" that the spec is silent and we follow XPath casting
(BuiltinFunctions.html: "Further ideas are taken from MathML, XPath, Java"),
"JPMML" that we follow the oracle.

| from → to | rule | basis | JPMML 1.7.7 |
|---|---|---|---|
| integer → float, double | widen (int64 → float rounds to nearest) | spec (less restriction) | same |
| float → double | exact | spec | same |
| double → float | IEEE round to nearest; overflow → ±INF, underflow → 0 | XSD float value space | same (`1e300` → `Infinity`) |
| double, float → integer | the value if integral and in range, else `Invalid(NotIntegral/Overflow)` | XSD; Functions.html "a double expression can not be implicitly cast as an integer" | TypeCheckException |
| number → string | Java text (`7`, `0.1`, `2.0`, `1.0E7`) | spec silent; Q1 | Java text |
| string → number, boolean | strict XSD reading of the text (`lexical.h`), else `Invalid(Malformed)` | XSD | lenient |
| boolean → number | true 1, false 0 | XPath; shipped in b4efd7d | same |
| number → boolean | 1 → true, 0 → false, else `Invalid(NoCast)` | XSD boolean lexical space `{true,false,1,0}`; no spec cast | double 1.0 → true; integer refused (D14) |
| boolean → string | `true` / `false` | XSD canonical | same |
| string → string, T → T | identity | — | — |
| anything else | load error when static; `Invalid(NoCast)` otherwise | — | — |

What an `Invalid` from an explicit cast does depends on where the cast sits:
in an Apply it goes to the Apply's `invalidValueTreatment` (§4.3); in a
DerivedField, OutputField or the target cast, which have no treatment
attribute, it is an invalid result for the record (Q2); in a MiningField
replacement it is handled by the slot-widening rule (§3.4).

Constant text with `dataType="float"` or `"double"` accepts `INF`, `-INF`,
`NaN` **case-insensitively** (Transformations.html: "These three constants are
case insensitive and only meaningful as floating point values"); everywhere
else — inputs, attributes, Array members — the XSD lexical forms stay strict
and case-sensitive as `lexical.h` implements them (D11).

### 2.7 Comparison rules

All comparisons come from one visitor returning
`std::optional<std::partial_ordering>`: `equal` is `equivalent`,
`lessOrEqual` is `less || equivalent`, and so on, so equality and ordering
cannot disagree (L4). `nullopt` is "no common type", which the binder refuses
at load, or a sentinel operand, which the caller turns into missing
(built-ins) or UNKNOWN (predicates).

| operands | compared as | basis | JPMML 1.7.7 | decision |
|---|---|---|---|---|
| numeric × numeric | the promoted type (§2.5), either order | Functions.html least-restrictive rule | converts the right operand to the **left** operand's type (`equal(f, 0.1)` true, `equal(0.1, f)` false) | spec, symmetric (D1); shipped for built-ins in 59513a2 |
| string × number, **equality** (`equal`, `notEqual`, `isIn`, `isNotIn`) | the number as text (§2.6), either order | string is the least restrictive type | text if the string is left; parses the string if the number is left | spec, symmetric (D2) |
| string × number, **ordering** | load error `E-TYPE-ORDER-MIXED` | DataDictionary.html "Categorical field values can only be tested for equality"; ordering text against a number (`"10" < "3"`) has no meaning the spec gives | "Categorical value cannot be used in comparison operations" | load error |
| string × string, equality | text | XSD | same | — |
| string × string, ordering | an ordinal field with Values: the declared order. An ordinal string without Values: code-unit order plus warning `W-ORDINAL-STRING-NO-VALUES`. A categorical operand: code-unit order plus warning (Q6) | DataDictionary.html: "For ordinal fields, if a sequence of Value elements is given, the sequence defines the ascending order"; "If no Value elements are given, the order is always assumed to be the ascending order of the representation type ... The notable exception to this rule are ordinal DataFields of type string: Ordinal DataFields of type string must always specify the order explicitly" | declared order; refuses categorical | as stated |
| boolean × boolean | false < true | XSD | same | — |
| boolean × other | no common type: load error | Functions.html "boolean dataTypes can not be mixed with other types" | evaluates some pairs | Q7 |
| any × missing | missing / UNKNOWN | BuiltinFunctions.html "any missing inputs to a built-in function will result in a missing value"; predicates' three-valued logic | missing / MissingArgumentException | spec |
| ±INF × the same INF | `unordered` → missing / UNKNOWN | Transformations.html "infinity is not single number, but a class of numbers ... the first infinity is not equal to the second"; BuiltinFunctions.html (lessThan): "any attempt to either with itself is indeterminate and will result in a missing value being returned" | IEEE (`INF == INF` true) | spec (D9) |
| NaN × anything | `unordered` → missing / UNKNOWN | **extrapolation**: the spec says only "All arithmetic expressions involving NaN will evaluate to NaN" | IEEE (false) | D9, marked as extrapolation |

Predicates use the same semantics, bound per type (§3.4). Interval margins and
`lowValue`/`highValue` are `NUMBER` = `xs:double`, so a float input is widened
to compare (L2).

**Hash containers.** The comparison visitor is not reflexive (INF, NaN), so
containers (MapValues tables, SimpleSetPredicate with more than 150 members,
NaiveBayes counts, distributions) use `KeyHash`/`KeyEqual`: same alternative,
same payload, `-0.0` equal to `0.0`, pooled strings by pointer. Keys are of one
type (the binder casts them at load). A NaN key is a load error (it can match
nothing). A probe that would compare unordered with some key — NaN, or ±INF
when the container holds the same infinity — is decided before the lookup:
the result is "indeterminate" (missing; UNKNOWN in a predicate) if the
container holds that infinity (a flag computed at load), "not found"
otherwise. The linear path below 150 members gives the same answers, so the
two paths cannot disagree (sketch: `KeyHash`, `KeyEqual`, `is_nonfinite`).
NaiveBayes counts and distributions are keyed by class labels, never probed
with non-finite values.

### 2.8 Missing versus invalid

DataDictionary.html: "Missing value: Input value is missing ... Invalid value:
The input value is not missing but it does not belong to a certain value range
... Valid value: A value which is neither missing nor invalid."
BuiltinFunctions.html: "An invalid value is not missing."

| source | missing | invalid |
|---|---|---|
| input absent, empty cell | missing → `missingValueReplacement` if given | — |
| input equal to a `<Value property="missing">` | missing; the raw text is compared first (a non-numeric marker such as `?` on a double field, as in `fix_datadict_missing_marker`), then, for a numeric field, the numeric values (`-999.0` equals the marker `-999`, B6) | — |
| input text not in the field's lexical space | — | invalid (malformed); `asIs` cannot keep a value that does not exist, so it is a record error as today |
| `<Value property="invalid">`, not in the valid set, outside every `Interval` | — | invalid → `invalidValueTreatment` |
| `invalidValueTreatment="asMissing"` | becomes missing → replacement | — |
| `invalidValueTreatment="asValue"` | — | replaced by `invalidValueReplacement` (MiningSchema.html: "asValue specifies that an invalid input value should be replaced with the value specified by attribute invalidValueReplacement which must be present in this case, or the PMML is invalid"; implemented at `miningfield.h:85-88`, missing replacement already a load error) |
| `invalidValueTreatment="asIs"` | — | keeps the typed value; the slot's validity flag is set so `isValid`/`isNotValid` answer correctly |
| `outliers="asExtremeValues"` | — | clamped to `lowValue`/`highValue` (`NUMBER`); on an integer field a non-integral bound widens the slot (§3.4) |
| function results | `0/0`, a missing input (§4.3) | `x/0` of integers (D20), `floor(INF)`, overflow, a domain error, a failed cast |
| model | no prediction | — |

MiningSchema.html: "returnInvalid is the default and specifies that, when an
invalid input is encountered, the model should return a value indicating an
invalid result has been returned. asIs means to use the input without
modification. asMissing specifies that an invalid input value should be
treated as a missing value and follow the behavior specified by the
missingValueReplacement attribute if present". "An invalid result" is, as
today, `cpmml::InvalidValueException` naming the field (or the DerivedField
and function), raised by the pipeline, never by a function body.

### 2.9 Java text

`java_text` for each parsed type is the existing writer
(`src/utils/java_format.h`): `Integer` → `Long.toString`, `Float` →
`Float.toString` via `write_float`, `Double` → `Double.toString` via
`write_double`, `Boolean` → `true/false`, `String` → its text. It is used at
the API boundary (`Result::as_string`, `str_outputs`, distribution keys,
selectAll collections) and, subject to Q1, for number → string casts. Because
the type is in the value, `java::Kind`, `java_kind(DataType, bool integer)`,
`InternalModel::Shown`, `label_text` and `OutputExpression::value_kind` all go
away.

### 2.10 Compiler notes

| | Apple clang 21 | g++-16.2 | em++ 7.0 (wasm32) |
|---|---|---|---|
| sketch compiles with `-std=c++20 -Wall -O3` | yes | yes (one pre-existing `-Wmaybe-uninitialized` in `java_format.h:175`) | yes, with `-fexceptions` as the library already sets |
| `sizeof(Value)` / `sizeof(StringRep)` | 16 / 32 | 16 / 32 | 16 / 16 |
| negative compile tests | rejected | rejected | rejected |
| ns per compare, 1M streamed values: old `Value` / bound fast path / generic `std::visit` | 0.47–0.99 / 0.64–1.09 / 5.4–6.8 | 0.68–1.24 / 0.57–0.74 / 3.4–3.6 | 0.52–0.57 / 0.96–1.40 / 21.5–24.2 |

The streaming micro-benchmark measures memory bandwidth more than tree
evaluation. The review's `design-review/tree.cc` (depth-12 tree over
L1-resident 16-slot frames) found the variant `get_if` path at about 11 ns per
11-compare traversal against 36 ns for the old layout on clang 21 / M4. Both
are only plausibility checks; the evidence for performance is the benchmark
gate of §6.4 on the library itself. What the numbers do establish: the generic
visit is 4–7× a plain compare natively and ~25× under Emscripten, so the
scoring path must not call it per record.

---

## 3. The bind/type-check pass

### 3.1 What it produces

The binder runs once per document, after XML parsing and before any model
object is built. Its output is:

- a `FrameLayout`: every field instance in every scope gets a `SlotId`; slots
  of a pass-through are aliases;
- for every expression a typed node with a static `TypeId`, with explicit
  `Cast` nodes where a type changes;
- for every model a `ModelBinding`: its MiningField steps, the DerivedFields it
  instantiates, its OutputFields in dependency order;
- `Diagnostics`: errors (load fails) and warnings (kept, Q9).

Models are built from the binding: wherever a model today calls
`indexer->get_index(name)` or `indexer->get_type(name)` it calls
`scope.resolve(name, node)` and receives a `BoundField{slot, type, optype,
order}`.

### 3.2 Scope model

FieldScope.html is quoted for each rule.

```
Document scope            DataDictionary DataFields; TransformationDictionary *definitions*
└─ Model scope M          (top-level model element)
   ├─ MiningFields of M   "Within a model element, all data must flow in through its MiningSchema."
   │                      top level: "their MiningSchema can refer only to DataFields in the DataDictionary"
   ├─ TD instantiations   "A DerivedField from the TransformationDictionary is instantiated in a model
   │                      element when it is referenced in the model element, either directly or
   │                      transitively through other DerivedFields." They "can only reference:
   │                      MiningFields from the MiningSchema of that model element [and] Other
   │                      DerivedFields which appear earlier in the TransformationDictionary."
   ├─ LocalTransformations of M   "a model can also define its 'own' derived fields"
   ├─ chain outputs       modelChain: "all Outputs from the segment models become part of the enclosing
   │                      mining model scope" — OutputField slots of earlier segments live in M
   ├─ Segment predicates  evaluated in M
   ├─ Output of M         may reference LocalTransformations of M (the one allowed forward reference)
   └─ Segment k → child model scope C_k
        MiningFields of C_k resolve in M: "their MiningSchema can refer to any fields defined within
        the parent model's scope, including The MiningFields of the parent model, DerivedFields defined
        in the LocalTransformations of the parent model ... OutputFields defined in Segments that appear
        above/earlier in the parent MiningModel"
```

Lookup of a name from an expression in model scope M, nearest first:

1. M's OutputFields (only from inside M's `Output`);
2. M's LocalTransformations DerivedFields;
3. TransformationDictionary DerivedFields, instantiated in M on first reference;
4. M's MiningFields;
5. *not found*: error `E-SCOPE-UNRESOLVED`, or, under Q3's recommended
   lenience, an implicit pass-through MiningField resolved in the enclosing
   scope plus warning `W-SCOPE-IMPLICIT`.

Naming rules become load errors where FieldScope.html says "must": "The names
of DerivedFields in the LocalTransformations must be unique from any other
names in their scope", "The names of OutputFields must be unique from any
other names in their scope, i.e. within a model or model segment and across
all model segments in a model chain", "The names of MiningFields cannot be
re-used within the same model element". The two allowed duplications get
separate slots: "DerivedField names in LocalTransformations can be duplicated
across different sibling sub-models" (B16) and "For model ensembles,
OutputField elements can be duplicated across different sibling sub-models"
(each sibling's OutputField has its own slot; outside a modelChain they are
not visible to the parent by name, and the MiningModel reports the one "that
comes from the Segment selected by the multipleModelMethod attribute").

**Aliasing.** A child MiningField with no treatment attributes (no
`missingValueReplacement`, `invalidValueTreatment` other than the default,
`outliers`, `optype` change) is an alias of the parent's slot: no step, no
copy. This is the common case in forests (AuditRandomForest has 200 segments).
JPMML passes such values through without re-validating (90847c1 known gap,
B26).

**TD instantiation and sharing (once per record).** Instantiation is per
model as the spec says; evaluation is shared whenever the values must be
equal. The binder gives a TD DerivedField instance the key
`(DerivedField id, the resolved slots of its MiningField inputs after
aliasing)` and places the instance in the **highest scope in which every one
of those slots is defined** — not the scope that referenced it. So if 200
sibling segments reference TD field `X` and all their MiningFields for `X`'s
inputs are aliases of the parent's slots, the 200 instances have one key, one
slot in the parent scope, and `X` is computed at most once per record, exactly
as `derivedfields_dag` does today — even though the parent itself never
references `X`. A segment whose MiningSchema treats an input (a replacement,
say) gets a different input slot, hence a different key and its own instance,
which is what the spec's per-model instantiation requires. The instance is
lazy (§5.3): it is computed on its first read in a record, by whichever
segment reads it first; under MULTITHREADING the parent computes the shared
instances its parallel children read before forking (§5.2).

### 3.3 Data structures

```cpp
namespace cpmml::bind {

using SlotId = std::uint32_t;
using ScopeId = std::uint32_t;

struct FieldType {
  types::TypeId type;                    // the slot's static type (§3.4 widening)
  OpType optype;
  const OrdinalOrder* order = nullptr;   // declared Value order of an ordinal field
};

enum class Origin : std::uint8_t { DataField, MiningField, DerivedField, OutputField, TargetPlaceholder };

struct Binding {                 // a name visible in one scope
  Origin origin;
  FieldType type;
  SlotId slot;
  const XmlNode* declaration;
  enum class State : std::uint8_t { Declared, Typing, Typed } state;   // two-phase (§3.5)
};

class Scope {
 public:
  enum class Kind : std::uint8_t { Document, Model, Output };
  const Binding* local(std::string_view name) const;
  const Binding& resolve(std::string_view name, const XmlNode& from, Diagnostics&);  // §3.2 rules
  ScopeId id() const;
  const Scope* enclosing() const;
};

struct SlotInfo {                // FrameLayout entry; also what DEBUG builds print
  std::string name; types::TypeId type; ScopeId scope; Origin origin;
  SlotId alias_of;               // == own id unless a pass-through
  bool lazy;                     // DerivedField: evaluated on first read (§5.3)
};

struct FrameLayout {
  std::vector<SlotInfo> slots;
  std::vector<std::pair<SlotId, SlotId>> scope_ranges;   // per scope: [first, last) — contiguous, so a
};                                                        // scope's state bytes never share a cache line
                                                          // with a sibling's (§5.2)
struct MiningFieldStep {         // one per non-alias MiningField
  SlotId source;                 // parent slot, or kInputSlot for a top-level field
  SlotId target;
  const MiningFieldSpec* spec;   // typed markers, valid set, intervals, replacements, treatments
};

struct ModelBinding {
  ScopeId scope;
  std::vector<MiningFieldStep> mining;
  std::vector<BoundDerivedField> derived;      // instances placed in this scope (§3.2 sharing)
  std::vector<BoundOutputField> outputs;       // dependency order
  BoundTarget target;                          // target field type, TargetValues typed
  std::vector<ModelBinding*> children;         // segments
};

class Binder {
 public:
  explicit Binder(const XmlNode& pmml);
  ModelBinding& bind_top(const XmlNode& model_element);
  ModelBinding& bind_child(const XmlNode& model_element, ModelBinding& parent, const ChainState* chain);
  std::unique_ptr<BoundExpr> bind_expression(const XmlNode& expr, Scope&, Context);
  const FrameLayout& layout() const;
  Diagnostics& diagnostics();
};
}
```

Typed expression nodes keep today's shape (a virtual `eval`) so the
interpreter stays familiar; each node is a concrete instantiation chosen by
the binder:

```cpp
struct BoundExpr {
  types::TypeId type;                              // static type; eval returns this or Missing
  virtual types::Value eval(EvalContext&) const = 0;
  virtual ~BoundExpr() = default;
};
template <class Op, class T> struct ArithNode;     // e.g. ArithNode<Plus, Double>
template <class T> struct FieldRefNode;            // reads frame[slot] (lazy check if derived)
template <class From, class To> struct CastNode;
struct ApplyNode;                                  // §4.3 wrapper around a FunctionKernel
```

### 3.4 Typing rules per element

`T(field)` is the referenced field's declared type. Every attribute is read
with its XSD type from `pmml-4-4.xsd`, not with the enclosing field's type
(L1, B11).

**Document text in a field's type (review K1).** Many attributes are
`xs:string` holding "a value of the field" (`SimplePredicate@value`,
`SimpleSetPredicate` Array members, `CategoricalPredictor@value`, MapValues
key cells, `Value@value`, missing markers, `NormDiscrete@value`, `PPCell@value`,
`missingValueReplacement`, `invalidValueReplacement`). Producers write integer
fields' values with a decimal point:
`benchmark/data/model/single_audit_dectree.xml` declares `Age` with
`dataType="integer"` (line 7) and tests `<SimplePredicate field="Age"
operator="lessOrEqual" value="33.0"/>` (line 147). The rule, for a field of
numeric type T:

1. the text is a T literal → a value of T;
2. else it is an `xs:double` literal → a **`Double`**; the comparison promotes
   the field's value to double (§2.5, §2.7);
3. else (non-numeric text) → load error, except a missing marker, which is
   matched as raw text first (§2.8) and is therefore allowed to be anything.

Spec basis: Functions.html allows implicit casts only "in the direction of
less restriction ... an integer expression can be implicitly cast as a double
... but a double expression can not be implicitly cast as an integer". Reading
`33.0` as the integer 33 would be the forbidden narrowing; reading it as the
double it spells and widening the integer field to compare is the allowed
direction. This is also what JPMML does (review probe `tree_int.pmml`: integer
`x`, `value="2.5"`, gives a / a / b for x = 1, 2, 3) and what cPMML does today
(everything is a double). For a `float` field every double literal is a float
literal, so rule 1 applies (`0.1` → `Float{0.1f}`, L5). For `boolean` and
`string` fields rule 2 does not exist: a boolean field's text must be an XSD
boolean, a string field's text is a string.

Where the value is **written into a slot** (`missingValueReplacement`,
`invalidValueReplacement`, an `asExtremeValues` bound on an integer field;
review N3), a rule-2 `Double` cannot be stored in an `Integer` slot. The
binder then **widens the slot**: its static type becomes the least restrictive
of the field type and every document value written into it, so the integer
field `Age` with `missingValueReplacement="2.5"` has a `Double` slot (inputs
are read as integers and widened), with warning `W-SLOT-WIDENED`. MiningSchema.html
says the replacement acts "as if the given value was found in the original
input"; an integer field cannot hold 2.5, so the document is inconsistent with
itself, and widening is the reading that scores it without inventing a
narrowing.

Verified against every fixture and benchmark model (`design/k1/scan.py`, 333
zipped fixtures + 23 benchmark models): text outside the field's lexical space
occurs 15 times — 12 `SimplePredicate@value` on integer `Age` (`27.0`, `33.0`,
`35.0` in `single_audit_dectree`, `ensemble_audit_dectree`, both the benchmark
XML and the zipped fixtures), 2 valid `Value@value="4.0"` on integer targets
(`fix_probability_tree_integer`, `fix_probability_vote_integer`), 1 missing
marker `?` on a double field (`fix_datadict_missing_marker`). Under a strict
"parse as T" rule all 15 are load errors in 7 documents; under the rule above
none is, and every comparison gives today's answer (an integer `Age` against
the double `33.0`, the integer target against the valid value `4.0`).
P2 adds the review's `tree_int.pmml` and a replacement-widening document as
fixtures.

**Absent `dataType` on Discretize and MapValues (review N4).** The XSD gives
no default and the spec text none. The type is the **nearest enclosing
declared type** (the DerivedField's or OutputField's `dataType`): JPMML does
exactly this (probe: `binValue="1.50"` gives `1.5` in a double DerivedField and
`1.50` in a string one), and so does cPMML today, and it is the reading
Functions.html gives ParameterFields ("assumed to inherit the dataType of the
expression"). When the attribute is present it is used (B11). Listed as D21.

| element | static type | attributes and children | notes |
|---|---|---|---|
| `Constant dataType=X` | X | text parsed strictly as X (floating INF/NaN case-insensitive, §2.6); malformed → load error | `missing="true"` → `Missing` of X |
| `Constant` (no dataType) in a function argument whose built-in parameter declares a dataType | **that dataType** | text parsed as it; malformed → load error | Transformations.html: "If the ParameterField definition includes a dataType, the Constant will inherit the dataType specified in the ParameterField" — `if(<Constant>true</Constant>, …)` and `not(<Constant>false</Constant>)` are booleans (review S2) |
| `Constant` (no dataType), otherwise | inferred: `[+-]?digits` integer; with a decimal point or an exponent double; anything else string; empty text the empty string | integer beyond int64 → load error | Transformations.html: "A Constant that consists solely of numbers will be treated as an integer, if a decimal point is present among the numbers the Constant will be treated as a double ... A scientific notation expression like 3.14e+5 or like 3.14E-5 should also be treated as a double. The presence of any non-numeric characters (other than e, E, -, + in scientific notation expressions) will result in the Constant being treated as a string. If the content is empty, then the constant will be interpreted as an empty string". Untyped `NaN`, `INF`, `-INF` (any case) are strings: "if datatype is set to its default value of "string", then the literal strings will be signified" (B9) |
| `FieldRef` | T(field) | `mapMissingTo` (`xs:string`) parsed as T(field) by the rule above | |
| `NormContinuous` | double | `orig`, `norm`, `mapMissingTo` are `NUMBER` → double; field must be numeric; input widened | outliers per Transformations.html |
| `NormDiscrete` | double | `value` (`xs:string`) read by the rule above; non-numeric text on a numeric field → load error; `mapMissingTo` `NUMBER` | "An element (f, v) defines that the unit has value 1.0 if the value of input field f is v" (B10) |
| `Discretize` | its `dataType`, else the nearest enclosing declared type | `binValue`, `mapMissingTo`, `defaultValue` parsed as that type; `Interval` margins `NUMBER` | field must be numeric |
| `MapValues` | its `dataType`, else the nearest enclosing declared type | key cells read by the rule above for each FieldColumnPair field; output cells, `mapMissingTo`, `defaultValue` as the MapValues type | lookup keyed by `KeyHash` |
| `TextIndex` | double | `textField` must be string | |
| `Apply` | from the function table (§4) | `mapMissingTo`, `defaultValue` (`xs:string`) parsed as the **Apply's** result type | children after `Extension*` (B25) |
| `Aggregate`, `Lag` | — | load error "unsupported", as today | |
| `DerivedField dataType=X` | X | `Cast<X>(expression)`; widening implicit, narrowing per §2.6 | `optype` required by the XSD |
| `OutputField` | its `dataType` (XSD: required; inferred with warning when absent, as today) | `predictedValue` → `Cast(dataType, prediction)`; `probability`, `residual`, … double; `transformedValue` → `Cast(dataType, expression)` | B5 |
| `SimplePredicate` | boolean (three-valued) | `value` read by the rule above | operator vs optype: §2.7, Q6 |
| `SimpleSetPredicate` | boolean | `Array` members read by the rule above | >150 members → hash set (§2.7) |
| `CompoundPredicate`, `True`, `False` | boolean | | |
| `MiningField` | the slot type (T, widened if needed) | `missingValueReplacement`, `invalidValueReplacement` read by the rule above; `lowValue`/`highValue` `NUMBER` | |
| `DataField` | its `dataType` | valid/invalid `Value@value` read by the rule above; missing markers raw text first | |
| `Target`/`TargetValue` | target field type | `TargetValue@value` read by the rule above; `defaultValue` `NUMBER`; `min`/`max`/`rescale*` double | |
| model-internal references (NumericPredictor, CategoricalPredictor, NeuralInput, PPCell, KNNInput, ClusteringField, BayesInput, Characteristic, VectorFields, …) | resolved like FieldRef | numeric attributes by XSD type (`NumericPredictor@exponent` is `INT-NUMBER`, B7) | category values by the rule above |
| `DefineFunction`, `ParameterField` (review S16) | — | **out of scope**: refused at load as today (`builtinfunctions.h:251`) | if added later: a user function is a row in a second table beside `kBuiltins` whose result rule is its declared `dataType`, and its body is bound in a scope containing only its ParameterFields (Functions.html: "The function body must not refer to fields other than the parameter fields"); Constants in calls inherit the ParameterField's type as above |

### 3.5 Forward references and order

FieldScope.html: "Forward referencing of fields is not allowed, with one
exception. OutputField with feature="transformedValue" in a PMML model can
refer to a DerivedField defined in LocalTransformations of that model in
addition to the other fields in its scope that are defined before it."

The binder is two-phase per container (DataDictionary, TransformationDictionary,
each LocalTransformations, each Output): phase 1 declares every name with its
declared type (`State::Declared`); phase 2 binds bodies, so a reference to a
later declaration has its type (L3). Cycles are detected by `State::Typing`
re-entry and are load errors ("DerivedFields in the TransformationDictionary
that have circular references ... are not valid PMML"). Forward references
other than the allowed one are accepted with warning `W-FORWARD-REF` (Q4).
JPMML accepts forward DerivedField references but refuses forward OutputField
references; cPMML deliberately accepts both (d23902a, D12). Evaluation order
is the dependency order, not document order.

### 3.6 Error reporting

```cpp
struct Diagnostic {
  enum class Severity : std::uint8_t { Error, Warning } severity;
  std::string_view code;        // stable: "E-SCOPE-UNRESOLVED", "E-TYPE-NO-COMMON", "W-FORWARD-REF", ...
  std::string message;          // "field \"agee\" is not in scope of TreeModel \"t3\"; in scope: age, income, ..."
  std::uint32_t line;           // from pugixml's offset_debug()
  // optional, later: std::string path (XPath-like); std::string_view rule (spec citation)
};
```

Code, message and line are enough to act on (review O4); the message names
the element and the enclosing model. The binder collects every error before
failing and throws one `cpmml::ParsingException` listing the first 20.
Nothing is caught and turned into a silent default (B13: the
`TransformedValue` catch is deleted).

### 3.7 Report mode

Before the binder becomes authoritative it runs in report mode (P1): the
existing engine scores, the binder only produces diagnostics. A test
(`test/bind_report.cc`) binds every fixture under `test/data/model`, every
benchmark model and the JPMML-derived fixtures, and compares the diagnostics
with a checked-in `test/data/bind_baseline.txt`. This is how each strictness
decision (Q3, Q4, Q6, Q7) is made with evidence.

---

## 4. The function table and the Apply wrapper

### 4.1 One table

```cpp
namespace cpmml::fn {

enum class Param : std::uint8_t { Any, Comparable, Numeric, Integer, Double, String, Boolean };
enum class Result : std::uint8_t {
  Fixed,               // the declared type in `fixed`
  LeastRestrictive,    // §2.5 over the numeric arguments
  AtLeastDouble,       // LeastRestrictive, integer → double (avg, median)
  IfBranches,          // common type of THEN and ELSE
};
enum class MissingRule : std::uint8_t {
  Propagate,           // "any missing inputs ... will result in a missing value"
  SkipAggregate,       // "Missing values in the input to an aggregate function are simply ignored"
  Receive,             // isMissing, isNotMissing, isValid, isNotValid
  LazyIf,              // only the condition and the branch taken are inputs
};

struct Signature {
  std::string_view name;
  std::uint8_t min_args, max_args;          // max_args == kVariadic for variadic functions
  std::array<Param, 3> params; Param rest;  // per position, then for the rest; a declared dataType
                                            // here is also what an untyped Constant argument inherits
  Result result; types::TypeId fixed;
  MissingRule missing;
  KernelFactory instantiate;                // (arg types, result type) -> monomorphic kernel
};

inline constexpr Signature kBuiltins[] = { /* §4.2 */ };
}
```

The binder looks the name up (unknown names keep today's specific messages for
`log` and `ceiling`), checks arity and parameter categories, types untyped
Constant arguments from the parameter (S2), computes the result type, inserts
implicit casts on the arguments, and asks the factory for a kernel specialised
to the argument types: `+(integer, double)` becomes `ArithNode<Plus, Double>`
over a `CastNode<Integer, Double>` and a `FieldRefNode<Double>`. Kernels are
written once against the scalar types of §2.1 (`Value plus(Integer, Integer)`,
`Value plus(Double, Double)`, …); how a kernel receives its operands from the
storage is a separate adapter (§7, P4/P5), which is what lets the function
table land before the representation changes.

### 4.2 The rows

"LR" is the least-restrictive numeric type of the arguments. Status: ✓
implemented today, ✗ refused at load today and in this design until the phase
named.

| function | arity | params | result | missing | invalid when | status |
|---|---|---|---|---|---|---|
| `+ - *` | 2 | numeric | LR | propagate | integer overflow | ✓ |
| `/` | 2 | numeric | LR | propagate; `0/0` → missing | integer `x/0` with `x ≠ 0` (D20); integer `MIN/-1` | ✓ |
| `min max sum product` | 1+ | numeric | LR | skip; all missing → missing | integer overflow | ✓ |
| `avg median` | 1+ | numeric | AtLeastDouble | skip | — | ✓ |
| `log10 ln sqrt exp` | 1 | numeric | double | propagate | — (NaN/±INF as specified) | ✓ |
| `abs` | 1 | numeric | LR | propagate | `abs(INT64_MIN)` | ✓ |
| `pow` | 2 | numeric | LR | propagate | integer with negative exponent (D8); overflow | ✓ |
| `threshold` | 2 | numeric | LR | propagate | — | ✓ |
| `floor ceil round` | 1 | numeric | integer | propagate | ±INF, NaN, out of int64 range | ✓ |
| `modulo` | 2 | numeric | LR | propagate | ±INF operand; divisor 0 | ✓ |
| `isMissing isNotMissing` | 1 | any | boolean | receive | — | ✓ |
| `isValid isNotValid` | 1 | field reference | boolean | receive ("a missing value will cause false to be returned") | — | ✗ → P7 (needs the validity flag) |
| `equal notEqual` | 2 | comparable | boolean | propagate; unordered → missing | — | ✓ |
| `lessThan lessOrEqual greaterThan greaterOrEqual` | 2 | comparable, not string × number | boolean | propagate; unordered → missing | — | ✓ |
| `and or` | 2+ | boolean | boolean | propagate | — | ✗ → P4 |
| `not` | 1 | boolean | boolean | propagate | — | ✓ |
| `isIn isNotIn` | 2+ | comparable | boolean | propagate | — | ✓ |
| `if` | 2–3 | boolean, any, any | IfBranches | lazy-if; no ELSE and false → missing | — | ✓ |
| `uppercase lowercase trimBlanks` | 1 | string | string | propagate | — | ✓ |
| `stringLength` | 1 | string | integer | propagate | — | ✓ |
| `substring` | 3 | string, integer, integer | string | propagate | — | ✓ |
| `concat` | 2+ | any (cast to string, Q1) | string | propagate | — | ✓ |
| `replace` | 3 | string ×3 | string | propagate | bad pattern → load error when constant | ✓ |
| `matches` | 2 | string ×2 | boolean | propagate | as above | ✓ |
| `formatNumber` | 2 | numeric, string | string | propagate | pattern not a numeric printf descriptor | ✗ → P4 |
| `formatDatetime`, `dateDaysSinceYear`, `dateSecondsSinceYear`, `dateSecondsSinceMidnight` | 2/2/2/1 | temporal (, integer / string) | string / integer | propagate | — | ✗ → P10 |
| `expm1 ln1p rint hypot`, `sin … tanh`, `erf` | 1 (hypot 2) | numeric | double (declared `dataType="double"`) | propagate | — | ✓ |
| `normalCDF normalPDF` | 3 | double | double | propagate | `sigma ≤ 0` | ✓ |
| `stdNormalCDF stdNormalPDF` | 1 | double | double | propagate | — | ✓ |
| `normalIDF stdNormalIDF` | 3/1 | double | double | propagate | p ∉ (0,1) | ✗ (needs an oracle-pinned approximation) |

Spec quotes behind the rows: "The functions -, *, and / are defined in the
same way"; "If a positive number is divided by zero, INF will be returned. If
zero is divided by zero, the result is indeterminate and a missing value will
be returned. If a negative number is divided by zero, -INF will be returned";
"Missing values in the input to an aggregate function are simply ignored ...
If all inputs are missing, the result evaluates to a missing value"; "Functions
floor, ceil, and round return an integer"; "Since INF and -INF can only be
floating point numbers, they will cause floor, ceil, round, and modulo to
return an invalid value"; "If the ELSE part is absent and the boolean value is
false then a missing value is returned"; the pseudo-declarations
(`not`/`if`: `ParameterField name="x" dataType="boolean"`; `substring`:
`startPos`/`length` `dataType="integer"`; `stringLength` `dataType="integer"`;
the distribution functions `dataType="double"`).

Notes where the design departs or the spec is silent:

- Integer `x/0` (D20). The spec says a positive number divided by zero returns
  INF, and also that INF and -INF "can only be floating point numbers"; an
  integer division (integer by inheritance, Functions.html) therefore has no
  value to return. The design makes it invalid, following XQuery, where
  integer division by zero raises `err:FOAR0001` (XQuery Functions and
  Operators, "Division by zero") — the basis is FOAR0001, not `idiv`. JPMML agrees (review
  probe `d.pmml`: N/A under asMissing). `0/0` of integers is missing per the
  spec sentence for zero over zero (probe `e.pmml`: N/A).
- Integer `/` rounding: truncation toward zero. The spec gives the type but
  not the rounding; this follows JPMML and Java integer division (and
  coincides with XQuery's `op:numeric-integer-divide`); shipped in 59513a2.
- `if` result type: the common type of THEN and ELSE (B19), so it is static;
  JPMML and today: the type of the branch taken (D7).
- `if` laziness: "Implements IF-THEN-ELSE logic" reads as a conditional; JPMML
  and cPMML evaluate lazily. Only the condition and the branch taken are
  inputs (so mapMissingTo applies when either is missing). D3 covers the
  missing condition.
- avg/median of integers: double (JPMML basis; shipped).

### 4.3 The Apply wrapper

One class implements the spec's "Output table for Apply" (Functions.html) for
every function:

```cpp
types::Value ApplyNode::eval(EvalContext& ctx) const {
  // 1. inputs, per the function's MissingRule
  bool any_missing = false;
  ArgBuffer args;                                     // inline storage, arity known at bind
  if (sig.missing == MissingRule::LazyIf) {
    const types::Value c = children[0]->eval(ctx);
    if (is_missing(c)) any_missing = true;
    else {
      const auto* branch = std::get<types::Boolean>(c).v ? children[1].get() : (children.size() > 2 ? children[2].get() : nullptr);
      args.push(branch ? branch->eval(ctx) : types::Value{types::Missing{}});
      any_missing = branch && is_missing(args.back());
    }
  } else {
    for (const auto& child : children) { args.push(child->eval(ctx)); any_missing |= is_missing(args.back()); }
  }
  // 2. "if it is specified and any of the input values of the function are missing then the
  //     function is not applied at all and the mapMissingTo value is returned instead"
  if (any_missing && has_map_missing_to && sig.missing != MissingRule::Receive) return map_missing_to;
  // 3. apply (aggregates and isMissing receive missing inputs; others propagate)
  types::Value result = (any_missing && sig.missing == MissingRule::Propagate)
                            ? types::Value{types::Missing{}}
                            : kernel(args, ctx);
  // 4. "the invalidValueTreatment value asMissing replaces the invalid value with a missing value";
  //    "in the context of an Apply, the value asIs is equivalent to returnInvalid"
  if (is_invalid(result)) {
    if (treatment != InvalidTreatment::AsMissing) ctx.raise_invalid(*this, std::get<types::Invalid>(result));
    result = types::Missing{};
  }
  // 5. "when a defaultValue value is provided, the function is applied first and if it produces a
  //     missing value the defaultValue is returned instead"
  if (is_missing(result) && has_default) return default_value;
  return result;
}
```

`raise_invalid` throws `cpmml::InvalidValueException("Apply " + function + " in
DerivedField " + field + ": " + reason)`. No `try/catch` remains in Apply
(B22). `asValue` on an Apply stays a load error. Kernels never throw for data;
they return `Invalid` (B21). `ArgBuffer` replaces the per-call
`std::vector<Value>` of today's `Apply::eval` (Housing*_PCA models carry 151
Applies).

---

## 5. One pipeline

### 5.1 Stages

```cpp
class InternalModel {
 public:
  InternalModel(const XmlNode& model, bind::ModelBinding& binding, ModelContext& shared);
  // The whole per-record pipeline, identical for a top-level model and a segment.
  ModelResult evaluate(Frame& frame, EvalContext& ctx) const;
  // What each model implements: raw scoring from prepared inputs. The default (P8) adapts the
  // legacy score_raw/predict_raw so models migrate one per commit.
  virtual RawScore raw_score(const Frame& frame, EvalContext& ctx) const;
  virtual ~InternalModel() = default;
 protected:
  const bind::ModelBinding& binding;
  Targets targets;            // typed
  OutputProgram outputs;      // typed, dependency order
};

ModelResult InternalModel::evaluate(Frame& frame, EvalContext& ctx) const {
  apply_mining_schema(binding.mining, frame, ctx);       // §2.8; top level reads ctx.input, a child its parent's slots
  frame.reset_lazy(binding.scope);                       // this scope's lazy DerivedFields: evaluated on first read
  RawScore raw = raw_score(frame, ctx);                  // model specific
  ModelResult result = targets.apply(std::move(raw));    // Targets.html order: min/max, rescaleFactor,
                                                         // rescaleConstant, castInteger; priors/defaultValue
                                                         // when no prediction; class stays raw (display noted)
  result.predicted = cast_to(binding.target.type, result.predicted, ctx);  // integer target: §2.6, Q2, B24
  outputs.run(frame, result, ctx);                       // each OutputField writes its slot and the result
  return result;
}
```

Targets.html: "they must be applied in sequence, which is: min and max,
rescaleFactor, rescaleConstant, castInteger" and "the displayValue attributes
are not used for identifying a target category within the model". So
`ModelResult::predicted` is the raw class or the post-Target number;
`ModelResult::display` points at the TargetValue whose displayValue names it,
and only `predictedDisplayValue` OutputFields and the API read it (B27). The
target cast makes a regression with an `integer` target and a non-integral
prediction (no `castInteger`) an invalid record; JPMML throws
TypeCheckException "Expected integer value, got double value" there (review
probe `b.pmml`); today cPMML returns the double (Q2).

```cpp
struct Distribution {                       // classes in the target's type, document order (B28, B29)
  std::vector<std::pair<types::Value, double>> entries;
};
struct RawScore {
  types::Value predicted;                   // Missing when the model makes no prediction
  Distribution probabilities;
  std::optional<double> confidence;
  Entity entity;                            // node id / cluster / segment / rule, typed
  ModelExtras extras;                       // variant: ranked ids, reason codes, matched rules, forecasts
};
struct ModelResult {                        // this is InternalScore, keeping the name cPMML.h declares
  types::Value predicted;
  const TargetValue* display = nullptr;
  Distribution probabilities;
  OutputValues outputs;                     // (OutputField index, Value) in document order
  ModelExtras extras;
  std::shared_ptr<const ModelContext> owner;        // keeps the document's StringPool and TargetValues alive
  std::unique_ptr<types::RecordArena> arena;        // strings made while scoring; allocated on first use
  mutable std::once_flag maps_once;                 // lazily built num_outputs/str_outputs (§5.5)
  mutable LazyMaps maps;
};
```

**Result lifetime (review S1).** `cpmml::Result` holds a
`shared_ptr<InternalScore>` independently of the `Model`'s
`shared_ptr<InternalEvaluator>` (`include/cPMML.h:260,361`), so a Result may
outlive its Model. Pooled `String`s and `display` point into the document's
`ModelContext` (pool, TargetValues), so `ModelResult` holds a
`shared_ptr<const ModelContext>`; the evaluator holds the same `shared_ptr`.
The arena is a `std::unique_ptr<RecordArena>` because
`std::pmr::monotonic_buffer_resource` is neither copyable nor movable; it is
allocated only when a record makes a string (most numeric models never do).
`test/api_guards.cc` gets a case where the `Result` outlives the `Model` and
reads every accessor under ASan.

### 5.2 Segments, modelChain and nested models

A segment is a child scope; the MiningModel's `raw_score` runs
`segment.model->evaluate(frame, ctx)` on **the same frame**. The child's
MiningField steps read the parent's slots (aliases cost nothing), the child's
own DerivedFields and OutputFields have slots in its scope range, and the
child applies its own Targets — the same code path as a top-level model, so
"segments skipped Targets" (01c1ecc) and "AnomalyDetection skipped the outer
MiningSchema" (90847c1) cannot recur.

- **modelChain**: OutputField slots of a chain segment are allocated in the
  MiningModel's scope ("all Outputs from the segment models become part of the
  enclosing mining model scope"), so later segments and the MiningModel's own
  OutputFields read them (B15). A segment whose predicate is false leaves its
  output slots missing ("OutputFields from Segments whose Predicates evaluate
  to false are, by convention, to be treated by subsequent Segments as missing
  values"). `ChainScope`'s save/restore disappears.
- **selectFirst / selectAll / votes / averages**: aggregate typed
  `ModelResult`s (B28); ties break in document order of the typed
  `Distribution` (B29). Top-level OutputFields "apply to the winning Segment"
  and "In the event of conflict ... the highest level specification
  prevails", implemented once in `MultipleModelMethod`.
- **`@segmentId` OutputFields** (B34): an OutputField with `segmentId` reads
  the named feature from that segment's `ModelResult` if its predicate was
  true, else missing (Output.html: "If there is no Segment matching segmentId
  or if the predicate of the matching Segment evaluated to false, then by
  convention the result delivered by this OutputField is missing").
- **nested MiningModel**: recursion of the same.
- **AnomalyDetectionModel**: an `InternalModel` whose MiningSchema runs in the
  pipeline and whose inner model is a child scope (B26).

**MULTITHREADING (review S9).** Today's OpenMP loop
(`multiplemodelmethod.h:298`) evaluates segments in parallel. In the design:

- segments write disjoint slots (their own scope ranges);
- per-slot lazy state and validity are **one byte per slot**, not bits, so two
  threads never read-modify-write the same word; each scope's slots are a
  contiguous range, and the layout pads scope ranges that run in parallel to a
  cache-line boundary to avoid false sharing;
- shared lazy instances placed in the parent scope (§3.2) are computed by the
  parent before the parallel region: the binder knows statically which parent
  lazy slots a parallel child can read, and the parent evaluates exactly those
  first. A failure is recorded in the slot's state byte and raised only when a
  child reads it, so the result is the same as the sequential lazy order;
- each thread gets its own `RecordArena`; the frame is never copied;
- gate: `test/concurrency.cc` under `-DSANITIZE=thread` with OpenMP (clang +
  libomp, which carries TSan annotations; libgomp is not TSan-instrumented),
  as a CI job, from P7 on.

The OpenMP path is benchmarked as part of the gate (§6.4). Before Q12 the
OpenMP build was faster only for AuditRandomForest (0.57×) and slower for
every other ensemble (up to 3.06×), because the parallel branch allocated per
segment even when the loop stayed sequential. Q12 (da1098e, §9) fixed it: an
ensemble not worth a parallel loop runs the plain loop, and the decision is
taken at load from estimated work. OpenMP head vs sequential head: geomean
0.977, 0 of 23 models fail the gate, AuditRandomForest 0.521. The OpenMP
build therefore joins the base-vs-head gate (§6.4 item 6).

### 5.3 Lazy DerivedFields

FieldScope.html: "DerivedFields from the TransformationDictionary which are
not referenced (directly or transitively) in a model are not instantiated in
that model element." Today every TD field whose inputs exist is evaluated per
record (`derivedfields_dag`). The design evaluates a DerivedField slot on its
first read in a record (one byte test per read; the scope's state bytes are
reset per evaluation). With the sharing rule of §3.2 a TD field read by N
segments is computed at most once per record. A DerivedField that fails
records the failure and raises it when read, so a record fails exactly when a
value it needs fails, independently of order, sharing or threads (§5.2). This
lands last in P7 (P7d), after scopes and the sharing rule, and only if the
TSan job is green (Q11).

### 5.4 Models plug in

Dispatch stays virtual (`raw_score`). A `ModelEvaluator` concept
(`requires { m.raw_score(f, c) } -> RawScore` plus the constructor shape)
could document the contract and type `ModelBuilder`'s factory table; it
duplicates the pure virtual, so it is optional (review O3).

| model | reads from the frame | change |
|---|---|---|
| TreeModel | bound predicates | predicates compiled per type; `raw_score` returns typed class/number and node id; Node@score parsed in the target type at load |
| RegressionModel | numeric `as_double()`; categorical predictors compare typed | CategoricalPredictor values read by the §3.4 rule |
| GeneralRegressionModel | covariates `as_double()`, factors compared typed | typed levels (315a815) become `Value`s |
| NeuralNetwork | NeuralInput DerivedFields `as_double()` | NeuralOutput classes typed |
| SupportVectorMachineModel | VectorFields `as_double()` | its own `validate` deleted (the pipeline validates, B26) |
| NearestNeighborModel, ClusteringModel | `as_double()`; binary measures cast to Boolean | B3 |
| NaiveBayesModel | counts keyed by `KeyHash`/`KeyEqual` | replaces the map keyed by string codes |
| Scorecard, RuleSetModel | bound predicates | — |
| BaselineModel, GaussianProcessModel | `as_double()` | — |
| AssociationModel | item text | items interned in the document pool; transactional inputs keep their own input path |
| TextModel, TextIndex | `String::text()` | the reverse lookup of string codes goes |
| TimeSeriesModel | no field inputs (`horizon`) | result typing only |
| AnomalyDetectionModel | child scope | becomes an `InternalModel` (§5.2) |
| MiningModel | segments | `MultipleModelMethod` over `ModelResult` |

### 5.5 Public API and compatibility

`include/cPMML.h` keeps its declarations; `Result` holds an opaque
`std::shared_ptr<InternalScore>`, which becomes `ModelResult` under the same
name, so the ABI of `Result` is unchanged.

| API | today | design | compatibility |
|---|---|---|---|
| `Model::evaluate(const Input&)` | flattens `Input` into a new `unordered_map<string,string>` per record | MiningField steps look names up in the caller's map and read `std::string_view`s; `int` inputs read as integers | same signature |
| `Result::as_string()` | displayValue when the Target has one, else Java text of the value | Java text of `predicted`, or the display per Q5 | unchanged until Q5 |
| `Result::as_double()` | parses the text | `as_double()` of a numeric prediction; boolean 1/0; else `double_min()` | same contract |
| `distribution()` | keys renamed after the fact | keys are `java_text` of the typed class | same values (B28) |
| `num_outputs()`, `str_outputs()` | maps filled during evaluate | built on first call under `std::call_once` (Result copies share one `InternalScore`, so the const accessors must be race-free; review N8) | same contents |
| `validate()` | re-runs preparation | runs the MiningSchema step only | same contract |
| new `Model::diagnostics()` | — | load warnings as `{code, message, line}` entries (Q9) | additive, ABI-safe |
| `Value::active_indexer`, `ScopedIndexer` | thread_local state | gone | internal |

---

## 6. Performance plan

### 6.1 Baseline (measured for this document)

Apple M4, macOS, Apple clang 21, Release (`-Wall -O3`, `CPMML_SHARED=ON`,
thin LTO), OpenMP not found (single-threaded), `master` at `b4efd7d`. Five
sequential runs of `model_benchmark.exe` per model; the tool times
`Model::evaluate` only. Raw output: `design/bench_baseline.txt`,
`design/bench_housing.txt`. (§6.4 has the interleaved A/A numbers, which are
the ones the gate uses.)

| benchmark (dataset) | eval median µs | eval min µs | load median ms |
|---|---|---|---|
| AuditBinaryReg (Audit) | 1.27 | 1.17 | 0.15 |
| AuditRandomForest (Audit) | 75.39 | 73.54 | 83.15 |
| AuditTree (Audit) | 1.02 | 0.98 | 0.23 |
| ensemble_audit_dectree (Audit) | 1.52 | 1.45 | 0.48 |
| ensemble_iris_dectree (Iris) | 1.04 | 1.01 | 0.17 |
| IrisLinearReg (Iris) | 0.54 | 0.49 | 0.07 |
| IrisMultinomReg (Iris) | 1.06 | 1.05 | 0.08 |
| IrisRandomForest (Iris) | 8.33 | 8.02 | 2.54 |
| IrisTree (Iris) | 0.73 | 0.72 | 0.10 |
| single_audit_dectree (Audit) | 0.95 | 0.92 | 0.18 |
| single_iris_dectree (Iris) | 0.52 | 0.48 | 0.09 |
| HousingGBTlassifier | 13.53 | 13.02 | 1.89 |
| HousingRFClassifier | 3.76 | 3.72 | 30.68 |
| HousingLRClassifier | 1.55 | 1.52 | 0.16 |
| HousingGBTlassifier_PCA | 17.13 | 16.92 | 1.99 |
| HousingRFClassifier_PCA | 7.96 | 7.55 | 33.43 |
| HousingLRClassifier_PCA | 5.22 | 5.06 | 0.44 |
| HousingGBTRegressor | 12.11 | 11.76 | 1.82 |
| HousingRFRegressor | 4.64 | 4.49 | 90.17 |
| HousingLinearRegressor | 1.35 | 1.31 | 0.16 |
| HousingGBTRegressor_PCA | 15.95 | 15.57 | 1.93 |
| HousingRFRegressor_PCA | 9.10 | 8.93 | 87.26 |
| HousingLinearRegressor_PCA | 5.17 | 4.84 | 0.42 |

### 6.2 Memory layout

| | today | design |
|---|---|---|
| `Value` | 40 B: `double`, `bool`, `std::string` | 16 B: 8 B payload + 1 B index, trivially copyable |
| per-field slot | `Feature` 40 B (+ name in DEBUG) | `Value` 16 B + 1 B state; names in `FrameLayout` |
| per-record frame | copy of `base_sample` (element-wise, `std::string` copy constructors) | one allocation (or a thread-local reused buffer) of `n_slots × 17 B`, filled with `Missing` |
| per-record input | `flatten_input` builds a `unordered_map<string,string>` copy | none: `string_view` into the caller's `Input` |
| strings | `svalue` copies; Indexer hash per input string | one pool lookup per string input; arena copy only for text the pool lacks |
| segment scope | `Sample` copy per segment with LocalTransformations, and in `OutputDictionary::add_output` when outputs read outputs | no copy: disjoint slot ranges in one frame |
| result | `InternalScore` 360 B with five maps, built eagerly | `ModelResult` with vectors; maps built on demand |

### 6.3 Expected cost per phase

These are expectations to be checked by the gate, not evidence.

| phase | costs | savings |
|---|---|---|
| P1, P2 | load only: one more walk of the document (AuditRandomForest walks 68k predicates) | — |
| P3 | none (accessors inline to today's fields) | — |
| P4 | an unpack/pack per Apply argument over the old storage | no `std::function` per Apply, no `std::vector<Value>` per Apply call, no `try/catch` |
| P5 | `std::get_if` index check per compare | 16 B trivially copyable slots; no Indexer per input string |
| P6 | per fix, small | — |
| P7 | lazy byte test per derived read | no `flatten_input`; no segment Sample copies; TD fields computed once per record and only when read |
| P8 | — | no text predictions between segments, no `label_text`/`present()`, lazy result maps |

### 6.4 Rules and gates

**Hot-path rules.**

- No `std::visit` per record. Generic entry points live in
  `cpmml::types::cold` and take a `ColdPath` argument whose constructor is
  private to load code (`bind/`), the API layer (`api/`) and tests (friend
  declarations); model and expression code cannot obtain one, so a generic
  visit from a scoring path does not compile (review S14).
- Bound predicates: a tree stores its predicates as a flat array of
  `struct BoundPredicate { std::uint8_t op; types::TypeId type; std::uint8_t kind; SlotId slot; std::uint32_t operand; }`
  (operand indexes a typed constant pool or a set table) and evaluates them
  with one `switch (op × type)`; no `std::function`, no bound `Value` copy
  (review N9).
- A bound typed access asserts its type in debug builds
  (`assert(std::holds_alternative<Missing>(v))` where a `Double` slot holds
  something else, sketch `DoubleLessOrEqual`), so a binder/runtime
  disagreement stops instead of taking the UNKNOWN branch (review S13). CI
  adds a Debug (assertions on) ctest job from P5.

**Noise, measured.** An interleaved A/A study (`design/aab.sh`, the same
sequential binary run as "A" and "B", 15 rounds × 23 models, order rotated
each round; `design/aab_stats.txt`) shows:

- per-model spread (max − min over median) 6–78%, typically 10–20%;
- the B/A ratio of medians ranges 0.952–1.035 with no real change;
- the revision-1 gate ("median delta > +3%") fired on 1 of 23 models
  (single_iris_dectree, 1.035) in an A/A pair, i.e. it would fail an unchanged
  commit; per-model 95% bootstrap intervals of the ratio are ±2–9% wide
  (AuditRandomForest [0.905, 1.089]);
- the geometric mean of the 23 ratios was 0.994.

**The gate (review S11).** `tools/bench-ab.sh BASE HEAD`:

1. Interleaved rounds, base and head alternating with the order rotated each
   round; 15 rounds per model, 31 for the NOISY models. One rule decides
   NOISY for every model: the five whose A/A interval is wider than ±5%
   (AuditRandomForest, IrisLinearReg, IrisMultinomReg, single_iris_dectree,
   HousingRFRegressor_PCA), plus every model wider than ±5% in at least 2 of
   the 4 build × mode groups of the P0 A/A run (ensemble_iris_dectree,
   IrisTree, HousingLRClassifier_PCA, HousingLinearRegressor_PCA,
   fix_nn_missing_classification, fix_svm_ovo_tie, fix_nb_threshold,
   IrisRuleSet, IrisText). The list lives in `benchmark/CMakeLists.txt`.
   One discarded warm-up pass precedes round 1.
2. Per model: ratio r = median(head)/median(base) and its 95% bootstrap
   interval (4 000 resamples of each side). **Fail** when the interval's lower
   bound exceeds 1.03, i.e. the slowdown is above 3% with 97.5% confidence.
   A failing model is re-run once with fresh rounds; it fails the gate only if
   it fails again. In the A/A study this rule fired on 0 of 23.
3. Across models: the geometric mean of r and its bootstrap interval
   (resampling rounds jointly, 4 000 resamples). Models have 15 or 31
   rounds, so each model draws k rounds per resample, uniform over its own
   k, with base and head sharing the draw: with c the fewest rounds of any
   model, a draw takes a shared round among the first c with probability
   c/k and otherwise one of the remaining k − c. Every round has probability
   1/k, and draws in the first c rounds stay aligned across models, where
   the sweeps coincide. **Fail** when the geomean exceeds 1.01 and its
   lower bound exceeds 1.00, i.e. a broad 1% slowdown that is statistically
   real. The A/A geomean, 0.994, sits within that band's noise.
4. Load: same method, per-model threshold 1.10.
   **Waiver W1 (decided 2026-11-18, expires at P7a.2).** From P2.1 the
   binder runs at every load in front of today's Indexer, which it does not
   yet replace. On models that load in under about 0.3 ms that adds a mostly
   fixed cost (measured in P2.1: 33–50 µs for the first bind in a process,
   8 µs warm, 30–40 ns per resolved reference), which fails item 4 on its
   own: IrisMultinomReg 1.28–1.35, the Housing linear models and IrisTree
   1.21–1.29, ensemble_iris_dectree 1.15–1.20. Until P7a.2 deletes the
   Indexer, P2.1 may fail item 4 only on models whose load is
   below 1 ms, and only by the binder's fixed cost; its message gives the
   ratios. The waiver covers load time only: items 2, 3 and 5–7 apply
   unchanged, and every commit after P2.1 still passes item 4 against its
   own predecessor (the waiver covers the binder's arrival, not later
   growth). **At P7a.2 the waiver ends and the load gate compares P7a.2
   against 425b709, the last commit before the binder ran at load; it must
   pass item 4 there with no waiver.**
5. Both benchmark modes: `evaluate` only, and `--read-all` (also calls
   `as_string`, `distribution`, `num_outputs`, `str_outputs`), so that P8's
   lazy result maps are not counted as a win they are not.
6. Both builds: sequential, and OpenMP (`-DOpenMP_ROOT=$(brew --prefix libomp)`
   locally, libgomp on the Linux CI runner), compared base-vs-head within each
   build.
7. Coverage: the 23 Regression/Tree/MiningModel benchmarks plus one per model
   family from the fixtures, scored repeatedly (`model_benchmark --zip
   --repeat N`, as the fixtures have 6–118 rows): `fix_nn_missing_classification`
   (NeuralNetwork), `fix_svm_ovo_tie` (SVM), `fix_minkowski_knn_average_p05`
   (KNN), `fix_clustering_equal_boolean_center` (Clustering), `fix_nb_threshold`
   (NaiveBayes), `GRM_OrdinalCauchit` (GeneralRegression), `IrisScorecard`,
   `IrisRuleSet`, `IrisText`, `fix_gp_missing_continue` (GaussianProcess),
   `IrisBaseline`, `fix_anomaly_miningschema_categorical`,
   `fix_association_output_defaults`. These are the models that call
   `as_double()` on every input. Their A/A noise is measured in P0 before
   they gate anything.

A failure is not waived by a sentence in the commit message; the commit is
reworked or split until the gate passes. The statistical method is what makes
that rule affordable.

Emscripten: the three Iris benchmarks under node, informational only.

---

## 7. Migration plan

Ground rules for every commit:

- **Green.** All tests pass (341 today plus the fixtures the commit adds).
- **Behaviour-preserving or behaviour-fixing, never both.** P0 adds
  `tools/fixture-dump-diff.sh`: it runs `pmml_tester --dump` over every
  fixture (and, once P0.4 adds it, the fuzz corpus, §8) with the base build
  and the head build and diffs the text of every cell. By default this
  checkout's tester is built over each revision, as `bench-ab.sh` does. The
  dump feeds every column of a fixture row, target included, exactly as ctest
  does, so residual OutputFields are covered; an empty or N/A cell is left
  out of the input. A refactoring commit shows **no** diff. A fix
  commit shows diffs only in the cells its message lists, each tied to a bug
  id and a fixture that fails on base with a quoted sentinel.
- **Small.** One concern per commit; the files touched are listed below.
- **Gated.** §6.4 on every commit that touches a scoring path.

Phase order: P0 scaffolding → P1 type library and binder report → P2 binder
decides at load → P3 accessor API → P4 function table and Apply over the
accessors → P5 representation swap → P6 semantic fixes on the new
representation → P7 frames and scopes → P8 one pipeline, one model per commit
→ P9 public API → P10 temporal types → P11 fuzzer as a gate. The function
table (P4) comes **before** the representation swap (P5), so P5 changes
storage only; every behaviour fix that the new representation enables (P6)
comes after it, one per commit.

### P0 — scaffolding (no behaviour change)

| commit | files | why green alone |
|---|---|---|
| P0.1 bench tooling | `tools/bench-ab.sh`, `tools/bench-compare.py`, `benchmark/model_benchmark.cc` (`--read-all`, `--zip`, `--repeat`), `benchmark/CMakeLists.txt` (`benchmark_all` with Housing and per-family models) | tools only |
| P0.2 dump diff | `tools/fixture-dump-diff.sh`, `tools/fixture-dump-cells.py`, `test/pmml_tester.cc` (`--dump` for time series) | tool only (done: d3bb746, 391e3db) |
| P0.3 tester checks the target column | `test/pmml_tester.cc` (B33). The target column holds the predicted value (JPMML's output) and is checked; for a model with a residual OutputField it holds the actual value, an input (Output.html: "the input data must include target values"), and is not checked | the target columns already agree (verified with P0.2 before committing; done: 4298b39) |
| P0.4 fuzzer v0 | `tools/fuzz/gen.py`, `tools/fuzz/JpmmlOracle.java`, `tools/fuzz/compare.py`, `tools/pmmltext.py` (shared with `tools/cross-validate-text.py`), `test/fuzz_runner.cc`, `tools/fuzz/departures.json`, `tools/fuzz/run.py`, `test/CMakeLists.txt` (`fuzz_runner.exe`, `EXCLUDE_FROM_ALL`) | not in ctest (done: fcc1ddf, 4467498) |
| P0.5 staged fixtures | `test/data/pending/` (20 fixtures, `MANIFEST.md` naming the commit that registers each) | not registered (done: 444fc92, 0d513c9) |

### P1 — type library and binder in report mode (nothing in the engine uses them)

| commit | files | why green alone |
|---|---|---|
| P1.1 types | new `src/core/types/{value.h,string_pool.h,parse.h,cast.h,compare.h,arithmetic.h,java_text.h,keys.h,cold.h}`; `test/typed_value.cc`; `test/types_negative/` (`try_compile` expected to fail); `CMakeLists.txt`, `test/CMakeLists.txt` | new code, unit-tested, unreferenced |
| P1.2 binder, report mode | new `src/bind/{scope.h,binder.h,typing.h,function_table.h,diagnostics.h}`; `test/bind_report.cc`; `test/data/bind_baseline.txt` | runs only in its test |

As built (P1.1): header-only under `cpmml::types`, reusing `utils/lexical.h` (grammar) and `utils/java_format.h` (Java text). `test/typed_value.cc` (221 checks after the follow-up below, each citing its rule) and three must-not-compile cases in `test/types_negative/` (a visitor without `String`, an uncategorised alternative, a `ColdPath` made outside its issuers), each run by ctest through `check_negative.cmake`: its control (`-DCPMML_NEGATIVE_CONTROL`) must compile and the case must fail with the intended diagnostic. Only `test/CMakeLists.txt` changed; the top-level `CMakeLists.txt` needed nothing. Choices the sections above left open: `Comparison` takes a `Purpose` (equality or ordering), so string × number gives `nullopt` for ordering and text for equality from one visitor; `cold::relate` and the arithmetic visitors throw `std::logic_error` on operands the binder refuses, instead of answering; a NaN probe of a hashed container is "indeterminate" when it has any key (what the linear path gives); `ColdPath` issuers are `cpmml::bind::ColdPathIssuer`, `cpmml::api::ColdPathIssuer`, `cpmml::types::testing::ColdPathIssuer`; `KeyHash`/`KeyEqual` are not behind `ColdPath` (a lookup is per record) and have typed overloads; float text is parsed straight to float (XSD §3.2.4, no double rounding); `+ - * /` follow Transformations.html's infinite-number rules (`INF + -INF`, `INF - INF`, `INF × 0`, `INF / INF` of one sign → missing; `INF / -INF` → the numerator), proposed as D27 (JPMML 1.7.7 gives an invalid value for all four, `Function "/" returned invalid value`, N/A under asMissing; IEEE gives NaN). Follow-up to P1.1 (working tree): a `-0.0` divisor gives INF signed by the numerator (`1 / -0.0` → INF, as the division sentence says and JPMML does; IEEE gives -INF); a missing operand of the arithmetic visitors gives missing even beside an invalid one, the order of the §4.3 Apply wrapper; `keys.h` states that the binder casts the probe to the key type before a lookup.

As built (P1.2): header-only under `cpmml::bind` (`cpmml::fn` for the table), included by `test/bind_report.cc` only; the test links pugixml and miniz, not the library, and only `test/CMakeLists.txt` changed. One `Binder` per document parses it with pugixml and produces the scope tree, a static type per declaration and expression, a `FrameLayout` (pass-through MiningFields are aliases) and `Diagnostics`; it never throws for a defect of the document. `test/bind_report.cc` has 100 rule checks (each a small document and the exact multiset of codes the cited rule gives) and the baseline: 377 documents (test/data/model, benchmark/data/model, test/data/pending/model), identical diagnostics within a document folded into one line with a count. Choices the sections above left open: a third severity, **Unsupported** (`U-ELEMENT`, `U-TEMPORAL`, `U-FUNCTION`, `U-USER-FUNCTION`, `U-SEGMENT-ID`), for what the binder does not type yet, so it is neither an error of the document nor silently skipped; in a model body every element is bound by a handler, or listed as inert (no EXPRESSION/PREDICATE content and no attribute holding a field value, per pmml-4-4.xsd; its FIELD-NAME attributes are still resolved), or reported once per model and element name with its count. Codes beyond those named above: `E-XML`, `E-SCOPE-TOP-LEVEL`, `E-NAME-DUPLICATE`, `E-CYCLE`, `E-ENUM`, `E-REQUIRED`, `E-FORBIDDEN`, `E-ATTRIBUTE-MALFORMED`, `E-VALUE-MALFORMED`, `E-TYPE-BOOLEAN-MIXED` (Q7), `E-TYPE-PARAM`, `E-ARITY`, `E-FUNCTION-UNKNOWN`, `E-KEY-DUPLICATE` (B42), `E-KEY-NAN` (D26), `E-NORM-ORDER`, `W-TEXT-WIDENED` (§3.4 rule 2 applied, D22: reported so P2 can see every occurrence; P2 may drop it), `W-CAST-NARROWING` (a declared type whose cast can give Invalid: to integer from float/double, string to number or boolean, number to boolean; Q2, B17, B24; it also fires for a boolean, integer or double OutputField over a string predictedValue or predictedDisplayValue, e.g. fix_output_boolean_class_text, fix_output_predicted_rescale, which is wider than B17/B24), `W-CAST-LOSSY` (double to float), `W-OUTPUT-NO-DATATYPE`, `W-NAME-SHADOWS-DATAFIELD` (a LocalTransformations DerivedField or OutputField named like a DataField that is not in the model's MiningSchema: FieldScope.html makes DataField names unique only "from the names of any other DataFields and the names of DerivedFields in the TransformationDictionary", the LocalTransformations-vs-DataFields sentence is quoted as PMML 3.0 history, so only "In general, field names in PMML should be unique" applies; no baseline document has one). The function table carries typing columns only (no kernel factory until P4), adds `Param::Temporal` (date functions, `U-FUNCTION` until P10) and `Param::FieldRef` (isValid/isNotValid take a FieldRef), and an `Engine` status column. A sub-model MiningField that names a TD field of its parent is read through the MiningSchema, not re-instantiated (the TD step of §3.2 yields to M's MiningFields for that name). Output precedes LocalTransformations in every model's content model, so an OutputField other than transformedValue reading LocalTransformations is `W-FORWARD-REF`. A NaN member of any SimpleSetPredicate, not only one above 150 members, is `E-KEY-NAN` (the linear path must give the hashed path's answers). A MapValues row without a key cell is `E-REQUIRED`. Item@category (not Item@value, which may be a field name, AssociationRules.html) is a value of Item@field; a GeneralRegression PPCell@value is a level for a factor and a NUMBER for a covariate. An untyped Constant's text is not trimmed: Constant extends xs:string (pmml-4-4.xsd, whiteSpace "preserve") and "The presence of any non-numeric characters ... will result in the Constant being treated as a string", so `<Constant> 1</Constant>` is the string " 1" (JPMML 1.7.7 agrees). A typed Constant is "empty" only when its xs:string content is: blanks-only text is read as a value of the type, whose whiteSpace facet collapses it to "", outside the numeric and boolean lexical spaces (`E-VALUE-MALFORMED`; JPMML 1.7.7 refuses it too), and a typed " 1 " reads as 1. pugixml's default drops whitespace-only PCDATA, so the binder parses with `parse_ws_pcdata_single` to see that content; the engine's reader keeps the default (P2 aligns it). In a modelChain a segment's OutputField is checked against the MiningModel's own OutputFields too (FieldScope.html: segment Outputs "become part of the enclosing mining model scope"). In the baseline, identical diagnostics folded across sibling models name only the first model, with "(+N more)" (e.g. fix_ensemble_localtrans_shadow). Baseline at this commit: 199 lines; E-NAME-DUPLICATE 9, E-SCOPE-UNRESOLVED 2, E-TYPE-BOOLEAN-MIXED 3, E-KEY-NAN 1, E-REQUIRED 1, W-CAST-LOSSY 113, W-CAST-NARROWING 15, W-TEXT-WIDENED 26, W-SCOPE-IMPLICIT 8, W-OUTPUT-NO-DATATYPE 4, U-ELEMENT 25; no E-TYPE-ORDER-MIXED, W-FORWARD-REF, W-ORDER-CATEGORICAL, W-ORDINAL-STRING-NO-VALUES, W-SLOT-WIDENED or E-KEY-DUPLICATE in any document.

### P2 — the binder decides types and names at load (old `Value` storage)

Each commit changes load-time decisions only; storage and scoring code are
untouched, and the dump diff names the cells that change.

| commit | files | closes | why green alone |
|---|---|---|---|
| P2.1 resolution and two-phase declarations | `core/predicatebuilder.h`, `core/derivedfield.h`, `core/transformationdictionary.h`, `core/internal_model.h` (construction only), `output/outputfield.h`, every model constructor's field lookup (`indexer->get_index/get_type` → `scope.resolve`, which still answers with today's Indexer slots) | B12 (fixture `vp/p2`) | same slots and types as today; the only new behaviour is that a forward reference no longer throws |
| P2.2 XSD attribute types and the §3.4 document-text rule | `expression/{normcontinuous,normdiscrete,discretize,mapvalues,fieldref,apply}.h`, `core/miningfield.h`, `core/datafield.h`, `regressionmodel/{numericpredictor,categoricalpredictor}.h`, `generalregressionmodel/generalregressionmodel.h` | B7, B10, B11 (fixtures `q3`, `q4`, `nd_i`, `tree_int`, replacement widening) | values still stored as doubles; only the text→number reading changes, and the K1 scan shows no existing document affected |
| P2.3 Constant typing | `expression/constant.h`, `bind/typing.h` | B8, B9, S2, S3 (fixtures `const_up`, spec `concat(month,"-",year)` row, `if(<Constant>true</Constant>…)`, `<Constant dataType="double">inf</Constant>`) | one element's typing |
| P2.4 node types replace the XML re-walk | `core/builtinfunctions.h` (delete `expression_kind`, `field_kind`, `declared_in`, `declared_kind`, `result_kind`, `common_kind`), `output/{transformedvalue,outputexpression,outputfield}.h`, `core/datatype.h` (`integer` gets its own `DataType` value; delete `is_integer`, `DataField::integer`, `OutputField::integer`, `java_kind(DataType,bool)`) | — | dump diff empty: the binder's types equal the re-walk's on every fixture (checked in P1's baseline) |
| P2.5 no silent fallbacks | `output/transformedvalue.h` (delete the catch), `expression/apply.h` (skip `Extension`) | B13, B25 | load errors only where the document was already unusable |
| P2.6 static `if` type | `bind/typing.h`, `core/builtinfunctions.h` | B19 | one function's result type |
| P2.7 string × number ordering is a load error | `bind/typing.h` | §2.7 (S5) | the P1 baseline shows which fixtures it would reject; if any, they are listed and re-derived in this commit |

As built (P2.0, not in the table; agreed with the user: CODE_REVIEW.md R1 finished before P2.1): `XmlNode` has no one-argument `get_attribute` (the text `"null"` for an absent attribute), `get_double_attribute` (`DBL_MIN`) or `get_long_attribute` (`LONG_MAX`); a reader says what absence means with `required_*` (pmml-4-4.xsd `use="required"`), `get_*(name, default)` (the XSD default, or `""` where the reader already keeps absence apart) or `optional_*` (`std::optional`, no default and absence means something). Every `== "null"` test (SVM classes and `alternateBinaryTargetCategory`, GRM `targetReferenceCategory`, Scorecard reason codes, `get_childs_byattribute`) and every attribute-derived `DBL_MIN` member or test (Target min/max/priorProbability/defaultValue, ScoreDistribution confidence/probability and the `is_double_min(recordCount)` tests of `TreeScore`, DataDictionary numberOfFields) now uses real presence. Node@recordCount and MiningField@importance, which no scoring rule reads, are checked at load and no longer stored: a first build kept them as `std::optional<double>`, and the larger `Node` failed the §6.4 gate (AuditRandomForest eval 1.10-1.14 seq, 1.06-1.07 omp, confirmed by re-run; seq geomean 1.012). Dropping the member left `Node` at 504 bytes, and the small trees (IrisTree, single_iris_dectree) measured 4-6% slower in seq eval than the base's 512; 425b709 therefore made `Node` `alignas(64)`, and its gate gave one literal FAIL (IrisScorecard seq eval 1.052 [1.036,1.094], re-run 1.057) that was committed as noise. Re-checked by the follow-up commit (full §6.4 gate, 36 models, both modes, seq and omp): (A) 3257e4d vs 425b709 passes with no model re-run (geomean eval/read-all seq 0.992/0.991, omp 1.000/0.997; IrisScorecard 1.014 [0.978,1.057] seq eval; no Scorecard or Target slowdown: IrisScorecard has no Target, and the Scorecard change only drops a string compare); (B) 3257e4d vs 425b709 without the `alignas` passes too, also with no re-run (seq 0.997/0.999, omp 1.004/0.998; IrisTree 0.989 and single_iris_dectree 1.015 seq eval), so the 4-6% small-tree slowdown did not reproduce. The follow-up removes the `alignas(64)` (a layout tweak with no semantic purpose) and passes its own gate against 425b709 (omp 1.002/1.004; seq 1.006/1.011, the read-all geomean a literal FAIL whose re-run passed at 1.006/1.008): the removal costs about 0.5-1% across all families, tree-less ones (NeuralNetwork, KNN, NaiveBayes) included, so it is code placement rather than `Node` size, and it leaves P2.0 level with 3257e4d. `Node` is 504 bytes again. Fixtures: `fix_svm_null_category` (base: "SupportVectorMachine has no targetCategory") `fix_scorecard_null_reasoncode` (base dropped the `null` codes) and `fix_nn_null_derivedfield` (unnamed NeuralInput DerivedFields were indexed under the text `null` and shadowed the DataField `null`; base 22.0 where JPMML gives 12.0). No fixture for GRM `targetReferenceCategory="null"`: JPMML scores it only when the target lists its values, and then base already agrees; base gave p=1.0 for every row only when the DataDictionary lists none. Newly required at load where the XSD is optional but the reader cannot work without the value (no oracle or JPMML agrees; no baseline document lacks them): RegressionTable@targetCategory of a classification model (Regression.html "each one must have the attribute targetCategory"; JPMML: "Required attribute RegressionTable@targetCategory is not defined"), DerivedField@name in TransformationDictionary/LocalTransformations (JPMML: "Required attribute DerivedField@name is not defined"; unnamed stays allowed in NeuralInput/NeuralOutput), Level@smoothedValue and Trend_ExpoSmooth@smoothedValue ("otherwise, the smoothed value is required"; JPMML does not support ExponentialSmoothing). TimeValue of the original series that ARIMA and GARCH read: "In addition, time or index must be used." (TimeSeriesModel.html; index is required only "In case of a logical TimeSeries"), so a point with neither is a load error; the history is ordered by @index when every point has one, else by @time when every point has one (base set an absent index to `(int)LONG_MAX`, an arbitrary order), and a series mixing index-only and time-only points is refused as a cPMML limitation (no single order). Seasonality_ExpoSmooth TimeValue@index stays required: it is the season number the value belongs to. Segment@id absent is its 1-based position (MultipleModels.html). Left for P5/R2: the score-time "no number" sentinels (`InternalScore::double_score`/`confidence`, `OutputExpression::eval_double`, `Value::value`'s default, the public `Result::as_double()`, §5.5), which are not attribute readers.

As built (P2.1): `core/modelbuilder.h` binds every document it loads before building the model (`bind/load_scope.h`, new): `LoadScope` runs a `Binder` in a new `Mode::Load` over the engine's own pugixml document (no second parse; parsed with pugixml's default flags, so a blanks-only typed Constant reads as empty there until P2.3 aligns the reader), throws one `ParsingException` listing the first 20 errors of names and scopes (§3.6), and stays current on the loading thread (a thread-local, as `Value::active_indexer` is, so no constructor signature changed; P7 passes scopes explicitly) while the model objects are built. The lookups `indexer->get_index/get_type(name)` of predicates (`core/predicatebuilder.h`), MapValues FieldColumnPairs (`expression/mapvalues.h`: the B12 site of the fuzzer's forward-reference class), PredictorTerm FieldRefs (`SimpleFieldRef`, `expression/fieldref.h`), Numeric/CategoricalPredictor, KNNInput and its training cells, ClusteringField and its centers, GaussianProcess InstanceFields, GRM offsetVariable and PPCell, SVM VectorFields, BaselineModel TestDistributions and BayesInput are `cpmml::bind::scope_resolve(indexer, node, name)`: `Binder::lookup` answers the binding by §3.2 from the referencing element (the nearest model element above it is M; an `Output` on the way enables step 1; a `TransformationDictionary` or no model means the TD definitions), the slot is today's `Indexer::get_or_set(name)` and the type is the binding's static type as the engine's `DataType` (integer and the temporal types as DOUBLE). A forward reference therefore has its later declaration's type (B12, which threw `std::out_of_range` "unordered_map::at: key not found"), and a name that resolves nowhere is a load error naming the field and the model. The table's "same slots and types as today" holds for slots but not for every type: the type is the declaration in scope, where the flat Indexer kept the first-registered one, so a name declared twice with different types in different scopes now types each reference by its own scope's declaration. That is the typing part of B16 (sibling segments' LocalTransformations fields of one name; the shared slot stays until P7b.1) and a LocalTransformations field over a parent's DataField of the same name; predicate and MapValues constants are read in that type. Probes (scratch, JPMML 1.7.7, `sum` of two TreeModel segments): LocalTransformations `d` string in segment 1 and double in segment 2, base 1|0|1|0, head and JPMML 101|100|1|100; the segments in the other order, base 100|100|0|100, head and JPMML 101|100|1|100; a segment's LocalTransformations `d` string over the parent's DataField `d` double, base 100|100|0|0, head and JPMML 101|100|1|0. Every change moves to JPMML and the spec; the first is fixture `fix_scope_sibling_localtrans_type` (base: "predicted: 1.0  expected: 101.0"). Not converted: the expression constructors that ask no type (FieldRef, NormContinuous, NormDiscrete, TextIndex use `get_or_set(name)`; their names are resolved by the binder, which binds every TD, LocalTransformations and Output body at load), Discretize (P2.2's file; it forces DOUBLE on a forward-referenced field's Indexer type, as before), the AssociationModel Item probes (`contains` asks whether Item@field/@value is a field at all, AssociationRules.html), and the model's own target (`outputfield.h`, `outputexpressionbuilder.h`, `residual.h`, `internal_model.h`): none is a reference in the document. `core/derivedfield.h`, `core/transformationdictionary.h` and `output/outputfield.h` needed no change: declarations keep their Indexer slots, and the two-phase types come from the binder at each reference. Load mode is narrower than report mode, because the report binder costs 25-56% of the load of every benchmark model: it binds the scope tree, every declaration (two-phase), the Output bodies and the names of the TD and LocalTransformations bodies (`resolve_expression_names`: the same resolutions, reads, forward references and TD instantiations as `bind_expression`, without the typing whose diagnostics stay report-only; a DerivedField's type is its declared dataType; the load errors of every fixture, benchmark model and fuzz case, and of 24 probes of each expression kind, are the same either way), but not the model bodies (their names are resolved by the engine through `lookup`, which caches the context of every element it walks so a tree predicate stops at its parent Node), nor DataField Values/Intervals (document text, P2.2); a pass-through MiningField maps to its source (Load mode keeps no FrameLayout), scope labels and diagnostic locations are made only when a message needs them, and lines are counted only for errors; scope name maps are flat (`NameMap`: linear up to 16 names, hashed beyond). Load errors are E-SCOPE-UNRESOLVED, E-SCOPE-TOP-LEVEL, E-NAME-DUPLICATE and E-CYCLE. A departure from JPMML among them: outside a modelChain, a Segment predicate or a sub-model body that names an earlier sibling segment's OutputField is E-SCOPE-UNRESOLVED (FieldScope.html: "Outputs are generally local to a model, except when the model is used in a model chain as part of a MiningModel. In a model chain, all Outputs from the segment models become part of the enclosing mining model scope."); base read the flat slot and JPMML, which evaluates lazily, scores such a `selectFirst` document (probe: 2.0, 2.8, 3.0); no fixture or benchmark document does it (strict_parse.cc checks it). Within a modelChain, `lookup` sees the outputs of the segments before the reference's only, as Report mode does by binding the segments in order (FieldScope.html: "All other fields in PMML must be defined before they are referenced"; Q4/D12 accept forward references within one container, not to a later segment's Output): a Segment's predicate (the binder keeps the position of every modelChain Segment, `Scope::chain_position`) and an implicit pass-through in a segment's model body (§3.2 step 5, through each enclosing chain) no longer reach their own or a later segment's OutputField. A first build of this commit loaded both and read the field missing; base threw `std::out_of_range`, and Report mode and JPMML ('Field "q" is not defined') refuse them; strict_parse.cc checks the predicate, the body, the segment's own output and the earlier segment's (which loads). A modelChain OutputField that duplicates a name of the chain's scope is E-NAME-DUPLICATE and still joins the chain's outputs (the first one kept), so a later segment's MiningField naming it does not add a follow-on E-SCOPE-UNRESOLVED; every other binder error stays report-only until the commit that owns it (P2.2 document text and E-KEY-NAN/E-KEY-DUPLICATE, P2.3 Constants, P2.7 ordering, Q7). The clashes that documents which load today contain are warnings, in both modes: W-NAME-SHADOWS-MININGFIELD (an OutputField or LocalTransformations DerivedField named like a MiningField of its model, decided 2026-11-18: AuditTree, IrisTree, IrisMultinomReg), W-NAME-SHADOWS-TD (a LocalTransformations DerivedField named like a TD DerivedField: fix_derivedfield_shadow, fix_ensemble_localtrans_shadow; FieldScope.html forbids it, kept so that they load, a decision for the user) and W-SCOPE-FLAT (a sub-model MiningField naming another model's LocalTransformations field, which the flat Indexer reaches: fix_ensemble_localtrans_scope, against MiningSchema.html "must refer to a field in the scope of the parent of the MiningSchema's model element", until P7, a decision for the user; the binding is that field's). The second declaration of a warned clash is not declared, so lookups find the first, which is the type the Indexer keeps (first declaration wins). ARIMA_DynReg keeps its report-mode E-SCOPE-UNRESOLVED (DynamicRegressor@field "x"): Load mode does not bind the TimeSeriesModel body and the engine never resolves that name, so it loads. B45: the record-validation error names the field and its input ("Invalid value for field: d (input "-1.25")"; it was "Sample: {...}didn't pass input validation"); no fixture row has that error, 933 cells of the 600-case fuzz corpus (seed 20261118) change only that text. The per-record path is base's `MiningSchema::validate`; the field is looked for (`first_invalid`, in a cold out-of-line throw) only after it fails. That was not what made a first build's HousingLinearRegressor eval 3-7% slower in every run (base's `internal_model.h` alone kept it): the binder's memory, freed after the load, is reused by every record's temporaries. A build that never freed the binder measured 0.98-0.99, one that kept only `lookup`'s 12 KiB context cache 1.01-1.02 and still failed the omp eval gate once (1.062, re-run 1.042 [1.016,1.064]). The `LoadScope` is therefore kept with the model (`InternalEvaluator::load_scope`, a `shared_ptr<const void>` set by `core/modelbuilder.h`, outside the table's files): 30 KiB for a small model, 150 KiB for HousingGBTlassifier_PCA, 320 KiB for AuditRandomForest (whose document alone takes 40 MiB to load), never read after the load (it points into the freed document); P7 makes its scopes the engine's. `lookup`'s cache is a member array (no allocation of its own). Fixtures (each failed on base with "unordered_map::at: key not found"; the design's `vp/p2` probe is not on disk, and B12 is covered by these, the NormDiscrete-over-a-later-field part being the design probe two_nd_fwd): `fix_forward_ref_derivedfield` (TD NormDiscrete and MapValues over later TD fields, a two-column LocalTransformations MapValues over later string and boolean fields; JPMML), `fix_forward_ref_tree` (tree predicates over a LocalTransformations MapValues of a later field; JPMML), `fix_forward_ref_output` (an OutputField's MapValues over a later OutputField, D12; JPMML refuses it, expectations from Transformations.html) and `fix_forward_ref_fuzz_c000049`, `_c000104`, `_c000494`, `_c000580` (tools/fuzz/gen.py --seed 20261118, context 2, the rows JPMML scores outside known departures; a predictedValue OutputField added). Two other forward cases of that corpus (c000196, c000332) now fail at load on their key text (`""` for an integer field, `TRUE` for a boolean one), as a backward reference already did. Dump diff against 425b709: 167 cells in 8 fixtures (163 in those 7, 4 in `fix_scope_sibling_localtrans_type`: rows 1, 2, 4, 5), none elsewhere. Bench (tools/bench-ab.sh against 95f476e, build directories, seq and omp, 36 models, both modes): the eval gate passes with no model failing or re-run (geomean eval/read-all seq 0.995/0.999, omp 0.999/0.994; HousingLinearRegressor 0.998/0.991 seq, 1.024/1.030 omp, every interval's lower bound under 1.005). The load gate passes on every model that loads in 1 ms or more (GBTs 1.06-1.10, IrisRandomForest 1.10-1.12 with lower bounds under 1.10, forests 1.02-1.06) and fails, under waiver W1 (§6.4 item 4), only on sub-millisecond loads by 10-45 µs each: IrisMultinomReg 1.33-1.41 (83-97 µs base), IrisLinearReg 1.20-1.28, IrisTree 1.22-1.33, HousingLRClassifier and HousingLinearRegressor 1.16-1.23, ensemble_iris_dectree 1.15-1.21, single_iris_dectree 1.20 (omp), AuditTree 1.15-1.18 (omp), AuditBinaryReg 1.16-1.18 (omp, about 0.15 ms base; found by the verifier). HousingLRClassifier stays 2-4% slower per record in seq and omp (seq eval 1.025 [1.002,1.045] on a 31-round re-run), within the eval gate (lower bound at most 1.03). A first build of this commit failed the load gate above 1 ms too (HousingGBTlassifier_PCA read-all 1.143, re-run 1.123, 2.25 ms base): besides the name-only bodies above, `check_param` and `bind_apply` now build their message strings only for a diagnostic and `resolve`/`bind_mining_schema` look names up by `string_view`, which took the binder from 124 to 75 µs on that model.

As built (P2.2; rebuilt on 529130a after a first version, 98e3274, was dropped, then reworked after a verification round): the engine's constructors read the document text of the table's files in its pmml-4-4.xsd type or by the §3.4 K1 rule. Values are still the old `Value`, in which `Value(text, DataType)` already is K1 (integer, float and double are doubles, so an integer field's `33.0` is the double 33, rule 2, and float text keeps its double reading until P6, B4): what changes is which type the text is read in. NormDiscrete@value is read in its field's type from `scope_resolve` (B10: always a string before, so it never matched an integer, double, float or boolean field; D29: `3.5` on an integer field loads and the unit is 0 for every input, fixture `fix_normdiscrete_integer_decimal`, spec-derived since JPMML 1.7.7 refuses the document). LinearNorm@orig/@norm, NormContinuous/NormDiscrete@mapMissingTo, MiningField@lowValue/@highValue and every Interval margin (DataField, Discretize) are NUMBER whatever the enclosing or field type (B11, L1, L2). FieldRef@mapMissingTo is read by K1 in T(field). Apply@mapMissingTo/@defaultValue are parsed strictly in the Apply's result type, which until P2.4 is `BuiltInFunction::expression_kind` (so `equal` with `mapMissingTo="true"` in a double DerivedField loads; base refused it). Discretize and MapValues values (binValue, output cells, mapMissingTo, defaultValue) are parsed strictly in their own dataType, else the nearest enclosing declared type (D21, taken from the enclosing DerivedField or OutputField element so that integer stays apart from double: `1.50` or `2.0` as an integer binValue is E-VALUE-MALFORMED; JPMML loads `2.0`). D28: NumericPredictor@exponent (pmml-4-4.xsd `type="INT-NUMBER"`) and a GeneralRegression covariate's PPCell@value (GeneralRegression.html: "the cell value is set to the exponent"; the XSD types it xs:string, so its integer reading is D28's decision, whose row names "NumericPredictor/GRM `exponent`") are xs:integer: `2.0`, `2e0`, `2.5` are load errors, `+2`, `02`, ` 2 ` are 2 (B7; fixtures `fix_numericpredictor_exponent_text` and `fix_grm_exponent_text`, strict_parse.cc). The GRM exponent is read once at load into `PPCell::exponent`; it was parsed as a double on every record that reached the cell, and malformed text failed each of them. MapValues key cells are read by K1 in each FieldColumnPair field's type; two rows whose keys are equal once typed are E-KEY-DUPLICATE (B42: integer `3`/`3.0`, double `3.5`/`3.50`, float `0.1`/`0.10000000149011612`, `0`/`-0` by KeyEqual, boolean `true`/`1`, one string twice; JPMML 1.7.7 refuses each lazily, "Element InlineTable is not valid" on the first lookup that reaches the table). A NaN key is E-KEY-NAN (D26), as is a NaN member of a SimpleSetPredicate of any size. `fix_mapvalues_nan_key` is re-derived: its NaN row is removed (it never matched, so no expected value changes), and strict_parse.cc checks that the document as registered gives E-KEY-NAN. The errors are thrown by the constructors as `ParsingException`s carrying the binder's codes (E-VALUE-MALFORMED, E-ATTRIBUTE-MALFORMED, E-KEY-NAN, E-KEY-DUPLICATE), not by `LoadScope::check`: Load mode binds neither expression nor model bodies, and the constructors read every such element, inline DerivedFields included.

Constants and slots (§3.4 "DerivedField dataType=X -> Cast<X>(expression)"). A FieldRef or Apply constant (mapMissingTo, defaultValue) that gives the value of a DerivedField or OutputField, as its expression or through the value arguments of `if` (`bind::slot_holder`), is cast at load to a number or boolean type the field declares (`bind::constant_store`): a string field's `5`, concat's `5`, and `if` over an untyped MapValues or Discretize with defaultValue `5` are the double 5, as base and JPMML give (fixture `fix_mapmissing_slot_cast`, which 98e3274 broke). A constant with no such value is E-VALUE-MALFORMED (base refused those too). In a string slot, decided 2026-11-18 (§9: the constant "keeps its text as well as its number"): the old storage holds one double per value, and equality readers and numeric readers need different ones, so the slot's value is the string, the text with its string code, as base stored it, and `Value::numeric_text` (a new bool in the padding after `missing`, so `sizeof(Value)` is unchanged) marks it as a number's or boolean's text (`bind::string_slot_constant`). Every equality reader (SimplePredicate, SimpleSetPredicate, MapValues key, NormDiscrete, `equal`, `notEqual`, `isIn`) compares the text as base did; string readers (a string OutputField, concat, uppercase) read the text; only an OutputField reads the number: a numeric or boolean OutputField over the slot, or over a number-typed DerivedField that copies it, takes `Value::text_number()` in `TransformedValue::eval_double` (the only per-record change; the in-place string edits clear the mark). Every other numeric reader still reads the string's code until P4.7's Cast nodes (or P6.8, B38, if a cast over the old storage still reads codes): the number-typed DerivedField itself holds the code (no cast is applied there before P4.7), so a NumericPredictor, a numeric Apply or a predicate's numeric comparison over that DerivedField, or over the slot directly, reads the code. Probe and pending fixture `test/data/pending/fix_mapmissing_string_slot_numeric` (decided by the main agent: not a regression, base gives the same, and no per-record work is added for it now): string DerivedFields `s` (FieldRef d mapMissingTo="5") and `s3` (FieldRef i mapMissingTo="2.5"), double DerivedFields `t` and `t3` copying them, NumericPredictors `t` (coefficient 1) and `t3` (100); with every input missing the OutputFields `ot` and `ot3` give 5.0 and 2.5 on P2.2 as JPMML 1.7.7 does (base: the codes 0.0 and 2.0), but the prediction is 200 on base and P2.2 (codes 0 and 2) where JPMML gives 255. One more text this storage gives: a number constant `-0` in a string slot keeps its text `-0` for string readers, and a double OutputField over it now reads the number -0, written `-0.0` (D18 keeps the sign); base read the string code and wrote `0.0`, as JPMML 1.7.7 does. Whether -0 is kept or normalised to +0 is Q13, decided before P5. Fixtures `fix_mapmissing_string_slot` (string, numeric and boolean readers) and `fix_mapmissing_string_slot_compare` (every equality reader, the tree included, and the `if` path), every cell JPMML 1.7.7's. A constant that is an argument of another function keeps its own type and carries its text (B11: `+` over FieldRef d mapMissingTo="5" adds 5; concat gives "5z"). NormContinuous/NormDiscrete@mapMissingTo (NUMBER) in a string slot are stored the same way, with the double's text: `mapMissingTo="9"` is "9.0" (§3.4 Cast<string> with D17's Java number text; JPMML gives "9.0", base "9", and the first rebuild no value). Discretize and MapValues values whose own type is of another kind (string, boolean, number) than the enclosing declaration are cast to it at load (§2.6; MapValues dataType="double" `1.50` in a string DerivedField is the text `1.5`, as JPMML gives), and a number cast to a string slot is marked too, so a double OutputField over it reads 2.5 as JPMML does (base read the string code). Gaps that remain are base's and wait for P4.7's Cast nodes, which cast every value of a slot, not only document constants: a number or boolean input in a string slot keeps its number, so equality against text fails (`test/data/pending/fix_mapmissing_string_slot_equal` row 2; a tree probe with b = true: JPMML node A, base and P2.2 node B) and its string readers get no value; a double OutputField over a string slot holding an input string reads the string's code; a string OutputField over a double DerivedField that copies the slot writes "5" where JPMML writes "5.0"; a numeric Apply over a string slot reads the code (for `+` JPMML refuses the document: "No PMML data type for the intersection of PMML data types string and integer"). Known and out of scope: `output/transformedvalue.h` catches a `ParsingException` from an OutputField's expression, so these load errors inside an OutputField leave the field silently empty instead of failing the load (NormDiscrete value="abc" on a double field gives N/A, not K1 rule 3's load error) until P2.5 deletes the catch (B13); an untyped Discretize binValue `2.50` in a string DerivedField read by a double OutputField gives a string code in base and P2.2 where JPMML gives 2.5 (D21 makes its type the enclosing string, so there is no number to keep; P2.3, P6).

Outside the table's files: `bind/load_scope.h` (the shared readers `type_id`, `declared_type_of`, `enclosing_type`, `engine_value`, `document_constant`, `k1_type`, `slot_holder`, `constant_store`, `string_slot_constant`, `slot_constant`, `number_attribute`, `number_constant`, `is_nan_key`), `core/value.h` (`numeric_text`, `text_number`), `output/transformedvalue.h`, `core/predicatebuilder.h` (the SimpleSetPredicate NaN member), `test/data/pending/MANIFEST.md` and the pending fixtures `fix_mapmissing_string_slot_equal` and `fix_mapmissing_string_slot_numeric`. `regressionmodel/categoricalpredictor.h` and the DataField Values needed no change (they already read `Value(text, the field's type)`); B7's GeneralRegression factor half ("level `1.5` loads, `1.0` matches") is what D22 decides (§3.4 rule 2), so it is kept. Departures from JPMML 1.7.7, the spec deciding: a NaN key loads there (D26); `exponent="2.5"` loads and scores x^1 there, and a GRM covariate `2.5` scores x^2.5 (D28); NormDiscrete `3.5` on an integer field is a NumberFormatException there (D29); an integer binValue `2.0` loads there (XSD integer lexical space); `missingValueReplacement="2.5"` and an asExtremeValues bound `1.5` on an integer field are refused there, and D22 widens the slot (`fix_miningfield_replacement_widening`, spec-derived).

Fixtures (base = 529130a) that fail on base: `fix_normdiscrete_field_type` (B10; "output 'o_ndm': predicted: 0  expected: 1.0"), `fix_normdiscrete_integer_decimal` (D29; "output 'nd30': predicted: 0  expected: 1.0"), `fix_attribute_type_elements` (B11; "output 'o_mvb': predicted: 1  expected: true"), `fix_attribute_type_missing` (B11; '"true" is not a valid double'), `fix_mapmissing_string_slot` (the decision; "output 'o2n': predicted: 0  expected: 5.0"), `fix_mapmissing_string_slot_compare` (the decision; "output 'o_nd7n': predicted: 11  expected: 7.0"); all JPMML-agreed except `nd35`. Guards the base build passes: `fix_mapmissing_slot_cast`, `fix_tree_integer_decimal_text` (review `tree_int.pmml`), `fix_miningfield_replacement_widening` (D22), `fix_numericpredictor_exponent_text` and `fix_grm_exponent_text` (D28). strict_parse.cc `document_text()`: 33 checks the base build fails. Dump diff against 529130a: 75 cells, all in the six fixtures above that fail on base (`fix_normdiscrete_field_type` 18, `fix_normdiscrete_integer_decimal` 1, `fix_attribute_type_elements` 8, `fix_attribute_type_missing` 25, `fix_mapmissing_string_slot` 9, `fix_mapmissing_string_slot_compare` 14), none in the other 350; the 23 benchmark models dump unchanged. Fuzz corpus (tools/fuzz/gen.py --seed 20261118, 600 cases), every changed case re-scored with JPMML 1.7.7 row by row: 1519 cells in 98 cases. Of the cells both builds score, 43 moved to JPMML (NormDiscrete over a boolean, double or float field in c000053, c000081, c000159, c000314, c000326, c000404, c000453; a pow mapMissingTo `-999` read by concat in a string DerivedField, c000131), 2 are a row JPMML refuses, and none moved away from JPMML. 49 cases now fail to load on the rules above (integer text of a Discretize, MapValues or Apply value 37, E-KEY-NAN 8, E-KEY-DUPLICATE 4, whose tables JPMML refuses at their first lookup; JPMML refuses every row of 8 of the 49), taking 1240 cells, 575 of which base scored as JPMML does; 35 cases base already refused fail with the new message; 6 cases load now (an Apply constant read in the Apply's result type), and of their 144 cells 48 are JPMML's, 87 are rows JPMML refuses, and 9 are rows cPMML refuses for an integer input `1.0` or `-999.0` (strict input text, P6.8). In `test/data/pending`, 36 cells of 4 fixtures change: 24 now hold the fixture's expected text (NormDiscrete over integers, MapValues own types, float NormDiscrete), and 8 in `fix_typed_derived_conversions` are string codes read as numbers (B1, B38; P6), not behaviour of this item: `o_td` rows 1-2 (4.0 -> 5.0, 8.0 -> 2.0: a double DerivedField over a string input reads its code), and `e_s_c3`, `e_s_c30`, `e_isin` rows 1 and 2 (row 1 true -> false, row 2 false -> true), which compare a string field with the untyped Constant 3 by code and now hit on row 2 instead of row 1 because fewer strings are interned at load; and 4 in `fix_mapmissing_string_slot_numeric` (added by this commit, MANIFEST note 9; it fails on base and on P2.2 with "predicted: 200.0  expected: 255.0"): `ot` 0.0 -> 5.0, `ot3` 2.0 -> 2.5, `os2n` 1.0 -> 10.0, `os4n` 3 -> 7 on row 1, each JPMML's. bind_baseline.txt: the new fixtures' warnings added (W-CAST-NARROWING, W-TEXT-WIDENED, W-SLOT-WIDENED; for the pending `fix_mapmissing_string_slot_numeric`, W-TEXT-WIDENED on `s3`'s `2.5` and W-CAST-NARROWING on `t`, `t3`, `os2n`, `os4n`) and the E-KEY-NAN line of `fix_mapvalues_nan_key` gone. Bench (tools/bench-ab.sh, build directories 529130a and this tree, 36 models, both modes; per-record code changed in `TransformedValue::eval_double`, the `Value` string edits and the GRM covariate term): the gate passes in the sequential build (geomean eval/read-all 1.009 [1.006, 1.015] and 1.008 [1.004, 1.012], under the 1.01 bound; highest per-model lower bound 1.025, ensemble_iris_dectree, under 1.03; tree models drift 2-5% in both directions between runs, as the run before this rework, with no per-record change, also showed) and in the OpenMP build (geomean 1.002 and 0.995; highest lower bound 1.006); no model failed or was re-run; load passes item 4 against 529130a in both builds with no waiver (highest lower bound 1.003). A sequential re-run after the verification round (same builds, fresh; load average 3-10 from other jobs on the machine, never under 2) passes too: geomean eval/read-all 1.008 [1.004, 1.016] and 1.008 [1.003, 1.012], highest per-model eval lower bound 1.024 (IrisLinearReg), highest load lower bound 0.997, no model failed or re-run; the verifier's idle re-run gave 1.009 and 1.006.

As built (P2.3, base 39b85e6): the engine's `Constant` (`expression/constant.h`) is typed by the §3.4 rows, with rules shared with the binder in `bind/typing.h`: `parameter_type` and `inherited_constant_type` (S2: the type an untyped Constant argument inherits from its built-in's parameter, read from `fn::kBuiltins` by its position after `Extension*`; the binder's own copy is gone; only where the spec's pseudo-declaration gives the ParameterField a dataType: the table's boolean parameters of `and` and `or` ("The function takes a variable number of fields as parameters", no ParameterField) and integer `referenceYear` of `dateDaysSinceYear`/`dateSecondsSinceYear` (`<ParameterField name="referenceYear" optype="continuous"/>`) still check the argument's type but give nothing to inherit, so an untyped `true` passed to `and` is a string, E-TYPE-BOOLEAN-MIXED in report mode, where JPMML 1.7.7 reads it as a boolean; the engine refuses `and`, `or` and the date functions as unsupported until P4 and P10) and `parse_constant` (the type's XSD lexical space, plus INF, -INF and NaN in any case for float and double, S3; the binder's `bind_constant` calls it). A Constant with `dataType` reads its text strictly in that type (Transformations.html: "The dataType of Constant can be optionally specified"; the strict reading is the XSD lexical space quoted next), so `<Constant dataType="integer">1.5</Constant>`, `1e3` and `INF` as an integer, `Infinity`, and `TRUE` as a boolean are load errors (E-VALUE-MALFORMED; base read integer text as a double and refused the specials in lower case; XML Schema Part 2 §3.3.13.1: integer is "a finite-length sequence of decimal digits (#x30-#x39) with an optional leading sign", §3.2.2.1: boolean "can have the following legal literals {true, false, 1, 0}"); an untyped Constant passed to a parameter that declares a dataType is read in it (S2: "If the ParameterField definition includes a dataType, the Constant will inherit the dataType specified in the ParameterField": `if(<Constant>true</Constant>, …)` and `not(<Constant>false</Constant>)` are booleans and an untyped `TRUE` there is a load error, `substring`'s `<Constant>1.0</Constant>` is a load error, `uppercase(<Constant>1.50</Constant>)` is "1.50"); any other untyped Constant is inferred by `infer_constant` (B9: `1e10` is a double, `3000000000` an int64, `NaN`/`INF`/`-INF` in any case strings, D10; beyond int64 a load error, D4) and a string keeps its text (B8: base stored only the string code, so `uppercase`, `concat`, `stringLength` and a string OutputField got no text; the spec's `concat(month, "-", year)` lost its "-"). Two rows of the Constant spec the base build did not read are applied too, since they are the §3.4 row of the same element: "If the content is empty ... unless dataType is something besides "string", in which case, it will be taken as a missing value of the specified type" (base: a load error) and "If the missing attribute is true, the constant will be taken as a missing value, regardless of the content" (base ignored the attribute; the binder's `bind_constant` no longer reads the text of an untyped missing Constant either, so `<Constant missing="true">99999999999999999999</Constant>` gives no E-VALUE-MALFORMED in report mode, bind_report rule `constant-missing-untyped`). Storage stays the old `Value`: integer and double are doubles, a float's text keeps its double reading (B4, P6), a boolean is 1/0, a string its code and text, a missing Constant `Value()`; the temporal dataTypes are read as before (P10). The reader is aligned with the binder's (P2.1 left it open): pugixml's default flags, which the engine keeps, drop a blanks-only text child, so a Constant with no child node is read as the binder reads it through `LoadScope::lone_blanks` (new, `bind/load_scope.h`, whose only other change is that `Active` drops its document): the loader's document text is parsed a second time, once per load and only when such a Constant asks, with the binder's `parse_default | parse_ws_pcdata_single`, and the element is found there by its element index under each ancestor (the two parses differ only in text nodes); the second document is dropped when the build ends (`LoadScope::Active`). Its blanks are therefore converted from the document's encoding and their line ends normalised exactly as the engine's text is (XML 1.0 §2.11: "translating both the two-character sequence #xD #xA and any #xD that is not followed by #xA to a single #xA"; JPMML agrees, `uppercase` of a CR LF Constant is one LF), or "" for `<Constant/>`, comments not being content. A typed `<Constant dataType="double"> </Constant>` is therefore E-VALUE-MALFORMED as in report mode, not missing, and an untyped or string one keeps its blanks. (A first draft read the blanks from the raw bytes at `offset_debug()`, which does not point into them once pugixml has converted an ISO-8859-1 document, and refused such documents; no fixture or benchmark model has an empty or blanks-only Constant.) Text split by a comment or CDATA section is still read up to its first text node, as base did (`1<!--c-->5` is 1; JPMML 15): not this item's. Not changed (smallest choice, for the user): the Constant's value is its own type's, never cast to an enclosing DerivedField's or OutputField's dataType (P4.7's Cast nodes; P2.2's `constant_store` cast covers attribute constants only), so a string OutputField over a double DerivedField whose expression is the untyped Constant `A` now writes `A` where base wrote the string code as a number (`0.0`, fuzz c000021, c000233, c000516; JPMML refuses the row: "For input string: A"; Q2 makes it an invalid record at P4.7); and `BuiltInFunction::expression_kind` (`core/builtinfunctions.h`, deleted by P2.4) still kinds an untyped `NaN`/`INF` as double and padded digits as integer when it picks a number's text or float/integer arithmetic for an Apply, which no fixture, benchmark model or fuzz case reaches. Departures from JPMML 1.7.7, the spec deciding (probed one OutputField per document): untyped `1e10` is a double (JPMML: a string, "No PMML data type for the intersection of PMML data types double and string"); `3000000000` an int64 (JPMML refuses it, D4); untyped `INF`, `inf`, `nan` are strings (JPMML: Infinity, NaN; D10); S2 strings keep their text where JPMML infers a number first (`stringLength(<Constant>1.50</Constant>)` 4, JPMML 3; `uppercase` "1.50", JPMML "1.5"; `stringLength(<Constant>INF</Constant>)` 3, JPMML 8), and `substring(s, <Constant>1.0</Constant>, …)` is a load error (JPMML scores it as 1). Typed `TRUE` as a boolean (JPMML: true, Java's `parseBoolean`), typed `1e3` as an integer (JPMML: 1000) and untyped `TRUE` as the condition of `if` (JPMML: true) are load errors by the XSD lexical spaces quoted above. These are D10 and S2 as decided; the S2 and XSD ones have no D-rule row of their own.

Fixtures (each fails on base): `fix_constant_untyped` (B8, B9, D10; "predicted: 1.5  expected: abcde", base dump: `cc_abc` N/A, `cc_my` 072000, `big` 7.5, `cc_sp` N/A), `fix_constant_parameter_type` (S2; base fails every record, "evaluating apply function") and `fix_constant_typed_text` (S3, empty and `missing` Constants; "predicted: 1.5  expected: Infinity": base refused `inf`, and the OutputField's catch, B13, left it N/A). JPMML 1.7.7 gives every cell except those named in each header (spec-derived: the departures above, and `dnan`, D23). strict_parse.cc `constants()`: 38 checks, 29 of which the base build fails (strict typed text, S3 case, empty and blanks-only content, the `missing` attribute, int64 overflow, S2 malformed text, CR LF and lone CR blanks read as one LF, blanks of an ISO-8859-1 document); every rule was mutation-checked (8 mutations of `constant.h`/`typing.h`, each caught by a fixture or a strict_parse check). Dump diff against 39b85e6: 81 cells, all in the three new fixtures (27, 33, 21), none in the other 356; the 23 benchmark models dump unchanged; in `test/data/pending`, 8 cells of `fix_builtin_text_concat` `c_spec` (rows 1-5, 7-9) N/A -> `-`: the untyped "-" keeps its text (B8), the integer month and year are still dropped until P4.6 (B23; MANIFEST sentinel updated). Fuzz corpus (seed 20261118, 600 cases): 375 cells in 25 cases, re-scored with JPMML row by row; none moved away from JPMML. 71 cells moved to JPMML's text (untyped `NaN` written NaN where base wrote 0.0 or nothing, c000048, c000262, c000545; untyped `1e10` 1.0E10 where base wrote 0.0, c000411; untyped `-0`, an integer, 0.0 where base wrote -0.0, c000210, c000360, c000507); 9 cases now fail to load on a typed integer Constant `2.5` or `1e10` (c000014, c000047, c000063, c000082, c000098, c000263, c000277, c000394, c000474; JPMML refuses every row of each), taking 232 cells; 2 cases base already refused fail with the new message (c000256, c000330); 61 cells are rows JPMML refuses, where head now writes a Constant's text (S2 `lowercase(<Constant>inf</Constant>)` c000017, S2 `substring` of `A` or `1e10` c000240 and c000419, the untyped `NaN` of a float field written as text in c000212, whose row JPMML refuses for a `+` giving NaN, D23, and the three cases of the not-cast gap above). bind_baseline.txt: only the document count (401 -> 404; the new fixtures bind with no diagnostic); bind_report.cc: 2 rule checks (`constant-no-inherit-and`, `constant-missing-untyped`), 125 in all. Load only: no code run per record changed (`Constant::eval` returns the value it read). Gate anyway, for the load time (tools/bench-ab.sh, build directories 39b85e6 and this tree, 36 models, both modes; the builds have no OpenMP, so the omp build was not run), re-run after the blanks reader moved to `LoadScope::lone_blanks`: GATE PASS with no model failing or re-run; load ratios 0.92-1.20 (the ends on sub-millisecond loads with wide intervals: fix_association_output_defaults, fix_gp_missing_continue), highest per-model load lower bound 0.978 (HousingRFRegressor_PCA); no benchmark model has an empty or blanks-only Constant, so none parses the document twice; eval geomean 1.002 [0.995, 1.006] and read-all 1.004 [0.997, 1.010].

As built (P2.3 follow-up `+INF`, base c76f5da; decided by the user 2026-11-19): D30 and D31 (§8.5). (1) Typed float and double text reads `+INF` as +∞, by XML Schema 1.1 Part 2 §3.3.4.2 (float) and §3.3.5.2 (double), "the ·literals· 'INF', '+INF', '-INF', and 'NaN'", the version the Constant text links ("see https://www.w3.org/TR/2012/REC-xmlschema11-2-20120405/#float"); XSD 1.0, which `lexical.h` cited, had no `+INF`. The one grammar is `lexical::parse_double` (`utils/lexical.h`, one more literal on its slow path; the fast path, which needs a digit or `.` first, is unchanged), so the rule reaches every reader of typed float/double text at once: attributes (`XmlNode`'s typed getters, the binder's `number_attr`), document Value text (`read_document_text`, `Value::to_double`, MapValues/table cells, Array members, `is_nan_key`), the types library's `parse_as<Float/Double>`, and input values of float/double fields (`MiningField::parse_input` already shares that grammar, so `+INF` is a valid input there rather than left to P6.8; an integer field still takes no INF). It is case-sensitive like the other literals (`+inf` stays malformed outside Constants, D11). Constant text adds `+inf` in any case to S3's case-insensitive specials (`types::detail::constant_special`, `core/types/parse.h`); `+NaN` is no literal, and an integer Constant takes no `+INF` ("No other data type will be accepted for these constants"). An untyped `+INF` stays a string (D10: `infer_constant` admits only digits, `.`, e, E, -, +). `expression/constant.h` needed no change (its float branch reads `+INF` through `lexical::parse_double`). (2) Untyped `1E400` is the double Infinity and `-1E400` -Infinity, as P2.3 committed ("A scientific notation expression like 3.14e+5 or like 3.14E-5 should also be treated as a double"; the out-of-range decimal rounds); now pinned by tests. JPMML 1.7.7 (probed one OutputField per document) refuses typed `+INF`/`+inf`/`+Inf` Constants and a MapValues key `+INF` (NumberFormatException), a LinearNorm `orig="+INF"` ("Required attribute LinearNorm@orig is not defined") and an input `+INF` ("Field "d" cannot accept invalid value"), loads an Interval `rightMargin="+INF"`, and reads untyped `1E400` as a string ("No PMML data type for the intersection of PMML data types double and string").

Fixtures: `fix_plus_inf` (new; base: "E-VALUE-MALFORMED: Constant text "+INF" is not a value of type double (§3.4)", a load error), and the `d=+INF` row of `fix_invalid_input_asmissing` and `fix_invalid_input_asvalue`, which was JPMML's "invalid" and is now +Infinity (`y` and `od` Infinity; Annotation updated; base: "predicted: 111.0  expected: Infinity" and "predicted: 117.0  expected: Infinity"). strict_parse.cc, typed_value.cc and bind_report.cc (`constant-plus-inf`, `constant-plus-inf-untyped-string`, `constant-1e400-double`, `attribute-plus-inf`, `attribute-plus-inf-lower`; 130 rule checks) pin both rules; built against the base library, typed_value fails 4 checks, bind_report 2 rule cases, and strict_parse 3 checks before aborting on the first `+INF` input. Mutation-checked, each caught: no `+INF` in `lexical.h`; `+inf` accepted there; `+INF` read as -INF; no `+inf` in `constant_special`; a float Constant refusing `+INF`; untyped out-of-range scientific notation as a string; untyped `+INF` as a number; inputs refusing `+INF` (8 mutations of a copy of the tree). Dump diff against c76f5da: 41 cells in 3 fixtures (36 cells of `fix_plus_inf` replacing base's load error, and `y`, `od` of row 5 in each of the two invalid-input fixtures), none in the other 364; `test/data/pending` (800 cells) and the 23 benchmark models (593720 cells) unchanged; no benchmark model or dataset contains `+INF`. bind_baseline.txt: only the document count (411 -> 412). Input parsing is per record, so the gate ran (tools/bench-ab.sh, build directories c76f5da and this tree, 36 models, both modes; no OpenMP build): GATE PASS with no model failing or re-run; eval geomean 1.004 [0.996, 1.007], read-all 1.008 [1.000, 1.014]; load ratios 0.79-1.04, highest per-model load lower bound 1.001 (HousingRFRegressor_PCA).

As built (P2.4, base 504479a): `integer` is its own `DataType` value (`core/datatype.h`: DataDictionary.html "The dataType reuses names and semantics of atomic types in XML Schema", and pmml-4-4.xsd's DATATYPE lists `integer` apart from `double`); `DataType::is_integer`, `DataField::integer`, `OutputField::integer` and the two-argument `java_kind` are gone, `MiningField::parse_input` reads an INTEGER input by the XSD integer lexical space as before, and `LoadScope`'s `engine_type`/`type_id` map Integer to INTEGER. Values are still the old `Value` (an integer is a double). The XML re-walk (`BuiltInFunction::expression_kind`, `field_kind`, `declared_in`, `declared_kind`, `result_kind`, `common_kind`, `arguments_of`) is deleted; `bind::expression_type` (`bind/load_scope.h`) gives the static type of an expression element by the binder's §3.4 rows (Load mode does not type expression bodies, so it is worked out on demand, without diagnostics): Constant by its dataType, S2 or `infer_constant`; FieldRef by `Binder::value_type` and §3.4's `mapMissingTo` rule (`value_type` is `lookup`'s binding, except that a LocalTransformations DerivedField, or from inside the Output an OutputField, that P2.1 accepts with W-NAME-SHADOWS-TD or W-NAME-SHADOWS-MININGFIELD and does not declare gives its own dataType: the engine computes it into the name's slot, so its value is the one read, and the re-walk typed it so; `Scope::local_shadows`/`output_shadows`, `bind/scope.h`); NormContinuous, NormDiscrete, TextIndex double; Discretize and MapValues by their dataType, else D21's enclosing DerivedField or OutputField; Apply by the table's result rule, `apply_result_type` (`bind/typing.h`, now shared with `Binder::bind_apply`). The Apply constructor (`core/builtinfunctions.h`: float narrowing, float result, integer `/` and `pow`), `Apply::result_constant` (`expression/apply.h`) and `TransformedValue::kind_of` read it; a type the binder does not know (an arithmetic operand that is boolean or string, Q7; a FieldRef to an OutputField whose type cannot be inferred) computes and writes as a double, as the re-walk's "double" fallback did. Files outside the table, each a one-line consequence of deleting the `integer` flags: `core/miningfield.h`, `core/datafield.h`, `core/internal_model.h`, `ensemblemodel/multiplemodelmethod.h`, `anomalydetectionmodel/anomalydetectionevaluator.h`, `expression/apply.h`, `bind/{binder,load_scope,typing,scope}.h`, `core/xmlnode.h` (comment); `output/outputexpression.h` is unchanged (`value_kind` stays until §2.9's P5). Smallest choices, for the user: (1) `if` keeps the engine's type, its THEN argument's (`engine_apply_type`, typing.h), and not the binder's D7, because the table closes B19 in P2.6 and P2.4 closes nothing; probe `/`(if(x>0, 7, 2.5), 2) in a double DerivedField: base and head 3.0 / 1.0, JPMML 3.0 / 1.25, D7 would give 3.5 / 1.25 (P2.6 deletes `engine_apply_type`). (2) An OutputField without a dataType over an integer target stays DOUBLE (`output/outputfield.h`), as it was when integer was carried as DOUBLE, so a selectAll collection still writes it `7.0`; the binder, which types it by its target (W-OUTPUT-NO-DATATYPE; pmml-4-4.xsd makes the attribute required), gives a FieldRef to it the type integer, so `/`(FieldRef p, 2) over such a predictedValue 7 is now 3.0 (base 3.5, JPMML 3.0). (3) A FieldRef outside the Output to a MiningField that an OutputField of the same model shadows (W-NAME-SHADOWS-MININGFIELD) has the MiningField's type, §3.2 step 1 making OutputFields visible only from inside the Output; the re-walk typed it by the OutputField. Probe: integer MiningField x, OutputField x double = d, LocalTransformations w = `/`(FieldRef x, 2), x = 7, d = 7.4: the engine reads the input 7 there (LocalTransformations are computed before the Output), base 3.5, head 3.0, JPMML 3.7 (it evaluates w lazily from the Output and reads the OutputField). The document is invalid (FieldScope.html: OutputField names "must be unique from any other names in their scope"); no fixture. The binder's types differ from the re-walk's only where no fixture, benchmark model or fuzz case looked: the dump diff over the 359 existing fixtures is empty, and the fuzz corpus (`gen.py --seed 20261118 --cases 2000`) changes two load-error messages only (c000140, c000482: the same E-VALUE-MALFORMED now names the integer slot and the float Apply). Where a valid document's result changes, a fixture pins it, each failing on base: `fix_apply_untyped_mapvalues_integer` (D21: an untyped MapValues or Discretize in an integer DerivedField, or in an integer OutputField, `o_q3`, is an integer, so `/` by 2 is integer division, 7/2 = 3; the re-walk typed them string and divided in doubles; base "predicted: 3.5  expected: 3.0"; JPMML 1.7.7 refuses the document, "No PMML data type for the intersection of PMML data types string and integer", typing the untyped MapValues as a string, so §8.5 D21's JPMML column, "same", does not hold for an Apply argument), `fix_apply_untyped_mapvalues_float` (D21 for float: an untyped MapValues in a float DerivedField is a float, so its 16777217 is the float 16777216 and `*` 3 gives 5.033165E7; the re-walk multiplied the double and narrowed 50331651 to 5.0331652E7; base "output 't_f': predicted: 5.0331652E7  expected: 5.033165E7"; JPMML refuses it as for integer and gives the fixture's values with dataType="float" on the MapValues) and `fix_fieldref_mapmissingto_widened` (D22: a FieldRef to an integer field with `mapMissingTo="2.5"`, or `"3.0"`, which has an integral value but is not an xs:integer literal, is a double, so 7/2 = 3.5 and a missing input 1.25 (1.5); the re-walk typed it by the field alone; base "predicted: 3.0  expected: 3.5"; JPMML gives 3.0 and refuses the missing row). `fix_fieldref_shadow_type` passes on base and pins `value_type`'s shadows (TD t integer, LocalTransformations t double; MiningField x integer, OutputField x double; `/` by 2 of each in doubles, 3.5; JPMML agrees): a first build of this commit typed both by the first declaration, as `lookup` gives it, and divided as integers (3; with x = 7, d = 7.4, 3.0 where base and JPMML give 3.7). Other rows the re-walk read differently and the binder now decides, unexercised: an untyped `NaN`/`INF` Constant is a string (D10, P2.3; was a double), an untyped Constant passed to a typed built-in parameter has that type (S2), and a FieldRef resolves by `Binder::lookup` (FieldScope.html) rather than the re-walk's own scope walk. Mutation check (in a copy): Integer mapped back to DOUBLE in the DATATYPE table, D21 enclosing type (and D21 for a DerivedField only), each of the two shadow lookups, the FieldRef widening, the Constant's inferred integer, integer `/`, float narrowing, `kind_of`'s integer text and the input's integer lexical space each fail at least one fixture; `if` as D7, S2 inheritance in `expression_type` (no engine path reads an inherited Constant's type: the parameters that declare a dataType belong to functions whose result type is fixed, such as `if`'s condition, `not`, the string and distribution functions, never to an argument of `+`, `/`, `pow` or an aggregate), `engine_type`'s INTEGER and the untyped-OutputField DOUBLE are not observed by any fixture. Gates: ctest 376/376 (372 + the four fixtures); dump diff against 504479a: 0 cells in the 359 existing fixtures and in `fix_fieldref_shadow_type`, 33 cells in the three fixtures that fail on base; `bind_baseline.txt` gains the four documents, two W-TEXT-WIDENED lines (`fix_fieldref_mapmissingto_widened`'s two mapMissingTo, D22), one W-NAME-SHADOWS-MININGFIELD and one W-NAME-SHADOWS-TD (`fix_fieldref_shadow_type`); §6.4 against 504479a passes in both builds with no model re-run and no load failure (geomean eval/read-all seq 1.003/1.004, omp 0.996/1.005; largest load r 1.047 seq, 1.066 omp, under 1.10).

As built (P2.5, base 6ef2d71): B13: `TransformedValue` (`output/transformedvalue.h`) no longer catches the `ParsingException` of its expression, so an OutputField's expression that fails to load fails the load, as it does in a DerivedField (§3.6: "Nothing is caught and turned into a silent default"); P2.2's and P2.3's load errors (E-VALUE-MALFORMED, E-KEY-DUPLICATE, E-KEY-NAN, NormDiscrete text on a numeric field) and the unsupported functions and elements (an unknown `Apply@function`, `formatNumber` until P4.9, `Aggregate`) now hold inside an OutputField too. Beyond the table's "delete the catch", the smallest choice for the same symptom: a transformedValue OutputField with no EXPRESSION is a load error (Output.html: "For this result feature, OutputField must contain an EXPRESSION, unless it is used to refer to a transformed value of a segment model through the segmentID attribute"; with `segmentId` the message says it is not supported, B34, P8.15). Base left such a field empty on every record, or, given `value="f"`, copied field f, a reading the spec does not define ("The attribute value is used in conjunction with result features referring to specific values"); JPMML 1.7.7 refuses both ("Required element OutputField/<Expression> is not defined"). No fixture, benchmark model or fuzz case had an expressionless transformedValue, nor an OutputField expression the catch hid, so none stops loading. B25: `Apply` (`expression/apply.h`) skips its `Extension` children (pmml-4-4.xsd Apply: `Extension` minOccurs=0 maxOccurs=unbounded, then the EXPRESSION group); base read one as an expression element and refused the document in a DerivedField, and, through the catch, left an OutputField empty. An Extension between arguments, against the XSD order, is skipped too, as `BuiltInFunction`'s argument count and the S2 positions of `constant.h` already did (JPMML 1.7.7 scores it so; smallest choice, for the user). The null-expression guards of `TransformedValue::eval`, `eval_str` and `eval_double` are gone (the expression is never null now): the only per-record change. Outside the table's files: `test/strict_parse.cc` (`no_silent_fallbacks()`, 16 checks, 12 of which the base build fails) and `test/data/pending/MANIFEST.md` (the sentinel of `fix_builtin_text_formatnumber`, which now fails at load: "unsupported value \"formatNumber\" for Apply@function"; base gave every formatNumber OutputField N/A). Fixture `fix_apply_extension` (B25; base: 'invalid value "Extension" for expression element in Apply'): Extensions in a DerivedField's Apply, two in an OutputField's, in a nested Apply, before concat's arguments, and before untyped Constants that inherit their parameter's dataType by position (S2: if's condition `true`, substring's startPos and length); every cell JPMML 1.7.7's. Mutation check (in a copy): the Extension skip removed (fix_apply_extension and 5 strict_parse checks fail), the catch restored (7), the expressionless field allowed (4), the `value` copy restored (1). Gates: ctest 377/377 (376 + the fixture); dump diff against 6ef2d71: 25 cells, all in `fix_apply_extension` (base LOAD-ERROR), none in the other 363 fixtures; the 23 benchmark models dump unchanged (2350 cells); the fuzz corpus (`gen.py --seed 20261118 --cases 2000`) scores identically; in `test/data/pending`, 190 cells of `fix_builtin_text_formatnumber` (N/A, and the prediction 0.25, become the load error above), none elsewhere. bind_baseline.txt: the document count only (408 -> 409; the fixture binds with no diagnostic). §6.4 against 6ef2d71 passes in both builds with no model failing or re-run: geomean eval/read-all seq 1.004/0.993, omp 0.987/0.986; highest per-model eval lower bound 1.020 (seq read-all, fix_nb_threshold); load passes item 4 with no waiver (largest load r 1.061 seq, 1.042 omp; highest lower bound 0.998).

As built (P2.6, base d518b9f): the engine types `if` as the binder does, by D7: `bind::engine_apply_type` (`bind/typing.h`), which kept the type of the THEN argument, is deleted, and the Apply constructor (`core/builtinfunctions.h`) and `bind::expression_type` (`bind/load_scope.h`, outside the table's files: the one call that read `engine_apply_type`, and its comment) take `apply_result_type`'s, whose IfBranches rule is the least restrictive type of THEN and ELSE (THEN alone with no ELSE). Basis: BuiltinFunctions.html declares no result dataType for `if` (`<DefineFunction name="if">` with `<ParameterField name="x" dataType="boolean"/>`, `<ParameterField name="A"/>` "THEN part is required", `<ParameterField name="B"/>` "ELSE part is optional"), so Functions.html's inheritance applies: "functions are assumed to inherit the dataType of the current input parameters unless otherwise specified ... When the input parameters have multiple dataTypes, the least restrictive dataType will be inherited by default ... string double float integer". JPMML 1.7.7 gives the type of the branch taken (D7). What reads the type, all at load: an enclosing function's integer `/` and `pow`, float narrowing of arguments and float result (`/`(if(c, 7, 2.5), 2) is 3.5, base and JPMML 3.0; `*`(if(c, 16777217, f float), 1) is the float 1.6777216E7, base and JPMML 1.6777217E7); `TransformedValue::kind_of`, the Java text a string OutputField writes (if(c, 7, 2.5) gives `7.0`, base and JPMML `7`); the Apply's own mapMissingTo/defaultValue, read in its result type (if(c, 7, z double) with defaultValue="1.5" loads; base refused it, "E-VALUE-MALFORMED: Apply defaultValue=\"1.5\" is not a value of type integer"). No per-record code changed: the values are still the old `Value`, and no cast is applied to the branch taken. Smallest choices, for the user: (1) THEN and ELSE with no common type (one boolean, Q7; the binder's E-TYPE-BOOLEAN-MIXED in report mode, not a load error before Q7 is decided) or one of unknown type (a FieldRef to an OutputField whose type cannot be inferred) give `if` no type, so it computes and writes as a double, as P2.4 decided for every type the binder does not know: probe if(x>0, lessThan(x, 5), 5) in a string OutputField, x = 1, -1: head `1.0` / `5.0`, base `true` / `true`, JPMML `true` / `5`; no fixture, benchmark model or fuzz case has one, and the document is invalid (Functions.html: boolean "can not be mixed with other types"). (2) A number branch of an `if` whose common type is string (if(c, 7, "abc")) is still the number in the old storage, with no text, so a string OutputField over it writes N/A for that branch (base wrote `7`, typing it by THEN; JPMML `7`); this is the gap every number in a string slot has until §3.4's Cast<string> (P4.7): base already gave N/A for if(c, "abc", 7) and for a string DerivedField over `+`, so the two branch orders now agree. Casting the branch at run time would be per-record code, which P2 does not add, and would change `equal` over that value (`equal`(FieldRef si, 7) over a string DerivedField si = if(c, "abc", 7) is true on base and in JPMML, base comparing the number); not done. (3) A float field read through an `if` of double type is no longer narrowed by an enclosing function (`*`(if(c, f, 2.5), 3) with f = 0.1: head 0.30000000000000004, base and JPMML 0.30000001192092896, the spec's float 0.1 widened then multiplied 0.30000000447034836): the float input is carried unnarrowed (P2.4's narrowing note), so the float-to-double cast of the branch waits for P5/P6 (B4); no fixture pins it. Fixtures, both failing on base: `fix_if_static_type` (o1 `/`(if(c, 7, 2.5), 2), o2 `/`(if(c, 2.5, 7), 2), o3 `*`(if(c, 16777217, f), 1), o4 `/`(if(c, 7, -3), 2) integer division, s1 and s3 the string OutputFields over if(c, 7, 2.5) and if(c, 16777217, f); spec-derived, JPMML departing as D7 says in o1, o2, o3, s1 and s3 for the rows where the branch taken has another type; base "predicted: 1.0  expected: 3.5") and `fix_if_static_type_default` (the defaultValue above; every cell JPMML 1.7.7's; base the E-VALUE-MALFORMED load error). Mutation check (in a copy): `if` typed by THEN again (both fixtures fail), by ELSE (o2), integer and float giving double (s3, o3), integers giving double (o4). Gates: ctest 379/379 (377 + the two fixtures); dump diff against d518b9f: 19 cells, all in the two new fixtures (10 in `fix_if_static_type`, 9 in `fix_if_static_type_default`, base LOAD-ERROR), none in the other 364; `test/data/pending` 0 of 800 cells; no benchmark model has an `if`; the fuzz corpus (`gen.py --seed 20261118 --cases 2000`, 35 cases with an `if`) scores identically. bind_baseline.txt: the document count only (409 -> 411; the binder already typed `if` by D7, rule check `if-static-type`, and the fixtures bind with no diagnostic). §6.4 against d518b9f (the seq build, both modes; no per-record code changed, so the gate was not required and the omp build was not run) passes with no model failing or re-run: geomean eval/read-all 0.992/0.993; highest per-model eval lower bound 1.001 (eval, fix_association_output_defaults); load passes item 4 with no waiver (largest load r 1.068, IrisScorecard read-all, lower bound 0.957; highest lower bound 1.010).

As built (P2.7, base af0485b): ordering a string against a number, in either order, is the load error E-TYPE-ORDER-MIXED (§2.7, review S5). The rule is one function shared by the binder and the engine, `bind::order_mixed` with `bind::is_ordering` (`bind/typing.h`: `lessThan`, `lessOrEqual`, `greaterThan`, `greaterOrEqual`; BuiltinFunctions.html lists them under one heading with the equality tests, "equal, notEqual, lessThan, lessOrEqual, greaterThan, greaterOrEqual ... Result is dependent on applying either function to two input parameters", and does not name the ordering subset, which is these four by what each tests, lessThan "Check if field A is less than field B"; `equal`, `notEqual`, `isIn`, `isNotIn` only test equality and keep D2). Basis: DataDictionary.html "Categorical field values can only be tested for equality, while ordinal field values in addition have an order defined" and "For ordinal fields, if a sequence of Value elements is given, the sequence defines the ascending order on the respective field" (an order of strings); Functions.html makes the common type of a string and a number string ("The inheritance precedence for mixed type input parameters is as follows: string double float integer"), and text order (`"10" < "3"`) is no order the spec gives the number. Outside the table's file, the smallest change that makes it a load error: Load mode does not type expression bodies (P2.1), so, as P2.2's and P2.3's errors are, it is raised by the engine's constructor, the Apply's `BuiltInFunction` (`core/builtinfunctions.h`), over the argument types it already takes from `bind::expression_type` (P2.4), with the binder's message (`bind::order_mixed_message`, `bind/load_scope.h`: 'E-TYPE-ORDER-MIXED: Apply "lessThan" in the TransformationDictionary orders string against integer (§2.7; ...)'); `Binder::check_comparable` (`bind/binder.h`) calls the shared rule. Every Apply the engine builds is checked (TransformationDictionary, LocalTransformations, Output, inline DerivedFields, nested Applies); an argument of unknown type (Q7 arithmetic, an OutputField whose type cannot be inferred) gives no error, as in report mode. The check runs before the Apply's arguments are built, so an ordering Apply whose argument has its own load error reports E-TYPE-ORDER-MIXED first (fuzz c000394, which base refused with E-VALUE-MALFORMED for `<Constant dataType="integer">1e10</Constant>`). No document rejected: the P1 baseline (and `test/data/bind_baseline.txt` now, 411 documents: every fixture, pending fixture and benchmark model) has no E-TYPE-ORDER-MIXED, so no fixture is re-derived and the baseline is unchanged. JPMML 1.7.7 (probes, scratch) refuses at evaluation, lazily: "Categorical value cannot be used in comparison operations" (a categorical string field on the left), NumberFormatException (a number on the left, or a non-numeric string Constant on the right of a double field: `abc`), 'Values "10" and "4" cannot be ordered' (an ordinal string field with Values against a number that is not one of them). Three departures, the spec deciding, each now E-TYPE-ORDER-MIXED: it orders a **continuous** string field against a double as text (`greaterThan(sc, d)`: false, false, true for `3`/3.5, `10`/2, `abc`/2.0; the optype gives text no numeric order); it orders an **ordinal** string field against a number whose text is one of its Values by the Value sequence (`so` Values 1, 2, 10, `lessThan(so, <Constant>2</Constant>)`: true, false, false for so = 1, 10, 2); and it orders a number field against a **numeric-text string Constant** as numbers (`lessThan(i, <Constant dataType="string">10</Constant>)`: true, false, true for i = 3, 20, 1). The second is a decision for the user: Functions.html makes the common type string ("string double float integer") and DataDictionary.html says "the sequence defines the ascending order", so the spec arguably orders a number whose text is a declared Value by that sequence, as JPMML does; §2.7 makes string × number ordering a load error with no carve-out, and this commit follows §2.7 (the smallest choice: one rule, no lookup of Values at load). Tests: `test/strict_parse.cc` `ordering()`, 26 checks (the ordinal-Value and numeric-text-Constant departures included), the 20 error checks failing on base (each document loads there; the 6 controls, string × number equality either order, string × string and number × number ordering, pass on both); mutation-checked in a copy (load check removed: 20 fail; one operand order only: 9; equality counted as ordering: 2 and bind_report; any string operand: 2; greaterOrEqual not ordering: 3; float not numeric: 1). Dump diff against af0485b: 0 cells in 366 fixtures, 0 of 800 in `test/data/pending`; no benchmark model has an ordering Apply. Fuzz corpus (seed 20261118, 600 cases): 3 cases change, all to E-TYPE-ORDER-MIXED: c000084 (`greaterOrEqual(i, <Constant>NaN</Constant>)`, an untyped NaN being a string by D10; base N/A on every row, JPMML false, reading NaN as a number), c000174 (`greaterThan(NormDiscrete, s)`; base true, JPMML "Categorical value cannot be used in comparison operations") and c000394 (above). Load only, no per-record code. Bench anyway, for the load gate (tools/bench-ab.sh, build directories af0485b and this tree, seq, 36 models, both modes): GATE PASS with no model failing or re-run, geomean eval/read-all 1.004/1.000, load ratios 0.79-1.07 (every interval passes; no waiver).

### P3 — accessor API (mechanical, no behaviour change)

Every direct access to `Value::value`, `Value::missing`, `Value::svalue` goes
through `as_double()`, `is_missing()`, `text()`, and two adapters are added:
`template <class T> T unpack(const Value&, TypeId static_type)` (old storage →
the P1 scalar types, using the type the binder fixed) and
`Value pack(const types::Value&)`. Files (35, from grep; review S15):
`anomalydetectionmodel/anomalydetectionevaluator.h`,
`associationmodel/associationmodel.h`, `baselinemodel/baselinemodel.h`,
`clusteringmodel/clusteringmodel.h`, `core/builtinfunctions.h`,
`core/datafield.h`, `core/derivedfield.h`, `core/fastpredicate.h`,
`core/internal_model.h`, `core/miningfield.h`, `core/predicate.h`,
`core/sample.h`, `core/target.h`, `core/value.h`,
`ensemblemodel/multiplemodelmethod.h`, `expression/apply.h`,
`expression/discretize.h`, `expression/fieldref.h`, `expression/mapvalues.h`,
`expression/normcontinuous.h`, `expression/normdiscrete.h`,
`expression/textindex.h`, `gaussianprocessmodel/gaussianprocessmodel.h`,
`generalregressionmodel/generalregressionmodel.h`, `knnmodel/knnmodel.h`,
`math/distance.h`, `naivebayesmodel/naivebayesmodel.h`,
`neuralnetwork/neuralnetworkmodel.h`, `output/residual.h`,
`output/transformedvalue.h`, `regressionmodel/categoricalpredictor.h`,
`regressionmodel/numericpredictor.h`, `regressionmodel/predictorterm.h`,
`regressionmodel/regressiontable.h`, `svmmodel/svmmodel.h`. May be split by
directory into several commits. Green: accessors inline to the same fields;
dump diff empty.

### P4 — function table and Apply wrapper, over the P3 accessors (old storage)

Kernels take P1 scalar types; `ApplyNode` unpacks its arguments with
`unpack<T>` and packs the result with `pack`. Comparison kernels delegate to a
`LegacyCompare` that calls today's `Value` operators, so string and float
comparison semantics do not change here (they change in P6).

| commit | files | closes | why green alone |
|---|---|---|---|
| P4.1 table, kernels, `ApplyNode` with today's semantics | new `src/fn/{table.h,kernels.h,apply_node.h}`; `expression/apply.h`, `expression/expressionbuilder.h`; delete `BuiltInFunction` (`core/builtinfunctions.h`) | — | kernels reproduce today's results including today's missing/invalid handling (behind per-rule switches the following commits remove); dump diff empty; gate on Housing (151 Applies) |
| P4.2 Apply output table | `fn/apply_node.h` | B22 | one rule |
| P4.3 `if` with a missing condition | `fn/kernels.h` | B20 | one rule |
| P4.4 division by zero, INF to integer | `fn/kernels.h` | B21, D20 | one rule |
| P4.5 int64 kernels, overflow invalid | `fn/kernels.h` | D4 | one rule |
| P4.6 `concat` casts | `fn/kernels.h` | B23 | one function |
| P4.7 Cast nodes on DerivedField, OutputField and the target | `core/derivedfield.h`, `output/outputfield.h`, `core/internal_model.h` (target cast) | B17, B24 | Q2 decides the failure mode |
| P4.8 `and`, `or` | `fn/table.h`, `fn/kernels.h` | — | new functions only |
| P4.9 `formatNumber` | `fn/kernels.h` (ported from the concat_formatnumber attempt with its verifier's float-narrowing fix) | — | new function only |

### P5 — representation swap (behaviour-preserving)

One commit, now small: `core/value.h` becomes `types::Value`; the document
`StringPool` replaces `Indexer::get_string_value`; `unpack`/`pack` become
`std::get`/identity; inputs are parsed into typed values; predicates become
`BoundPredicate` arrays (§6.4). Files: `core/value.h`, `core/indexer.h`
(string map removed), `core/internal_evaluator.h` and `core/internal_model.h`
(`ScopedIndexer` removed), `core/sample.h`, `core/miningfield.h`,
`core/datafield.h`, `core/predicate.h`, `core/fastpredicate.h`,
`core/predicatebuilder.h`, `treemodel/node.h`, and the bodies of the P3
adapters.

Two compatibility switches keep today's semantics exactly, so the dump diff is
empty: (1) `float` fields are stored as `Double` (today's behaviour); (2)
`LegacyCompare` reproduces code equality for strings (pooled strings by
pointer; every unpooled string carries the shared "unseen" id; a string
function's result carries its argument's id). Each switch is deleted by one P6
commit. Deletes: `Indexer::get_string_value`, `string_value_map`, `freeze`;
`Value::active_indexer`, `ScopedIndexer` and its uses; `Value::infer_value`,
`Value::to_double`, `Value::ValueHash`, `createValues`,
`createUnorderedValues`; the `Value` operator set; `BuiltInFunction::typed`
leftovers. Green: dump diff empty; gate on all models (the riskiest commit
for performance, now without semantic changes mixed in).

### P6 — semantic fixes on the new representation, one per commit

| commit | change | closes | fixtures registered |
|---|---|---|---|
| P6.1 | strings compare by text (delete switch 2) | B1, B2 | `fix_string_unseen_equal`, `fix_string_builtin_recode`, probes `unseen_eq`, `up_eq` |
| P6.2 | float is a float (delete switch 1) | B4, B5 (inputs and casts) | `fix_float_equal`, `fix_float_set_narrow`, `fix_output_double_float_target`, `vp/p6`, `vp/p8`, `vx/probe/pa,pb,pd` |
| P6.3 | symmetric typed comparisons (delete `LegacyCompare`) | B18, D1, D2 | `fix_string_number_equal`, `fix_typed_equal_mixed` (re-derived cells) |
| P6.4 | typed missing markers | B6 | new marker fixture |
| P6.5 | binary measures over strings | B3 | `fix_binsim_*` (re-derived cells) |
| P6.6 | trailing blanks of categorical inputs | — | new fixture |
| P6.7 | remaining fixtures of the failed patches | — | `fix_string_derived_set`, `fix_string_typed_values`, `fix_typed_derived_conversions`, `fix_string_sum_avg`, `fix_derived_cast_novalue`, `fix_output_integer_nonintegral` |
| P6.8 | inputs parsed by the XSD lexical rules: `""` is the empty string or invalid, `NaN` is a float/double value, a string never reaches a numeric slot as a code | B37, B38, B39 | fuzzer v0 cases as fixtures |

Each touches the comparison/cast/marker code it names and nothing else; each
is green because its dump diff is exactly its listed cells.

### P7 — frames and scopes

| commit | files | closes | why green alone |
|---|---|---|---|
| P7a.1 `Frame` = renamed `Sample` | the 48 files that name `Sample` (mechanical rename, `Feature` removed) | — | rename; dump diff empty |
| P7a.2 slots from the binder | the files that use `Indexer` for name→index (56 mention it), via a thin `SlotResolver` with the Indexer's interface, then its uses removed directory by directory; `Indexer` deleted at the end | — | the binder's flat layout reproduces today's numbering semantics (one scope); dump diff empty |
| P7b.1 child scopes for LocalTransformations names | `bind/scope.h`, `core/internal_model.h`, `core/transformationdictionary.h` | B16 | sibling-name fixture |
| P7b.2 child MiningFields: aliases and steps | `core/miningschema.h`, `core/miningfield.h`, `ensemblemodel/segment.h`, `svmmodel/svmmodel.h` (delete `validate`); delete `in_segment_scope`, `segment_dag`, `scope_under`, the Sample copies | B26 (re-validation, SVM asIs) | 90847c1 probes as fixtures |
| P7b.3 TD instantiation and sharing (§3.2) | `bind/binder.h`, `core/dagbuilder.h` (deleted, with its silent `removed` set), `core/internal_model.h` | B14 | outer-LT/TD segment fixture; gate on GBT/RF (sharing must keep TD fields once per record) |
| P7b.4 chain output slots in the parent scope | `ensemblemodel/multiplemodelmethod.h` (`model_chain`), `core/internal_model.h` (delete `ChainScope`, `writes`, `augment` write-back) | B15 | `vp/p5`, `fix_output_integer_chain` |
| P7b.5 MiningField step rewrite | `core/miningfield.h`, `core/miningschema.h` | B31, N3 (widening), validity flag; `isValid`/`isNotValid` | new fixtures |
| P7c MULTITHREADING layout | `core/frame.h` (byte state, padded scope ranges), `ensemblemodel/multiplemodelmethod.h` (parent pre-evaluates shared instances); CI TSan+OpenMP job | — | TSan job green; OpenMP gate |
| P7d lazy DerivedFields | `core/frame.h`, `expression/fieldref.h`, `core/derivedfield.h` | Q11 | only after P7c; dump diff limited to records whose only failure was an unread field (listed) |

### P8 — one pipeline, one model per commit

| commit | change | why green alone |
|---|---|---|
| P8.0 | `InternalModel::evaluate`, `RawScore`, `ModelResult` (= `InternalScore`, with `owner` and lazy `arena`, S1), default `raw_score()` adapting each model's existing `score_raw`/`predict_raw` output (text parsed in the target type); `api/prediction.cc` reads the typed result; Result-outlives-Model test | every model still runs its old code through the adapter; dump diff empty |
| P8.1–P8.13 | one leaf model per commit: Tree, Regression, GeneralRegression, NeuralNetwork, SVM, NearestNeighbor, Clustering, NaiveBayes, Scorecard, RuleSet, Baseline, GaussianProcess, Association (+Text) — each implements `raw_score` natively and deletes its `predict_raw` and old `score_raw` | one model's code; its fixtures and its gate benchmark |
| P8.14 | MiningModel: `MultipleModelMethod` over `ModelResult` (delete the string paths `predict(segment)`, `is_missing(std::string)`, `values_of`; `Segment::score/result/predict`) | B28, B29; all leaves already native |
| P8.15 | `@segmentId` OutputFields | B34 |
| P8.16 | AnomalyDetectionModel as an `InternalModel` (delete `apply_mining_schema`) | B26 (write-back) |
| P8.17 | typed Targets and display only at the API; delete the adapter, `predict`, the string `finish`, `present`, `label_text`, `Shown`, `augment_last`, `displayed`, `InternalScore`'s string fields, `PredictedValue::eval_double`'s class-text parsing, `Category::same` | B27 (internal), B30, B5 (outputs) |

### P9 — public API decisions

Q5 (`as_string` and displayValue), Q9 (`Model::diagnostics()`), README and
CHANGELOG, version 2.2 or 3.0 depending on Q5.

### P10 — temporal types (optional)

Add the temporal alternatives (the compiler lists every visitor to extend),
XSD `date`/`time`/`dateTime` lexical forms, the `*Since` types, their casts
(DataDictionary.html: "These type casts are analogous to, e.g., casting an
integer to a double") and comparisons, `formatDatetime`, `dateDaysSinceYear`,
`dateSecondsSinceYear`, `dateSecondsSinceMidnight`. Closes B32.

### P11 — the fuzzer as a gate

Quick fuzz (fixed seed) in ctest under label `fuzz`; nightly long runs;
coverage feedback, the shrinker and the function-table JSON export (§8);
`departures.json` published as an appendix of the README.

### P-fixtures — reusing the failed patches' fixtures

`fix_grm_factor_level` is already committed (315a815). "Re-derive" means the
expected values encode a rule this design decides differently; the affected
cells are recomputed from the spec and the change is stated in the header.

| fixture | from | commit | re-derive? |
|---|---|---|---|
| fix_string_unseen_equal | both | P6.1 | no |
| fix_string_builtin_recode | type_value_tag | P6.1 | no |
| fix_float_equal, fix_float_set_narrow | typed_values / type_value_tag | P6.2 | fix_float_equal: yes for `e_isin` (untyped Constants are double, D1: rows 1–2 true → false); fix_float_set_narrow: no |
| fix_output_double_float_target | both | P6.2 | no |
| fix_string_number_equal | both | P6.3 | no (it has no cell with the number on the left) |
| fix_typed_equal_mixed | typed_values | P6.3 | yes: float vs double compares in double (`e_f_c01` false), and `e_d_s3` (rows 1–2) and `e_i_s30` (rows 1, 2, 4) under D2; `e_b_ct`, `e_b_c1`, `e_s_b` mix boolean with another type, a load error under §2.7/Q7, so they move to a separate load-error check |
| fix_binsim_clustering_string, _declared, fix_binsim_knn_string | type_value_tag | P6.5 | yes: strings `1`/`0` are XSD boolean literals (D15) |
| fix_string_derived_set, fix_string_typed_values, fix_typed_derived_conversions, fix_string_sum_avg | both | P6.7 | fix_typed_derived_conversions: yes for `e_d_s3` (D2: row 1 true → false); the others no (checked against Q1) |
| fix_derived_cast_novalue, fix_output_integer_nonintegral | both | P6.7 (after P4.7) | yes, Q2 is "record invalid": fix_derived_cast_novalue rows 3–6, fix_output_integer_nonintegral rows 4–6 and 8 become invalid records; P4.7 adds an invalid-record marker to `pmml_tester` |
| fix_output_integer_chain | both | P7b.4 | yes (Q2): rows 2 and 5 become invalid records |
| fix_builtin_text_if | concat_formatnumber | P4.3 | no |
| fix_builtin_text_concat | concat_formatnumber | P4.6 | yes if Q1 is Java text (it uses XQuery casts); add the spec's `month-year` row |
| fix_builtin_text_formatnumber | concat_formatnumber | P4.9 | add the float-narrowing column |

fix_float_set_narrow and fix_string_sum_avg have an empty `<Header/>`; the
commit that registers them writes a header quoting the spec.

The verifiers' probe models become fixtures in the commit that closes them:
`vprobe2/q3,q4` (P2.2), `vp/p2` (P2.1), `vp/p6,p8` and `vx/probe/pa,pb,pd`
(P6.2), `vp/p5` (P7b.4), this document's `design/probe/` (§1.3) and the
review's `tree_int.pmml`, `b.pmml`, `d.pmml`, `e.pmml` (P2.2, P4.7, P4.4).

---

## 8. The differential fuzzer

### 8.1 Goal

Find disagreements between cPMML and JPMML on small documents, classify each
as a cPMML bug, a listed departure, or a formatting/rounding difference, and
turn every unexplained one into a fixture. It searches function × argument
types × edge values × context, the space where the hand probes kept finding
defects one at a time.

### 8.2 Components

| file | role | phase |
|---|---|---|
| `tools/fuzz/gen.py` | seeded generator of cases (stdlib Python) | P0 |
| `tools/fuzz/JpmmlOracle.java` | loads many PMML files in one JVM (`javac -cp tools/pmml-evaluator-...jar`) and scores each case's rows, writing per-row values or the exception class and message | P0 |
| `test/fuzz_runner.cc` | the same for cPMML, in one process | P0 |
| `tools/pmmltext.py` | `same_number`, `classify`, CSV helpers moved out of `tools/cross-validate-text.py`, which imports them | P0 |
| `tools/fuzz/run.py` | drives both runners over a corpus; a case on which `fuzz_runner` crashes is recorded as `CRASH` and the run resumes | P0 |
| `tools/fuzz/compare.py` | cell classification and departure matching | P0 |
| `tools/fuzz/departures.json` | the known departures (§8.5) | P0 |
| coverage feedback, `tools/fuzz/shrink.py`, `fuzz_runner --dump-function-table` | biasing toward unhit (function, types) pairs; delta debugging to a minimal fixture; the generator reading `kBuiltins` | P11 (review O2) |

### 8.3 Cases

A case is one document with one expression under test, so an error isolates
to one expression (JPMML aborts the record on any error). The generator is
type directed: `gen(type, depth)` picks a production that yields `type`:

- leaves: `FieldRef` to a field of that type; `Constant` typed, or untyped with
  edge text (`7`, `007`, `-0`, `1e10`, `3.14E-5`, `NaN`, `inf`, empty, `abc`,
  ` a`), and untyped Constants in parameters that declare a type (S2);
- `Apply` of a function whose result rule can give `type` (a hand-written list
  in P0; the exported table from P11), with optional `mapMissingTo`,
  `defaultValue`, `invalidValueTreatment`;
- `NormContinuous`, `NormDiscrete`, `Discretize`, `MapValues` (with and without
  `dataType`);
- 10% ill-typed productions to exercise load errors.

Contexts: (1) a TransformationDictionary DerivedField of a random declared
dataType read by a string and a typed `transformedValue` OutputField;
(2) LocalTransformations, including a forward reference; (3) a TreeModel whose
SimplePredicates compare the field with typed and decimal-on-integer values
(K1); (4) segment templates: selectFirst and modelChain with the field in the
outer LocalTransformations or TD read through child MiningSchemas, an outer
OutputField reading a chain output, and N segments sharing a TD field (B14,
B15, B16, S8); (5) DataDictionary Values/Intervals/missing markers and
MiningFields with random treatments.

Inputs: eight rows from per-type edge lists — integer `0, -1, 2147483648,
9007199254740993, 1.0, ''`; float `0.1, 3.4e39, 1e-50, -0, NaN, INF`; double
likewise; string `'', ' a', 'a ', a declared Value, an unseen text, '3',
'3.0', 'true'`; boolean `true, false, 1, 0, TRUE` — plus N/A.

### 8.4 Comparison

Per cell: identical text → pass; `FORMAT` and `ROUNDING` as in
`cross-validate-text.py`; otherwise `VALUE`, `ERROR_CPMML_ONLY`,
`ERROR_JPMML_ONLY`, `ERROR_BOTH` (pass), and at load `LOAD_CPMML_ONLY`,
`LOAD_JPMML_ONLY`, `LOAD_BOTH` (pass: both sides refuse the document at
load, nothing else) and `LOAD_CPMML_ROWS_JPMML` (a finding: cPMML refuses at
load, JPMML loads and fails every row; grouped by JPMML's error), and
`CRASH`. A non-pass cell is matched against the
departure rules; a match is reported as expected with the rule id; anything
else is a finding, written (by hand in P0–P10, by the shrinker from P11) as
`test/data/pending/model/fuzz_<case-id>.zip` + `test/data/pending/dataset/fuzz_<case-id>.csv` with a header annotation (the layout `tools/cross-validate-text.py` and `fixture-dump-diff.sh` read with `--fixtures test/data/pending`). In v0 the departure rules key on the case's tags, not on the failing sub-expression, so they are approximate; the P11 shrinker narrows them.
Every case is reproducible from `--seed S --case N`. Budgets: quick mode 500
cases with a fixed seed (~30 s, ctest label `fuzz` in P11); nightly 50 000.

### 8.5 Known departures

Departures from JPMML that follow the spec, and departures from the spec
(marked **spec**), each with its reason.

| id | cPMML (this design) | JPMML 1.7.7 | basis / reason |
|---|---|---|---|
| D1 | numeric mixed-type comparisons in the promoted type, either order | converts to the left operand's type | Functions.html least-restrictive precedence |
| D2 | string × number equality compared as text, either order | parses the string when the number is left | same |
| D3 | `if` with a missing condition → missing (then mapMissingTo/defaultValue) | MissingArgumentException | "any missing inputs to a built-in function will result in a missing value" |
| D4 | integers are int64; overflow invalid | 32-bit; refuses `1e10` | XSD integer is unbounded; int64 is the representable subset (Q10) |
| D5 | `rint`, `expm1`, `ln1p`, `hypot` return double | `rint(3)` is `3` | their pseudo-declarations: `dataType="double"` |
| D6 | an aggregate's type comes from declared argument types even when one is skipped (`max(int, double missing)` is `-7.0`) | `-7` | "the least restrictive dataType will be inherited" |
| D7 | `if` has the common type of THEN and ELSE | type of the branch taken | static typing; spec silent |
| D8 | pow of integers with a negative exponent is invalid and follows invalidValueTreatment | throws regardless | Apply output table |
| D9 | ±INF against the same INF → missing; **NaN in comparisons → missing (extrapolation)** | Java `Double.compare`: `equal(INF, INF)` and `equal(NaN, NaN)` are true (measured in P0) | Transformations.html / BuiltinFunctions.html for INF; for NaN the spec says only that arithmetic with NaN is NaN — chosen so NaN never silently compares false |
| D10 | untyped Constant `NaN`/`INF`/`-INF` is a string | a number: NaN, Infinity, -Infinity (measured in P0) | Transformations.html Constant |
| D11 | **spec**: inputs, attributes and Array members use the strict, case-sensitive XSD lexical forms (`TRUE`, `inf`, `1d` invalid); only Constant text accepts INF/NaN in any case | lenient | DataDictionary.html → XML Schema Part 2; the case-insensitivity sentence is about Constants (§2.6) |
| D12 | forward-referenced OutputFields and DerivedFields accepted, ordered by dependency | refuses forward OutputField references | leniency on invalid PMML (Q4) |
| D13 | a FieldRef to a missing predictedValue applies mapMissingTo | fails the record | FieldRef: "The optional attribute mapMissingTo may be used to map a missing result" |
| D14 | number → boolean only for 0 and 1 | the same rule: accepts integer 0/1 and double 0.0/1.0, refuses 2, NaN, INF (measured in P0; not a departure, kept for the record) | XSD boolean lexical space |
| D15 | binary measures read strings `1`/`0` as booleans | only `true`/`false` | XSD boolean lexical space |
| D16 | `median` implemented | no `median` | BuiltinFunctions.html lists it |
| D17 | (Q1) number → string text inside expressions is Java text, including in `concat` | Java text | **spec**: `concat` points to XQuery `fn:concat`, whose casts write a whole double as `2`; one text rule for the whole library is preferred (Q1) |
| D18 | input `-0` stays `-0.0` | `0.0` (measured in P0) | XSD double has −0 |
| D19 | ordering over a categorical or Values-less ordinal string operand evaluates with a warning | refuses categorical | Q6 |
| D20 | **spec**: integer `x/0` (x ≠ 0) is invalid, not INF | invalid (N/A under asMissing) | the spec's "INF will be returned" meets "INF and -INF can only be floating point numbers"; XQuery raises `err:FOAR0001` for integer division by zero |
| D21 | Discretize/MapValues without `dataType` take the nearest enclosing declared type | same | spec and XSD silent; JPMML and today's cPMML |
| D22 | **spec**: integer-field document text with a decimal point (`33.0`) is a double and the field is widened to compare; a replacement or clamp bound that is not an integer widens the slot | scores | Functions.html implicit casts only toward less restriction (§3.4) |
| D23 | an arithmetic result NaN is NaN | fails the record (`Function "+" returned invalid value`) | Transformations.html: "All arithmetic expressions involving NaN will evaluate to NaN" (`departures.json` P0-NaN-result) |
| D24 | a missing argument of any built-in gives missing (then mapMissingTo/defaultValue), except where the spec says otherwise; D3 is the case of `if` | `+` and `-` give missing; `abs`, `equal`, `threshold`, `floor`, `ceil`, `round`, `sqrt`, `uppercase`, `substring`, `replace`, … throw MissingArgumentException | BuiltinFunctions.html: "Except as noted below, any missing inputs to a built-in function will result in a missing value being returned" (P0-missing-argument) |
| D25 | an invalid value in any MiningField with returnInvalid makes the record invalid, read or not | checks only fields the model reads | MiningSchema.html returnInvalid: "when an invalid input is encountered, the model should return a value indicating an invalid result" (P0-unread-invalid-input) |
| D26 | a `NaN` key in a hashed table (MapValues InlineTable, large SimpleSetPredicate) is a load error (§2.7) | loads; the row never matches | spec silent; a NaN key can match nothing, so the row is dead text and most likely an export error. Until P2.2 cPMML loads it like JPMML (B35) |
| D27 | `+`, `-`, `*`, `/` over infinities follow Transformations.html's "Brief Summary of Operations Involving Infinite Numbers": `INF + -INF`, `INF - INF`, `INF * 0` and `INF / INF` of one sign give missing; `INF / -INF` gives the numerator | fails the record (`Function "+" returned invalid value`; measured in P1.1) | the spec's table; IEEE would give NaN (decided 2026-11-18) |
| D28 | **spec**: NumericPredictor/GRM `exponent` is `INT-NUMBER` (xs:integer): `2.0`, `2e0`, `2.5` are load errors; `+2`, `02`, ` 2 ` read as 2 | loads non-integer text and scores x^1 | pmml-4-4.xsd `exponent` type="INT-NUMBER" (decided 2026-11-18) |
| D29 | **spec**: NormDiscrete `value` is read in its field's type by K1/D22; `3.5` on an integer field loads, the comparison widens, and the unit is 0 for every integer input | refuses the document (NumberFormatException) | Transformations.html NormDiscrete "has value 1.0 if the value of input field f is v, otherwise it is 0"; xsd `value` is xs:string (decided 2026-11-18) |
| D30 | **spec**: typed float and double text reads `+INF` as +∞ wherever `INF` is read: Constant text (any case, as S3 reads `INF`: `+inf`, `+Inf`), float/double attributes (xs:double `NUMBER`, e.g. Interval margins, coefficients), document Value text read in a float/double type (MapValues and table cells, Array members), and input values of float/double fields (they share `lexical::parse_double`); case-sensitive outside Constants (`+inf` stays malformed, D11); no `+NaN`, no `+INF` for integer; an untyped `+INF` is a string (D10) | refuses it everywhere: NumberFormatException `For input string: "+INF"` for a typed Constant or a MapValues key, "Required attribute LinearNorm@orig is not defined" for `orig="+INF"`, `Field "d" cannot accept invalid value "+INF"` for an input (probed 2026-11-19); it loads an Interval `rightMargin="+INF"` | XML Schema 1.1 Part 2 §3.3.4.2 (float) and §3.3.5.2 (double): "the ·literals· 'INF', '+INF', '-INF', and 'NaN'", `(\+|-)?([0-9]+(\.[0-9]*)?|\.[0-9]+)([Ee](\+|-)?[0-9]+)? |(\+|-)?INF|NaN`; the PMML Constant text links that version ("see https://www.w3.org/TR/2012/REC-xmlschema11-2-20120405/#float"); XSD 1.0 had no `+INF` (decided 2026-11-19) |
| D31 | an untyped Constant `1E400` (scientific notation beyond the double range) is the double Infinity, `-1E400` -Infinity, as committed in P2.3 | reads it as a string: `d + 1E400` fails, "No PMML data type for the intersection of PMML data types double and string" (probed 2026-11-19; a double OutputField over the Constant alone casts the string to Infinity) | Transformations.html, Constant: "A scientific notation expression like 3.14e+5 or like 3.14E-5 should also be treated as a double"; the out-of-range decimal rounds to ±INF (XSD 1.1 double lexical mapping, as `lexical.h` already reads every double; decided 2026-11-19) |

---

## 9. Open questions

Decided 2026-11-17: every question below is accepted as recommended. The commit that needs the answer is in brackets.

**Q1. Text of a number cast to string inside expressions** (concat, a string
DerivedField or OutputField over a number, string × number equality) [P4.6].
Options: (a) Java text, as the API writes it (`2.0`, `1.0E7`); (b) XPath/XQuery
casting (`2`, `1.0E7`), which the spec points to for `concat` only; (c) XSD
canonical form (`2.0E0`). *Recommendation: (a)*, one number→text rule in the
whole library, equal to what users receive from the API and what JPMML writes;
the existing fixtures' string OutputFields already encode Java text
(ceb6cec). D17 records the `concat` departure.

**Q2. A narrowing cast that fails at runtime** — an integer DerivedField or
OutputField over `3.5`, a double DerivedField over MapValues text `abc`, and a
regression whose `integer` target receives a non-integral prediction without
`castInteger` [P4.7]. Options: (a) load error whenever the static types narrow;
(b) an invalid result for the record, `InvalidValueException` naming the field
or target (JPMML: TypeCheckException, also for the target, review `b.pmml`);
(c) the value is missing. *Recommendation: (b)*. (a) rejects documents
producers commonly emit (a `float` DerivedField over double arithmetic; an
integer field over a product that is integral for real inputs); (c) silently
changes scores. The integer-target case changes today's output (cPMML returns
the double) and is listed in P4.7's message.

**Q3. A field referenced in a model but not in its MiningSchema** [P2.1/P7b].
*Recommendation: implicit pass-through with warning `W-SCOPE-IMPLICIT`, a
strict option available; switch the default to strict if the P1 baseline is
clean.*

**Q4. Forward references** [P2.1]. *Recommendation: accept with
`W-FORWARD-REF`, evaluate in dependency order, refuse cycles* — both cPMML and
JPMML score forward DerivedField references; JPMML refuses forward OutputField
references, which cPMML accepts on purpose (D12).

**Q5. What `Result::as_string()` returns for a classification whose Target
has displayValues** [P9]. *Recommendation: the raw class plus a new
`Result::display_string()`, in 3.0*; until then today's behaviour, computed at
the API boundary from the typed result.

**Q6. Ordering over operands whose optype or declaration does not define an
order** [P2.7]. Revised per the review: string × number ordering is a **load
error** (§2.7), not a warning. What remains open is ordering of two values of
one type where the spec says the field is unordered: a categorical operand
("Categorical field values can only be tested for equality"), and an ordinal
**string** field without Values, which the spec requires to "always specify
the order explicitly". *Recommendation: warnings*
(`W-ORDER-CATEGORICAL`, `W-ORDINAL-STRING-NO-VALUES`), ordering by code units
for strings and by value for numbers, because producers set optype loosely
(integer-coded categoricals compared with `lessOrEqual` are common in tree
exports). Ordinal non-string fields without Values are not in question: "the
order is always assumed to be the ascending order of the representation
type".

**Q7. Boolean mixed with numbers** (`equal(b, 1)`, `+(b, 1)`) [P2]. Functions.html:
boolean "can not be mixed with other types". *Recommendation: load error*,
unless the P1 baseline shows fixtures that rely on it; then an explicit cast
through the XSD lexical space `{1,0}` with a warning.

**Q8. Temporal types** [P10]. *Recommendation: add the alternatives and
functions in P10, after the core (review O1); accept both an XSD date and an
integer day count for the `*Since` types, pinned against JPMML in that phase.*
Until then, unchanged.

**Q9. Diagnostics API** [P9]. *Recommendation: add
`std::vector<cpmml::Diagnostic> Model::diagnostics() const` with
`struct Diagnostic { std::string code; std::string message; unsigned line; }`*
declared in `cPMML.h` — structured, so callers can filter by code; additive
and ABI-safe; empty for a clean document; no logging from the library.

**Q10. Integer range** [P4.5]. *Recommendation: int64, overflow invalid* (D4).

**Q11. Lazy DerivedField evaluation** [P7d]. *Recommendation: lazy, but only
after P7b.3 (the sharing rule, S8) and P7c (byte-per-slot state, parent
pre-evaluation before forking, TSan job green, S9) have landed.* With lazy
evaluation a record whose only failing DerivedField is never read scores
instead of failing, and unreferenced TD fields cost nothing (FieldScope.html:
"not instantiated in that model element").

**Q12. The OpenMP segment loop** (new; out of this design's scope)
[follow-up]. The OpenMP build is 1.25–3× slower than the sequential one on
every ensemble benchmark but AuditRandomForest (§5.2). *Recommendation: open a
separate item to move the per-segment allocation out of the parallel branch
and choose the parallel threshold by measured work rather than segment count;
until then the gate compares OpenMP base vs OpenMP head only.*
**Resolved.** An ensemble that is not worth a parallel loop now runs the same
plain loop as the sequential build, with no OpenMP region and no allocation
per segment; the parallel loop keeps its results in one
`std::vector<std::optional<R>>` and rethrows the error of the first failing
segment in document order, as the sequential loop does; a nested ensemble
inside a parallel one runs sequentially (`omp_in_parallel()`). The decision
is taken once at load (`EnsembleModel::parallel_worth_it`) from an estimate
of one record's work: per segment 1 plus the mean leaf depth of a TreeModel,
the recursive sum of a nested MiningModel, or the element count of any other
model; parallel at 1500 units or more (`-DCPMML_PARALLEL_WORK=N` overrides
it; `0` forces every ensemble parallel). Calibration
(`benchmark/omp/forced.txt`): the parallel loop costs about 10 µs a record
with libomp and 4 threads, and was faster only on AuditRandomForest
(estimate 2439, 70 µs sequential) — slower on IrisRandomForest (1037, 8 µs),
HousingGBT* (400, 12 µs), HousingRF* (95–118, 4–9 µs) and the dectree
ensembles (22–32). Measured with this document's method, 31 interleaved
rounds of base/head × sequential/OpenMP (`benchmark/omp/ab.sh`,
`ab_stats.txt`): OpenMP head vs sequential head fires the gate on 0 of 23
models, geomean 0.977, AuditRandomForest 0.521; sequential head vs base 0 of
23, geomean 0.999; OpenMP head vs base 0.32–0.97 on every ensemble. Every
fixture dumps byte-identical under sequential, OpenMP and forced-parallel
builds. The OpenMP build can therefore join the base-vs-head gate (§6.4
item 6). The threshold is tuned on one machine (Apple M-series, Homebrew
libomp); libgomp on Linux should be re-measured when P7 records both builds.

**Q13. Text of negative zero** (new, from P1.1) [P5]. D18 keeps an input
`-0` as `-0.0`, whose Java text is `-0.0`; JPMML writes `0.0`, and the XSD
float/double description says there is "only one zero" in some readings and
distinguishes ±0 in others. *Decide before P5, when inputs are parsed into
typed values: keep −0 (D18) or normalise it to +0 on input.*

**Decided 2026-11-18 after P1.** D27 is accepted (spec over IEEE and
JPMML). An OutputField, LocalTransformations or DerivedField name equal to
a MiningField of the same model (AuditTree, IrisTree, IrisMultinomReg) is
reported as a **warning**, not a load error, when the binder becomes
authoritative in P2.1: FieldScope.html forbids it, but real exporters write
it and JPMML scores those documents. The P1 agents' doc notes (the
Unsupported severity and codes, the ColdPath friends, the comparison
Purpose, NaN probes, the §3.4 table rows) are accepted as written in the P1
section.

**Decided 2026-11-18 after P2.1.** `InternalEvaluator::load_scope` (the
load binder kept with each model, 30-320 KiB, pointing into the freed
document, to avoid a per-record slowdown from allocator reuse) is accepted
until P7a.2, which must remove it or make its scopes the engine's. P2.2's
departures from JPMML follow the spec: D28 (`exponent` is xs:integer) and
D29 (NormDiscrete `value` in its field's type). A Constant of number or
boolean type that lands in a string slot (a FieldRef or Apply
`mapMissingTo`/`defaultValue` in a string DerivedField) keeps its text as
well as its number: string readers get the text (§3.4 `Cast<string>`, what
base and JPMML give: `"5"`, `"true"`, `"1"`), numeric readers the number.
The first P2.2 commit, which stored only the number, was dropped from
local history and P2.2 is rebuilt.

---

## 10. Revision log

Revision 2 answers `design-review/review.md` (approve with changes).

| item | resolution |
|---|---|
| K1 strict "parse as T(field)" | §3.4 rule: T literal → T; else xs:double literal → Double and the comparison promotes; else load error (missing markers matched as raw text first). Slot widening for replacements and clamp bounds (with N3). Grounded in Functions.html's implicit-cast direction; verified by `design/k1/scan.py` over 333 fixtures and 23 benchmark models (15 occurrences in 7 documents, all accepted with today's answers); D22 |
| K2 big-bang phases | §7 reordered and split: function table and Apply (P4) over P3's accessors before the swap (P5); P5 behaviour-preserving behind two switches; each fix its own commit (P6.1–P6.7, P4.2–P4.9, P2.1–P2.7); P7 split into rename, slots, scopes (five commits), MT layout, lazy; P8 one model per commit through a default `raw_score` adapter; files and the reason each commit is green, backed by `tools/fixture-dump-diff.sh` |
| S1 Result lifetime, arena | `ModelResult` holds `shared_ptr<const ModelContext>` and `unique_ptr<RecordArena>` allocated on first use; API test with a Result outliving its Model (§5.1) |
| S2 Constant and ParameterField | untyped Constant arguments inherit the built-in parameter's declared type (§3.4, §4.1) |
| S3 INF/NaN case | case-insensitive in Constant text, strict elsewhere; D11 states the remaining departure (§2.6) |
| S4 integer x/0 | kept invalid; listed as D20 with the FOAR0001 basis; `idiv` reference corrected (§4.2) |
| S5 string vs number ordering | load error (§2.7, P2.7) |
| S6 ordinal strings without Values | non-string ordinals without Values are fine and removed from Q6; for **string** ordinals the spec's next sentence requires Values, so that case stays a warning in Q6 (see disagreements below) |
| S7 hash containers and INF/NaN | reflexive `KeyHash`/`KeyEqual`; NaN keys refused at load; non-finite probes decided before lookup with a load-time flag, so hashed and linear paths agree (§2.7, sketch) |
| S8 TD sharing | instance placed in the highest scope where all its input slots are defined; a TD field read by N segments computed once per record (§3.2) |
| S9 MULTITHREADING | byte-per-slot state, padded scope ranges, parent pre-evaluates shared instances before forking, per-thread arenas, TSan+libomp CI job; OpenMP path measured now and gated (§5.2, §6.4, Q12) |
| S10 one pool per document | §2.3; pooled-distinct reps are unequal without reading text (`string_equal`, sketch) |
| S11 benchmark gate | A/A study measured; per-model bootstrap CI on the ratio of medians (fail if lower bound > 1.03, rerun once), geomean gate, more rounds for noisy models, both modes and both builds, one benchmark per model family (§6.4) |
| S12 sketch measures bandwidth | §2.10 says so, cites the review's `tree.cc`, and names the gate as the evidence |
| S13 silent UNKNOWN | debug assertion in bound accessors (sketch updated); Debug ctest job from P5 (§6.4) |
| S14 grep cannot enforce | `ColdPath` tag with private constructor (§2.4, §6.4) |
| S15 file counts | P3 lists its 35 files; P7a names the 48 `Sample` and 56 `Indexer` files and how they are split |
| S16 DefineFunction | out of scope; where it would plug in (§3.4) |
| N1 NaN | D9 marks it as an extrapolation; the INF quote's bracketed insertion removed |
| N2 asValue | added to §2.8 |
| N3 asExtremeValues on integer | slot widening (§3.4, §2.8) |
| N4 Discretize/MapValues default | nearest enclosing declared type, with the JPMML probe; D21 |
| N5 integer target cast | §5.1 and Q2; B24 extended |
| N6 segmentId | B34, §5.2, P8.15 |
| N7 sibling OutputField duplicates | §3.2 |
| N8 lazy maps | `std::call_once` (§5.1, §5.5) |
| N9 predicate storage | `BoundPredicate` flat array with a switch (§6.4) |
| N10 "17 models" | 14 overrides plus the base virtual (§1.1) |
| O1 temporal alternatives | deferred to P10; variant has 7 alternatives (sketch builds both) |
| O2 fuzzer extras | coverage feedback, shrinker and table export deferred to P11 |
| O3 ModelEvaluator concept | optional (§5.4) |
| O4 Diagnostic fields | code, message, line; path and rule deferred (§3.6) |
| Open questions | Q2 includes the integer target; Q6 rewritten per S5/S6; Q9 structured entries; Q11 conditional on S8/S9; Q12 added |

**After P0 (2026-11-18).** The P0 agents' corrections: §5.2 and Q12 record
the OpenMP fix; §6.4 states one NOISY rule and how models with different
round counts are resampled jointly; §7 P0 rows list the files the commits
actually touch and the target-column rule; §8.2 adds `run.py`; §8.4 defines
`LOAD_BOTH`, `LOAD_CPMML_ROWS_JPMML` and the pending layout; §8.5 corrects
D9, D10, D14 and D18 from measurement and adds D23–D26; the P-fixtures table
lists the cells that need re-deriving; B33 is closed, and B35–B46 are the
fuzzer v0 findings not already in the table.

**Where this revision disagrees with the review.**

- *S6.* The review quotes "If no Value elements are given, the order is always
  assumed to be the ascending order of the representation type" and concludes
  that ordering an ordinal string without Values is not a violation. The same
  paragraph continues: "The notable exception to this rule are ordinal
  DataFields of type string: Ordinal DataFields of type string must always
  specify the order explicitly via respective Value elements". So for strings
  it is a violation; the design removes the non-string case from Q6 and keeps
  the string case as a warning.
- *S11, the geomean threshold.* The review suggests a flat ≤ +1% geomean
  gate. The measured A/A geomean moved 0.6% with no change, so a flat 1% would
  fire on noise too often; the design requires the geomean to exceed 1.01
  **and** its bootstrap lower bound to exceed 1.00.
- *S12.* Agreed that the streaming benchmark measures bandwidth; the review's
  `tree.cc` result (variant path faster) is likewise a micro-benchmark, so
  neither is cited as evidence of library performance.
