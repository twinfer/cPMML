# Staged fixtures (typed-core P0.5)

Twenty of these fixtures come from three attempts that were not merged:
`type_value_tag`, `typed_values` and `concat_formatnumber`. Each one encodes a
bug that the typed-core design (`docs/design/typed-core.md`, §7
"P-fixtures") fixes in a later commit. They are kept here so that work is
not lost. They are **not registered in ctest**. `test/CMakeLists.txt` globs
only `test/data/model/*.zip`, and nothing in the build or the tools searches
subdirectories, so `pending/` is never picked up.

Two more, `fix_mapmissing_string_slot_equal` and
`fix_mapmissing_string_slot_numeric`, are known gaps that design commit
P2.2 left for P4.7 (notes 8 and 9). `fix_segment_outer_localtrans_derived`
is bug B14, found by design item P2.8c and left for P7b.3 (note 10).
Four more were added by design item P3.4b (coverage fixtures for
ComplexPartialScore, the SVM alternate category and the GRM offset, base
28fbb83), each the rows on which cPMML disagrees with JPMML and the spec:
`fix_scorecard_complex_score_missing` (B49), `fix_scorecard_reasoncode_negative`
(B50), `fix_grm_offset_general_linear` (B51) and `fix_svm_binary_no_alternate`
(B48) (notes 11-14). Design item P4.3b registered
`fix_scorecard_reasoncode_negative` (B50 decided not a bug, D35) and added
`fix_grm_offset_multinomial` (B51, multinomialLogistic; note 15).

Layout: `model/<name>.zip` (one `<name>.pmml` inside) and
`dataset/<name>.csv`, the same layout as `test/data/`, with LF line endings.
The tools that take a fixture tree run on it directly, for example
`tools/cross-validate-text.py --fixtures test/data/pending` or
`tools/fixture-dump-diff.sh --fixtures test/data/pending`.

**Registering a fixture.** The design commit named below moves the fixture's
two files to `test/data/model/` and `test/data/dataset/` (`git mv`). If the
fixture is marked "re-derive", that commit also recomputes the cells listed
from the spec and says so in the PMML `Header`. The fixture must fail on that
commit's base, and the commit records the sentinel line. When the last
fixture has moved, delete this directory.

## Provenance

| source attempt | fixtures |
|---|---|
| `typed_values` | fix_string_unseen_equal, fix_float_equal, fix_output_double_float_target, fix_string_number_equal, fix_typed_equal_mixed, fix_string_derived_set, fix_string_typed_values, fix_typed_derived_conversions, fix_derived_cast_novalue, fix_output_integer_nonintegral, fix_output_integer_chain |
| `type_value_tag` | fix_string_builtin_recode, fix_float_set_narrow, fix_binsim_clustering_string, fix_binsim_clustering_string_declared, fix_binsim_knn_string, fix_string_sum_avg |
| `concat_formatnumber` | fix_builtin_text_if, fix_builtin_text_concat, fix_builtin_text_formatnumber |

Seven fixtures are in both `type_value_tag` and `typed_values`. The
`typed_values` copy is the one staged: it is the later one, and its headers
quote the spec. In six of the seven the two copies differ only in the
`Header`. The exception is `fix_string_derived_set`: the `type_value_tag`
copy had two more DerivedFields and three more score bits, which the
`typed_values` copy dropped:

```
<DerivedField name="rs" optype="categorical" dataType="string"><Apply function="round"><FieldRef field="d"/></Apply></DerivedField>
<DerivedField name="ps" optype="categorical" dataType="string"><Apply function="+"><FieldRef field="i"/><FieldRef field="i"/></Apply></DerivedField>
c9  512  SimpleSetPredicate rs isIn {"3" x}
c10 1024 SimpleSetPredicate rs isIn {"3.0" x}
c11 2048 SimpleSetPredicate ps isIn {"6" x}
(equal(ds, "3") was bit 4096 there, 512 here)
```

P6.7 may add them back. Their expected values would then come from the spec:
`round` returns an integer (BuiltinFunctions, quoted in design §4.2:
"Functions floor, ceil, and round return an integer"), and `+` of two integers is an integer (§2.5). Under Q1 their text
is Java text, so `rs` is `3` and `ps` is `6`.

## Fixtures

"Base" is commit 4467498 (this commit's parent). Its sentinel is the first
line `pmml_tester.exe` prints on that build. "JPMML" is JPMML 1.7.7 on the
fixture's inputs. Whole-file results are from `tools/cross-validate-text.py`.
For the fixtures JPMML refuses, each row was also scored on its own, because
JPMML gives up on the whole file at the first row it refuses.

| fixture | registered by | re-derive? | base (4467498) | JPMML 1.7.7 |
|---|---|---|---|---|
| fix_string_unseen_equal | P6.1 | no | FAIL `predicted: same  expected: diff` | all 8 rows agree |
| fix_string_builtin_recode | P6.1 | no | FAIL `predicted: other  expected: up` | all 10 rows agree |
| fix_float_equal | P6.2 | **yes: `e_isin`** (see note 1) | FAIL `predicted: 72.0  expected: 9.0` | all 6 rows agree |
| fix_float_set_narrow | P6.2 | no; **header to write** (note 2) | FAIL `predicted: 240.0  expected: 0.20000000149011612` (the `o_fpd` column) | all 7 rows agree |
| fix_output_double_float_target | P6.2 | no | FAIL `output 'tv_f': predicted: 1e+300  expected: Infinity` | all 7 rows agree |
| fix_string_number_equal | P6.3 | no (note 3) | FAIL `predicted: 0.0  expected: 15.0` | all 6 rows agree |
| fix_typed_equal_mixed | P6.3 | **yes, and restructure** (note 4) | FAIL `output 'e_s_i': predicted: false  expected: true` | all 4 rows agree |
| fix_binsim_clustering_string | P6.5 | **yes: row 6** (note 5) | FAIL `predicted:   expected: K2` | rows 1-4 agree; rows 5-7 refused (`Expected false or true, got "n"` / `"1"` / `"TRUE"`) |
| fix_binsim_clustering_string_declared | P6.5 | check (note 5) | **PASS** (a guard, not a sentinel) | every row refused (`Expected false or true, got "n"`) |
| fix_binsim_knn_string | P6.5 | **yes: every row** (note 5) | FAIL `predicted: 12.0  expected: no prediction` | every row refused (`Expected false or true, got "0"`) |
| fix_string_derived_set | **registered by P4.7** (planned P6.7; moved to `test/data/`, as are its files; note 16) | no (checked against Q1) | FAIL `predicted: 138.0  expected: 511.0` | all 7 rows agree |
| fix_string_typed_values | P6.7 | no (checked against Q1) | FAIL `predicted: 1.0  expected: 3.0` | all 4 rows agree |
| fix_typed_derived_conversions | P6.7 | **yes: `e_d_s3` row 1** (note 3) | FAIL `predicted: 1.0  expected: 3.0` | all 5 rows agree |
| fix_string_sum_avg | **registered by P4.7** (planned P6.7; moved to `test/data/`, as are its files; note 16) | no; **header written by P4.7** (note 2) | FAIL `predicted: 0.0  expected: 6` (the `o_sii` column) | all 7 rows agree |
| fix_derived_cast_novalue | **registered by P4.7** (planned P6.7 after P4.7; moved to `test/data/`; note 16) | **done: rows 3-6 are invalid records** (notes 6, 16) | FAIL `output 'o_tsd': predicted: 1  expected: 1.0` | rows 1-2 agree; rows 3-6 refused (TypeCheckException) |
| fix_output_integer_nonintegral | **registered by P4.7** (planned P6.7 after P4.7; moved to `test/data/`; note 16) | **done: rows 4-6, 8 are invalid records** (notes 6, 16) | FAIL `predicted: 3.5  expected: INVALID` (row 4; before P4.7's INVALID marker the pending row read `output 'ti': predicted: 3.5  expected: no value`) | rows 1-3 agree; rows 4-8 refused (TypeCheckException; row 7, 1e10, refused as 32-bit, D4) |
| fix_output_integer_chain | P7b.4 | **yes: rows 2, 5** (notes 6, 16) | FAIL `predicted: 103.5  expected: no prediction` | rows 1, 3, 4 agree; rows 2, 5 refused (TypeCheckException) |
| fix_mapmissing_string_slot_equal | **registered by P4.7** (moved to `test/data/`, as are its files) | no (note 8) | FAIL `output 'o': predicted: false  expected: true` (row 2) | both rows agree |
| fix_mapmissing_string_slot_numeric | **registered by P4.7** (moved to `test/data/`, as are its files; notes 9, 16) | **yes: `os3` row 2** (note 16) | FAIL `predicted: 200.0  expected: 255.0` (row 1) | both rows agree |
| fix_builtin_text_if | **registered by P2.8b4** (planned P4.3; moved to `test/data/`, as are its files) | no | FAIL `output 'if_full': predicted: E  expected: no value` (also the sentinel on P2.8b4's base a6eeee4) | rows 1, 2, 4, 5 agree; rows 3, 6 refused (`Function "if" cannot accept missing value`); spec-derived there (D3) |
| fix_builtin_text_concat | **registered by P4.6** (moved to `test/data/`, as are its files) | **done: every double and float cell re-derived** (note 7) | FAIL `output 'c_spec': predicted: -  expected: 2-2000` (since P2.3 the untyped Constant `-` keeps its text, B8; the integer month and year are dropped until P4.6, B23); with the re-derived cells, on P4.6's base 73cf6ea: the same line (row 1) | rows 1-5, 7-9 agree on every cell (the re-derived Java text); row 6 refused at evaluation (MissingFieldException: `Field "pv" is not defined`), spec-derived |
| fix_builtin_text_formatnumber | **registered by P4.9** (moved to `test/data/`, as are its files) | **done: the float-narrowing columns added** (`f_flt` %.10f, `f_flt_g` %.9g, `f_flt_d` %d of a float field `f`: 0.1 is `[0.1000000015]` and `[0.100000001]`, 16777217 is `[16777216]`), with `f_fix2` (%.2f), `f_hexf` (%a), `f_field` (the pattern read from a field `pat`, per record) and a row 10 (x = 1.005, a binary value below the decimal tie); every cell from an independent C program calling snprintf (the existing cells unchanged, which it reproduces) | FAIL `unsupported value "formatNumber" for Apply@function; ...` (a load error since P2.5, B13: the OutputField catch had hidden it, giving `output 'f_pct': model declares it but produced no value  expected: rate 0.5%`); the same line on P4.9's base 9bd050f, its list ending `, and, or` | scored column by column and row by row: every cell agrees but those the header lists (Java's Formatter: %d/%i of a double or a float, %#g and %.2f of an integer refused; %g keeps trailing zeros; half-up rounding of the decimal; %a `0x1.0p-1`; 32-bit two's complement for %x/%o of a negative; `0x0`; -0.0 read as 0.0, D18) and row 7 (a missing input, refused, D24): spec-derived throughout (POSIX printf) |
| fix_segment_outer_localtrans_derived | P7b.3 | no (note 10) | FAIL `predicted:   expected: 1553.0` (base 9123c4f, row 1) | all 3 rows agree |
| fix_scorecard_complex_score_missing | a fix item before P5, P8.9 at the latest (note 11) | no | FAIL `predicted: 29.0  expected: no prediction` (base 28fbb83, row 2) | row 1 agrees; rows 2-3 refused (UndefinedResultException `Undefined result`), N/A here |
| fix_scorecard_reasoncode_negative | **registered by P4.3b** (planned P8.9; B50 decided not a bug, D35; moved to `test/data/`, as are its files) | **yes: every reason-code cell, re-derived from Scorecard.html's routine** (note 12) | FAIL `output 'rc3': predicted: RC_INCOME  expected: no value` (base 28fbb83, row 1, the old JPMML cells); with the spec cells, PASS on P4.3b's base cc0f81e | row 3 agrees; rows 1, 2, 4, 5 differ (JPMML drops totals below 0, D35) |
| fix_grm_offset_general_linear | P8.3 (note 13) | no | FAIL `predicted: 1.9  expected: 0.9` (base 28fbb83, row 2) | all 7 rows agree |
| fix_grm_offset_multinomial | P8.3 (note 15) | no | FAIL `predicted: c  expected: a` (base cc0f81e, row 1: x = 0, off = -2, cPMML pa 0.17095 for 0.50648) | all 6 rows agree (within 1e-15) |
| fix_svm_binary_no_alternate | P8.5, after a decision (note 14) | **expectation is a load error** | loads and scores `pos`, `neg`, `pos` (`--dump`, base 28fbb83; no `LOAD-ERROR` cell); `pmml_tester` itself stops at `output column 'cls' not found in fixture CSV` | rows 1, 3 refused (MissingAttributeException `Required attribute SupportVectorMachine@alternateTargetCategory is not defined`); row 2 scores `neg` |

### Notes

1. **fix_float_equal, `e_isin`.** The cell is `isIn(ff, 0.1, 0.2)` with
   untyped Constants. Transformations, Constant: "A Constant that consists
   solely of numbers will be treated as an integer, if a decimal point is
   present among the numbers the Constant will be treated as a double." The
   operands are therefore float and double, and design D1 / §2.7 compares
   them in the promoted type, double. Functions: "When the input parameters
   have multiple dataTypes, the least restrictive dataType will be inherited
   by default ... string, double, float, integer". For f = 0.1 the float
   0.10000000149011612 is not the double 0.1, and for f = 0.2 the same holds
   against 0.2. Rows 1-2 change from `true` to `false`. The fixture has
   `true`, which is JPMML's left-operand-type rule. The predicate score bits
   (SimplePredicate and SimpleSetPredicate values read in the field's type,
   §3.4) do not change. The design's table says "no" for this fixture, and
   that is the design gap.
2. **fix_float_set_narrow, fix_string_sum_avg** have an empty `<Header/>`.
   The project rule asks for a header that states the behaviour and quotes
   the spec, so the registering commit writes one. For sum_avg, `avg` of
   integers is double and `avg` of floats is float (design `AtLeastDouble`,
   §4.1-4.2). For set_narrow, `f * i` is float (§2.5 table) and `f + d` is
   double. Both fixtures are Scorecards with no predictedValue OutputField,
   so `pmml_tester` checks their first OutputField. It prints the
   prediction in the `predicted:` part of its message, which is why the
   base sentinel shows `240.0` against `0.20000000149011612`.
3. **String x number equality (D2).** §2.7 compares the number as text, in
   either order, where JPMML parses the string when the number is on the
   left. In `fix_string_number_equal` the string is on the left in every
   comparison, so nothing changes. P6.3 may add number-on-the-left cells.
   `fix_typed_derived_conversions` has one such cell, `e_d_s3` =
   `equal(d, "3")`: d = 3 is the text `3.0`, which is not `3`, so row 1
   changes from `true` to `false`. The design's table says "no" for this
   fixture, and that is the design gap.
4. **fix_typed_equal_mixed.**
   - Under D1/D2, three columns change:
     - `e_d_s3` = `equal(d, "3")`: rows 1-2 change from `true` to `false`.
     - `e_i_s30` = `equal(i, "3.0")`: the text `3` is not `3.0`, so rows 1,
       2 and 4 change from `true` to `false`.
     - `e_f_c01` = `equal(f, 0.1)`: float and double compare in double, so
       rows 1-3 change from `true` to `false`.
   - `e_b_ct` = `equal(b, "true")`, `e_b_c1` = `equal(b, 1)` and `e_s_b` =
     `equal(s, b)` mix boolean with another type. Under §2.7 and Q7 that is a
     load error. Functions: "The various date, time and dateTime as well as
     boolean dataTypes can not be mixed with other types in a defined
     function without explicitly defining the expected output dataType".
     P6.3 must move these three OutputFields into a separate load-error
     check, or the whole document stops loading. This is a correction to
     the design's table: it lists only `e_f_c01`.
5. **Binary similarity over strings (D15).** XML Schema Part 2, 3.2.2.1:
   "An instance of a datatype that is defined as boolean can have the
   following legal literals {true, false, 1, 0}." JPMML accepts only
   `true`/`false`.
   - `fix_binsim_clustering_string` row 6 (`1,0,0`) becomes a scored row,
     with the same inputs as row 1 (K2). Row 5 (`n`) and row 7 (`TRUE`,
     not a literal under D11) stay without a prediction.
   - `fix_binsim_knn_string`: the training cells and the queries `0`/`1`
     are booleans, so every row gets a prediction computed from the spec
     (simpleMatching over the booleans).
   - `fix_binsim_clustering_string_declared` (`n`/`y`) does not change
     under D15 and already passes on base. It registers as a guard.
   - P6.5 also decides whether a string that is not a boolean literal, in a
     binary measure, gives "no prediction" (the cells as written) or an
     invalid record (§2.8). If it is an invalid record, these N/A cells
     need the same invalid-record expectation as note 6.
6. **Q2 = (b): a failed narrowing cast is an invalid record.** The
   fixtures write N/A for the field that has no value, because "JPMML 1.7.7
   does not score these records at all ... which a fixture cannot express".
   Under Q2 the record is invalid (`InvalidValueException` naming the field
   or target), so these rows need an invalid-record expectation:
   `fix_derived_cast_novalue` rows 3-6, `fix_output_integer_nonintegral`
   rows 4-6 and 8 (3.5, -2.5, 3.0000000001, INF), and
   `fix_output_integer_chain` rows 2 and 5. Row 7 of nonintegral (1e10)
   stays: it is an int64 (D4/Q10). `pmml_tester` has no marker for an
   invalid record today, so P4.7 must add one (for example a cell `ERROR`,
   or a header column), and the commit that registers these fixtures uses
   it.
7. **fix_builtin_text_concat.** The fixture writes every argument as the
   XQuery cast to `xs:string` (BuiltinFunctions, concat: "See XQuery
   fn:concat"). Q1 = (a) chose Java text for every number-to-string cast,
   including `concat`. That choice is departure D17 from the spec, recorded
   in the design. P4.6 therefore rewrites every double and float cell in
   Java text (`2.0`, `2000.0`, `1.0E-4`) and states D17 in the header. The
   integer and boolean cells are the same under both rules. The spec's own
   example, `c_spec` = concat(month, "-", year) giving `2-2000`, is already
   a column. JPMML 1.7.7 loads the model and scores rows 1-5 and 7-9; there
   it differs from the fixture only in number text, Java against XQuery
   (`c_k` `2000|2.0|0.1|true`, `c_x` `1.0E-4|` and `1234567.0|`, `chain`
   `1.0|0.75`, and `y`/`pred`/`pv`/`tv1` values such as `1.000000025E7`,
   `2.0E7`, `3.0E-7`). That Java text is what D17 asks for, so JPMML is a
   usable oracle for those 8 rows when P4.6 re-derives them (score row by
   row; the whole-file run stops at row 6). Row 6, every input missing, is
   refused at evaluation: the transformedValue OutputField `chain` reads
   the predictedValue OutputField `pv`, which has no value, and JPMML
   throws MissingFieldException (`Field "pv" is not defined`). Only row 6
   is spec-derived.
   **Done in P4.6:** every double and float cell (`y`, `pred`, `pv`, `tv1`
   and the concat columns over them, `c_f`, `c_k`, `c_ap`) re-derived by an
   independent Java program applying Double.toString, Float.toString,
   Long.toString and Boolean.toString to each argument; it agrees with JPMML
   1.7.7 on every cell of rows 1-5 and 7-9. Row 6's `c_k` is
   `2000|2.0|0.1|true`. The header states D17 and quotes the spec. The
   fixture moved to `test/data/`; `fix_concat_number_text` (P4.6) adds the
   types the fixture does not reach.
8. **fix_mapmissing_string_slot_equal** (added by design commit P2.2, not
   from the three attempts; its base is P2.2's parent, 529130a). A string
   DerivedField `x` holds `FieldRef b` with `mapMissingTo="true"`, and an
   OutputField asks `equal(x, "true")` (§2.7: string x string equality is
   text; §3.4 `DerivedField dataType=X -> Cast<X>(expression)`), which
   JPMML 1.7.7 gives `true` on both rows. Row 1 (b missing) holds the
   constant, which P2.2 stores as the string `true` keeping its number
   (decided 2026-11-18, design §9), so it is `true` on P2.2 and on the base
   build (and in the registered fixture
   `fix_mapmissing_string_slot_compare`). Row 2 (b = true, not missing)
   holds the boolean input, which no cast turns into the string `true`
   before P4.7, so `equal` compares the number 1 with the string's code and
   gives `false` on both builds. P4.7's Cast node makes `x` a string.
9. **fix_mapmissing_string_slot_numeric** (added by design commit P2.2,
   base 529130a; bug B38, "a numeric field computed over a string input
   uses the string's internal code", the string-code class of B1). String
   DerivedFields `s`..`s5` hold a FieldRef with a number or boolean
   `mapMissingTo` (`5`, `1e1`, `2.5` of an integer field, `07`, `0` of a
   boolean field); double DerivedFields `t` and `t3` copy `s` and `s3`
   (§3.4 `DerivedField dataType=X -> Cast<X>(expression)`) and are read by
   the RegressionTable's NumericPredictors (coefficients 1 and 100) and by
   the double OutputFields `ot` and `ot3`. Row 1 (every input missing):
   JPMML 1.7.7 gives `t` = 5 and `t3` = 2.5, so the prediction is
   5 + 100 * 2.5 = 255. P2.2 keeps the constant's number for OutputFields
   only (decided 2026-11-18, design §9): `ot` 5.0, `ot3` 2.5, `os2n` 10.0
   and `os4n` 7 are right on P2.2 and string codes on the base build (the
   4 cells of the P2.2 dump diff), but `t` and `t3` themselves still hold
   the codes, so the NumericPredictors read 0 and 2 and the prediction is
   200 on both builds. Row 2 (d = 3, i = 4, b = true) puts numbers and a
   boolean into the string slots, which no cast turns into text before
   P4.7, so the string OutputFields `os`..`os5` have no value on both
   builds (JPMML: `3.0`, `3.0`, `4`, `4`, `true`, Java number text, D17).
   P4.7's Cast nodes make `t` = Cast<double>(`s`) and `s` =
   Cast<string>(`d`); if P4.7's Cast<double> over the old storage still
   reads a string's code, the fixture moves with B38 to P6.8 ("a string
   never reaches a numeric slot as a code").
10. **fix_segment_outer_localtrans_derived** (added by design item P2.8c,
   base 9123c4f; bug B14). The MiningModel's LocalTransformations define
   `g` = round(x) (integer) and `gd` = x * 2 (double). Segment 1 lists `g` in
   its MiningSchema and its own LocalTransformations reads it: `h` = g / 2.
   Segment 2 does the same with `gd`: `k` = gd + 1. The prediction is the
   `sum` h + 100 * k. FieldScope.html allows this: "For sub-models, their
   MiningSchema can refer to any fields defined within the parent model's
   scope, including ... DerivedFields defined in the LocalTransformations of
   the parent model". JPMML 1.7.7 gives 1553, -1253 and 150. `h` is an
   integer division that truncates (`7 / 2` = 3, `-7 / 2` = -3, design
   §4.2). On base every row has no prediction. The cause is
   `core/dagbuilder.h`. The segment copies the parent's TransformationDictionary,
   so `g` is in it, and the segment builds its DerivedField DAG against its
   own MiningSchema. `g`'s input `x` is not a segment MiningField, so `g`
   goes into the silent `removed` set, and so does `h`, which reads `g`.
   `scope_under` then leaves `h` out of `segment_dag`, and `h` is never
   evaluated. The integer type and `round` play no part: a double `g` and a
   `g` = FieldRef x fail the same way. A NumericPredictor that reads `g`
   directly scores, and so does the same document with `x` added to each
   segment's MiningSchema (1553, -1253, 150 on base). That control rules out
   every cause except the DAG. P7b.3 deletes `dagbuilder.h`: a parent-scope
   field is an input of the segment and is not instantiated again. JPMML
   refuses a missing `x` (`Function "round" cannot accept missing value`),
   so the fixture has no missing row.
11. **fix_scorecard_complex_score_missing** (design item P3.4b, base
   28fbb83; bug B49). The model is `fix_scorecard_complex_partial_score`'s.
   Row 2 (age 45, income missing) takes the Attribute whose
   ComplexPartialScore is `income / 8`; row 3 (age and income missing)
   takes the `isMissing(age)` Attribute, `income / 10`. Scorecard.html:
   "Whenever element ComplexPartialScore is used, the actual partial score
   is the value returned by the EXPRESSION". That value is missing, and the
   page gives no replacement, so the score has no value. JPMML 1.7.7 throws
   UndefinedResultException on both rows, written N/A as
   `fix_scorecard_reasoncode` writes its own UndefinedResultException row.
   cPMML adds `as_double()` of the missing Value, 0, and scores 29.0. The
   fix belongs before P5, because P5 changes what a missing Value holds
   under `as_double()`, and a behaviour-preserving swap would have to keep
   this 0 on purpose.
12. **fix_scorecard_reasoncode_negative** (P3.4b, base 28fbb83; B50,
   registered by P4.3b, base cc0f81e). The model is
   `fix_scorecard_complex_partial_score`'s. P3.4b staged it with JPMML
   1.7.7's cells, which drop every reason code whose total points missed is
   below 0 (and keep a total of 0). Decided 2026-11-19: B50 is not a bug
   (D35). Scorecard.html's routine ranks every total: "For each unique
   reason code, R_1, R_2, ..., R_n, initialize points missed P_i = 0 ... For
   each scorecard characteristic ... compute point differential d_j ...
   pointsBelow d_j = baselineScore_j - partialScore_j; pointsAbove d_j =
   partialScore_j - baselineScore_j. Note that negative differences are
   possible and should be expected when the baselineMethod is other than min
   or max ... add d_j to that P_i. Rank the total points, P_1, ... P_n from
   largest to smallest, and return the corresponding reason codes", with
   ties to "the one associated with the Attribute or Characteristic that
   appears first in the PMML file", and baselineMethod "is purely
   informational". The example's "meaningless" RC3 is illustration, not the
   rule. P4.3b re-derived every reason-code cell with an independent Python
   implementation of that routine (not from cPMML) and added two tie rows:
   row 4 ties RC_AGE and RC_INCOME at -20 (RC_AGE first in the file), row 5
   ties RC_INCOME and RC_DEBT at -8 (RC_INCOME first in the file, against
   alphabetical order). Two points the routine leaves open are read as
   cPMML already does: the unique reason codes are those the scored
   Attributes cite (RC_YOUNG, uncited on rows 1 and 3-5, is not ranked at
   0), and a rank past the last code has no value (`fix_scorecard_reasoncode`
   expects the same; the example's "the second and third reason codes would
   be populated with the same code" is followed by neither evaluator). The
   fixture passes on cc0f81e: cPMML ranks with a stable sort in order of
   first citation (`scorecard/scorecardmodel.h:203`).
13. **fix_grm_offset_general_linear** (P3.4b, base 28fbb83; bug B51).
   `fix_grm_offset_variable` with `modelType="generalLinear"` and the
   identity link. GeneralRegression.html: offsetVariable "if present, this
   variable is used during scoring generalizedLinear, ordinalMultinomial, or
   multinomialLogistic models" (offsetValue has the same sentence).
   generalLinear is not one of them, and JPMML 1.7.7 does not read the
   offset there: y = 0.5 + 0.2x, also when the offset is missing (row 7).
   cPMML adds the offset for every model type (rows 2-5) and gives no
   prediction on row 7. P8.3 also decides whether such an offset is
   ignored silently or with a binder warning (project rule: no silent
   fallbacks).
14. **fix_svm_binary_no_alternate** (P3.4b, base 28fbb83; bug B48). One
   SupportVectorMachine, OneAgainstOne, `targetCategory="neg"` and no
   `alternateTargetCategory`. The target DataField declares `pos` and
   `neg`. SupportVectorMachine.html: alternateTargetCategory "is required
   in case of binary classification models with only one
   SupportVectorMachine element". cPMML loads the document and takes the
   first other Value of the target, `pos`, as the alternate category
   (`src/svmmodel/svmmodel.h:154`). JPMML 1.7.7 loads it too, but throws
   MissingAttributeException on every record that needs the alternate. The
   spec's answer is a load error. A fixture CSV cannot state one, so the
   dataset holds the three inputs only, and the registering commit adds a
   `test/strict_parse.cc` check instead of moving the files to
   `test/data/`. Refusing it makes a document that loads today stop
   loading, so the user decides first. The alternative is to keep the
   inference with a binder warning and a D-rule; then the dataset gains
   the `cls` column from the spec's threshold rule ("If maxWins is true
   and the prediction is larger than the threshold or maxWins is false and
   the prediction is smaller than the threshold, the class label is the
   targetCategory attribute, otherwise, it is the alternateTargetCategory
   attribute"; maxWins false, threshold 0, decision x1 - x2 = 3, -3, 0):
   `pos`, `neg`, `pos`. This is the
   only document that reaches `svmmodel.h:154`, so until then no
   registered fixture covers that line.
15. **fix_grm_offset_multinomial** (P4.3b, base cc0f81e; bug B51,
   multinomialLogistic). Categories a, b and the reference c ("the one from
   DataDictionary that does not appear in the ParamMatrix"), r_a = 0.5 + x,
   r_b = -0.5 + 0.3x, offsetVariable `off`. GeneralRegression.html lists
   multinomialLogistic among the models that use the offset, but its
   procedure has no offset term: "Set r_j = <x,beta_j> and s_j = exp r_j.
   The probability that our case falls into category j is then p_j = s_j /
   (s_1 + ... + s_k)". Decided 2026-11-19: the offset is added to every
   category's linear predictor, the reference's 0 included, so it cancels
   and the probabilities are those without it. The cells were computed by
   that formula in Python and JPMML 1.7.7 agrees on every cell (within
   1e-15). Row 2 (off = 0) is a control. cPMML adds the offset to the
   non-reference categories only: row 1 gives pa 0.17095 (0.50648), and
   rows 1, 4 and 6 change their predicted category (c, a, b for a, c, c).
   No missing-offset row: JPMML scores it (the offset is ignored), cPMML
   gives no prediction (`generalregressionmodel.h:182`), and the spec says
   nothing; P8.3 decides, with the generalLinear question of note 13.
16. **Design commit P4.7 (Cast nodes on DerivedField, OutputField and the
   target; base 2114e55)** registered six fixtures, each failing on its base
   (the first line `pmml_tester.exe` prints there) and passing with the
   casts: `fix_mapmissing_string_slot_equal` (`output 'o': predicted: false
   expected: true`, row 2), `fix_mapmissing_string_slot_numeric`
   (`predicted: 200.0  expected: 255.0`, row 1), `fix_derived_cast_novalue`
   (`output 'o_tsd': predicted: 1  expected: 1.0`, row 1),
   `fix_output_integer_nonintegral` (`predicted: 3.5  expected: INVALID`,
   row 4), `fix_string_sum_avg` (`predicted: 0.0  expected: 6`, row 1) and
   `fix_string_derived_set` (`predicted: 138.0  expected: 511.0`, row 1).
   The last two were planned for P6.7, but P4.7's Cast<string> is what
   fixes them, and a fixture must fail on the base of the commit that
   registers it. P4.7 added the invalid-record marker note 6 asks for: a
   cell `INVALID` in the predicted column (`pmml_tester.cc`) expects
   `cpmml::InvalidValueException` and checks nothing else in the row. Rows
   re-derived with it: `fix_derived_cast_novalue` rows 3-6 and
   `fix_output_integer_nonintegral` rows 4-6 and 8 (the headers say why,
   quoting XML Schema Part 2; JPMML 1.7.7 scores none of them). In
   `fix_mapmissing_string_slot_numeric`, `os3` on row 2 is `4.0`, not
   JPMML's `4`: `s3` is FieldRef i with mapMissingTo="2.5", which is not an
   integer literal, so by the design's K1 rule 2 the FieldRef's static type
   is the least restrictive of integer and double, double, and its value 4
   is cast to string as the double 4.0 (proposed D43; the header says so).
   `fix_string_sum_avg` got the header note 2 asks for. Still pending, and
   why: `fix_output_integer_chain` (P7b.4) fails on P4.7 with `predicted:
   103.5  expected: no prediction` (row 2), because a modelChain step writes
   its OutputFields for the next segments through
   `InternalModel::_write_outputs_to_sample`, which does not go through
   `OutputField::add_output` and so not through its cast; P7b.4 also has to
   write rows 2 and 5 as `INVALID` (Q2) rather than N/A.
   `fix_typed_derived_conversions` (P6.7) still fails on `e_d_s3` (note 3,
   D2: equal(d, "3") compares the double's text), although P4.7 gave its
   string DerivedFields their text (`o_ds` N/A -> 3.0 ...); and
   `fix_float_set_narrow` (P6.2) on the float narrowing of inputs (B4)
   (its `o_fpd` cells changed from N/A to the expected values).
