# Staged fixtures (typed-core P0.5)

These 20 fixtures come from three attempts that were not merged:
`type_value_tag`, `typed_values` and `concat_formatnumber`. Each one encodes a
bug that the typed-core design (`docs/design/typed-core.md`, §7
"P-fixtures") fixes in a later commit. They are kept here so that work is
not lost. They are **not registered in ctest**. `test/CMakeLists.txt` globs
only `test/data/model/*.zip`, and nothing in the build or the tools searches
subdirectories, so `pending/` is never picked up.

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
| fix_string_derived_set | P6.7 | no (checked against Q1) | FAIL `predicted: 138.0  expected: 511.0` | all 7 rows agree |
| fix_string_typed_values | P6.7 | no (checked against Q1) | FAIL `predicted: 1.0  expected: 3.0` | all 4 rows agree |
| fix_typed_derived_conversions | P6.7 | **yes: `e_d_s3` row 1** (note 3) | FAIL `predicted: 1.0  expected: 3.0` | all 5 rows agree |
| fix_string_sum_avg | P6.7 | no; **header to write** (note 2) | FAIL `predicted: 0.0  expected: 6` (the `o_sii` column) | all 7 rows agree |
| fix_derived_cast_novalue | P6.7 (after P4.7) | **yes: rows 3-6** (note 6) | FAIL `output 'o_tsd': predicted: 1  expected: 1.0` | rows 1-2 agree; rows 3-6 refused (TypeCheckException) |
| fix_output_integer_nonintegral | P6.7 (after P4.7) | **yes: rows 4-6, 8** (note 6) | FAIL `output 'ti': predicted: 3.5  expected: no value` | rows 1-3 agree; rows 4-8 refused (TypeCheckException; row 7, 1e10, refused as 32-bit, D4) |
| fix_output_integer_chain | P7b.4 | **yes: rows 2, 5** (note 6) | FAIL `predicted: 103.5  expected: no prediction` | rows 1, 3, 4 agree; rows 2, 5 refused (TypeCheckException) |
| fix_builtin_text_if | P4.3 | no | FAIL `output 'if_full': predicted: E  expected: no value` | rows 1, 2, 4, 5 agree; rows 3, 6 refused (`Function "if" cannot accept missing value`); spec-derived there (D3) |
| fix_builtin_text_concat | P4.6 | **yes: every double and float cell** (note 7) | FAIL `output 'c_spec': model declares it but produced no value  expected: 2-2000` | load error (`Field "pv" is not defined`): spec-derived throughout |
| fix_builtin_text_formatnumber | P4.9 | add the float-narrowing column (design) | FAIL `output 'f_pct': model declares it but produced no value  expected: rate 0.5%` | every row refused (`%i` pattern; missing input): spec-derived throughout (POSIX printf) |

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
   a column. The fixture also has a transformedValue OutputField (`chain`)
   that reads the predictedValue OutputField `pv`, which JPMML refuses at
   load (`Field "pv" is not defined`). Every value in the fixture is
   therefore spec-derived.
