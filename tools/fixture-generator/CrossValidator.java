package org.cpmml;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import java.util.zip.*;

import org.dmg.pmml.DataField;
import org.dmg.pmml.Field;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Interval;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.OpType;
import org.jpmml.evaluator.EvaluatorUtil;
import org.jpmml.evaluator.HasGroupFields;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.LoadingModelEvaluatorBuilder;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.evaluator.TargetField;

/**
 * CrossValidator — generates oracle fixtures from jpmml and validates cPMML against them.
 *
 * For each PMML file under --jpmml-pmml-dir:
 *   1. Load the model via jpmml API.
 *   2. Inspect InputField list → derive a small set of test rows from each
 *      field's declared domain (Interval or Value elements in the DataDictionary).
 *   3. Score each row with jpmml → oracle output.
 *   4. Write combined CSV (inputs + oracle outputs) and a PMML zip to --output-dir.
 *
 * Usage:
 *   java -cp pmml-evaluator-example-executable-1.7.7.jar:. org.cpmml.CrossValidator \
 *        --jpmml-pmml-dir PATH  --output-dir PATH
 */
public class CrossValidator {

    public static void main(String[] args) throws Exception {
        String jpmmlPmmlDir = null;
        String outputDir    = null;

        for (int i = 0; i < args.length - 1; i++) {
            switch (args[i]) {
                case "--jpmml-pmml-dir": jpmmlPmmlDir = args[i + 1]; break;
                case "--output-dir":     outputDir    = args[i + 1]; break;
            }
        }

        if (jpmmlPmmlDir == null || outputDir == null) {
            System.err.println("Usage: CrossValidator --jpmml-pmml-dir PATH --output-dir PATH");
            System.exit(1);
        }

        Path pmmlBase = Paths.get(jpmmlPmmlDir);
        Path outDir   = Paths.get(outputDir);
        Files.createDirectories(outDir);

        List<Path> pmmlFiles = Files.walk(pmmlBase)
            .filter(p -> {
                String s = p.toString();
                return s.endsWith(".pmml") || s.endsWith(".xml");
            })
            .sorted()
            .collect(Collectors.toList());

        int ok = 0, skipped = 0;
        List<String> skipList = new ArrayList<>();

        System.out.println("Generating oracle fixtures from " + pmmlFiles.size() + " PMML files...");
        System.out.println();

        for (Path pmmlFile : pmmlFiles) {
            String rel  = pmmlBase.relativize(pmmlFile).toString();
            String stem = rel.replaceAll("[\\\\/]", "_")
                            .replaceAll("\\.(pmml|xml)$", "");

            String reason = processFile(pmmlFile, stem, outDir);
            if (reason == null) {
                System.out.printf("  %-55s OK%n", rel);
                ok++;
            } else {
                System.out.printf("  %-55s SKIP (%s)%n", rel, reason);
                skipped++;
                skipList.add(rel + ": " + reason);
            }
        }

        System.out.println();
        System.out.println("Generated: " + ok + "  Skipped: " + skipped);
    }

    // -----------------------------------------------------------------------
    // Returns null on success, or a short skip reason on failure.
    // -----------------------------------------------------------------------

    static String processFile(Path pmmlFile, String stem, Path outDir) {
        // Pre-check: root XML element must be <PMML>
        String rootTag = rootElementName(pmmlFile);
        if (rootTag == null) return "XML parse error";
        if (!"PMML".equals(rootTag))
            return "not a PMML document (root element: " + rootTag + ")";

        Evaluator evaluator;
        try (InputStream is = Files.newInputStream(pmmlFile)) {
            evaluator = new LoadingModelEvaluatorBuilder()
                .load(is)
                .build();
            evaluator.verify(); // run embedded ModelVerification if present
        } catch (Exception e) {
            return abbrev(e);
        }

        List<InputField>  inputFields  = evaluator.getInputFields();
        List<org.jpmml.evaluator.OutputField> outputFields = evaluator.getOutputFields();
        List<TargetField> targetFields = evaluator.getTargetFields();

        // Detect transactional association model (group + item schema)
        boolean isTransactional = (evaluator instanceof HasGroupFields)
            && !((HasGroupFields) evaluator).getGroupFields().isEmpty();

        if (inputFields.isEmpty() && !isTransactional) {
            return "no input fields";
        }

        // Build per-field candidate values from the DataDictionary domain
        List<String>         inputCols  = new ArrayList<>();
        List<List<Object>>   fieldVals  = new ArrayList<>();

        if (isTransactional) {
            // Transactional association: item field needs Collection-valued inputs
            List<Object> baskets = transactionalBaskets(evaluator);
            if (baskets.isEmpty()) return "no items in transactional association model";
            for (InputField inputField : inputFields) {
                inputCols.add(inputField.getName());
                fieldVals.add(baskets);
            }
        } else {
            for (InputField inputField : inputFields) {
                inputCols.add(inputField.getName());
                fieldVals.add(domainValues(inputField));
            }
        }

        // Output column names: prefer OutputFields, fall back to TargetFields
        List<String> outputCols = new ArrayList<>();
        if (!outputFields.isEmpty()) {
            for (org.jpmml.evaluator.OutputField of : outputFields)
                if (of.getName() != null) outputCols.add(of.getName());
        }
        if (outputCols.isEmpty()) {
            for (TargetField tf : targetFields) {
                String name = tf.getName();
                outputCols.add(name != null ? name : "_target");
            }
        }
        if (outputCols.isEmpty()) return "no output fields";

        // Generate rows by round-robin across field value lists
        int nRows = fieldVals.stream().mapToInt(List::size).max().orElse(1);
        List<Map<String, Object>> inputRows = new ArrayList<>();
        for (int r = 0; r < nRows; r++) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (int f = 0; f < inputFields.size(); f++) {
                List<Object> vals = fieldVals.get(f);
                Object val = vals.get(r % vals.size());
                if (val != null) {
                    row.put(inputFields.get(f).getName(), val);
                }
                // null → omit from map → treated as missing by jpmml
            }
            inputRows.add(row);
        }

        // Score each row with jpmml (tolerate individual failures)
        List<Map<String, Object>> oracleRows    = new ArrayList<>();
        List<Map<String, Object>> validInputRows = new ArrayList<>();
        String lastError = null;

        for (Map<String, Object> row : inputRows) {
            try {
                Map<String, ?> result = evaluator.evaluate(row);
                Map<String, Object> decoded = new LinkedHashMap<>();
                for (Map.Entry<String, ?> e : result.entrySet()) {
                    String key = e.getKey();
                    if (key == null || key.isEmpty()) key = "_target";
                    Object val;
                    try {
                        val = EvaluatorUtil.decode(e.getValue());
                    } catch (Exception ex) {
                        continue; // skip un-decodable entries (e.g. model internals)
                    }
                    decoded.put(key, val);
                }
                oracleRows.add(decoded);
                validInputRows.add(row);
            } catch (Exception e) {
                lastError = abbrev(e);
            }
        }

        // If all rows failed, retry with missing (null) values — handles models
        // like scorecards where isMissing predicates are the only matching path.
        if (oracleRows.isEmpty() && !isTransactional) {
            Map<String, Object> nullRow = new LinkedHashMap<>();
            // all fields omitted → treated as missing by jpmml
            try {
                Map<String, ?> result = evaluator.evaluate(nullRow);
                Map<String, Object> decoded = new LinkedHashMap<>();
                for (Map.Entry<String, ?> e : result.entrySet()) {
                    String key = e.getKey();
                    if (key == null || key.isEmpty()) key = "_target";
                    Object val;
                    try { val = EvaluatorUtil.decode(e.getValue()); }
                    catch (Exception ex) { continue; }
                    decoded.put(key, val);
                }
                oracleRows.add(decoded);
                validInputRows.add(nullRow);
            } catch (Exception e) {
                // null row also failed
            }
        }

        if (oracleRows.isEmpty()) {
            return "scoring failed: " + (lastError != null ? lastError : "all rows failed");
        }

        // Build combined column list: input cols + output cols
        List<String> allCols = new ArrayList<>(inputCols);
        for (String oc : outputCols) {
            if (!allCols.contains(oc)) allCols.add(oc);
        }

        // Write CSV — transactional models expand each basket into multiple rows
        Path csvPath = outDir.resolve(stem + ".csv");
        try {
            if (isTransactional && !inputCols.isEmpty()) {
                String itemCol = inputCols.get(0);
                // Add transaction column at the front
                List<String> txCols = new ArrayList<>();
                txCols.add("transaction");
                txCols.addAll(allCols);
                writeCsvTransactional(csvPath, txCols, itemCol, validInputRows, oracleRows);
            } else {
                writeCsv(csvPath, allCols, validInputRows, oracleRows);
            }
        } catch (IOException e) {
            return "CSV write failed: " + e.getMessage();
        }

        // Write zip containing the PMML
        Path zipPath = outDir.resolve(stem + ".zip");
        try {
            writeZip(zipPath, pmmlFile, stem + ".pmml");
        } catch (IOException e) {
            return "ZIP write failed: " + e.getMessage();
        }

        return null; // success
    }

    // -----------------------------------------------------------------------
    // Check root XML element (rejects standalone ModelVerification fragments)
    // -----------------------------------------------------------------------

    static String rootElementName(Path file) {
        try (InputStream is = Files.newInputStream(file)) {
            javax.xml.stream.XMLInputFactory xif = javax.xml.stream.XMLInputFactory.newFactory();
            javax.xml.stream.XMLStreamReader xsr = xif.createXMLStreamReader(is);
            try {
                while (xsr.hasNext()) {
                    if (xsr.next() == javax.xml.stream.XMLStreamConstants.START_ELEMENT) {
                        return xsr.getLocalName();
                    }
                }
            } finally {
                xsr.close();
            }
        } catch (Exception e) {
            // fall through
        }
        return null;
    }

    // -----------------------------------------------------------------------
    // Build transaction baskets from association model items for transactional
    // schema (group + item fields).
    // -----------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    static List<Object> transactionalBaskets(Evaluator evaluator) {
        List<String> itemValues = new ArrayList<>();
        if (evaluator instanceof ModelEvaluator) {
            org.dmg.pmml.Model model = ((ModelEvaluator<?>) evaluator).getModel();
            if (model instanceof org.dmg.pmml.association.AssociationModel) {
                org.dmg.pmml.association.AssociationModel am =
                    (org.dmg.pmml.association.AssociationModel) model;
                for (org.dmg.pmml.association.Item item : am.getItems()) {
                    itemValues.add(item.getValue());
                }
            }
        }
        if (itemValues.isEmpty()) return Collections.emptyList();

        List<Object> baskets = new ArrayList<>();
        // Single-item transactions
        for (String v : itemValues) {
            baskets.add(Collections.singletonList(v));
        }
        // Consecutive pairs
        for (int i = 0; i < itemValues.size() - 1; i++) {
            baskets.add(Arrays.asList(itemValues.get(i), itemValues.get(i + 1)));
        }
        return baskets;
    }

    // -----------------------------------------------------------------------
    // Convert Collection to pipe-delimited string for CSV output
    // -----------------------------------------------------------------------

    static String collectionToString(Collection<?> coll) {
        return coll.stream()
            .map(String::valueOf)
            .collect(Collectors.joining("|"));
    }

    // -----------------------------------------------------------------------
    // Extract candidate test values for one InputField from its DataField
    // domain declaration.  Falls back to type-based defaults when no domain
    // information is present.
    // -----------------------------------------------------------------------

    static List<Object> domainValues(InputField inputField) {
        List<Object> result = new ArrayList<>();

        Field<?> field = inputField.getField();

        if (field instanceof DataField) {
            DataField df = (DataField) field;
            OpType    op = inputField.getOpType();

            // Categorical / ordinal: one row per valid Value element
            if (op == OpType.CATEGORICAL || op == OpType.ORDINAL) {
                for (org.dmg.pmml.Value v : df.getValues()) {
                    org.dmg.pmml.Value.Property prop = v.getProperty();
                    if (prop == null
                            || prop == org.dmg.pmml.Value.Property.VALID) {
                        result.add(v.getValue());
                    }
                }
            }

            // Continuous: sample left, mid, right from Interval(s)
            if (result.isEmpty() && !df.getIntervals().isEmpty()) {
                for (Interval iv : df.getIntervals()) {
                    double lo = iv.getLeftMargin()  != null ? iv.getLeftMargin().doubleValue()  : 0.0;
                    double hi = iv.getRightMargin() != null ? iv.getRightMargin().doubleValue() : 1.0;
                    result.add(lo);
                    result.add((lo + hi) / 2.0);
                    result.add(hi);
                    break; // first interval is sufficient
                }
            }
        }

        // Fallback when no domain info is available
        if (result.isEmpty()) {
            DataType dt = inputField.getDataType();
            if (dt == DataType.BOOLEAN) {
                result.add(Boolean.FALSE);
                result.add(Boolean.TRUE);
            } else if (dt == DataType.STRING) {
                result.add("value");
            } else {
                result.add(0.0);
                result.add(1.0);
            }
        }

        return result;
    }

    // -----------------------------------------------------------------------
    // CSV writer — combines input rows and oracle output rows into one file.
    // -----------------------------------------------------------------------

    static void writeCsv(Path path,
                          List<String> cols,
                          List<Map<String, Object>> inputRows,
                          List<Map<String, Object>> oracleRows) throws IOException {
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path))) {
            w.println(cols.stream().map(CrossValidator::csvCell).collect(Collectors.joining(",")));
            for (int i = 0; i < inputRows.size(); i++) {
                Map<String, Object> in  = inputRows.get(i);
                Map<String, Object> out = oracleRows.get(i);
                List<String> cells = new ArrayList<>();
                for (String col : cols) {
                    Object v = in.containsKey(col) ? in.get(col) : out.get(col);
                    cells.add(csvCell(v != null ? v.toString() : ""));
                }
                w.println(String.join(",", cells));
            }
        }
    }

    // Transactional CSV: each basket (Collection) input becomes multiple rows
    // with the same transaction ID. Output values are repeated for each row.
    @SuppressWarnings("unchecked")
    static void writeCsvTransactional(Path path,
                                       List<String> cols,
                                       String itemCol,
                                       List<Map<String, Object>> inputRows,
                                       List<Map<String, Object>> oracleRows) throws IOException {
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path))) {
            w.println(cols.stream().map(CrossValidator::csvCell).collect(Collectors.joining(",")));
            for (int i = 0; i < inputRows.size(); i++) {
                Map<String, Object> in  = inputRows.get(i);
                Map<String, Object> out = oracleRows.get(i);
                int txId = i + 1;

                // Get basket items from the item field
                Object itemVal = in.get(itemCol);
                List<String> items;
                if (itemVal instanceof Collection) {
                    items = ((Collection<?>) itemVal).stream()
                        .map(String::valueOf)
                        .collect(Collectors.toList());
                } else {
                    items = Collections.singletonList(
                        itemVal != null ? itemVal.toString() : "");
                }

                // One row per item in the basket
                for (String item : items) {
                    List<String> cells = new ArrayList<>();
                    for (String col : cols) {
                        if ("transaction".equals(col)) {
                            cells.add(csvCell(String.valueOf(txId)));
                        } else if (col.equals(itemCol)) {
                            cells.add(csvCell(item));
                        } else {
                            Object v = in.containsKey(col) ? in.get(col) : out.get(col);
                            cells.add(csvCell(v != null ? v.toString() : ""));
                        }
                    }
                    w.println(String.join(",", cells));
                }
            }
        }
    }

    static String csvCell(String s) {
        if (s == null) return "";
        if (s.contains(",") || s.contains("\"") || s.contains("\n") || s.contains("\r")) {
            return "\"" + s.replace("\"", "\"\"") + "\"";
        }
        return s;
    }

    // -----------------------------------------------------------------------
    // ZIP writer
    // -----------------------------------------------------------------------

    static void writeZip(Path zipPath, Path pmmlFile, String entryName) throws IOException {
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(zipPath))) {
            zos.putNextEntry(new ZipEntry(entryName));
            Files.copy(pmmlFile, zos);
            zos.closeEntry();
        }
    }

    // -----------------------------------------------------------------------
    // Short exception summary for skip messages
    // -----------------------------------------------------------------------

    static String abbrev(Exception e) {
        String msg = e.getMessage();
        if (msg != null && msg.length() > 80) msg = msg.substring(0, 77) + "...";
        return e.getClass().getSimpleName() + (msg != null ? ": " + msg : "");
    }
}
