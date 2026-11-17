/*
 * JpmmlOracle -- the JPMML 1.7.7 side of the differential fuzzer
 * (docs/design/typed-core.md §8.2). One JVM scores every case of a corpus.
 *
 * Build and run (tools/fuzz/run.py does both):
 *   javac -cp tools/pmml-evaluator-example-executable-1.7.7.jar -d OUT tools/fuzz/JpmmlOracle.java
 *   java -cp OUT:tools/pmml-evaluator-example-executable-1.7.7.jar JpmmlOracle MANIFEST
 *
 * MANIFEST has one case per line, tab-separated:
 *   case-id  model.pmml  inputs.csv  output.csv
 * inputs.csv is the case's input columns (RFC 4180; an unquoted N/A is a
 * missing value, any other cell -- the empty one included -- is that text).
 * output.csv gets the columns EvaluationExample writes, in its order: the
 * target fields, then the output fields; one row per input row, each cell
 * EvaluatorUtil.decode(value).toString(), N/A for null, as EvaluationExample's
 * cell formatter does. A row whose evaluation throws has every cell
 * "ERROR: <exception class>: <message>" (EvaluationExample aborts the whole
 * file instead). A model that does not load or verify gives a one-column file
 * "LOAD-ERROR" with the exception in its only row.
 *
 * The model is read as EvaluationExample reads it by default
 * (Example.readPMML, a ModelEvaluatorBuilder with no optimizer or interner
 * batteries), so a value printed here is the one tools/cross-validate-text.py
 * would get from EvaluationExample.
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.EvaluatorUtil;
import org.jpmml.evaluator.ModelEvaluatorBuilder;
import org.jpmml.evaluator.OutputField;
import org.jpmml.evaluator.TargetField;
import org.jpmml.evaluator.example.Example;

public class JpmmlOracle {

  static final String MISSING = "N/A";

  public static void main(String[] args) throws IOException {
    if (args.length != 1) {
      System.err.println("usage: JpmmlOracle MANIFEST");
      System.exit(2);
    }
    int cases = 0;
    for (String line : Files.readAllLines(Paths.get(args[0]), StandardCharsets.UTF_8)) {
      if (line.isEmpty()) continue;
      String[] f = line.split("\t");
      runCase(new File(f[1]), Paths.get(f[2]), Paths.get(f[3]));
      cases++;
    }
    System.err.println("JpmmlOracle: " + cases + " cases");
  }

  static String describe(Throwable e) {
    String message = e.getMessage() == null ? "" : e.getMessage();
    return e.getClass().getSimpleName() + ": " + message.replace('\n', ' ').replace('\r', ' ');
  }

  static void runCase(File model, java.nio.file.Path inputs, java.nio.file.Path output) throws IOException {
    Evaluator evaluator;
    List<String> columns = new ArrayList<>();
    try {
      PMML pmml = Example.readPMML(model);
      evaluator = new ModelEvaluatorBuilder(pmml).build();
      evaluator.verify();
      for (TargetField t : evaluator.getTargetFields()) columns.add(t.getName());
      for (OutputField o : evaluator.getOutputFields()) columns.add(o.getName());
    } catch (Throwable e) {
      try (BufferedWriter w = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
        w.write("LOAD-ERROR\n");
        w.write(cell("LOAD-ERROR: " + describe(e)) + "\n");
      }
      return;
    }

    List<List<String>> rows = readCsv(inputs);
    List<String> header = rows.get(0);
    try (BufferedWriter w = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
      writeRow(w, columns);
      for (int r = 1; r < rows.size(); r++) {
        List<String> row = rows.get(r);
        Map<String, Object> arguments = new LinkedHashMap<>();
        for (int c = 0; c < header.size(); c++) arguments.put(header.get(c), c < row.size() ? row.get(c) : null);
        List<String> cells = new ArrayList<>();
        try {
          Map<String, ?> results = evaluator.evaluate(arguments);
          for (String column : columns) {
            Object value = EvaluatorUtil.decode(results.get(column));
            cells.add(value == null ? MISSING : value.toString());
          }
        } catch (Throwable e) {
          cells.clear();
          String text = "ERROR: " + describe(e);
          for (int i = 0; i < columns.size(); i++) cells.add(text);
        }
        writeRow(w, cells);
      }
    }
  }

  static String cell(String text) {
    if (text.indexOf(',') < 0 && text.indexOf('"') < 0 && text.indexOf('\n') < 0) return text;
    return "\"" + text.replace("\"", "\"\"") + "\"";
  }

  static void writeRow(BufferedWriter w, List<String> cells) throws IOException {
    for (int i = 0; i < cells.size(); i++) {
      if (i > 0) w.write(',');
      w.write(cell(cells.get(i)));
    }
    w.write('\n');
  }

  // RFC 4180 rows; an unquoted N/A is null (a missing value).
  static List<List<String>> readCsv(java.nio.file.Path path) throws IOException {
    List<List<String>> rows = new ArrayList<>();
    try (BufferedReader r = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
      String line;
      while ((line = r.readLine()) != null) {
        List<String> row = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean quoted = false, wasQuoted = false;
        for (int i = 0; i < line.length(); i++) {
          char ch = line.charAt(i);
          if (quoted) {
            if (ch == '"' && i + 1 < line.length() && line.charAt(i + 1) == '"') { current.append('"'); i++; }
            else if (ch == '"') quoted = false;
            else current.append(ch);
          } else if (ch == '"') { quoted = true; wasQuoted = true; }
          else if (ch == ',') { row.add(finish(current, wasQuoted)); current.setLength(0); wasQuoted = false; }
          else current.append(ch);
        }
        row.add(finish(current, wasQuoted));
        rows.add(row);
      }
    }
    return rows;
  }

  static String finish(StringBuilder current, boolean wasQuoted) {
    String text = current.toString();
    return (!wasQuoted && text.equals(MISSING)) ? null : text;
  }
}
