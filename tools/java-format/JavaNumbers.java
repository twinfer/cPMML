// Regenerates test/data/java_format/{double,float}.txt, the oracle for
// test/java_format.cc:
//   javac -d /tmp/jn tools/java-format/JavaNumbers.java
//   java -cp /tmp/jn JavaNumbers test/data/java_format/double.txt test/data/java_format/float.txt
// Needs JDK 19 or later (shortest-decimal Double.toString, JDK-4511638).
import java.util.*;
import java.io.*;

// Writes Double.toString / Float.toString for a fixed set of values, keyed by
// their IEEE bit patterns, for cPMML's formatter unit test.
public class JavaNumbers {
  static LinkedHashSet<Long> d = new LinkedHashSet<>();
  static LinkedHashSet<Integer> f = new LinkedHashSet<>();
  static void D(double x) { d.add(Double.doubleToRawLongBits(x)); f.add(Float.floatToRawIntBits((float) x)); }
  static void Dn(double x) { D(x); D(Math.nextUp(x)); D(Math.nextDown(x)); D(-x); }
  static void F(float x) { f.add(Float.floatToRawIntBits(x)); }
  static void Fn(float x) { F(x); F(Math.nextUp(x)); F(Math.nextDown(x)); F(-x); }
  public static void main(String[] a) throws Exception {
    D(0.0); D(-0.0); D(Double.NaN); D(Double.POSITIVE_INFINITY); D(Double.NEGATIVE_INFINITY);
    Dn(Double.MIN_VALUE); Dn(Double.MAX_VALUE); Dn(Double.MIN_NORMAL); Dn(2 * Double.MIN_VALUE);
    Dn(1e7); Dn(1e-3); Dn(9999999.999999998); Dn(9999999.5); Dn(0.0009999999999999998); Dn(0.00099999);
    Dn(1e-4); Dn(1e6); Dn(9999999.0); Dn(10000000.5); Dn(0.001000001);
    Fn(1e7f); Fn(1e-3f); Fn(9999999f); Fn(Float.MIN_VALUE); Fn(Float.MAX_VALUE); Fn(Float.MIN_NORMAL);
    Fn(0.1f); Fn(1.1f); Fn(3.3f); Fn(16777216f); Fn(16777217f); Fn(1.0e-45f); Fn(3.4028235e38f);
    for (int e = -325; e <= 309; e++) { double x = Double.parseDouble("1e" + e); D(x); D(-x); D(Math.nextUp(x)); D(Math.nextDown(x)); D(Double.parseDouble("2e" + e)); D(Double.parseDouble("5e" + e)); D(Double.parseDouble("9.5e" + e)); D(Double.parseDouble("1.5e" + e)); }
    for (int e = -1074; e <= 1023; e += 7) D(Math.scalb(1.0, e));
    for (int e = -149; e <= 127; e += 3) F(Math.scalb(1.0f, e));
    for (int i = 0; i <= 200; i++) { D(i); D(-i); D(i / 10.0); D(i / 100.0); D(i / 1000.0); D(i / 3.0); D(i / 7.0); D(i * 0.1); }
    D(0.1 + 0.2); D(1.0 / 3); D(2.0 / 3); D(Math.PI); D(Math.E); D(Math.sqrt(2));
    D(1e23); D(2e23); D(8.41e21); D(5e-324); D(1.0E22); D(1.0E21); D(123456789012345680.0); D(4.35e15);
    D(9007199254740992.0); D(9007199254740993.0); D(9007199254740994.0); D(1.0E16); D(1.2345678901234567E16);
    D(2.2250738585072014E-308); D(2.225073858507201E-308); D(4.9E-324); D(9.9E-324); D(1.0E-322);
    D(Long.MAX_VALUE); D(Long.MIN_VALUE); D(Integer.MAX_VALUE); D(Integer.MIN_VALUE); D(1234567.0); D(12345678.0);
    D(0.5); D(0.25); D(0.125); D(0.0625); D(0.333); D(0.6666666666666666); D(0.1234567890123); D(98.76543210987654);
    D(1.5); D(2.5); D(100.0); D(1234.5); D(0.00123); D(0.000123); D(1.0E-5); D(5.0E-5); D(1.25E-4);
    Random r = new Random(20261117L);
    for (int i = 0; i < 600; i++) D(Double.longBitsToDouble(r.nextLong()));
    for (int i = 0; i < 600; i++) { double x = Math.pow(10, r.nextDouble() * 24 - 12) * (r.nextBoolean() ? 1 : -1); D(x); }
    for (int i = 0; i < 200; i++) { double x = Math.round(r.nextDouble() * 1e6) / 1e3; D(x); }
    for (int i = 0; i < 500; i++) F(Float.intBitsToFloat(r.nextInt()));
    for (int i = 0; i < 500; i++) F((float) (Math.pow(10, r.nextDouble() * 16 - 8) * (r.nextBoolean() ? 1 : -1)));
    try (PrintWriter w = new PrintWriter(new FileWriter(a[0]))) {
      w.print("# Double.toString, JDK " + System.getProperty("java.version") + ": IEEE 754 bits (hex), text\n");
      for (long b : d) w.print(String.format("%016x", b) + " " + Double.toString(Double.longBitsToDouble(b)) + "\n");
    }
    try (PrintWriter w = new PrintWriter(new FileWriter(a[1]))) {
      w.print("# Float.toString, JDK " + System.getProperty("java.version") + ": IEEE 754 bits (hex), text\n");
      for (int b : f) w.print(String.format("%08x", b) + " " + Float.toString(Float.intBitsToFloat(b)) + "\n");
    }
  }
}
