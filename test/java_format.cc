/*******************************************************************************
 * The Java-compatible formatter (src/utils/java_format.h) against Java itself.
 *
 * data/java_format/double.txt and float.txt hold Double.toString and
 * Float.toString of some 12000 values, keyed by their IEEE bits, written by
 * tools/java-format/JavaNumbers.java on JDK 19+: the thresholds 1e7 and 1e-3
 * and their neighbours, subnormals, MIN/MAX values, powers of two and ten,
 * integers held as doubles up to and past 2^53, random bit patterns and
 * random decades. Every float is also checked where its shortest float text
 * differs from the double's (0.1f).
 *
 * Usage: java_format.exe   (run from the test build directory)
 *******************************************************************************/

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "core/value.h"
#include "utils/java_format.h"

static int failures = 0;

static void check(const std::string& what, const std::string& actual, const std::string& expected) {
  if (actual == expected) return;
  if (++failures <= 30) std::cerr << what << ": got \"" << actual << "\", Java gives \"" << expected << "\"" << std::endl;
}

template <class Bits, class F>
static size_t check_table(const std::string& path, const bool single) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "cannot open " << path << std::endl;
    failures++;
    return 0;
  }
  size_t rows = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::string hex, expected;
    fields >> hex >> expected;
    const Bits bits = static_cast<Bits>(std::stoull(hex, nullptr, 16));
    F value;
    std::memcpy(&value, &bits, sizeof(value));
    char buffer[java::MAX_TEXT];
    const char* end = single ? java::write_float(buffer, static_cast<float>(value))
                             : java::write_double(buffer, static_cast<double>(value));
    check(std::string(single ? "float " : "double ") + hex, std::string(static_cast<const char*>(buffer), end), expected);

    // The text reads back as the same value (NaN aside).
    double back = 0;
    if (!java::parse(expected, back)) {
      check("parse " + expected, "unreadable", expected);
    } else if (!std::isnan(static_cast<double>(value))) {
      const bool same = single ? static_cast<float>(back) == static_cast<float>(value)
                               : back == static_cast<double>(value) &&
                                     std::signbit(back) == std::signbit(static_cast<double>(value));
      if (!same) check("round trip " + expected, java::double_text(back), expected);
    }
    rows++;
  }
  return rows;
}

int main() {
  const size_t doubles = check_table<uint64_t, double>("data/java_format/double.txt", false);
  const size_t floats = check_table<uint32_t, float>("data/java_format/float.txt", true);
  if (doubles < 8000 || floats < 3000) {
    std::cerr << "oracle tables too small: " << doubles << " doubles, " << floats << " floats" << std::endl;
    failures++;
  }

  // The rules, spelled out (each is also in the tables).
  check("1.5", java::double_text(1.5), "1.5");
  check("2", java::double_text(2), "2.0");
  check("1e7", java::double_text(1e7), "1.0E7");
  check("below 1e7", java::double_text(9999999.999999998), "9999999.999999998");
  check("1e-3", java::double_text(1e-3), "0.001");
  check("below 1e-3", java::double_text(9.999999999999998E-4), "9.999999999999998E-4");
  check("1e-4", java::double_text(1e-4), "1.0E-4");
  check("min subnormal", java::double_text(std::numeric_limits<double>::denorm_min()), "4.9E-324");
  check("max", java::double_text(std::numeric_limits<double>::max()), "1.7976931348623157E308");
  check("2^53+2", java::double_text(9007199254740994.0), "9.007199254740994E15");
  check("1e23", java::double_text(1e23), "1.0E23");
  check("-0", java::double_text(-0.0), "-0.0");
  check("NaN", java::double_text(std::numeric_limits<double>::quiet_NaN()), "NaN");
  check("inf", java::double_text(std::numeric_limits<double>::infinity()), "Infinity");
  check("-inf", java::double_text(-std::numeric_limits<double>::infinity()), "-Infinity");
  check("0.1 as float", java::float_text(0.1), "0.1");
  check("1.1 as float", java::float_text(1.1), "1.1");
  check("(double) 1.1f", java::double_text(static_cast<float>(1.1)), "1.100000023841858");
  check("1e300 as float", java::float_text(1e300), "Infinity");
  check("123456789 as float", java::float_text(123456789), "1.2345679E8");
  check("(double) 123456789f", java::double_text(static_cast<float>(123456789)), "1.23456792E8");
  check("integer", java::text({7, java::Kind::INTEGER}), "7");
  check("negative integer", java::text({-123456789, java::Kind::INTEGER}), "-123456789");
  check("integer 2^62", java::text({4611686018427387904.0, java::Kind::INTEGER}), "4611686018427387904");
  check("integer, not whole", java::text({2.5, java::Kind::INTEGER}), "2.5");
  check("boolean true", java::text({1, java::Kind::BOOLEAN}), "true");
  check("boolean false", java::text({0, java::Kind::BOOLEAN}), "false");

  double parsed = 0;
  if (java::parse("Infinity ", parsed) || java::parse("inf", parsed) || java::parse("1.5abc", parsed)) {
    std::cerr << "java::parse accepts what it should not" << std::endl;
    failures++;
  }

#if defined(__cpp_lib_format)
  check("format double", std::format("{}", java::Number{1e7}), "1.0E7");
  check("format float", std::format("{}", java::Number{0.1, java::Kind::FLOAT}), "0.1");
  check("format integer", std::format("[{}]", java::Number{42, java::Kind::INTEGER}), "[42]");
  check("format boolean", std::format("{}", java::Number{0, java::Kind::BOOLEAN}), "false");
  // std::format's own double output is shortest, but not Java's notation.
  check("std::format double", std::format("{}", 1e7), "1e+07");

  // Value: a missing one is empty, a string its text, a number a Java double.
  check("format missing Value", std::format("{}", Value()), "");
  check("format numeric Value", std::format("{}", Value(2.0)), "2.0");
  const cpmml::StringScope strings(cpmml::StringScope::Loading{}, std::make_shared<cpmml::DocumentStrings>());
  check("format string Value", std::format("{}", Value::string("abc")), "abc");
#else
  std::cout << "no <format> in this standard library: std::formatter checks skipped" << std::endl;
#endif

  if (failures) {
    std::cerr << failures << " mismatches" << std::endl;
    return 1;
  }
  std::cout << doubles << " doubles and " << floats << " floats as Java writes them" << std::endl;
  return 0;
}
