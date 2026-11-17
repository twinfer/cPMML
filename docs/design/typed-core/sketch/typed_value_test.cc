// Compile and behaviour test for typed_value.h, plus a micro-benchmark of the
// comparison paths. Uses the repository's own strict reader and Java writer.
#include "typed_value.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "utils/java_format.h"
#include "utils/lexical.h"

namespace pmml {

std::optional<Integer> parse(std::type_identity<Integer>, std::string_view text, ParseContext&) {
  long parsed = 0;
  if (!lexical::parse_long(text, parsed)) return std::nullopt;
  return Integer{parsed};
}
std::optional<Float> parse(std::type_identity<Float>, std::string_view text, ParseContext&) {
  double parsed = 0;  // xs:float has the xs:double lexical space; the value is rounded to single
  if (!lexical::parse_double(text, parsed)) return std::nullopt;
  return Float{static_cast<float>(parsed)};
}
std::optional<Double> parse(std::type_identity<Double>, std::string_view text, ParseContext&) {
  double parsed = 0;
  if (!lexical::parse_double(text, parsed)) return std::nullopt;
  return Double{parsed};
}
std::optional<Boolean> parse(std::type_identity<Boolean>, std::string_view text, ParseContext&) {
  bool parsed = false;
  if (!lexical::parse_boolean(text, parsed)) return std::nullopt;
  return Boolean{parsed};
}
std::optional<String> parse(std::type_identity<String>, std::string_view text, ParseContext& ctx) {
  if (ctx.pool) return String{ctx.pool->intern(text)};
  if (ctx.frozen)
    if (const StringRep* pooled = ctx.frozen->find(text)) return String{pooled};  // known text: pooled rep
  return String{ctx.arena->make(text)};
}

void java_text(const Integer& x, std::string& out) { out += java::integer_text(x.v); }
void java_text(const Float& x, std::string& out) {
  char buffer[java::MAX_TEXT];
  out.append(buffer, java::write_float(buffer, x.v));
}
void java_text(const Double& x, std::string& out) {
  char buffer[java::MAX_TEXT];
  out.append(buffer, java::write_double(buffer, x.v));
}
void java_text(const Boolean& x, std::string& out) { out += x.v ? "true" : "false"; }
void java_text(const String& x, std::string& out) { out += x.text(); }

}  // namespace pmml

using namespace pmml;

static_assert(PmmlNumeric<Integer> && PmmlNumeric<Float> && PmmlNumeric<Double>);
static_assert(PmmlParsedType<Boolean> && PmmlParsedType<String> && !PmmlNumeric<String>);
#if PMML_WITH_TEMPORAL
static_assert(Temporal<DateDaysSince1960> && Temporal<Date> && !Temporal<Double>);
#endif
static_assert(std::is_trivially_copyable_v<Value> && std::is_trivially_destructible_v<Value>);  // a frame copy is a memcpy

int main() {
  StringPool pool;
  RecordArena arena;
  ParseContext load{&pool, nullptr, nullptr};
  ParseContext run{nullptr, &pool, &arena};
  std::string scratch;

  std::printf("sizeof(Value)=%zu alignof=%zu alternatives=%zu sizeof(StringRep)=%zu\n", sizeof(Value),
              alignof(Value), std::variant_size_v<Value>, sizeof(StringRep));

  auto cmp = [&](const Value& a, const Value& b) { return compare_values(a, b, run, scratch); };
  auto is = [](std::optional<std::partial_ordering> r, std::partial_ordering o) { return r && *r == o; };

  // float against double: compared in double, either order (the two failed
  // patches compared in float when the float came first).
  const Value f = Float{0.1f}, d = Double{0.1};
  assert(is(cmp(f, d), std::partial_ordering::greater));
  assert(is(cmp(d, f), std::partial_ordering::less));
  // equal and lessOrEqual come from one ordering: equal => lessOrEqual.
  const Value f2 = *parse(std::type_identity<Float>{}, "0.1", load);
  assert(is(cmp(f, f2), std::partial_ordering::equivalent));

  // string against integer: compared as text, whichever side.
  const Value s3 = String{pool.intern("3")}, s30 = String{pool.intern("3.0")}, i3 = Integer{3};
  assert(is(cmp(s3, i3), std::partial_ordering::equivalent));
  assert(is(cmp(i3, s3), std::partial_ordering::equivalent));
  assert(!is(cmp(s30, i3), std::partial_ordering::equivalent));

  // an input naming a pooled string gets the pooled rep; one the pool never
  // saw is equal to another only if their texts are.
  const Value in3 = *parse(std::type_identity<String>{}, "3", run);
  assert(std::get<String>(in3).rep == std::get<String>(s3).rep);
  const Value u1 = String{arena.make("zz")}, u2 = String{arena.make("yy")}, u3 = String{arena.make("zz")};
  assert(!is(cmp(u1, u2), std::partial_ordering::equivalent));
  assert(is(cmp(u1, u3), std::partial_ordering::equivalent));

  // boolean against integer: no common type.
  assert(!cmp(Boolean{true}, Integer{1}).has_value());
  // INF against itself is indeterminate; NaN is unordered.
  const double inf = std::numeric_limits<double>::infinity();
  assert(is(cmp(Double{inf}, Double{inf}), std::partial_ordering::unordered));
  assert(is(cmp(Double{std::nan("")}, Double{1}), std::partial_ordering::unordered));

  // arithmetic: promotion, integer overflow, division by zero.
  assert(std::holds_alternative<Double>(plus(Integer{1}, Double{0.5})));
  assert(std::holds_alternative<Float>(plus(Integer{1}, Float{0.5f})));
  assert(std::get<Integer>(divide(Integer{-7}, Integer{2})).v == -3);
  assert(std::holds_alternative<Missing>(divide(Double{0}, Double{0})));
  assert(std::holds_alternative<Invalid>(divide(Integer{1}, Integer{0})));
  assert(std::get<Double>(divide(Double{1}, Double{0})).v == inf);
  assert(std::holds_alternative<Invalid>(plus(Integer{std::numeric_limits<std::int64_t>::max()}, Integer{1})));
  assert(std::holds_alternative<Missing>(plus(Missing{}, Integer{1})));

  // casts: a double into an integer field keeps no fraction.
  auto cast_int = [&](const Value& v) { return std::visit(CastTo<Integer>{&run, &scratch}, v); };
  assert(std::get<Integer>(cast_int(Double{3.0})).v == 3);
  assert(std::holds_alternative<Invalid>(cast_int(Double{3.5})));
  assert(std::get<Float>(std::visit(CastTo<Float>{&run, &scratch}, Value{Double{0.1}})).v == 0.1f);
  const Value t = std::visit(CastTo<String>{&run, &scratch}, Value{Double{2.0}});
  assert(std::get<String>(t).text() == "2.0");
  assert(std::holds_alternative<Invalid>(std::visit(CastTo<Boolean>{&run, &scratch}, Value{Integer{2}})));
  assert(std::get<Boolean>(std::visit(CastTo<Boolean>{&run, &scratch}, Value{Double{1.0}})).v);

  // Java text of each type
  std::string text;
  for (const Value& v : {Value{Integer{7}}, Value{Float{0.1f}}, Value{Double{1e7}}, Value{Boolean{true}}, s3}) {
    std::visit(JavaText{&text}, v);
    text += '|';
  }
  assert(text == "7|0.1|1.0E7|true|3|");

  // container keys: reflexive, -0 == 0, pooled strings by pointer
  {
    std::unordered_map<Value, int, KeyHash, KeyEqual> table;
    table[Double{inf}] = 1;
    table[Double{0.0}] = 2;
    table[s3] = 3;
    assert(table.count(Double{inf}) == 1 && table.at(Double{-0.0}) == 2);
    assert(table.at(in3) == 3 && table.count(String{arena.make("3")}) == 1);
    assert(string_equal(std::get<String>(s3), std::get<String>(in3)));
    assert(!string_equal(std::get<String>(s3), std::get<String>(s30)));
    assert(is_nonfinite(Double{inf}) && !is_nonfinite(Double{1}));
  }

  // ---- micro-benchmark: SimplePredicate field <= constant over 1M records
  constexpr size_t N = 1 << 20;
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> uni(0, 1);
  struct OldValue {  // today's Value: double, missing flag, std::string
    double value;
    bool missing;
    std::string svalue;
  };
  std::vector<OldValue> old_values(N);
  std::vector<Value> new_values(N);
  for (size_t i = 0; i < N; i++) {
    const double x = uni(rng);
    const bool m = x < 0.05;
    old_values[i] = {x, m, {}};
    new_values[i] = m ? Value{Missing{}} : Value{Double{x}};
  }
  auto time = [](auto&& body) {
    double best = 1e30;
    size_t sink = 0;
    for (int rep = 0; rep < 7; rep++) {
      const auto t0 = std::chrono::steady_clock::now();
      sink += body();
      const auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    return std::pair{best / N, sink};
  };
  const auto [t_old, s_old] = time([&] {
    size_t n = 0;
    for (const auto& v : old_values) n += !v.missing && v.value <= 0.5;
    return n;
  });
  const DoubleLessOrEqual bound{0, 0.5};
  const auto [t_bound, s_bound] = time([&] {
    size_t n = 0;
    for (size_t i = 0; i < N; i++) n += bound(&new_values[i]) == Tri::True;
    return n;
  });
  const Value half = Double{0.5};
  const auto [t_visit, s_visit] = time([&] {
    size_t n = 0;
    for (const auto& v : new_values) {
      const auto r = compare_values(v, half, run, scratch);
      n += r && (*r == std::partial_ordering::less || *r == std::partial_ordering::equivalent);
    }
    return n;
  });
  assert(s_old == s_bound && s_bound == s_visit);
  std::printf("ns/compare: old Value %.2f | bound fast path %.2f | generic std::visit %.2f\n", t_old, t_bound,
              t_visit);
  std::printf("all assertions passed\n");
  return 0;
}
