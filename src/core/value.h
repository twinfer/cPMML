/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_VALUE_H
#define CPMML_VALUE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "core/types/cast.h"
#include "core/types/parse.h"
#include "core/types/string_pool.h"
#include "core/types/value.h"
#include "datatype.h"
#include "utils/utils.h"

/**
 * Design note -- the engine's value (docs/design/typed-core.md §2.1, §2.3,
 * §7 P5).
 *
 * A Value holds a types::Value: Missing, or a value of one PMML dataType
 * (Integer, Float, Double, Boolean, String). Numbers and booleans are the
 * value itself (a float is an IEEE single since P6.2: XML Schema Part 2
 * §3.2.4, "float is patterned after the IEEE single-precision 32-bit floating
 * point type"); a String is a pointer to the StringRep of its text, which is
 * one of
 *
 *   - the document's StringPool (DocumentStrings::pool): every string the
 *     load names, interned in the order the load names them, so two pooled
 *     reps are equal iff they are the same pointer. Frozen after the load,
 *     then only read (find), from any number of threads;
 *   - an arena: the evaluation's (StringScope, per record) for a text the
 *     pool lacks (an input the document never names, a string a function
 *     builds), or the document's load arena for a string the load edits.
 *
 * There is no Indexer string map any more: the pool is that map (§2.3). P5
 * kept today's semantics with two compatibility switches, each one place
 * that one P6 commit deletes. Switch 1 (legacy::float_value: a float stored
 * as the Double it was read as, B4, B5) was deleted by P6.2: a float is a
 * types::Float. What remains:
 *
 *   (2) legacy: a String's as_double() is its legacy code, its rep's pool
 *       id, -1 for a string the pool lacks, for the numeric readers of a
 *       string (B38, P6.8). Comparisons no longer read it: P6.1 made two
 *       strings equal iff their texts are (types::string_equal), and P6.3
 *       deleted LegacyCompare: every comparison is typed (core/compare.h,
 *       §2.7; B18, D1, D2, D9), and containers keyed by values of one type
 *       read cpmml::ContainerKey (a string's key is its pool id).
 *
 * A String's rep is canonical (P6.1): a string whose text the pool holds has
 * the pooled rep, whoever made it (an input, a cast, an edit: uppercase,
 * lowercase, substring, trimBlanks), so the code of its rep is the code of
 * its text, and a container keyed by code (MapValues, a SimplePredicate's or
 * SimpleSetPredicate's value, CategoricalPredictor) finds an edited string
 * by its text. Only a string whose text the pool lacks has an arena rep, and
 * none of those is a key (keys are document text, interned at load).
 *
 * Two more fields carry what the old storage carried beside its double and
 * that a typed value has no room for; each is read through the accessors
 * only: a number's text (text_: a document constant given its Cast<string>
 * text at load, §3.4, kept for the string readers), and the numeric_text mark
 * (a number or boolean constant held as text in a string slot, §9 "Decided
 * 2026-11-18 after P2.1"). A Value is 32 bytes and trivially copyable: a
 * Sample is copied with memcpy and no string is copied with it.
 */
namespace cpmml {

// The strings of one document (§2.3), shared by every model and segment of it.
struct DocumentStrings {
  types::StringPool pool;
  // Reps made at load that the pool does not index: the text the load gives
  // a number (set_text; a string's text is interned, Value::put_text). They
  // live as long as the document.
  types::RecordArena load_arena;
};

/**
 * Where the strings made on this thread go (§2.3, §5.1: the per-evaluation
 * RecordArena). P5 has no frame yet to own the arena (P7), so a scope on the
 * stack of each entry point does: a load scope (ModelBuilder::build) while the
 * document is read, a scoring scope (score, predict, validate, the entry
 * points of the evaluators, and each OpenMP segment worker) while a record is
 * scored. The outermost scoring scope of a thread uses the thread's arena,
 * from its first string the pool lacks, and empties it when it ends; a scope
 * nested in one of the same document shares its arena, and one of another
 * document has its own. Strings made in a scope die with it, so no Value made while
 * scoring may outlive the entry point that made it (results are text,
 * InternalScore).
 */
class StringScope {
 public:
  struct Loading {};
  struct Scoring {};

  StringScope(Loading, std::shared_ptr<DocumentStrings> document)
      : loading_(std::move(document)), document_(loading_.get()), previous_(current()) {
    if (!document_) throw std::logic_error("StringScope: no document");
    arena_ = &document_->load_arena;
    current() = this;
  }

  StringScope(Scoring, DocumentStrings& document) : document_(&document), previous_(current()) {
    if (previous_ && previous_->document_ == document_) {
      arena_owner_ = previous_->arena_owner_;  // an evaluation nested in this document's: its arena
    } else if (previous_) {
      own_arena_ = std::make_unique<types::RecordArena>();
      arena_ = own_arena_.get();
    }
    // else the outermost scope of the thread: the thread's arena, taken on
    // first use (most records make no string the pool lacks).
    current() = this;
  }

  ~StringScope() {
    current() = previous_;
    if (reset_on_exit_) arena_->reset();
  }

  StringScope(const StringScope&) = delete;
  StringScope& operator=(const StringScope&) = delete;

  // The scope of this thread; a string made outside every scope is a
  // programming error, never a string of some other numbering.
  static StringScope& active() {
    if (!current()) throw std::logic_error("no StringScope: a string Value is made outside a load or an evaluation");
    return *current();
  }

  // The document being loaded on this thread (models keep it to score).
  static std::shared_ptr<DocumentStrings> loading() {
    for (StringScope* s = current(); s; s = s->previous_)
      if (s->loading_) return s->loading_;
    throw std::logic_error("StringScope::loading: no document is being loaded on this thread");
  }

  // The rep of the string `text`: the pooled one (interned while the pool is
  // open, i.e. at load), else one made in this scope's arena.
  const types::StringRep* rep(std::string_view text) {
    types::StringPool& pool = document_->pool;
    if (!pool.frozen()) return pool.intern(text);
    if (const types::StringRep* pooled = pool.find(text)) return pooled;
    return arena().make(text);
  }

  // A rep the pool does not index, in this scope's arena.
  const types::StringRep* unindexed(std::string_view text) { return arena().make(text); }

 private:
  static StringScope*& current() {
    thread_local StringScope* scope = nullptr;
    return scope;
  }
  types::RecordArena& arena() {
    StringScope* owner = arena_owner_;
    if (!owner->arena_) {
      owner->arena_ = &thread_arena();
      owner->reset_on_exit_ = true;
    }
    return *owner->arena_;
  }
  static types::RecordArena& thread_arena() {
    alignas(std::max_align_t) thread_local unsigned char buffer[4096];
    thread_local types::RecordArena arena(buffer, sizeof(buffer));
    return arena;
  }

  std::shared_ptr<DocumentStrings> loading_;
  DocumentStrings* document_;
  types::RecordArena* arena_ = nullptr;   // this scope's own, or null until first use
  StringScope* arena_owner_ = this;       // the scope whose arena this one uses
  std::unique_ptr<types::RecordArena> own_arena_;
  bool reset_on_exit_ = false;
  StringScope* previous_;
};

namespace legacy {
// Switch 2 (see the note): the code of a string the pool lacks.
inline constexpr std::uint32_t kUnseen = types::StringRep::kUnpooled;
}  // namespace legacy

}  // namespace cpmml

class Value;
namespace cpmml::legacy {
inline double code_of(const ::Value& v);  // switch 2, below
}
namespace cpmml {
// B38 (P6.8): the record error of a string that reaches a numeric reader,
// below (Value::as_number).
[[noreturn]] [[gnu::cold]] [[gnu::noinline]] inline void raise_string_as_number(std::string_view reader,
                                                                                std::string_view field,
                                                                                const ::Value& v);
}  // namespace cpmml

/**
 * @class Value
 *
 * The engine's value: a types::Value and what the old storage carried beside
 * it (see the note above). Every reader goes through the accessors.
 */
class Value {
 public:
  Value() = default;
  // A double (and the old storage's every number).
  explicit Value(const double number) : v_(cpmml::types::Double{number}) {}
  explicit Value(const cpmml::types::Integer number) : v_(number) {}
  // A float (XSD float, an IEEE single; P6.2 deleted switch 1).
  explicit Value(const cpmml::types::Float number) : v_(number) {}
  explicit Value(const cpmml::types::Boolean flag) : v_(flag) {}

  // The string `text` in this thread's StringScope: its pooled rep, or one
  // made for it (canonical, see the note).
  static Value string(std::string_view text) {
    Value v;
    v.v_ = cpmml::types::String{cpmml::StringScope::active().rep(text)};
    return v;
  }

  // An integer slot's number as the old storage held it, the double: an
  // Integer when it is an int64 other than -0 (XSD integer has one zero; the
  // old storage kept -0, Q13/D18), else the Double.
  static Value integer_slot(const double number) {
    if (std::trunc(number) == number && number >= -9223372036854775808.0 && number < 9223372036854775808.0 &&
        !(number == 0 && std::signbit(number)))
      return Value(cpmml::types::Integer{static_cast<std::int64_t>(number)});
    return Value(number);
  }

  // A value of `datatype` read from its document text (an attribute, an
  // Array member, a table cell), as the load always read it: a string is
  // the string; a boolean the XSD boolean; a float the XSD float reading (the
  // float closest to the decimal, ±INF beyond the range: XML Schema Part 2
  // §3.2.4, P6.2); a double or integer the xs:double reading (an integer
  // field's "2.5" is the double, §3.4 K1 rule 2). Malformed text throws ParsingException, which at load is a load
  // error. Inputs do not come here (MiningField::parse_input).
  static Value document(std::string_view text, const DataType& datatype);

  // ---- readers
  inline bool is_missing() const { return std::holds_alternative<cpmml::types::Missing>(v_); }

  // The number the old storage held: a number or a boolean (1 or 0) itself,
  // a string its legacy code (switch 2), a missing value double_min().
  inline double as_double() const {
    if (const auto* d = std::get_if<cpmml::types::Double>(&v_)) [[likely]]
      return d->v;
    // A string's code inline too: the string equalities of a tree read it per
    // node (switch 2).
    if (std::holds_alternative<cpmml::types::String>(v_)) return cpmml::legacy::code_of(*this);
    return as_double_other();
  }

  // The number of a numeric reader (a numeric built-in's argument, a
  // NumericPredictor, a PredictorTerm, NormContinuous, Discretize; B38,
  // P6.8): as_double() of a number or a boolean; a string is no number, an
  // invalid record (Q2) naming `reader` and `field`, where as_double() gives
  // its legacy code, the category identity the categorical readers use
  // (switch 2). §2.5: "no arithmetic accepts" string; Regression.html
  // NumericPredictor "Defines a numeric independent variable", PredictorTerm
  // "The type of all fields referenced within PredictorTerm must be
  // continuous"; Transformations.html NormContinuous and Discretize read a
  // numeric field. Missing gives double_min() as as_double() does (callers
  // test is_missing() first).
  inline double as_number(std::string_view reader, std::string_view field) const {
    if (const auto* d = std::get_if<cpmml::types::Double>(&v_)) [[likely]]
      return d->v;
    if (std::holds_alternative<cpmml::types::String>(v_)) [[unlikely]]
      cpmml::raise_string_as_number(reader, field, *this);
    return as_double_other();
  }
  inline bool is_string() const { return std::holds_alternative<cpmml::types::String>(v_); }

  // The text: a string's own, the text a number carries, else empty.
  inline std::string_view text() const {
    const cpmml::types::StringRep* rep = text_rep();
    return rep ? rep->text : std::string_view();
  }
  inline bool has_numeric_text() const { return numeric_text_; }

  // The typed value (for unpack).
  inline const cpmml::types::Value& typed() const { return v_; }

  // The legacy code of a string (switch 2): its rep's pool id, or kUnseen
  // (a string the pool lacks, or a value that is no string).
  inline std::uint32_t legacy_code() const {
    if (const auto* s = std::get_if<cpmml::types::String>(&v_)) return s->rep->id;
    return cpmml::legacy::kUnseen;
  }

  // ---- writers
  // The value becomes the number `number`; a text it carries stays (the old
  // storage wrote its double and left the rest).
  inline void set_double(const double number) {
    if (is_missing()) throw std::logic_error("Value::set_double on a missing value");
    if (const auto* s = std::get_if<cpmml::types::String>(&v_)) text_ = s->rep;
    v_ = cpmml::types::Double{number};
  }
  // The same for the float `number` (a cast to float, §2.6).
  inline void set_float(const float number) {
    if (is_missing()) throw std::logic_error("Value::set_float on a missing value");
    if (const auto* s = std::get_if<cpmml::types::String>(&v_)) text_ = s->rep;
    v_ = cpmml::types::Float{number};
  }
  // The value keeps what it is and carries `new_text` (a string becomes the
  // string of that text, with its canonical rep).
  inline void set_text(std::string_view new_text) {
    if (is_missing()) throw std::logic_error("Value::set_text on a missing value");
    put_text(new_text);
  }
  inline void mark_numeric_text() { numeric_text_ = true; }

  // The number of a numeric_text value: its text, validated at load as a
  // number literal (an XSD double, or Java text: "Infinity") or a boolean
  // literal (1 or 0).
  inline double text_number() const {
    double number = 0;
    if (java::parse(text(), number)) return number;
    bool flag = false;
    if (lexical::parse_boolean(text(), flag)) return flag ? 1 : 0;
    throw std::logic_error("Value::text_number: \"" + std::string(text()) +
                           "\" is not a number or boolean literal");
  }

  // ---- string edits in place: a string becomes the string of its new text
  // (its canonical rep, so it equals that text and not its argument: B2,
  // closed by P6.1); the numeric_text mark goes.
  inline void lowercase() {
    std::string t(text());
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    edit(t);
  }
  inline void uppercase() {
    std::string t(text());
    std::transform(t.begin(), t.end(), t.begin(), ::toupper);
    edit(t);
  }
  inline void substr(const size_t& start, const size_t& size) { edit(std::string(text().substr(start, size))); }
  inline void trim_blanks() {
    std::string t(text());
    edit(::trim(t));
  }

  // std::regex's constructor parses and compiles the pattern. Both of these
  // rebuilt it per record, though the pattern comes from a Constant the
  // document fixes at parse time. The cache is thread_local: building a regex
  // into shared state from several scoring threads is a race, and one model
  // uses a handful of distinct patterns, so per-thread duplication is cheap.
  inline static const std::regex& cached_regex(std::string_view pattern) {
    static thread_local std::unordered_map<std::string, std::regex> cache;
    auto it = cache.find(std::string(pattern));
    if (it == cache.end()) it = cache.emplace(std::string(pattern), std::regex(std::string(pattern))).first;

    return it->second;
  }

  inline Value replace(std::string_view regex, std::string_view replacement) const {
    const std::string input(text());
    return Value::string(std::regex_replace(input, cached_regex(regex), std::string(replacement)));
  }
  inline bool matches(std::string_view pattern) const {
    const std::string input(text());
    return std::regex_match(input, cached_regex(pattern));
  }

 private:
  inline const cpmml::types::StringRep* text_rep() const {
    if (const auto* s = std::get_if<cpmml::types::String>(&v_)) return s->rep;
    return text_;
  }

  // A string's new text is the string of that text: its canonical rep
  // (StringScope::rep, the pooled one when the pool holds the text; at load,
  // where the pool is open, the text is interned, as Value::string does). A
  // number's text is no string identity: a rep the pool does not index, so
  // that it shifts no string code (B18, the order of strings: P6.3).
  inline void put_text(std::string_view new_text) {
    if (std::holds_alternative<cpmml::types::String>(v_))
      v_ = cpmml::types::String{cpmml::StringScope::active().rep(new_text)};
    else
      text_ = cpmml::StringScope::active().unindexed(new_text);
  }

  // An edit of a value without text leaves it as it is (the old storage
  // edited its empty text), but for the mark.
  inline void edit(std::string_view new_text) {
    numeric_text_ = false;
    if (!text_rep()) return;
    put_text(new_text);
  }

  double as_double_other() const;

  cpmml::types::Value v_ = cpmml::types::Missing{};
  const cpmml::types::StringRep* text_ = nullptr;  // a number's text (a String's is its rep)
  bool numeric_text_ = false;
};

static_assert(std::is_trivially_copyable_v<Value> && std::is_trivially_destructible_v<Value>,
              "a Sample is copied with memcpy");
static_assert(sizeof(Value) <= 32);

namespace cpmml::legacy {

// What remains of switch 2 (design P5; P6.1 took its string equality, P6.3
// deleted LegacyCompare, the comparisons by this double: core/compare.h):
// the double as_double() gives a string, its code (Value::legacy_code), -1
// for a string the pool lacks. No comparison reads it; a numeric reader of a
// string does (B38, P6.8).
inline double code_of(const ::Value& v) {
  return v.legacy_code() == kUnseen ? -1.0 : static_cast<double>(v.legacy_code());
}

}  // namespace cpmml::legacy

inline void cpmml::raise_string_as_number(std::string_view reader, std::string_view field, const ::Value& v) {
  throw cpmml::InvalidValueException(std::string(reader) + (field.empty() ? "" : " \"" + std::string(field) + "\"") +
                                     ": the string \"" + std::string(v.text()) +
                                     "\" where a number is expected (§2.5: no arithmetic accepts string; B38)");
}

inline double Value::as_double_other() const {
  switch (static_cast<cpmml::types::TypeId>(v_.index())) {
    case cpmml::types::TypeId::Missing:
      return double_min();
    case cpmml::types::TypeId::Integer:
      return static_cast<double>(std::get<cpmml::types::Integer>(v_).v);
    case cpmml::types::TypeId::Float:
      return static_cast<double>(std::get<cpmml::types::Float>(v_).v);
    case cpmml::types::TypeId::Double:
      return std::get<cpmml::types::Double>(v_).v;
    case cpmml::types::TypeId::Boolean:
      return std::get<cpmml::types::Boolean>(v_).v ? 1.0 : 0.0;
    case cpmml::types::TypeId::String:
      return cpmml::legacy::code_of(*this);
    case cpmml::types::TypeId::Invalid:
      break;
  }
  throw std::logic_error("Value::as_double: an Invalid value is never stored");
}

inline Value Value::document(std::string_view text, const DataType& datatype) {
  switch (datatype.value) {
    case DataType::DataTypeValue::STRING:
      return Value::string(text);
    case DataType::DataTypeValue::BOOLEAN: {
      bool parsed = false;
      if (!lexical::parse_boolean(text, parsed))
        throw cpmml::ParsingException("\"" + std::string(text) + "\" is not a valid boolean (true, false, 1 or 0)");
      return Value(cpmml::types::Boolean{parsed});
    }
    case DataType::DataTypeValue::FLOAT: {
      // XML Schema Part 2 §3.2.4: "A literal in the lexical space
      // representing a decimal number d maps to the normalized value in the
      // value space of float that is closest to d"; XML Schema 1.1 Part 2
      // E.1 floatLexicalMap: "Set nV to floatingPointRound(nV, 24, -149,
      // 104) when nV is not zero. (floatingPointRound may nonetheless return
      // zero, or INF or -INF.)" (types::detail::read_float: the decimal
      // rounded once, to the float; beyond the float range ±INF).
      float parsed = 0;
      if (!cpmml::types::detail::read_float(text, parsed))
        throw cpmml::ParsingException("\"" + std::string(text) + "\" is not a valid float");
      return Value(cpmml::types::Float{parsed});
    }
    case DataType::DataTypeValue::INTEGER:
      return Value::integer_slot(::to_double(text));
    case DataType::DataTypeValue::DOUBLE:
      return Value(::to_double(text));
  }
  throw std::logic_error("Value::document: unknown DataType");
}

// A Value as text, as JPMML writes it (utils/java_format.h): nothing for a
// missing value, the text of a string, Float.toString of a float,
// Double.toString of any other number. A caller that knows the type writes
// java::Number{v.as_double(), kind} instead (OutputField does, from the
// expression's type).
inline std::string java_text(const Value& value) {
  if (value.is_missing()) return std::string();
  if (!value.text().empty()) return std::string(value.text());
  if (const auto* f = std::get_if<cpmml::types::Float>(&value.typed()))
    return java::float_text(static_cast<double>(f->v));
  return java::double_text(value.as_double());
}

#if defined(__cpp_lib_format)
// std::format("{}", value): java_text(value).
template <>
struct std::formatter<Value, char> {
  constexpr auto parse(std::format_parse_context& ctx) {
    auto it = ctx.begin();
    if (it != ctx.end() && *it != '}') throw std::format_error("Value takes no format spec");
    return it;
  }
  template <class FormatContext>
  auto format(const Value& value, FormatContext& ctx) const {
    const std::string text = java_text(value);
    return std::copy(text.begin(), text.end(), ctx.out());
  }
};
#endif

// ---------------------------------------------------------------- adapters
// Between the engine's Value and the P1 typed value (core/types/value.h).
// The Value holds a types::Value (P5), so both are the identity where the
// value is stored in its static type, and the explicit cast where it is not.
//
// unpack<T>(v, static_type) reads v as a value of static_type, the type the
// binder fixed for the expression that produced it. A value stored in that
// type is itself (std::get). Otherwise the number it holds is cast (§2.6,
// types::CastTo), as the old storage, which held every number as a double,
// was read: Double the double (-0, NaN, INF kept); Float its IEEE round to
// nearest (a value of a float slot that is no Float, a double the engine
// computed); Integer the value if
// integral and within int64, else Invalid(NotIntegral) or Invalid(Overflow)
// (Functions.html: "a double expression can not be implicitly cast as an
// integer"); Boolean 1 -> true, 0 -> false, else Invalid(NoCast) (the XSD
// boolean lexical space {true, false, 1, 0}, D14). A String is the value's
// own when it is one; the text of anything else (a number in a string slot
// carries its text), with a rep made by `ctx` (types::make_string), so a
// String static type needs the three-argument form. A missing v is Missing
// whatever its static type.
//
// T is types::Value (any of the above) or the scalar type of static_type
// itself. A scalar cannot hold Missing or Invalid, so unpack<Scalar> of a
// missing v, of a v that is not a value of the type, or with another static
// type is a programming error (std::logic_error), never a default.
//
// pack(x) is the Value of a typed value: a number or boolean itself (a Float
// a Float since P6.2), Missing Value(). A String is the string of its
// text in this thread's StringScope (Value::string: the pooled rep and its
// code, or an arena's and the unseen code), not the rep it came with: the
// binder reads document text with pools of its own (bind::document_constant),
// whose reps and ids are not the document's. Invalid has no Value: the engine
// raises an exception where a value is invalid, so packing one is a
// programming error.
//
// Round trip: pack(unpack(v, t)) is v, accessor for accessor, for every v
// the engine builds as a value of t, but for two: a numeric_text string
// loses the mark; and -0 in an integer or boolean slot packs as +0 (XSD
// integer has one zero).
namespace cpmml {

namespace detail {
// v (not missing) as a value of static_type.
inline types::Value unpack_present(const ::Value& v, const types::TypeId static_type, types::ParseContext* ctx) {
  if (types::type_of(v.typed()) == static_type) return v.typed();
  const types::Double d{v.as_double()};
  switch (static_type) {
    case types::TypeId::Integer:
      return types::CastTo<types::Integer>{nullptr, nullptr}(d);
    case types::TypeId::Float:
      return types::CastTo<types::Float>{nullptr, nullptr}(d);
    case types::TypeId::Double:
      return d;
    case types::TypeId::Boolean:
      return types::CastTo<types::Boolean>{nullptr, nullptr}(d);
    case types::TypeId::String:
      if (!ctx) throw std::logic_error("unpack: a string needs a ParseContext to make its rep");
      return types::String{types::make_string(*ctx, v.text())};
    case types::TypeId::Missing:
    case types::TypeId::Invalid:
      break;
  }
  throw std::logic_error("unpack: Missing and Invalid are not static types");
}

template <class T>
T unpack(const ::Value& v, const types::TypeId static_type, types::ParseContext* ctx) {
  if constexpr (std::is_same_v<T, types::Value>) {
    if (static_type == types::TypeId::Missing || static_type == types::TypeId::Invalid)
      throw std::logic_error("unpack: Missing and Invalid are not static types");
    if (v.is_missing()) return types::Missing{};
    return unpack_present(v, static_type, ctx);
  } else {
    static_assert(types::PmmlParsedType<T>, "unpack<T>: T is types::Value or a P1 scalar type");
    if (static_type != T::type_id)
      throw std::logic_error("unpack<" + std::string(T::name) + ">: the static type is another");
    if (v.is_missing()) throw std::logic_error("unpack<" + std::string(T::name) + ">: the value is missing");
    const types::Value x = unpack_present(v, static_type, ctx);
    if (const T* scalar = std::get_if<T>(&x)) return *scalar;
    throw std::logic_error("unpack<" + std::string(T::name) + ">: " + java::double_text(v.as_double()) +
                           " is not a value of the type");
  }
}
}  // namespace detail

// A number, boolean or missing value (T = types::Value, or Integer, Float,
// Double, Boolean); a String static type needs the form below.
template <class T>
T unpack(const ::Value& v, const types::TypeId static_type) {
  static_assert(!std::is_same_v<T, types::String>, "unpack<String> needs a ParseContext");
  return detail::unpack<T>(v, static_type, nullptr);
}

// Any static type; a String's rep is made by ctx.
template <class T>
T unpack(const ::Value& v, const types::TypeId static_type, types::ParseContext& ctx) {
  return detail::unpack<T>(v, static_type, &ctx);
}

inline ::Value pack(const types::Value& x) {
  if (const auto* i = std::get_if<types::Integer>(&x)) return ::Value(*i);
  if (const auto* f = std::get_if<types::Float>(&x)) return ::Value(*f);
  if (const auto* d = std::get_if<types::Double>(&x)) return ::Value(d->v);
  if (const auto* b = std::get_if<types::Boolean>(&x)) return ::Value(*b);
  if (const auto* s = std::get_if<types::String>(&x)) return ::Value::string(s->text());
  if (types::is_missing(x)) return ::Value();
  throw std::logic_error("pack: an Invalid value has no Value");
}

}  // namespace cpmml

#endif  // CPMML_VALUE_H
