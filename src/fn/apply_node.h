/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_FN_APPLY_NODE_H
#define CPMML_FN_APPLY_NODE_H

/**
 * Design note -- the Apply wrapper (§4.3).
 *
 * One class implements Functions.html's handling of an Apply for every
 * built-in: it evaluates the arguments by the function's MissingRule (the
 * table, fn/table.h), short-circuits a missing input to mapMissingTo, calls
 * the kernel (fn/kernels.h) resolved at load, applies invalidValueTreatment
 * to an invalid result, and defaultValue to a missing one. Functions.html,
 * Apply: "The optional attribute mapMissingTo defines the output value for
 * the cases when any of the function inputs are missing. This means that if
 * it is specified and any of the input values of the function are missing
 * then the function is not applied at all and the mapMissingTo value is
 * returned instead"; "when a defaultValue value is provided, the function is
 * applied first and if it produces a missing value the defaultValue is
 * returned instead"; BuiltinFunctions.html: "Except as noted below, any
 * missing inputs to a built-in function will result in a missing value
 * being returned".
 *
 * An invalid result (P4.2, B22), Functions.html, Apply: "This attribute's
 * default value returnInvalid causes the model to return a value indicating
 * an invalid result. On the other hand, the invalidValueTreatment value
 * asMissing replaces the invalid value with a missing value. Note that, in
 * this case, if a defaultValue is also specified then the default value is
 * returned instead of a missing value. Finally, in the context of an Apply,
 * the value asIs is equivalent to returnInvalid as an invalid value cannot
 * be propagated further." So returnInvalid and asIs make the record invalid
 * (cpmml::InvalidValueException naming the function and the field, §4.3),
 * and asMissing gives a missing result, then defaultValue; mapMissingTo is
 * only for a missing input ("the output value for the cases when any of the
 * function inputs are missing"), never for an invalid result.
 *
 * P4.1 (design commit) kept today's semantics exactly, including where they
 * departed from that output table (legacy::B22, removed by P4.2), and `if`
 * evaluated apart (legacy::D3, removed by P4.3: its laziness is now the
 * table's MissingRule::LazyIf, step 1, and its result a kernel). It replaces core/builtinfunctions.h's
 * BuiltInFunction, a std::function per Apply called with a
 * std::vector<Value> allocated per record: the arguments are now built in
 * place in an ArgBuffer (inline storage for up to four, §4.3), and the
 * kernel is a function pointer chosen at load.
 *
 * The load errors are the engine's, word for word as BuiltInFunction gave
 * them and in the same order (they are LOAD-ERROR cells of the fixture
 * dumps): the function's name, its arity (E-ARITY, the table's), a string
 * ordered against a number (E-TYPE-ORDER-MIXED, §2.7), then mapMissingTo,
 * defaultValue and invalidValueTreatment.
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "bind/load_scope.h"
#include "cPMML.h"
#include "core/invalidvaluetreatmentmethod.h"
#include "core/value.h"
#include "core/xmlnode.h"
#include "expression/expression.h"
#include "fn/kernels.h"
#include "fn/table.h"

namespace cpmml::fn {

// The arguments of one call, constructed in place from each argument's
// eval (guaranteed copy elision: no Value is moved), inline up to kInline.
class ArgBuffer {
 public:
  static constexpr std::size_t kInline = 4;

  explicit ArgBuffer(std::size_t capacity)
      : data_(capacity <= kInline ? reinterpret_cast<::Value*>(inline_)
                                  : std::allocator<::Value>().allocate(capacity)),
        capacity_(capacity) {}
  ~ArgBuffer() {
    for (std::size_t i = 0; i < size_; i++) data_[i].~Value();
    if (capacity_ > kInline) std::allocator<::Value>().deallocate(data_, capacity_);
  }
  ArgBuffer(const ArgBuffer&) = delete;
  ArgBuffer& operator=(const ArgBuffer&) = delete;

  // Appends make()'s result, constructed in place.
  template <class Make>
  const ::Value& emplace(Make&& make) {
    ::Value* slot = ::new (static_cast<void*>(data_ + size_)) ::Value(make());
    size_++;
    return *slot;
  }
  const ::Value* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  alignas(::Value) unsigned char inline_[kInline * sizeof(::Value)];
  ::Value* data_;
  std::size_t capacity_;
  std::size_t size_ = 0;
};

// The row of the built-in an Apply names (its `function` attribute), or the
// load error BuiltInFunction gave: a hint for "log" and "ceiling", the
// names that are no built-in, and the built-ins the engine refuses (with the
// list of those it evaluates, in the table's order).
inline const Signature& engine_row(const std::string& function) {
  const std::string_view hint = hint_for(function);
  if (!hint.empty())
    throw cpmml::ParsingException("function \"" + function + "\" is not a PMML built-in: " + std::string(hint));
  const Signature* sig = find(function);
  if (!sig)
    throw cpmml::ParsingException("invalid value \"" + function +
                                  "\" for Apply@function: not a PMML built-in function, and user-defined "
                                  "functions (DefineFunction) are unsupported");
  if (sig->op == Op::None) {
    std::string supported;
    for (const auto& s : kBuiltins) {
      if (s.op == Op::None) continue;
      if (!supported.empty()) supported += ", ";
      supported += s.name;
    }
    throw cpmml::ParsingException("unsupported value \"" + function + "\" for Apply@function; cPMML supports " +
                                  supported);
  }
  return *sig;
}

class ApplyNode : public Expression {
 public:
  ApplyNode() = default;

  // Everything but the arguments' expressions, which the caller (Apply,
  // expression/apply.h) builds and adds with add_argument, in order.
  ApplyNode(const XmlNode& node, const size_t& output_index, const DataType& output_type,
            const std::shared_ptr<Indexer>& indexer)
      : Expression(output_index, output_type, indexer),
        sig_(&engine_row(node.required_attribute("function"))),
        missing_(sig_->missing),
        kernel_(instantiate_checked(node, *sig_, bound_)),
        has_map_missing_to_(node.exists_attribute("mapMissingTo")),
        map_missing_to_(has_map_missing_to_ ? result_constant(node, "mapMissingTo", output_type) : Value()),
        has_default_(node.exists_attribute("defaultValue")),
        default_value_(has_default_ ? result_constant(node, "defaultValue", output_type) : Value()),
        treatment_(InvalidValueTreatmentMethod::read(node)) {
    // asValue names MiningField/@invalidValueReplacement; an Apply has no
    // replacement to give, so the token is refused rather than read as asIs.
    if (treatment_.value == InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_VALUE)
      throw cpmml::ParsingException("Apply: invalidValueTreatment=\"asValue\" is only defined for a MiningField");
    locate(node);
  }

  const Signature& signature() const { return *sig_; }

  inline Value eval(const Sample& sample) const override {
    // 1. The inputs, by the function's MissingRule: every argument, in
    // order; for `if` (LazyIf), the condition, then the argument it takes
    // (adapter::if_taken, fn/kernels.h), the other branch not evaluated.
    const std::size_t n = arguments_.size();
    ArgBuffer args(n);
    bool missing_input = false;
    if (missing_ == MissingRule::LazyIf) {
      const Value& condition = args.emplace([&] { return arguments_[0]->eval(sample); });
      if (condition.is_missing()) {
        missing_input = true;
      } else if (const std::size_t taken = adapter::if_taken(condition, n); taken < n) {
        missing_input = args.emplace([&] { return arguments_[taken]->eval(sample); }).is_missing();
      }
    } else {
      for (std::size_t i = 0; i < n; i++)
        missing_input |= args.emplace([&] { return arguments_[i]->eval(sample); }).is_missing();
    }

    // 2. A missing input: mapMissingTo, for every function but isMissing and
    // isNotMissing, which receive it; else a missing result, then
    // defaultValue: at once for the functions that propagate it, from the
    // kernel for the aggregates, which skip it, and for `if`, whose result
    // is then its missing input (if_then_else).
    if (missing_input && missing_ != MissingRule::Receive) {
      if (has_map_missing_to_) return map_missing_to_;
      if (missing_ == MissingRule::Propagate) return has_default_ ? default_value_ : Value();
    }

    // 3. The function, over the inputs.
    Value result;
    if (kernel_(bound_, args.data(), args.size(), result) == Status::Invalid) {
      // 4. An invalid result: asMissing "replaces the invalid value with a
      // missing value" (then step 5's defaultValue, not mapMissingTo: no
      // input is missing); returnInvalid, and asIs, which "in the context of
      // an Apply ... is equivalent to returnInvalid", make the record
      // invalid. (asValue is refused at load.)
      if (treatment_.value != InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_MISSING)
        raise_invalid();
      result = Value();
    }

    // 5. A missing result: defaultValue.
    if (result.is_missing() && has_default_) return default_value_;
    return result;
  }

 protected:
  void add_argument(std::shared_ptr<Expression> argument) { arguments_.push_back(std::move(argument)); }

 private:
  // The kernel of `sig`, after the load checks of the call: the arity (the
  // table's min_args/max_args, E-ARITY), for `if` the condition's static type
  // and for not, and, or every argument's (boolean, E-TYPE-BOOLEAN-MIXED,
  // B53: P4.3c, P4.8), and for the numeric functions and
  // the comparisons the arguments' static types (bind::expression_type), a
  // string ordered against a number (§2.7, E-TYPE-ORDER-MIXED) and the
  // typing flags of `bound`; for concat the static type each argument is
  // cast to string from (Bound::argument_type, P4.6). The arguments are the
  // element children after Extension* (pmml-4-4.xsd Apply, B25), as the
  // binder counts them.
  static KernelFn instantiate_checked(const XmlNode& apply, const Signature& sig, Bound& bound) {
    using cpmml::bind::TypeId;
    std::size_t n = 0;
    for (pugi::xml_node argument : apply.xml().children())
      if (argument.type() == pugi::node_element && std::strcmp(argument.name(), "Extension") != 0) n++;
    if (!cpmml::bind::arity_fits(sig, n)) throw cpmml::ParsingException(cpmml::bind::arity_message(apply, sig, n));
    const KernelFn kernel = instantiate(sig.op);
    if (sig.op == Op::If) {
      // B53 (P4.3c): the condition is a boolean parameter; a condition of
      // another static type is a load error (bind::if_condition_message).
      // Unknown (nullopt) is left to the binder's report, as for the other
      // load checks here.
      pugi::xml_node condition = apply.xml().first_child();
      while (condition.type() != pugi::node_element || std::strcmp(condition.name(), "Extension") == 0)
        condition = condition.next_sibling();
      const cpmml::bind::StaticType type = cpmml::bind::expression_type(condition);
      if (type && *type != TypeId::Boolean)
        throw cpmml::ParsingException(cpmml::bind::if_condition_message(apply, *type));
      return kernel;
    }
    if (sig.op == Op::Not || sig.op == Op::And || sig.op == Op::Or) {
      // B53 (P4.8): not declares a boolean parameter, and and/or evaluate
      // "two or more boolean values"; an argument of another static type is a
      // load error, as for if's condition (D36). Unknown (nullopt) is left to
      // the binder's report.
      std::size_t i = 0;
      for (pugi::xml_node argument : apply.xml().children()) {
        if (argument.type() != pugi::node_element || std::strcmp(argument.name(), "Extension") == 0) continue;
        i++;
        const cpmml::bind::StaticType type = cpmml::bind::expression_type(argument);
        if (type && *type != TypeId::Boolean)
          throw cpmml::ParsingException(boolean_argument_message(apply, sig, i, *type));
      }
      return kernel;
    }
    if (sig.op == Op::Concat) {
      // P4.6 (B23): each argument is cast to string from its static type.
      for (pugi::xml_node argument : apply.xml().children())
        if (argument.type() == pugi::node_element && std::strcmp(argument.name(), "Extension") != 0)
          bound.argument_type.push_back(adapter::text_type(cpmml::bind::expression_type(argument)));
      return kernel;
    }
    const bool comparison = compares(sig.op);
    if (!numeric(sig.op) && !comparison) return kernel;
    std::vector<cpmml::bind::StaticType> argument_types;
    argument_types.reserve(n);
    for (pugi::xml_node argument : apply.xml().children())
      if (argument.type() == pugi::node_element && std::strcmp(argument.name(), "Extension") != 0)
        argument_types.push_back(cpmml::bind::expression_type(argument));
    bound.float_argument.resize(n);
    for (std::size_t i = 0; i < n; i++)
      if (argument_types[i] == TypeId::Float) {
        bound.float_argument[i] = 1;
        bound.narrow_arguments = true;
      }
    if (comparison) {
      // §2.7 (design commit P2.7): string x number ordering is a load error.
      if (cpmml::bind::is_ordering(sig.name))
        for (std::size_t i = 1; i < argument_types.size(); i++)
          if (cpmml::bind::order_mixed(argument_types[0], argument_types[i]))
            throw cpmml::ParsingException(cpmml::bind::order_mixed_message(apply, std::string(sig.name),
                                                                           *argument_types[0], *argument_types[i]));
      return kernel;
    }
    const cpmml::bind::StaticType result = cpmml::bind::apply_result_type(sig, argument_types).type;
    bound.float_result = result == TypeId::Float;
    // P4.5 (D4): an integer result has its int64 kernel.
    return instantiate(sig.op, result == TypeId::Integer);
  }

  // The load error of a not, and or or argument (position `i`, from 1) whose
  // static type is not boolean (design item P4.8, B53), in the words of
  // bind::if_condition_message (P4.3c) and of the binder's report of the same
  // check (Binder::check_param, fn::Param::Boolean), with each function's
  // basis: 'E-TYPE-BOOLEAN-MIXED: Apply "and" in RegressionModel: argument 2
  // of and is integer, where the parameter is declared boolean (...)'.
  static std::string boolean_argument_message(const XmlNode& apply, const Signature& sig, std::size_t i,
                                              cpmml::bind::TypeId type) {
    const std::string name(sig.name);
    const char* basis = sig.op == Op::Not
                            ? "BuiltinFunctions.html: <ParameterField name=\"x\" dataType=\"boolean\"/>"
                            : "BuiltinFunctions.html: and, or \"Evaluate the results of two or more boolean values\"";
    return std::string(cpmml::bind::code::kTypeBooleanMixed) + ": " + cpmml::bind::apply_label(apply, name) +
           ": argument " + std::to_string(i) + " of " + name + " is " + std::string(cpmml::bind::type_name(type)) +
           ", where the parameter is declared boolean (" + basis +
           "; Functions.html: boolean \"can not be mixed with other types\", Q7)";
  }

  // The field whose expression holds the Apply, for the record error of an
  // invalid result (raise_invalid): the nearest enclosing DerivedField or
  // OutputField; an Apply outside both (a ComplexPartialScore, a
  // DefineFunction) names its nearest non-Apply ancestor element instead.
  // Only the name is kept at load (it fits the short-string buffer of most
  // field names); the message is built when it is raised.
  void locate(const XmlNode& node) {
    pugi::xml_node fallback;
    for (pugi::xml_node n = node.xml().parent(); n && n.type() == pugi::node_element; n = n.parent()) {
      if (std::strcmp(n.name(), "DerivedField") == 0 || std::strcmp(n.name(), "OutputField") == 0) {
        where_kind_ = n.name()[0] == 'D' ? Where::DerivedField : Where::OutputField;
        where_name_ = n.attribute("name").value();
        return;
      }
      if (!fallback && std::strcmp(n.name(), "Apply") != 0) fallback = n;
    }
    where_kind_ = Where::Element;
    where_name_ = fallback ? fallback.name() : "the document";
  }

  // §4.3 raise_invalid: an invalid result under returnInvalid or asIs makes
  // the record invalid, naming the function and the field, e.g. 'Apply
  // "pow" in DerivedField "p": invalid result
  // (invalidValueTreatment="returnInvalid")'.
  [[noreturn]] [[gnu::noinline]] [[gnu::cold]] void raise_invalid() const {
    std::string where;
    switch (where_kind_) {
      case Where::DerivedField:
        where = "DerivedField \"" + where_name_ + "\"";
        break;
      case Where::OutputField:
        where = "OutputField \"" + where_name_ + "\"";
        break;
      case Where::Element:
        where = where_name_;
        break;
    }
    throw cpmml::InvalidValueException("Apply \"" + std::string(sig_->name) + "\" in " + where +
                                       ": invalid result (invalidValueTreatment=\"" + treatment_.to_string() + "\")");
  }

  // §3.4: Apply "mapMissingTo, defaultValue (xs:string) parsed as the
  // **Apply's** result type", not the enclosing DerivedField's (B11): they
  // stand for the function's result, so they are kept in its representation,
  // as the values the function returns are. The result type is the binder's
  // (design commit P2.4, bind::expression_type: the function table's result
  // rule over the arguments' static types). Where the binder does not know
  // it (arithmetic over a boolean or string operand, Q7, reported in report
  // mode and not yet a load error) the text is read as a double, the type the
  // function computes in. Stored as cpmml::bind::slot_constant says: `concat`
  // with mapMissingTo="5" in a double DerivedField gives the double 5, as
  // does `if` with defaultValue="5" over an untyped MapValues; `+` with
  // mapMissingTo="5" in a string DerivedField gives the string "5" that keeps
  // its number 5 (decided 2026-11-18, §9; JPMML 1.7.7 and the base build
  // agree).
  static Value result_constant(const XmlNode& node, const std::string& attribute, const DataType& slot) {
    using cpmml::bind::TypeId;
    const TypeId type = cpmml::bind::expression_type(node).value_or(TypeId::Double);
    return cpmml::bind::slot_constant(node, attribute, node.required_attribute(attribute), type, slot);
  }

  const Signature* sig_ = nullptr;
  MissingRule missing_ = MissingRule::Propagate;
  Bound bound_;
  KernelFn kernel_ = nullptr;
  bool has_map_missing_to_ = false;
  Value map_missing_to_;
  bool has_default_ = false;
  Value default_value_;
  InvalidValueTreatmentMethod treatment_;
  enum class Where : std::uint8_t { DerivedField, OutputField, Element };
  Where where_kind_ = Where::Element;
  std::string where_name_;
  std::vector<std::shared_ptr<Expression>> arguments_;
};

}  // namespace cpmml::fn

#endif  // CPMML_FN_APPLY_NODE_H
