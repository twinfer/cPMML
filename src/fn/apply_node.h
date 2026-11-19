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
 * P4.1 (design commit): today's semantics exactly, including where they
 * depart from that output table (legacy::B22, legacy::D3; kernels.h). It
 * replaces core/builtinfunctions.h's BuiltInFunction, a std::function per
 * Apply called with a std::vector<Value> allocated per record: the arguments
 * are now built in place in an ArgBuffer (inline storage for up to four,
 * §4.3), and the kernel is a function pointer chosen at load.
 *
 * The load errors are the engine's, word for word as BuiltInFunction gave
 * them and in the same order (they are LOAD-ERROR cells of the fixture
 * dumps): the function's name, its arity (E-ARITY, the table's), a string
 * ordered against a number (E-TYPE-ORDER-MIXED, §2.7), then mapMissingTo,
 * defaultValue and invalidValueTreatment.
 */

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
  }

  const Signature& signature() const { return *sig_; }

  inline Value eval(const Sample& sample) const override {
    if constexpr (legacy::D3)
      if (missing_ == MissingRule::LazyIf) return eval_if(sample);

    // 1. The inputs.
    const std::size_t n = arguments_.size();
    ArgBuffer args(n);
    bool missing_input = false;
    for (std::size_t i = 0; i < n; i++)
      missing_input |= args.emplace([&] { return arguments_[i]->eval(sample); }).is_missing();

    // 2. A missing input: mapMissingTo, for every function but isMissing and
    // isNotMissing, which receive it; else, but for the aggregates, which
    // skip it, a missing result, then defaultValue.
    if (missing_input && missing_ != MissingRule::Receive) {
      if (has_map_missing_to_) return map_missing_to_;
      if (missing_ != MissingRule::SkipAggregate) return has_default_ ? default_value_ : Value();
    }

    // 3. The function.
    Value result;
    if (kernel_(bound_, args.data(), n, result) == Status::Invalid) {
      // 4. An invalid result.
      if constexpr (legacy::B22) {
        switch (treatment_.value) {
          case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::RETURN_INVALID:
            throw cpmml::InvalidValueException("evaluating apply function");
          case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_MISSING:
            if (has_map_missing_to_) return map_missing_to_;
            return has_default_ ? default_value_ : Value();
          case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_IS:
          case InvalidValueTreatmentMethod::InvalidValueTreatmentMethodValue::AS_VALUE:  // refused at load
            result = Value();
            break;
        }
      }
    }

    // 5. A missing result: defaultValue.
    if (result.is_missing() && has_default_) return default_value_;
    return result;
  }

 protected:
  void add_argument(std::shared_ptr<Expression> argument) { arguments_.push_back(std::move(argument)); }

 private:
  // The kernel of `sig`, after the load checks of the call: the arity (the
  // table's min_args/max_args, E-ARITY), and for the numeric functions and
  // the comparisons the arguments' static types (bind::expression_type), a
  // string ordered against a number (§2.7, E-TYPE-ORDER-MIXED) and the
  // typing flags of `bound`. The arguments are the element children after
  // Extension* (pmml-4-4.xsd Apply, B25), as the binder counts them.
  static KernelFn instantiate_checked(const XmlNode& apply, const Signature& sig, Bound& bound) {
    using cpmml::bind::TypeId;
    std::size_t n = 0;
    for (pugi::xml_node argument : apply.xml().children())
      if (argument.type() == pugi::node_element && std::strcmp(argument.name(), "Extension") != 0) n++;
    if (!cpmml::bind::arity_fits(sig, n)) throw cpmml::ParsingException(cpmml::bind::arity_message(apply, sig, n));
    const KernelFn kernel = instantiate(sig.op);
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
    bound.integer_division = sig.op == Op::Divide && result == TypeId::Integer;
    bound.integer_power = sig.op == Op::Pow && result == TypeId::Integer;
    return kernel;
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

  // `if` (design commit P2.8b4; §4.2 "if laziness", §4.3 LazyIf; D3, D24),
  // evaluated here while legacy::D3 stands (P4.3 moves the rule into the
  // kernel). BuiltinFunctions.html, if: "Implements IF-THEN-ELSE logic. The
  // ELSE part is optional. If the ELSE part is absent and the boolean value
  // is false then a missing value is returned." Read as a conditional, the
  // inputs are the condition and the branch it takes (§4.2), evaluated in
  // that order; the other branch is not evaluated, so its missing value is
  // not an input. A missing condition is a missing input (D3): mapMissingTo,
  // else a missing result, then defaultValue. A false condition without an
  // ELSE part is a missing result, not a missing input: defaultValue applies
  // and mapMissingTo does not. The condition is true when the double it
  // holds is not 0. JPMML 1.7.7 agrees on every case but the missing
  // condition without mapMissingTo, which it refuses
  // (MissingArgumentException; D3).
  inline Value eval_if(const Sample& sample) const {
    const Value condition = arguments_[0]->eval(sample);
    if (condition.is_missing()) {
      if (has_map_missing_to_) return map_missing_to_;
      return has_default_ ? default_value_ : Value();
    }
    const size_t taken = condition.as_double() != 0.0 ? 1 : 2;
    if (taken >= arguments_.size()) return has_default_ ? default_value_ : Value();
    const Value result = arguments_[taken]->eval(sample);
    if (!result.is_missing()) return result;
    if (has_map_missing_to_) return map_missing_to_;
    return has_default_ ? default_value_ : result;
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
  std::vector<std::shared_ptr<Expression>> arguments_;
};

}  // namespace cpmml::fn

#endif  // CPMML_FN_APPLY_NODE_H
