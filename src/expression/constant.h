/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_CONSTANT_H
#define CPMML_CONSTANT_H

#include <cstddef>
#include <string>
#include <string_view>

#include "bind/load_scope.h"
#include "bind/typing.h"
#include "expression.h"

/**
 * @class Constant
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_Constant">PMML
 * Constant</a>.
 *
 * It provides a constant value for other PMML Expressions.
 *
 * Typing (docs/design/typed-core.md §3.4, design commit P2.3; the rules are
 * bind/typing.h's, which the binder applies too). Transformations.html,
 * Constant:
 *
 *   - dataType given ("The dataType of Constant can be optionally
 *     specified"): the text is read in that type, strictly by its lexical
 *     space (XML Schema Part 2 §3.3.13.1, §3.2.2.1), except that for float
 *     and double "These three constants [NaN, INF, -INF] are case
 *     insensitive" (S3). Malformed text is
 *     a load error, E-VALUE-MALFORMED.
 *   - no dataType, an argument of a built-in whose parameter declares one:
 *     "If the ParameterField definition includes a dataType, the Constant
 *     will inherit the dataType specified in the ParameterField" (S2):
 *     if(<Constant>true</Constant>, ...) and not(<Constant>false</Constant>)
 *     are booleans, uppercase(<Constant>1.50</Constant>) is the string "1.50".
 *   - otherwise inferred: "A Constant that consists solely of numbers will be
 *     treated as an integer, if a decimal point is present among the numbers
 *     the Constant will be treated as a double ... A scientific notation
 *     expression like 3.14e+5 or like 3.14E-5 should also be treated as a
 *     double. The presence of any non-numeric characters (other than e, E, -,
 *     + in scientific notation expressions) will result in the Constant being
 *     treated as a string" (B9: 1e10 is a double, not a string past the int
 *     range; NaN, INF, -INF are strings, D10: "if datatype is set to its
 *     default value of "string", then the literal strings will be
 *     signified"). An integer beyond int64 is a load error (§3.4, D4). A
 *     string keeps its text (B8: uppercase(<Constant>abc</Constant>) is ABC,
 *     and the spec's own concat(month, <Constant>-</Constant>, year) keeps
 *     the "-"; the base build kept only the string's code).
 *   - "If the content is empty, then the constant will be interpreted as an
 *     empty string, unless dataType is something besides "string", in which
 *     case, it will be taken as a missing value of the specified type." "If
 *     the missing attribute is true, the constant will be taken as a missing
 *     value, regardless of the content."
 *
 * The content is the element's text, blanks included (Constant extends
 * xs:string, whiteSpace "preserve"): pugixml's default flags drop a
 * blanks-only text, so it is read as the binder reads it (content(),
 * LoadScope::lone_blanks). A blanks-only typed number is therefore malformed,
 * not missing, as in the binder (P1.2; JPMML 1.7.7 refuses it too). Text
 * split by a comment or CDATA section is read up to the first text node, as
 * before P2.3, not as the whole xs:string content.
 *
 * Storage is the old Value (P2): integer, float and double are doubles (a
 * float's text keeps its double reading until P6, B4), a boolean is 1 or 0, a
 * string its code with its text. The value is the Constant's own; the cast to
 * an enclosing DerivedField's or OutputField's dataType is P4.7's.
 */
class Constant : public Expression {
 public:
  bool datatype_exists = false;
  Value value;

  Constant() = default;

  Constant(const XmlNode& node, const unsigned int& output_index, const DataType& output_type,
           const std::shared_ptr<Indexer>& indexer)
      : Expression(output_index, output_type, indexer),
        datatype_exists(node.exists_attribute("dataType")),
        value(read(node)) {}

  inline Value eval(const Sample& sample) const override { return value; }

  // The Constant's value, typed as the class comment says.
  static Value read(const XmlNode& node) {
    using cpmml::bind::TypeId;
    namespace code = cpmml::bind::code;
    const cpmml::bind::StaticType declared = cpmml::bind::declared_type_of(node);  // E-ENUM if not a DATATYPE
    // The temporal types are read as today until P10 (§2.1: the binder maps
    // them to Double "exactly as today").
    if (declared && cpmml::bind::parse_datatype(node.required_attribute("dataType"))->temporal)
      return Value::document(node.value(), DataType::read(node));
    if (node.get_bool_attribute("missing", false)) return Value();

    const std::string text = content(node);
    const cpmml::bind::StaticType type = declared ? declared : inherited_type(node);
    types_parse_context ctx;
    if (type) {
      if (text.empty() && *type != TypeId::String) return Value();
      const cpmml::types::Value v = cpmml::bind::parse_constant(*type, text, ctx.ctx);
      if (cpmml::types::is_invalid(v))
        throw cpmml::ParsingException(std::string(code::kValueMalformed) + ": Constant text \"" + text +
                                      "\" is not a value of type " + std::string(cpmml::bind::type_name(*type)) +
                                      (declared ? std::string(" (§3.4)")
                                                : std::string(" (inherited from the parameter of ") +
                                                      node.xml().parent().attribute("function").value() + ", S2)"));
      if (*type == TypeId::String) return Value::string(text);
      // A float's text keeps its double reading (B4, P6): 0.1 is the double
      // 0.1, as DataType::read stored it; INF and NaN are the same either way.
      if (const auto* f = std::get_if<cpmml::types::Float>(&v)) {
        double d = 0;
        return cpmml::legacy::float_value(lexical::parse_double(text, d) ? d : static_cast<double>(f->v));
      }
      return cpmml::bind::engine_value(v);
    }
    using cpmml::bind::ConstantKind;
    const ConstantKind kind = cpmml::bind::infer_constant(text);
    if (kind == ConstantKind::String) return Value::string(text);
    if (kind == ConstantKind::IntegerOverflow)
      throw cpmml::ParsingException(std::string(code::kValueMalformed) + ": untyped Constant integer \"" + text +
                                    "\" is beyond the 64-bit range (§3.4, D4)");
    return cpmml::bind::engine_value(
        cpmml::bind::parse_constant(kind == ConstantKind::Integer ? TypeId::Integer : TypeId::Double, text, ctx.ctx));
  }

 private:
  struct types_parse_context {
    cpmml::types::StringPool pool;
    cpmml::types::ParseContext ctx{&pool, nullptr, nullptr};
  };

  // S2: the dataType an untyped Constant inherits from the built-in parameter
  // it is an argument of, or nullopt.
  static cpmml::bind::StaticType inherited_type(const XmlNode& node) {
    const pugi::xml_node apply = node.xml().parent();
    if (std::string_view(apply.name()) != "Apply") return std::nullopt;
    std::size_t position = 0;  // among the arguments: the children after Extension* (B25)
    for (pugi::xml_node c = apply.first_child(); c && c != node.xml(); c = c.next_sibling())
      if (c.type() == pugi::node_element && std::string_view(c.name()) != "Extension") position++;
    return cpmml::bind::inherited_constant_type(apply.attribute("function").value(), position);
  }

  // The element's text content. pugixml's default flags (the engine's,
  // core/modelbuilder.h) drop a text child that is only blanks, which the
  // binder keeps (parse_ws_pcdata_single, P1.2); an element with no child is
  // therefore read as the binder's reader reads it (LoadScope::lone_blanks):
  // its blanks, line ends normalised (XML 1.0 §2.11), or "" for <Constant/>
  // and <Constant></Constant>. Comments and processing instructions are not
  // content. (Text split by a comment or CDATA is read up to the first text
  // node, as before P2.3; not this item's.)
  static std::string content(const XmlNode& node) {
    const pugi::xml_node n = node.xml();
    if (n.first_child()) return node.value();
    const cpmml::bind::LoadScope* scope = cpmml::bind::LoadScope::current();
    if (!scope) throw std::logic_error("Constant: built outside a model load");
    return scope->lone_blanks(n);
  }
};

#endif
