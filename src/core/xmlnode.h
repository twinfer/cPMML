
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_XMLNODE_H
#define CPMML_XMLNODE_H

#include <optional>
#include <pugixml.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "utils/utils.h"

/**
 * @class XmlNode
 *
 * Non-owning wrapper of pugi::xml_node, implementing utility functions used
 * throughout cPMML for navigating and querying PMML documents.
 */
class XmlNode {
 public:
  XmlNode() = default;

  explicit XmlNode(pugi::xml_node node) : node(node) {}

  std::string name() const { return node.name(); }

  std::string parent_name() const { return node.parent().name(); }

  std::string value() const { return node.child_value(); }

  bool exists_attribute(const std::string& attribute_name) const {
    return static_cast<bool>(node.attribute(attribute_name.c_str()));
  }

  // There is no one-argument get_attribute: it answered the text "null" for an
  // absent attribute, which a document can also hold (CODE_REVIEW.md R1). A
  // caller says what absence means: required_attribute (use="required"),
  // get_attribute(name, default) (the XSD default, or "" where the caller
  // keeps "absent" apart with exists_attribute), or optional_attribute.
  std::string required_attribute(const std::string& attribute_name) const { return required(attribute_name).value(); }

  std::optional<std::string> optional_attribute(const std::string& attribute_name) const {
    pugi::xml_attribute attr = node.attribute(attribute_name.c_str());
    if (!attr) return std::nullopt;
    return std::string(attr.value());
  }

  std::string get_attribute(const std::string& attribute_name, const std::string& default_value) const {
    pugi::xml_attribute attr = node.attribute(attribute_name.c_str());
    if (!attr) return default_value;
    return attr.value();
  }

  // ---- typed attributes ------------------------------------------------------
  //
  // Each reads the attribute through the strict lexical rules (lexical.h). An
  // attribute that is present but malformed is a load error in every one of
  // them -- coefficient="3abc" used to read as 3, and coefficient="abc" as the
  // default or a sentinel, so a corrupt document loaded and scored. They
  // differ only in what an absent attribute means:
  //   required_*            absent is a load error (use="required");
  //   get_*(name, default)  absent takes the PMML default;
  //   optional_*            absent is std::nullopt (no default in the XSD and
  //                         absence means something to the reader).
  // There is no one-argument get_*: it answered a sentinel (DBL_MIN, LONG_MAX)
  // that a document can also hold (CODE_REVIEW.md R1).

  double required_double_attribute(const std::string& attribute_name) const {
    return parse_double(required(attribute_name), attribute_name);
  }

  double get_double_attribute(const std::string& attribute_name, const double default_value) const {
    pugi::xml_attribute attr = node.attribute(attribute_name.c_str());
    return attr ? parse_double(attr, attribute_name) : default_value;
  }

  std::optional<double> optional_double_attribute(const std::string& attribute_name) const {
    pugi::xml_attribute attr = node.attribute(attribute_name.c_str());
    if (!attr) return std::nullopt;
    return parse_double(attr, attribute_name);
  }

  long required_long_attribute(const std::string& attribute_name) const {
    return parse_long(required(attribute_name), attribute_name);
  }

  long get_long_attribute(const std::string& attribute_name, const long default_value) const {
    pugi::xml_attribute attr = node.attribute(attribute_name.c_str());
    return attr ? parse_long(attr, attribute_name) : default_value;
  }

  std::optional<long> optional_long_attribute(const std::string& attribute_name) const {
    pugi::xml_attribute attr = node.attribute(attribute_name.c_str());
    if (!attr) return std::nullopt;
    return parse_long(attr, attribute_name);
  }

  bool required_bool_attribute(const std::string& attribute_name) const {
    return parse_bool(required(attribute_name), attribute_name);
  }

  // There is no sentinel overload: every boolean PMML attribute has a default,
  // and the old one answered false for an absent attribute and for "yes".
  bool get_bool_attribute(const std::string& attribute_name, const bool default_value) const {
    pugi::xml_attribute attr = node.attribute(attribute_name.c_str());
    return attr ? parse_bool(attr, attribute_name) : default_value;
  }

  XmlNode get_child(const std::string& child_name) const { return XmlNode(node.child(child_name.c_str())); }

  XmlNode get_child() const { return XmlNode(node.first_child()); }

  bool exists_child(const std::string& child_name) const { return static_cast<bool>(node.child(child_name.c_str())); }

  std::vector<XmlNode> get_childs(const std::string& child_name) const {
    std::vector<XmlNode> result;
    for (auto child : node.children(child_name.c_str())) result.push_back(XmlNode(child));
    return result;
  }

  std::vector<XmlNode> get_childs() const {
    std::vector<XmlNode> result;
    for (auto child : node.children()) result.push_back(XmlNode(child));
    return result;
  }

  XmlNode get_child_bylist(const std::unordered_set<std::string>& list) const {
    for (auto child : node.children()) {
      if (list.find(child.name()) != list.cend()) return XmlNode(child);
    }
    return XmlNode();
  }

  bool exists_child_bylist(const std::unordered_set<std::string>& list) const {
    for (auto child : node.children()) {
      if (list.find(child.name()) != list.cend()) return true;
    }
    return false;
  }

  std::vector<XmlNode> get_childs_byattribute(const std::string& child_name, const std::string& attribute_name,
                                              const std::string& attribute_value) const {
    std::vector<XmlNode> result;
    for (auto child : node.children(child_name.c_str())) {
      XmlNode child_node(child);
      if (child_node.optional_attribute(attribute_name) == attribute_value) result.push_back(child_node);
    }
    return result;
  }

  std::vector<XmlNode> get_childs_bylist(const std::unordered_set<std::string>& list) {
    std::vector<XmlNode> result;
    for (auto child : node.children()) {
      if (list.find(child.name()) != list.cend()) result.push_back(XmlNode(child));
    }
    return result;
  }

  bool is_empty() const { return node.empty(); }

  // The underlying pugixml node, for load-time lookups across the document
  // (BuiltInFunction::field_kind finds the dataType a FieldRef names).
  pugi::xml_node xml() const { return node; }

 private:
  pugi::xml_node node;

  pugi::xml_attribute required(const std::string& attribute_name) const {
    pugi::xml_attribute attr = node.attribute(attribute_name.c_str());
    if (!attr)
      throw cpmml::ParsingException(std::string(node.name()) + ": required attribute " + attribute_name +
                                    " is missing");
    return attr;
  }

  [[noreturn]] void malformed(const pugi::xml_attribute& attr, const std::string& attribute_name,
                              const char* type) const {
    throw cpmml::ParsingException(std::string(node.name()) + "@" + attribute_name + ": \"" + attr.value() +
                                  "\" is not a valid " + type);
  }

  double parse_double(const pugi::xml_attribute& attr, const std::string& attribute_name) const {
    double parsed = 0.0;
    if (!lexical::parse_double(attr.value(), parsed)) malformed(attr, attribute_name, "double");
    return parsed;
  }

  long parse_long(const pugi::xml_attribute& attr, const std::string& attribute_name) const {
    long parsed = 0;
    if (!lexical::parse_long(attr.value(), parsed)) malformed(attr, attribute_name, "integer");
    return parsed;
  }

  bool parse_bool(const pugi::xml_attribute& attr, const std::string& attribute_name) const {
    bool parsed = false;
    if (!lexical::parse_boolean(attr.value(), parsed)) malformed(attr, attribute_name, "boolean (true, false, 1 or 0)");
    return parsed;
  }
};

#endif
