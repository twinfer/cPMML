
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_XMLNODE_H
#define CPMML_XMLNODE_H

#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <pugixml.hpp>

#include "options.h"
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

  std::string value() const { return node.child_value(); }

  bool exists_attribute(const std::string &attribute_name) const {
    return static_cast<bool>(node.attribute(attribute_name.c_str()));
  }

  std::string get_attribute(const std::string &attribute_name) const {
    pugi::xml_attribute attr = node.attribute(attribute_name.c_str());
    if (!attr) return "null";
    return attr.value();
  }

  double get_double_attribute(const std::string &attribute_name) const {
    try {
      return std::stod(get_attribute(attribute_name));
    } catch (const std::invalid_argument &) {
      return std::numeric_limits<double>::min();
    }
  }

  bool get_bool_attribute(const std::string &attribute_name) const {
    std::string tmp = get_attribute(attribute_name);
    return to_lower(tmp) == "true" || tmp == "1";
  }

  long get_long_attribute(const std::string &attribute_name) const {
    try {
      return std::stol(get_attribute(attribute_name));
    } catch (const std::invalid_argument &) {
      return std::numeric_limits<long>::max();
    }
  }

  XmlNode get_child(const std::string &child_name) const {
    return XmlNode(node.child(child_name.c_str()));
  }

  XmlNode get_child() const { return XmlNode(node.first_child()); }

  bool exists_child(const std::string &child_name) const {
    return static_cast<bool>(node.child(child_name.c_str()));
  }

  std::vector<XmlNode> get_childs(const std::string &child_name) const {
    std::vector<XmlNode> result;
    for (auto child : node.children(child_name.c_str()))
      result.push_back(XmlNode(child));
    return result;
  }

  std::vector<XmlNode> get_childs() const {
    std::vector<XmlNode> result;
    for (auto child : node.children())
      result.push_back(XmlNode(child));
    return result;
  }

  XmlNode get_child_bypattern(const std::string &pattern) const {
    for (auto child : node.children()) {
      XmlNode child_node(child);
      if (to_lower(child_node.name()).find(to_lower(pattern)) != std::string::npos)
        return child_node;
    }
    return XmlNode();
  }

  XmlNode get_child_bylist(const std::unordered_set<std::string> &list) const {
    for (auto child : node.children()) {
      if (list.find(child.name()) != list.cend())
        return XmlNode(child);
    }
    return XmlNode();
  }

  bool exists_child_bylist(const std::unordered_set<std::string> &list) const {
    for (auto child : node.children()) {
      if (list.find(child.name()) != list.cend())
        return true;
    }
    return false;
  }

  std::vector<XmlNode> get_childs_bypattern(const std::string &pattern) const {
    std::vector<XmlNode> result;
    for (auto child : node.children()) {
      if (to_lower(std::string(child.name())).find(pattern) != std::string::npos)
        result.push_back(XmlNode(child));
    }
    return result;
  }

  std::vector<XmlNode> get_childs_byattribute(const std::string &child_name,
                                               const std::string &attribute_name,
                                               const std::string &attribute_value) const {
    std::vector<XmlNode> result;
    for (auto child : node.children(child_name.c_str())) {
      XmlNode child_node(child);
      if (child_node.get_attribute(attribute_name) == attribute_value)
        result.push_back(child_node);
    }
    return result;
  }

  std::vector<XmlNode> get_childs_bylist(const std::unordered_set<std::string> &list) {
    std::vector<XmlNode> result;
    for (auto child : node.children()) {
      if (list.find(child.name()) != list.cend())
        result.push_back(XmlNode(child));
    }
    return result;
  }

  bool is_empty() const { return node.empty(); }

 private:
  pugi::xml_node node;
};

#endif
