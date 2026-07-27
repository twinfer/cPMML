
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef TREE_TABLE_NODE_H
#define TREE_TABLE_NODE_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @class TreeTableNode
 *
 * Basic building block of a TreeTable, it can be seen as a column of the table.
 */
template <class K, class V, class H>
class TreeTableNode {
 public:
  K key;
  V value;
  std::unordered_map<K, std::shared_ptr<TreeTableNode>, H> children;

  TreeTableNode() = default;

  explicit TreeTableNode(const V& value) : value(value) {}

  // Both descents used to copy the remaining keys into a fresh vector at every
  // level -- one allocation per column, per lookup, on the scoring path.
  void add(const std::vector<K>& keys, const V& value, const size_t level = 0) {
    if (level == keys.size()) {
      this->value = value;
      return;
    }
    if (children.find(keys[level]) == children.cend()) children[keys[level]] = std::make_shared<TreeTableNode>();
    children[keys[level]]->add(keys, value, level + 1);
  }

  inline V get(const std::vector<K>& keys, const size_t level = 0) const {
    if (children.size() == 0) return value;
    return children.at(keys[level])->get(keys, level + 1);
  }
};

#endif
