
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
    // One lookup, then use the node it found or made. A NaN key never equals
    // itself, so a second lookup (operator[] after find) inserts a fresh,
    // empty slot and the old code dereferenced its null pointer. A NaN row is
    // stored but unreachable, as a lookup with NaN finds nothing either.
    auto& child = children[keys[level]];
    if (!child) child = std::make_shared<TreeTableNode>();
    child->add(keys, value, level + 1);
  }

  inline V get(const std::vector<K>& keys, const size_t level = 0) const {
    if (children.size() == 0) return value;
    return children.at(keys[level])->get(keys, level + 1);
  }
};

#endif
