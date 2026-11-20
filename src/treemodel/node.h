
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_NODE_H
#define CPMML_NODE_H

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/fastpredicate.h"
#include "core/predicatebuilder.h"
#include "core/xmlnode.h"
#include "scoredistribution.h"
#include "treescore.h"

/**
 * @class Node
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/TreeModel.html#xsdElement_Node">PMML Node</a>.
 *
 * It is a node of the decision tree, containing a Predicate and a TreeScore.
 * The score represents the prediction associated to a sample matching the
 * predicate.
 */
class Node {
 public:
  std::string id;
  std::string simple_score;
  std::string default_child;
  std::vector<Node> children;
  // Its predicate, in the tree's BoundPredicates (TreeModel::predicates).
  std::uint32_t predicate = 0;
  bool root = false;
  bool leaf = false;
  TreeScore score;

  Node() = default;

  Node(const XmlNode& node, bool root, const PredicateBuilder& predicate_builder, BoundPredicates& predicates,
       const DataType& target_datatype)
      : id(node.get_attribute("id", "")),
        simple_score(node.get_attribute("score", "")),
        default_child(node.get_attribute("defaultChild", "")),
        children(to_nodes(node.get_childs("Node"), predicate_builder, predicates, target_datatype)),
        predicate(predicates.add(predicate_builder.build_from_parent(node))),
        root(root),
        leaf(children.size() == 0),
        score(simple_score, id, target_datatype, node.get_childs("ScoreDistribution")) {
    // pmml-4-4.xsd: recordCount is an optional NUMBER without a default. No
    // scoring rule reads it, so it is checked and not kept: a member here
    // makes every Node, which scoring walks per record, larger.
    (void)node.optional_double_attribute("recordCount");
  };

  inline PredicateResult evaluate(const BoundPredicates& predicates, const Sample& sample) const {
    return predicates.evaluate(predicate, sample);
  }

  inline bool has_score() const { return !simple_score.empty(); }

  static std::vector<Node> to_nodes(const std::vector<XmlNode>& nodes, const PredicateBuilder& predicateBuilder,
                                    BoundPredicates& predicates, const DataType& target_datatype) {
    std::vector<Node> result;
    for (auto node : nodes) result.push_back(Node(node, false, predicateBuilder, predicates, target_datatype));

    return result;
  }
};

#endif
