
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_TREEEVALUATOR_H
#define CPMML_TREEEVALUATOR_H

#include <sstream>
#include <string>

#include "core/datadictionary.h"
#include "core/header.h"
#include "core/internal_evaluator.h"
#include "core/xmlnode.h"
#include "treemodel.h"

/**
 * @class TreeEvaluator
 *
 * Implementation of InternalEvaluator, it is used as a wrapper of TreeModel.
 */
class TreeEvaluator : public InternalEvaluator {
 public:
  explicit TreeEvaluator(const XmlNode& node)
      : InternalEvaluator(node),
        tree(node.get_child("TreeModel"), data_dictionary, transformation_dictionary, indexer) {};

  TreeModel tree;

  inline std::unique_ptr<InternalScore> evaluate(const Input& arguments) const override {
    return tree.score(flatten_input(arguments));
  }

  inline bool validate(const Input& arguments) const override {
    return tree.validate(flatten_input(arguments));
  }

  inline std::string get_target_name() const override { return tree.target_field.name; }
  inline std::string output_name() const override { return tree.output_name(); }
  inline std::string mining_function_name() const override { return tree.mining_function.to_string(); }
};

#endif
