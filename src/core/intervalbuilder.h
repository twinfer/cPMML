
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_INTERVALBUILDER_H
#define CPMML_INTERVALBUILDER_H

#include <limits>
#include <optional>
#include <string>

#include "predicate.h"
#include "xmlnode.h"

/**
 * @class IntervalBuilder
 *
 * Class used to build the constraints for continous features of the
 * DataDictionary. See also Closure.
 */
class IntervalBuilder {
 public:
  static Predicate build(const XmlNode& node, const size_t& index,
                         const DataType& dataType = DataType::DataTypeValue::DOUBLE) {
    using Op = PredicateOpType::PredicateOpTypeValue;
    std::vector<Predicate> tmp_constraints;

    Closure closure(node);
    // An absent margin means the interval is unbounded on that side.
    const std::optional<std::string> left = node.optional_attribute("leftMargin");
    const std::optional<std::string> right = node.optional_attribute("rightMargin");
    Value left_margin = left ? Value(*left, dataType) : Value(std::numeric_limits<double>::lowest(), dataType);
    Value right_margin = right ? Value(*right, dataType) : Value(std::numeric_limits<double>::max(), dataType);
    switch (closure.value) {
      case Closure::ClosureValue::CLOSED_CLOSED:
        tmp_constraints.push_back(Predicate(index, Op::GREATER_OR_EQUAL, left_margin));
        tmp_constraints.push_back(Predicate(index, Op::LESS_OR_EQUAL, right_margin));
        break;
      case Closure::ClosureValue::OPEN_OPEN:
        tmp_constraints.push_back(Predicate(index, Op::GREATER_THAN, left_margin));
        tmp_constraints.push_back(Predicate(index, Op::LESS_THAN, right_margin));
        break;
      case Closure::ClosureValue::CLOSED_OPEN:
        tmp_constraints.push_back(Predicate(index, Op::GREATER_OR_EQUAL, left_margin));
        tmp_constraints.push_back(Predicate(index, Op::LESS_THAN, right_margin));
        break;
      case Closure::ClosureValue::OPEN_CLOSED:
        tmp_constraints.push_back(Predicate(index, Op::GREATER_THAN, left_margin));
        tmp_constraints.push_back(Predicate(index, Op::LESS_OR_EQUAL, right_margin));
        break;
    }

    return Predicate(tmp_constraints, Op::AND);
  }
};

#endif
