
/*******************************************************************************
 * Copyright 2019 AMADEUS. All rights reserved.
 * Author: Paolo Iannino
 *******************************************************************************/

#ifndef CPMML_NORMCONTINUOUS_H
#define CPMML_NORMCONTINUOUS_H

#include <cstdlib>
#include <utility>

#include "bind/load_scope.h"
#include "expression.h"

/**
 * @class NormContinuous
 *
 * Class representing <a
 * href="http://dmg.org/pmml/v4-4/Transformations.html#xsdElement_NormContinuous">PMML
 * NormContinuous</a>.
 *
 * It is used to normalize input fields through a piecewise linear
 * interpolation.
 */
class NormContinuous : public Expression {
 public:
  bool exist_missingreplacement = false;
  Value mapmissing_to;
  std::string field_name;
  size_t index = std::numeric_limits<size_t>::max();
  OutlierTreatmentMethod outliertreatment_method;
  std::vector<Value> orig;
  std::vector<Value> norm;

  NormContinuous() = default;

  NormContinuous(const XmlNode& node, const unsigned int& output_index, const DataType& output_type,
                 const std::shared_ptr<Indexer>& indexer)
      : Expression(output_index, output_type, indexer),
        exist_missingreplacement(node.exists_attribute("mapMissingTo")),
        // pmml-4-4.xsd: NormContinuous@mapMissingTo and LinearNorm@orig/@norm
        // are NUMBER (xs:double), whatever the enclosing DerivedField's
        // dataType (§3.4, B11: in a string DerivedField they were read as
        // string codes and interpolated).
        mapmissing_to(exist_missingreplacement ? cpmml::bind::number_constant(node, "mapMissingTo", output_type) : Value()),
        field_name(node.required_attribute("field")),
        index(indexer->get_or_set(field_name)),
        outliertreatment_method(OutlierTreatmentMethod::read(node)),
        orig(get_values("orig", node)),
        norm(get_values("norm", node)) {
    if (orig.size() < 2 || orig.size() != norm.size())
      throw cpmml::ParsingException("NormContinuous requires at least two LinearNorm elements");
    inputs.insert(field_name);
  }

  static std::vector<Value> get_values(const std::string& linearnorm_type, const XmlNode& node) {
    std::vector<Value> result;

    for (const auto& linear_norm : node.get_childs("LinearNorm"))
      result.push_back(cpmml::bind::number_attribute(linear_norm, linearnorm_type));

    return result;
  }

  inline unsigned int get_interval(
      const Value& value) const {  // if value == orig then the closest interval will be right and
                                   // the interpolation also (look at formula and ifs to see it)
    unsigned int i, min_i = std::numeric_limits<unsigned int>::min();
    Value min(std::numeric_limits<double>::max(), DataType::DataTypeValue::DOUBLE), a, b;
    for (i = 0; i < (orig.size() - 1); i++) {
      if (value > orig[i] && value < orig[i + 1]) {
        return i;  // correct interval
      } else {
        a = value.diff(orig[i]);
        b = value.diff(orig[i + 1]);
        if (a < min || b < min) {
          min_i = i;
          if (a < min) min = a;
          if (b < min) min = b;
        }
      }
    }

    return min_i;  // nearest interval (see outlier below)
  }

  inline Value interpolate(const Value& value) const {
    unsigned int i = get_interval(value);
    return norm[i] + (value - orig[i]) / (orig[i + 1] - orig[i]) *
                         (norm[i + 1] - norm[i]);  // that's why input and output type must be equal
  }

  inline Value eval(const Sample& sample) const override {
    Value input = sample[index].value;

    if (input.is_missing()) return mapmissing_to;  // missing when no mapMissingTo is given

    if (input < orig.front() || input > orig.back()) {  // outlier
      switch (outliertreatment_method.value) {
        case OutlierTreatmentMethod::OutlierTreatmentMethodValue::AS_IS:
          return interpolate(input);  // extrapolate from the nearest interval
        case OutlierTreatmentMethod::OutlierTreatmentMethodValue::AS_MISSING_VALUES:
          return mapmissing_to;
        case OutlierTreatmentMethod::OutlierTreatmentMethodValue::AS_EXTREME_VALUES:
          return input < orig.front() ? norm.front() : norm.back();
      }
    }

    return interpolate(input);  // no outlier
  }
};

#endif
