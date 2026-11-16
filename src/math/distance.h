
/*******************************************************************************
 * Shared distance and similarity metrics for ClusteringModel and KNN.
 *******************************************************************************/

#ifndef CPMML_DISTANCE_H
#define CPMML_DISTANCE_H

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/xmlnode.h"
#include "utils/utils.h"

/**
 * Per-field comparison functions used by both ClusteringModel and
 * NearestNeighborModel.  These correspond to the PMML ComparisonMeasure
 * compareFunction attribute values.
 */
enum class CompareFunc { ABS_DIFF, GAUSS_SIM, DELTA, EQUAL };

/**
 * One field of a distance computation, as declared by a PMML ClusteringField or
 * a KNNInput. Both carry field, compareFunction and fieldWeight, and were
 * transcribed into two structs that differed only in the one attribute
 * ClusteringField adds -- similarityScale, which KNNInput does not have and
 * which stays at its default of 1 there.
 */
struct ComparisonField {
  std::string field_name;
  size_t field_index = 0;
  CompareFunc compare_func = CompareFunc::ABS_DIFF;
  double field_weight = 1.0;
  double similarity_scale = 1.0;
};

// COMPARE-FUNCTION is spelled absDiff / gaussSim / delta / equal. This used to
// require the caller to lower-case first, and NearestNeighborModel did not:
// "gaussSim" matched nothing and fell through to the absDiff fallback, so a
// KNNInput asking for a Gaussian similarity silently got an absolute
// difference. It lower-cases its own argument now.
inline CompareFunc parse_compare_func(const std::string& raw, CompareFunc fallback = CompareFunc::ABS_DIFF) {
  const std::string s = to_lower(raw);
  if (s == "absdiff") return CompareFunc::ABS_DIFF;
  if (s == "gausssim") return CompareFunc::GAUSS_SIM;
  if (s == "delta") return CompareFunc::DELTA;
  if (s == "equal") return CompareFunc::EQUAL;
  return fallback;  // absent/unknown → the model-level default
}

// Every compare function returns a DISSIMILARITY (0 = identical), so that
// compute_distance is uniformly "smaller = closer" for ranking. Similarity
// compare functions from the spec are therefore flipped: gaussSim's
// c(x,y) = exp(-ln(2)·(x−y)²/s²) becomes 1−c (s is the PMML similarityScale
// attribute); equal's 1-if-equal becomes 0-if-equal.
inline double field_compare(double xi, double yi, CompareFunc cf, double scale = 1.0) {
  switch (cf) {
    case CompareFunc::ABS_DIFF:
      return std::abs(xi - yi);
    case CompareFunc::GAUSS_SIM: {
      const double d = xi - yi;
      const double s2 = scale * scale;
      return 1.0 - std::exp(-std::log(2.0) * d * d / (s2 > 0 ? s2 : 1.0));
    }
    case CompareFunc::DELTA:
      return (xi != yi) ? 1.0 : 0.0;
    case CompareFunc::EQUAL:
      return (xi != yi) ? 1.0 : 0.0;
  }
  return std::abs(xi - yi);
}

/**
 * Distance metrics supported by PMML ComparisonMeasure.
 * CENTER_BASED metrics (pick min distance): EUCLIDEAN, SQUARED_EUCLIDEAN,
 *   CITYBLOCK, CHEBYSHEV, MINKOWSKI.
 * BINARY metrics: SIMPLE_MATCHING, JACCARD, TANIMOTO, BINARY_SIMILARITY.
 */
enum class DistanceMetric {
  EUCLIDEAN,
  SQUARED_EUCLIDEAN,
  CITYBLOCK,
  CHEBYSHEV,
  MINKOWSKI,
  SIMPLE_MATCHING,
  JACCARD,
  TANIMOTO,
  BINARY_SIMILARITY
};

/**
 * Generic weighted distance computation between a query vector and a row
 * stored in a flat double array.
 *
 * @tparam GetQuery   callable(size_t i) -> double  (query feature i)
 * @tparam GetRow     callable(size_t i) -> double  (train/centroid feature i)
 * @tparam GetWeight  callable(size_t i) -> double  (field weight i)
 * @tparam GetCF      callable(size_t i) -> CompareFunc
 * @tparam GetScale   callable(size_t i) -> double  (similarityScale for gaussSim)
 */
template <typename GetQuery, typename GetRow, typename GetWeight, typename GetCF, typename GetScale>
double compute_distance(DistanceMetric metric, size_t n, double minkowski_p, GetQuery q, GetRow r, GetWeight w,
                        GetCF cf, GetScale sc) {
  switch (metric) {
    case DistanceMetric::EUCLIDEAN: {
      double sum = 0.0;
      for (size_t i = 0; i < n; i++) {
        double d = field_compare(q(i), r(i), cf(i), sc(i));
        sum += w(i) * d * d;
      }
      return std::sqrt(sum);
    }
    case DistanceMetric::SQUARED_EUCLIDEAN: {
      double sum = 0.0;
      for (size_t i = 0; i < n; i++) {
        double d = field_compare(q(i), r(i), cf(i), sc(i));
        sum += w(i) * d * d;
      }
      return sum;
    }
    case DistanceMetric::CITYBLOCK: {
      double sum = 0.0;
      for (size_t i = 0; i < n; i++) sum += w(i) * field_compare(q(i), r(i), cf(i), sc(i));
      return sum;
    }
    case DistanceMetric::CHEBYSHEV: {
      double mx = 0.0;
      for (size_t i = 0; i < n; i++) mx = std::max(mx, w(i) * field_compare(q(i), r(i), cf(i), sc(i)));
      return mx;
    }
    case DistanceMetric::MINKOWSKI: {
      double sum = 0.0;
      for (size_t i = 0; i < n; i++) sum += w(i) * std::pow(field_compare(q(i), r(i), cf(i), sc(i)), minkowski_p);
      return std::pow(sum, 1.0 / minkowski_p);
    }
    case DistanceMetric::SIMPLE_MATCHING:
    case DistanceMetric::JACCARD:
    case DistanceMetric::TANIMOTO:
    case DistanceMetric::BINARY_SIMILARITY:
      // The binary measures are similarities over 0/1 counts, with their own
      // parameters and undefined cases: binary_measure_similarity below. This
      // used to score them here, binarySimilarity as simpleMatching.
      throw std::logic_error("compute_distance: binary measures are scored by binary_measure_similarity");
  }
  return 0.0;
}

/**
 * The p of a ComparisonMeasure <minkowski> element, shared by ClusteringModel
 * and NearestNeighborModel. The schema declares p-parameter use="required" and
 * the ComparisonMeasure text restricts it to p-parameter>0, so a missing,
 * non-numeric, non-positive or non-finite value is a load error rather than a
 * default. JPMML 1.7.7 agrees for a missing attribute and a negative value; it
 * accepts p-parameter="0" and then scores every neighbour at an infinite
 * distance, which the spec does not allow.
 */
inline double parse_minkowski_p(const XmlNode& minkowski) {
  if (!minkowski.exists_attribute("p-parameter")) {
    const std::string hint = minkowski.exists_attribute("p") ? " (the attribute is p-parameter, not p)" : "";
    throw cpmml::ParsingException("minkowski: required attribute p-parameter is missing" + hint);
  }
  const std::string text = minkowski.get_attribute("p-parameter");
  const double p = to_double(text);
  if (!std::isfinite(p) || !(p > 0.0))
    throw cpmml::ParsingException("minkowski: p-parameter must be a finite number > 0, got \"" + text + "\"");
  return p;
}

/**
 * The parameters of a ComparisonMeasure <binarySimilarity> element:
 *
 *   s = (c11*a11 + c10*a10 + c01*a01 + c00*a00) /
 *       (d11*a11 + d10*a10 + d01*a01 + d00*a00)
 *
 * where aXY counts the fields with x = X and y = Y. The schema declares all
 * eight c??-parameter / d??-parameter attributes use="required" and gives no
 * default, so a missing or non-numeric one is a load error, as in JPMML 1.7.7
 * ("Required attribute binarySimilarity@c00-parameter is not defined").
 */
struct BinarySimilarityParams {
  double c00 = 0.0, c01 = 0.0, c10 = 0.0, c11 = 0.0;
  double d00 = 0.0, d01 = 0.0, d10 = 0.0, d11 = 0.0;
};

inline BinarySimilarityParams parse_binary_similarity(const XmlNode& node) {
  const auto read = [&](const char* name) {
    if (!node.exists_attribute(name))
      throw cpmml::ParsingException(std::string("binarySimilarity: required attribute ") + name + " is missing");
    const std::string text = node.get_attribute(name);
    double v = 0.0;
    try {
      v = to_double(text);
    } catch (const cpmml::ParsingException&) {
      v = std::numeric_limits<double>::quiet_NaN();
    }
    if (!std::isfinite(v))
      throw cpmml::ParsingException(std::string("binarySimilarity: ") + name + " must be a finite number, got \"" +
                                    text + "\"");
    return v;
  };
  BinarySimilarityParams p;
  p.c00 = read("c00-parameter");
  p.c01 = read("c01-parameter");
  p.c10 = read("c10-parameter");
  p.c11 = read("c11-parameter");
  p.d00 = read("d00-parameter");
  p.d01 = read("d01-parameter");
  p.d10 = read("d10-parameter");
  p.d11 = read("d11-parameter");
  return p;
}

inline bool is_binary_metric(DistanceMetric metric) {
  return metric == DistanceMetric::SIMPLE_MATCHING || metric == DistanceMetric::JACCARD ||
         metric == DistanceMetric::TANIMOTO || metric == DistanceMetric::BINARY_SIMILARITY;
}

/**
 * The similarity s of a query x and a row y under one of the binary measures
 * of ComparisonMeasure (ClusteringModel.html, 4.4.1), over the counts aXY of
 * fields with x = X and y = Y:
 *
 *   simpleMatching    (a11 + a00) / (a11 + a10 + a01 + a00)
 *   jaccard           a11 / (a11 + a10 + a01)
 *   tanimoto          (a11 + a00) / (a11 + 2*(a10 + a01) + a00)
 *   binarySimilarity  (c11*a11 + c10*a10 + c01*a01 + c00*a00) /
 *                     (d11*a11 + d10*a10 + d01*a01 + d00*a00)
 *
 * `params` is read for binarySimilarity only. NaN, for the caller to turn into
 * no prediction, where s is undefined, as JPMML 1.7.7 throws there:
 *  - a value other than 0 or 1 on either side, which the counts do not cover
 *    (JPMML: "Expected false or true, got 0.7"). Values used to be read as 1
 *    when > 0.5;
 *  - a zero denominator (JPMML: "Undefined result"): jaccard for an all-zero
 *    query against an all-zero row, which used to score as s = 1, and
 *    binarySimilarity wherever the d's of the present counts sum to 0.
 * fieldWeight and compareFunction take no part, as in JPMML.
 */
template <typename GetQuery, typename GetRow>
double binary_measure_similarity(DistanceMetric metric, const BinarySimilarityParams& params, size_t n, GetQuery q,
                                 GetRow r) {
  constexpr double undefined = std::numeric_limits<double>::quiet_NaN();
  double a00 = 0.0, a01 = 0.0, a10 = 0.0, a11 = 0.0;
  for (size_t i = 0; i < n; i++) {
    const double xi = q(i), yi = r(i);
    if ((xi != 0.0 && xi != 1.0) || (yi != 0.0 && yi != 1.0)) return undefined;
    if (xi == 1.0)
      (yi == 1.0 ? a11 : a10)++;
    else
      (yi == 1.0 ? a01 : a00)++;
  }

  double num = 0.0, den = 0.0;
  switch (metric) {
    case DistanceMetric::SIMPLE_MATCHING:
      num = a11 + a00;
      den = a11 + a10 + a01 + a00;
      break;
    case DistanceMetric::JACCARD:
      num = a11;
      den = a11 + a10 + a01;
      break;
    case DistanceMetric::TANIMOTO:
      num = a11 + a00;
      den = a11 + 2.0 * (a10 + a01) + a00;
      break;
    case DistanceMetric::BINARY_SIMILARITY:
      num = params.c11 * a11 + params.c10 * a10 + params.c01 * a01 + params.c00 * a00;
      den = params.d11 * a11 + params.d10 * a10 + params.d01 * a01 + params.d00 * a00;
      break;
    default:
      throw std::logic_error("binary_measure_similarity: not a binary measure");
  }
  return den != 0.0 ? num / den : undefined;
}

/**
 * The query of a distance computation: one value per comparison field, which
 * of them are present, and AdjustM = sum[Qi] / sum[nonmissing(Xi)*Qi] from the
 * ComparisonMeasure text. A missing entry's value is never read.
 * ClusteringModel takes Qi from MissingValueWeights; a NearestNeighborModel has
 * none, so every Qi is 1 and AdjustM = n / present. Each model kept its own
 * copy of this. Callers have checked that the present Qi sum to > 0.
 */
struct Query {
  Eigen::VectorXd values;
  std::vector<size_t> present;
  double adjust = 1.0;
};

// Templated on the sample type so math/ does not depend on core/.
template <class SampleT, class GetQi>
Query build_query(const SampleT& sample, const std::vector<ComparisonField>& fields, GetQi qi) {
  Query query;
  query.values = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(fields.size()));
  double total_weight = 0.0;
  double present_weight = 0.0;
  for (size_t i = 0; i < fields.size(); i++) {
    total_weight += qi(i);
    const auto& value = sample[fields[i].field_index].value;
    if (value.missing) continue;
    query.values[static_cast<Eigen::Index>(i)] = value.value;
    query.present.push_back(i);
    present_weight += qi(i);
  }
  if (query.present.size() < fields.size()) query.adjust = total_weight / present_weight;

  return query;
}

/**
 * A center-based distance over the present fields (SumNM), with AdjustM
 * applied where each aggregation puts it: inside the root for euclidean and
 * minkowski, outside it for squaredEuclidean, cityBlock and chebychev. With
 * nothing missing AdjustM is 1 and this is the plain distance.
 */
inline double apply_adjust(DistanceMetric metric, double distance, double adjust, double minkowski_p) {
  if (adjust == 1.0) return distance;
  switch (metric) {
    case DistanceMetric::EUCLIDEAN:
      return distance * std::sqrt(adjust);
    case DistanceMetric::MINKOWSKI:
      return distance * std::pow(adjust, 1.0 / minkowski_p);
    default:
      return distance * adjust;
  }
}

// Convenience overload without per-field similarityScale (scale = 1).
template <typename GetQuery, typename GetRow, typename GetWeight, typename GetCF>
double compute_distance(DistanceMetric metric, size_t n, double minkowski_p, GetQuery q, GetRow r, GetWeight w,
                        GetCF cf) {
  return compute_distance(metric, n, minkowski_p, q, r, w, cf, [](size_t) { return 1.0; });
}

#endif  // CPMML_DISTANCE_H
