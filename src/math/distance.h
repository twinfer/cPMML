
/*******************************************************************************
 * Shared distance and similarity metrics for ClusteringModel and KNN.
 *******************************************************************************/

#ifndef CPMML_DISTANCE_H
#define CPMML_DISTANCE_H

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

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

/**
 * The query vector for a distance computation: one entry per comparison field,
 * read from the prepared sample in field order.
 *
 * ClusteringModel and NearestNeighborModel each carried their own copy of this,
 * differing only in the wording of the exception -- neither of which said which
 * field was missing. Templated on the sample type so math/ does not depend on
 * core/.
 */
template <class SampleT>
inline Eigen::VectorXd build_query(const SampleT& sample, const std::vector<ComparisonField>& fields,
                                   const char* element) {
  Eigen::VectorXd query(static_cast<Eigen::Index>(fields.size()));
  for (size_t i = 0; i < fields.size(); i++) {
    const auto& value = sample[fields[i].field_index].value;
    if (value.missing)
      throw cpmml::MissingValueException(std::string("missing value for ") + element + " \"" + fields[i].field_name +
                                         "\"");
    query[static_cast<Eigen::Index>(i)] = value.value;
  }

  return query;
}

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
    case DistanceMetric::SIMPLE_MATCHING: {
      int match = 0;
      for (size_t i = 0; i < n; i++)
        if ((q(i) > 0.5) == (r(i) > 0.5)) match++;
      return (n > 0) ? 1.0 - static_cast<double>(match) / n : 0.0;
    }
    case DistanceMetric::JACCARD: {
      int a11 = 0, union_count = 0;
      for (size_t i = 0; i < n; i++) {
        bool xi = q(i) > 0.5, yi = r(i) > 0.5;
        if (xi || yi) union_count++;
        if (xi && yi) a11++;
      }
      return (union_count > 0) ? 1.0 - static_cast<double>(a11) / union_count : 0.0;
    }
    case DistanceMetric::TANIMOTO: {
      int a11 = 0, a10 = 0, a01 = 0, a00 = 0;
      for (size_t i = 0; i < n; i++) {
        bool xi = q(i) > 0.5, yi = r(i) > 0.5;
        if (xi && yi)
          a11++;
        else if (xi)
          a10++;
        else if (yi)
          a01++;
        else
          a00++;
      }
      int denom = a11 + 2 * (a10 + a01) + a00;
      return (denom > 0) ? 1.0 - static_cast<double>(a11 + a00) / denom : 0.0;
    }
    case DistanceMetric::BINARY_SIMILARITY:
      // binarySimilarity is equivalent to simpleMatching when custom c/d
      // parameters are not provided (PMML default c11=c10=c01=c00=d11=d10=d01=d00=1).
      return compute_distance(DistanceMetric::SIMPLE_MATCHING, n, minkowski_p, q, r, w, cf, sc);
  }
  return 0.0;
}

// Convenience overload without per-field similarityScale (scale = 1).
template <typename GetQuery, typename GetRow, typename GetWeight, typename GetCF>
double compute_distance(DistanceMetric metric, size_t n, double minkowski_p, GetQuery q, GetRow r, GetWeight w,
                        GetCF cf) {
  return compute_distance(metric, n, minkowski_p, q, r, w, cf, [](size_t) { return 1.0; });
}

#endif  // CPMML_DISTANCE_H
