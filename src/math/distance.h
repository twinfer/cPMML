
/*******************************************************************************
 * Shared distance and similarity metrics for ClusteringModel and KNN.
 *******************************************************************************/

#ifndef CPMML_DISTANCE_H
#define CPMML_DISTANCE_H

#include <cmath>
#include <limits>

/**
 * Per-field comparison functions used by both ClusteringModel and
 * NearestNeighborModel.  These correspond to the PMML ComparisonMeasure
 * compareFunction attribute values.
 */
enum class CompareFunc { ABS_DIFF, GAUSS_SIM, DELTA, EQUAL };

inline CompareFunc parse_compare_func(const std::string& s) {
  // Caller is expected to pass a lowered string.
  if (s == "gausssim") return CompareFunc::GAUSS_SIM;
  if (s == "delta") return CompareFunc::DELTA;
  if (s == "equal") return CompareFunc::EQUAL;
  return CompareFunc::ABS_DIFF;
}

inline double field_compare(double xi, double yi, CompareFunc cf) {
  switch (cf) {
    case CompareFunc::ABS_DIFF:
      return std::abs(xi - yi);
    case CompareFunc::GAUSS_SIM:
      return 1.0 - std::exp(-(xi - yi) * (xi - yi));
    case CompareFunc::DELTA:
      return (xi != yi) ? 1.0 : 0.0;
    case CompareFunc::EQUAL:
      return (xi == yi) ? 1.0 : 0.0;
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
 */
template <typename GetQuery, typename GetRow, typename GetWeight, typename GetCF>
double compute_distance(DistanceMetric metric, size_t n, double minkowski_p, GetQuery q, GetRow r, GetWeight w,
                        GetCF cf) {
  switch (metric) {
    case DistanceMetric::EUCLIDEAN: {
      double sum = 0.0;
      for (size_t i = 0; i < n; i++) {
        double d = field_compare(q(i), r(i), cf(i));
        sum += w(i) * d * d;
      }
      return std::sqrt(sum);
    }
    case DistanceMetric::SQUARED_EUCLIDEAN: {
      double sum = 0.0;
      for (size_t i = 0; i < n; i++) {
        double d = field_compare(q(i), r(i), cf(i));
        sum += w(i) * d * d;
      }
      return sum;
    }
    case DistanceMetric::CITYBLOCK: {
      double sum = 0.0;
      for (size_t i = 0; i < n; i++) sum += w(i) * field_compare(q(i), r(i), cf(i));
      return sum;
    }
    case DistanceMetric::CHEBYSHEV: {
      double mx = 0.0;
      for (size_t i = 0; i < n; i++) mx = std::max(mx, w(i) * field_compare(q(i), r(i), cf(i)));
      return mx;
    }
    case DistanceMetric::MINKOWSKI: {
      double sum = 0.0;
      for (size_t i = 0; i < n; i++) sum += w(i) * std::pow(field_compare(q(i), r(i), cf(i)), minkowski_p);
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
      return compute_distance(DistanceMetric::SIMPLE_MATCHING, n, minkowski_p, q, r, w, cf);
  }
  return 0.0;
}

#endif  // CPMML_DISTANCE_H
