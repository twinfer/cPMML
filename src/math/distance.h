
/*******************************************************************************
 * Shared distance and similarity metrics for ClusteringModel and KNN.
 *
 * Both models read their fields with read_comparison_field and rank rows with
 * query_dissimilarity, which applies the ComparisonMeasure inner functions
 * (field_compare) exactly as the spec defines them and then the measure.
 *******************************************************************************/

#ifndef CPMML_DISTANCE_H
#define CPMML_DISTANCE_H

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/enum_parser.h"
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

// COMPARE-FUNCTION, on ComparisonMeasure (the default for every field, absDiff
// when absent), ClusteringField and KNNInput. parse_compare_func used to
// lower-case the value and fall back to absDiff for anything it did not know,
// so compareFunction="table" -- a valid PMML value cPMML does not implement --
// and any misspelling scored as an absolute difference. JPMML 1.7.7 refuses
// table on a field ("Attribute with value ClusteringField@compareFunction=table
// is not supported") and reads an unknown value as absent.
inline constexpr EnumEntry<CompareFunc> COMPARE_FUNCTION_TABLE[] = {
    {"absDiff", CompareFunc::ABS_DIFF},
    {"gaussSim", CompareFunc::GAUSS_SIM},
    {"delta", CompareFunc::DELTA},
    {"equal", CompareFunc::EQUAL},
    {"table", CompareFunc::ABS_DIFF, nullptr, EnumSupport::UNSUPPORTED},
};

inline CompareFunc read_compare_function(const XmlNode& node, CompareFunc when_absent) {
  return read_enum(node, "compareFunction", COMPARE_FUNCTION_TABLE, when_absent);
}

// The inner function c(x,y) of ComparisonMeasure (ClusteringModel.html, 4.4.1),
// exactly as the spec writes it, for ClusteringModel and NearestNeighborModel
// alike:
//
//   absDiff   c(x,y) = |x - y|
//   gaussSim  c(x,y) = exp(-ln(2) * z*z / (s*s)), z = x - y, s = similarityScale
//   delta     c(x,y) = 0 if x = y, 1 else
//   equal     c(x,y) = 1 if x = y, 0 else
//
// c(x,y) goes into the measure as it stands -- euclidean sums w*c^2, cityBlock
// w*c, chebychev takes the max of w*c -- so under a distance measure a matching
// value adds 1 with equal and an identical value adds 1 with gaussSim. That is
// what the spec says and what JPMML 1.7.7 computes, in both models. This used to
// turn every similarity into a dissimilarity first: gaussSim into 1 - c and
// equal into 0-if-equal, i.e. delta, which inverted every such field. KNN had
// worked around it for equal (1be5d0f); ClusteringModel had not.
//
// x and y are the same kind of double: a number for a numeric or boolean field
// (the query's boolean is 0/1, a cluster center any number, a mean such as
// 0.37 included), the document's legacy code for a string field (switch 2,
// core/value.h; a string
// center is the Java text of its number, see ClusteringModel::parse_clusters;
// a KNN training cell is read with the field's dataType). So x = y is equality
// of the field's values. Value carries no type tag to compare by, so this is
// done by reading both sides alike. s is never 0 (read_comparison_field), and
// missing values never reach this.
inline double field_compare(double xi, double yi, CompareFunc cf, double scale) {
  switch (cf) {
    case CompareFunc::ABS_DIFF:
      return std::abs(xi - yi);
    case CompareFunc::GAUSS_SIM: {
      const double z = xi - yi;
      return std::exp(-std::log(2.0) * z * z / (scale * scale));
    }
    case CompareFunc::DELTA:
      return xi == yi ? 0.0 : 1.0;
    case CompareFunc::EQUAL:
      return xi == yi ? 1.0 : 0.0;
  }
  throw std::logic_error("field_compare: unknown CompareFunc");
}

/**
 * Reads the attributes a ClusteringField and a KNNInput share -- field,
 * compareFunction (`default_cf`, the ComparisonMeasure value, when absent) and
 * fieldWeight -- and, for a ClusteringField, similarityScale. The caller
 * resolves field_index.
 *
 * gaussSim needs s: "s is the value of attribute similarityScale (required in
 * this case) in the ClusteringField", the distance at which the similarity is
 * 0.5. So a field whose compare function is gaussSim, its own or inherited, is
 * a load error
 *  - on a ClusteringField without similarityScale, or with similarityScale 0
 *    or not finite. It used to default to 1, and 0 was replaced by 1. JPMML
 *    1.7.7 refuses the first ("Element ClusteringField is not valid") and gives
 *    NaN for an identical value with the second;
 *  - on any KNNInput, which has no similarityScale attribute. It used to score
 *    with s = 1. JPMML 1.7.7 refuses it ("Element KNNInput is not valid").
 * The same goes for an s so small that s*s is 0 (1e-320): JPMML 1.7.7 scores
 * it with the NaN that 0/0 gives an identical value. A negative s is accepted:
 * only s*s enters c, as in JPMML. JPMML also refuses ComparisonMeasure
 * compareFunction="gaussSim" whenever a ClusteringField inherits it, even one
 * with its own similarityScale; the spec makes it the default for every field
 * without one and asks only for s on the field, so cPMML scores that.
 *
 * similarityScale is read strictly whenever it is present, as every numeric
 * attribute is (6914ecd), so similarityScale="abc" is a load error even on a
 * field that is not gaussSim and never uses it. JPMML 1.7.7 ignores it there
 * and scores; that is a deliberate departure.
 */
inline ComparisonField read_comparison_field(const XmlNode& node, CompareFunc default_cf, bool has_similarity_scale) {
  const char* element = has_similarity_scale ? "ClusteringField" : "KNNInput";
  ComparisonField f;
  f.field_name = node.required_attribute("field");
  f.compare_func = read_compare_function(node, default_cf);
  f.field_weight = node.get_double_attribute("fieldWeight", 1.0);
  if (has_similarity_scale && node.exists_attribute("similarityScale"))
    f.similarity_scale = node.required_double_attribute("similarityScale");
  if (f.compare_func != CompareFunc::GAUSS_SIM) return f;

  const std::string where = std::string(element) + " \"" + f.field_name + "\": compareFunction gaussSim" +
                            (node.exists_attribute("compareFunction") ? "" : " (from ComparisonMeasure)");
  if (!has_similarity_scale)
    throw cpmml::ParsingException(where + " needs a similarityScale, which KNNInput does not have");
  if (!node.exists_attribute("similarityScale"))
    throw cpmml::ParsingException(where + " requires the similarityScale attribute");
  // s*s, not s: a subnormal s such as 1e-320 squares to 0, and c would be
  // 0/0 = NaN for an identical value.
  if (!std::isfinite(f.similarity_scale) || !(f.similarity_scale * f.similarity_scale > 0.0))
    throw cpmml::ParsingException(where + " needs a finite similarityScale whose square is not 0, got \"" +
                                  node.required_attribute("similarityScale") + "\"");
  return f;
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
  const std::string text = minkowski.required_attribute("p-parameter");
  const double p = minkowski.required_double_attribute("p-parameter");
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
    const std::string text = node.required_attribute(name);
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

// ComparisonMeasure holds exactly one measure element (an xs:choice of the
// nine below), besides any Extension. A missing, unknown or second measure is
// a load error. JPMML 1.7.7 refuses the first two ("Required element
// ComparisonMeasure/<Measure> is not defined"); it skips an unknown element
// next to a known one and takes the last of two measures, both schema
// violations cPMML refuses. ClusteringModel used to take euclidean for a
// missing or unrecognised measure (<Euclidean/> included).
inline constexpr EnumEntry<DistanceMetric> MEASURE_TABLE[] = {
    {"euclidean", DistanceMetric::EUCLIDEAN},
    {"squaredEuclidean", DistanceMetric::SQUARED_EUCLIDEAN},
    {"chebychev", DistanceMetric::CHEBYSHEV},
    {"cityBlock", DistanceMetric::CITYBLOCK},
    {"minkowski", DistanceMetric::MINKOWSKI},
    {"simpleMatching", DistanceMetric::SIMPLE_MATCHING},
    {"jaccard", DistanceMetric::JACCARD},
    {"tanimoto", DistanceMetric::TANIMOTO},
    {"binarySimilarity", DistanceMetric::BINARY_SIMILARITY},
};

inline DistanceMetric read_measure(const XmlNode& comparison_measure) {
  bool found = false;
  DistanceMetric result = DistanceMetric::EUCLIDEAN;
  for (const auto& child : comparison_measure.get_childs()) {
    const std::string name = child.name();
    if (name.empty() || name == "Extension") continue;  // text, or an Extension
    const DistanceMetric metric = read_enum_element(child, MEASURE_TABLE, "measure element in ComparisonMeasure");
    if (found)
      throw cpmml::ParsingException("ComparisonMeasure has more than one measure (<" + name + "> after another)");
    found = true;
    result = metric;
  }
  if (!found)
    throw cpmml::ParsingException("ComparisonMeasure has no measure element; allowed elements are " +
                                  enum_reader::list_tokens(MEASURE_TABLE, false));
  return result;
}

// ComparisonMeasure@kind is required. Nothing downstream depends on it -- each
// measure is a distance or a similarity by definition -- but it is read so a
// document that misspells it is refused like any other.
inline constexpr EnumEntry<bool> MEASURE_KIND_TABLE[] = {
    {"distance", false},
    {"similarity", true},
};

inline bool read_measure_kind_is_similarity(const XmlNode& comparison_measure) {
  return read_enum(comparison_measure, "kind", MEASURE_KIND_TABLE);
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
    if (value.is_missing()) continue;
    query.values[static_cast<Eigen::Index>(i)] = value.as_double();
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

/**
 * The dissimilarity of a query and one center or training instance, the one
 * implementation ClusteringModel and NearestNeighborModel both rank by
 * (smaller = closer). `row(i)` is the row's value for field i, typed like the
 * query (see field_compare).
 *
 * The center-based measures run SumNM over the present fields with each
 * field's compare function, fieldWeight and similarityScale, then AdjustM
 * (apply_adjust). The binary measures are similarities s, ranked as 1 - s;
 * callers reach them only with every field present, and they are NaN where s
 * is undefined (binary_measure_similarity), which the callers turn into no
 * prediction.
 */
template <typename GetRow>
double query_dissimilarity(DistanceMetric metric, double minkowski_p, const BinarySimilarityParams& binary_params,
                           const std::vector<ComparisonField>& fields, const Query& query, GetRow row) {
  const std::vector<size_t>& present = query.present;
  const auto q = [&](size_t j) { return query.values[static_cast<Eigen::Index>(present[j])]; };
  const auto r = [&](size_t j) { return row(present[j]); };
  if (is_binary_metric(metric)) return 1.0 - binary_measure_similarity(metric, binary_params, present.size(), q, r);
  const double distance = compute_distance(
      metric, present.size(), minkowski_p, q, r, [&](size_t j) { return fields[present[j]].field_weight; },
      [&](size_t j) { return fields[present[j]].compare_func; },
      [&](size_t j) { return fields[present[j]].similarity_scale; });
  return apply_adjust(metric, distance, query.adjust, minkowski_p);
}

#endif  // CPMML_DISTANCE_H
