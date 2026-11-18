// MUST NOT COMPILE (docs/design/typed-core.md §2.4, §6.4, review S14): code
// outside the ColdPath issuers -- here, an expression evaluating a record --
// making a ColdPath to call a generic std::visit entry point. The constructor
// is private; only cpmml::bind::, cpmml::api:: and the tests' issuer are
// friends. With CPMML_NEGATIVE_CONTROL the tag comes from an issuer and the
// file compiles: the failure is the construction alone.
#include <string>

#include "core/types/compare.h"

#ifdef CPMML_NEGATIVE_CONTROL
namespace cpmml::types::testing {
struct ColdPathIssuer {
  static cold::ColdPath make() { return {}; }
};
}  // namespace cpmml::types::testing
#endif

namespace cpmml::expression {
inline bool equal_on_scoring_path(const types::Value& a, const types::Value& b, std::string& scratch) {
#ifdef CPMML_NEGATIVE_CONTROL
  const types::cold::ColdPath tag = types::testing::ColdPathIssuer::make();
#else
  const types::cold::ColdPath tag{};
#endif
  return types::cold::relate(tag, types::Relation::Equal, a, b, scratch) == types::Tri::True;
}
}  // namespace cpmml::expression

bool use(const cpmml::types::Value& a, std::string& s) { return cpmml::expression::equal_on_scoring_path(a, a, s); }
