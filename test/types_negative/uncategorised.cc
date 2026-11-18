// MUST NOT COMPILE (docs/design/typed-core.md §2.2): an alternative that is in
// no category. Weird is not a PmmlType (no compare(), wrong index), so it is
// neither Sentinel, PmmlNumeric, Textual, Logical nor Temporal, and the
// check every Value goes through, all_categorised, rejects the variant. With
// CPMML_NEGATIVE_CONTROL the variant has only categorised alternatives.
#include <variant>

#include "core/types/value.h"

namespace cpmml::types {
struct Weird {
  static constexpr TypeId type_id = TypeId::Double;
  double v;
};
}  // namespace cpmml::types

#ifdef CPMML_NEGATIVE_CONTROL
using Extended = std::variant<cpmml::types::Missing, cpmml::types::Double>;
#else
using Extended = std::variant<cpmml::types::Missing, cpmml::types::Double, cpmml::types::Weird>;
#endif
static_assert(cpmml::types::all_categorised(static_cast<Extended*>(nullptr)),
              "every Value alternative is in exactly one category");
