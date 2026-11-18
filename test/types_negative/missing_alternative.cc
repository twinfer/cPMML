// MUST NOT COMPILE (docs/design/typed-core.md §2.2): a visitor over Value that
// forgets one alternative. Visitors are overload sets constrained by the
// categories, with no unconstrained catch-all, so std::visit finds no
// overload for String. With CPMML_NEGATIVE_CONTROL the String overload is
// there and the file compiles: the failure is that one missing overload.
#include <variant>

#include "core/types/value.h"

using namespace cpmml::types;

struct Incomplete {
  template <Sentinel T>
  int operator()(const T&) const {
    return 0;
  }
  template <PmmlNumeric T>
  int operator()(const T&) const {
    return 1;
  }
  template <Logical T>
  int operator()(const T&) const {
    return 2;
  }
#ifdef CPMML_NEGATIVE_CONTROL
  template <Textual T>
  int operator()(const T&) const {
    return 3;
  }
#endif
};

int visit_all(const Value& v) { return std::visit(Incomplete{}, v); }
