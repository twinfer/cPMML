// Must NOT compile: a visitor that forgets one alternative (String).
#include "typed_value.h"
using namespace pmml;
struct Incomplete {
  template <class T> requires(Sentinel<T>) int operator()(const T&) const { return 0; }
  template <PmmlNumeric T> int operator()(const T&) const { return 1; }
  template <Logical T> int operator()(const T&) const { return 2; }
  template <Temporal T> int operator()(const T&) const { return 3; }
};
int f(const Value& v) { return std::visit(Incomplete{}, v); }
