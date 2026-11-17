// Must NOT compile: a new alternative that satisfies no category concept.
#include "typed_value.h"
namespace pmml { struct Weird { static constexpr TypeId type_id = TypeId::Double; }; }
using V2 = std::variant<pmml::Missing, pmml::Weird>;
static_assert(pmml::all_categorised(static_cast<V2*>(nullptr)));
