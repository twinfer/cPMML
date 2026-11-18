/*******************************************************************************
 * Copyright 2026 Khalid Daoud. All rights reserved.
 * Author: Khalid Daoud
 *******************************************************************************/

#ifndef CPMML_TYPES_COLD_H
#define CPMML_TYPES_COLD_H

/**
 * Design note -- the ColdPath tag (§2.4, §6.4, review S14).
 *
 * The generic operations over a whole Value -- the std::visit entry points in
 * cpmml::types::cold (cast.h, compare.h, arithmetic.h, java_text.h,
 * parse.h) -- cost 4 to 25 times a typed operation (§2.10). The scoring path
 * must use the monomorphic operations the binder chose instead. That rule is
 * kept by the API, not by grep: every cold:: entry point takes a ColdPath,
 * and a ColdPath can only be made by the three issuers below, which are
 * declared here and defined by the code allowed to run cold:
 *
 *   cpmml::bind::ColdPathIssuer           load code (src/bind/, P1.2)
 *   cpmml::api::ColdPathIssuer            the API layer (src/api/)
 *   cpmml::types::testing::ColdPathIssuer tests
 *
 * Model and expression code cannot construct one (the constructor is
 * private), so a generic visit from there does not compile
 * (test/types_negative/cold_path.cc). A ColdPath may be copied: code that was
 * handed one by an issuer is cold by construction.
 */

namespace cpmml::bind {
struct ColdPathIssuer;
}
namespace cpmml::api {
struct ColdPathIssuer;
}
namespace cpmml::types::testing {
struct ColdPathIssuer;
}

namespace cpmml::types::cold {

class ColdPath {
 private:
  constexpr ColdPath() = default;
  friend struct ::cpmml::bind::ColdPathIssuer;
  friend struct ::cpmml::api::ColdPathIssuer;
  friend struct ::cpmml::types::testing::ColdPathIssuer;
};

}  // namespace cpmml::types::cold

#endif  // CPMML_TYPES_COLD_H
