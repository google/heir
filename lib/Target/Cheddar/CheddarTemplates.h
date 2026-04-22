#ifndef LIB_TARGET_CHEDDAR_CHEDDARTEMPLATES_H_
#define LIB_TARGET_CHEDDAR_CHEDDARTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace cheddar {

// Includes emitted at the top of generated files
// CHEDDAR headers use unnamespaced paths (core/, extension/).
constexpr std::string_view kCheddarInclude = R"cpp(
#include "core/Context.h"
#include "core/Container.h"
#include "core/Parameter.h"
#include "core/Encode.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"
#include "UserInterface.h"
)cpp";

constexpr std::string_view kCheddarExtensionInclude = R"cpp(
#include "extension/BootContext.h"
#include "extension/LinearTransform.h"
#include "extension/EvalPoly.h"
#include "extension/Hoist.h"
)cpp";

constexpr std::string_view kStdIncludes = R"cpp(
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
)cpp";

constexpr std::string_view kJsonInclude = R"cpp(
#include <json/json.h>
#include <fstream>
)cpp";

// The type alias prelude, parametric on word type
constexpr std::string_view kTypeAliasPrelude64 = R"cpp(
  using namespace cheddar;
  using word = uint64_t;
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Const = Constant<word>;
  using Evk = EvaluationKey<word>;
  using EvkMapT = EvkMap<word>;
  using CtxPtr = std::shared_ptr<Context<word>>;
  using Param = Parameter<word>;
  using UI = UserInterface<word>;
  using Enc = Encoder<word>;
  using Complex = std::complex<double>;
)cpp";

constexpr std::string_view kTypeAliasPrelude32 = R"cpp(
  using namespace cheddar;
  using word = uint32_t;
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Const = Constant<word>;
  using Evk = EvaluationKey<word>;
  using EvkMapT = EvkMap<word>;
  using CtxPtr = std::shared_ptr<Context<word>>;
  using Param = Parameter<word>;
  using UI = UserInterface<word>;
  using Enc = Encoder<word>;
  using Complex = std::complex<double>;
)cpp";

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_CHEDDAR_CHEDDARTEMPLATES_H_
