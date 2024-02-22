#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace openfhe {

// The `// from @openfhe` is a bit of a hack to avoid out default copybara
// transforms for HEIR includes.

// clang-format off
constexpr std::string_view kModulePrelude = R"cpp(
#include "src/pke/include/openfhe.h" // from @openfhe

using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using CryptoContextT = CryptoContext<DCRTPoly>;
)cpp";
// clang-format on

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_
