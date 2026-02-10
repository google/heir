#ifndef LIB_TARGET_LATTIGO_LATTIGOTEMPLATES_H_
#define LIB_TARGET_LATTIGO_LATTIGOTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace lattigo {

// prevent format mangling
// clang-format off
constexpr std::string_view kRlweImport =
    "\"github.com/tuneinsight/lattigo/v6/core/rlwe\"";
constexpr std::string_view kBgvImport =
    "\"github.com/tuneinsight/lattigo/v6/schemes/bgv\"";
constexpr std::string_view kCkksImport =
    "\"github.com/tuneinsight/lattigo/v6/schemes/ckks\"";
constexpr std::string_view kLintransImport =
  "\"github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans\"";
constexpr std::string_view kCkksCircuitPolynomialImport =
  "\"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial\"";
constexpr std::string_view kLattigoBignumImport=
  "\"github.com/tuneinsight/lattigo/v6/utils/bignum\"";
constexpr std::string_view kBootstrappingImport =
  "\"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping\"";
constexpr std::string_view kLattigoUtilsImport =
  "\"github.com/tuneinsight/lattigo/v6/utils\"";
// clang-format on

constexpr std::string_view kMathImport = "\"math\"";
constexpr std::string_view kSlicesImport = "\"slices\"";
constexpr std::string_view kMathBigImport = "\"math/big\"";

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_LATTIGO_LATTIGOTEMPLATES_H_
