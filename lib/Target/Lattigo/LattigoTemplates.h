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
// clang-format on
constexpr std::string_view kMathImport = "\"math\"";
constexpr std::string_view kSlicesImport = "\"slices\"";

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_LATTIGO_LATTIGOTEMPLATES_H_
