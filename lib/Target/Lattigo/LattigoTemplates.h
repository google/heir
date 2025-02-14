#ifndef LIB_TARGET_LATTIGO_LATTIGOTEMPLATES_H_
#define LIB_TARGET_LATTIGO_LATTIGOTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace lattigo {

// clang-format off
constexpr std::string_view kModulePreludeBGVTemplate = R"go(
import (
    "github.com/tuneinsight/lattigo/v6/core/rlwe"
    "github.com/tuneinsight/lattigo/v6/schemes/bgv"
)
)go";

constexpr std::string_view kModulePreludeCKKSTemplate = R"go(
import (
    "github.com/tuneinsight/lattigo/v6/core/rlwe"
    "github.com/tuneinsight/lattigo/v6/schemes/ckks"
)
)go";
// clang-format on

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_LATTIGO_LATTIGOTEMPLATES_H_
