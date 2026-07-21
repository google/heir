#ifndef LIB_TARGET_POULPY_POULPYTEMPLATES_H_
#define LIB_TARGET_POULPY_POULPYTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace poulpy {
constexpr std::string_view kModulePrelude =
    R"poulpy(This is the kModulePrelude)poulpy";
}  // namespace poulpy
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_POULPY_POULPYTEMPLATES_H_