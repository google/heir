#ifndef LIB_TARGET_OPTALYSYS_OPTALYSYSTEMPLATES_H_
#define LIB_TARGET_OPTALYSYS_OPTALYSYSTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace optalysys {

// clang-format off
constexpr std::string_view kModulePreludeTemplate = R"cpp(
#include <opt_api.h>

typedef uint_t(*lut_fn)(uint_t);

)cpp";
// clang-format on

}  // namespace optalysys
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPTALYSYS_OPTALYSYSTEMPLATES_H_
