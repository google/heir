#ifndef LIB_TARGET_TFHERUSTBOOL_TFHERUSTBOOLTEMPLATES_H_
#define LIB_TARGET_TFHERUSTBOOL_TFHERUSTBOOLTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace tfhe_rust_bool {

constexpr std::string_view kModulePrelude = R"rust(
use tfhe::boolean::prelude::*;
)rust";

}  // namespace tfhe_rust_bool
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_TFHERUSTBOOL_TFHERUSTBOOLTEMPLATES_H_
