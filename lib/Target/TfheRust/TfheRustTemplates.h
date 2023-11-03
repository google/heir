#ifndef LIB_TARGET_TFHERUST_TFHERUSTTEMPLATES_H_
#define LIB_TARGET_TFHERUST_TFHERUSTTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace tfhe_rust {

constexpr std::string_view kModulePrelude = R"rust(
use tfhe::shortint;
use tfhe::shortint::prelude::*;
use tfhe::shortint::CiphertextBig as Ciphertext;
)rust";

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_TFHERUST_TFHERUSTTEMPLATES_H_
