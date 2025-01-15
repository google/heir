#ifndef LIB_TARGET_TFHERUST_TFHERUSTHLTEMPLATES_H_
#define LIB_TARGET_TFHERUST_TFHERUSTHLTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace tfhe_rust {

constexpr std::string_view kModulePrelude = R"rust(
use std::collections::BTreeMap;
use tfhe::{FheUint8, FheUint16, FheUint32, FheUint64};
use tfhe::prelude::*;
use tfhe::ServerKey;
)rust";

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_TFHERUST_TFHERUSTHLTEMPLATES_H_
