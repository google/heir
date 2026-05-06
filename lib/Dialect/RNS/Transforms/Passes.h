#ifndef LIB_DIALECT_RNS_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_RNS_TRANSFORMS_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/Transforms/LowerConvertBasis.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace rns {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/RNS/Transforms/Passes.h.inc"

}  // namespace rns
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_RNS_TRANSFORMS_PASSES_H_
