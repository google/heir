#ifndef LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/Transforms/AllocToInPlace.h"
#include "lib/Dialect/Openfhe/Transforms/ConfigureCryptoContext.h"
#include "lib/Dialect/Openfhe/Transforms/ConvertToExtendedBasis.h"
#include "lib/Dialect/Openfhe/Transforms/CountAddAndKeySwitch.h"
#include "lib/Dialect/Openfhe/Transforms/FastRotationPrecompute.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_H_
