#ifndef LIB_DIALECT_LWE_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_LWE_TRANSFORMS_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/Transforms/AddDebugPort.h"
#include "lib/Dialect/LWE/Transforms/ImplementTrivialEncryptionAsAddition.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_TRANSFORMS_PASSES_H_
