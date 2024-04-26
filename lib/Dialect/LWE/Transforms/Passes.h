#ifndef LIB_DIALECT_LWE_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_LWE_TRANSFORMS_PASSES_H_

#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/Transforms/SetDefaultParameters.h"

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/LWE/Transforms/Passes.h.inc"

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LWE_TRANSFORMS_PASSES_H_
