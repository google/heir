#ifndef INCLUDE_DIALECT_LWE_TRANSFORMS_PASSES_H_
#define INCLUDE_DIALECT_LWE_TRANSFORMS_PASSES_H_

#include "include/Dialect/LWE/IR/LWEDialect.h"
#include "include/Dialect/LWE/Transforms/SetDefaultParameters.h"

namespace mlir {
namespace heir {
namespace lwe {

#define GEN_PASS_REGISTRATION
#include "include/Dialect/LWE/Transforms/Passes.h.inc"

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // INCLUDE_DIALECT_LWE_TRANSFORMS_PASSES_H_
