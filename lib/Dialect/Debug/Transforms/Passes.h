#ifndef LIB_DIALECT_DEBUG_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_DEBUG_TRANSFORMS_PASSES_H_

#include "lib/Dialect/Debug/IR/DebugDialect.h"
#include "lib/Dialect/Debug/Transforms/ValidateNames.h"

namespace mlir {
namespace heir {
namespace debug {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Debug/Transforms/Passes.h.inc"

}  // namespace debug
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_DEBUG_TRANSFORMS_PASSES_H_
