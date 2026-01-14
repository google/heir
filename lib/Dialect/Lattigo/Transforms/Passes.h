#ifndef LIB_DIALECT_LATTIGO_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_LATTIGO_TRANSFORMS_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/Transforms/AllocToInPlace.h"
#include "lib/Dialect/Lattigo/Transforms/ConfigureCryptoContext.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace lattigo {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LATTIGO_TRANSFORMS_PASSES_H_
