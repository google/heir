#ifndef LIB_DIALECT_LATTIGO_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_LATTIGO_TRANSFORMS_PASSES_H_

#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/Transforms/AllocToInplace.h"
#include "lib/Dialect/Lattigo/Transforms/ConfigureCryptoContext.h"
#include "lib/Dialect/Lattigo/Transforms/HoistRotations.h"

namespace mlir {
namespace heir {
namespace lattigo {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Lattigo/Transforms/Passes.h.inc"

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_LATTIGO_TRANSFORMS_PASSES_H_
