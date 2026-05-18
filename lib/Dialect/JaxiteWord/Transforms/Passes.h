#ifndef LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_H_

// IWYU pragma: begin_keep
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/JaxiteWord/Transforms/ConfigureCryptoContext.h"
// IWYU pragma: end_keep

namespace mlir {
namespace heir {
namespace jaxiteword {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/JaxiteWord/Transforms/Passes.h.inc"

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_JAXITEWORD_TRANSFORMS_PASSES_H_
